from deepface import DeepFace
from milvus import (
    get_primary_metadata as _get_primary_metadata,
    get_secondary_metadata as _get_secondary_metadata,
    update_secondary_metadata as _update_secondary_metadata,
    find_similar_users as _find_similar_users,
    set_primary_metadata as _set_primary_metadata,
    disable_user         as _disable_user,
    get_user             as _get_user
)
from minio_uploader import put_secondary_photo, put_primary_photo

from PIL import Image
import numpy as np, io
from deepface.commons import functions, distance

_model = "SFace"
_detector = "skip"
_similarity_metric = "euclidean_l2"

def represent(img_path, model_name, detector_backend, enforce_detection, align):
    result = {}
    embedding_objs = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = embedding_objs
    return result


def verify(
    img1_path, img2_path, model_name, detector_backend, distance_metric, enforce_detection, align
):
    obj = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )
    return obj


def analyze(img_path, actions, detector_backend, enforce_detection, align):
    result = {}
    demographies = DeepFace.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = demographies
    return result


def set_primary_photo(user_id: str, photo_stream):
    user = _get_user(user_id)
    if user is not None and user["disabled_at"] > 0:
        raise UserDisabled(f"User {user_id} was disabled at {user['disabled_at']}")
    existing_md = _get_primary_metadata(user_id)
    if existing_md is not None:
        raise MetadataAlreadyExists(f"User {user_id} already owns primary face uploaded at {existing_md['uploaded_at']}")
    md = distance.l2_normalize(DeepFace.represent(
        img_path=loadImageFromStream(photo_stream),
        model_name=_model,
        enforce_detection=True,
        detector_backend=_detector,
        align=True,
        normalization="base",
    )[0]["embedding"])
    similar_users, distances = _find_similar_users(user_id,md, distance.findThreshold(_model,_similarity_metric))
    if similar_users[0] != user_id:
        if _disable_user(user_id):
            raise UserDisabled(f"Face is matching with user {similar_users[0]}, distance {distances[0]}")
    else:
        url = put_primary_photo(user_id,photo_stream.stream)
        now, rows = _set_primary_metadata(user_id, md, url)
        if rows > 0:
            return now
        else:
            now, rows = _set_primary_metadata(user_id, md, url)
            return now


def check_similar_user_and_register_metadata(user_id: str, pics: list):
    user_reference_metadata = _get_primary_metadata(user_id)
    if user_reference_metadata is None:
        raise MetadataNotFound(f"User {user_id} have no registered primary metadata yet")
    user_reference_metadata = user_reference_metadata["face_metadata"]
    i = 0
    metadata_to_compare = [user_reference_metadata]
    try:
        metadata_to_compare.extend([
            DeepFace.represent(
                img_path=loadImageFromStream(p),
                model_name=_model,
                enforce_detection=True,
                detector_backend=_detector
            )[0]["embedding"] for p in pics
        ])
    except ValueError as e:
        raise NoFaces("No faces detected", e)
    threshold = distance.findThreshold(_model,_similarity_metric)
    bestIndex, euclidian = compare_metadatas(metadata_to_compare, threshold)
    if bestIndex == -1:
        raise NotSameUser(f"user mismatch: distance is greater than {threshold}")
    url = put_secondary_photo(user_id,pics[bestIndex].stream)
    now, rows = _update_secondary_metadata(user_id,metadata_to_compare[bestIndex], url)
    if rows > 0:
        return bestIndex, euclidian, now
    else:
        now, rows = _update_secondary_metadata(user_id,metadata_to_compare[bestIndex])
        return bestIndex, euclidian, now


def compare_metadatas(metadatas: list, threshold: float):
    normalizedRefMetadata = distance.l2_normalize(metadatas[0])
    metadatas = metadatas[1:]
    distances = [distance.findEuclideanDistance(normalizedRefMetadata, distance.l2_normalize(md)) for md in metadatas]
    distances.sort()
    indexes = [distances.index(d) for d in distances if d <= threshold]
    if len(indexes) != len(metadatas):
        return -1,-1
    else:
        return indexes[0], distances[indexes[0]]

def loadImageFromStream(p):
    im = Image.open(p.stream)
    img = np.asarray(im)
    return img


def get_status(user_id: str):
    primary = _get_primary_metadata(user_id)
    secondary = _get_secondary_metadata(user_id)
    primaryUploaded = primary is not None
    if primaryUploaded and secondary is not None:
        lastVerified = max(primary["uploaded_at"], secondary["uploaded_at"])
    elif secondary is not None:
        lastVerified = secondary["uploaded_at"]
    elif primary is not None:
        lastVerified = primary["uploaded_at"]
    else:
        lastVerified = 0
    disabled = _get_user(user_id)
    return {
        "userID": user_id,
        "primaryPhotoUploaded": primaryUploaded,
        "lastVerified": lastVerified,
        "disabled": disabled is not None and disabled["disabled_at"] > 0
    }

class MetadataNotFound(Exception):
    def __init__(self, message):
        super().__init__(message)
class MetadataAlreadyExists(Exception):
    def __init__(self, message):
        super().__init__(message)

class NotSameUser(Exception):
    def __init__(self, message):
        super().__init__(message)

class UserDisabled(Exception):
    def __init__(self, message):
        super().__init__(message)
class UserNotFound(Exception):
    def __init__(self, message):
        super().__init__(message)
class NoFaces(Exception):
    def __init__(self, message, e):
        super(e).__init__(message)