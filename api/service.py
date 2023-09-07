import time

from deepface import DeepFace
from milvus import (
    get_primary_metadata as _get_primary_metadata,
    get_secondary_metadata as _get_secondary_metadata,
    update_secondary_metadata as _update_secondary_metadata,
    find_similar_users as _find_similar_users,
    set_primary_metadata as _set_primary_metadata,
    update_user          as _update_user,
    disable_user         as _disable_user,
    get_user             as _get_user,
    update_emotion_sequence_and_best_score as _update_emotion_sequence_and_best_score,
    update_session_ended_at as _update_session_ended_at
)
from minio_uploader import put_secondary_photo, put_primary_photo
from flask import current_app
import uuid
import requests
import glob
import os
import random
import time
import shutil
from os.path import exists

from PIL import Image
import cv2
import numpy as np, io, requests
from deepface.commons import functions, distance

_model = "SFace"
_model_fallback = "ArcFace"#"Facenet" #"VGG-Face"
_detector_high_quality = "yunet"
_detector_low_quality = "yunet" # TODO: test with skip, if we gonna get proper photos from FE
_similarity_metric = "euclidean_l2"
_min_images_with_emotions_to_proceed = 1

TOTAL_BEST_PICTURES = 7
DEFAULT_EMOTIONS_NUM = 3
ONE_CALL_IMAGES_COUNT = 15
DEFAULT_EMOTIONS_LIST = DeepFace.build_model("Emotion").idx_to_class

# TODO: take them from current_app.config.
IMG_STORAGE_PATH = '/tmp/upload'
SESSION_DURATION = 10 * 60 * int(1e9)
LIMIT_RATE = 1 * 60 * int(1e9)
BASE_SIMILARITY_ENDPOINT = 'http://127.0.0.1:5000/v1w/face-auth/similarity/'

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

def init_models():
    DeepFace.build_model(_model)
    DeepFace.build_model(_model_fallback)
    DeepFace.build_model("Emotion")
    samplePerson = requests.get(
        url="https://thispersondoesnotexist.com/", verify=False
    )
    if samplePerson.status_code == 200:
        img = loadImageFromStream(io.BytesIO(samplePerson.content))
        DeepFace.represent(img_path=img, detector_backend=_detector_high_quality, model_name=_model)
        DeepFace.represent(img_path=img, detector_backend=_detector_high_quality, model_name=_model_fallback)

def set_primary_photo(current_user, user_id: str, photo_stream):
    user = _get_user(user_id)
    if user is not None and user["disabled_at"] > 0:
        raise UserDisabled(f"User {user_id} was disabled at {user['disabled_at']}")
    existing_md = _get_primary_metadata(user_id)
    if existing_md is not None:
        raise MetadataAlreadyExists(f"User {user_id} already owns primary face uploaded at {existing_md['uploaded_at']}")
    img = loadImageFromStream(photo_stream)
    try:
        md = distance.l2_normalize(DeepFace.represent(
            img_path=img,
            model_name=_model,
            enforce_detection=True,
            detector_backend=_detector_high_quality,
            align=True,
            normalization="base",
            target_size=(1080,1080)
        )[0]["embedding"])
    except ValueError:
        raise NoFaces("No faces detected")
    threshold = distance.findThreshold(_model,_similarity_metric)
    similar_users, distances = _find_similar_users(user_id,md, threshold)
    if similar_users[0] != user_id:
        # make sure it is not a false positive, let's check other picture as well
        secondary_md = _get_secondary_metadata(similar_users[0])
        if secondary_md:
            bestIndex, euclidian = compare_metadatas([secondary_md["face_metadata"],md], threshold)
            if bestIndex != -1 and _disable_user(user_id):
                raise UserDisabled(f"Face is matching with user {similar_users[0]}, distance {distances[0]} {euclidian} < {threshold}")
        else:
            # that similar user dont have 2nd pic yet,but we can re-check with fallback model
            simiar_user_picture = get_primary_photo(similar_users[0])
            res = DeepFace.verify(
                img1_path=img,
                img2_path=loadImageFromStream(io.BytesIO(simiar_user_picture)),
                detector_backend=_detector_high_quality,
                model_name=_model_fallback,
                distance_metric=_similarity_metric,
                normalization="base",
                align=True
            )
            print(res['distance'],res['threshold'])
            if res["verified"]:
                disabled_at, disabled = _disable_user(user_id)
                if disabled:
                    callback(current_user, None,None, {"disabled_at": disabled_at})
                    raise UserDisabled(f"Face is matching with user {similar_users[0]}, distance {distances[0]} {res['distance']} < {res['threshold']}")
    url = put_primary_photo(user_id,photo_stream.stream)
    upd, rows = _set_primary_metadata(user_id, md, url)
    if rows > 0:
        try:
            callback(current_user, upd,None,user)
        except UnauthorizedFromWebhook as e:
            _delete_metadata(upd["user_picture_id"])
            raise e
        except requests.RequestException as e:
            _delete_metadata(upd["user_picture_id"])
            raise e # goes to 5xx

def check_similarity_and_update_secondary_photo(current_user, user_id: str, raw_pics: list):
    user_reference_metadata = _get_primary_metadata(user_id)
    if user_reference_metadata is None:
        raise MetadataNotFound(f"User {user_id} have no registered primary metadata yet")
    md_vector = user_reference_metadata["face_metadata"]
    pics = [loadImageFromStream(p) for p in raw_pics]
    md, bestIndex, euclidian,threshold = extract_and_compare_metadatas(md_vector, pics,_model)
    if bestIndex == -1:
        # user is not the same as on primary photo - let's try with more complex but slower model as well to reduce false-negatives
        primary_photo = get_primary_photo(user_id)
        md_vector = distance.l2_normalize(DeepFace.represent(
            img_path=loadImageFromStream(io.BytesIO(primary_photo)),
            model_name=_model_fallback,
            enforce_detection=True,
            detector_backend=_detector_high_quality,
            align=True,
            normalization="base",
        )[0]["embedding"])
        mdFallback, bestIndex, euclidian,threshold = extract_and_compare_metadatas(md_vector, pics,_model_fallback)
        if bestIndex == -1:
            raise NotSameUser(f"user mismatch: distance is greater than {threshold}: {euclidian}")
    url = put_secondary_photo(user_id,raw_pics[bestIndex].stream)
    prev_state = _get_secondary_metadata(user_id)
    upd, rows = _update_secondary_metadata(user_id,md[bestIndex], url)
    if rows > 0:
        try:
            callback(current_user, user_reference_metadata,upd,_get_user(user_id))
        except UnauthorizedFromWebhook as e:
            if prev_state is not None:
                _update_secondary_metadata(user_id,prev_state["face_metadata"], prev_state["url"])
            raise e
        except requests.RequestException as e:
            if prev_state is not None:
                _update_secondary_metadata(user_id,prev_state["face_metadata"], prev_state["url"])
            raise e # goes to 5xx
        return bestIndex, euclidian, upd["uploaded_at"]
    else:
        upd, rows = _update_secondary_metadata(user_id,md[bestIndex])
        return bestIndex, euclidian, upd["uploaded_at"]


def extract_and_compare_metadatas(user_reference_metadata: list, pics, model):
    metadata_to_compare = [user_reference_metadata]
    try:
        metadata_to_compare.extend([
            distance.l2_normalize(DeepFace.represent(
                img_path=p,
                model_name=model,
                enforce_detection=False,
                detector_backend=_detector_low_quality,
                align=True,
                normalization="base",
            )[0]["embedding"]) for p in pics
        ])
    except ValueError as e:
        raise NoFaces("No faces detected")
    threshold = distance.findThreshold(model,_similarity_metric)
    bestIndex, euclidian = compare_metadatas(metadata_to_compare, threshold)
    return metadata_to_compare,bestIndex, euclidian, threshold

def compare_metadatas(metadatas: list, threshold: float):
    normalizedRefMetadata = metadatas.pop(0)
    distances = [distance.findEuclideanDistance(normalizedRefMetadata, md) for md in metadatas]
    m = distances[0]
    indexes = [(distances.index(d), m := min(d,m)) for d in distances if d <= threshold]
    indexes.sort(key=lambda x: x[1])
    if len(indexes) != len(metadatas) and len(indexes) < _min_images_with_emotions_to_proceed:
        return -1, min([i for i in distances if i > threshold])
    else:
        return indexes[0]

def loadImageFromStream(p):
    chunk_arr = np.frombuffer(p.read(), dtype=np.uint8)
    img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
    # im = Image.open(p.stream)
    # img = np.asarray(im)
    # return img
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

class UpsertException(Exception):
    def __init__(self, message):
        super().__init__(message)

class SessionTimeOutException(Exception):
    def __init__(self, message):
        super().__init__(message)

class NoDataException(Exception):
    def __init__(self, message):
        super().__init__(message)

class IOException(Exception):
    def __init__(self, message):
        super().__init__(message)

class WrongEmotionException(Exception):
    def __init__(self, message):
        super().__init__(message)

class SessionNotFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)

class RateLimitException(Exception):
    def __init__(self, message):
        super().__init__(message)

class SessionEndedException(Exception):
    def __init__(self, message):
        super().__init__(message)

class EmotionSequenceNotUsed(Exception):
    def __init__(self, message):
        super().__init__(message)

def _save_image(image, idx, user_id):
    local_path = f'{IMG_STORAGE_PATH}{user_id}/'
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    filename = f'{idx}.jpg'
    image.seek(0)
    image.save(os.path.join(local_path, filename))

def _remove_user_images(user_id):
    dirname = f"{IMG_STORAGE_PATH}{user_id}"
    if os.path.isdir(dirname) == False:
        return

    for filename in glob.glob(f"{dirname}/*.jpg"):
        os.remove(filename)

    if _count_user_images(user_id=user_id) == 0:
        shutil.rmtree(dirname)

def _count_user_images(user_id):
    return len(glob.glob(f"{IMG_STORAGE_PATH}{user_id}/*.jpg"))

def _send_best_images(token, user_id, best_images_indexes):
    multiple_files = []
    for img_idx in best_images_indexes:
        file_path = f"{IMG_STORAGE_PATH}{user_id}/{img_idx}.jpg"
        multiple_files.append(('image', (f"{img_idx}.jpg", open(file_path, 'rb'), 'image/jpeg')))

    response = requests.post(f"{BASE_SIMILARITY_ENDPOINT}{user_id}", files=multiple_files)

    return response.status_code == 200

def _get_unique_emotion(current_emotions_list: list):
    if len(current_emotions_list) == 0:
        choice = random.choice(list(DEFAULT_EMOTIONS_LIST.values()))

        return choice, False

    diff  = set(DEFAULT_EMOTIONS_LIST.values()) - set(current_emotions_list)
    if len(diff) == 0:
        return None, True

    return random.choice(list(diff)), False

def emotions(user_id):
    now = int(time.time()*1e9)
    usr = _get_user(user_id)

    if usr is not None:
        if usr['disabled_at'] is not None and usr['disabled_at'] > 0:
            return None, usr['session_id'], True

        if now - usr['session_started_at'] < LIMIT_RATE:
            raise RateLimitException(f'rate limit exception for user_id:{user_id}')

        _remove_user_images(user_id=user_id)

    session_id = str(uuid.uuid4())
    emotions_list = []
    for _ in range(0, DEFAULT_EMOTIONS_NUM):
        new_emotion, completed = _get_unique_emotion(list(emotions_list))
        emotions_list.append(new_emotion)
    res = _update_user(user_id, session_id, ",".join(emotions_list), now, 0, 0, 0)
    if res is False:
        raise UpsertException(f"can't insert user:{user_id}")

    return emotions_list, session_id, False

def add_additional_emotion(session_id, user_id):
    usr = _get_user(user_id)
    if usr is None:
        raise SessionNotFoundException(f"user:{user_id} not found")

    if usr['disabled_at'] is not None and usr['disabled_at'] > 0:
        return None, usr['session_id'], True

    if usr['session_id'] != session_id:
        raise SessionNotFoundException(f"wrong session:{session_id} for user:{user_id}")

    now = int(time.time()*1e9)
    if now - usr['session_started_at'] > SESSION_DURATION:
        raise SessionTimeOutException(f"session of user:{user_id} timed out")

    if usr['session_ended_at'] is not None and usr['session_ended_at'] > 0:
        raise SessionEndedException(f"session of user:{user_id} ended")

    current_emotions_list = usr['emotions'].split(',')
    if usr['emotion_sequence'] != len(current_emotions_list):
        raise EmotionSequenceNotUsed(f"session:{session_id} was finished for user:{user_id}")

    # TODO: do we need to rate limit here as well?

    emotion_sequence = usr['emotion_sequence']
    emotions_list = []
    new_emotion, completed = _get_unique_emotion(usr['emotions'].split(','))
    if completed is True:
        new_emotion, completed = _get_unique_emotion(list())
        emotions_list = [new_emotion]
        emotion_sequence = 0
    else:
        emotions_list = usr['emotions'].split(',')
        emotions_list.append(new_emotion)

    res = _update_user(user_id, session_id, ",".join(emotions_list), usr['session_started_at'], 0, 0, emotion_sequence)
    if res is False:
        raise UpsertException(f"can't update user:{user_id} by new emotion")

    return emotions_list, session_id, False

def process_images(user_id: str, session_id: str, images:list):
    usr = _get_user(user_id)
    if usr is None:
        raise SessionNotFoundException(f"user:{user_id} not found")

    if usr['session_ended_at'] is not None and usr['session_ended_at'] > 0:
        return False, True

    if usr['session_id'] != session_id:
        raise SessionNotFoundException(f"wrong session:{session_id} for user:{user_id}")

    now = int(time.time()*1e9)
    if now - usr['session_started_at'] > SESSION_DURATION:
        raise SessionTimeOutException(f"session of user:{user_id} timed out")

    current_emotions_list = usr['emotions'].split(',')
    if usr['emotion_sequence'] >= len(current_emotions_list):
        return False, False

    current_emotion = current_emotions_list[usr['emotion_sequence']]
    model = DeepFace.build_model("Emotion")
    emotions, scores = model.predict_multi_emotions(face_img_list=[loadImageFromStream(i) for i in images])
    emotion = max(emotions, key=emotions.count)
    if emotion != current_emotion:
        if _update_emotion_sequence_and_best_score(user_id=user_id, emotion_sequence=usr['emotion_sequence']+1, best_score = usr["best_pictures_score"]) is not True:
            raise UpsertException(f'can\'t update emotion sequence for user_id:{user_id}')

        return False, False

    images_count = _count_user_images(user_id=user_id)
    for idx, image in enumerate(images):
        try:
            _save_image(
                image=image,
                user_id=user_id,
                idx=idx + images_count
            )
        except OSError:
            raise IOException(f"can't save image for user:{user_id}")

    idx = images_count % 15
    neutral_idx = model.class_to_idx.get("Neutral")
    neutral_scores = [s[neutral_idx]for s in scores]
    usr["best_pictures_score"][idx * ONE_CALL_IMAGES_COUNT:(idx+1)*ONE_CALL_IMAGES_COUNT] = neutral_scores
    if _update_emotion_sequence_and_best_score(user_id=user_id, emotion_sequence=usr['emotion_sequence']+1, best_score = usr["best_pictures_score"]) is not True:
        raise UpsertException(f'can\'t update emotion sequence and best score for user_id:{user_id}')

    if _count_user_images(user_id=user_id) == ONE_CALL_IMAGES_COUNT * 3:
        best_indexies = []
        for idx in np.argpartition(usr['best_pictures_score'], -TOTAL_BEST_PICTURES)[-TOTAL_BEST_PICTURES:]:
            best_indexies.append(str(idx))

        _send_best_images(token='', user_id=user_id, best_images_indexes=best_indexies)

        _remove_user_images(user_id=user_id)

        _update_session_ended_at(user_id=user_id)

        return True, True

    return True, False
