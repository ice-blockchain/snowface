import uuid
import glob
import os
import random
import time
import shutil
from os.path import exists
from webhook import callback, UnauthorizedFromWebhook
from flask import current_app

from milvus import (
    get_primary_metadata                   as _get_primary_metadata,
    get_secondary_metadata                 as _get_secondary_metadata,
    update_secondary_metadata              as _update_secondary_metadata,
    find_similar_users                     as _find_similar_users,
    set_primary_metadata                   as _set_primary_metadata,
    delete_metadata                        as _delete_metadata,
    update_user                            as _update_user,
    disable_user                           as _disable_user,
    get_user                               as _get_user,
    update_emotion_sequence_and_best_score as _update_emotion_sequence_and_best_score,
    remove_session                         as _remove_session,
    update_last_negative_request_at        as _update_last_negative_request_at
)
from PIL import Image
import cv2

import numpy as np, io, requests
from deepface.commons import functions, distance
from concurrent.futures import ThreadPoolExecutor, wait
from deepface import DeepFace
from deepface.commons import distance
from minio_uploader import put_secondary_photo, put_primary_photo, get_primary_photo
import numpy as np

_model = "SFace"
_model_fallback = "ArcFace"#"Facenet" #"VGG-Face"
_detector_high_quality = "yunet"
_detector_low_quality = "yunet" # TODO: test with skip, if we gonna get proper photos from FE
_similarity_metric = "euclidean_l2"
_max_executor_workers = 2
_default_emotions_num = 3
_images_count_per_call = 15
_min_images_with_emotions_to_proceed = 1
_default_emotions_list = DeepFace.build_model("Emotion").idx_to_class

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
    emotion = DeepFace.build_model("Emotion")
    samplePerson = requests.get(
        url="https://thispersondoesnotexist.com/", verify=False
    )
    if samplePerson.status_code == 200:
        img = loadImageFromStream(io.BytesIO(samplePerson.content))
        DeepFace.represent(img_path=img, detector_backend=_detector_high_quality, model_name=_model)
        DeepFace.represent(img_path=img, detector_backend=_detector_high_quality, model_name=_model_fallback)
        emotion.predict(img)


def set_primary_photo(current_user, user_id: str, photo_stream):
    now = time.time_ns()
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
            if bestIndex != -1 and _disable_user(now, user_id):
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
                disabled = _disable_user(now,user_id)
                if disabled:
                    callback(current_user, None,None, {"disabled_at": now})
                    raise UserDisabled(f"Face is matching with user {similar_users[0]}, distance {distances[0]} {res['distance']} < {res['threshold']}")
    url = put_primary_photo(user_id,photo_stream.stream)
    upd, rows = _set_primary_metadata(now, user_id, md, url)
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
    now = time.time_ns()
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
    upd, rows = _update_secondary_metadata(now,user_id,md[bestIndex], url)
    if rows > 0:
        try:
            callback(current_user, user_reference_metadata,upd,_get_user(user_id))
        except UnauthorizedFromWebhook as e:
            if prev_state is not None:
                _update_secondary_metadata(prev_state["uploaded_at"],user_id,prev_state["face_metadata"], prev_state["url"])
            raise e
        except requests.RequestException as e:
            if prev_state is not None:
                _update_secondary_metadata(prev_state["uploaded_at"],user_id,prev_state["face_metadata"], prev_state["url"])
            raise e # goes to 5xx
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

class WrongEmotionException(Exception):
    def __init__(self, message):
        super().__init__(message)

class SessionNotFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)

class RateLimitException(Exception):
    def __init__(self, message):
        super().__init__(message)

class WrongImageSizeException(Exception):
    def __init__(self, message):
        super().__init__(message)

def _count_user_images(user_id):
    return len(glob.glob(f"{current_app.config['IMG_STORAGE_PATH']}{user_id}/*.jpg"))

def _remove_user_images(user_id):
    dirname = f"{current_app.config['IMG_STORAGE_PATH']}{user_id}"
    if os.path.isdir(dirname) == False:
        return

    for filename in glob.glob(f"{dirname}/*.jpg"):
        os.remove(filename)

    if _count_user_images(user_id) == 0:
        shutil.rmtree(dirname)

def _remove_not_best_user_images(img_storage_path, user_id, best_images_indexes):
    dirname = f"{img_storage_path}{user_id}"

    if os.path.isdir(dirname) == False:
        return False

    all_images = glob.glob(f"{dirname}/*.jpg")
    for full_name in all_images:
        filename = full_name[full_name.rfind('/')+1:]
        parts = filename.split('.')
        if parts[0] not in best_images_indexes:
            os.remove(full_name)

def _get_unique_emotion(current_emotions_list: list):
    if len(current_emotions_list) == 0:
        choice = random.choice(list(_default_emotions_list.values()))

        return choice.lower(), False

    diff  = set(_default_emotions_list.values()) - set(current_emotions_list)
    if len(diff) == 0:
        return None, True

    choice = random.choice(list(diff))

    return choice.lower(), False

def emotions(user_id):
    now = int(time.time()*1e9)
    usr = _get_user(user_id)

    if usr is not None:
        if usr['disabled_at'] is not None and usr['disabled_at'] > 0:
            raise UserDisabled(f"user:{usr['user_id']} disabled")

        if now - usr['session_started_at'] <= current_app.config['LIMIT_RATE']:
            raise RateLimitException(f'rate limit exception for user_id:{user_id}')

        _remove_session(user_id)
        _remove_user_images(user_id)

    session_id = str(uuid.uuid4())
    emotions_list = []
    for _ in range(0, _default_emotions_num):
        new_emotion, _ = _get_unique_emotion(list(emotions_list))
        emotions_list.append(new_emotion)

    res = _update_user(
        user_id=user_id,
        session_id=session_id,
        emotions=",".join(emotions_list),
        session_started_at=now,
        disabled_at=0,
        emotion_sequence=0,
        best_pictures_score=np.array([0.0]*45),
        now=now
    )
    if res is False:
        raise UpsertException(f"can't insert user:{user_id}")

    return emotions_list, session_id

def _validate_session(usr, user_id, session_id):
    if usr is None:
        raise SessionNotFoundException(f"user:{user_id} not found")

    if usr['disabled_at'] is not None and usr['disabled_at'] > 0:
        raise UserDisabled(f"user:{usr['user_id']} disabled")

    now = int(time.time()*1e9)
    if now - usr['last_negative_request_at'] <= current_app.config['LIMIT_RATE_NEGATIVE']:
        raise RateLimitException(f"limit rate time didn't pass from the last negative request for user:{user_id}")

    if usr['session_id'] != session_id:
        raise SessionNotFoundException(f"wrong session:{session_id} for user:{usr['user_id']}")

    if now - usr['session_started_at'] >= current_app.config['SESSION_DURATION']:
        raise SessionTimeOutException(f"session of user:{usr['user_id']} timed out")

def _generate_emotions(usr):
    emotions_list = usr['emotions'].split(',')
    new_emotion, completed = _get_unique_emotion(usr['emotions'].split(','))
    if completed is True:
        new_emotion, _ = _get_unique_emotion(list())
    emotions_list.append(new_emotion)

    return emotions_list

def add_additional_emotion(session_id, user_id):
    now = int(time.time()*1e9)
    usr = _get_user(user_id)

    _validate_session(usr=usr, user_id=user_id, session_id=session_id)

    current_emotions_list = usr['emotions'].split(',')
    if usr['emotion_sequence'] != len(current_emotions_list) or len(current_emotions_list) >= current_app.config['MAX_EMOTION_COUNT']:
        return current_emotions_list, session_id

    emotion_sequence = usr['emotion_sequence']
    emotions_list = _generate_emotions(usr)

    if _update_user(
        user_id=user_id,
        session_id=session_id,
        emotions=",".join(emotions_list),
        session_started_at=usr['session_started_at'],
        disabled_at=0,
        emotion_sequence=emotion_sequence,
        best_pictures_score=usr['best_pictures_score'],
        now=now
    ) is False:
        raise UpsertException(f"can't update user:{user_id} by new emotion")

    return emotions_list, session_id

def _generate_best_scores(usr, scores, model, images_count):
    idx = int(images_count / _images_count_per_call)

    neutral_idx = model.class_to_idx.get("neutral")
    neutral_scores = [s[neutral_idx]for s in scores]
    usr['best_pictures_score'][idx * _images_count_per_call:(idx+1)*_images_count_per_call] = neutral_scores

    if _update_emotion_sequence_and_best_score(
        user_id=usr['user_id'],
        emotion_sequence=usr['emotion_sequence']+1,
        best_score=usr["best_pictures_score"]
    ) is not True:
        raise UpsertException(f"can't update emotion sequence and best score for user_id:{usr['user_id']}")

def _predict(user_id, session_id, model, images, now):
    face_img_list = []
    for img in images:
        loaded_image = loadImageFromStream(img)
        if loaded_image.shape[0] != model.img_size or loaded_image.shape[1] != model.img_size:
            if _update_last_negative_request_at(user_id=user_id, now=now) is False:
                raise UpsertException(f"update last negative request time failed for user:{user_id}")

            raise WrongImageSizeException(f"wrong image size for user:{user_id}, session:{session_id}")

        face_img_list.append(loaded_image)

    emotions, scores = model.predict_multi_emotions(face_img_list=face_img_list)
    emotion = max(emotions, key=emotions.count)

    return emotion, scores

def _rollback_images_device_modulo_15(user_id: str):
    images_count = _count_user_images(user_id)
    if images_count > _images_count_per_call:
        obsolete = images_count % _images_count_per_call
        if obsolete > 0:
            local_path = f"{current_app.config['IMG_STORAGE_PATH']}{user_id}/"
            print('images_count - obsolete: ', images_count - obsolete)
            print('images_count: ', images_count)
            for idx in range(images_count - obsolete, images_count):
                os.remove(f"{local_path}{idx}.jpg")
    elif images_count > 0 and images_count < _images_count_per_call:
        _remove_user_images(user_id)

def _save_image(image, idx, user_id):
    try:
        local_path = f"{current_app.config['IMG_STORAGE_PATH']}{user_id}/"
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        filename = f'{idx}.jpg'
        image.seek(0)
        image.save(os.path.join(local_path, filename))
    except Exception as e:
        _rollback_images_device_modulo_15(user_id)
        raise e

def _send_best_images(img_storage_path, base_similarity_endpoint, token, user_id, best_images_indexes):
    files = []
    for img_idx in best_images_indexes:
        file_path = f"{img_storage_path}{user_id}/{img_idx}.jpg"
        files.append(('image', (f"{img_idx}.jpg", open(file_path, 'rb'), 'image/jpeg')))

    response = requests.post(
        url=f"{base_similarity_endpoint}{user_id}",
        files=files,
        headers={"Authorization": f"Bearer {token}"}
    )

    return response.status_code == 200

def _finish_session(usr, token):
    best_indexes = []
    for idx in np.argpartition(usr['best_pictures_score'], -current_app.config['TOTAL_BEST_PICTURES'])[-current_app.config['TOTAL_BEST_PICTURES']:]:
        best_indexes.append(str(idx))

    with ThreadPoolExecutor(max_workers=_max_executor_workers) as executor:
        futures = [
            executor.submit(
                _remove_not_best_user_images,
                current_app.config['IMG_STORAGE_PATH'],
                usr['user_id'],
                best_indexes
            ),
            executor.submit(
                _send_best_images,
                current_app.config['IMG_STORAGE_PATH'],
                current_app.config['BASE_SIMILARITY_ENDPOINT'],
                token, usr['user_id'],
                best_indexes
            )
        ]

        wait(futures)

    _remove_user_images(user_id=usr['user_id'])
    _remove_session(user_id=usr['user_id'])

def process_images(token: str, user_id: str, session_id: str, images:list):
    now = int(time.time()*1e9)
    usr = _get_user(user_id)

    _validate_session(usr=usr, user_id=user_id, session_id=session_id)

    current_emotions_list = usr['emotions'].split(',')
    if usr['emotion_sequence'] >= len(current_emotions_list):
        if _update_last_negative_request_at(user_id) is False:
            raise UpsertException(f"update last negative request time failed for user:{user_id}")

        return False, False

    current_emotion = current_emotions_list[usr['emotion_sequence']]
    if len(current_emotions_list) >= current_app.config['MAX_EMOTION_COUNT']:
        if _update_last_negative_request_at(user_id) is False:
            raise UpsertException(f"update last negative request time failed for user:{user_id}")

        return False, True

    model = DeepFace.build_model("Emotion")
    emotion, scores = _predict(
        user_id=user_id,
        session_id=session_id,
        model=model,
        images=images,
        now=now
    )

    if emotion != current_emotion:
        if _update_emotion_sequence_and_best_score(
            user_id=user_id,
            emotion_sequence=usr['emotion_sequence']+1,
            best_score=usr['best_pictures_score']
        ) is False:
            raise UpsertException(f"can\'t update emotion sequence for user_id:{user_id}")

        if _update_last_negative_request_at(user_id=user_id, now=now) is False:
            raise UpsertException(f"update last negative request time failed for user:{user_id}")

        return False, False

    images_count = _count_user_images(user_id)
    for idx, img in enumerate(images):
        _save_image(
            image=img,
            user_id=user_id,
            idx=idx + images_count
        )

    _generate_best_scores(
        usr=usr,
        scores=scores,
        model=model,
        images_count=images_count
    )

    if _count_user_images(user_id) == _images_count_per_call * _default_emotions_num:
        _finish_session(usr, token)

        return True, True

    return True, False
