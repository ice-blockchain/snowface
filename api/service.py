import uuid
import glob
import logging
import os
import random
import time
import shutil
from os.path import exists
from datetime import datetime
from webhook import callback, UnauthorizedFromWebhook
from flask import current_app
import exceptions

from milvus import (
    get_primary_metadata                   as _get_primary_metadata,
    get_secondary_metadata                 as _get_secondary_metadata,
    update_secondary_metadata              as _update_secondary_metadata,
    find_similar_users                     as _find_similar_users,
    set_primary_metadata                   as _set_primary_metadata,
    delete_metadatas                        as _delete_metadatas,
    update_user                            as _update_user,
    disable_user                           as _disable_user,
    get_user                               as _get_user,
    update_emotions_and_best_score         as _update_emotions_and_best_score,
    remove_session                         as _remove_session,
    update_last_negative_request_at        as _update_last_negative_request_at,
    get_users_collection                   as _get_users_collection
)
from PIL import Image
import cv2

import numpy as np, io, requests
from deepface.commons import functions, distance
from concurrent.futures import ThreadPoolExecutor, wait
from deepface import DeepFace
from deepface.commons import distance
from minio_uploader import put_secondary_photo, put_primary_photo, get_primary_photo, delete_photos as _delete_photos
import numpy as np

_model = "SFace"
_model_fallback = "ArcFace"#"Facenet" #"VGG-Face"
_detector_high_quality = "yunet"
_detector_low_quality = "yunet" # TODO: test with skip, if we gonna get proper photos from FE
_similarity_metric = "euclidean_l2"
_picture_extension = '.jpg'
_max_executor_workers = 2
_images_count_per_call = 15
_min_images_with_emotions_to_proceed = 1
_time_format = '%Y-%m-%dT%H:%M:%S.%fZ%Z'
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
    try:
        samplePerson = requests.get(
            url="https://thispersondoesnotexist.com/", verify=False
        )
        if samplePerson.status_code == 200:
            img = loadImageFromStream(io.BytesIO(samplePerson.content))
            DeepFace.represent(img_path=img, detector_backend=_detector_high_quality, model_name=_model)
            DeepFace.represent(img_path=img, detector_backend=_detector_high_quality, model_name=_model_fallback)
            emotion.predict_multi_emotions(face_img_list=[img])
    except requests.RequestException as e:
        logging.error(e, exc_info=e)


def set_primary_photo(current_user, user_id: str, photo_stream):
    now = time.time_ns()
    user = _get_user(user_id)
    if user is not None and user["disabled_at"] > 0:
        raise exceptions.UserDisabled(f"User {user_id} was disabled at {user['disabled_at']}")
    existing_md = _get_primary_metadata(user_id)
    if existing_md is not None:
        raise exceptions.MetadataAlreadyExists(f"User {user_id} already owns primary face uploaded at {existing_md['uploaded_at']}")
    img = loadImageFromStream(photo_stream)
    try:
        md = distance.l2_normalize(DeepFace.represent(
            img_path=img,
            model_name=_model,
            enforce_detection=True,
            detector_backend=_detector_high_quality,
            align=True,
            normalization="base",
            target_size=(640,640)
        )[0]["embedding"])
    except ValueError:
        raise exceptions.NoFaces("No faces detected, userId: {user_id}")
    threshold = distance.findThreshold(_model,_similarity_metric)
    similar_users, distances = _find_similar_users(user_id,md, threshold)
    if similar_users[0] != user_id:
        # make sure it is not a false positive, let's check other picture as well
        secondary_md = _get_secondary_metadata(similar_users[0])
        if secondary_md:
            bestIndex, euclidian = compare_metadatas([secondary_md["face_metadata"],md], threshold)
            if bestIndex != -1 and _disable_user(now, user_id):
                callback(
                    current_user=current_user,
                    primary_md=None,
                    secondary_md=None,
                    user={"disabled_at": now}
                )
                logging.info(f"Face {user_id} is matching with user {similar_users[0]}, distance {distances[0]} {euclidian} < {threshold}")
                raise exceptions.UserDisabled(f"Face {user_id} is matching with user {similar_users[0]}, distance {distances[0]} {euclidian} < {threshold}")
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
            if res["verified"]:
                disabled = _disable_user(now,user_id)
                if disabled:
                    callback(
                        current_user=current_user,
                        primary_md=None,
                        secondary_md=None,
                        user={"disabled_at": now}
                    )
                    logging.info(f"Face {user_id} is matching with user {similar_users[0]}, distance {distances[0]} {res['distance']} < {res['threshold']}")
                    raise exceptions.UserDisabled(f"Face {user_id} is matching with user {similar_users[0]}, distance {distances[0]} {res['distance']} < {res['threshold']}")
    url = put_primary_photo(user_id,photo_stream.stream)
    upd, rows = _set_primary_metadata(now, user_id, md, url)
    if rows > 0:
        try:
            callback(
                current_user=current_user,
                primary_md=upd,
                secondary_md=None,
                user=user
            )
        except UnauthorizedFromWebhook as e:
            _delete_metadatas([upd["user_picture_id"]])
            raise e
        except requests.RequestException as e:
            _delete_metadatas([upd["user_picture_id"]])
            raise e # goes to 5xx

def check_similarity_and_update_secondary_photo(current_user, user_id: str, raw_pics: list):
    now = time.time_ns()
    user_reference_metadata = _get_primary_metadata(user_id)
    if user_reference_metadata is None:
        raise exceptions.MetadataNotFound(f"User {user_id} have no registered primary metadata yet")
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
            raise exceptions.NotSameUser(f"user mismatch: distance is greater than {threshold}: {euclidian}")
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
        raise exceptions.NoFaces("No faces detected")
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
        "userId": user_id,
        "primaryPhotoUploaded": primaryUploaded,
        "lastVerified": lastVerified,
        "disabled": disabled is not None and disabled["disabled_at"] > 0
    }

def _count_user_images(user_id):
    return len(glob.glob(f"{current_app.config['IMG_STORAGE_PATH']}{user_id}/*{_picture_extension}"))

def _remove_user_images(user_id):
    dirname = f"{current_app.config['IMG_STORAGE_PATH']}{user_id}"
    if os.path.isdir(dirname) == False:
        return

    for filename in glob.glob(f"{dirname}/*{_picture_extension}"):
        os.remove(filename)

    if _count_user_images(user_id) == 0:
        shutil.rmtree(dirname)

def _remove_not_best_user_images(img_storage_path, user_id, best_images_indexes):
    dirname = f"{img_storage_path}{user_id}"

    if os.path.isdir(dirname) == False:
        return False

    all_images = glob.glob(f"{dirname}/*{_picture_extension}")
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
    now = time.time_ns()
    usr = _get_user(user_id)

    if usr is not None:
        if usr['disabled_at'] is not None and usr['disabled_at'] > 0:
            raise exceptions.UserDisabled(f"user:{usr['user_id']} disabled")
        if usr['last_negative_request_at'] > 0 and now - usr['last_negative_request_at'] <= current_app.config['LIMIT_RATE_NEGATIVE']:
            raise exceptions.NegativeRateLimitException(f"limit rate time didn't pass from the last negative try for user:{user_id} time: {usr['last_negative_request_at']}")

        secondary = _get_secondary_metadata(user_id)
        if secondary is not None and secondary['uploaded_at'] is not None and now - secondary['uploaded_at'] <= current_app.config['LIMIT_RATE']:
            raise exceptions.RateLimitException(f"rate limit exception for user_id:{user_id}, already passed the liveness at {secondary['uploaded_at']}")

        _remove_session(user_id)
        _remove_user_images(user_id)

    session_id = str(uuid.uuid4())
    emotions_list = []
    for _ in range(0, current_app.config['INITIAL_EMOTION_COUNT']):
        new_emotion, completed = _get_unique_emotion(list(emotions_list))
        if completed:
            new_emotion, _ = _get_unique_emotion([emotions_list[-1]])
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
        raise exceptions.UpsertException(f"can't insert user:{user_id}")

    return emotions_list, session_id, datetime.utcfromtimestamp((now + current_app.config['SESSION_DURATION'])/1e9).strftime(_time_format)

def _validate_session(usr, user_id, session_id, now):
    if usr is None:
        raise exceptions.SessionNotFoundException(f"user:{user_id} not found")

    if usr['disabled_at'] is not None and usr['disabled_at'] > 0:
        raise exceptions.UserDisabled(f"user:{usr['user_id']} disabled")

    if usr['session_id'] != session_id:
        raise exceptions.SessionNotFoundException(f"wrong session:{session_id} for user:{usr['user_id']}")

    if now - usr['session_started_at'] >= current_app.config['SESSION_DURATION']:
        raise exceptions.SessionTimeOutException(f"session of user:{usr['user_id']} timed out")

def _generate_emotions(usr):
    emotions_list = usr['emotions'].split(',')
    new_emotion, completed = _get_unique_emotion(usr['emotions'].split(','))
    if completed is True:
        new_emotion, _ = _get_unique_emotion(list())
    emotions_list.append(new_emotion)

    return ",".join(emotions_list)

def _add_additional_emotion(usr):
    user_id = usr["user_id"]
    current_emotions_list = usr['emotions'].split(',')
    emotion_sequence = usr['emotion_sequence']
    if emotion_sequence != len(current_emotions_list) or emotion_sequence >= current_app.config['MAX_EMOTION_COUNT']:
        return ",".join(current_emotions_list)

    emotions_list = _generate_emotions(usr)

    return emotions_list

def _generate_best_scores(usr, current_emotions_list,scores, model, images_count, session_success):
    idx = int(images_count / _images_count_per_call)

    neutral_idx = model.class_to_idx.get("neutral")
    neutral_scores = [s[neutral_idx]for s in scores]
    usr['best_pictures_score'][idx * _images_count_per_call:(idx+1)*_images_count_per_call] = neutral_scores
    usr['emotion_sequence'] = usr['emotion_sequence']+1
    if usr['emotion_sequence'] >= len(current_emotions_list) and (not session_success):
        usr['emotions'] = _add_additional_emotion(usr)
    if _update_emotions_and_best_score(
        usr=usr,
        emotions = usr['emotions'],
        emotion_sequence=usr['emotion_sequence'],
        best_score=usr["best_pictures_score"]
    ) is not True:
        raise exceptions.UpsertException(f"can't update emotion sequence and best score for user_id:{usr['user_id']}")
    return usr['emotions']

def _predict(usr, model, images, now, awaited_emotion):
    t = time.time()
    face_img_list = []
    for img in images:
        loaded_image = loadImageFromStream(img)
        if loaded_image.shape[0] != model.img_size or loaded_image.shape[1] != model.img_size:
            raise exceptions.WrongImageSizeException(f"wrong image size for user:{usr['user_id']}, session:{usr['session_id']}")

        face_img_list.append(loaded_image)
    awaited_idx = model.class_to_idx[awaited_emotion]
    emotions, scores = model.predict_multi_emotions(face_img_list=face_img_list, logits = False)
    averages = np.mean(scores, axis=0)
    logging.debug(f"[U:{usr['user_id']}][S:{usr['session_id']}] Prediction took {time.time()-t}")
    return averages[awaited_idx]*100.0, scores, model.idx_to_class[np.argmax(averages)], max(averages)*100, dict([(v,averages[k]*100) for k,v in model.idx_to_class.items()])

def _rollback_images_devide_modulo_15(user_id: str):
    images_count = _count_user_images(user_id)
    if images_count > _images_count_per_call:
        obsolete = images_count % _images_count_per_call
        if obsolete > 0:
            local_path = f"{current_app.config['IMG_STORAGE_PATH']}{user_id}/"
            for idx in range(images_count - obsolete, images_count):
                os.remove(f"{local_path}{idx}{_picture_extension}")
    elif images_count > 0 and images_count < _images_count_per_call:
        _remove_user_images(user_id)

def _save_image(image, idx, user_id):
    try:
        local_path = f"{current_app.config['IMG_STORAGE_PATH']}{user_id}/"
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        filename = f'{idx}{_picture_extension}'
        image.seek(0)
        image.save(os.path.join(local_path, filename))
    except Exception as e:
        _rollback_images_devide_modulo_15(user_id)
        raise e

def _send_best_images(img_storage_path:str, similarity_server:str, token, user_id, best_images_indexes):
    files = []
    for img_idx in best_images_indexes:
        file_path = f"{img_storage_path}{user_id}/{img_idx}{_picture_extension}"
        files.append(('image', (f"{img_idx}{_picture_extension}", open(file_path, 'rb'), 'image/jpeg')))

    response = requests.post(
        url=f"{similarity_server[:-1] if similarity_server.endswith('/') else similarity_server}/v1w/face-auth/similarity/{user_id}",
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
                current_app.config['SIMILARITY_SERVER'],
                token, usr['user_id'],
                best_indexes
            )
        ]

        wait(futures)

    _remove_user_images(user_id=usr['user_id'])
    _remove_session(user_id=usr['user_id'])

def process_images(token: str, user_id: str, session_id: str, images:list):
    now = time.time_ns()
    usr = _get_user(user_id)

    _validate_session(usr=usr, user_id=user_id, session_id=session_id, now=now)

    if usr['last_negative_request_at'] > 0 and now - usr['last_negative_request_at'] <= current_app.config['LIMIT_RATE_NEGATIVE']:
        raise exceptions.NegativeRateLimitException(f"limit rate time didn't pass from the last negative try for user:{user_id} time: {usr['last_negative_request_at']}")

    current_emotions_list = usr['emotions'].split(',')

    if usr['emotion_sequence'] >= current_app.config['MAX_EMOTION_COUNT']:
        if _update_last_negative_request_at(usr=usr, now=now) is False:
            raise exceptions.UpsertException(f"update last negative request time failed for user:{user_id}")

        return False, True, usr['emotions']

    current_emotion = current_emotions_list[usr['emotion_sequence']]
    if usr['emotion_sequence'] >= len(current_emotions_list):
        usr['emotions'] = _add_additional_emotion(usr)
        if _update_emotions_and_best_score(
                usr=usr,
                emotions = usr['emotions'],
                emotion_sequence=usr['emotion_sequence'],
                best_score=usr['best_pictures_score'],
                last_negative_request_at = usr['last_negative_request_at']
        ) is False:
            raise exceptions.UpsertException(f"can't update emotion sequence for user_id:{user_id}")
        return False, False, usr['emotions']
    model = DeepFace.build_model("Emotion")
    awaited_score, scores, max_emotion, max_score, averages = _predict(
        usr=usr,
        model=model,
        images=images,
        now=now,
        awaited_emotion = current_emotion,
    )
    relative_score = awaited_score*100.0/max_score
    logging.info(f"[U:{user_id}][S:{session_id}] awaited {current_emotion}/{awaited_score} it is {relative_score} of ({max_emotion}/{max_score}=100)  < {current_app.config['TARGET_EMOTION_SCORE']} all:{averages}")
    if relative_score < current_app.config['TARGET_EMOTION_SCORE']:
        usr['emotion_sequence'] = usr['emotion_sequence']+1
        session_ended = usr['emotion_sequence'] >= current_app.config['MAX_EMOTION_COUNT']
        if session_ended:
            usr['last_negative_request_at'] = now
        usr['best_pictures_score'][0] = usr['emotion_sequence']
        if usr['emotion_sequence'] >= len(current_emotions_list):
            usr['emotions'] = _add_additional_emotion(usr)
        if _update_emotions_and_best_score(
            usr=usr,
            emotions = usr['emotions'],
            emotion_sequence=usr['emotion_sequence'],
            best_score=usr['best_pictures_score'],
            last_negative_request_at = usr['last_negative_request_at']
        ) is False:
            raise exceptions.UpsertException(f"can't update emotion sequence for user_id:{user_id}")
        return False, session_ended, usr['emotions']

    images_count = _count_user_images(user_id)
    for idx, img in enumerate(images):
        _save_image(
            image=img,
            user_id=user_id,
            idx=idx + images_count
        )
    session_success = _count_user_images(user_id) == _images_count_per_call * current_app.config['TARGET_EMOTION_COUNT']
    session_ended = usr['emotion_sequence'] >= current_app.config['MAX_EMOTION_COUNT']
    emotions = _generate_best_scores(
        usr=usr,
        current_emotions_list=current_emotions_list,
        scores=scores,
        model=model,
        images_count=images_count,
        session_success = session_success
    )

    if session_success:
        _finish_session(usr, token)

        return True, True, emotions

    return True, session_ended, emotions

def delete_temporary_user_data(user_id:str):
    _remove_user_images(user_id = user_id)
    _remove_session(user_id)
def delete_user_photos_and_metadata(user_id:str):
    errs = _delete_photos(user_id)
    if len(errs) > 0:
        raise Exception(str(errs))
    if _delete_metadatas([f"{user_id}~0", f"{user_id}~1"]) == 0:
        raise exceptions.MetadataNotFound(f"face metadata for userId {user_id} was not deleted")

def proxy_delete(current_user):
    similarity_server = current_app.config['SIMILARITY_SERVER']
    api_token = current_app.config['METADATA_UPDATED_SECRET']
    response = requests.delete(
        url=f"{similarity_server[:-1] if similarity_server.endswith('/') else similarity_server}/v1w/face-auth/",
        headers={"X-API-Key": api_token,"Authorization": f"Bearer {current_user.raw_token}", "X-Account-Metadata": current_user.metadata}
    )
    return response.content, response.status_code