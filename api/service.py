import json
import uuid
import glob
import logging
import os
import random
import time
import shutil
from os.path import exists
from datetime import datetime
from webhook import callback, callback_migrate_phone_login, UnauthorizedFromWebhook, MigratePhoneLoginWebhookBadRequest, MigratePhoneLoginWebhookConflict, MigratePhoneLoginWebhookRateLimit
from flask import current_app
import exceptions
import jwt
from faces import (
    get_primary_metadata                   as _get_primary_metadata,
    get_secondary_metadata                 as _get_secondary_metadata,
    update_secondary_metadata              as _update_secondary_metadata,
    find_similar_users                     as _find_similar_users,
    set_primary_metadata                   as _set_primary_metadata
)

from users import (
    update_user                               as _update_user,
    get_user                                  as _get_user,
    update_emotions_and_best_score            as _update_emotions_and_best_score,
    remove_session                            as _remove_session,
    update_last_negative_request_at           as _update_last_negative_request_at,
    get_expired_sessions                      as _get_expired_sessions,
    decrease_available_retries                as _decrease_available_retries,
    enable_user                               as _enable_user,
    get_disabled_user_for_selfie_reprocessing as _get_disabled_user_for_selfie_reprocessing,
    put_disabled_user_for_selfie_reprocessing as _put_disabled_user_for_selfie_reprocessing,
    get_admin_token                           as _get_admin_token,
    unregister_wrongfully_disabled_users_worker   as _unregister_wrongfully_disabled_users_worker,
    register_wrongfully_disabled_users_worker   as _register_wrongfully_disabled_users_worker,
    mark_user_for_manual_review                as _mark_user_for_manual_review,
    is_review_disabled                         as _is_review_disabled,
    get_user_similarity_resp                   as _get_user_similarity_resp,
    put_user_similarity_resp                   as _put_user_similarity_resp,
    update_secondary_metadata_pending          as _update_secondary_metadata_pending,
    get_pending_face                           as _get_pending_face,
    add_possible_duplicate_with                as _add_possible_duplicate_with,
    register_unique_email_and_phone_number       as _register_unique_email_and_phone_number
)
import review, primary_photo

from PIL import Image
import cv2

import numpy as np, io, requests
from deepface.commons import distance
from concurrent.futures import ThreadPoolExecutor, wait
from deepface import DeepFace
from minio_uploader import (put_secondary_photo, put_primary_photo, get_primary_photo, get_secondary_photo,
                            delete_photos as _delete_photos,
                            put_disabled_photo as _put_disable_photo,
                            get_disabled_photo as _get_disabled_photo,
                            get_review_photo   as _get_review_photo,
                            put_review_photo   as _put_review_photo
)
import metrics
import numpy as np

from auth import Token

from flask_executor import Executor

_model = primary_photo._model
_model_fallback = primary_photo._model_fallback #"Facenet" #"VGG-Face"
_detector_high_quality = primary_photo._detector_high_quality
_detector_low_quality = "yunet" # TODO: test with skip, if we gonna get proper photos from FE
_similarity_metric = "euclidean_l2"
_picture_extension = '.jpg'
_max_executor_workers = 2
_images_count_per_call = 15
_min_images_with_emotions_to_proceed = 1
_time_format = '%Y-%m-%dT%H:%M:%S.%fZ%Z'
_invalidated_session='00000000-0000-0000-0000-000000000000'
_user_not_the_same = "USER_NOT_THE_SAME"
_default_emotions_list = [
    ['anger', 'surprise', 'happiness', 'neutral'],
    ['contempt','sadness', 'fear'],
    ['disgust']
]
_default_session_duration = 600

__admin_token = None
__stop_wrongfully_disabled_users_worker = False

_invalid_properties = "INVALID_PROPERTIES"
_conflict_with_another_user = "CONFLICT_WITH_ANOTHER_USER"
_too_many_requests = "TOO_MANY_REQUESTS"

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
    if current_app.config["MINIO_URI"]:
        DeepFace.build_model(_model_fallback)
    logging.warning(f"building emotion model PID:{os.getpid()}")
    emotion = DeepFace.build_model("Emotion")
    logging.warning(f"emotion model built PID:{os.getpid()}")
    try:
        samplePerson = requests.get(
            url="https://thispersondoesnotexist.com/", verify=False, timeout=10
        )
        if samplePerson.status_code == 200:
            img = loadImageFromStream(io.BytesIO(samplePerson.content))
            if img is not None:
                if current_app.config["MINIO_URI"]:
                    DeepFace.represent(img_path=img, detector_backend=_detector_high_quality, model_name=_model_fallback, enforce_detection=False)
                else:
                    logging.warning(f"initing face extractor {_detector_high_quality} PID:{os.getpid()}")
                    DeepFace.extract_faces(img_path=img,detector_backend=_detector_high_quality, enforce_detection=False)
                    logging.warning(f"face extractor initialized {_detector_high_quality} PID:{os.getpid()}")
                logging.warning(f"initializing emotion extractor PID:{os.getpid()}")
                emotion.predict_multi_emotions(face_img_list=[img])
                logging.warning(f"initialized emotion extractor PID:{os.getpid()}")
    except requests.RequestException as e:
        logging.error(e, exc_info=e)

def set_primary_photo_internal(now: int,user_id: str, photo_stream, attempt):
    img_to_represent, md, sface_md = primary_photo.extract_metadatas(user_id,photo_stream)
    threshold = current_app.config['PRIMARY_PHOTO_ARCFACE_DISTANCE']
    similar_users, distances = _find_similar_users(user_id, md, threshold)
    if similar_users[0] != user_id:
        similar_user_md = _get_primary_metadata(similar_users[0], model=_model_fallback, search_growing=False)["face_metadata"]
        # make sure it is not a false positive, let's check other picture as well
        secondary_md = _get_secondary_metadata(similar_users[0], model=_model)
        existing_arcface_secondary_md = _get_secondary_metadata(similar_users[0], model=_model_fallback)
        if existing_arcface_secondary_md:
            existing_arcface_secondary_md=existing_arcface_secondary_md["face_metadata"]
        if secondary_md:
            secondary_md = secondary_md["face_metadata"]
            bestIndex, euclidian, bestNotFittingIndex = compare_metadatas([sface_md,secondary_md], current_app.config["PRIMARY_PHOTO_ARCFACE_DISTANCE"])
            if bestIndex != -1:
                raise exceptions.FailedTryToDisable(message=f"[secondary photo] Face {user_id} attempt:{attempt} is matching with user {similar_users[0]}, distance ({distances[0]}) < {threshold}, ({euclidian}) < {current_app.config['PRIMARY_PHOTO_ARCFACE_DISTANCE']}",
                    sface_distance=distances[0],
                    arface_distance=euclidian,
                    matching_user_id=similar_users[0],
                    arcface_meta=md,
                    similar_users = similar_users
                )
            else:
                if not existing_arcface_secondary_md:
                    most_similar_user_photo = get_secondary_photo(similar_users[0])
                    if most_similar_user_photo:
                        existing_arcface_secondary_md = distance.l2_normalize(DeepFace.represent(
                            img_path=loadImageFromStream(io.BytesIO(most_similar_user_photo)),
                            model_name=_model_fallback,
                            detector_backend="yunet",
                            normalization="base",
                            target_size=(112, 112),
                        )[0]["embedding"])
                        _update_secondary_metadata(now,user_id=similar_users[0],metadata=existing_arcface_secondary_md,model=_model_fallback, email=similar_user_md.get("email",""), phone_number=similar_user_md.get("phone_number", ""))
                if existing_arcface_secondary_md is not None:
                    bestIndex, euclidian, bestNotFittingIndex = compare_metadatas([md,existing_arcface_secondary_md], current_app.config["PRIMARY_PHOTO_ARCFACE_DISTANCE"])
                    if bestIndex != -1:
                        raise exceptions.FailedTryToDisable(message=f"[secondary photo] Face {user_id} attempt:{attempt} is matching with user {similar_users[0]}, distance ({distances[0]}) < {threshold}, ({euclidian}) < {current_app.config['PRIMARY_PHOTO_ARCFACE_DISTANCE']}",
                                                            sface_distance=distances[0],
                                                            arface_distance=euclidian,
                                                            matching_user_id=similar_users[0],
                                                            arcface_meta=md,
                                                            similar_users = similar_users
                                                            )
                    else:
                        logging.info(f"[primary photo, positive arcface] Face {user_id}, attempt:{attempt}: {similar_users[0]}, distance ({euclidian}) < {current_app.config['PRIMARY_PHOTO_ARCFACE_DISTANCE']}")
                        return md, sface_md, similar_users

        # that similar user dont have 2nd pic yet,but we can re-check with fallback model
        bestIndex, euclidian, bestNotFittingIndex = compare_metadatas([md,similar_user_md], current_app.config["PRIMARY_PHOTO_ARCFACE_DISTANCE"])

        if bestIndex != -1:
            logging.info(f"[primary photo, no secondary] Face {user_id}, attempt:{attempt} is matching with user {similar_users[0]}, distance {euclidian} < {current_app.config['PRIMARY_PHOTO_ARCFACE_DISTANCE']}")

            raise exceptions.FailedTryToDisable(
                message=f"[primary photo, no secondary] Face {user_id}, attempt:{attempt} is matching with user {similar_users[0]}, distance {euclidian} < {current_app.config['PRIMARY_PHOTO_ARCFACE_DISTANCE']}",
                sface_distance=distances[0],
                arface_distance=euclidian,
                matching_user_id=similar_users[0],
                arcface_meta=md,
                similar_users = similar_users
            )
        else:
            logging.info(f"[primary photo, no secondary, positive arface] Face {user_id}, attempt:{attempt}: {similar_users[0]}, distance ({euclidian}) < {current_app.config['PRIMARY_PHOTO_ARCFACE_DISTANCE']}")
    else:
        logging.info(f"[milvus, positive] Face {user_id}, attempt:{attempt}: {similar_users}, distance ({distances}) < {current_app.config['PRIMARY_PHOTO_SFACE_DISTANCE']}")
    if similar_users[0] == user_id:
        similar_users.pop()
    return md, sface_md, similar_users

def set_primary_photo(current_user, client_ip, user_id: str, photo_stream, email: str, phone_number: str):
    now = time.time_ns()
    user = _get_user(user_id, search_growing=False)
    if user is not None and user["disabled_at"] > 0:
        raise exceptions.UserDisabled(f"User {user_id} was disabled at {user['disabled_at']}")
    if user is not None and user["possible_duplicate_with"]:
        raise exceptions.RateLimitException(f"User was forwarded to manual review")
    existing_md = _get_primary_metadata(user_id, search_growing=False, model=_model_fallback)
    if existing_md is not None:
        # retry webhook as normally user should not get here
        try:
            callback(
                current_user=current_user,
                primary_md=existing_md,
                secondary_md=None,
                user=user
            )
            return
        except UnauthorizedFromWebhook as e:
            raise e
        except Exception as e:
            raise exceptions.MetadataAlreadyExists(f"User {user_id} already owns primary face uploaded at {existing_md['uploaded_at']} but exception occured while syncing webhook: {str(e)}")

    if user is not None and user["available_retries"] != 0:
        attempt = user["available_retries"]
    else:
        attempt = current_app.config["PRIMARY_PHOTO_RETRIES"]
    if user is None:
        user={
            "email": email,
            "phone_number": phone_number
        }
    try:
        md, md_sface, similar_users = set_primary_photo_internal(now, user_id=user_id, photo_stream=photo_stream, attempt=current_app.config["PRIMARY_PHOTO_RETRIES"] - attempt + 1)
    except exceptions.NoFaces as e:
        raise e
    except exceptions.FailedTryToDisable as e:
        _decrease_available_retries(user, user_id)

        if user is not None and attempt <= 1:
            if _is_review_disabled():
                primary_photo.primary_photo_declined(e, now, current_user, current_user.user_id, photo_stream.stream)
            else:
                _register_unique_email_and_phone_number(user_id,user)
                review.primary_photo_to_review(now, current_user, user_id, user, photo_stream, e.arcface_metadata, e.similar_users, client_ip, e)
            return
        raise e
    _register_unique_email_and_phone_number(user_id,user)
    primary_photo.primary_photo_passed(now, current_user, user_id, user, photo_stream.stream, md_sface, md, attempt)


def check_similarity_and_update_secondary_photo(current_user, user_id: str, raw_pics: list, migrate_phone_login: bool, emotionSessionId = None):
    now = time.time_ns()
    user_reference_metadata = _get_primary_metadata(user_id, model = _model, search_growing = False)
    if user_reference_metadata is None:
        raise exceptions.MetadataNotFound(f"User {user_id} have no registered primary metadata yet")

    md_vector = user_reference_metadata["face_metadata"]
    pics = [loadImageFromStream(p) for p in raw_pics]
    best_md, bestIndex, euclidian,threshold, bestNotFittingIndex = extract_and_compare_metadatas(md_vector, pics, _model)
    if bestIndex == -1:
        euclidian_sface = euclidian
        sface_threshold = threshold
        best_md, bestIndex, euclidian, threshold, bestNotFittingIndex = recheck_similarity_using_sface(md_vector, user_id, pics, [best_md], bestNotFittingIndex)
        if bestIndex == -1:
            metrics.register_similarity_failure(euclidian_sface, euclidian)
            raise exceptions.NotSameUser(f"user mismatch for user_id {user_id}: distance is greater than {sface_threshold} {threshold}: {euclidian_sface} {euclidian}")
    if not migrate_phone_login:
        put_secondary_photo(user_id,raw_pics[bestIndex].stream)

        prev_state = _get_secondary_metadata(user_id, model=_model)
        upd, rows = _update_secondary_metadata_pending(now, user_id, best_md, model=_model)
        ###
        # with metrics.represent_time.labels(model = _model_fallback).time():
        #     m = DeepFace.build_model(_model_fallback)
        #     md_arcface = distance.l2_normalize(m.predict(np.expand_dims(pics[bestIndex][::2,::2], axis=0), verbose = 0)[0].tolist())
        #     _update_secondary_metadata(now, user_id, md_arcface, url, model=_model_fallback)
        ###
        if rows > 0:
            try:
                callback(current_user, user_reference_metadata, upd, _get_user(user_id))
            except UnauthorizedFromWebhook as e:
                if prev_state is not None: pass
                    #_update_secondary_metadata(prev_state["uploaded_at"], user_id, prev_state["face_metadata"], prev_state["url"], model=_model)

                raise e
            except Exception as e:
                if prev_state is not None: pass
                    #_update_secondary_metadata(prev_state["uploaded_at"], user_id, prev_state["face_metadata"], prev_state["url"], model=_model)

                raise e # goes to 5xx

            return bestIndex, euclidian, upd["uploaded_at"]

    return bestIndex, euclidian, None

def recheck_similarity_using_sface(primary_md, user_id: str, pics: list, sface_metadatas: list, bestNotFittingIndex: int):
    secondary_md = _get_secondary_metadata(user_id, model=_model)
    threshold = _similarity_threshold(_model_fallback)
    if not secondary_md:
        return sface_metadatas[0], -1, -1, threshold, bestNotFittingIndex

    try:
        new_pic_md = distance.l2_normalize(DeepFace.represent(
            img_path=pics[bestNotFittingIndex],
            model_name=_model,
            detector_backend=_detector_low_quality,
            align=True,
            normalization="base",
        )[0]["embedding"])
    except ValueError as e:
        raise exceptions.NoFaces("No faces detected on recheck")

    bestIndex, euclidian, bestNotFittingIndex = compare_metadatas([secondary_md["face_metadata"], new_pic_md], threshold)
    if bestIndex == -1:
        bestIndex, euclidian, bestNotFittingIndex = compare_metadatas([primary_md, new_pic_md], threshold)

    return new_pic_md, bestIndex, euclidian, threshold, bestNotFittingIndex

def recheck_similarity_using_arcface(primary_md, user_id: str, pics: list,sface_metadatas: list, bestNotFittingIndex: int):
    # user is not the same as on primary photo - let's try with more complex but slower model as well to reduce false-negatives
    primary_photo = get_primary_photo(user_id)
    t = time.time()
    md_vector = distance.l2_normalize(DeepFace.represent(
        img_path=loadImageFromStream(io.BytesIO(primary_photo)),
        model_name=_model_fallback,
        enforce_detection=True,
        detector_backend=_detector_high_quality,
        align=True,
        normalization="base",
    )[0]["embedding"])

    threshold = _similarity_threshold(_model_fallback)
    bestIndex, euclidian, bestNotFittingIndex = compare_metadatas([md_vector,distance.l2_normalize(DeepFace.represent(
        img_path=pics[bestNotFittingIndex],
        model_name=_model_fallback,
        enforce_detection=True,
        detector_backend=_detector_high_quality,
        align=True,
        normalization="base",
    )[0]["embedding"])], threshold)

    return bestIndex, euclidian, threshold, bestNotFittingIndex

def extract_and_compare_metadatas(user_reference_metadata: list, pics, model):
    m = DeepFace.build_model(model)
    def predict_pic(p):
        try:
            _,f = DeepFace.extract_faces_custom(img_path=p, target_size=(112, 112), detector_backend=_detector_low_quality, align=True)
            repr = f[0]['face']
        except ValueError as e:
            raise exceptions.NoFaces("No faces detected on metadata comparison")
        with metrics.represent_time.labels(model = model).time():
            return distance.l2_normalize(DeepFace.represent(
                img_path=repr,
                model_name=_model,
                detector_backend="skip",
                normalization="base",
                target_size=(112, 112),
                align=True,
                enforce_detection = False,
            )[0]["embedding"])
        # try:
        #     _,f = DeepFace.extract_faces_custom(img_path=p, target_size=(112, 112), detector_backend=_detector_low_quality, align=True)
        #     p = f[0]['face']
        # except ValueError as e:
        #     raise exceptions.NoFaces("No faces detected on metadata comparison")
        # with metrics.represent_time.labels(model = model).time():
        #     return distance.l2_normalize(m.predict(np.expand_dims(p, axis=0))[0].tolist())
    threshold = _similarity_threshold(model)
    d = threshold + 1
    idx = 0
    best_idx = idx
    best_distance = d
    md = []
    while d > threshold and idx < len(pics):
        try:
            md = predict_pic(pics[idx])
        except exceptions.NoFaces as e:
            if idx >= len(pics) - 1:
                raise e # last pic, we have to fail anyway
            else:
                idx += 1
                continue
        current_distance = distance.findEuclideanDistance(user_reference_metadata, md)
        if min(best_distance, current_distance) < best_distance:
            best_idx = idx
            best_distance = d
        d = current_distance
        idx += 1
    if d > threshold:
        return md, -1, d,threshold, best_idx
    return md, best_idx,d,threshold, best_idx

def _similarity_threshold(model: str):
    return current_app.config[f"SIMILARITY_{model.upper()}_DISTANCE"]

def compare_metadatas(metadatas: list, threshold: float):
    normalizedRefMetadata = metadatas.pop(0)
    distances = [distance.findEuclideanDistance(normalizedRefMetadata, md) for md in metadatas]
    m = distances[0]
    indexes = [(distances.index(d), m := min(d,m)) for d in distances if d <= threshold]
    indexes.sort(key=lambda x: x[1])

    if len(indexes) != len(metadatas) and len(indexes) < _min_images_with_emotions_to_proceed:
        return -1, min([i for i in distances if i > threshold]), int(np.argmin(distances))
    else:
        return indexes[0][0], m, indexes[0][0]

def loadImageFromStream(p):
    chunk_arr = np.frombuffer(p.read(), dtype=np.uint8)
    img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)

    return img

def get_status(user_id: str):
    primary = _get_primary_metadata(user_id, model=_model_fallback)
    secondary = _get_secondary_metadata(user_id, model=_model)
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
    p = os.environ.get('IMG_STORAGE_PATH')
    p = p if p.endswith("/") else p+"/"
    return len(glob.glob(f"{p}{user_id}/*{_picture_extension}"))

def _remove_user_images(user_id):
    p = os.environ.get('IMG_STORAGE_PATH')
    p = p if p.endswith("/") else p+"/"
    dirname = f"{p}{user_id}"
    if os.path.isdir(dirname) == False:
        return

    for filename in glob.glob(f"{dirname}/*{_picture_extension}"):
        os.remove(filename)

    if _count_user_images(user_id) == 0:
        shutil.rmtree(dirname)

def _get_unique_emotion(current_emotions_list: list, excluded_emotions = frozenset()):
    if len(current_emotions_list) == 0:
        choice = random.choice(_default_emotions_list[0])

        return choice.lower(), False

    last_weight = [current_emotions_list[-1] in emotions_by_weight for emotions_by_weight in _default_emotions_list].index(True)
    diff = []
    excluded=set(excluded_emotions)
    excluded.update(current_emotions_list)
    while last_weight < len(_default_emotions_list):
        diff = set(_default_emotions_list[last_weight]) - set(excluded)
        if len(diff) == 0:
            last_weight += 1
        else:
            break
    if last_weight >= len(_default_emotions_list) and len(diff) == 0:
        return None, True

    choice = random.choice(list(diff))

    return choice.lower(), False

def emotions(user_id: str, migrate_phone_login: bool):
    now = time.time_ns()
    usr = _get_user(user_id, search_growing=False)
    if usr is not None:
        if usr['disabled_at'] is not None and usr['disabled_at'] > 0:
            raise exceptions.UserDisabled(f"user:{usr['user_id']} disabled")
        # if user is on review with possible_duplicate_with flag then
        # in normal conditions he cannot reach here, but we cannot return disabled due to FE dialog
        if usr['possible_duplicate_with'] or (usr['last_negative_request_at'] > 0 and now - usr['last_negative_request_at'] <= current_app.config['LIMIT_RATE_NEGATIVE']):
            raise exceptions.NegativeRateLimitException(f"limit rate time didn't pass from the last negative try for user:{user_id} time: {usr['last_negative_request_at']}")

        if not migrate_phone_login:
            secondary = _get_secondary_metadata(user_id, model=_model)
            if secondary is not None and secondary['uploaded_at'] is not None and now - secondary['uploaded_at'] <= current_app.config['LIMIT_RATE']:
                raise exceptions.RateLimitException(f"rate limit exception for user_id:{user_id}, already passed the liveness at {secondary['uploaded_at']}")

        _remove_user_images(user_id)

    session_id = str(uuid.uuid4())
    emotions_list = []
    total_emotions_count = sum([len(i) for i in _default_emotions_list])
    for i in range(0, current_app.config['INITIAL_EMOTION_COUNT']):
        new_emotion, completed = _get_unique_emotion(list(emotions_list))
        if completed:
            idx = int(i/total_emotions_count)*total_emotions_count
            new_emotion, _ = _get_unique_emotion(emotions_list[idx:],excluded_emotions=set(emotions_list[-1]))
        emotions_list.append(new_emotion)

    res = _update_user(
        user_id=user_id,
        session_id=session_id,
        emotions=",".join(emotions_list),
        session_started_at=now,
        last_negative_request_at=usr["last_negative_request_at"] if usr else 0,
        disabled_at=0,
        emotion_sequence=0,
        best_pictures_score=np.array([0.0]*45)
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
    total_emotions_count = sum([len(i) for i in _default_emotions_list])
    new_emotion, completed = _get_unique_emotion(usr['emotions'].split(','))
    if completed is True:
        idx = int((len(emotions_list))/total_emotions_count)*total_emotions_count
        new_emotion, _ = _get_unique_emotion(emotions_list[idx:], excluded_emotions=emotions_list[-1])
    emotions_list.append(new_emotion)
    return ",".join(emotions_list)

def _add_additional_emotion(usr):
    current_emotions_list = usr['emotions'].split(',')
    emotion_sequence = usr['emotion_sequence']
    if emotion_sequence != len(current_emotions_list) or emotion_sequence >= current_app.config['MAX_EMOTION_COUNT']:
        return ",".join(current_emotions_list)

    emotions_list = _generate_emotions(usr)

    return emotions_list

def _generate_best_scores(usr, current_emotions_list,scores, model, images_count, session_success):
    idx = int(images_count / _images_count_per_call)
    if idx >= current_app.config["TARGET_EMOTION_COUNT"]: idx = current_app.config["TARGET_EMOTION_COUNT"] - 1
    neutral_idx = model.class_to_idx.get("neutral")
    neutral_scores = [s[neutral_idx]for s in scores]
    usr['best_pictures_score'][idx * _images_count_per_call:(idx+1)*_images_count_per_call] = neutral_scores
    usr['emotion_sequence'] = usr['emotion_sequence']+1
    if usr['emotion_sequence'] >= len(current_emotions_list) and (not session_success):
        usr['emotions'] = _add_additional_emotion(usr)

    return usr['emotions']

def _predict(usr, model, images, now, awaited_emotion):
    t = time.time()
    face_img_list = []
    for img in images:
        loaded_image = loadImageFromStream(img)
        if loaded_image.shape[0] != model.img_size or loaded_image.shape[1] != model.img_size:
            raise exceptions.WrongImageSizeException(f"wrong image size for user:{usr['user_id']}, session:{usr['session_id']}")

        face_img_list.append(loaded_image)
    try:
        DeepFace.extract_faces(face_img_list[1+int(len(face_img_list)/2)], target_size=(112, 112), detector_backend=_detector_low_quality, enforce_detection=True, landmarks_verification=False)
    except ValueError as e:
        raise exceptions.NoFaces(f"No faces detected, userId: {usr['user_id']}")

    awaited_idx = model.class_to_idx[awaited_emotion]
    emotions, scores = model.predict_multi_emotions(face_img_list=face_img_list, logits = False)

    averages = np.average(scores, axis=0, weights=[i for i in range(len(scores))]) # last frames weights more
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

def _send_best_images(similarity_server:str, current_user, user_id, files, emotion_session_id, migrate_phone_login):
    response = None
    try:
        t = time.time()
        logging.debug(f"Initiating similarity call for user_id {user_id} S:{emotion_session_id}")

        if not migrate_phone_login:
            headers = {"Authorization": f"Bearer {current_user.raw_token}","X-Account-Metadata": current_user.metadata, "x-queued-time": str(float(time.time()))}
        else:
            headers = {
                "X-Migrate-Phone-Number-To-Email": "",
                "X-Migrate-Phone-Number-Language": current_user.language,
                "X-Migrate-Phone-Number-Device-Unique-Id": current_user.device_unique_id,
                "X-Migrate-Phone-Number-Email": current_user.email,
                "x-queued-time": str(float(time.time()))
            }

        response = requests.post(
            url=f"{similarity_server[:-1] if similarity_server.endswith('/') else similarity_server}/v1w/face-auth/similarity/{user_id}?sessionId={emotion_session_id}",
            files=files,
            headers=headers,
            timeout=15
        )
    except Exception as e:
        if response:
            response.close()
        logging.warning(f"Similarity check userID {user_id}: {str(e)}")

        raise e
    logging.debug(f"Similarity e2e took for user_id {user_id} S:{emotion_session_id}: {time.time() - t}")
    _put_user_similarity_resp(time.time_ns(),user_id,response.status_code,response.content)
    if response.status_code == 200:
        response.close()
        return True
    else:
        logging.warning(f"Similarity check userID {user_id}: {response.status_code} {response.text}")
        if response.status_code == 400 and ["code"] == _user_not_the_same:
            response.close()
            return False

    return True

def check_emotions_similarity(usr, current_user, migrate_phone_login):
    best_indexes = []
    for idx in np.argpartition(usr['best_pictures_score'], -current_app.config['TOTAL_BEST_PICTURES'])[-current_app.config['TOTAL_BEST_PICTURES']:]:
        best_indexes.append(str(idx))
    files = []
    for img_idx in best_indexes:
        file_path = f"{current_app.config['IMG_STORAGE_PATH']}{usr['user_id']}/{img_idx}{_picture_extension}"
        files.append(('image', (f"{img_idx}{_picture_extension}", open(file_path, 'rb'), 'image/jpeg')))

    executor = current_app.extensions["snowfaceexecutor"]
    futures = [
        executor.submit(
            _send_best_images,
            current_app.config['SIMILARITY_SERVER'],
            current_user, usr['user_id'],
            files, usr['session_id'],
            migrate_phone_login
        )
    ]

def _send_magic_link(user_id, current_user):
    similarity_server = current_app.config['SIMILARITY_SERVER']
    headers = {
                "X-Migrate-Phone-Number-To-Email": "true",
                "X-Migrate-Phone-Number-Language": current_user.language,
                "X-Migrate-Phone-Number-Device-Unique-Id": current_user.device_unique_id,
                "X-Migrate-Phone-Number-Email": current_user.email,
                "X-Send-Email-Magic-Link": "true"
            }
    response = requests.post(
            url=f"{similarity_server[:-1] if similarity_server.endswith('/') else similarity_server}/v1w/face-auth/similarity/{user_id}",
            headers=headers,
            timeout=25
        )

    if response.status_code == 400:
        raise MigratePhoneLoginWebhookBadRequest(response.content)
    elif response.status_code == 409:
        raise MigratePhoneLoginWebhookConflict(response.content)
    elif response.status_code == 429:
        raise MigratePhoneLoginWebhookRateLimit(response.content)
    elif response.status_code != 200:
        raise Exception()

    d = response.content.decode("utf-8")
    response.close()
    return d

def _finish_session(usr, current_user, migrate_phone_login):
    #identity_match = _send_best_images(current_app.config['SIMILARITY_SERVER'], token,usr['user_id'], files)
    login_session = None
    identity_match = False
    similarity_code, similarity_resp = _get_user_similarity_resp(usr["user_id"])

    if similarity_code is None and similarity_resp is None:
       logging.warning(f"Similarity check userID {usr['user_id']} not finished yet")
       identity_match = True

    if not identity_match and int(similarity_code) == 200:
        identity_match = True

    if not identity_match and int(similarity_code) == 400:
        body = json.loads(similarity_resp)
        if body["code"] == _user_not_the_same:
            identity_match = False

    if identity_match:
        if not migrate_phone_login:
            url, uploadedat, face = _get_pending_face(usr['user_id'])
            if uploadedat is not None and face is not None:
                _update_secondary_metadata(int(uploadedat),usr['user_id'], [float(x) for x in face], usr.get("email", ""),usr.get("phone_number", ""), _model)
        else:
            login_session = _send_magic_link(usr['user_id'], current_user)

        _remove_user_images(user_id=usr['user_id'])
        _remove_session(usr['user_id'])

    return identity_match, login_session

def process_images(current_user, user_id: str, session_id: str, images:list, migrate_phone_login: bool):
    now = time.time_ns()
    usr = _get_user(user_id, search_growing=False)
    _validate_session(usr=usr, user_id=user_id, session_id=session_id, now=now)

    if usr['possible_duplicate_with'] or (usr['last_negative_request_at'] > 0 and now - usr['last_negative_request_at'] <= current_app.config['LIMIT_RATE_NEGATIVE']):
        raise exceptions.NegativeRateLimitException(f"limit rate time didn't pass from the last negative try for user:{user_id} time: {usr['last_negative_request_at']}")

    current_emotions_list = usr['emotions'].split(',')

    if usr['emotion_sequence'] >= current_app.config['MAX_EMOTION_COUNT']:
        if _update_last_negative_request_at(usr=usr, now=now) is False:
            raise exceptions.UpsertException(f"update last negative request time failed for user:{user_id}")

        return False, True, usr['emotions'], None

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

        return False, False, usr['emotions'], None

    model = DeepFace.build_model("Emotion")
    tpredict = time.time()
    try:
        awaited_score, scores, max_emotion, max_score, averages = _predict(
            usr=usr,
            model=model,
            images=images,
            now=now,
            awaited_emotion = current_emotion,
        )
    except exceptions.NoFaces as e:
        _remove_user_images(user_id = user_id)
        usr['session_id'] = _invalidated_session
        if _update_emotions_and_best_score(
                usr=usr,
                emotions = usr['emotions'],
                emotion_sequence=usr['emotion_sequence'],
                best_score=usr['best_pictures_score'],
                last_negative_request_at = usr['last_negative_request_at']
        ) is False:
            raise exceptions.UpsertException(f"can't invalidate session user_id:{user_id}")

        raise e

    relative_score = awaited_score*100.0/max_score
    logging.info(f"[U:{user_id}][S:{session_id}] awaited {current_emotion}/{awaited_score} it is {relative_score} of ({max_emotion}/{max_score}=100)  < {current_app.config['TARGET_EMOTION_SCORE']} all:{averages}, took {time.time() - tpredict}")
    if relative_score < current_app.config['TARGET_EMOTION_SCORE']:
        metrics.register_emotion_failure(model,current_emotion,scores, averages)
        usr['emotion_sequence'] = usr['emotion_sequence']+1
        session_ended = usr['emotion_sequence'] >= current_app.config['MAX_EMOTION_COUNT']
        if session_ended:
            metrics.register_session_failure()
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

        return False, session_ended, usr['emotions'], None

    metrics.register_emotion_success(model,current_emotion,scores, averages)
    images_count = _count_user_images(user_id)
    for idx, img in enumerate(images):
        _save_image(
            image=img,
            user_id=user_id,
            idx=idx + images_count
        )

    session_success = _count_user_images(user_id) >= _images_count_per_call * current_app.config['TARGET_EMOTION_COUNT']

    session_ended = usr['emotion_sequence'] >= current_app.config['MAX_EMOTION_COUNT']
    emotions = _generate_best_scores(
        usr=usr,
        current_emotions_list=current_emotions_list,
        scores=scores,
        model=model,
        images_count=images_count,
        session_success = session_success
    )
    time_to_check_for_similarity = (_count_user_images(user_id) >= _images_count_per_call * (current_app.config['TARGET_EMOTION_COUNT']-1)) and (not (session_ended or session_success))
    # actual photos verification -> result in redis -> session ends -> check redis ->swaps metas in milvus and callback to eskimo (for non migrate case)
    if time_to_check_for_similarity:
        check_emotions_similarity(usr, current_user, migrate_phone_login)

    result = True
    login_session = None
    if session_success:
        result, login_session = _finish_session(usr, current_user, migrate_phone_login)

        if not result:
            usr['last_negative_request_at'] = now
            metrics.register_session_failure()

        session_ended = True
        if result:
            metrics.register_session_length(usr['emotion_sequence']+1)
    if (not result) or (not session_ended):
        if _update_emotions_and_best_score(
                usr=usr,
                emotions = usr['emotions'],
                emotion_sequence=usr['emotion_sequence'],
                best_score=usr["best_pictures_score"]
        ) is not True:
            raise exceptions.UpsertException(f"can't update emotion sequence and best score for user_id:{usr['user_id']}")

    return result, session_ended, emotions, login_session

def delete_temporary_user_data(user_id:str):
    _remove_user_images(user_id)

def delete_user_photos_and_metadata(current_user, to_delete_user_id = "", force_user_id = ""):
    primary_photo.delete_user_photos_and_metadata(current_user, to_delete_user_id, force_user_id)

def proxy_delete_if_not_exists(similarity_server, current_user, user_id = ""):

    if not user_id:
        user = _get_user(current_user.user_id)
        if user is None:
            primary_photo = _get_primary_metadata(current_user.user_id,model=_model_fallback, search_growing=False)
            if not primary_photo:
                return "", 204
    executor = current_app.extensions["snowfaceexecutor"]
    futures = [
        executor.submit(
            proxy_delete,
            current_app.config['SIMILARITY_SERVER'],
            current_user, user_id,
        )
    ]
    return "", 200

def proxy_delete(similarity_server, current_user, user_id = ""):
    url = f"{similarity_server[:-1] if similarity_server.endswith('/') else similarity_server}/v1w/face-auth/"
    payload = None
    if user_id != "":
        url = f"{url}?userId={user_id}"
        payload = {"userId":user_id}
    response = requests.delete(
        url=url,
        headers={
            "Authorization": f"Bearer {current_user.raw_token}",
            "X-Account-Metadata": current_user.metadata,
            "x-queued-time": str(float(time.time()))},
        json=payload,
        timeout=5
    )
    d, s = response.content, response.status_code
    response.close()
    return d, s
def emotions_cleanup():
    now = time.time_ns()
    duration = int(os.environ.get('SESSION_DURATION', _default_session_duration))*int(1e9)
    sessions = _get_expired_sessions(now,duration)
    logging.info(f"cleaning outdated sessions ({len(sessions)})...")

    for session in sessions:
        try:
            _remove_user_images(session)
        except OSError as e:
            logging.warning(str(e))
def reenable_user(current_user, user_id: str, duplicated_face: str):
    primary = _get_primary_metadata(user_id, model = _model_fallback, search_growing = False)
    secondary = _get_secondary_metadata(user_id, model=_model)

    _enable_user(user_id)

    callback(current_user, primary, secondary, None, user_id=user_id)

    if duplicated_face:
        try:
            delete_user_photos_and_metadata(current_user, force_user_id=duplicated_face)
        except exceptions.MetadataNotFound as e:
            pass

def review_duplicates(current_user: Token, user_id:str, decision: str, most_similar_duplicate: str = None):
    now = time.time_ns()
    if user_id and decision:
        review.make_decision(now,current_user,user_id, decision, most_similar_duplicate)
    return review.next_user_for_review(current_user.user_id)


def _reprocess_wrongfully_disabled_users():
    global __admin_token, __stop_wrongfully_disabled_users_worker
    while 1:
        if __stop_wrongfully_disabled_users_worker: break
        try:
            now = time.time_ns()
            user_id = _get_disabled_user_for_selfie_reprocessing()
            if user_id is None:
                logging.debug(f"[reprocess_wrongfully_disabled_users] No user to process")
                time.sleep(60)
                continue
            __admin_token = _get_admin_token()
            if __admin_token is None:
                logging.warning(f"[reprocess_wrongfully_disabled_users] users are presented ({user_id}) but admin token is not presented")
                _put_disabled_user_for_selfie_reprocessing(user_id)
                time.sleep(60)
                continue
            data = jwt.decode(__admin_token, options={"verify_signature": False})
            expiration = data.get("exp",int(time.time()-60))
            if expiration <= int(time.time()):
                logging.warning(f"[reprocess_wrongfully_disabled_users] users are presented ({user_id}) but admin token is expired")
                _put_disabled_user_for_selfie_reprocessing(user_id)
                time.sleep(60)
                continue
            photo = _get_disabled_photo(user_id)
            if not photo:
                logging.warning(f"[reprocess_wrongfully_disabled_users] user {user_id} do not have disabled photo to reprocess")
                continue
            logging.info(f"[reprocess_wrongfully_disabled_users] picked {user_id}")
            user = _get_user(user_id, search_growing=False)
            _enable_user(user_id)
            if user is None:
                user = {"user_id": user_id}
            user["disabled_at"] = 0
            try:
                md, md_sface,similar_users = set_primary_photo_internal(now,user_id=user_id, photo_stream=io.BytesIO(photo), attempt=-1)
            except exceptions.FailedTryToDisable as e:
                md = e.arcface_metadata
                md_sface = distance.l2_normalize(DeepFace.represent(
                    img_path=loadImageFromStream(io.BytesIO(photo)),
                    model_name=_model,
                    detector_backend="skip",
                    normalization="base",
                    target_size=(112, 112),
                )[0]["embedding"])
            except Exception as e:
                _put_disabled_user_for_selfie_reprocessing(user_id)
                logging.error(f"[reprocess_wrongfully_disabled_users] User {user_id}: "+str(e), exc_info=e)
                continue
            try:
                put_primary_photo(user_id,io.BytesIO(photo))
                upd, rows = _set_primary_metadata(now, user_id, md,user.get("email", ""),user.get("phone_number", ""), model=_model_fallback)
                if rows > 0:
                    _set_primary_metadata(now, user_id, md_sface, user.get("email", ""),user.get("phone_number", ""), model=_model)
                    metrics.register_primary_photo_uploaded(1)
                    if __admin_token is None:
                        __admin_token = _get_admin_token()
                    try:
                        callback(
                            current_user=Token(__admin_token,"","","",""),
                            primary_md=upd,
                            secondary_md=None,
                            user=user,
                            user_id=user_id
                        )
                    except UnauthorizedFromWebhook as e:
                        __admin_token = _get_admin_token()
                        callback(
                            current_user=Token(__admin_token,"","","",""),
                            primary_md=upd,
                            secondary_md=None,
                            user=user,
                            user_id=user_id,
                        )
            except Exception as e:
                _put_disabled_user_for_selfie_reprocessing(user_id)
                logging.error(f"[reprocess_wrongfully_disabled_users] User {user_id}: "+str(e), exc_info=e)
        except Exception as e:
            _put_disabled_user_for_selfie_reprocessing(user_id)
            logging.error(f"[reprocess_wrongfully_disabled_users] User {user_id}: "+str(e), exc_info=e)



def stop_wrongfully_disabled_users_worker():
    global __stop_wrongfully_disabled_users_worker
    _unregister_wrongfully_disabled_users_worker()
    __stop_wrongfully_disabled_users_worker = True


def start_wrongfully_disabled_users_worker():
    w =  _register_wrongfully_disabled_users_worker()
    if w > current_app.config["WRONGFULLY_DISABLED_USERS_WORKERS"]:
        logging.info(f"Worker {os.getpid()} skipping WRONGFULLY_DISABLED_USERS due to count is {w} {current_app.config['WRONGFULLY_DISABLED_USERS_WORKERS']}")
        return

    wrongfully_disabled_users_executor = Executor(current_app, "wrongfully_disabled_users_processor")
    with current_app.test_request_context():
        wrongfully_disabled_users_executor.submit(_reprocess_wrongfully_disabled_users)