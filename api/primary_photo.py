from users import (
    rollback_disabled_user                    as _rollback_disabled_user,
    disable_user                              as db_disable_user,
    get_user                                  as _get_user,
    full_user_reset                           as _full_user_reset,
    remove_expired                            as _remove_expired,
    set_expired                               as _set_expired,
    update_user                               as _update_user,
    rollback_manual_review                    as _rollback_manual_review

)
from faces import (
    set_primary_metadata as _set_primary_metadata,
    delete_metadatas as _delete_metadatas,
    update_secondary_metadata              as _update_secondary_metadata,
    set_primary_metadata                   as _set_primary_metadata,
    delete_metadatas                       as _delete_metadatas
)
from webhook import callback,UnauthorizedFromWebhook
import requests, io
from minio_uploader import (put_primary_photo, put_disabled_photo as _put_disable_photo, delete_photos as _delete_photos, put_secondary_photo as _put_secondary_photo)
from flask import current_app
import metrics
import exceptions

from deepface import DeepFace
from deepface.commons import distance
import numpy as np, cv2


_model = "SFace"
_model_fallback = "ArcFace"#"Facenet" #"VGG-Face
_detector_high_quality = "yunet"
def primary_photo_passed(now, current_user, user_id, user, photo_stream, md_sface, md, attempt):
    url = put_primary_photo(user_id,photo_stream)
    upd, rows = _set_primary_metadata(now, user_id, md, url, model=_model_fallback)
    if rows > 0:
        _set_primary_metadata(now, user_id, md_sface, url, model=_model)
        metrics.register_primary_photo_uploaded(current_app.config["PRIMARY_PHOTO_RETRIES"] - attempt + 1)
        try:
            callback(
                current_user=current_user,
                primary_md=upd,
                secondary_md=None,
                user=user
            )
        except UnauthorizedFromWebhook as e:
            _delete_metadatas(user_id, [upd["user_picture_id"]])
            if user.get("possible_duplicate_with",[]):
                _rollback_manual_review(user_id)
            raise e
        except Exception as e:
            _delete_metadatas(user_id, [upd["user_picture_id"]])
            if user.get("possible_duplicate_with",[]):
                _rollback_manual_review(user_id)
            raise e # goes to 5xx

def primary_photo_declined(e, now, current_user, user_id, photo_stream):
    disabled = _disable_user(now,user_id, photo_stream)
    if disabled:
        metrics.register_disabled_user(e.sface_distance, e.arface_distance)
        try:
            callback(
                current_user=current_user,
                primary_md=None,
                secondary_md=None,
                user={"disabled_at": now},
                user_id=user_id
            )
        except UnauthorizedFromWebhook as ex:
            _rollback_disabled_user(user_id)

            raise ex
        except Exception as ex:
            _rollback_disabled_user(user_id)

            raise ex # goes to 5xx

        raise exceptions.UserDisabled(f"Face {user_id}  is matching with user {e.matching_user_id}, attempt:{current_app.config['PRIMARY_PHOTO_RETRIES']}, distance {e.sface_distance} < {current_app.config['PRIMARY_PHOTO_SFACE_DISTANCE']}, {e.arface_distance} < {current_app.config['PRIMARY_PHOTO_ARCFACE_DISTANCE']}")

def _disable_user(now, user_id, photo_content):
    _put_disable_photo(user_id,photo_content)

    return db_disable_user(now,user_id)

def extract_metadatas(user_id, photo_stream):
    try:
        img_objs, resp_objs = DeepFace.extract_faces_custom(
            img_path=loadImageFromStream(photo_stream),
            detector_backend=_detector_high_quality,
            align=True,
            landmarks_verification=True)
    except ValueError:
        raise exceptions.NoFaces(f"No faces detected, userId: {user_id}")
    img_to_represent = resp_objs[0]['face']
    md = distance.l2_normalize(DeepFace.represent(
        img_path=img_to_represent,
        model_name=_model_fallback,
        detector_backend="skip",
        normalization="base",
        target_size=(112, 112),
    )[0]["embedding"])
    sface_md = distance.l2_normalize(DeepFace.represent(
        img_path=img_to_represent,
        model_name=_model,
        detector_backend="skip",
        normalization="base",
        target_size=(112, 112),
    )[0]["embedding"])
    return img_to_represent, md, sface_md


def delete_user_photos_and_metadata(current_user, to_delete_user_id = "", force_user_id = "", keep_retries = False):
    if to_delete_user_id != "":
        user_id = to_delete_user_id
    elif force_user_id != "":
        user_id = force_user_id
    else:
        user_id = current_user.user_id

    prev_state = _get_user(user_id, search_growing=False)

    main, secondary, errs = _delete_photos(user_id)
    if len(errs) > 0:
        raise Exception(str(errs))

    main_md, secondary_md, deleted_mds = _delete_metadatas(user_id, [f"{user_id}~0", f"{user_id}~1"])

    if force_user_id == "":
        _full_user_reset(user_id, prev_state=prev_state if keep_retries else None)
        if prev_state is not None:
            _remove_expired(prev_state['session_started_at'], user_id)

    if force_user_id != "":
        callback_user_id = force_user_id
    elif to_delete_user_id != "":
        callback_user_id = user_id
    else:
        callback_user_id = ""

    try:
        callback(current_user, None, None, None, user_id = callback_user_id)
    except UnauthorizedFromWebhook as e:
        _rollback_deletion(prev_state, user_id, main, secondary, main_md, secondary_md)

        raise e
    except requests.HTTPError as e:
        if (e is not None) and (e.response is not None) and e.response.status_code == 404:
            raise exceptions.MetadataNotFound(f"face metadata for userId {user_id} was not deleted")
        _rollback_deletion(prev_state, user_id, main, secondary, main_md, secondary_md)
        raise e
    except Exception as e:
        _rollback_deletion(prev_state, user_id, main, secondary, main_md, secondary_md)

        raise e # goes to 5xx
    if (deleted_mds == 0 or (main_md is None and secondary_md is None)) and prev_state is None:
        raise exceptions.MetadataNotFound(f"face metadata for userId {user_id} was not deleted")

def _rollback_expired(user_id: str, prev_state):
    if prev_state['session_started_at'] is not None:
        _set_expired(prev_state['session_started_at'], user_id)


def _rollback_deletion(prev_state, user_id, main, secondary, main_md, secondary_md):
    if main is not None:
        put_primary_photo(user_id=user_id, photo_content=io.BytesIO(main))
    if secondary is not None:
        _put_secondary_photo(user_id=user_id, photo_content=io.BytesIO(secondary))
    if main_md:
        _set_primary_metadata(main_md["uploaded_at"], user_id, main_md["face_metadata"], main_md["url"], model=_model_fallback)
    if secondary_md:
        _update_secondary_metadata(secondary_md["uploaded_at"], user_id, secondary_md["face_metadata"], secondary_md["url"], model=_model_fallback)

    if prev_state is not None:
        _rollback_user_state(user_id, prev_state)
        _rollback_expired(user_id, prev_state)


def _rollback_user_state(user_id: str, prev_state):
    _update_user(
        user_id=user_id,
        session_id=prev_state['session_id'],
        emotions=prev_state['emotions'],
        session_started_at=prev_state['session_started_at'],
        last_negative_request_at=prev_state['last_negative_request_at'],
        disabled_at=prev_state['disabled_at'],
        emotion_sequence=prev_state['emotion_sequence'],
        best_pictures_score=prev_state['best_pictures_score'],
    )


def loadImageFromStream(p):
    chunk_arr = np.frombuffer(p.read(), dtype=np.uint8)
    img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)

    return img
