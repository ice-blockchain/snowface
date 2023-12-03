from users import (
rollback_disabled_user as _rollback_disabled_user
)
from faces import set_primary_metadata as _set_primary_metadata, delete_metadatas as _delete_metadatas
from webhook import callback,UnauthorizedFromWebhook
from minio_uploader import put_primary_photo
from flask import current_app
import metrics


_model = "SFace"
_model_fallback = "ArcFace"#"Facenet" #"VGG-Face"
def _primary_photo_passed(now, current_user,user_id,user, photo_stream, md_sface, md, attempt):
    url = put_primary_photo(user_id,photo_stream.stream)
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

            raise e
        except Exception as e:
            _delete_metadatas(user_id, [upd["user_picture_id"]])

            raise e # goes to 5xx

def _primary_photo_declined(e,now,current_user,user_id, photo_stream):
    disabled = _disable_user(now,user_id, photo_stream.stream)
    if disabled:
        metrics.register_disabled_user(e.sface_distance, e.arface_distance)
        try:
            callback(
                current_user=current_user,
                primary_md=None,
                secondary_md=None,
                user={"disabled_at": now}
            )
        except UnauthorizedFromWebhook as ex:
            _rollback_disabled_user(user_id)

            raise ex
        except Exception as ex:
            _rollback_disabled_user(user_id)

            raise ex # goes to 5xx

        raise exceptions.UserDisabled(f"Face {user_id}  is matching with user {e.matching_user_id}, attempt:{current_app.config['PRIMARY_PHOTO_RETRIES']}, distance {e.sface_distance} < {current_app.config['PRIMARY_PHOTO_SFACE_DISTANCE']}, {e.arface_distance} < {current_app.config['PRIMARY_PHOTO_ARCFACE_DISTANCE']}")
