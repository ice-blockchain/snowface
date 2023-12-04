import io
import logging
from flask import current_app
from users import (
    mark_user_for_manual_review as _mark_user_for_manual_review,
    user_reviewed as _user_reviewed,
    rollback_reviewed as _rollback_reviewed,
    get_user as _get_user,
    allocate_review_user as _allocate_review_user
)
from minio_uploader import (
    put_review_photo as _put_review_photo,
    get_review_photo as _get_review_photo,
    get_primary_photo as _get_primary_photo,
    delete_review_photo as _delete_review_photo
)
from webhook import callback, UnauthorizedFromWebhook
import primary_photo
from auth import Token
import exceptions

class UserForReview:
    def __init__(self, user, primary_photo, possible_duplicates):
        self.user_id = user.get("user_id")
        self.ip = user.get("ip")
        self.retries = user.get("duplicate_review_count", 0)
        self.primary_photo = primary_photo
        self.possible_duplicates = possible_duplicates

class Duplicate:
    def __init__(self, user_id, primary_photo):
        self.user_id = user_id
        self.primary_photo = primary_photo

def primary_photo_to_review(now, current_user, user_id, user, photo_stream, similar_users, ip, e):
    logging.info(f"Face {user_id}  is matching with user {similar_users[0]}, user is forwarded to manual review: distance {e.sface_distance} < {current_app.config['PRIMARY_PHOTO_SFACE_DISTANCE']}, {e.arface_distance} < {current_app.config['PRIMARY_PHOTO_ARCFACE_DISTANCE']}")
    _mark_user_for_manual_review(user_id,ip,similar_users,user.get("duplicate_review_count",0))
    _put_review_photo(user_id,photo_stream)
    # we have noting to rollback, so exception straight to 5xx to retry from FE
    callback(
        current_user=current_user,
        primary_md={"uploaded_at": now},
        secondary_md=None,
        user=user,
        potentially_duplicate=True
    )

def make_decision(now: int, admin_current_user: Token, user_id:str, decision: str):
    user = _get_user(user_id)
    photo = _get_review_photo(user_id)
    if decision == "duplicate":
        _user_reviewed(admin_id=admin_current_user.user_id,user_id=user_id,retry=False)
        try:
            primary_photo.primary_photo_declined(exceptions.DisableByAdmin("manually disabled by admin", -1, -1, user.get("possible_duplicate_with",[])), now, admin_current_user, user_id, io.BytesIO(photo))
            _delete_review_photo(user_id)
        except exceptions.UserDisabled as e:
            pass
        except Exception as e:
            _rollback_reviewed(admin_id=admin_current_user.user_id,user_id=user_id,user = user,retry=False)
            raise e
    elif decision == "retry":
        _user_reviewed(admin_id=admin_current_user.user_id,user_id=user_id,retry=True)
        primary_photo.delete_user_photos_and_metadata(current_user=admin_current_user,to_delete_user_id=user_id)
    elif decision == "not_duplicate":
        _user_reviewed(admin_id=admin_current_user.user_id,user_id=user_id,retry=False)
        try:
            _, md, sface_md = primary_photo.extract_metadatas(user_id,io.BytesIO(photo))
            primary_photo.primary_photo_passed(now, admin_current_user, user_id,user, io.BytesIO(photo), sface_md, md, -1)
            _delete_review_photo(user_id)
        except Exception as e:
            _rollback_reviewed(admin_id=admin_current_user.user_id,user_id=user_id,user=user,retry=False)
            raise e
    else:
        raise Exception(f"invalid decision:{decision}")


def next_user_for_review(admin_id):
    user_id = _allocate_review_user(admin_id)
    if user_id:
        user = _get_user(user_id)
        selfie = _get_review_photo(user_id)
        return UserForReview(user,primary_photo=selfie, possible_duplicates = [fetch_duplicate(id) for id in user.get("possible_duplicate_with",[])] )
    return None

def fetch_duplicate(user_id):
    photo = _get_primary_photo(user_id)
    return Duplicate(user_id, photo)