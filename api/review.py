import io

from users import (
    mark_user_for_manual_review as _mark_user_for_manual_review,
    user_reviewed as _user_reviewed,
    get_user as _get_user,
    allocate_review_user as _allocate_review_user
)
from minio_uploader import (
    put_review_photo as _put_review_photo,
    get_review_photo as _get_preview_photo
)
from webhook import callback
import primary_photo
from auth import Token
class UserForReview:
    def __init__(self, user, primary_photo, possible_duplicates):
        self.user_id = user.get("user_id")
        self.ip = user.get("ip")
        self.retries = user.get("duplicate_review_count", 0)
        self.primary_photo = primary_photo
        self.possible_duplicates = possible_duplicates


def _primary_photo_to_review(now,current_user,user_id, user, photo_stream,similar_users, ip):
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
    if decision == "duplicate":
        _user_reviewed(admin_id=admin_current_user.user_id,user_id=user_id,retry=False)
        photo = _get_preview_photo(user_id)
        primary_photo._primary_photo_declined(None,now,admin_current_user,user_id, photo)
    elif decision == "retry":
        _user_reviewed(admin_id=admin_current_user.user_id,user_id=user_id,retry=True)
    elif decision == "not_duplicate":
        _user_reviewed(admin_id=admin_current_user.user_id,user_id=user_id,retry=False)
        photo = _get_preview_photo(user_id)
        primary_photo._primary_photo_passed(now,admin_current_user,user_id, photo)
    else:
        raise Exception(f"invalid decision:{decision}")


def next_user_for_review(admin_id):
    user_id = _allocate_review_user(admin_id)
    user = _get_user(user_id)
    selfie = _get_preview_photo(user_id)
    return UserForReview(user,primary_photo=selfie, possible_duplicates = [])