from users import (
    mark_user_for_manual_review as _mark_user_for_manual_review,
    get_user as _get_user,
    allocate_review_user as _allocate_review_user
)
from minio_uploader import (
    put_review_photo as _put_review_photo,
    get_review_photo as _get_preview_photo
)
from webhook import callback

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

def make_decision(user_id:str, decision: str):
    pass

def next_user_for_review(admin_id):
    user_id = _allocate_review_user(admin_id)
    user = _get_user(user_id)
    selfie = _get_preview_photo(user_id)
    return UserForReview(user,primary_photo=selfie, possible_duplicates = [])