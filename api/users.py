import redis, os
from flask import current_app
from typing import List
_client = None

def _get_client():
    global _client
    url = os.environ.get("REDIS_URI")
    if not _client:
        _client = redis.Redis.from_url(url)
        _client.ping()

    return _client

def _userKey(userId: str):
    return "users:"+userId

def _expirationKey(session_started_at: int, duration: int):
    session_idx = int(session_started_at / duration)

    return "to_expire:"+str(session_idx)

def disable_user(now: int, user_id: str):
    r = _get_client()

    return r.hmset(_userKey(user_id), mapping={
        "user_id": user_id,
        "disabled_at": now,
    })

def rollback_disabled_user(user_id: str):
    r = _get_client()

    return r.hdel(_userKey(user_id), "disabled_at")

def update_user(
    user_id: str,
    session_id: str,
    emotions: str,
    session_started_at,
    last_negative_request_at,
    disabled_at,
    emotion_sequence,
    best_pictures_score
):
    r = _get_client()
    if not set_expired(session_started_at, user_id):
        return False

    return r.hmset(_userKey(user_id),mapping={
        "user_id": user_id,
        "session_id": session_id,
        "emotions": emotions,
        "session_started_at": session_started_at,
        "disabled_at": disabled_at,
        "last_negative_request_at": last_negative_request_at,
        "emotion_sequence": emotion_sequence,
        "best_pictures_score": ":".join([str(b) for b in best_pictures_score])
    })

def update_emotions_and_best_score(usr,emotions: list, emotion_sequence: int, best_score: list, last_negative_request_at = None):
    r = _get_client()
    if last_negative_request_at or usr['last_negative_request_at']:
        r.hdel(_expirationKey(usr["session_started_at"],current_app.config["SESSION_DURATION"]),usr['user_id'])

    return r.hmset(_userKey(usr['user_id']),mapping={
        "user_id": usr['user_id'],
        "emotions": emotions,
        "last_negative_request_at": last_negative_request_at or usr['last_negative_request_at'],
        "emotion_sequence": emotion_sequence,
        "best_pictures_score": ":".join([str(b) for b in best_score])
    })

def update_last_negative_request_at(usr, now: int):
    r = _get_client()

    res = r.hset(_userKey(usr["user_id"]),mapping={
        "user_id": usr["user_id"],
        "last_negative_request_at": now,
    })
    res = res and r.hdel(_expirationKey(usr["session_started_at"], current_app.config["SESSION_DURATION"]),usr['user_id'])

    return res

def decrease_available_retries(usr, user_id):
    r = _get_client()

    available_retries = current_app.config["PRIMARY_PHOTO_RETRIES"] - 1
    if usr is not None and usr["available_retries"] != 0:
        available_retries = usr["available_retries"] - 1

    return r.hset(_userKey(user_id), mapping={
        "user_id": user_id,
        "available_retries": available_retries,
    })

def get_user(user_id: str, search_growing = True):
    r = _get_client()
    hkeys = [
              "session_id",
              "emotions",
              "session_started_at",
              "disabled_at",
              "last_negative_request_at",
              "emotion_sequence",
              "best_pictures_score",
              "available_retries",
              "possible_duplicate_with",
              "duplicate_review_count",
              "ip"
            ]
    mappers = {
        "session_id": lambda x: str(x, encoding = "utf-8") if x else "",
        "emotions": lambda x: str(x, encoding = "utf-8") if x else "",
        "session_started_at": int,
        "disabled_at": int,
        "last_negative_request_at": int,
        "emotion_sequence": lambda x: int(x) if x else 0,
        "best_pictures_score": lambda x: x,
        "available_retries": int,
        "duplicate_review_count": lambda x: int(x) if x else 0,
        "possible_duplicate_with": lambda x: str(x, encoding = "utf-8").split(",") if x else [],
        "ip":lambda x: str(x, encoding = "utf-8") if x else ""
    }
    res = r.hmget(_userKey(user_id),hkeys)
    if res.count(None) == len(hkeys):
        return None

    res = dict(zip(hkeys,res))
    if res.get("best_pictures_score"):
        res['best_pictures_score'] = [float(str(b)) for b in str(res['best_pictures_score'], encoding = "utf-8").split(":")]
    if res.get('session_started_at') is None:
        res['session_started_at'] = 0
    if res.get('disabled_at') is None:
        res['disabled_at'] = 0
    if res.get('last_negative_request_at') is None:
        res['last_negative_request_at'] = 0
    if res.get('available_retries') is None:
        res['available_retries'] = 0
    for k in res:
        res[k] = mappers[k](res[k])
    res['user_id'] = user_id

    return res

def get_expired_sessions(now, duration):
    r = _get_client()
    expired_users = r.hgetall(_expirationKey(now-duration, duration)).keys()
    if expired_users is None or None in expired_users:
        return []

    return [str(e, encoding = "utf-8") for e in expired_users]

def remove_expired(session_started_at, user_id):
    r = _get_client()

    if session_started_at is not None:
        return r.hdel(_expirationKey(session_started_at, current_app.config["SESSION_DURATION"]), user_id)

    return True

def remove_session(user_id: str):
    r = _get_client()

    return r.hdel(_userKey(user_id), "session_id")

def full_user_reset(user_id: str, prev_state = None):
    r = _get_client()

    res = r.delete(_userKey(user_id))
    if res:
        r.srem("users_pending_duplicate_review", user_id)
    if prev_state:
        r.hset(_userKey(user_id),mapping = {
            "user_id": user_id,
            "duplicate_review_count": prev_state.get("duplicate_review_count",0)
        })
    return res

def enable_user(user_id: str):
    r = _get_client()

    return r.hdel(_userKey(user_id), "disabled_at")

def set_expired(session_started_at, user_id):
    r = _get_client()

    return r.hmset(_expirationKey(session_started_at, current_app.config["SESSION_DURATION"]), mapping = {
        user_id: 1
    })

def get_disabled_user_for_selfie_reprocessing():
    r = _get_client()
    user_id  = r.spop("wrongfully_disabled_users", count=1)
    if user_id is None or len(user_id) == 0:
        return None
    return str(user_id[0], encoding= "utf-8")
def get_admin_token():
    r = _get_client()
    t = r.get("admin_token")
    if t:
        return str(t, encoding="utf-8")
    return None
def put_disabled_user_for_selfie_reprocessing(user_id: str):
    r = _get_client()
    return r.sadd("wrongfully_disabled_users", user_id) > 0

def register_wrongfully_disabled_users_worker():
    r = _get_client()
    r.sadd("wrongfully_disabled_users_workers", os.getpid())
    return int(r.scard("wrongfully_disabled_users_workers"))
def unregister_wrongfully_disabled_users_worker():
    r = _get_client()
    r.spop("wrongfully_disabled_users_workers",1)
    return int(r.scard("wrongfully_disabled_users_workers"))
def clean_wrongfully_disabled_users_workers():
    r = _get_client()
    r.delete("wrongfully_disabled_users_workers",1)

def mark_user_for_manual_review(user_id: str, ip: str, similar_users: List[str], duplicate_review_count: int):
    r = _get_client()
    if r.hset(_userKey(user_id),mapping = {
        "ip":ip,
        "possible_duplicate_with": ",".join(similar_users)
    }) > 0:
        return r.sadd("users_pending_duplicate_review", user_id) > 0
    return False
def rollback_manual_review(user_id: str):
    r = _get_client()
    print("rollback")
    if r.hdel(_userKey(user_id), "possible_duplicate_with") > 0:
        return r.srem("users_pending_duplicate_review", user_id) > 0
    return False

def allocate_review_user(admin_id: str):
    r = _get_client()
    user_id = r.get(f"user_pending_duplicate_review_{admin_id}")
    if user_id and len(user_id):
        return str(user_id,encoding = "utf-8")
    user_id = r.spop("users_pending_duplicate_review")
    if user_id:
        user_id = str(user_id, encoding = "utf-8")
        r.set(f"user_pending_duplicate_review_{admin_id}", user_id)
        return user_id
    return None
def user_reviewed(admin_id: str, user_id: str, retry = False):
    r = _get_client()
    with r.pipeline(transaction=True) as p:
        if retry:
            p.hincrby(_userKey(user_id),"duplicate_review_count")
        else:
            p.hdel(_userKey(user_id),"duplicate_review_count")
        p.hdel(_userKey(user_id),"possible_duplicate_with")
        p.delete(f"user_pending_duplicate_review_{admin_id}")
        p.execute()
def pop_possible_duplicate_with(user_id, user, most_similar_user_id):
    r = _get_client()
    dupls = user.get("possible_duplicate_with", [])
    if most_similar_user_id in dupls:
        dupls.remove(most_similar_user_id)
    else:
        return user
    user["possible_duplicate_with"] = dupls
    r.hset(_userKey(user_id), mapping = {"possible_duplicate_with":",".join(dupls)})
    return user
def rollback_pop_possible_duplicate_with(user_id, user, most_similar_user_id):
    r = _get_client()
    dupls = user.get("possible_duplicate_with", [])
    if not most_similar_user_id in dupls:
        dupls.append(most_similar_user_id)
    else:
        return user
    user["possible_duplicate_with"] = dupls
    r.hset(_userKey(user_id), mapping = {"possible_duplicate_with":",".join(dupls)})
    return user
def rollback_reviewed(admin_id: str, user_id: str, user: dict, retry = False,):
    r = _get_client()
    with r.pipeline(transaction=True) as p:
        if retry:
            p.hincr(_userKey(user_id),"duplicate_review_count", -1)
        else:
            p.hset(_userKey(user_id),mapping = {"duplicate_review_count":user.get("duplicate_review_count", 0)})
        p.hset(_userKey(user_id), mapping = {"possible_duplicate_with":",".join(user.get("possible_duplicate_with",[]))})
        p.set(f"user_pending_duplicate_review_{admin_id}", user_id)
        p.execute()

def is_review_disabled():
    r = _get_client()
    return r.get("disable_duplicate_review") is not None

def ping():
    r = _get_client()

    return r.ping()
