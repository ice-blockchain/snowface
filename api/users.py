import redis, os
from flask import current_app

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
    return r.hset(_userKey(user_id),mapping={
        "user_id": user_id,
        "disabled_at": now,
    })
def update_user(
    user_id: str,
    session_id: str,
    emotions: str,
    session_started_at,
    last_negative_request_at,
    disabled_at,
    emotion_sequence,
    best_pictures_score,
    now
):
    r = _get_client()
    if not r.hset(_expirationKey(session_started_at, current_app.config["SESSION_DURATION"]),mapping = {
        user_id:1
    }):
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
    if usr is not None:
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
              "available_retries"
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


def remove_session(user_id: str, duration:int = 0):
    if not duration:
        duration = current_app.config["SESSION_DURATION"]
    r = _get_client()
    u = get_user(user_id)
    if not u:
        return
    res = r.delete(_userKey(user_id))
    res = res and r.hdel(_expirationKey(u["session_started_at"], duration),user_id)
    return res

def ping():
    r = _get_client()
    return r.ping()