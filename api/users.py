import json
import uuid

import redis, os
from flask import current_app
from typing import List
from exceptions import EmailOrPhoneNumberNotUnique, NoEmailAndPhoneNumber
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
def _emailsKey():
    return "emails"
def _phoneNumbersKey():
    return "phoneNumbers"
def _pendingFace(userId: str):
    return "pendingFace:"+userId
def _reviewMetadata(userId: str):
    return "reviewMetadata:"+userId

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
              "phone_number",
              "email",
              "ip",
              "primary_uploaded_at",
              "secondary_uploaded_at"
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
        "email": lambda x: str(x, encoding = "utf-8") if x else "",
        "phone_number": lambda x: str(x, encoding = "utf-8") if x else "",
        "ip":lambda x: str(x, encoding = "utf-8") if x else "",
        "primary_uploaded_at": lambda x: int(x) if x else 0,
        "secondary_uploaded_at": lambda x: int(x) if x else 0,
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
    r.hdel(_userKey(user_id), "session_id", "similarity_response","similarity_code", "url","uploaded_at")
    r.delete(_pendingFace(user_id))

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

def mark_user_for_manual_review(user_id: str, ip: str, similar_users: List[str], duplicate_review_count: int, metadata):
    r = _get_client()
    with r.pipeline(transaction=True) as p:
        r.delete(_reviewMetadata(user_id))
        r.rpush(_reviewMetadata(user_id), *metadata)
        if r.hset(_userKey(user_id),mapping = {
            "ip":ip,
            "possible_duplicate_with": ",".join(similar_users),
            "duplicate_review_count": duplicate_review_count
        }) > 0:
            r.sadd("users_pending_duplicate_review", user_id)
        p.execute()
    return False

def get_face_metadata_pending_review(user_id: str):
    r = _get_client()
    return r.lrange(_reviewMetadata(user_id), 0, -1)
def rollback_manual_review(user_id: str):
    r = _get_client()
    print("rollback")
    if r.hdel(_userKey(user_id), "possible_duplicate_with") > 0:
        return r.srem("users_pending_duplicate_review", user_id) > 0
    return False

def allocate_review_user(admin_id: str):
    r = _get_client()
    user_id = r.get(f"user_pending_duplicate_review_{admin_id}")
    review_queue_len = r.scard("users_pending_duplicate_review")
    if user_id and len(user_id):
        return str(user_id,encoding = "utf-8"), int(review_queue_len)
    user_id = r.spop("users_pending_duplicate_review")
    if user_id:
        user_id = str(user_id, encoding = "utf-8")
        r.set(f"user_pending_duplicate_review_{admin_id}", user_id)
        return user_id, int(review_queue_len)-1
    return None, 0
def user_reviewed(admin_id: str, user_id: str, retry = False):
    r = _get_client()
    with r.pipeline(transaction=True) as p:
        if retry:
            p.hincrby(_userKey(user_id),"duplicate_review_count")
        else:
            p.hdel(_userKey(user_id),"duplicate_review_count")
        p.hdel(_userKey(user_id),"possible_duplicate_with","ip")
        p.delete(f"user_pending_duplicate_review_{admin_id}")
        p.delete(_reviewMetadata(user_id))
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
def add_possible_duplicate_with(user_id, user, most_similar_user_ids):
    r = _get_client()
    dupls = user.get("possible_duplicate_with", [])
    diff = set(most_similar_user_ids)-set(dupls)
    if len(diff) > 0:
        dupls.extend(diff)
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

def update_secondary_metadata_pending(now: int, user_id:str, metadata: list, model: str):
    r = _get_client()
    with r.pipeline(transaction=True) as p:
        r.hset(_userKey(user_id),mapping = {
            "uploaded_at": now,
        })
        r.delete(_pendingFace(user_id))
        r.rpush(_pendingFace(user_id), *metadata)
        p.execute()
    return {
        "user_picture_id": f"{user_id}~2",
        "user_id": user_id,
        "picture_id": 2,
        "face_metadata": metadata,
        "uploaded_at": now
    }, 1

def get_user_similarity_resp(user_id: str):
    r = _get_client()
    return r.hmget(_userKey(user_id), ["similarity_code", "similarity_response"])

def get_pending_face(user_id: str):
    r = _get_client()
    res = r.hmget(_userKey(user_id), ["url", "uploaded_at"])
    res.append(r.lrange(_pendingFace(user_id), 0, -1))
    return res

def put_user_similarity_resp(now:int, user_id: str, code: int, resp: bytes):
    r = _get_client()
    r.hset(_userKey(user_id), mapping= {
        "similarity_response": resp,
        "similarity_code": code,
        "similarity_finished_at": now
    })

def is_review_disabled():
    r = _get_client()
    return r.get("disable_duplicate_review") is not None

def ping():
    r = _get_client()

    return r.ping()

def mark_photo_uploaded(user_id, face_md, primary):
    if primary:
        hkey = "primary_uploaded_at"
    else:
        hkey = "secondary_uploaded_at"
    r = _get_client()
    r.hset(_userKey(user_id),mapping = {
        hkey: face_md["uploaded_at"] if face_md else 0
    })

def register_unique_email_and_phone_number(user_id, user):
    r = _get_client()
    email = user.get("email","")
    phone_number = user.get("phone_number","")
    mapping = {}
    unique = False
    ion_id = ionID(email, phone_number)
    ids = json.dumps({"user_id": user_id,"ion_id":ion_id})
    if email:
        unique = r.hset(_emailsKey(), mapping={
            email: ids
        }) == 1
        mapping["email"] = email
    if (not email or (email and unique)) and phone_number:
        hs = r.hset(_phoneNumbersKey(), mapping={
            phone_number: ids
        })
        unique = hs == 1
        mapping["phone_number"] = phone_number
    if mapping:
        mapping["ion_id"] = ion_id
        if unique:
            r.hset(_userKey(user_id), mapping=mapping)
        else:
            raise EmailOrPhoneNumberNotUnique("not unique "+ ("phone number" if "phone_number" in mapping.keys() else "email "))
def rollback_unique_email_and_phone_number(user_id, user):
    r = _get_client()
    email = user.get("email","")
    phone_number = user.get("phone_number","")
    if email:
        r.hdel(_emailsKey(), email)
    if phone_number:
        r.hdel(_phoneNumbersKey(), phone_number)
    r.hdel(_userKey(user_id), "email", "phone_number")

def ionID(email, phone_number):
    data = None
    if email:
        id = _get_ionID(_emailsKey(), email)
        if not id:
            data = f"e:{email}"
    elif phone_number:
        id = _get_ionID(_phoneNumbersKey(), phone_number)
        if not id:
            data = f"p:{phone_number}"
    else:
        raise NoEmailAndPhoneNumber("at least one of (email, phone_number) must be provided")
    if data:
        uid = uuid.uuid5(uuid.UUID("00000000-0000-0000-0000-000000000000"),data)
        id = str(uid)
    return id

def _get_ionID(key: str, search: str):
    r = _get_client()
    val = r.hget(key, search)
    if val is None:
        return None
    ids = json.loads(str(val, encoding = "utf-8"))
    return ids["ion_id"]