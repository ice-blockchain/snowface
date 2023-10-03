import time

from flask import g
import os
from pymilvus import CollectionSchema, FieldSchema, DataType, utility, connections, Collection, Milvus, MilvusException
import numpy as np
from flask import current_app
_conn_prefix = "snowface-"+str(os.getpid())
_faces_collection = None
_users_collection = None

_picture_primary = 0
_picture_secondary = 1

def connect_milvus():
    connections.connect(
        alias=_conn_prefix,
        user=current_app.config["MILVUS_USER"],
        password=current_app.config["MILVUS_PASSWORD"],
        uri=current_app.config['MILVUS_URI']
    )

def init_milvus():
    if not connections.has_connection(_conn_prefix):
        connect_milvus()
    init_schema()
def close_milvus():
    connections.disconnect(alias="default")

def on_exit(arbiter):
    connections.disconnect(alias="default")

def init_collection(name, create_fn):
    db = None
    if not utility.has_collection(name, using=_conn_prefix):
        db = create_fn(name)
        try: db.load()
        except MilvusException as e:
            if e.code == 5:
                db.release()
                db.load()
    else:
        db = Collection(name=name, using=_conn_prefix)
        try: db.load()
        except MilvusException as e:
            if e.code == 5:
                db.release()
                db.load()
    return db

def init_schema():
    _users_collection = init_collection("users", create_users_collection)
    _faces_collection = init_collection("faces", create_faces_collection)

def create_users_collection(name):
    fields = [
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=40, is_primary=True),
        FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="emotions", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="session_started_at", dtype=DataType.INT64),
        FieldSchema(name="disabled_at", dtype=DataType.INT64),
        FieldSchema(name="last_negative_request_at", dtype=DataType.INT64),
        FieldSchema(name="emotion_sequence", dtype=DataType.INT64),
        FieldSchema(name="best_pictures_score", dtype=DataType.FLOAT_VECTOR, dim = 45),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="users"
    )
    users = Collection(
        name=name,
        schema=schema,
        using=_conn_prefix
    )
    users.create_index(
        field_name="best_pictures_score",
        index_params={
            "metric_type":"L2",
            "index_type":"HNSW",
            "params":{"efConstruction":512, "M":16}
        },

    )

    return users

def create_faces_collection(name):
    faces = Collection(
        name=name,
        schema=CollectionSchema(
            fields=[FieldSchema(
                name="user_picture_id", #user_id~picture_id
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=50, #40+str(2^32)
            ), FieldSchema(
                name="user_id",
                dtype=DataType.VARCHAR,
                # iceIDs are 4prefix(ice_) + 36 uuid, firebase 36 max as well
                # https://groups.google.com/g/firebase-talk/c/5ENGCX8y04M
                max_length=40
            ), FieldSchema(
                name="picture_id",
                dtype=DataType.INT32
            ), FieldSchema(
                name="face_metadata",
                dtype=DataType.FLOAT_VECTOR,
                dim=128
            ), FieldSchema(
                name="url",
                dtype=DataType.VARCHAR,
                max_length = 1024
            ),FieldSchema(
                name="uploaded_at",
                dtype=DataType.INT64,
            )],
            description="Faces search "
        ),
        using=_conn_prefix,
        # partitions??? Shards???
    )
    faces.create_index(
        field_name="face_metadata",
        index_params= {
            "metric_type":"L2",
            "index_type":"HNSW",
            "params":{"efConstruction":512, "M":16}
        })
    faces.create_index(
        field_name="picture_id",
        index_name="picture_id_idx")
    return faces

def get_faces_collection():
    global _faces_collection
    if not connections.has_connection(_conn_prefix):
        connect_milvus()
    if _faces_collection is None:
        _faces_collection = init_collection("faces", create_faces_collection)

    return _faces_collection

def get_users_collection():
    global _users_collection
    if not connections.has_connection(_conn_prefix):
        connect_milvus()
    if _users_collection is None:
        _users_collection = init_collection("users", create_users_collection)

    return _users_collection

def get_primary_metadata(user_id):
    faces = get_faces_collection()
    res = faces.query(
        expr = f"user_picture_id == \"{user_id}~{_picture_primary}\"",
        offset = 0,
        limit = 1,
        output_fields = ["user_id","picture_id","face_metadata","uploaded_at", "url"],
        ignore_growing = False,
        consistency_level = "Strong"
    )
    if len(res) == 0:
        return None
    return res[0]

def get_secondary_metadata(user_id):
    faces = get_faces_collection()
    res = faces.query(
        expr = f"user_picture_id == \"{user_id}~{_picture_secondary}\"",
        offset = 0,
        limit = 1,
        output_fields = ["user_id","picture_id","face_metadata","uploaded_at", "url"],
    )
    if len(res) == 0:
        return None
    return res[0]

def find_similar_users(user_id: str,metadata: list, threshold: float):
    faces = get_faces_collection()
    results = faces.search(
        data=[metadata],
        anns_field="face_metadata",
        param={
            "metric_type": "L2",
            "offset": 0,
            "ignore_growing": False,
            "params": {"ef": 20}
        },
        limit=5,
        expr = f"picture_id == {_picture_primary}",
        output_fields=['user_id']
    )
    if len(results) == 0:
        return [user_id],[]
    if len(results[0].ids) == 0:
        return [user_id],[]
    r = zip(results[0].ids, results[0].distances)
    found_user_ids = []
    distances = []
    [(found_user_ids.append(found_user_id.split("~")[0]),distances.append(distance))
      for found_user_id, distance in r
      if (distance <= threshold or distance == 0.0) and found_user_id not in found_user_ids]
    #print(len(results),results[0].ids, results[0].distances, len(found_user_ids))
    if len(found_user_ids) == 0:
        return [user_id],[]
    return found_user_ids, distances


def update_secondary_metadata(now: int, user_id:str, metadata: list, url: str):
    faces = get_faces_collection()
    pk = f"{user_id}~{_picture_secondary}"
    rowsCount = faces.upsert([[pk],[user_id],[np.int32(_picture_secondary)],[metadata],[url],[now]]).upsert_count
    return {
        "user_picture_id": pk,
        "user_id": user_id,
        "picture_id": np.int32(_picture_secondary),
        "face_metadata": metadata,
        "url": url,
        "uploaded_at": now
    }, rowsCount

def set_primary_metadata(now: int, user_id:str, metadata: list, url: str):
    faces = get_faces_collection()
    pk = f"{user_id}~{_picture_primary}"
    insertedRows = faces.insert([[pk],[user_id],[np.int32(_picture_primary)],[metadata],[url],[now]]).insert_count
    # t = time.time()
    # faces.flush()
    # print("flush took", time.time() - t)
    return {
        "user_picture_id": pk,
        "user_id": user_id,
        "picture_id": np.int32(_picture_primary),
        "face_metadata": metadata,
        "url": url,
        "uploaded_at": now
    }, insertedRows
def delete_metadatas(user_id: str, pk: list):
    faces = get_faces_collection()
    primary = get_primary_metadata(user_id)
    secondary = get_secondary_metadata(user_id)
    return primary, secondary, faces.delete(f"user_picture_id in {pk}").delete_count

def disable_user(now: int, user_id: str):
    users = get_users_collection()
    user = get_user(user_id)
    if user is not None:
        insertedRows = users.upsert([[user_id], [user["session_id"]], [user["emotions"]],[user["session_started_at"]], [now], [user["last_negative_request_at"]], [user["emotion_sequence"]], [user["best_pictures_score"]]]).upsert_count
    else:
        insertedRows = users.upsert([[user_id], [""], [""],[0], [now], [0], [0], [np.array([0.0]*45)]]).upsert_count
    return insertedRows > 0

def update_user(
    user_id: str,
    session_id: str,
    emotions: str,
    session_started_at,
    disabled_at,
    emotion_sequence,
    best_pictures_score,
    now
):
    users = get_users_collection()
    user = get_user(user_id)
    if user is not None:
        insertedRows = users.upsert([
            [user_id],
            [session_id],
            [emotions],
            [session_started_at],
            [disabled_at],
            [user['last_negative_request_at']],
            [emotion_sequence],
            [best_pictures_score]
        ]).upsert_count
    else:
        insertedRows = users.upsert([
            [user_id],
            [session_id],
            [emotions],
            [now],
            [0],
            [0],
            [0],
            [np.array([0.0]*45)]
        ]).upsert_count

    return insertedRows > 0

def update_emotions_and_best_score(usr,emotions: list, emotion_sequence: int, best_score: list, last_negative_request_at = None):
    users = get_users_collection()
    insertedRows = users.upsert([
        [usr['user_id']],
        [usr['session_id']],
        [emotions],
        [usr['session_started_at']],
        [usr['disabled_at']],
        [last_negative_request_at or usr['last_negative_request_at']],
        [emotion_sequence],
        [best_score]
    ]).upsert_count

    return insertedRows > 0

def update_last_negative_request_at(usr, now: int):
    users = get_users_collection()
    insertedRows = users.upsert([
        [usr['user_id']],
        [usr['session_id']],
        [usr['emotions']],
        [usr['session_started_at']],
        [usr['disabled_at']],
        [now],
        [usr['emotion_sequence']],
        [usr['best_pictures_score']]
    ]).upsert_count

    return insertedRows > 0

def get_user(user_id: str):
    users = get_users_collection()
    res = users.query(
        expr = f"user_id == \"{user_id}\"",
        offset = 0,
        limit = 1,
        output_fields = ["user_id", "session_id", "emotions", "session_started_at", "disabled_at", "last_negative_request_at", "emotion_sequence", "best_pictures_score"],
        ignore_growing = False,
        consistency_level = "Strong"
    )
    if len(res) == 0:
        return None

    return res[0]

def remove_session(user_id: str):
    users = get_users_collection()
    expr = f"user_id in [\"{user_id}\"]"
    users.delete(expr)
