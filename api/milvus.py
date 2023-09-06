import time

from flask import g
import os
from pymilvus import CollectionSchema, FieldSchema, DataType, utility, connections, Collection, Milvus
import numpy as np

_conn_prefix = "default"
_faces_collection = None
_disabled_users_collection = None

_picture_primary = 0
_picture_secondary = 1

def init_milvus():
    if not connections.has_connection(_conn_prefix):
        connections.connect(
            alias=_conn_prefix,
            user=os.getenv("MILVUS_USER","root"),
            password=os.getenv("MILVUS_PASSWORD","Milvus"),
            uri=os.getenv("MILVUS_URI", "http://localhost:19530")
        )
    init_schema()
def close_milvus():
    connections.disconnect(alias="default")

def on_exit(arbiter):
    connections.disconnect(alias="default")

def init_collection(name, create_fn):
    db = None
    if not utility.has_collection(name):
        db = create_fn(name)
        db.load()
    else:
        db = Collection(name=name)
        db.load()
    return db

def init_schema():
    init_collection("disabled_users", create_disabled_users_collection)
    init_collection("faces", create_faces_collection)

def create_disabled_users_collection(name):
    users = Collection(
        name = name,
        schema = CollectionSchema(
            fields = [
                FieldSchema(
                    name = "user_id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length = 40
                ),
                FieldSchema(
                    name = "disabled",
                    dtype=DataType.FLOAT_VECTOR,
                    dim = 1
                ),
                FieldSchema(
                    name = "disabled_at",
                    dtype=DataType.INT64
                ),
            ]
        )
    )
    users.create_index(
        field_name="disabled",
        index_params= {
            "metric_type":"L2",
            "index_type":"IVF_FLAT",
            "params":{"nlist":1}
        })
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
        using='default',
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
        connections.connect(
            alias=_conn_prefix,
            user=os.getenv("MILVUS_USER","root"),
            password=os.getenv("MILVUS_PASSWORD","Milvus"),
            uri=os.getenv("MILVUS_URI", "http://localhost:19530")
        )
    if _faces_collection is None:
        _faces_collection = init_collection("faces", create_faces_collection)

    return _faces_collection

def get_disabled_users_collection():
    global _disabled_users_collection
    if not connections.has_connection(_conn_prefix):
        connections.connect(
            alias=_conn_prefix,
            user=os.getenv("MILVUS_USER","root"),
            password=os.getenv("MILVUS_PASSWORD","Milvus"),
            uri=os.getenv("MILVUS_URI", "http://localhost:19530")
        )
    if _disabled_users_collection is None:
        _disabled_users_collection = init_collection("disabled_users", create_disabled_users_collection)

    return _disabled_users_collection

def get_primary_metadata(user_id):
    faces = get_faces_collection()
    res = faces.query(
        expr = f"user_picture_id == \"{user_id}~{_picture_primary}\"",
        offset = 0,
        limit = 1,
        output_fields = ["user_id","picture_id","face_metadata","uploaded_at", "url"],
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
    [(found_user_ids.append(found_user_id),distances.append(distance))
      for found_user_id, distance in r
      if distance <= threshold and found_user_id not in found_user_ids]
    return found_user_ids, distances


def update_secondary_metadata(user_id:str, metadata: list, url: str):
    faces = get_faces_collection()
    pk = f"{user_id}~{_picture_secondary}"
    now = int(time.time()*1e9)
    rowsCount = faces.upsert([[pk],[user_id],[np.int32(_picture_secondary)],[metadata],[url],[now]]).upsert_count
    return now, rowsCount

def set_primary_metadata(user_id:str, metadata: list, url: str):
    faces = get_faces_collection()
    pk = f"{user_id}~{_picture_primary}"
    now = int(time.time()*1e9)
    insertedRows = faces.insert([[pk],[user_id],[np.int32(_picture_primary)],[metadata],[url],[now]]).insert_count
    return now, insertedRows

def disable_user(user_id: str):
    users = get_disabled_users_collection()
    now = int(time.time()*1e9)
    insertedRows = users.insert([[user_id], [np.array([1.0],dtype=float)], [now]]).insert_count
    return insertedRows > 0

def disabled_user(user_id: str):
    users = get_disabled_users_collection()
    res = users.query(
        expr = f"user_id == \"{user_id}\"",
        offset = 0,
        limit = 1,
        output_fields = ["user_id","disabled_at"],
    )
    if len(res) == 0:
        return None
    return res[0]