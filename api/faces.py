import time

from flask import g
import os
from pymilvus import CollectionSchema, FieldSchema, DataType, utility, connections, Collection, Milvus, MilvusException
from pymilvus.client.types import LoadState
from pymilvus.exceptions import IndexNotExistException
import numpy as np
from flask import current_app
_conn_prefix = "snowface-"+str(os.getpid())
_faces_collection = None
_users_collection = None

_picture_primary = 0
_picture_secondary = 1

_default_user = 'root'
_default_password = 'Milvus'

def connect_milvus():
    uri = os.environ.get('MILVUS_URI')
    usr = os.environ.get('MILVUS_USER', _default_user)
    passw = os.environ.get('MILVUS_PASSWORD', _default_password)
    connections.connect(
        alias=_conn_prefix,
        user=usr,
        password=passw,
        uri=uri
    )

def init_milvus():
    if not connections.has_connection(_conn_prefix):
        connect_milvus()
    init_schema()
def close_milvus():
    connections.disconnect(alias="default")

def on_exit(arbiter):
    connections.disconnect(alias="default")

def init_collection(name, create_fn, extra_indexes = None):
    db = None
    if not utility.has_collection(name, using=_conn_prefix):
        db = create_fn(name)
        if extra_indexes:
            for ind_field in extra_indexes.keys():
                index = [idx for idx in db.indexes if idx.field_name == ind_field]
                if len(index) == 0:
                    extra_indexes.get(ind_field, lambda x: None)(db)
        try: db.load()
        except MilvusException as e:
            if e.code == 5:
                db.release()
                db.load()
    else:
        db = Collection(name=name, using=_conn_prefix)
        if extra_indexes:
            for ind_field in extra_indexes.keys():
                index = [idx for idx in db.indexes if idx.field_name == ind_field]
                if len(index) == 0:
                    extra_indexes.get(ind_field, lambda x: None)(db)
        try: db.load()
        except MilvusException as e:
            if e.code == 5:
                db.release()
                db.load()
    return db

def init_schema():
    _faces_collection = init_collection("faces", create_faces_collection)


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
                dim=512
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

def get_primary_metadata(user_id, search_growing = True):
    faces = get_faces_collection()
    res = faces.query(
        expr = f"user_picture_id == \"{user_id}~{_picture_primary}\"",
        offset = 0,
        limit = 1,
        output_fields = ["user_id","picture_id","face_metadata","uploaded_at", "url"],
        ignore_growing = False,
        consistency_level = "Strong" if search_growing else "Bounded"
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

def ping(timeout = 30):
    if _faces_collection is None: get_faces_collection()
    state = utility.load_state(_faces_collection.name, using=_conn_prefix, timeout=timeout)
    if state != LoadState.Loaded:
        raise Exception(f"Collection {_faces_collection.name} is in {state} state")