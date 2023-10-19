import os

import minio.error
import requests
from minio import Minio
from minio.error import S3Error
from minio.deleteobjects import DeleteObject
import io, datetime
from flask import current_app

_minio_client = None

_bucket_name = "photos"
_disabled_users_bucket_name = "disabled"

_picture_primary = 0
_picture_secondary = 1


def _get_minio_client():
    global _minio_client
    if _minio_client is None:
        ssl = current_app.config["MINIO_SSL"]
        _minio_client = Minio(
            current_app.config["MINIO_URI"],
            access_key=current_app.config["MINIO_ACCESS_KEY"],
            secret_key=current_app.config["MINIO_SECRET_KEY"],
            secure=ssl
        )
    return _minio_client

def _client_with_initialized_bucket():
    client = _get_minio_client()
    if not client.bucket_exists(_bucket_name):
        try: client.make_bucket(_bucket_name)
        except minio.error.S3Error as e:
            if e.code != "BucketAlreadyOwnedByYou":
                raise e
    if not client.bucket_exists(_disabled_users_bucket_name):
        try: client.make_bucket(_disabled_users_bucket_name)
        except minio.error.S3Error as e:
            if e.code != "BucketAlreadyOwnedByYou":
                raise e
    return client

def put_primary_photo(user_id: str, photo_content):
    return put_proto(user_id, _picture_primary, photo_content)

def get_primary_photo(user_id: str):
    return get_photo(user_id, _picture_primary)
def put_secondary_photo(user_id: str, photo_content):
    return put_proto(user_id, _picture_secondary, photo_content)

def put_disabled_photo(user_id: str, photo_content):
    obj_name = f"{user_id}"
    return _upload(_disabled_users_bucket_name,obj_name,photo_content)


def put_proto(user_id: str, photo_id: int, photo_content):
    obj_name = f"{user_id}/{photo_id}"
    return _upload(_bucket_name,obj_name,photo_content)


def _upload(bucket, obj_name: str, photo_content):
    l = photo_content.seek(0, os.SEEK_END)
    photo_content.seek(0, os.SEEK_SET)
    client = _client_with_initialized_bucket()
    res = client.put_object(
        bucket_name=bucket,
        object_name=obj_name,
        data=photo_content,
        length=l,
    )
    return "/" + _bucket_name + "/" + obj_name

def get_photo(user_id: str, photo_id: int):
    try:
        client = _client_with_initialized_bucket()
        obj_name = f"{user_id}/{photo_id}"
        res = client.get_object(_bucket_name, obj_name)
        data = res.data
        res.close()
        res.release_conn()
        return data
    except minio.error.S3Error as e:
        if e.code == "NoSuchKey":
            return None

def delete_photos(user_id):
    client = _client_with_initialized_bucket()
    main_photo = get_primary_photo(user_id)
    secondary = get_photo(user_id, _picture_secondary)
    folder_obj_name = f"{user_id}"

    errs = client.remove_objects(_bucket_name,[DeleteObject(i.object_name) for i in
                                               client.list_objects(_bucket_name,prefix=folder_obj_name,recursive=True)]
                                            + [DeleteObject(folder_obj_name)]
                                 )
    list(client.remove_objects(_disabled_users_bucket_name,[DeleteObject(f"{user_id}")]))
    return main_photo, secondary, list(errs)

def ping(timeout = 30):
    client = _client_with_initialized_bucket()
    healthcheck_url = "/minio/health/cluster"
    ssl = current_app.config["MINIO_SSL"]
    minio_url = current_app.config["MINIO_URI"]
    resp = requests.get(url=("https://" if ssl else "http://") + minio_url +healthcheck_url,
                        verify=not ssl,
                        timeout=timeout)
    resp.raise_for_status()