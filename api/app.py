# 3rd parth dependencies
from flask import Flask
from routes import blueprint

from milvus import init_milvus, close_milvus
from auth import _get_firebase_client
from minio_uploader import _client_with_initialized_bucket
from service import init_models
from routes import init_rate_limiters
import os

def create_app():
    app = Flask(__name__)
    app.register_blueprint(blueprint)
    jwt_secret = os.environ.get('JWT_SECRET')
    app.config['JWT_SECRET'] = jwt_secret
    if not app.config['JWT_SECRET']:
        raise Exception("JWT_SECRET not set")
    fp = os.environ.get("AUTH_CREDENTIALS_FILE_PATH")
    if fp:
        firebase_file_content = open(fp,'r').read()
    else:
        firebase_file_content = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not firebase_file_content:
        raise Exception("failed to init firebase auth: GOOGLE_APPLICATION_CREDENTIALS not set")
    app.config['MILVUS_URI'] = os.environ.get('MILVUS_URI')
    app.config['MILVUS_USER'] = os.environ.get('MILVUS_USER', 'root')
    app.config['MILVUS_PASSWORD'] = os.environ.get('MILVUS_PASSWORD', 'Milvus')

    app.config['MINIO_URI'] = os.environ.get('MINIO_URI')
    app.config['MINIO_ACCESS_KEY'] = os.environ.get('MINIO_ACCESS_KEY','minioadmin')
    app.config['MINIO_SECRET_KEY'] = os.environ.get('MINIO_SECRET_KEY','minioadmin')
    app.config['MINIO_SSL'] = os.environ.get("MINIO_SSL", 'False').lower() in ('true', '1')

    app.config['GOOGLE_APPLICATION_CREDENTIALS'] = firebase_file_content
    app.config['METADATA_UPDATED_CALLBACK_URL'] = os.environ.get("METADATA_UPDATED_CALLBACK_URL")
    app.config['METADATA_UPDATED_SECRET'] = os.environ.get("METADATA_UPDATED_SECRET")
    app.config['PRIMARY_PHOTO_ERROR_LIMIT'] = os.environ.get("PRIMARY_PHOTO_ERROR_LIMIT")
    init_rate_limiters(app)
    app.config['IMG_STORAGE_PATH'] = os.environ.get('IMG_STORAGE_PATH')
    app.config['SESSION_DURATION'] = int(os.environ.get('SESSION_DURATION', 600)) * int(1e9)
    app.config['LIMIT_RATE'] = int(os.environ.get('LIMIT_RATE', 60)) * int(1e9)
    app.config['LIMIT_RATE_NEGATIVE'] = int(os.environ.get('LIMIT_RATE_NEGATIVE', 1)) * int(1e9)
    app.config['BASE_SIMILARITY_ENDPOINT'] = os.environ.get('BASE_SIMILARITY_ENDPOINT')
    app.config['TOTAL_BEST_PICTURES'] = int(os.environ.get('TOTAL_BEST_PICTURES', 7))
    app.config['MAX_EMOTION_COUNT'] = int(os.environ.get('MAX_EMOTION_COUNT', 10))
    app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_CONTENT_LENGTH', 16 * 1000 * 1000))

    with app.app_context():
        when_ready(app)

    return app

def when_ready(app):
    init_milvus()
    _get_firebase_client()
    if app.config['MINIO_URI']:
        _client_with_initialized_bucket()
    init_models()
def on_exit(arbiter):
    close_milvus()

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv("../.env")
    app = create_app()
    app.debug = False
    app.run()