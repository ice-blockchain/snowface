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
    fp = os.environ.get("AUTH_CREDENTIALS_FILE_PATH")
    if fp:
        firebase_file_content = open(fp,'r').read()
    else:
        firebase_file_content = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not firebase_file_content:
        raise Exception("failed to init firebase auth: GOOGLE_APPLICATION_CREDENTIALS not set")
    app.config['GOOGLE_APPLICATION_CREDENTIALS'] = firebase_file_content
    app.config['METADATA_UPDATED_CALLBACK_URL'] = os.environ.get("METADATA_UPDATED_CALLBACK_URL")
    app.config['PRIMARY_PHOTO_ERROR_LIMIT'] = os.environ.get("PRIMARY_PHOTO_ERROR_LIMIT")
    init_rate_limiters(app)
    app.config['IMG_STORAGE_PATH'] = os.environ.get('IMG_STORAGE_PATH')
    app.config['SESSION_DURATION'] = os.environ.get('SESSION_DURATION')
    app.config['LIMIT_RATE'] = os.environ.get('LIMIT_RATE')
    app.config['BASE_SIMILARITY_ENDPOINT'] = os.environ.get('BASE_SIMILARITY_ENDPOINT')

    with app.app_context():
        is_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")
        if not is_gunicorn:
            when_ready("mock")

    return app

def when_ready(arbiter):
    init_milvus()
    _get_firebase_client()
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