# 3rd parth dependencies
from flask import Flask
from routes import blueprint

from milvus import init_milvus, close_milvus
from auth import _get_firebase_client
from minio_uploader import _client_with_initialized_bucket
from deepface import DeepFace
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
    with app.app_context():
        is_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")
        if not is_gunicorn:
            when_ready("mock")
    return app

def when_ready(arbiter):
    init_milvus()
    _get_firebase_client()
    _client_with_initialized_bucket()
    DeepFace.build_model("SFace")
    DeepFace.build_model("VGG-Face")
def on_exit(arbiter):
    close_milvus()

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv("../.env")
    app = create_app()
    app.debug = False
    app.run()