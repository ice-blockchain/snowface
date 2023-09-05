# 3rd parth dependencies
from flask import Flask
from routes import blueprint

from milvus import init_milvus, close_milvus
from deepface import DeepFace
def create_app():
    app = Flask(__name__)
    app.register_blueprint(blueprint)
    return app

def when_ready(arbiter):
    init_milvus()
    DeepFace.build_model("SFace")
def on_exit(arbiter):
    close_milvus()

if __name__ == '__main__':
    app = create_app()
    app.debug = True
    app.run()