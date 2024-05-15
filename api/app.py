# 3rd parth dependencies
import logging
import random
import time

from flask import Flask
from routes import blueprint

from faces import init_milvus, close_milvus, _default_user, _default_password
from users import _get_client as init_redis
from auth import _get_firebase_client
from minio_uploader import _client_with_initialized_bucket
from service import init_models, _model, _model_fallback, _similarity_metric, _default_session_duration, start_wrongfully_disabled_users_worker, emotions_cleanup
from routes import init_rate_limiters
import os
from flask_executor import Executor
from deepface.commons.distance import findThreshold
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_SCHEDULER_SHUTDOWN
from apscheduler.jobstores.redis import RedisJobStore
from redis.connection import parse_url
executor = None
logging.basicConfig(level=os.environ.get('LOGGING_LEVEL','INFO'), format="%(asctime)s.%(msecs)d %(levelname)s:%(name)s:PID:%(process)d %(message)s")
scheduler = None
def create_app():
    app = Flask(__name__)
    app.register_blueprint(blueprint)
    executor = Executor(app, "snowface")
    executor.EXECUTOR_MAX_WORKERS = 256
    app.config['LOGGING_LEVEL'] = os.environ.get('LOGGING_LEVEL','INFO')
    logging.basicConfig(level=app.config['LOGGING_LEVEL'], format="%(asctime)s.%(msecs)d %(levelname)s:%(name)s:PID:%(process)d %(message)s")
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
    app.config['MILVUS_USER'] = os.environ.get('MILVUS_USER', _default_user)
    app.config['MILVUS_PASSWORD'] = os.environ.get('MILVUS_PASSWORD', _default_password)

    app.config['REDIS_URI'] = os.environ.get('REDIS_URI')
    if not app.config['REDIS_URI']:
        raise Exception("REDIS_URI was not set")

    app.config['MINIO_URI'] = os.environ.get('MINIO_URI')
    app.config['MINIO_ACCESS_KEY'] = os.environ.get('MINIO_ACCESS_KEY','minioadmin')
    app.config['MINIO_SECRET_KEY'] = os.environ.get('MINIO_SECRET_KEY','minioadmin')
    app.config['MINIO_SSL'] = os.environ.get("MINIO_SSL", 'False').lower() in ('true', '1')

    app.config['GOOGLE_APPLICATION_CREDENTIALS'] = firebase_file_content
    app.config['METADATA_UPDATED_CALLBACK_URL'] = os.environ.get("METADATA_UPDATED_CALLBACK_URL")
    app.config['METADATA_UPDATED_SECRET'] = os.environ.get("METADATA_UPDATED_SECRET","").replace("\n","")
    app.config['PRIMARY_PHOTO_ERROR_LIMIT'] = os.environ.get("PRIMARY_PHOTO_ERROR_LIMIT")
    app.config["PRIMARY_PHOTO_SFACE_DISTANCE"] = float(os.environ.get('PRIMARY_PHOTO_SFACE_DISTANCE', findThreshold(_model,_similarity_metric)))
    app.config["PRIMARY_PHOTO_ARCFACE_DISTANCE"] = float(os.environ.get('PRIMARY_PHOTO_ARCFACE_DISTANCE', findThreshold(_model_fallback,_similarity_metric)))
    app.config["PRIMARY_PHOTO_RETRIES"] = int(os.environ.get('PRIMARY_PHOTO_RETRIES', 3))
    app.config['MIGRATE_PHONE_LOGIN_CALLBACK_URL'] = os.environ.get("MIGRATE_PHONE_LOGIN_CALLBACK_URL")

    init_rate_limiters(app)

    if not app.config['METADATA_UPDATED_SECRET'] and app.config['METADATA_UPDATED_CALLBACK_URL']:
        raise Exception("METADATA_UPDATED_SECRET was not set")
    if not app.config['MIGRATE_PHONE_LOGIN_CALLBACK_URL'] and app.config['MIGRATE_PHONE_LOGIN_CALLBACK_URL']:
        raise Exception("MIGRATE_PHONE_LOGIN_CALLBACK_URL was not set")
    app.config['SESSION_DURATION'] = int(os.environ.get('SESSION_DURATION', _default_session_duration)) * int(1e9)
    app.config['LIMIT_RATE'] = int(os.environ.get('LIMIT_RATE', 60)) * int(1e9)
    app.config['LIMIT_RATE_NEGATIVE'] = int(os.environ.get('LIMIT_RATE_NEGATIVE', 1)) * int(1e9)
    app.config['SIMILARITY_SERVER'] = os.environ.get('SIMILARITY_SERVER')
    app.config["SIMILARITY_SFACE_DISTANCE"] = float(os.environ.get('SIMILARITY_SFACE_DISTANCE', findThreshold(_model,_similarity_metric)))
    app.config["SIMILARITY_ARCFACE_DISTANCE"] = float(os.environ.get('SIMILARITY_ARCFACE_DISTANCE', findThreshold(_model_fallback,_similarity_metric)))
    app.config['TOTAL_BEST_PICTURES'] = int(os.environ.get('TOTAL_BEST_PICTURES', 7))
    app.config['MAX_EMOTION_COUNT'] = int(os.environ.get('MAX_EMOTION_COUNT', 10))
    app.config['TARGET_EMOTION_COUNT'] = int(os.environ.get('TARGET_EMOTION_COUNT', 3))
    app.config['TARGET_EMOTION_SCORE'] = int(os.environ.get('TARGET_EMOTION_SCORE', 70))
    app.config['INITIAL_EMOTION_COUNT'] = int(os.environ.get('INITIAL_EMOTION_COUNT', 3))
    app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_CONTENT_LENGTH', 16 * 1000 * 1000))
    app.config['IMG_STORAGE_PATH'] = os.environ.get('IMG_STORAGE_PATH')
    if (not app.config['IMG_STORAGE_PATH']) and app.config['SIMILARITY_SERVER']:
        raise Exception("IMG_STORAGE_PATH was not set")
    if app.config['IMG_STORAGE_PATH'] and app.config['IMG_STORAGE_PATH'].endswith('/') is False:
        app.config['IMG_STORAGE_PATH'] = app.config['IMG_STORAGE_PATH'] + '/'
    app.config['METRICS_USER'] = os.environ.get('METRICS_USER','metrics')
    app.config['METRICS_PASSWORD'] = os.environ.get('METRICS_PASSWORD')
    app.config['WRONGFULLY_DISABLED_USERS_WORKERS'] = int(os.environ.get('WRONGFULLY_DISABLED_USERS_WORKERS',1))
    with app.app_context():
        when_ready(app)

    return app

def when_ready(app):
    if app.config["MINIO_URI"]:
        time.sleep(random.randint(1,int(os.environ.get("DISTRIBUTE_WORKERS_TIME",120))))
    logging.warning(f"initing milvus: PID:{os.getpid()}")
    init_milvus()
    logging.warning(f"initing redis PID:{os.getpid()}")
    init_redis()
    logging.warning(f"initing firebase PID:{os.getpid()}")
    _get_firebase_client()
    if app.config['MINIO_URI']:
        logging.warning(f"initing minio PID:{os.getpid()}")
        _client_with_initialized_bucket()
    init_models()
    if os.environ.get('MINIO_URI'):
        start_wrongfully_disabled_users_worker()
    if os.environ.get('IMG_STORAGE_PATH'):
        #start_emotion_photo_cleaner()
        pass


def on_exit(arbiter):
    close_milvus()

def start_emotion_photo_cleaner():
    global scheduler
    redisStore = RedisJobStore(**parse_url(os.environ.get("REDIS_URI")))
    if not scheduler:
        scheduler = BackgroundScheduler(jobstores={
            'default': redisStore
        })
    if scheduler and not scheduler.running:
        scheduler.start()
        if not scheduler.get_job("emotions_cleanup"):
            scheduler.add_job(
                id = "emotions_cleanup",
                func=emotions_cleanup,
                trigger="interval", seconds = 60,
                max_instances=1
            )


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv("../.env")
    app = create_app()
    app.debug = True
    app.run(port=int(os.environ.get("SNOWFACE_PORT",5000)))