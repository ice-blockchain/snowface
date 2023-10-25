from prometheus_client import make_asgi_app, multiprocess, CollectorRegistry, generate_latest, Counter, Histogram, REGISTRY, Summary,GC_COLLECTOR,PLATFORM_COLLECTOR,PROCESS_COLLECTOR, metrics
from deepface.extendedmodels.hsefer import HSEmotionRecognizer
import threading
from flask import current_app
def latest():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry=registry)

REGISTRY.unregister(GC_COLLECTOR)
REGISTRY.unregister(PLATFORM_COLLECTOR)
REGISTRY.unregister(PROCESS_COLLECTOR)

_frames_scores_with_emotion = Histogram("emotion_frames_scores", "Summary with scores distibution by frame with requested emotion", labelnames=["expected_emotion", "actual_emotion", "frame"], buckets=[i*5 for i in range(20)])
_emotion_avg_score = Histogram("emotion_avg_score", "Summary with avg scores by requested emotion", labelnames=["expected_emotion", "actual_emotion"], buckets=[i*5 for i in range(20)])
_emotions_success = Counter("emotion_success", f"Counter of successfully passed emotion liveness checks for emotion", labelnames=["expected_emotion"])
_emotions_failure = Counter("emotion_failure", f"Counter of failed emotion liveness checks for emotion", labelnames=["expected_emotion"])
_emotion_session_success = Counter("emotion_session_success","Counter of successfully passed emotion liveness sessions")
_emotion_session_failure = Counter("emotion_session_failure","Counter of failed emotion liveness sessions")

_session_length = None
_gunicorn_queue = Histogram("gunicorn_queue_request_time", "Time per request spent in queue", labelnames=["path"],buckets=(.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0,15.0, 20.0, 30.0, 40.0,50.0,59.9, metrics.INF))
REQUEST_TIME = Histogram('request_processing_seconds', 'Time spent processing request', labelnames=["path"])

_similarity_failure_distance_sface = Histogram("similarity_failure_distance_sface", "Euclidian (L2) distances of faces between primary users photo and failed secondary (for primary model: sface)", buckets=(1.05,1.1,1.15,1.2,1.25,1.3,1.5,1.75,2.0))
_similarity_failure_distance_arcface = Histogram("similarity_failure_distance_arcface", "Euclidian (L2) distances of faces between primary users photo and failed secondary (for fallback model: arcface)", buckets=(1.1,1.15,1.2,1.25,1.3,1.5,1.75,2.0))
_disabled_user_similarity_sface = Histogram("disabled_user_similarity_sface", "Euclidian (L2) distances of faces between most similar user and disabled one (for primary model: sface)", buckets=(0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.06))
_disabled_user_similarity_arcface = Histogram("disabled_user_similarity_arcface", "Euclidian (L2) distances of faces between most similar user and disabled one (for fallback model: arcface)", buckets=(0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15))
_primary_photo_passed = Counter("primary_photo_passed","Counter of successfully passed primary photo uploads (not disabled)")
def register_emotion_success(model: HSEmotionRecognizer, emotion: str, scores_by_frame: list, averages: dict):
    _emotions_success.labels(expected_emotion=emotion).inc()
    _update_frames(model, emotion,scores_by_frame, averages)
def register_emotion_failure(model: HSEmotionRecognizer, emotion: str, scores_by_frame: list, averages: dict):
    _emotions_failure.labels(expected_emotion=emotion).inc()
    _update_frames(model, emotion,scores_by_frame, averages)

def register_session_length(length:int):
    global _session_length
    if not _session_length:
        _session_length = Histogram("emotion_session_length", "Length of the session", buckets=[i for i in range(current_app.config['MAX_EMOTION_COUNT'])])
    _session_length.observe(length)
    _emotion_session_success.inc()
def register_session_failure():
    _emotion_session_failure.inc()

def register_gunicorn_latency(path, latency):
    _gunicorn_queue.labels(path).observe(latency)
def _update_frames(model, emotion, scores_by_frame, averages):
    for i in range(len(scores_by_frame)):
        for actual_emotion in model.class_to_idx:
            idx = model.class_to_idx.get(actual_emotion)
            _frames_scores_with_emotion.labels(expected_emotion=emotion, actual_emotion = actual_emotion,frame = i).observe(scores_by_frame[i][idx])
    for actual_emotion in averages:
        _emotion_avg_score.labels(expected_emotion=emotion, actual_emotion = actual_emotion).observe(averages[actual_emotion])


def register_similarity_failure(sface_distance, arcface_distance):
    _similarity_failure_distance_sface.observe(sface_distance)
    _similarity_failure_distance_arcface.observe(arcface_distance)
def register_disabled_user(sface_distance, arcface_distance):
    _disabled_user_similarity_sface.observe(sface_distance)
    if arcface_distance != -1:
        _disabled_user_similarity_arcface.observe(arcface_distance)
def register_primary_photo_uploaded():
    _primary_photo_passed.inc()