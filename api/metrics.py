from prometheus_client import make_asgi_app, multiprocess, CollectorRegistry, generate_latest, Counter, Histogram, REGISTRY, Summary,GC_COLLECTOR,PLATFORM_COLLECTOR,PROCESS_COLLECTOR
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
_session_length = None
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
def _update_frames(model, emotion, scores_by_frame, averages):
    for i in range(len(scores_by_frame)):
        for actual_emotion in model.class_to_idx:
            idx = model.class_to_idx.get(actual_emotion)
            _frames_scores_with_emotion.labels(expected_emotion=emotion, actual_emotion = actual_emotion,frame = i).observe(scores_by_frame[i][idx])
    for actual_emotion in averages:
        _emotion_avg_score.labels(expected_emotion=emotion, actual_emotion = actual_emotion).observe(averages[actual_emotion])