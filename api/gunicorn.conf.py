from prometheus_client import multiprocess
import time, os
import metrics
from service import emotions_cleanup, _default_session_duration, stop_wrongfully_disabled_users_worker
from apscheduler.schedulers.background import BackgroundScheduler

def worker_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)
    stop_wrongfully_disabled_users_worker()
def pre_request(worker, request):
    q_header = [float(h[1]) for h in request.headers if h[0].lower() == "x-queued-time"]
    queued_time = None
    if len(q_header) > 0:
        queued_time = q_header[0]
    if queued_time:
        latency = time.time() - queued_time
        userIdIdx = -1
        if "/liveness/" in request.path:
            userIdIdx = -2
        metrics.register_gunicorn_latency("/".join(request.path.split("/")[:userIdIdx]), latency)

scheduler = BackgroundScheduler()
def when_ready(server):
    global scheduler
    if os.environ.get('IMG_STORAGE_PATH'):
        scheduler.add_job(
            id = "emotions_cleanup",
            func=emotions_cleanup,
            trigger="interval", seconds = 60,
        )
        scheduler.start()