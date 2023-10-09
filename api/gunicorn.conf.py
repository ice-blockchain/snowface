from prometheus_client import multiprocess
import time
import metrics
def worker_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)
def pre_request(worker, request):
    q_header = [float(h[1]) for h in request.headers if h[0].lower() == "x-queued-time"]
    queued_time = None
    if len(q_header) > 0:
        queued_time = q_header[0]
    if queued_time:
        latency = time.time() - queued_time
        metrics.register_gunicorn_latency("/".join(request.path.split("/")[:-1]), latency)