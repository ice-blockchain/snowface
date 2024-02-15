import logging

from prometheus_client import multiprocess
import time, os
import metrics
from service import emotions_cleanup, _default_session_duration, stop_wrongfully_disabled_users_worker
from apscheduler.schedulers.background import BackgroundScheduler
from users import clean_wrongfully_disabled_users_workers
def worker_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)
    stop_wrongfully_disabled_users_worker()

def on_starting(server):
    clean_wrongfully_disabled_users_workers()