import logging
import subprocess
import time
from multiprocessing import Process

from celery import Celery

from ypl.backend.utils.json import json_dumps

NUM_WORKERS = 2
CELERY_BROKER_URL = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND = "redis://localhost:6379/0"

_celery_app = None


def init_celery() -> Celery:
    """Create a new celery app instance with the standard configuration, or return existing one."""
    global _celery_app
    if _celery_app is not None:
        return _celery_app

    app = Celery(
        "ypl",
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND,
        include=["ypl.backend.jobs.tasks"],
    )

    app.conf.update(
        task_always_eager=False,  # Async tasks
        worker_concurrency=NUM_WORKERS,
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
    )

    _celery_app = app
    logging.info("Celery app initialized")
    return app


def _run_celery_worker() -> None:
    app = init_celery()
    app.worker_main(argv=["worker", "--loglevel=info"])


def _check_redis_running() -> bool:
    # Check if redis is installed.
    try:
        subprocess.run(["redis-cli", "--version"], check=True, capture_output=True)
    except Exception as e:
        raise RuntimeError("Redis not installed") from e

    try:
        result = subprocess.run(["redis-cli", "ping"], check=True, capture_output=True)
        return result.stdout.decode().strip() == "PONG"
    except Exception:
        return False


def start_redis(wait_seconds: int = 10) -> None:
    """Start Redis server if not already running."""
    if _check_redis_running():
        logging.info("Redis server is already running")
        return

    logging.info("Redis not running, starting...")
    try:
        subprocess.Popen(["redis-server"])
        for _ in range(wait_seconds):
            if _check_redis_running():
                logging.info("Redis server started")
                return
            time.sleep(1)
        raise RuntimeError(f"Failed to start after {wait_seconds} seconds")
    except Exception as e:
        log_dict = {
            "message": "Redis server failed to start",
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise e


def start_celery_workers() -> list[Process]:
    """Start celery workers (should only be called by the main server)"""
    if _celery_app is None:
        raise RuntimeError("Celery app not initialized. Call init_celery() first.")

    processes = [Process(target=_run_celery_worker, daemon=True) for _ in range(NUM_WORKERS)]
    for process in processes:
        process.start()

    logging.info(f"{NUM_WORKERS} celery workers started")
    return processes
