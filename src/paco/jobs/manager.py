"""Running long work in the background of this process, one job at a time."""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

logger = logging.getLogger(__name__)


class JobManager:
    """Runs jobs one after another in a background thread, and knows which are still alive.

    A job keeps its progress and results on disk, in its own record; the manager only knows which
    jobs this process is running, so that a record a stopped server left running can be told
    apart from a live one. One job at a time: each already uses every worker process it is given.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paco-job")
        self._live: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, job_id: str, work: Callable[[], object]) -> None:
        """Queue `work` as job `job_id`: it runs after the jobs submitted before it."""
        with self._lock:
            self._live.add(job_id)
        self._executor.submit(work).add_done_callback(lambda future: self._finished(job_id, future))

    def is_live(self, job_id: str) -> bool:
        """Whether job `job_id` is queued or running in this process."""
        with self._lock:
            return job_id in self._live

    def _finished(self, job_id: str, future: Future[object]) -> None:
        with self._lock:
            self._live.discard(job_id)
        # A job records its own failures; anything else is a bug, for the server's terminal.
        if (error := future.exception()) is not None:
            logger.error("Job %s crashed", job_id, exc_info=error)
