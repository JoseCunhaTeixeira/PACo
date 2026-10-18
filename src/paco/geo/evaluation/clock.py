"""A play's CPU and wall-clock time (docs/geo_plan.md §12, "Cost"). The servers run in this process,
their work in pools of worker processes, which the forkserver starts (Python 3.14's default on
Linux): not this process's children, and an inversion's chains in pools of their own inside a
worker. Their time reaches no rusage of this one, so the clock samples every process descended from
it in /proc, twice a second, and adds their CPU time to this process's own. A worker that lives less
than a sample apart is missed; elsewhere than Linux, only this process's time counts."""

import os
import resource
import threading
import time
from pathlib import Path

SAMPLE_S = 0.5  # how often the descendants are read
_PROC = Path("/proc")


class Clock:
    """The CPU time (this process's, user and system, and its descendants') and the wall-clock
    time from `start` to `stop`, in seconds."""

    def __init__(self) -> None:
        self.cpu_s = 0.0
        self.wall_s = 0.0
        self._before: dict[int, float] = {}
        self._seen: dict[int, float] = {}
        self._gone = 0.0  # the time of processes whose pid came back for another
        self._stopping = threading.Event()
        self._sampler: threading.Thread | None = None
        self._own = 0.0
        self._wall = 0.0

    def start(self) -> None:
        self._own, self._wall = _own(), time.perf_counter()
        self._before = descendants()
        self._seen = dict(self._before)
        self._stopping.clear()
        self._sampler = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._sampler.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._sampler is not None:
            self._sampler.join()
        self._sample()
        workers = self._gone + sum(
            cpu - self._before.get(pid, 0.0) for pid, cpu in self._seen.items()
        )
        self.cpu_s = round(_own() - self._own + workers, 1)
        self.wall_s = round(time.perf_counter() - self._wall, 1)

    def _sample_until_stopped(self) -> None:
        while not self._stopping.wait(SAMPLE_S):
            self._sample()

    def _sample(self) -> None:
        for pid, cpu in descendants().items():
            known = self._seen.get(pid)
            if known is not None and cpu < known:
                # Another process under a pid seen before: the first one's time kept.
                self._gone += known - self._before.pop(pid, 0.0)
            self._seen[pid] = cpu


def descendants(root: int | None = None) -> dict[int, float]:
    """The CPU time (user and system, their own, not their children's) of every live process
    descended from `root` (this one by default), by pid, from /proc; none where it is not."""
    root = os.getpid() if root is None else root
    try:
        stats = list(_PROC.glob("[0-9]*/stat"))
    except OSError:
        return {}
    ticks = os.sysconf("SC_CLK_TCK")
    parents: dict[int, int] = {}
    times: dict[int, float] = {}
    for path in stats:
        try:
            # The fields after the command's name, which may hold spaces and parentheses.
            fields = path.read_text().rsplit(")", 1)[1].split()
        except OSError, IndexError:
            continue  # gone since it was listed
        pid = int(path.parent.name)
        parents[pid] = int(fields[1])
        times[pid] = (int(fields[11]) + int(fields[12])) / ticks
    found: dict[int, float] = {}
    level = {root}
    while level:
        level = {pid for pid, parent in parents.items() if parent in level} - found.keys()
        found |= {pid: times[pid] for pid in level}
    return found


def _own() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime
