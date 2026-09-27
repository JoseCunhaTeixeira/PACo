"""Stopping the agent's work on request: a person's Stop in PAC's chat, which stops the answer
running and everything it started. The host sets a Signal for its conversation in a context
variable, and gives it a new event for each answer; a tool call pins the event of the answer it
runs in (a later answer's cannot restart it), and a background job keeps the event of the answer
that started it (the agent follows a job to its end within that answer).

The work reads the event where it waits on its tasks: sigpipe's loops, and PACo's own
(sigpipe.masw.runs.stopping): at once, what finished kept, nothing half-written, a window redone
given back its previous attempt. Without a host that sets a Signal (the terminal, the MCP
server alone, the evaluation) nothing can stop, as before."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from sigpipe.masw.runs.stopping import Stopped

__all__ = ["SIGNAL", "Signal", "Stopped", "bound", "check", "current", "pin"]


@dataclass
class Signal:
    """A conversation's stop: the event of the answer running."""

    event: threading.Event = field(default_factory=threading.Event)

    def renew(self) -> None:
        """A new answer: its own event, unset."""
        self.event = threading.Event()

    def stop(self) -> None:
        self.event.set()


SIGNAL: contextvars.ContextVar[Signal | None] = contextvars.ContextVar("paco_signal", default=None)


def current() -> threading.Event | None:
    """The event of the answer this work runs for; None when nothing can stop it."""
    signal = SIGNAL.get()
    return signal.event if signal is not None else None


def check() -> None:
    """Raise Stopped when the answer this work runs for was stopped: between steps."""
    event = current()
    if event is not None and event.is_set():
        raise Stopped()


def pin() -> None:
    """In a tool call's thread: the event of the answer as it is now, kept for the whole call,
    whatever answer comes next (the call runs in its own copy of the context)."""
    signal = SIGNAL.get()
    if signal is not None:
        SIGNAL.set(Signal(signal.event))


def bound[T](work: Callable[[], T]) -> Callable[[], T]:
    """`work`, to run in another thread (a background job's), with this call's stop."""
    event = current()

    def run() -> T:
        if event is None:
            return work()
        token = SIGNAL.set(Signal(event))
        try:
            return work()
        finally:
            SIGNAL.reset(token)

    return run
