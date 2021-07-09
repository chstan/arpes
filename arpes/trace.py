"""Provides lightweight perf and tracing tools which also provide light logging functionality."""
import functools
from dataclasses import dataclass, field
from typing import Callable
import time

__all__ = [
    "traceable",
]


@dataclass
class Trace:
    silent: bool = False
    start_time: float = field(default_factory=time.time_ns)

    def __call__(self, message):
        if self.silent:
            return

        now = time.time_ns()
        elapsed = (now - self.start_time) // 1000000  # to ms
        print(f"{elapsed} ms: {message}")


def traceable(original: Callable) -> Callable:
    """A decorator which takes a function and feeds a trace instance through its parameters.

    The call API of the returned function is that there is a `trace=` parameter which expects
    a bool (feature gate).

    Internally, this decorator turns that into a `Trace` instance and silences it if tracing is
    to be disabled (the user passed trace=False or did not pass trace= by keyword).

    Args:
        original: The function to decorate

    Returns:
        The decorated function which accepts a trace= keyword argument.
    """

    @functools.wraps(original)
    def _inner(*args, **kwargs):
        trace = kwargs.get("trace", False)

        # this allows us to pass Trace instances into function calls
        if not isinstance(trace, Trace):
            trace = Trace(silent=not trace)

        kwargs["trace"] = trace
        return original(*args, **kwargs)

    return _inner
