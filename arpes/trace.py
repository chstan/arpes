import functools
from dataclasses import dataclass, field
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


def traceable(original):
    @functools.wraps(original)
    def _inner(*args, **kwargs):
        trace = kwargs.get("trace", False)

        # this allows us to pass Trace instances into function calls
        if not isinstance(trace, Trace):
            trace = Trace(silent=not trace)

        kwargs["trace"] = trace
        return original(*args, **kwargs)

    return _inner
