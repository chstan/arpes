import traceback
import sys
from collections import namedtuple

__all__ = ('patched_excepthook',)


def patched_excepthook(exc_type, exc_value, exc_tb):
    enriched_tb = _add_missing_frames(exc_tb) if exc_tb else exc_tb
    traceback.print_exception(exc_type, exc_value, enriched_tb)


def _add_missing_frames(tb):
    result = fake_tb(tb.tb_frame, tb.tb_lasti, tb.tb_lineno, tb.tb_next)
    frame = tb.tb_frame.f_back
    while frame:
        result = fake_tb(frame, frame.f_lasti, frame.f_lineno, result)
        frame = frame.f_back
    return result


fake_tb = namedtuple('fake_tb', ('tb_frame', 'tb_lasti', 'tb_lineno', 'tb_next'))
