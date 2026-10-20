"""Structured HTTP/operation events; no request bodies, credentials or image IDs."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import time

REQUEST_ID = ContextVar('vreid_request_id', default=None)
LOGGER = logging.getLogger('vreid.http')
LOGGER.setLevel(logging.INFO)


def event(name, **fields):
    LOGGER.info(json.dumps({'event': name, 'time': datetime.now(timezone.utc).isoformat(),
                            'request_id': REQUEST_ID.get(), **fields}, ensure_ascii=False))


@contextmanager
def operation(phase, **fields):
    started = time.perf_counter()
    error = None
    try:
        yield
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        event('model.operation', phase=phase, duration_ms=round((time.perf_counter()-started)*1000, 3),
              outcome='error' if error else 'ok', error_type=error, **fields)
