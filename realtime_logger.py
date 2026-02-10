import logging
import os
from typing import Optional


class _FlushFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


_CONFIGURED = False


def setup_realtime_logging(
    *,
    log_path: str = "./uvicorn.log",
    level: int = logging.INFO,
) -> None:
    """Configure realtime file logging.

    Idempotent: safe to call multiple times.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    handler: logging.Handler = _FlushFileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicated handlers if some other code already added a file handler.
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == getattr(handler, "baseFilename", None):
            _CONFIGURED = True
            return

    root.addHandler(handler)

    # Ensure uvicorn loggers propagate to root.
    logging.getLogger("uvicorn").propagate = True
    logging.getLogger("uvicorn.error").propagate = True
    logging.getLogger("uvicorn.access").propagate = True

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    setup_realtime_logging()
    return logging.getLogger(name if name else "policyReader")
