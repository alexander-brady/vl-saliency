import logging
import os
from functools import cache
from typing import cast

LIB_NAME = "vl_saliency"
DEFAULT_LOG_LEVEL = "INFO"
ENV_LOG_LEVEL_KEY = "VL_SALIENCY_LOG_LEVEL"


def _env_log_level() -> str:
    """Get log level from environment variable if set."""
    return os.getenv(ENV_LOG_LEVEL_KEY, DEFAULT_LOG_LEVEL).upper()


@cache
def warning_once(logger: logging.Logger, *args, **kwargs):
    """Log a warning message only once."""
    logger.warning(*args, **kwargs)


@cache
def info_once(logger: logging.Logger, *args, **kwargs):
    """Log an info message only once."""
    logger.info(*args, **kwargs)


class Logger(logging.Logger):
    def warning_once(self, *args, **kwargs):
        """Log a warning message only once."""
        warning_once(self, *args, **kwargs)

    def info_once(self, *args, **kwargs):
        """Log an info message only once."""
        info_once(self, *args, **kwargs)


def get_logger(name: str = LIB_NAME) -> Logger:
    """Get a logger for the library with the specified name."""
    old_class = logging.getLoggerClass()
    logging.setLoggerClass(Logger)
    logger = logging.getLogger(name)
    logging.setLoggerClass(old_class)

    logger.setLevel(_env_log_level())
    return cast(Logger, logger)
