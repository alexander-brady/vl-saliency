import logging
import sys
from datetime import datetime


class MinimalFormatter(logging.Formatter):
    def format(self, record):
        time_str = datetime.now().strftime("%H:%M:%S")
        level = record.levelname.upper()
        return f"{time_str} [{level}] {record.getMessage()}"


def get_logger(name: str = "app", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(MinimalFormatter())
        logger.addHandler(handler)

    logger.propagate = False  # Avoid double logging
    logger.setLevel(level)
    return logger
