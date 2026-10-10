from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler
import logging
import time
from datetime import datetime, timezone


class UTCJsonFormatter(jsonlogger.JsonFormatter):
    converter = time.gmtime

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def init_logging(logger_level: str, file_path: str = "log_file.jsonl"):
    logger = logging.getLogger()
    logger.setLevel(get_logger_level(logger_level))
    logger.handlers.clear()

    formatter = UTCJsonFormatter("%(asctime)s %(levelname)s %(process)d %(name)s %(message)s", json_ensure_ascii=False)

    # Terminal
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # File
    file_handler = RotatingFileHandler(
        file_path,
        encoding="utf-8",
        maxBytes=20 * 1024 * 1024,
        backupCount=2
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger_level(logger_level: str):
    if logger_level:
        logger_level = logger_level.upper()
        if logger_level == "DEBUG":
            return logging.DEBUG
        if logger_level == "WARN":
            return logging.WARN
        if logger_level == "ERROR":
            return logging.ERROR
        if logger_level == "CRITICAL":
            return logging.CRITICAL
    return logging.INFO
