import os
import re
import schedule
import time
import logging
import subprocess
import sys

from common.loggingext import init_logging


logger = None
error_in_row_count = 0


def init_schedule(schedule_interval: str):
    match = re.fullmatch(r"daily/(\d{2}:\d{2})", schedule_interval)
    if match:
        schedule.every().day.at(match.group(1)).do(run_app)
        logger.info(f"Запуск будет происходить раз в сутки в {match.group(1)}")
        return

    match = re.fullmatch(r"(\d+)\s*sec", schedule_interval)
    if match:
        schedule.every(int(match.group(1))).seconds.do(run_app)
        logger.info(f"Запуск будет происходить каждые {match.group(1)} секунд")
        return

    raise ValueError(f"В переменной окружения SCHEDULE_INTERVAL задано недопустимое значение '{schedule_interval}'")


def try_parse_int(str_value: str, fallback_value: int) -> int:
    try:
        return int(str_value)
    except (ValueError, TypeError):
        return fallback_value


def run_app():
    global logger
    global error_in_row_count
    try:
        logger.info("Запуск приложения")

        result = subprocess.run([sys.executable, "app.py"], capture_output=False)

        if result.returncode == 0:
            logger.info(f"Приложение успешно завершено")
            error_in_row_count = 0
        else:
            logger.error(f"Приложение завершилось с ошибкой: '{result.returncode}'")
            error_in_row_count += 1
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при запуске приложения: {e}")
        error_in_row_count += 1


if __name__ == "__main__":
    try:
        logger_level = os.getenv("LOGGER_LEVEL", "INFO")
        # schedule_interval = os.getenv("SCHEDULE_INTERVAL", "daily/22:00")
        schedule_interval = os.getenv("SCHEDULE_INTERVAL", "60 sec")
        error_in_row_threshold = try_parse_int(os.getenv("ERROR_IN_ROW_THRESHOLD", "5"), 5)

        init_logging(logger_level, "log_file_scheduler.jsonl")
        logger = logging.getLogger(__name__)

        logger.info("Инициализация расписания обновления базы знаний RAG-ассистента")

        init_schedule(schedule_interval)

        run_app()

        while True:
            schedule.run_pending()
            time.sleep(1)

            if error_in_row_threshold > 0 and error_in_row_count >= error_in_row_threshold:
                raise ValueError(f"Приложение завершилось с ошибкой более {error_in_row_count} раз подряд. Расписание остановлено.")
    except Exception as e:
        if logger:
            logger.error(f"В работе расписания произошла ошибка: {e}")
        else:
            print(f"В работе расписания произошла ошибка: {e}")
        raise
