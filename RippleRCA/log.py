"""Lightweight logger wrapper for RippleRCA."""

from __future__ import annotations
import logging
import os
from datetime import datetime
from typing import Optional


def get_cur_time() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


class Logger:
    base_log_path = f"{get_cur_time()}.log"

    def __init__(
        self,
        loggername: str,
        log_path_prefix: Optional[str] = None,
        loglevel: int = logging.DEBUG,
    ) -> None:
        if log_path_prefix is None:
            filename = self.base_log_path
        else:
            filename = f"{log_path_prefix}{self.base_log_path}"
        log_dir = os.path.join(os.path.dirname(__file__), "log")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, filename)
        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(loglevel)
        if not self.logger.handlers:
            formatter = logging.Formatter(
                "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d: %(message)s"
            )
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(loglevel)
            file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(loglevel)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)
