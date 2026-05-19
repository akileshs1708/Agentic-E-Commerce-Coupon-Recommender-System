"""
Structured JSON logging configuration shared across all services.
Provides consistent log format with service name, timestamps, and request context.
"""
import logging
import json
import sys
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    Custom JSON log formatter that outputs structured logs.
    Each log line is valid JSON for easy ingestion by ELK/Loki/etc.
    """

    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self.service_name,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Attach extra fields if provided (e.g., user_id, coupon_code)
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Attach exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


def get_logger(name: str, service_name: Optional[str] = None) -> logging.Logger:
    if service_name is None:
        service_name = os.getenv("SERVICE_NAME", "unknown-service")

    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = JSONFormatter(service_name=service_name)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)

        file_path = os.path.join(log_dir, f"{service_name}.log")
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    return logger


class LogContext:
    """
    Helper to attach contextual fields to log messages.

    Usage:
        logger.info("User request", extra=LogContext(user_id="u123", coupon="SAVE10").as_extra())
    """

    def __init__(self, **kwargs):
        self.fields = kwargs

    def as_extra(self) -> Dict[str, Any]:
        return {"extra_fields": self.fields}
