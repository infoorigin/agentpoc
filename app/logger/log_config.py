# app/logger/log_config.py

import logging
import structlog
from app.logger.cw_handler import cloud_handler
from pythonjsonlogger import json
import uuid

class SourceAwareJsonFormatter(json.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['logger'] = record.name
        log_record['level'] = record.levelname
        log_record['timestamp'] = self.formatTime(record, self.datefmt)
        return log_record


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = structlog.threadlocal.get_threadlocal()
        record.request_id = context.get('request_id', f'{__name__}+{uuid.uuid4()}')
        return True


def configure_logging():
    # === Stdlib formatter (used by third-party logs) ===
    stdlib_formatter = SourceAwareJsonFormatter()

    # === Structlog formatter (used by structlog logs only) ===
    structlog_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=[
            structlog.threadlocal.merge_threadlocal,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.format_exc_info,
        ]
    )

    # === Handlers ===
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(structlog_formatter)   # for structlog logs
    stream_handler.addFilter(RequestIDFilter())

    file_handler = logging.FileHandler('app.log', mode='a')
    file_handler.setFormatter(stdlib_formatter)        # for third-party logs
    file_handler.addFilter(RequestIDFilter())

    cloud_handler.setFormatter(stdlib_formatter)       # for third-party logs
    cloud_handler.addFilter(RequestIDFilter())

    logging.root.handlers = []
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(stream_handler)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(cloud_handler)

    # 🎯 Uvicorn loggers (important)
    # for uvicorn_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    #     logger = logging.getLogger(uvicorn_logger)
    #     logger.handlers = []
    #     logger.propagate = True
    #     logger.setLevel(logging.DEBUG)


    # === Structlog Configuration ===
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.threadlocal.merge_threadlocal,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  # 🔑 only wrap for structlog logs
        ],
        context_class=structlog.threadlocal.wrap_dict(dict),
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )