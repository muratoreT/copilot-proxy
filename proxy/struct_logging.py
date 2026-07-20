"""Request/response logging with structlog."""

import logging

import structlog

from .models import LoggingConfig

# Structlog logger bound to the proxy module.
log = structlog.get_logger("copilot-proxy")


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog for console output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def log_request(
    method: str,
    path: str,
    client_ip: str,
    model: str = "",
    prompt_chars: int = 0,
    max_tokens_original: int | None = None,
    max_tokens_effective: int | None = None,
    stream: bool = False,
    logging_config: LoggingConfig | None = None,
) -> None:
    """Log an incoming proxy request."""
    if logging_config and not logging_config.requests:
        return

    extra = {}
    if model:
        extra["model"] = model
    if prompt_chars:
        extra["prompt_chars"] = prompt_chars
    if max_tokens_original is not None and max_tokens_effective is not None:
        if max_tokens_original != max_tokens_effective:
            extra["max_tokens_clamped"] = (
                f"{max_tokens_original}->{max_tokens_effective}"
            )
        else:
            extra["max_tokens"] = max_tokens_effective
    extra["stream"] = stream
    extra["client"] = client_ip

    log.info("request.incoming", method=method, path=path, **extra)


def log_response(
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    logging_config: LoggingConfig | None = None,
) -> None:
    """Log an upstream response."""
    if logging_config and not logging_config.responses:
        return

    log.info(
        "response.outgoing",
        method=method,
        path=path,
        status=status_code,
        latency_ms=round(latency_ms, 2),
    )
