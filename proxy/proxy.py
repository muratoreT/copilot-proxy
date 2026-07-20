"""Main FastAPI proxy application."""

import json
import logging
import pathlib
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .config import ConfigState, ConfigWatcher, load_config
from .debuglogging import DebugLogger
from .health import router as health_router
from .struct_logging import log_request, log_response
from .metrics import MetricsTracker
from .middleware import TimingMiddleware
from .rewrite import extract_request_info, rewrite_request
from .streaming import forward_request

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup/shutdown."""
    config_path = app.state.config_path

    # Startup
    logger.info("Proxy starting up")

    # Load configuration
    config = await load_config(config_path)
    app.state.config_state = ConfigState(config, config_path)

    # Initialize debug logger
    debug_logger = DebugLogger(config.debug)

    # Start cleanup loop (if debug logging is enabled)
    await debug_logger.start_cleanup_loop()
    app.state.debug_logger = debug_logger

    # Create shared HTTP client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=5.0,
            read=300.0,  # Long timeout for streaming
            write=10.0,
            pool=5.0,
        ),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
    )
    app.state.http_client = http_client

    # Start config watcher
    config_watcher = ConfigWatcher(app.state.config_state, interval=5.0)
    await config_watcher.start()
    app.state.config_watcher = config_watcher

    logger.info(
        "Proxy started: host=%s, port=%s, local_ai_server_url=%s",
        config.listen.host,
        config.listen.port,
        config.localAIServer.url,
    )

    yield

    # Shutdown
    logger.info("Proxy shutting down")

    if hasattr(app.state, "debug_logger") and app.state.debug_logger:
        await app.state.debug_logger.stop_cleanup_loop()

    if hasattr(app.state, "config_watcher") and app.state.config_watcher:
        await app.state.config_watcher.stop()

    if hasattr(app.state, "http_client") and app.state.http_client:
        await app.state.http_client.aclose()

    logger.info("Proxy shutdown complete")


def create_app(config_path: Optional[pathlib.Path] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config_path is None:
        config_path = pathlib.Path("config.yaml")

    app = FastAPI(
        title="Copilot Proxy",
        description="OpenAI-compatible reverse proxy for a local AI server",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store config path on app.state for lifespan to use
    app.state.config_path = config_path

    # Create metrics instance and store on app.state
    metrics_tracker = MetricsTracker()
    app.state.metrics = metrics_tracker

    # Add middleware
    app.add_middleware(TimingMiddleware, metrics=metrics_tracker)

    # Register health endpoints
    app.include_router(health_router)

    # Metrics endpoint
    @app.get("/metrics")
    async def get_metrics() -> JSONResponse:
        """Return current metrics snapshot."""
        snapshot = await app.state.metrics.get_snapshot()
        return JSONResponse(content=snapshot.__dict__)

    # Catch-all route for proxying
    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def proxy_request(request: Request, path: str) -> Response:
        """Catch-all route that forwards requests to the upstream local AI server."""
        config_state = app.state.config_state
        config = await config_state.get()

        client_ip = request.client.host if request.client else "unknown"

        # Read request body
        body_bytes = await request.body()
        content_type = request.headers.get("content-type", "")

        # Determine original max_tokens for logging
        original_max_tokens = None
        effective_max_tokens = None
        model_name = ""
        prompt_chars = 0
        is_stream = False

        # Handle JSON requests
        if "application/json" in content_type and body_bytes:
            try:
                body = json.loads(body_bytes)
            except json.JSONDecodeError:
                body = {}
        else:
            body = {}

        # Extract request info for logging
        if body:
            request_info = extract_request_info(body)
            model_name = request_info.get("model", "")
            prompt_chars = request_info.get("prompt_chars", 0)
            original_max_tokens = request_info.get("max_tokens")
            is_stream = request_info.get("stream", False)

        # Apply request rewriting
        rewritten_body = None
        if body:
            rewritten_body, was_modified = rewrite_request(body, config)
            if was_modified:
                await app.state.metrics.increment_rewritten_requests()

            # Get effective max_tokens after rewriting
            effective_max_tokens = rewritten_body.get("max_tokens")

        # Log the request
        log_request(
            method=request.method,
            path=f"/{path}",
            client_ip=client_ip,
            model=model_name,
            prompt_chars=prompt_chars,
            max_tokens_original=original_max_tokens,
            max_tokens_effective=effective_max_tokens,
            stream=is_stream,
            logging_config=config.logging,
        )

        # Capture request for debug logging
        debug_logger = app.state.debug_logger
        debug_conv_key = ""
        debug_exchange_num = 0
        if debug_logger and debug_logger.enabled and rewritten_body:
            debug_conv_key, debug_exchange_num = await debug_logger.start_exchange(
                body=rewritten_body,
                client_ip=client_ip,
                method=request.method,
                path=f"/{path}",
                headers=dict(request.headers),
            )

        # Forward to upstream
        http_client = app.state.http_client
        if http_client is None:
            return JSONResponse(
                content={"error": "Proxy not initialized"},
                status_code=500,
            )

        response = await forward_request(
            client=http_client,
            local_ai_server_url=config.localAIServer.url,
            method=request.method,
            path=path,
            headers=dict(request.headers),
            body=rewritten_body if rewritten_body else None,
            raw_body=body_bytes if not body else None,
            proxy_config=config,
            debug_logger=debug_logger,
            debug_conv_key=debug_conv_key,
            debug_exchange_num=debug_exchange_num,
        )

        # Log the response
        log_response(
            method=request.method,
            path=f"/{path}",
            status_code=response.status_code,
            latency_ms=0,  # Latency tracked by middleware
            logging_config=config.logging,
        )

        return response

    return app
