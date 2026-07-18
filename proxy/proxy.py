"""Main FastAPI proxy application."""

import json
import logging
import pathlib
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from . import config as config_module
from .config import ConfigWatcher
from .debuglogging import DebugLogger
from .health import router as health_router
from .logging import log_request, log_response, setup_logging
from .metrics import metrics
from .middleware import TimingMiddleware
from .models import ProxyConfig
from .rewrite import extract_request_info, rewrite_request
from .streaming import forward_request

logger = logging.getLogger(__name__)

# Global state
_http_client: Optional[httpx.AsyncClient] = None
_config_watcher: Optional[ConfigWatcher] = None
_config: Optional[ProxyConfig] = None
_debug_logger: Optional[DebugLogger] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup/shutdown."""
    global _http_client, _config_watcher, _config, _debug_logger

    # Startup
    logger.info("Proxy starting up")

    # Load configuration
    config_path = pathlib.Path("config.yaml")
    _config = await config_module.load_config(config_path)

    # Initialize debug logger
    _debug_logger = DebugLogger(_config.debug)

    # Create shared HTTP client
    _http_client = httpx.AsyncClient(
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

    # Start config watcher
    _config_watcher = ConfigWatcher(config_path, interval=5.0)
    await _config_watcher.start()

    logger.info(
        "Proxy started: host=%s, port=%s, local_ai_server_url=%s",
        _config.listen.host,
        _config.listen.port,
        _config.localAIServer.url,
    )

    yield

    # Shutdown
    logger.info("Proxy shutting down")

    if _config_watcher:
        await _config_watcher.stop()

    if _http_client:
        await _http_client.aclose()

    logger.info("Proxy shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Copilot Proxy",
        description="OpenAI-compatible reverse proxy for a local AI server",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Setup logging
    setup_logging("INFO")

    # Add middleware
    app.add_middleware(TimingMiddleware, metrics=metrics)

    # Register health endpoints
    app.include_router(health_router)

    # Metrics endpoint
    @app.get("/metrics")
    async def get_metrics() -> JSONResponse:
        """Return current metrics snapshot."""
        snapshot = metrics.get_snapshot()
        return JSONResponse(content=snapshot.__dict__)

    # Catch-all route for proxying
    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def proxy_request(request: Request, path: str) -> Response:
        """Catch-all route that forwards requests to the upstream local AI server."""
        global _config

        # Get current config
        if _config is None:
            _config = await config_module.get_config()

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
            rewritten_body, was_modified = rewrite_request(body, _config)
            if was_modified:
                metrics.increment_rewritten_requests()

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
            logging_config=_config.logging,
        )

        # Capture request for debug logging
        debug_conv_key = ""
        debug_exchange_num = 0
        if _debug_logger and _debug_logger.enabled and rewritten_body:
            debug_conv_key, debug_exchange_num = await _debug_logger.start_exchange(
                body=rewritten_body,
                client_ip=client_ip,
                method=request.method,
                path=f"/{path}",
                headers=dict(request.headers),
            )

        # Forward to upstream
        if _http_client is None:
            return JSONResponse(
                content={"error": "Proxy not initialized"},
                status_code=500,
            )

        response = await forward_request(
            client=_http_client,
            local_ai_server_url=_config.localAIServer.url,
            method=request.method,
            path=path,
            headers=dict(request.headers),
            body=rewritten_body if rewritten_body else None,
            raw_body=body_bytes if not body else None,
            debug_logger=_debug_logger,
            debug_conv_key=debug_conv_key,
            debug_exchange_num=debug_exchange_num,
        )

        # Log the response
        log_response(
            method=request.method,
            path=f"/{path}",
            status_code=response.status_code,
            latency_ms=0,  # Latency tracked by middleware
            logging_config=_config.logging,
        )

        return response

    return app
