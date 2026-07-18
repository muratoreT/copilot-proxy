"""Request timing and connection tracking middleware."""

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .metrics import MetricsTracker


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks request latency and active connections."""

    def __init__(self, app, metrics: MetricsTracker):
        super().__init__(app)
        self.metrics = metrics

    async def dispatch(self, request: Request, call_next) -> Response:
        # Track active connections
        self.metrics.increment_active_connections()
        self.metrics.increment_requests()

        start_time = time.monotonic()

        try:
            response = await call_next(request)

            # Calculate latency
            latency_ms = (time.monotonic() - start_time) * 1000

            # Record metrics
            self.metrics.record_latency(request.url.path, latency_ms)

            # Track errors
            if response.status_code >= 400:
                self.metrics.increment_forward_errors(response.status_code)

            return response

        finally:
            self.metrics.decrement_active_connections()
