"""Request metrics tracking."""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MetricsSnapshot:
    """Immutable snapshot of current metrics."""

    requests_total: int
    streaming_requests: int
    avg_latency_ms: float
    active_connections: int
    rewritten_requests: int
    forward_errors: int
    latency_by_path: Dict[str, float]
    errors_by_status: Dict[int, int]


@dataclass
class PathMetrics:
    """Per-path latency tracking."""

    total_latency: float = 0.0
    count: int = 0

    @property
    def avg_latency(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_latency / self.count


class MetricsTracker:
    """Thread-safe metrics tracker for the proxy."""

    def __init__(self):
        self._lock = threading.Lock()
        self._requests_total: int = 0
        self._streaming_requests: int = 0
        self._active_connections: int = 0
        self._rewritten_requests: int = 0
        self._forward_errors: int = 0
        self._total_latency: float = 0.0
        self._path_metrics: Dict[str, PathMetrics] = {}
        self._errors_by_status: Dict[int, int] = {}

    def increment_requests(self) -> None:
        """Increment total request counter."""
        with self._lock:
            self._requests_total += 1

    def increment_streaming_requests(self) -> None:
        """Increment streaming request counter."""
        with self._lock:
            self._streaming_requests += 1

    def increment_active_connections(self) -> None:
        """Increment active connection counter."""
        with self._lock:
            self._active_connections += 1

    def decrement_active_connections(self) -> None:
        """Decrement active connection counter."""
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)

    def increment_rewritten_requests(self) -> None:
        """Increment rewritten request counter."""
        with self._lock:
            self._rewritten_requests += 1

    def increment_forward_errors(self, status_code: int = 0) -> None:
        """Increment forward error counter."""
        with self._lock:
            self._forward_errors += 1
            if status_code:
                self._errors_by_status[status_code] = (
                    self._errors_by_status.get(status_code, 0) + 1
                )

    def record_latency(self, path: str, latency_ms: float) -> None:
        """Record latency for a specific path."""
        with self._lock:
            self._total_latency += latency_ms
            if path not in self._path_metrics:
                self._path_metrics[path] = PathMetrics()
            self._path_metrics[path].total_latency += latency_ms
            self._path_metrics[path].count += 1

    def get_snapshot(self) -> MetricsSnapshot:
        """Get a point-in-time snapshot of all metrics."""
        with self._lock:
            avg_latency = 0.0
            if self._requests_total > 0:
                avg_latency = self._total_latency / self._requests_total

            latency_by_path = {
                path: pm.avg_latency
                for path, pm in self._path_metrics.items()
            }

            return MetricsSnapshot(
                requests_total=self._requests_total,
                streaming_requests=self._streaming_requests,
                avg_latency_ms=round(avg_latency, 2),
                active_connections=self._active_connections,
                rewritten_requests=self._rewritten_requests,
                forward_errors=self._forward_errors,
                latency_by_path=latency_by_path,
                errors_by_status=dict(self._errors_by_status),
            )

    def reset(self) -> None:
        """Reset all metrics to zero."""
        with self._lock:
            self._requests_total = 0
            self._streaming_requests = 0
            self._active_connections = 0
            self._rewritten_requests = 0
            self._forward_errors = 0
            self._total_latency = 0.0
            self._path_metrics.clear()
            self._errors_by_status.clear()


# Global metrics instance.
metrics = MetricsTracker()
