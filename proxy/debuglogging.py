"""Raw debug logging — saves full request/response exchanges to disk per conversation."""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from .models import DebugConfig

logger = logging.getLogger(__name__)

# Headers that contain sensitive data and should be stripped before logging
SENSITIVE_HEADERS = frozenset(
    "authorization proxy-authorization cookie set-cookie x-api-key "
    "x-forwarded-for x-real-ip sec-ch-ua sec-ch-ua-platform "
    "sec-ch-ua-mobile sec-ch-ua-full-version-list sec-ch-ua-arch "
    "sec-ch-ua-model sec-ch-ua-bitness sec-ch-ua-wow64 "
    "sec-ch-ua-full-version sec-ch-ua-preference-levels sec-ch-ua-fledging ".
    split()
)


@dataclass
class ExchangeData:
    """Holds all captured data for a single request/response exchange."""

    exchange_num: int
    conversation_key: str

    # Request side
    request_timestamp: Optional[str] = None
    request_method: Optional[str] = None
    request_path: Optional[str] = None
    request_headers: Optional[Dict[str, str]] = None
    request_body: Optional[Any] = None

    # Response side
    response_timestamp: Optional[str] = None
    response_status_code: Optional[int] = None
    response_headers: Optional[Dict[str, str]] = None
    response_body: Optional[str] = None
    response_duration_ms: Optional[float] = None

    # Retry info
    retry_count: int = 0

    # Internal
    _response_start_time: Optional[float] = None


def strip_sensitive_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Remove sensitive headers from a headers dict (case-insensitive match)."""
    return {k: v for k, v in headers.items() if k.lower() not in SENSITIVE_HEADERS}


class DebugLogger:
    """
    Captures full request/response exchanges and writes them to disk
    organized by conversation folders.
    """

    def __init__(self, config: DebugConfig):
        self.config = config
        self.enabled = config.enabled
        self.log_dir = Path(config.log_dir)
        self.max_response_bytes = config.max_response_size_mb * 1024 * 1024

        # Per-conversation state
        self._counters: Dict[str, int] = {}
        self._exchanges: Dict[str, List[ExchangeData]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Debug logger enabled: log_dir=%s", self.log_dir)

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a per-conversation lock."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    @staticmethod
    def compute_conversation_key(body: Dict[str, Any], client_ip: str) -> str:
        """
        Compute a conversation key from the first user message content + client IP.
        Returns a 16-char hex string (truncated SHA-256).
        """
        first_user_content = ""
        messages = body.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    first_user_content = content
                    break
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            first_user_content += part["text"]
                    break

        raw = f"{first_user_content}|{client_ip}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _conversation_dir(self, key: str) -> Path:
        """Get the folder path for a conversation key."""
        return self.log_dir / key

    def _exchange_filename(self, num: int) -> str:
        """Generate exchange filename like exchange_001.json."""
        return f"exchange_{num:03d}.json"

    async def start_exchange(
        self,
        body: Dict[str, Any],
        client_ip: str,
        method: str,
        path: str,
        headers: Dict[str, str],
    ) -> tuple:
        """
        Record the start of a new exchange (request captured).
        Returns (conversation_key, exchange_num) for later use.
        """
        if not self.enabled:
            return ("", 0)

        conv_key = self.compute_conversation_key(body, client_ip)
        lock = self._get_lock(conv_key)

        async with lock:
            # Increment counter
            self._counters[conv_key] = self._counters.get(conv_key, 0) + 1
            exchange_num = self._counters[conv_key]

            # Ensure conversation folder exists
            conv_dir = self._conversation_dir(conv_key)
            conv_dir.mkdir(parents=True, exist_ok=True)

            # Strip sensitive headers before storing
            safe_headers = strip_sensitive_headers(headers)

            # Create exchange data
            exchange = ExchangeData(
                exchange_num=exchange_num,
                conversation_key=conv_key,
                request_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                request_method=method,
                request_path=path,
                request_headers=safe_headers,
                request_body=body,
            )

            # Track exchange
            if conv_key not in self._exchanges:
                self._exchanges[conv_key] = []
            self._exchanges[conv_key].append(exchange)

            logger.debug(
                "Debug: exchange started conv=%s num=%d", conv_key, exchange_num
            )

            return (conv_key, exchange_num)

    async def complete_exchange(
        self,
        conv_key: str,
        exchange_num: int,
        status_code: int,
        headers: Dict[str, str],
        body: str,
        duration_ms: float,
        retry_count: int = 0,
    ) -> None:
        """
        Complete an exchange with response data and write to disk.
        """
        if not self.enabled or not conv_key:
            return

        lock = self._get_lock(conv_key)

        async with lock:
            exchanges = self._exchanges.get(conv_key, [])

            # Find the matching exchange
            target = None
            for ex in exchanges:
                if ex.exchange_num == exchange_num:
                    target = ex
                    break

            if target is None:
                logger.warning(
                    "Debug: exchange %d not found for conv %s, cannot complete",
                    exchange_num,
                    conv_key,
                )
                return

            # Populate response data
            target.response_timestamp = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            target.response_status_code = status_code
            target.response_headers = strip_sensitive_headers(headers)
            target.response_body = body
            target.response_duration_ms = round(duration_ms, 2)
            target.retry_count = retry_count

            # Truncation
            body_bytes = body.encode("utf-8", errors="replace")
            if len(body_bytes) > self.max_response_bytes:
                logger.warning(
                    "Debug: response truncated for conv=%s exchange=%d "
                    "(exceeded %d MB limit)",
                    conv_key,
                    exchange_num,
                    self.config.max_response_size_mb,
                )
                target.response_body = (
                    body_bytes[:self.max_response_bytes].decode("utf-8", errors="replace")
                    + f"\n\n[TRUNCATED: response exceeded {self.config.max_response_size_mb} MB limit, "
                    f"original size: {len(body_bytes)} bytes]"
                )
            else:
                target.response_body = body

            # Write to disk
            await self._write_exchange(conv_key, target)

            # Clean up from in-memory list (keep last 5 to avoid unbounded growth)
            if len(exchanges) > 5:
                self._exchanges[conv_key] = exchanges[-5:]

    async def _write_exchange(self, conv_key: str, exchange: ExchangeData) -> None:
        """Atomically write an exchange to disk using temp file + rename."""
        conv_dir = self._conversation_dir(conv_key)
        filename = self._exchange_filename(exchange.exchange_num)
        target_path = conv_dir / filename

        data = {
            "exchange": exchange.exchange_num,
            "conversation_key": exchange.conversation_key,
            "retry_count": exchange.retry_count,
            "request": {
                "timestamp": exchange.request_timestamp,
                "method": exchange.request_method,
                "path": exchange.request_path,
                "headers": exchange.request_headers,
                "body": exchange.request_body,
            },
            "response": {
                "timestamp": exchange.response_timestamp,
                "status_code": exchange.response_status_code,
                "headers": exchange.response_headers,
                "body": exchange.response_body,
                "duration_ms": exchange.response_duration_ms,
            },
        }

        try:
            # Write to temp file in same directory, then rename (atomic on same filesystem)
            fd, tmp_path = tempfile.mkstemp(dir=conv_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, target_path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            logger.debug(
                "Debug: wrote exchange file conv=%s num=%d path=%s",
                conv_key,
                exchange.exchange_num,
                target_path,
            )
        except Exception as e:
            logger.error(
                "Debug: failed to write exchange file conv=%s num=%d: %s",
                conv_key,
                exchange.exchange_num,
                e,
            )

    def get_response_start_time(self, conv_key: str, exchange_num: int) -> Optional[float]:
        """Get the monotonic time when response started (for duration calculation)."""
        exchanges = self._exchanges.get(conv_key, [])
        for ex in exchanges:
            if ex.exchange_num == exchange_num:
                return ex._response_start_time
        return None

    def set_response_start_time(self, conv_key: str, exchange_num: int) -> float:
        """Record when the response started arriving."""
        now = time.monotonic()
        exchanges = self._exchanges.get(conv_key, [])
        for ex in exchanges:
            if ex.exchange_num == exchange_num:
                ex._response_start_time = now
                break
        return now

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_sync(self, cutoff_ts: float) -> tuple:
        """Synchronous helper: scan log_dir and delete old conversation folders.

        Returns (conversations_deleted, total_freed_bytes).
        """
        import shutil

        deleted = 0
        freed = 0

        if not self.log_dir.exists():
            return (0, 0)

        for entry in self.log_dir.iterdir():
            if not entry.is_dir():
                continue

            # Find the newest file's mtime inside the conversation folder
            newest = 0.0
            for child in entry.iterdir():
                try:
                    mt = child.stat().st_mtime
                except OSError:
                    continue
                if mt > newest:
                    newest = mt

            if newest == 0.0:
                # Empty folder — treat as old
                newest = 0.0

            if newest < cutoff_ts:
                try:
                    size = sum(
                        f.stat().st_size
                        for f in entry.rglob("*")
                        if f.is_file()
                    )
                    freed += size
                    shutil.rmtree(entry)
                    deleted += 1
                except OSError as e:
                    logger.warning(
                        "Debug cleanup: failed to remove %s: %s", entry, e
                    )

        return (deleted, freed)

    async def cleanup_old_logs(self) -> tuple:
        """Delete conversation folders whose newest file is older than retention_days.

        Returns (conversations_deleted, total_freed_bytes).
        """
        from datetime import timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.retention_days)
        cutoff_ts = cutoff.timestamp()

        loop = asyncio.get_running_loop()
        deleted, freed = await loop.run_in_executor(None, self._cleanup_sync, cutoff_ts)

        if deleted:
            logger.info(
                "cleanup.complete conversations_deleted=%d total_freed_mb=%.2f",
                deleted,
                freed / (1024 * 1024),
            )
        return (deleted, freed)

    async def start_cleanup_loop(self) -> None:
        """Start the periodic cleanup background task."""
        if not self.enabled:
            return

        self._cleanup_shutdown = asyncio.Event()
        interval_seconds = self.config.cleanup_interval_hours * 3600

        async def _loop():
            while not self._cleanup_shutdown.is_set():
                try:
                    await asyncio.wait_for(
                        self._cleanup_shutdown.wait(),
                        timeout=interval_seconds,
                    )
                    break  # Shutdown signal received
                except asyncio.TimeoutError:
                    # Interval elapsed — run cleanup
                    try:
                        await self.cleanup_old_logs()
                    except Exception:
                        logger.exception("Debug cleanup: unexpected error during cleanup")

        self._cleanup_task = asyncio.create_task(_loop())
        logger.info(
            "Debug cleanup loop started: retention=%d days, interval=%d hours",
            self.config.retention_days,
            self.config.cleanup_interval_hours,
        )

    async def stop_cleanup_loop(self) -> None:
        """Signal the cleanup loop to stop and wait for it to finish."""
        if hasattr(self, "_cleanup_shutdown"):
            self._cleanup_shutdown.set()

        if hasattr(self, "_cleanup_task") and self._cleanup_task is not None:
            await self._cleanup_task
            logger.info("Debug cleanup loop stopped")
