"""YAML configuration loader with hot-reload support."""

import asyncio
import copy
import logging
import pathlib
from typing import Optional

import yaml

from .models import ProxyConfig

logger = logging.getLogger(__name__)


def _load_yaml(path: pathlib.Path) -> dict:
    """Load and parse a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    return data


async def load_config(path: pathlib.Path) -> ProxyConfig:
    """Load configuration from a YAML file and validate it."""
    data = _load_yaml(path)
    config = ProxyConfig(**data)
    logger.info("Configuration loaded: %s", path)
    return config


class ConfigState:
    """Holds the current config and supports hot-reload."""

    def __init__(self, config: ProxyConfig, path: pathlib.Path):
        self._config = config
        self._path = path
        self._lock = asyncio.Lock()

    @property
    def path(self) -> pathlib.Path:
        return self._path

    async def get(self) -> ProxyConfig:
        """Get a copy of the current configuration (thread-safe)."""
        async with self._lock:
            return copy.copy(self._config)

    async def reload(self) -> Optional[ProxyConfig]:
        """Reload configuration from disk."""
        async with self._lock:
            try:
                data = _load_yaml(self._path)
            except FileNotFoundError:
                logger.warning(
                    "Config file not found during reload, keeping current config"
                )
                return None
            new_config = ProxyConfig(**data)
            self._config = new_config
        logger.info("Configuration reloaded: %s", self._path)
        return new_config


class ConfigWatcher:
    """Watches for config file changes and reloads asynchronously."""

    def __init__(
        self,
        config_state: ConfigState,
        interval: float = 5.0,
    ):
        self._config_state = config_state
        self.path = config_state.path
        self.interval = interval
        self._last_mtime: float = 0.0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start watching the config file for changes."""
        self._running = True
        self._last_mtime = self.path.stat().st_mtime
        self._task = asyncio.create_task(self._watch_loop())
        logger.info(
            "Config watcher started: path=%s, interval=%s", self.path, self.interval
        )

    async def stop(self):
        """Stop the config watcher."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Config watcher stopped")

    async def _watch_loop(self):
        """Periodically check for config file changes."""
        while self._running:
            try:
                current_mtime = self.path.stat().st_mtime
                if current_mtime != self._last_mtime:
                    self._last_mtime = current_mtime
                    logger.info("Config file changed, reloading")
                    await self._config_state.reload()
            except FileNotFoundError:
                logger.warning(
                    "Config file not found during watch, skipping reload"
                )
            except OSError as e:
                logger.error("Error checking config file: %s", e)
            await asyncio.sleep(self.interval)