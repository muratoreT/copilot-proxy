"""YAML configuration loader with hot-reload support."""

import asyncio
import logging
import pathlib
from typing import Optional

import yaml

from .models import ProxyConfig

logger = logging.getLogger(__name__)

# Global config instance (updated atomically via lock).
_current_config: Optional[ProxyConfig] = None
_config_lock = asyncio.Lock()

# Path to the config file.
_config_path: Optional[pathlib.Path] = None


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
    global _current_config, _config_path
    data = _load_yaml(path)
    config = ProxyConfig(**data)
    async with _config_lock:
        _current_config = config
        _config_path = path
    logger.info("Configuration loaded: %s", path)
    return config


async def get_config() -> ProxyConfig:
    """Get the current configuration (thread-safe)."""
    async with _config_lock:
        if _current_config is None:
            raise RuntimeError(
                "Configuration not loaded. Call load_config() first."
            )
        return _current_config


async def reload_config() -> Optional[ProxyConfig]:
    """Reload configuration from disk if the path is set."""
    global _current_config
    async with _config_lock:
        if _config_path is None:
            logger.warning("No config path set, cannot reload")
            return None
        data = _load_yaml(_config_path)
        new_config = ProxyConfig(**data)
        _current_config = new_config
    logger.info("Configuration reloaded: %s", _config_path)
    return new_config


class ConfigWatcher:
    """Watches for config file changes and reloads asynchronously."""

    def __init__(
        self,
        path: pathlib.Path,
        interval: float = 5.0,
    ):
        self.path = path
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
                    await reload_config()
            except FileNotFoundError:
                logger.warning(
                    "Config file not found during watch, skipping reload"
                )
            except OSError as e:
                logger.error("Error checking config file: %s", e)
            await asyncio.sleep(self.interval)