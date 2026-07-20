#!/usr/bin/env python3
"""Entry point for the copilot-proxy application."""

import asyncio
import logging
import pathlib
import sys

import uvicorn

from proxy.config import load_config
from proxy.proxy import create_app
from proxy.struct_logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    # Determine config path
    config_path = pathlib.Path("config.yaml")
    if not config_path.exists():
        logger.error("Config file not found: config.yaml")
        sys.exit(1)

    # Setup logging before creating the app
    setup_logging("INFO")

    # Load config to get listen settings
    config = asyncio.run(load_config(config_path))

    # Create the app and run with uvicorn (handles signals internally)
    app = create_app(config_path)
    uvicorn.run(
        app,
        host=config.listen.host,
        port=config.listen.port,
        log_level="info",
        access_log=False,  # We handle access logging ourselves
    )


if __name__ == "__main__":
    main()
