#!/usr/bin/env python3
"""Entry point for the copilot-proxy application."""

import asyncio
import logging
import pathlib
import signal
import sys

import uvicorn

from proxy.config import load_config
from proxy.proxy import create_app

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    # Determine config path
    config_path = pathlib.Path("config.yaml")
    if not config_path.exists():
        logger.error("Config file not found: config.yaml")
        sys.exit(1)

    # Load config to get listen settings
    config = asyncio.run(load_config(config_path))

    # Create the app
    app = create_app()

    # Configure uvicorn
    uvicorn_config = uvicorn.Config(
        app,
        host=config.listen.host,
        port=config.listen.port,
        log_level="info",
        access_log=False,  # We handle access logging ourselves
        loop="asyncio",
    )

    server = uvicorn.Server(uvicorn_config)

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(signum, frame):
        logger.info(
            f"Received signal {signum}, initiating graceful shutdown..."
        )
        server.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.shutdown())
        loop.close()
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
