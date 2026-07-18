"""Health and readiness endpoints."""

import logging

import httpx
from fastapi import APIRouter, HTTPException

from . import proxy as proxy_module
from .models import ProxyConfig

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check() -> dict:
    """
    Readiness check that verifies upstream vLLM is reachable.

    Probes vLLM's /v1/models endpoint to confirm the server is alive.
    """
    config = proxy_module._config
    if config is None:
        raise HTTPException(status_code=503, detail="Config not loaded")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{config.vllm.url.rstrip('/')}/v1/models"
            )
            if response.status_code == 200:
                return {"status": "ready", "upstream": config.vllm.url}
            else:
                logger.warning(
                    "Upstream returned non-200: %d", response.status_code
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Upstream vLLM unhealthy (status {response.status_code})",
                )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to upstream vLLM server",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=503,
            detail="Upstream vLLM server timed out",
        )
