"""Health and readiness endpoints."""

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check(request: Request) -> dict:
    """
    Readiness check that verifies the upstream local AI server is reachable.

    Probes /v1/models on the upstream server to confirm it is alive.
    """
    config_state = getattr(request.app.state, "config_state", None)
    if config_state is None:
        raise HTTPException(status_code=503, detail="Config not loaded")

    config = await config_state.get()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{config.localAIServer.url.rstrip('/')}/v1/models"
            )
            if response.status_code == 200:
                return {"status": "ready", "upstream": config.localAIServer.url}
            else:
                logger.warning(
                    "Upstream returned non-200: %d", response.status_code
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Upstream local AI server unhealthy (status {response.status_code})",
                )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to upstream local AI server",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=503,
            detail="Upstream local AI server timed out",
        )
