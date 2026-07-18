"""Tests for proxy/proxy.py - main FastAPI app, health, metrics, and forwarding."""

import httpx
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock

from proxy.proxy import create_app
from proxy.models import ProxyConfig, LocalAIServerConfig, ListenConfig, DefaultsConfig, RewriteConfig, LoggingConfig


@pytest.fixture
def test_config():
    return ProxyConfig(
        listen=ListenConfig(host="127.0.0.1", port=9999),
        localAIServer=LocalAIServerConfig(url="http://127.0.0.1:8000"),
        logging=LoggingConfig(requests=True, responses=False, body_preview_chars=400),
        defaults=DefaultsConfig(max_tokens=4096, temperature=0.1, top_p=1.0),
        rewrite=RewriteConfig(clamp_max_tokens=False),
        models={},
    )


@pytest.fixture
def app(test_config):
    """Create the FastAPI app for testing."""
    return create_app()


@pytest_asyncio.fixture
async def client(test_config, app):
    """Create an async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    """Health and readiness endpoints."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestReadinessEndpoint:
    """Readiness endpoint probes the local AI server."""

    @pytest.mark.asyncio
    async def test_ready_returns_ready_when_local_ai_server_up(self, client, test_config):
        """Ready endpoint returns ready when the local AI server is reachable."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__.return_value = mock_client
        mock_client_cm.__aexit__.return_value = None

        with patch("proxy.health.httpx.AsyncClient", return_value=mock_client_cm):
            with patch("proxy.proxy._config", test_config):
                response = await client.get("/ready")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_ready_returns_not_ready_when_local_ai_server_down(self, client, test_config):
        """Ready endpoint returns not_ready when the local AI server is unreachable."""
        with patch("proxy.health.httpx.AsyncClient") as mock_client_class:
            mock_client_class.side_effect = httpx.ConnectError("Connection refused")
            with patch("proxy.proxy._config", test_config):
                response = await client.get("/ready")
                assert response.status_code == 503


class TestMetricsEndpoint:
    """Metrics endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_returns_json(self, client):
        response = await client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "active_connections" in data


class TestUnknownEndpointForwarding:
    """Unknown endpoints are forwarded to upstream."""

    @pytest.mark.asyncio
    async def test_unknown_endpoint_forwarded(self, client, test_config):
        """Unknown path is forwarded to the local AI server upstream."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=b'{"models": []}')

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.stream = MagicMock(return_value=mock_stream_cm)

        with patch("proxy.proxy._config", test_config):
            with patch("proxy.proxy._http_client", mock_http_client):
                response = await client.get("/v1/models")
                assert response.status_code == 200


class TestErrorForwarding:
    """Error responses from upstream are forwarded unchanged."""

    @pytest.mark.asyncio
    async def test_error_status_forwarded(self, client, test_config):
        """Upstream error status is preserved."""
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=b'{"error": "internal error"}')

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.stream = MagicMock(return_value=mock_stream_cm)

        with patch("proxy.proxy._config", test_config):
            with patch("proxy.proxy._http_client", mock_http_client):
                response = await client.post(
                    "/v1/chat/completions",
                    json={"model": "test", "messages": []},
                )
                assert response.status_code == 500
