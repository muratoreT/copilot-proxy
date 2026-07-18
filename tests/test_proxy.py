"""Tests for proxy/proxy.py - main FastAPI app, health, metrics, and forwarding."""

import httpx
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock

from proxy.proxy import create_app
from proxy.models import ProxyConfig, VLLMConfig, ListenConfig, DefaultsConfig, RewriteConfig, LoggingConfig


@pytest.fixture
def test_config():
    return ProxyConfig(
        listen=ListenConfig(host="127.0.0.1", port=9999),
        vllm=VLLMConfig(url="http://127.0.0.1:8000"),
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
    """Readiness endpoint probes vLLM."""

    @pytest.mark.asyncio
    async def test_ready_returns_ready_when_vllm_up(self, client, test_config):
        """Ready endpoint returns ready when vLLM is reachable."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client_class = MagicMock()
        async def aenter(self):
            return self
        async def aexit(self, *args):
            pass
        mock_client_class.__aenter__ = aenter
        mock_client_class.__aexit__ = aexit
        mock_client_class.get = mock_get

        with patch("proxy.health.httpx.AsyncClient", return_value=mock_client_class()):
            with patch("proxy.proxy._config", test_config):
                response = await client.get("/ready")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_ready_returns_not_ready_when_vllm_down(self, client, test_config):
        """Ready endpoint returns not_ready when vLLM is unreachable."""
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
        """Unknown path is forwarded to vLLM upstream."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}

        async def aenter(self):
            return self

        async def aexit(self, *args):
            pass

        async def aread(self):
            return b'{"models": []}'

        mock_response.__aenter__ = aenter
        mock_response.__aexit__ = aexit
        mock_response.aread = aread

        mock_http_client = MagicMock()
        mock_http_client.stream = MagicMock(return_value=mock_response)

        with patch("proxy.proxy._config", test_config):
            with patch("proxy.proxy._http_client", mock_http_client):
                response = await client.get("/v1/models")
                assert response.status_code == 200


class TestErrorForwarding:
    """Error responses from upstream are forwarded unchanged."""

    @pytest.mark.asyncio
    async def test_error_status_forwarded(self, client, test_config):
        """Upstream error status is preserved."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "application/json"}

        async def aenter(self):
            return self

        async def aexit(self, *args):
            pass

        async def aread(self):
            return b'{"error": "internal error"}'

        mock_response.__aenter__ = aenter
        mock_response.__aexit__ = aexit
        mock_response.aread = aread

        mock_http_client = MagicMock()
        mock_http_client.stream = MagicMock(return_value=mock_response)

        with patch("proxy.proxy._config", test_config):
            with patch("proxy.proxy._http_client", mock_http_client):
                response = await client.post(
                    "/v1/chat/completions",
                    json={"model": "test", "messages": []},
                )
                assert response.status_code == 500
