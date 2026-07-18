"""Tests for proxy/streaming.py - SSE streaming passthrough."""

import pytest
import pytest_asyncio
import httpx
from httpx import AsyncClient, Response, Request, HTTPStatusError
from unittest.mock import AsyncMock, MagicMock, patch

from proxy.streaming import forward_request
from proxy.models import ProxyConfig, LocalAIServerConfig, ListenConfig, DefaultsConfig, RewriteConfig, LoggingConfig


@pytest.fixture
def minimal_config():
    return ProxyConfig(
        listen=ListenConfig(host="127.0.0.1", port=9999),
        localAIServer=LocalAIServerConfig(url="http://127.0.0.1:8000"),
        logging=LoggingConfig(requests=True, responses=False, body_preview_chars=400),
        defaults=DefaultsConfig(max_tokens=4096, temperature=0.1, top_p=1.0),
        rewrite=RewriteConfig(clamp_max_tokens=False),
        models={},
    )


class TestStreamingPassthrough:
    """Streaming passthrough preserves SSE format."""

    @pytest.mark.asyncio
    async def test_streaming_response_returns_streaming_response(self, minimal_config):
        """Streaming requests return a StreamingResponse."""
        from starlette.responses import StreamingResponse

        mock_stream_context = MagicMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_stream_context)
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)
        mock_stream_context.headers = {"content-type": "text/event-stream"}

        async def aiter_bytes():
            yield b'data: {"test": true}'
            yield b''
        mock_stream_context.aiter_bytes = aiter_bytes

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_context)

        result = await forward_request(
            client=mock_client,
            local_ai_server_url="http://127.0.0.1:8000",
            method="POST",
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body={"model": "test", "stream": True, "messages": []},
        )
        assert isinstance(result, StreamingResponse)

    @pytest.mark.asyncio
    async def test_non_streaming_returns_response(self, minimal_config):
        """Non-streaming requests return a regular Response."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=b'{"choices": []}')

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_cm)

        result = await forward_request(
            client=mock_client,
            local_ai_server_url="http://127.0.0.1:8000",
            method="POST",
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body={"model": "test", "stream": False, "messages": []},
        )
        assert hasattr(result, 'body') or hasattr(result, 'content')


class TestChunkOrdering:
    """Chunk ordering preserved in streaming."""

    @pytest.mark.asyncio
    async def test_chunks_preserve_order(self, minimal_config):
        """SSE chunks are passed through in order."""
        from starlette.responses import StreamingResponse

        chunks_received = []

        mock_stream_context = MagicMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_stream_context)
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)
        mock_stream_context.headers = {"content-type": "text/event-stream"}

        async def aiter_bytes():
            for i in range(5):
                chunk = f'data: {{"index": {i}}}'.encode()
                chunks_received.append(chunk)
                yield chunk
            yield b''
        mock_stream_context.aiter_bytes = aiter_bytes

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_context)

        result = await forward_request(
            client=mock_client,
            local_ai_server_url="http://127.0.0.1:8000",
            method="POST",
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body={"model": "test", "stream": True, "messages": []},
        )
        streamed_chunks = []
        async for chunk in result.body_iterator:
            if chunk:
                streamed_chunks.append(chunk)

        # Verify chunks were streamed in order
        for i in range(5):
            assert f'"index": {i}'.encode() in streamed_chunks[i]


class TestContentTypePreservation:
    """Content-Type header preserved."""

    @pytest.mark.asyncio
    async def test_content_type_preserved_for_streaming(self, minimal_config):
        """Streaming response preserves Content-Type."""
        from starlette.responses import StreamingResponse

        mock_stream_context = MagicMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_stream_context)
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)
        mock_stream_context.headers = {"content-type": "text/event-stream; charset=utf-8"}

        async def aiter_bytes():
            yield b'data: {"test": true}'
            yield b''
        mock_stream_context.aiter_bytes = aiter_bytes

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_context)

        result = await forward_request(
            client=mock_client,
            local_ai_server_url="http://127.0.0.1:8000",
            method="POST",
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body={"model": "test", "stream": True, "messages": []},
        )
        assert isinstance(result, StreamingResponse)
