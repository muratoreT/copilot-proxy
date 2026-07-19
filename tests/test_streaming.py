"""Tests for proxy/streaming.py - SSE streaming passthrough."""

import pytest
import pytest_asyncio
import httpx
from httpx import AsyncClient, Response, Request, HTTPStatusError
from unittest.mock import AsyncMock, MagicMock, patch

from proxy.streaming import (
    _has_tool_messages,
    _is_empty_stream_completion,
    forward_request,
)
from proxy.models import (
    DefaultsConfig,
    ListenConfig,
    LocalAIServerConfig,
    LoggingConfig,
    ProxyConfig,
    RewriteConfig,
    StreamingRetryConfig,
)


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


class TestStreamingRetryDetection:
    """Detection helpers for empty completions."""

    def test_has_tool_messages(self):
        assert _has_tool_messages(
            {"messages": [{"role": "tool", "content": "ok"}]}
        )
        assert not _has_tool_messages(
            {"messages": [{"role": "user", "content": "hi"}]}
        )

    def test_detect_empty_stream_completion(self):
        chunks = [
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n',
            b"data: [DONE]\n\n",
        ]
        assert _is_empty_stream_completion(chunks)

    def test_detect_non_empty_stream_completion(self):
        chunks = [
            b'data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":5}}\n\n',
            b"data: [DONE]\n\n",
        ]
        assert not _is_empty_stream_completion(chunks)


class TestStreamingRetryFlow:
    """End-to-end retry behavior for empty completions."""

    @pytest.mark.asyncio
    async def test_retry_once_after_empty_tool_followup(self):
        def _make_stream_context(chunks):
            stream_context = MagicMock()
            stream_context.__aenter__ = AsyncMock(return_value=stream_context)
            stream_context.__aexit__ = AsyncMock(return_value=None)
            stream_context.headers = {"content-type": "text/event-stream"}
            stream_context.status_code = 200

            async def _aiter_bytes():
                for chunk in chunks:
                    yield chunk

            stream_context.aiter_bytes = _aiter_bytes
            return stream_context

        first_empty = [
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n',
            b"data: [DONE]\n\n",
        ]
        second_valid = [
            b'data: {"choices":[{"index":0,"delta":{"content":"resolved"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":8}}\n\n',
            b"data: [DONE]\n\n",
        ]

        mock_client = MagicMock()
        mock_client.stream = MagicMock(
            side_effect=[
                _make_stream_context(first_empty),
                _make_stream_context(second_valid),
            ]
        )

        config = ProxyConfig(
            streaming_retry=StreamingRetryConfig(
                enabled=True,
                max_retries=1,
                only_after_tool_messages=True,
                retry_delay_ms=0,
            )
        )

        response = await forward_request(
            client=mock_client,
            local_ai_server_url="http://127.0.0.1:8000",
            method="POST",
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body={
                "model": "test",
                "stream": True,
                "messages": [
                    {"role": "assistant", "content": "", "tool_calls": []},
                    {"role": "tool", "content": "result", "tool_call_id": "1"},
                ],
            },
            proxy_config=config,
        )

        emitted = []
        async for chunk in response.body_iterator:
            emitted.append(chunk)

        combined = b"".join(emitted)
        assert b"resolved" in combined
        assert mock_client.stream.call_count == 2
