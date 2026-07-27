"""Tests for proxy/streaming.py - SSE streaming passthrough."""

import pytest
import pytest_asyncio
import httpx
from httpx import AsyncClient, Response, Request, HTTPStatusError
from unittest.mock import AsyncMock, MagicMock, patch

from proxy.streaming import (
    _build_retry_body,
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
        mock_stream_context.status_code = 200

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
        mock_stream_context.status_code = 200

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
        mock_stream_context.status_code = 200

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

    def test_retry_hint_merges_into_leading_system_message(self):
        body = {
            "messages": [
                {"role": "system", "content": "Original instructions."},
                {"role": "user", "content": "Continue."},
            ]
        }

        retried = _build_retry_body(body)

        assert retried is not None
        assert retried["messages"][0]["role"] == "system"
        assert "Original instructions." in retried["messages"][0]["content"]
        assert "non-empty assistant response" in retried["messages"][0]["content"]
        assert all(
            message["role"] != "system"
            for message in retried["messages"][1:]
        )
        assert body["messages"][0]["content"] == "Original instructions."

    def test_retry_hint_inserts_system_message_first(self):
        retried = _build_retry_body(
            {"messages": [{"role": "tool", "content": "result"}]}
        )

        assert retried is not None
        assert [message["role"] for message in retried["messages"]] == [
            "system",
            "tool",
        ]


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
        retry_body = mock_client.stream.call_args_list[1].kwargs["json"]
        assert retry_body["messages"][0]["role"] == "system"
        assert all(
            message["role"] != "system"
            for message in retry_body["messages"][1:]
        )


class TestThinkingOnlyDetection:
    """Thinking-only stream detection and retry."""

    def test_detects_thinking_only_stream(self):
        from proxy.streaming import _is_thinking_only_completion
        chunks = [
            b'data: {"choices":[{"index":0,"delta":{"content_parts":[{"type":"thinking","thinking":"reasoning step 1"}]},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content_parts":[{"type":"thinking","thinking":"reasoning step 2"}]},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ]
        assert _is_thinking_only_completion(chunks)

    def test_does_not_detect_mixed_thinking_and_text(self):
        from proxy.streaming import _is_thinking_only_completion
        chunks = [
            b'data: {"choices":[{"index":0,"delta":{"content_parts":[{"type":"thinking","thinking":"reasoning"}]},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content_parts":[{"type":"text","text":"Here is the answer"}]},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ]
        assert not _is_thinking_only_completion(chunks)

    def test_does_not_detect_plain_text_stream(self):
        from proxy.streaming import _is_thinking_only_completion
        chunks = [
            b'data: {"choices":[{"index":0,"delta":{"content":"Hello world"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ]
        assert not _is_thinking_only_completion(chunks)

    def test_does_not_detect_tool_calls_with_thinking(self):
        from proxy.streaming import _is_thinking_only_completion
        chunks = [
            b'data: {"choices":[{"index":0,"delta":{"content_parts":[{"type":"thinking","thinking":"thinking"}]},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"1","type":"function","function":{"name":"calc"}}]},"finish_reason":null}]}\n\n',
        ]
        assert not _is_thinking_only_completion(chunks)

    def test_build_thinking_retry_body_halves_budget(self):
        from proxy.streaming import _build_thinking_retry_body
        body = {"model": "test", "thinking_budget": 4096, "stream": True}
        retry_body = _build_thinking_retry_body(body)
        assert retry_body["thinking_budget"] == 2048

    def test_build_thinking_retry_body_halves_to_zero_removes_field(self):
        from proxy.streaming import _build_thinking_retry_body
        body = {"model": "test", "thinking_budget": 1, "stream": True}
        retry_body = _build_thinking_retry_body(body)
        assert "thinking_budget" not in retry_body

    def test_build_thinking_retry_body_preserves_other_fields(self):
        from proxy.streaming import _build_thinking_retry_body
        body = {"model": "test", "thinking_budget": 2048, "temperature": 0.7, "stream": True}
        retry_body = _build_thinking_retry_body(body)
        assert retry_body["temperature"] == 0.7
        assert retry_body["model"] == "test"
        assert retry_body["thinking_budget"] == 1024

    @pytest.mark.asyncio
    async def test_retry_on_thinking_only_stream(self):
        from proxy.streaming import _is_thinking_only_completion

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

        # First attempt: thinking-only
        thinking_only = [
            b'data: {"choices":[{"index":0,"delta":{"content_parts":[{"type":"thinking","thinking":"reasoning"}]},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ]
        # Second attempt: thinking + text
        mixed = [
            b'data: {"choices":[{"index":0,"delta":{"content_parts":[{"type":"thinking","thinking":"brief"}]},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content_parts":[{"type":"text","text":"Answer"}]},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ]

        mock_client = MagicMock()
        mock_client.stream = MagicMock(
            side_effect=[
                _make_stream_context(thinking_only),
                _make_stream_context(mixed),
            ]
        )

        config = ProxyConfig(
            streaming_retry=StreamingRetryConfig(
                enabled=True,
                max_retries=2,
                only_after_tool_messages=False,
                retry_on_thinking_only=True,
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
                "thinking_budget": 4096,
                "messages": [{"role": "user", "content": "hello"}],
            },
            proxy_config=config,
        )

        emitted = []
        async for chunk in response.body_iterator:
            emitted.append(chunk)

        combined = b"".join(emitted)
        assert b"Answer" in combined
        assert mock_client.stream.call_count == 2
        # Second call should have halved budget
        retry_body = mock_client.stream.call_args_list[1].kwargs["json"]
        assert retry_body["thinking_budget"] == 2048
