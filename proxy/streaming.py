"""Streaming passthrough to upstream local AI server."""

import asyncio
import copy
import json
import logging
import time
from typing import Optional

import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse

from .debuglogging import DebugLogger
from .models import ProxyConfig

logger = logging.getLogger(__name__)


def _has_tool_messages(body: Optional[dict]) -> bool:
    """Return True when request history includes one or more tool messages."""
    if not body:
        return False
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(msg, dict) and msg.get("role") == "tool"
        for msg in messages
    )


def _is_empty_stream_completion(chunks: list[bytes]) -> bool:
    """Detect empty assistant completion from SSE stream chunks."""
    if not chunks:
        return False

    text = b"".join(chunks).decode("utf-8", errors="replace")
    has_content = False
    has_tool_calls = False
    finish_reason = None
    completion_tokens = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue

        payload = line[6:].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = event.get("choices", [])
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str) and content.strip():
                        has_content = True
                    elif isinstance(content, list) and content:
                        has_content = True
                    if delta.get("tool_calls"):
                        has_tool_calls = True
                fr = choice.get("finish_reason")
                if fr is not None:
                    finish_reason = fr

        usage = event.get("usage")
        if isinstance(usage, dict):
            ct = usage.get("completion_tokens")
            if isinstance(ct, int):
                completion_tokens = ct

    return (
        finish_reason == "stop"
        and not has_content
        and not has_tool_calls
        and completion_tokens is not None
        and completion_tokens <= 1
    )


def _build_retry_body(body: Optional[dict]) -> Optional[dict]:
    """Create a retried request body with a non-empty-response hint."""
    if body is None:
        return None

    retried = copy.deepcopy(body)
    messages = retried.get("messages")
    if not isinstance(messages, list):
        messages = []
        retried["messages"] = messages

    messages.append(
        {
            "role": "system",
            "content": (
                "After tool results, return a non-empty assistant response. "
                "Do not end with an empty completion."
            ),
        }
    )
    return retried


async def forward_request(
    client: httpx.AsyncClient,
    local_ai_server_url: str,
    method: str,
    path: str,
    headers: dict,
    body: Optional[dict] = None,
    raw_body: Optional[bytes] = None,
    proxy_config: Optional[ProxyConfig] = None,
    debug_logger: Optional[DebugLogger] = None,
    debug_conv_key: str = "",
    debug_exchange_num: int = 0,
) -> Response:
    """
    Forward a request to the upstream local AI server.

    Args:
        client: Shared httpx AsyncClient instance.
        local_ai_server_url: Base URL of the local AI server.
        method: HTTP method (GET, POST, etc.).
        path: Request path to forward.
        headers: Request headers to forward.
        body: JSON body (already rewritten if applicable).
        raw_body: Raw body bytes (for non-JSON requests).
        debug_logger: Optional debug logger instance.
        debug_conv_key: Conversation key for debug grouping.
        debug_exchange_num: Exchange number for debug tracking.

    Returns:
        FastAPI Response (streaming or non-streaming).
    """
    url = f"{local_ai_server_url.rstrip('/')}/{path.lstrip('/')}"

    # Filter out hop-by-hop headers that shouldn't be forwarded
    # Also remove content-length since the body may have been rewritten
    # and httpx will compute the correct value automatically
    hop_by_hop = {
        "connection",
        "content-length",
        "keep-alive",
        "proxy-authorization",
        "proxy-authenticate",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
    }
    filtered_headers = {
        k: v for k, v in headers.items() if k.lower() not in hop_by_hop
    }

    # Determine request content
    content = None
    json_body = None
    if body is not None:
        json_body = body
    elif raw_body is not None:
        content = raw_body

    # Check if this is a streaming request
    is_streaming = False
    if body and body.get("stream"):
        is_streaming = True

    try:
        if is_streaming:
            return _handle_streaming_response(
                client,
                method,
                url,
                filtered_headers,
                json_body,
                proxy_config,
                path,
                debug_logger,
                debug_conv_key,
                debug_exchange_num,
            )
        else:
            start_time = time.monotonic()
            async with client.stream(
                method,
                url,
                headers=filtered_headers,
                json=json_body,
                content=content,
            ) as response:
                # Read the full response
                response_body = await response.aread()
                duration_ms = (time.monotonic() - start_time) * 1000
                # Build response with exact headers
                response_headers = dict(response.headers)
                # Remove hop-by-hop headers from response too
                for h in hop_by_hop:
                    response_headers.pop(h, None)

                # Debug: capture non-streaming response
                if debug_logger and debug_logger.enabled and debug_conv_key:
                    body_str = response_body.decode("utf-8", errors="replace")
                    await debug_logger.complete_exchange(
                        conv_key=debug_conv_key,
                        exchange_num=debug_exchange_num,
                        status_code=response.status_code,
                        headers=response_headers,
                        body=body_str,
                        duration_ms=duration_ms,
                    )

                return Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=response_headers,
                )

    except httpx.ConnectError as e:
        logger.error("Connection error forwarding request: %s", e)
        return Response(
            content=json.dumps(
                {"error": "Unable to connect to upstream server"}
            ),
            status_code=502,
            media_type="application/json",
        )
    except httpx.TimeoutException as e:
        logger.error("Timeout forwarding request: %s", e)
        return Response(
            content=json.dumps({"error": "Upstream server timeout"}),
            status_code=504,
            media_type="application/json",
        )
    except httpx.HTTPError as e:
        logger.error("HTTP error forwarding request: %s", e)
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=502,
            media_type="application/json",
        )


def _handle_streaming_response(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict,
    json_body: Optional[dict],
    proxy_config: Optional[ProxyConfig],
    path: str = "",
    debug_logger: Optional[DebugLogger] = None,
    debug_conv_key: str = "",
    debug_exchange_num: int = 0,
) -> StreamingResponse:
    """
    Handle a streaming SSE response by passing chunks through verbatim.

    Opens the httpx stream inside the generator so it stays alive for the
    full duration of the StreamingResponse consumption.

    Returns a StreamingResponse that yields each chunk as received from upstream.
    """
    # Determine hop-by-hop headers to strip from response
    response_hop_by_hop = {
        "connection",
        "transfer-encoding",
        "content-encoding",
        "content-length",
    }

    async def generate():
        response = None
        response_headers = {}
        accumulated_body = bytearray()
        response_status_code = 0
        max_bytes = (
            debug_logger.config.max_response_size_mb * 1024 * 1024
            if debug_logger and debug_logger.enabled
            else 0
        )
        truncated = False
        start_time = time.monotonic()

        retry_cfg = proxy_config.streaming_retry if proxy_config else None
        retry_enabled = bool(retry_cfg and retry_cfg.enabled)
        if retry_enabled and retry_cfg.only_after_tool_messages:
            retry_enabled = _has_tool_messages(json_body)
        retries_left = retry_cfg.max_retries if retry_enabled else 0
        active_json_body = json_body

        try:
            while True:
                buffered_chunks: list[bytes] = []
                async with client.stream(
                    method,
                    url,
                    headers=headers,
                    json=active_json_body,
                ) as response:
                    response_headers = dict(response.headers)
                    response_status_code = response.status_code
                    # Record response start for debug
                    if debug_logger and debug_logger.enabled and debug_conv_key:
                        debug_logger.set_response_start_time(
                            debug_conv_key, debug_exchange_num
                        )

                    async for chunk in response.aiter_bytes():
                        if retry_enabled:
                            buffered_chunks.append(chunk)
                        else:
                            yield chunk

                        # Accumulate for debug logging (with size cap)
                        if (
                            debug_logger
                            and debug_logger.enabled
                            and debug_conv_key
                            and not truncated
                        ):
                            if (
                                len(accumulated_body) + len(chunk)
                                <= max_bytes
                            ):
                                accumulated_body.extend(chunk)
                            else:
                                accumulated_body.extend(
                                    chunk[: max_bytes - len(accumulated_body)]
                                )
                                truncated = True
                                logger.warning(
                                    "Debug: response truncated for conv=%s "
                                    "exchange=%d (exceeded %d MB limit)",
                                    debug_conv_key,
                                    debug_exchange_num,
                                    debug_logger.config.max_response_size_mb,
                                )

                if retry_enabled and _is_empty_stream_completion(buffered_chunks):
                    if retries_left > 0:
                        retries_left -= 1
                        logger.warning(
                            "Empty completion detected for %s; retrying once",
                            path or url,
                        )
                        active_json_body = _build_retry_body(active_json_body)
                        if retry_cfg.retry_delay_ms > 0:
                            await asyncio.sleep(retry_cfg.retry_delay_ms / 1000.0)
                        continue

                if retry_enabled:
                    for chunk in buffered_chunks:
                        yield chunk
                break
        except httpx.StreamError as e:
            logger.error("Stream error during SSE passthrough: %s", e)
            # Send error as SSE event
            error_event = json.dumps(
                {"error": "Stream interrupted", "detail": str(e)}
            )
            yield f"data: {error_event}\n\n".encode()
        finally:
            duration_ms = (time.monotonic() - start_time) * 1000

            # Debug: write accumulated response to disk
            if (
                debug_logger
                and debug_logger.enabled
                and debug_conv_key
            ):
                body_str = accumulated_body.decode("utf-8", errors="replace")
                if truncated:
                    body_str += (
                        f"\n\n[TRUNCATED: response exceeded "
                        f"{debug_logger.config.max_response_size_mb} MB limit]"
                    )

                # Strip hop-by-hop headers from response headers
                clean_headers = {
                    k: v
                    for k, v in response_headers.items()
                    if k.lower() not in response_hop_by_hop
                }

                await debug_logger.complete_exchange(
                    conv_key=debug_conv_key,
                    exchange_num=debug_exchange_num,
                    status_code=response_status_code,
                    headers=clean_headers,
                    body=body_str,
                    duration_ms=duration_ms,
                )

            # Ensure the response is closed if something went wrong
            if response is not None:
                try:
                    await response.aclose()
                except Exception:
                    pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
