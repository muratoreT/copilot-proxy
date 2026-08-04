"""Streaming passthrough to upstream local AI server."""

import asyncio
import copy
import json
import logging
import re
import time
from typing import Optional

import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse

from .debuglogging import DebugLogger
from .models import ProxyConfig

logger = logging.getLogger(__name__)

# Some models (notably Qwen3 finetunes) emit control tokens as literal text
# instead of terminating. Left in the stream they render as garbage, and they
# poison the next request's history once the client echoes them back.
_RESPONSE_SPECIAL_TOKEN_RE = re.compile(
    r"<\|(?:endoftext|im_start|im_end|end_of_text|eot_id|eom_id|"
    r"start_header_id|end_header_id|tool_call|/tool_call)\|>"
)


class _SSETokenScrubber:
    """Removes literal special tokens from an SSE byte stream.

    Buffers partial lines so a token split across chunk boundaries is still
    matched. Lines without a ``<|`` marker are passed through byte-for-byte.
    """

    def __init__(self) -> None:
        self._pending = b""

    def process(self, chunk: bytes) -> bytes:
        data = self._pending + chunk
        head, sep, tail = data.rpartition(b"\n")
        if not sep:
            self._pending = data
            return b""
        self._pending = tail
        return self._scrub(head + sep)

    def flush(self) -> bytes:
        remainder = self._pending
        self._pending = b""
        return self._scrub(remainder) if remainder else b""

    @staticmethod
    def _scrub(data: bytes) -> bytes:
        if b"<|" not in data:
            return data
        # Line boundaries never split a UTF-8 sequence, so decoding is safe.
        text = data.decode("utf-8", errors="replace")
        return _RESPONSE_SPECIAL_TOKEN_RE.sub("", text).encode("utf-8")


def _extract_reasoning_from_delta(delta: dict) -> Optional[str]:
    """Extract reasoning content from a streaming delta, checking all known fields.

    vLLM / OpenAI-compatible backends may emit reasoning under different keys:
    - ``reasoning_content`` (OpenAI standard)
    - ``reasoning`` (vLLM / some backends)
    - ``content_parts[type=thinking]`` (Qwen ext:thinking format)

    Returns the first non-empty reasoning string found, or None.
    """
    # Check reasoning_content (OpenAI standard)
    rc = delta.get("reasoning_content")
    if isinstance(rc, str) and rc:
        return rc

    # Check reasoning (vLLM / some backends)
    reasoning = delta.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning

    # Check content_parts[type=thinking] (Qwen ext:thinking)
    content_parts = delta.get("content_parts", [])
    if isinstance(content_parts, list):
        for part in content_parts:
            if isinstance(part, dict) and part.get("type") == "thinking":
                pc = part.get("content", "")
                if isinstance(pc, str) and pc:
                    return pc

    return None


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


def _buffer_has_content(chunks: list[bytes]) -> bool:
    """Quick check: does the buffered SSE data contain any content delta?

    Used to decide when to stop buffering and start streaming through
    (optimistic passthrough for retry path).
    """
    if not chunks:
        return False
    text = b"".join(chunks).decode("utf-8", errors="replace")
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
                        return True
                    if isinstance(content, list) and content:
                        return True
                    if delta.get("tool_calls"):
                        return True
    return False


def _is_empty_stream_completion(chunks: list[bytes]) -> bool:
    """Detect empty assistant completion from SSE stream chunks.

    Returns True when the stream has finish_reason=stop, no text content,
    no tool calls, and completion_tokens <= 1.

    Also checks reasoning fields (reasoning_content, reasoning, content_parts)
    so reasoning-only streams are not misclassified as "empty" here —
    they are handled by _is_thinking_only_completion instead.
    """
    if not chunks:
        return False

    text = b"".join(chunks).decode("utf-8", errors="replace")
    has_content = False
    has_tool_calls = False
    has_reasoning = False
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
                    # Check reasoning fields so reasoning-only streams are not
                    # misclassified as "empty" here (they go to _is_thinking_only_completion)
                    reasoning = _extract_reasoning_from_delta(delta)
                    if reasoning is not None:
                        has_reasoning = True
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


def _append_system_hint(retried: dict, hint: str) -> None:
    """Merge a hint into the leading system message, or insert one."""
    messages = retried.get("messages")
    if not isinstance(messages, list):
        messages = []
        retried["messages"] = messages

    if (
        messages
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "system"
    ):
        system_content = messages[0].get("content")
        if isinstance(system_content, list):
            system_content.append({"type": "text", "text": hint})
        else:
            messages[0]["content"] = (
                f"{system_content}\n\n{hint}"
                if isinstance(system_content, str) and system_content
                else hint
            )
    else:
        messages.insert(0, {"role": "system", "content": hint})


def _build_retry_body(body: Optional[dict]) -> Optional[dict]:
    """Create a retried request body with a non-empty-response hint."""
    if body is None:
        return None

    retried = copy.deepcopy(body)
    _append_system_hint(
        retried,
        "After tool results, return a non-empty assistant response. "
        "Do not end with an empty completion.",
    )
    return retried


def _is_thinking_only_completion(chunks: list[bytes]) -> bool:
    """Detect streams that produced only thinking content with no text output.

    Covers both the Qwen ``content_parts[type=thinking]`` format and the
    vLLM/OpenAI ``reasoning_content`` / ``reasoning`` delta fields. Returns
    True when reasoning was produced but there is zero text content and no
    tool calls.
    """
    if not chunks:
        return False

    text = b"".join(chunks).decode("utf-8", errors="replace")
    has_text = False
    has_tool_calls = False
    has_thinking = False

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
                    # Plain string content counts as text
                    content = delta.get("content")
                    if isinstance(content, str) and content.strip():
                        has_text = True
                    # Mixed content_parts format (Qwen ext:thinking)
                    content_parts = delta.get("content_parts", [])
                    if isinstance(content_parts, list):
                        for part in content_parts:
                            if isinstance(part, dict):
                                if part.get("type") == "thinking":
                                    has_thinking = True
                                elif part.get("type") == "text":
                                    has_text = True
                    # reasoning_content / reasoning fields (vLLM, OpenAI)
                    if _extract_reasoning_from_delta(delta) is not None:
                        has_thinking = True
                    if delta.get("tool_calls"):
                        has_tool_calls = True

    return has_thinking and not has_text and not has_tool_calls


def _build_thinking_retry_body(body: Optional[dict]) -> Optional[dict]:
    """Create a retry body for thinking-only streams (halve thinking_budget)."""
    if body is None:
        return None

    retried = copy.deepcopy(body)
    current_budget = retried.get("thinking_budget")

    if current_budget is not None and isinstance(current_budget, int):
        halved = current_budget // 2
        if halved > 0:
            retried["thinking_budget"] = halved
        else:
            # Budget reached 0 — remove the field entirely
            retried.pop("thinking_budget", None)

    # Backends without thinking_budget support need a prompt-level nudge,
    # otherwise the retry is an exact repeat of the failing request.
    _append_system_hint(
        retried,
        "Keep internal reasoning short and always finish the turn with a "
        "visible assistant message or a tool call. Never write raw model "
        "control tokens as output text.",
    )

    return retried


def _parse_sse_to_reconstructed(body_bytes: bytes) -> dict:
    """Parse raw SSE byte stream into a structured response dict.

    Returns a dict with keys: content, reasoning, tool_calls,
    finish_reason, usage, metadata.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_map: dict[int, dict] = {}  # index -> tool_call dict
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    metadata: Optional[dict] = None

    text = body_bytes.decode("utf-8", errors="replace")

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

        # Capture metadata from first event
        if metadata is None:
            metadata = {
                k: v
                for k, v in event.items()
                if k in ("id", "model", "created")
            }

        # Capture usage (last one wins)
        if "usage" in event and isinstance(event["usage"], dict):
            usage = event["usage"]

        choices = event.get("choices", [])
        if not isinstance(choices, list):
            continue

        for choice in choices:
            if not isinstance(choice, dict):
                continue

            # Capture finish_reason (last one wins)
            fr = choice.get("finish_reason")
            if fr is not None:
                finish_reason = fr

            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue

            # Plain string content — preserve whitespace-only deltas
            cn = delta.get("content")
            if isinstance(cn, str):
                content_parts.append(cn)

            # Reasoning content — use centralized extractor
            reasoning = _extract_reasoning_from_delta(delta)
            if reasoning is not None:
                reasoning_parts.append(reasoning)

            # Tool calls
            tc_list = delta.get("tool_calls", [])
            if isinstance(tc_list, list):
                for tc in tc_list:
                    if not isinstance(tc, dict):
                        continue
                    idx = tc.get("index")
                    if idx is None:
                        continue
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {}
                    # id comes incrementally across chunks and should be concatenated
                    tc_id = tc.get("id")
                    if tc_id is not None and isinstance(tc_id, str):
                        existing_id = tool_calls_map[idx].get("id", "")
                        tool_calls_map[idx]["id"] = existing_id + tc_id
                    # type is set once and should not be concatenated
                    tc_type = tc.get("type")
                    if tc_type is not None:
                        tool_calls_map[idx]["type"] = tc_type
                    # name and arguments may be nested under "function" (OpenAI format)
                    # or at the top level; handle both
                    func = tc.get("function")
                    if isinstance(func, dict):
                        for key in ("name", "arguments"):
                            if key in func and func[key] is not None:
                                existing = tool_calls_map[idx].get(key)
                                if existing and isinstance(existing, str):
                                    tool_calls_map[idx][key] = existing + func[key]
                                else:
                                    tool_calls_map[idx][key] = func[key]
                    else:
                        for key in ("name", "arguments"):
                            if key in tc and tc[key] is not None:
                                existing = tool_calls_map[idx].get(key)
                                if existing and isinstance(existing, str):
                                    tool_calls_map[idx][key] = existing + tc[key]
                                else:
                                    tool_calls_map[idx][key] = tc[key]

    tool_calls = [
        tool_calls_map[i] for i in sorted(tool_calls_map)
    ]

    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": usage,
        "metadata": metadata,
    }


def _parse_non_streaming_to_reconstructed(body_str: str) -> dict:
    """Parse a non-streaming JSON response into a structured response dict.

    Same output shape as _parse_sse_to_reconstructed.
    """
    content = ""
    reasoning = ""
    tool_calls: list[dict] = []
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    metadata: Optional[dict] = None

    try:
        data = json.loads(body_str)
    except json.JSONDecodeError:
        return {
            "content": "",
            "reasoning": "",
            "tool_calls": [],
            "finish_reason": None,
            "usage": None,
            "metadata": None,
        }

    # Metadata
    metadata = {
        k: v
        for k, v in data.items()
        if k in ("id", "model", "created")
    }

    # Usage
    if "usage" in data and isinstance(data["usage"], dict):
        usage = data["usage"]

    choices = data.get("choices", [])
    if not isinstance(choices, list):
        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "usage": usage,
            "metadata": metadata,
        }

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        fr = choice.get("finish_reason")
        if fr is not None:
            finish_reason = fr

        message = choice.get("message", {})
        if not isinstance(message, dict):
            continue

        # Content — can be string or list of parts
        cn = message.get("content")
        if isinstance(cn, str):
            content = cn
        elif isinstance(cn, list):
            for part in cn:
                if isinstance(part, dict) and part.get("type") == "text":
                    txt = part.get("text", "")
                    if isinstance(txt, str):
                        content += txt

        # Reasoning — check both reasoning_content (OpenAI) and reasoning (vLLM)
        rc = message.get("reasoning_content")
        if isinstance(rc, str):
            reasoning = rc
        else:
            rc = message.get("reasoning")
            if isinstance(rc, str):
                reasoning = rc

        # Tool calls
        tc = message.get("tool_calls")
        if isinstance(tc, list):
            tool_calls = [
                {k: v for k, v in item.items()}
                for item in tc
                if isinstance(item, dict)
            ]
            break  # Only first choice has tool_calls typically

    return {
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": usage,
        "metadata": metadata,
    }


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
    filtered_headers = {}
    for k, v in headers.items():
        if k.lower() in hop_by_hop:
            continue
        # Drop Authorization headers with empty Bearer tokens
        # (httpx rejects "Bearer " as an illegal header value)
        if k.lower() == "authorization" and isinstance(v, str):
            stripped = v.strip()
            if stripped == "Bearer" or stripped == "Bearer ":
                logger.warning(
                    "Dropping empty Authorization header: %r", v
                )
                continue
        filtered_headers[k] = v

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
                    reconstructed = _parse_non_streaming_to_reconstructed(
                        body_str
                    )
                    await debug_logger.complete_exchange(
                        conv_key=debug_conv_key,
                        exchange_num=debug_exchange_num,
                        status_code=response.status_code,
                        headers=response_headers,
                        body=reconstructed,
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
        max_retries = retry_cfg.max_retries if retry_enabled else 0
        retries_left = max_retries
        active_json_body = json_body
        retry_count = 0  # Track how many retries were actually performed
        scrub_enabled = bool(
            proxy_config and proxy_config.rewrite.scrub_response_special_tokens
        )

        try:
            while True:
                buffered_chunks: list[bytes] = []
                scrubber = _SSETokenScrubber() if scrub_enabled else None
                # Track whether we've started streaming through (optimistic passthrough)
                streaming_through = False
                async with client.stream(
                    method,
                    url,
                    headers=headers,
                    json=active_json_body,
                ) as response:
                    response_headers = dict(response.headers)
                    response_status_code = response.status_code

                    # Finding E: check upstream status before committing to stream
                    if response_status_code != 200:
                        # Read the error body and yield it as a single SSE error event
                        error_body = await response.aread()
                        logger.warning(
                            "Upstream returned %d for streaming request %s: %s",
                            response_status_code,
                            path or url,
                            error_body[:500],
                        )
                        error_event = json.dumps(
                            {
                                "error": f"Upstream returned HTTP {response_status_code}",
                                "detail": error_body.decode("utf-8", errors="replace")[:1000],
                            }
                        )
                        yield f"data: {error_event}\n\n".encode()
                        yield b"data: [DONE]\n\n"
                        break

                    # Record response start for debug
                    if debug_logger and debug_logger.enabled and debug_conv_key:
                        debug_logger.set_response_start_time(
                            debug_conv_key, debug_exchange_num
                        )

                    async for chunk in response.aiter_bytes():
                        if scrubber is not None:
                            chunk = scrubber.process(chunk)
                            if not chunk:
                                continue
                        # Finding D: stream through optimistically once we see real content
                        # This avoids buffering the entire stream (which kills TTFT)
                        # while still catching empty completions for retry.
                        if retry_enabled and not streaming_through:
                            buffered_chunks.append(chunk)
                            # Check if buffer contains actual content delta
                            if _buffer_has_content(buffered_chunks):
                                # Real content detected — flush buffer and stream through
                                for bc in buffered_chunks:
                                    yield bc
                                buffered_chunks.clear()
                                streaming_through = True
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

                    if scrubber is not None:
                        tail = scrubber.flush()
                        if tail:
                            if retry_enabled and not streaming_through:
                                buffered_chunks.append(tail)
                            else:
                                yield tail
                            if (
                                debug_logger
                                and debug_logger.enabled
                                and debug_conv_key
                                and not truncated
                                and len(accumulated_body) + len(tail) <= max_bytes
                            ):
                                accumulated_body.extend(tail)

                # Empty completion retry (only if we never started streaming through)
                if retry_enabled and not streaming_through and _is_empty_stream_completion(buffered_chunks):
                    if retries_left > 0:
                        retries_left -= 1
                        retry_count += 1
                        logger.warning(
                            "Empty completion detected for %s; retrying (attempt %d)",
                            path or url,
                            retry_count,
                        )
                        active_json_body = _build_retry_body(active_json_body)
                        if retry_cfg.retry_delay_ms > 0:
                            await asyncio.sleep(retry_cfg.retry_delay_ms / 1000.0)
                        continue

                # Thinking-only retry (independent of only_after_tool_messages gate)
                thinking_retry_enabled = bool(
                    retry_cfg and retry_cfg.retry_on_thinking_only
                )
                if thinking_retry_enabled and not streaming_through and _is_thinking_only_completion(buffered_chunks):
                    if retries_left > 0:
                        retries_left -= 1
                        retry_count += 1
                        logger.warning(
                            "Thinking-only stream detected for %s; retrying with halved budget (attempt %d)",
                            path or url,
                            retry_count,
                        )
                        active_json_body = _build_thinking_retry_body(active_json_body)
                        if retry_cfg.retry_delay_ms > 0:
                            await asyncio.sleep(retry_cfg.retry_delay_ms / 1000.0)
                        continue

                # Flush any remaining buffered chunks (for short/empty responses that didn't retry)
                if buffered_chunks:
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
                reconstructed = _parse_sse_to_reconstructed(
                    bytes(accumulated_body)
                )
                if truncated:
                    reconstructed["_truncated"] = True
                    reconstructed["_truncated_msg"] = (
                        f"Response exceeded "
                        f"{debug_logger.config.max_response_size_mb} MB limit"
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
                    body=reconstructed,
                    duration_ms=duration_ms,
                    retry_count=retry_count,
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
