"""Streaming passthrough to upstream vLLM server."""

import json
import logging
import time
from typing import Optional

import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse

from .models import ProxyConfig

logger = logging.getLogger(__name__)


async def forward_request(
    client: httpx.AsyncClient,
    vllm_url: str,
    method: str,
    path: str,
    headers: dict,
    body: Optional[dict] = None,
    raw_body: Optional[bytes] = None,
) -> Response:
    """
    Forward a request to the upstream vLLM server.

    Args:
        client: Shared httpx AsyncClient instance.
        vllm_url: Base URL of the vLLM server.
        method: HTTP method (GET, POST, etc.).
        path: Request path to forward.
        headers: Request headers to forward.
        body: JSON body (already rewritten if applicable).
        raw_body: Raw body bytes (for non-JSON requests).

    Returns:
        FastAPI Response (streaming or non-streaming).
    """
    url = f"{vllm_url.rstrip('/')}/{path.lstrip('/')}"

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
                client, method, url, filtered_headers, json_body, path
            )
        else:
            async with client.stream(
                method,
                url,
                headers=filtered_headers,
                json=json_body,
                content=content,
            ) as response:
                # Read the full response
                response_body = await response.aread()
                # Build response with exact headers
                response_headers = dict(response.headers)
                # Remove hop-by-hop headers from response too
                for h in hop_by_hop:
                    response_headers.pop(h, None)

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
    path: str = "",
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
        try:
            async with client.stream(
                method, url, headers=headers, json=json_body
            ) as response:
                # Stream each chunk verbatim
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.StreamError as e:
            logger.error("Stream error during SSE passthrough: %s", e)
            # Send error as SSE event
            error_event = json.dumps(
                {"error": "Stream interrupted", "detail": str(e)}
            )
            yield f"data: {error_event}\n\n".encode()
        finally:
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
