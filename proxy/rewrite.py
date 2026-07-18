"""Request body rewriting logic."""

import copy
from typing import Any, Dict, Optional, Tuple

from .models import ProxyConfig


def _extract_model(body: Dict[str, Any]) -> str:
    """Extract the model name from a request body."""
    return body.get("model", "")


def _extract_prompt_length(body: Dict[str, Any]) -> int:
    """Estimate the prompt character count from messages or prompt field."""
    messages = body.get("messages", [])
    if messages:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total += len(part["text"])
        return total

    prompt = body.get("prompt", "")
    if isinstance(prompt, str):
        return len(prompt)
    if isinstance(prompt, list):
        return sum(len(str(p)) for p in prompt)
    return 0


def rewrite_request(
    body: Dict[str, Any], config: ProxyConfig
) -> Tuple[Dict[str, Any], bool]:
    """
    Rewrite a JSON request body according to the proxy configuration.

    Returns:
        Tuple of (rewritten_body, was_modified).
        was_modified is True if any rewriting was applied.
    """
    modified = copy.deepcopy(body)
    was_modified = False

    model_name = _extract_model(body)

    # --- max_tokens clamping ---
    if config.rewrite.clamp_max_tokens:
        effective_limit = config.get_model_max_tokens(model_name)
        original_max_tokens = body.get("max_tokens")

        if effective_limit is not None:
            if original_max_tokens is None:
                # Inject default max_tokens
                modified["max_tokens"] = effective_limit
                was_modified = True
            elif original_max_tokens > effective_limit:
                # Clamp to effective limit
                modified["max_tokens"] = effective_limit
                was_modified = True

    # --- temperature override ---
    effective_temperature = config.get_model_temperature(model_name)
    if (
        effective_temperature is not None
        and "temperature" not in body
    ):
        modified["temperature"] = effective_temperature
        was_modified = True

    # --- top_p override ---
    effective_top_p = config.get_model_top_p(model_name)
    if effective_top_p is not None and "top_p" not in body:
        modified["top_p"] = effective_top_p
        was_modified = True

    # --- thinking_budget injection ---
    if (
        config.rewrite.inject_thinking_budget
        and config.rewrite.thinking_budget is not None
    ):
        if "thinking_budget" not in body:
            modified["thinking_budget"] = config.rewrite.thinking_budget
            was_modified = True

    # --- remove_reasoning ---
    if config.rewrite.remove_reasoning:
        # Remove reasoning-related fields if present
        if "reasoning" in modified:
            del modified["reasoning"]
            was_modified = True
        if "reasoning_effort" in modified:
            del modified["reasoning_effort"]
            was_modified = True

    return modified, was_modified


def extract_request_info(
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract logging-relevant info from a request body."""
    model_name = _extract_model(body)
    prompt_chars = _extract_prompt_length(body)
    max_tokens = body.get("max_tokens")
    stream = body.get("stream", False)

    return {
        "model": model_name,
        "prompt_chars": prompt_chars,
        "max_tokens": max_tokens,
        "stream": stream,
    }
