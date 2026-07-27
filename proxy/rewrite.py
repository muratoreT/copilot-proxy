"""Request body rewriting logic."""

import copy
import re
from typing import Any, Dict, Optional, Tuple

from .models import ProxyConfig

# Patterns for common special tokens that can leak into user content
# and be misinterpreted by the tokenizer as control tokens.
# We replace them with a visually identical but token-safe version.
_SPECIAL_TOKEN_PATTERNS = [
    # Qwen-style: <|endoftext|>, <|im_start|>, <|im_end|>, etc.
    (re.compile(r"<\|endoftext\|>"), "<\u200b|endoftext\u200b|>"),
    (re.compile(r"<\|im_start\|>"), "<\u200b|im_start\u200b|>"),
    (re.compile(r"<\|im_end\|>"), "<\u200b|im_end\u200b|>"),
    # Generic <|...|> patterns (catch-all for unknown control tokens)
    (re.compile(r"<\|[^|]+\|>"), lambda m: m.group(0).replace("|", "\u200b|")),
    # Qwen3 thinking tags: <think> and</think>
    (re.compile(r"<\/?thinking>"), lambda m: m.group(0).replace("<", "<\u200b")),
]


def _sanitize_special_tokens(text: str) -> str:
    """Replace literal special-token strings with safe equivalents.

    Prevents user content (tool output, reviewed code, etc.) from containing
    raw special tokens that vLLM's tokenizer could interpret as control tokens
    (EOS, chat delimiters, etc.), which can cause premature stop or malformed
    chat templates.
    """
    if not text:
        return text
    for pattern, repl in _SPECIAL_TOKEN_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def _sanitize_message_content(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize all string content in a message."""
    if not isinstance(msg, dict):
        return msg
    result = dict(msg)
    content = msg.get("content")
    if isinstance(content, str):
        result["content"] = _sanitize_special_tokens(content)
    elif isinstance(content, list):
        sanitized_parts = []
        for part in content:
            if isinstance(part, dict):
                sp = dict(part)
                if "text" in sp and isinstance(sp["text"], str):
                    sp["text"] = _sanitize_special_tokens(sp["text"])
                sanitized_parts.append(sp)
            else:
                sanitized_parts.append(part)
        result["content"] = sanitized_parts
    # Sanitize tool call arguments (they may contain special tokens from code)
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        sanitized_tc = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                stc = dict(tc)
                func = tc.get("function")
                if isinstance(func, dict):
                    sfunc = dict(func)
                    args = func.get("arguments", "")
                    if isinstance(args, str):
                        sfunc["arguments"] = _sanitize_special_tokens(args)
                    stc["function"] = sfunc
                sanitized_tc.append(stc)
            else:
                sanitized_tc.append(tc)
        result["tool_calls"] = sanitized_tc
    return result


def _sanitize_messages(messages: list) -> list:
    """Sanitize all messages in a conversation."""
    return [_sanitize_message_content(msg) for msg in messages]


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


def _extract_tool_vision_parts(
    content: Any,
) -> Tuple[list[str], list[Dict[str, Any]]]:
    """Extract text and valid OpenAI image parts from tool message content."""
    if not isinstance(content, list):
        return [], []

    text_parts = []
    image_parts = []
    for part in content:
        if not isinstance(part, dict):
            continue

        if part.get("type") == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
            continue

        image_url = part.get("image_url")
        if part.get("type") != "image_url" or not isinstance(image_url, dict):
            continue

        url = image_url.get("url")
        if not isinstance(url, str) or not url:
            continue

        normalized_image_url = {"url": url}
        if image_url.get("detail") in {"auto", "low", "high"}:
            normalized_image_url["detail"] = image_url["detail"]
        image_parts.append(
            {"type": "image_url", "image_url": normalized_image_url}
        )

    return text_parts, image_parts


def _normalize_tool_vision_messages(
    messages: Any,
) -> Tuple[Any, bool]:
    """Move tool-result images to user messages accepted by LM Studio."""
    if not isinstance(messages, list):
        return messages, False

    normalized_messages = []
    pending_vision_parts = []
    was_modified = False

    def flush_pending_vision() -> None:
        if not pending_vision_parts:
            return
        normalized_messages.append(
            {"role": "user", "content": list(pending_vision_parts)}
        )
        pending_vision_parts.clear()

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            flush_pending_vision()
            normalized_messages.append(message)
            continue

        text_parts, image_parts = _extract_tool_vision_parts(
            message.get("content")
        )
        if not image_parts:
            normalized_messages.append(message)
            continue

        normalized_message = copy.deepcopy(message)
        normalized_message["content"] = (
            "\n".join(text_parts)
            if text_parts
            else "Tool completed successfully; visual output follows."
        )
        normalized_messages.append(normalized_message)

        tool_call_id = message.get("tool_call_id")
        tool_label = (
            f"tool call {tool_call_id}"
            if isinstance(tool_call_id, str) and tool_call_id
            else "the preceding tool call"
        )
        pending_vision_parts.append(
            {
                "type": "text",
                "text": f"Visual output returned by {tool_label}.",
            }
        )
        pending_vision_parts.extend(image_parts)
        was_modified = True

    flush_pending_vision()
    return normalized_messages, was_modified


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

    # --- special token sanitization (opt-out via config) ---
    if (
        config.rewrite.sanitize_special_tokens
        and "messages" in modified
        and isinstance(modified["messages"], list)
    ):
        sanitized = _sanitize_messages(modified["messages"])
        if sanitized != modified["messages"]:
            modified["messages"] = sanitized
            was_modified = True

    # --- tool-result vision normalization ---
    if config.rewrite.normalize_tool_vision and "messages" in modified:
        normalized_messages, messages_modified = _normalize_tool_vision_messages(
            modified["messages"]
        )
        if messages_modified:
            modified["messages"] = normalized_messages
            was_modified = True

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

    # --- thinking_budget clamping + injection ---
    max_budget = config.rewrite.max_thinking_budget
    if max_budget is not None and "thinking_budget" in body:
        if body["thinking_budget"] > max_budget:
            modified["thinking_budget"] = max_budget
            was_modified = True

    if (
        config.rewrite.inject_thinking_budget
        and config.rewrite.thinking_budget is not None
    ):
        if "thinking_budget" not in body:
            # Inject the configured budget, capped by max if set
            effective = (
                min(config.rewrite.thinking_budget, max_budget)
                if max_budget is not None
                else config.rewrite.thinking_budget
            )
            modified["thinking_budget"] = effective
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
