"""Shared fixtures for tests."""

import pytest

from proxy.models import (
    DefaultsConfig,
    ListenConfig,
    LoggingConfig,
    ModelOverride,
    ProxyConfig,
    RewriteConfig,
    VLLMConfig,
)


@pytest.fixture
def default_config():
    """Return a default ProxyConfig for testing."""
    return ProxyConfig(
        listen=ListenConfig(host="127.0.0.1", port=9999),
        vllm=VLLMConfig(url="http://127.0.0.1:8000"),
        logging=LoggingConfig(requests=True, responses=False, body_preview_chars=400),
        defaults=DefaultsConfig(max_tokens=4096, temperature=0.1, top_p=1.0),
        rewrite=RewriteConfig(
            clamp_max_tokens=True,
            remove_reasoning=False,
            inject_thinking_budget=False,
            thinking_budget=1024,
        ),
        models={},
    )


@pytest.fixture
def config_with_model_overrides():
    """Return a ProxyConfig with model-specific overrides."""
    return ProxyConfig(
        listen=ListenConfig(host="127.0.0.1", port=9999),
        vllm=VLLMConfig(url="http://127.0.0.1:8000"),
        logging=LoggingConfig(requests=True, responses=False, body_preview_chars=400),
        defaults=DefaultsConfig(max_tokens=4096, temperature=0.1, top_p=1.0),
        rewrite=RewriteConfig(
            clamp_max_tokens=True,
            remove_reasoning=False,
            inject_thinking_budget=False,
            thinking_budget=1024,
        ),
        models={
            "model-large": ModelOverride(max_tokens=8192),
            "model-small": ModelOverride(max_tokens=2048),
        },
    )
