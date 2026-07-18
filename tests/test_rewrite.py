"""Tests for proxy/rewrite.py - request body rewriting."""

import copy

from proxy.models import DefaultsConfig, ProxyConfig, RewriteConfig, LocalAIServerConfig, ListenConfig, LoggingConfig, ModelOverride
from proxy.rewrite import rewrite_request


class TestMaxTokensClamping:
    """max_tokens clamping: requested > limit → clamped."""

    def test_clamps_max_tokens_when_exceeding_default(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096),
            rewrite=RewriteConfig(clamp_max_tokens=True),
        )
        body = {"model": "test", "max_tokens": 8192, "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert modified is True
        assert result["max_tokens"] == 4096

    def test_does_not_clamp_when_under_limit(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096),
            rewrite=RewriteConfig(clamp_max_tokens=True),
        )
        body = {"model": "test", "max_tokens": 2048, "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert result["max_tokens"] == 2048

    def test_injects_max_tokens_when_absent(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096),
            rewrite=RewriteConfig(clamp_max_tokens=True),
        )
        body = {"model": "test", "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert modified is True
        assert result["max_tokens"] == 4096

    def test_no_clamping_when_disabled(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096, temperature=None, top_p=None),
            rewrite=RewriteConfig(clamp_max_tokens=False),
        )
        body = {"model": "test", "max_tokens": 8192, "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert modified is False
        assert result["max_tokens"] == 8192


class TestModelSpecificOverrides:
    """Model-specific max_tokens overrides."""

    def test_clamps_to_model_override(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096),
            rewrite=RewriteConfig(clamp_max_tokens=True),
            models={"my-model": ModelOverride(max_tokens=2048)},
        )
        body = {"model": "my-model", "max_tokens": 8192, "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert modified is True
        assert result["max_tokens"] == 2048

    def test_model_override_not_applied_to_other_models(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096),
            rewrite=RewriteConfig(clamp_max_tokens=True),
            models={"my-model": ModelOverride(max_tokens=2048)},
        )
        body = {"model": "other-model", "max_tokens": 8192, "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert modified is True
        assert result["max_tokens"] == 4096


class TestTemperatureAndTopP:
    """temperature/top_p injection when not configured by client."""

    def test_injects_temperature_when_absent(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096, temperature=0.7, top_p=0.9),
            rewrite=RewriteConfig(clamp_max_tokens=False),
        )
        body = {"model": "test", "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert result["temperature"] == 0.7

    def test_injects_top_p_when_absent(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096, temperature=0.7, top_p=0.9),
            rewrite=RewriteConfig(clamp_max_tokens=False),
        )
        body = {"model": "test", "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert result["top_p"] == 0.9

    def test_does_not_override_client_temperature(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096, temperature=0.7, top_p=0.9),
            rewrite=RewriteConfig(clamp_max_tokens=False),
        )
        body = {"model": "test", "temperature": 0.3, "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert result["temperature"] == 0.3

    def test_does_not_override_client_top_p(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096, temperature=0.7, top_p=0.9),
            rewrite=RewriteConfig(clamp_max_tokens=False),
        )
        body = {"model": "test", "top_p": 0.5, "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert result["top_p"] == 0.5


class TestThinkingBudget:
    """thinking_budget injection."""

    def test_injects_thinking_budget_when_enabled(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096),
            rewrite=RewriteConfig(
                clamp_max_tokens=False,
                inject_thinking_budget=True,
                thinking_budget=512,
            ),
        )
        body = {"model": "test", "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert modified is True
        assert result["thinking_budget"] == 512

    def test_does_not_override_existing_thinking_budget(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096),
            rewrite=RewriteConfig(
                clamp_max_tokens=False,
                inject_thinking_budget=True,
                thinking_budget=512,
            ),
        )
        body = {"model": "test", "thinking_budget": 256, "stream": True}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert result["thinking_budget"] == 256


class TestReasoningRemoval:
    """reasoning field removal."""

    def test_removes_reasoning_when_enabled(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096, temperature=None, top_p=None),
            rewrite=RewriteConfig(
                clamp_max_tokens=False,
                remove_reasoning=True,
            ),
        )
        body = {"model": "test", "stream": True, "reasoning": "some reasoning", "reasoning_effort": "high"}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert modified is True
        assert "reasoning" not in result
        assert "reasoning_effort" not in result

    def test_keeps_reasoning_when_disabled(self):
        config = ProxyConfig(
            listen=ListenConfig(),
            localAIServer=LocalAIServerConfig(),
            defaults=DefaultsConfig(max_tokens=4096, temperature=None, top_p=None),
            rewrite=RewriteConfig(
                clamp_max_tokens=False,
                remove_reasoning=False,
            ),
        )
        body = {"model": "test", "stream": True, "reasoning": "some reasoning"}
        result, modified = rewrite_request(copy.deepcopy(body), config)
        assert result["reasoning"] == "some reasoning"
