"""Tests for proxy/config.py and proxy/models.py - YAML config loading and validation."""

import asyncio
import tempfile
import os
import pathlib

import pytest

from proxy.models import ProxyConfig, ModelOverride
from proxy.config import load_config


class TestYAMLLoading:
    """YAML loading and validation."""

    @pytest.mark.asyncio
    async def test_load_valid_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = """
listen:
  host: 0.0.0.0
  port: 8081
localAIServer:
  url: http://127.0.0.1:8000
"""
            f.write(yaml_content)
            f.flush()
            config = await load_config(pathlib.Path(f.name))
            assert config is not None
            assert config.listen.host == "0.0.0.0"
            assert config.listen.port == 8081
            assert config.localAIServer.url == "http://127.0.0.1:8000"
        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_load_config_with_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = """
listen:
  host: 127.0.0.1
  port: 9090
localAIServer:
  url: http://localhost:8000
defaults:
  max_tokens: 2048
  temperature: 0.5
  top_p: 0.8
streaming_retry:
    enabled: true
    max_retries: 1
    only_after_tool_messages: true
    retry_delay_ms: 10
"""
            f.write(yaml_content)
            f.flush()
            config = await load_config(pathlib.Path(f.name))
            assert config.defaults.max_tokens == 2048
            assert config.defaults.temperature == 0.5
            assert config.defaults.top_p == 0.8
            assert config.streaming_retry.enabled is True
            assert config.streaming_retry.max_retries == 1
        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_invalid_port_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = """
listen:
  host: 127.0.0.1
  port: 99999
localAIServer:
  url: http://127.0.0.1:8000
"""
            f.write(yaml_content)
            f.flush()
            try:
                await load_config(pathlib.Path(f.name))
                assert False, "Expected validation error"
            except Exception:
                pass  # Expected
        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_missing_required_field(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = """
listen:
  host: 127.0.0.1
"""
            f.write(yaml_content)
            f.flush()
            try:
                await load_config(pathlib.Path(f.name))
                assert False, "Expected validation error"
            except Exception:
                pass  # Expected
        os.unlink(f.name)


class TestModelOverrideResolution:
    """Model override resolution."""

    def test_get_model_max_tokens_with_override(self):
        config = ProxyConfig(
            models={
                "my-model": ModelOverride(max_tokens=8192),
            },
        )
        assert config.get_model_max_tokens("my-model") == 8192

    def test_get_model_max_tokens_without_override(self):
        config = ProxyConfig(
            models={
                "other-model": ModelOverride(max_tokens=8192),
            },
        )
        # Falls back to defaults.max_tokens (default is 4096)
        assert config.get_model_max_tokens("my-model") == 4096

    def test_get_model_temperature_with_override(self):
        config = ProxyConfig(
            models={
                "my-model": ModelOverride(temperature=0.5),
            },
        )
        assert config.get_model_temperature("my-model") == 0.5

    def test_get_model_temperature_without_override(self):
        config = ProxyConfig(
            models={
                "other-model": ModelOverride(temperature=0.5),
            },
        )
        # Falls back to defaults.temperature (default is 0.1)
        assert config.get_model_temperature("my-model") == 0.1


class TestStreamingRetryConfig:
    """Streaming retry config validation including hybrid mode fields."""

    def test_streaming_retry_defaults(self):
        from proxy.models import StreamingRetryConfig
        cfg = StreamingRetryConfig()
        assert cfg.enabled is False
        assert cfg.streaming_mode == "hybrid"
        assert cfg.empty_detection_timeout_ms == 1000

    def test_streaming_retry_hybrid_mode(self):
        from proxy.models import StreamingRetryConfig
        cfg = StreamingRetryConfig(enabled=True, streaming_mode="hybrid", empty_detection_timeout_ms=2000)
        assert cfg.streaming_mode == "hybrid"
        assert cfg.empty_detection_timeout_ms == 2000

    def test_streaming_retry_buffered_mode(self):
        from proxy.models import StreamingRetryConfig
        cfg = StreamingRetryConfig(enabled=True, streaming_mode="buffered")
        assert cfg.streaming_mode == "buffered"

    def test_streaming_retry_invalid_mode_rejected(self):
        from proxy.models import StreamingRetryConfig
        with pytest.raises(Exception):
            StreamingRetryConfig(enabled=True, streaming_mode="invalid")

    def test_empty_detection_timeout_ms_below_min_rejected(self):
        from proxy.models import StreamingRetryConfig
        with pytest.raises(Exception):
            StreamingRetryConfig(empty_detection_timeout_ms=100)

    def test_empty_detection_timeout_ms_above_max_rejected(self):
        from proxy.models import StreamingRetryConfig
        with pytest.raises(Exception):
            StreamingRetryConfig(empty_detection_timeout_ms=20000)

    @pytest.mark.asyncio
    async def test_load_hybrid_config_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = """
listen:
  host: 127.0.0.1
  port: 8081
localAIServer:
  url: http://127.0.0.1:8000
streaming_retry:
  enabled: true
  streaming_mode: hybrid
  empty_detection_timeout_ms: 1500
"""
            f.write(yaml_content)
            f.flush()
            config = await load_config(pathlib.Path(f.name))
            assert config.streaming_retry.enabled is True
            assert config.streaming_retry.streaming_mode == "hybrid"
            assert config.streaming_retry.empty_detection_timeout_ms == 1500
        os.unlink(f.name)
