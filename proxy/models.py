"""Pydantic models for proxy configuration."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ListenConfig(BaseModel):
    """Configuration for the proxy listen address."""

    host: str = Field(default="0.0.0.0", description="Host to bind to.")
    port: int = Field(default=8081, ge=1, le=65535, description="Port to bind to.")


class LocalAIServerConfig(BaseModel):
    """Configuration for the upstream local AI server."""

    url: str = Field(
        default="http://127.0.0.1:8000",
        description="Base URL of the local AI server.",
    )


class LoggingConfig(BaseModel):
    """Configuration for request/response logging."""

    requests: bool = Field(default=True, description="Log incoming requests.")
    responses: bool = Field(default=False, description="Log upstream responses.")
    body_preview_chars: int = Field(
        default=400, ge=0, description="Max chars of body to log."
    )


class DefaultsConfig(BaseModel):
    """Default values injected into requests when not set by the client."""

    max_tokens: Optional[int] = Field(
        default=4096,
        ge=1,
        description="Default max_tokens to inject or clamp against.",
    )
    temperature: Optional[float] = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Default temperature value.",
    )
    top_p: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Default top_p value.",
    )


class RewriteConfig(BaseModel):
    """Configuration for request rewriting rules."""

    clamp_max_tokens: bool = Field(
        default=True,
        description="If True, clamp max_tokens to the configured limit.",
    )
    remove_reasoning: bool = Field(
        default=False,
        description="If True, strip reasoning-related fields.",
    )
    inject_thinking_budget: bool = Field(
        default=False,
        description="If True, inject a thinking_budget field.",
    )
    thinking_budget: Optional[int] = Field(
        default=1024,
        ge=0,
        description="Thinking budget value to inject when enabled.",
    )


class DebugConfig(BaseModel):
    """Configuration for raw debug logging of request/response exchanges."""

    enabled: bool = Field(
        default=False,
        description="If True, save full request/response exchanges to disk.",
    )
    log_dir: str = Field(
        default="logs/debug",
        description="Base directory for per-conversation debug log folders.",
    )
    max_response_size_mb: int = Field(
        default=20,
        ge=1,
        description="Max MB to buffer for streaming responses before truncating.",
    )


class ModelOverride(BaseModel):
    """Per-model override configuration."""

    max_tokens: Optional[int] = Field(
        default=None, ge=1, description="Override max_tokens for this model."
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Override temperature for this model."
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Override top_p for this model."
    )


class ProxyConfig(BaseModel):
    """Top-level configuration combining all sub-configs."""

    listen: ListenConfig = Field(default_factory=ListenConfig)
    localAIServer: LocalAIServerConfig = Field(default_factory=LocalAIServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    rewrite: RewriteConfig = Field(default_factory=RewriteConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    models: Dict[str, ModelOverride] = Field(
        default_factory=dict,
        description="Per-model override map keyed by model name/path.",
    )

    def get_model_max_tokens(self, model_name: str) -> Optional[int]:
        """Resolve the effective max_tokens for a given model name."""
        if model_name in self.models:
            override = self.models[model_name]
            if override.max_tokens is not None:
                return override.max_tokens
        return self.defaults.max_tokens

    def get_model_temperature(self, model_name: str) -> Optional[float]:
        """Resolve the effective temperature for a given model name."""
        if model_name in self.models:
            override = self.models[model_name]
            if override.temperature is not None:
                return override.temperature
        return self.defaults.temperature

    def get_model_top_p(self, model_name: str) -> Optional[float]:
        """Resolve the effective top_p for a given model name."""
        if model_name in self.models:
            override = self.models[model_name]
            if override.top_p is not None:
                return override.top_p
        return self.defaults.top_p
