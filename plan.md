I think this is a great use case for another coding agent. The key is to make the requirements precise so it doesn't "improvise" the OpenAI protocol or break streaming.

I'd give it the following specification.

---

# Project Goal

Build a production-ready Python reverse proxy for GitHub Copilot that sits between VS Code and a local vLLM server.

```
GitHub Copilot
        │
        ▼
 OpenAI-compatible Proxy
        │
        ▼
     vLLM Server
```

The proxy must be completely transparent to the client except for configurable request rewriting.

---

# Technology

Use:

- Python 3.12+
- FastAPI
- Uvicorn
- httpx
- PyYAML
- pydantic-settings (optional)
- structlog or standard logging

No Flask.

No synchronous networking.

Everything must be async.

---

# Folder structure

```
copilot-proxy/

    proxy.py

    config.yaml

    requirements.txt

    README.md

    systemd/

        copilot-proxy.service

    logs/

    proxy/

        __init__.py

        config.py

        models.py

        middleware.py

        rewrite.py

        streaming.py

        logging.py

        health.py

        proxy.py
```

The code should be modular.

Do NOT implement everything in one file.

---

# Configuration

Configuration must be YAML.

Example:

```yaml
listen:
  host: 0.0.0.0

  port: 8081

vllm:
  url: http://127.0.0.1:8000

logging:
  requests: true

  responses: false

  body_preview_chars: 400

defaults:
  max_tokens: 4096

  temperature: 0.1

  top_p: 1.0

rewrite:
  clamp_max_tokens: true

  remove_reasoning: false

  inject_thinking_budget: false

  thinking_budget: 1024

models:
  "/mnt/models/vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8":
    max_tokens: 8192

  "/mnt/models/vllm/Qwen/Qwen3-0.6B":
    max_tokens: 2048
```

The proxy must reload config without restarting.

---

# API compatibility

Must proxy ALL endpoints transparently.

Examples:

```
GET /v1/models

POST /v1/chat/completions

POST /v1/completions

GET /health

GET /metrics
```

Unknown paths should simply forward.

---

# Streaming

This is the most important part.

The proxy MUST preserve Server Sent Events exactly.

Requirements:

- no buffering

- no parsing SSE

- no modifying streamed chunks

- preserve keep-alive messages

- preserve chunk ordering

- preserve content type

- preserve transfer encoding

The proxy may only rewrite the request BEFORE forwarding.

Never rewrite the response stream.

---

# Request rewriting

Before forwarding the JSON request:

If

```
max_tokens
```

exists

Clamp:

```
max_tokens = min(requested, configured_limit)
```

If absent

Inject:

```
max_tokens = configured_limit
```

Model-specific settings override defaults.

---

Support future rewrite hooks:

```
temperature

top_p

thinking_budget

reasoning

presence_penalty

frequency_penalty

repetition_penalty
```

These should all be optional.

---

# Logging

Log every request.

Example:

```
POST /v1/chat/completions

Model:
Qwen3-Coder

Prompt chars:
18234

max_tokens:
58555 -> 4096

Stream:
true

Client:
192.168.201.14
```

Do NOT log the full prompt by default.

Only preview the first configurable number of characters.

---

# Metrics

Expose

```
GET /metrics
```

Return JSON.

Example:

```
requests_total

streaming_requests

avg_latency

active_connections

rewritten_requests

forward_errors
```

---

# Error handling

If vLLM returns an error

Return it unchanged.

Do not wrap it.

Do not rewrite it.

Status code must remain identical.

---

# Timeouts

Do not use request timeout.

Streaming requests may run for hours.

Use:

```
timeout=None
```

---

# Connection reuse

Reuse AsyncClient.

Do NOT create a new AsyncClient per request.

Create one during FastAPI startup.

Close it on shutdown.

---

# Health endpoint

Implement

```
GET /health
```

Returns

```
{
  "status": "ok"
}
```

---

# Readiness endpoint

Implement

```
GET /ready
```

Proxy should call

```
GET /v1/models
```

on vLLM.

Return

```
ready
```

only if backend responds.

---

# Graceful shutdown

SIGTERM

SIGINT

Finish active streams.

Close AsyncClient.

Exit cleanly.

---

# Systemd

Provide service file.

```
Restart=always

RestartSec=5

WorkingDirectory=/opt/copilot-proxy

User=ubuntuai
```

---

# README

Include

Installation

Configuration

Running

systemd installation

Troubleshooting

Common GitHub Copilot issues

How to change max_tokens

How to add model-specific configuration

---

# Unit tests

Use pytest.

Tests for

- max_tokens clamping

- YAML loading

- model overrides

- request rewriting

- streaming passthrough

- unknown endpoint forwarding

---

# Code quality

Use

- type hints

- dataclasses or Pydantic

- Ruff-compatible formatting

- no global mutable state

- no duplicated code

---

# Future extension points

Design the architecture so the following can be added later without major refactoring:

- authentication (API keys)
- HTTPS/TLS
- Prometheus metrics
- rate limiting
- OpenTelemetry tracing
- per-client quotas
- caching of `/v1/models`
- request recording/replay
- token accounting
- request validation
- multiple upstream vLLM servers with load balancing and failover
- model routing (e.g., send Qwen3-Coder requests to one server and Qwen3-0.6B to another)

---

## One additional requirement I'd add

Ask the agent to **preserve all HTTP headers and status codes unless there is a specific reason to modify them**. In particular, it should forward headers like `Authorization`, `Accept`, `Content-Type`, and streaming-related headers correctly. Many OpenAI-compatible clients are sensitive to subtle differences in HTTP behavior, so minimizing changes at the proxy layer will make it much more robust.
