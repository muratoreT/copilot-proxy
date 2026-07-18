# Implementation Plan: copilot-proxy

## Phase 1: Project Scaffolding

### Step 1: Create folder structure

- Create the following directories:
  - `proxy/`
  - `systemd/`
  - `logs/`
  - `tests/`

### Step 2: Create `requirements.txt`

- Add dependencies:
  - `fastapi`
  - `uvicorn[standard]`
  - `httpx`
  - `pyyaml`
  - `pydantic-settings`
  - `structlog`
  - `pytest`
  - `pytest-asyncio`
  - `pytest-httpx`

### Step 3: Initialize `proxy/__init__.py`

- Empty file or minimal package docstring.

---

## Phase 2: Configuration Layer

### Step 4: Implement `proxy/models.py`

- Define Pydantic dataclasses for:
  - `ListenConfig` (host, port)
  - `VLLMConfig` (url)
  - `LoggingConfig` (requests, responses, body_preview_chars)
  - `DefaultsConfig` (max_tokens, temperature, top_p)
  - `RewriteConfig` (clamp_max_tokens, remove_reasoning, inject_thinking_budget, thinking_budget)
  - `ModelConfig` (per-model overrides)
  - `ProxyConfig` (top-level config combining all above)

### Step 5: Implement `proxy/config.py`

- YAML loader function that reads `config.yaml` and validates against `ProxyConfig`.
- Config reload mechanism:
  - File watcher or periodic reload trigger.
  - Thread-safe config access (e.g., `asyncio.Lock`).

### Step 6: Create `config.yaml`

- Write the example config from the spec as the default file.

---

## Phase 3: Logging & Metrics

### Step 7: Implement `proxy/logging.py`

- Configure structlog with console output.
- Create a request logger function that logs:
  - HTTP method and path
  - model name (extracted from request body)
  - prompt char count
  - max_tokens rewriting (original → clamped)
  - stream flag
  - client IP
- Respect `body_preview_chars` limit.

### Step 8: Implement metrics tracking

- Create a metrics module (e.g., `proxy/metrics.py`) or embed in the main proxy file.
- Track:
  - `requests_total`
  - `streaming_requests`
  - `avg_latency`
  - `active_connections`
  - `rewritten_requests`
  - `forward_errors`
- Expose `GET /metrics` endpoint returning JSON.

---

## Phase 4: Request Rewriting

### Step 9: Implement `proxy/rewrite.py`

- Function `rewrite_request(request_body: dict, config: ProxyConfig) -> dict`:
  - Extract `model` from request body.
  - Resolve effective `max_tokens` limit (model-specific override → default).
  - Apply `clamp_max_tokens`: `min(requested, limit)` or inject if absent.
  - Placeholder hooks for future rewrites:
    - `temperature`
    - `top_p`
    - `thinking_budget`
    - `reasoning`
    - `presence_penalty`
    - `frequency_penalty`
    - `repetition_penalty`
- Return the modified request body.

---

## Phase 5: Streaming Passthrough

### Step 10: Implement `proxy/streaming.py`

- Function to forward the rewritten request to vLLM using httpx.
- For streaming responses:
  - Use `httpx.AsyncClient().stream()` to iterate over backend chunks.
  - Yield each chunk verbatim — no parsing, no modification.
  - Preserve `Content-Type`, `Transfer-Encoding`, and keep-alive messages.
- For non-streaming responses:
  - Read and return the full response body unchanged.
- Forward all headers and status codes exactly.

---

## Phase 6: Middleware & Health Endpoints

### Step 11: Implement `proxy/middleware.py`

- Request timing middleware (for latency tracking).
- Active connection counter middleware.
- Error counting middleware.

### Step 12: Implement `proxy/health.py`

- `GET /health` → `{"status": "ok"}`
- `GET /ready` → proxy to vLLM's `GET /v1/models`, return `{"status": "ready"}` on success, error on failure.

---

## Phase 7: Main Proxy Application

### Step 13: Implement `proxy/proxy.py`

- FastAPI app setup.
- Lifespan context manager:
  - Startup: create shared `httpx.AsyncClient`, load config, start config watcher.
  - Shutdown: close AsyncClient, finish active streams.
- Register health and readiness endpoints.
- Register `/metrics` endpoint.
- Catch-all route for transparent proxying:
  - Read request body.
  - If JSON, apply rewriting via `proxy/rewrite.py`.
  - Forward to vLLM via `proxy/streaming.py`.
  - Return response (streaming or not) to the client.
- Handle unknown paths by forwarding as-is.

### Step 14: Create `proxy.py` (entry point)

- Minimal script that creates the Uvicorn config and runs the app.
- Read host/port from config.
- Wire up graceful shutdown (SIGTERM/SIGINT handling via Uvicorn's built-in support).

---

## Phase 8: Systemd Integration

### Step 15: Create `systemd/copilot-proxy.service`

- Service file with:
  - `Restart=always`
  - `RestartSec=5`
  - `WorkingDirectory=/opt/copilot-proxy`
  - `User=ubuntuai`
  - ExecStart pointing to the Python entry point.

---

## Phase 9: Documentation

### Step 16: Create `README.md`

- Installation instructions.
- Configuration guide (YAML schema reference).
- Running instructions (direct and via systemd).
- Troubleshooting section:
  - Common GitHub Copilot issues.
  - How to change `max_tokens`.
  - How to add model-specific configuration.

---

## Phase 10: Testing

### Step 17: Write unit tests

- `tests/test_rewrite.py`:
  - max_tokens clamping (requested > limit → clamped).
  - max_tokens injection (absent → injected).
  - model-specific overrides.
  - temperature/top_p passthrough when not configured.
- `tests/test_config.py`:
  - YAML loading and validation.
  - Invalid config rejection.
  - model override resolution.
- `tests/test_streaming.py`:
  - Streaming passthrough preserves SSE format.
  - Chunk ordering preserved.
  - Content-Type preserved.
- `tests/test_proxy.py`:
  - Unknown endpoint forwarding.
  - Error forwarding (status code unchanged).
  - Health and readiness endpoints.
  - Metrics endpoint.

### Step 18: Run tests and fix issues

- Run `pytest` and iterate on any failures.

---

## Phase 11: Final Review

### Step 19: Code quality pass

- Verify type hints on all functions.
- Run Ruff (or equivalent linter) and fix issues.
- Ensure no global mutable state.
- Ensure no duplicated code.
- Verify async throughout (no sync networking).

### Step 20: Integration test

- Start vLLM (or mock it).
- Start the proxy.
- Send real OpenAI-compatible requests through the proxy.
- Verify streaming works end-to-end.
- Verify config reload works without restart.

---

## Summary of Files to Create

| #   | File                            | Phase |
| --- | ------------------------------- | ----- |
| 1   | `proxy/__init__.py`             | 1     |
| 2   | `requirements.txt`              | 1     |
| 3   | `proxy/models.py`               | 2     |
| 4   | `proxy/config.py`               | 2     |
| 5   | `config.yaml`                   | 2     |
| 6   | `proxy/logging.py`              | 3     |
| 7   | `proxy/metrics.py`              | 3     |
| 8   | `proxy/rewrite.py`              | 4     |
| 9   | `proxy/streaming.py`            | 5     |
| 10  | `proxy/middleware.py`           | 6     |
| 11  | `proxy/health.py`               | 6     |
| 12  | `proxy/proxy.py`                | 7     |
| 13  | `proxy.py` (entry)              | 7     |
| 14  | `systemd/copilot-proxy.service` | 8     |
| 15  | `README.md`                     | 9     |
| 16  | `tests/test_rewrite.py`         | 10    |
| 17  | `tests/test_config.py`          | 10    |
| 18  | `tests/test_streaming.py`       | 10    |
| 19  | `tests/test_proxy.py`           | 10    |
