# copilot-proxy

**Main purpose:** Improve compatibility between VS Code GitHub Copilot and a local AI model.

VS Code's GitHub Copilot extension is built to talk to OpenAI's cloud API, but many developers want to use a local AI model instead — for privacy, cost, or offline access. The problem is that local models often don't speak the same "dialect" as OpenAI: they may not support the same parameters, handle tool calls differently, or choke on request shapes that Copilot sends.

This proxy sits in the middle to bridge that gap. It intercepts the requests from Copilot, rewrites them so the local model can understand them, and translates the responses back. Think of it as an adapter that lets you swap out OpenAI for any local AI server without Copilot noticing.

```
GitHub Copilot
        │
        ▼
 OpenAI-compatible Proxy
        │
        ▼
  Local AI Server
```

---

## Features

- **Transparent proxying** — forwards all OpenAI-compatible API calls to the local AI server unchanged
- **Request rewriting** — configurable `max_tokens` clamping, temperature/top_p defaults, thinking budget injection
- **Tool vision compatibility** — moves tool-result images into LM Studio-compatible user messages
- **Model-specific overrides** — per-model configuration in YAML
- **Hot-reload config** — watches `config.yaml` for changes and reloads without restart
- **SSE streaming passthrough** — preserves Server-Sent Events verbatim (no parsing, no buffering)
- **Streaming retry** — automatic retry on empty or thinking-only streaming responses
- **Debug logging** — full request/response exchange logging organized by conversation
- **Log viewer** — web-based viewer for browsing debug logs
- **Structured logging** — request/response logging with structlog
- **Metrics endpoint** — `GET /metrics` exposes request counts, latency, active connections, and error rates
- **Health & readiness checks** — `GET /health` and `GET /ready` for container orchestration

---

## Installation

### Prerequisites

- Python 3.12+
- A running local AI server (default: `http://127.0.0.1:8000`)

### Setup

```bash
# Clone or copy the project
cd copilot-proxy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Edit `config.yaml` to customize the proxy behavior:

```yaml
listen:
  host: 0.0.0.0
  port: 8081

localAIServer:
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
  normalize_tool_vision: true
  remove_reasoning: false
  inject_thinking_budget: false
  thinking_budget: 1024
  max_thinking_budget: null

debug:
  enabled: false
  log_dir: logs/debug
  max_response_size_mb: 20

streaming_retry:
  enabled: false
  max_retries: 1
  only_after_tool_messages: true
  retry_delay_ms: 100
  retry_on_thinking_only: false

models:
  "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8":
    max_tokens: 8192
  "Qwen/Qwen3-0.6B":
    max_tokens: 2048
```

### Configuration Reference

#### `listen` — Proxy Bind Address

| Key    | Type   | Default   | Description                       |
| ------ | ------ | --------- | --------------------------------- |
| `host` | string | `0.0.0.0` | Host address to bind the proxy to |
| `port` | int    | `8081`    | Port to listen on (1–65535)       |

> **When to change:** Set `host` to `127.0.0.1` if you only want local access. Use `0.0.0.0` to allow connections from other machines (e.g., when VS Code runs in a container).

#### `localAIServer` — Upstream AI Server

| Key   | Type   | Default                 | Description                                                |
| ----- | ------ | ----------------------- | ---------------------------------------------------------- |
| `url` | string | `http://127.0.0.1:8000` | Base URL of your local AI server (LM Studio, Ollama, etc.) |

> **When to change:** Always update this to match your AI server's address and port.

#### `logging` — Structured Request/Response Logging

| Key                  | Type | Default | Description                                    |
| -------------------- | ---- | ------- | ---------------------------------------------- |
| `requests`           | bool | `true`  | Log incoming request details to console        |
| `responses`          | bool | `false` | Log upstream response details to console       |
| `body_preview_chars` | int  | `400`   | Max characters of request/response body to log |

> **When to change:** Enable `responses` for troubleshooting upstream issues. Reduce `body_preview_chars` if logs are too verbose.

#### `defaults` — Default Request Values

| Key           | Type  | Default | Description                                             |
| ------------- | ----- | ------- | ------------------------------------------------------- |
| `max_tokens`  | int   | `4096`  | Default max_tokens injected when client doesn't set one |
| `temperature` | float | `0.1`   | Default temperature (0.0–2.0)                           |
| `top_p`       | float | `1.0`   | Default top_p (0.0–1.0)                                 |

> **When to change:** Lower `max_tokens` if your model has a small context window. Adjust `temperature` for more/less deterministic outputs.

#### `rewrite` — Request Body Rewriting Rules

| Key                      | Type | Default | Description                                                     |
| ------------------------ | ---- | ------- | --------------------------------------------------------------- |
| `clamp_max_tokens`       | bool | `true`  | Clamp `max_tokens` to the configured limit                      |
| `normalize_tool_vision`  | bool | `false` | Move image content from tool messages into user messages        |
| `remove_reasoning`       | bool | `false` | Strip reasoning-related fields from requests                    |
| `inject_thinking_budget` | bool | `false` | Inject a `thinking_budget` field when absent                    |
| `thinking_budget`        | int  | `1024`  | Thinking budget value to inject (0 disables)                    |
| `max_thinking_budget`    | int  | `null`  | Hard ceiling for thinking_budget; clamps client values above it |

> **When to change:**
>
> - Enable `clamp_max_tokens` to prevent requests from exceeding your model's token limits.
> - Enable `normalize_tool_vision` when using LM Studio with tool-use + vision (it doesn't support images in tool messages).
> - Set `max_thinking_budget` to cap how much the model can "think" regardless of what the client requests.
> - Enable `inject_thinking_budget` for models that support extended thinking (e.g., Qwen3 reasoning models).

#### `debug` — Raw Debug Logging

| Key                    | Type   | Default      | Description                                                |
| ---------------------- | ------ | ------------ | ---------------------------------------------------------- |
| `enabled`              | bool   | `false`      | Save full request/response exchanges to disk               |
| `log_dir`              | string | `logs/debug` | Directory for per-conversation debug log folders           |
| `max_response_size_mb` | int    | `20`         | Max MB to buffer for streaming responses before truncating |

> **When to change:** Enable for debugging specific conversations. Each conversation gets its own folder with numbered exchange JSON files. Use the built-in log viewer (`python -m viewer`) to browse them. Disable in production to avoid disk usage.

#### `streaming_retry` — Automatic Retry on Empty Streams

| Key                        | Type | Default | Description                                                   |
| -------------------------- | ---- | ------- | ------------------------------------------------------------- |
| `enabled`                  | bool | `false` | Enable retry on empty streaming completions                   |
| `max_retries`              | int  | `1`     | Max retry attempts (0–3)                                      |
| `only_after_tool_messages` | bool | `true`  | Only retry when the request includes tool messages            |
| `retry_delay_ms`           | int  | `100`   | Delay between retry attempts (0–5000 ms)                      |
| `retry_on_thinking_only`   | bool | `false` | Retry when the stream produces only thinking content, no text |

> **When to change:**
>
> - Enable `enabled` if your upstream server occasionally returns empty streams (common with some local AI servers under load).
> - Set `only_after_tool_messages: false` to retry all empty streams, not just those following tool calls.
> - Enable `retry_on_thinking_only` if the model sometimes gets stuck producing only internal reasoning with no output.

#### `models` — Per-Model Overrides

| Key           | Type  | Default | Description                                   |
| ------------- | ----- | ------- | --------------------------------------------- |
| `max_tokens`  | int   | —       | Override max_tokens for this model            |
| `temperature` | float | —       | Override temperature for this model (0.0–2.0) |
| `top_p`       | float | —       | Override top_p for this model (0.0–1.0)       |

> **When to change:** Add entries for models that need different limits than the defaults. Key by the model name/path as it appears in requests.

```yaml
models:
  "qwen/qwen3.6-27b":
    max_tokens: 8192
    temperature: 0.2
  "qwen3-0.6b":
    max_tokens: 2048
```

### How Request Rewriting Works

1. **max_tokens clamping**: If `clamp_max_tokens` is `true`:
   - If the request has no `max_tokens`, the effective limit is injected
   - If the request's `max_tokens` exceeds the effective limit, it is clamped down
   - The effective limit is resolved: model-specific override → default

2. **Temperature/top_p defaults**: Injected only if the client did not specify a value

3. **Tool vision normalization**: If `normalize_tool_vision` is `true`, image parts are removed from `tool` messages and appended after the complete tool-result block in a multimodal `user` message. This preserves tool-call ordering while using the message shape accepted by LM Studio.

4. **Thinking budget**: If `inject_thinking_budget` is `true`, a `thinking_budget` field is added when absent

5. **Thinking budget ceiling**: If `max_thinking_budget` is set, any client-provided `thinking_budget` exceeding this value is clamped down

### How Streaming Retry Works

When `streaming_retry.enabled` is `true`:

1. The proxy monitors streaming responses for empty completions
2. If an empty response is detected and retry conditions are met:
   - `only_after_tool_messages` is `false`, OR the request contains tool messages
   - `retry_on_thinking_only` is `false`, OR the stream contained only thinking content
3. The request is re-sent to the upstream server after `retry_delay_ms`
4. Retries continue up to `max_retries` attempts
5. If all retries fail, the empty response is passed through to the client

### How Debug Logging Works

When `debug.enabled` is `true`:

1. Each request/response exchange is captured with full payloads
2. Exchanges are organized by conversation key (derived from the first user message)
3. Files are written as `exchange_NNN.json` in per-conversation subdirectories
4. Sensitive headers (authorization, cookies, etc.) are stripped before logging
5. Streaming responses are buffered up to `max_response_size_mb` before truncation
6. Use `python -m viewer --log-dir logs/debug` to browse logs in a web interface

---

## Running

### Direct execution

```bash
python proxy_entry.py
```

The proxy will start on the address configured in `config.yaml` (default: `0.0.0.0:8081`).

### Via systemd (Linux)

```bash
# Copy the service file
sudo cp systemd/copilot-proxy.service /etc/systemd/system/

# Edit the service file to match your environment
# (WorkingDirectory, ExecStart, User/Group)

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now copilot-proxy

# Check status
sudo systemctl status copilot-proxy

# View logs
journalctl -u copilot-proxy -f
```

---

## API Endpoints

### Proxy Endpoints (built-in)

| Method | Path       | Description                                    |
| ------ | ---------- | ---------------------------------------------- |
| GET    | `/health`  | Health check → `{"status": "ok"}`              |
| GET    | `/ready`   | Readiness check → probes upstream `/v1/models` |
| GET    | `/metrics` | Metrics snapshot (JSON)                        |

### Proxied Endpoints

All other paths are forwarded to the upstream local AI server:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- Any other path → forwarded as-is

---

## Troubleshooting

### GitHub Copilot won't connect to the proxy

- Ensure the proxy is running and listening on the correct port
- Check that `config.yaml` has the correct `listen` settings
- Verify VS Code / Copilot is configured to use the proxy URL

### How to change max_tokens

1. Edit `config.yaml`
2. Set `defaults.max_tokens` for a global limit
3. Or add a model-specific override under `models:`
4. The proxy will reload automatically (within 5 seconds)

### How to add model-specific configuration

```yaml
models:
  "your-model-name-or-path":
    max_tokens: 8192
    temperature: 0.5
    top_p: 0.9
```

### Upstream connection errors

- Verify the local AI server is running at the configured URL
- Check `GET /ready` — it should return `{"status": "ready"}`
- View proxy logs: `journalctl -u copilot-proxy -f` (systemd) or console output

### Viewing metrics

```bash
curl http://localhost:8081/metrics | python -m json.tool
```

### Debugging empty or stuck responses

1. Enable debug logging in `config.yaml`:
   ```yaml
   debug:
     enabled: true
   ```
2. Reproduce the issue
3. Browse the captured exchanges with the log viewer:
   ```bash
   python -m viewer --log-dir logs/debug
   ```
4. Open `http://127.0.0.1:8082` in a browser to inspect request/response payloads

### Enabling streaming retry for flaky upstreams

If your local AI server occasionally returns empty streams:

```yaml
streaming_retry:
  enabled: true
  max_retries: 2
  retry_delay_ms: 200
```

This will automatically retry empty streaming responses up to 2 times with a 200ms delay between attempts.

---

## Project Structure

```
copilot-proxy/
├── proxy_entry.py              # Entry point
├── config.yaml                 # Configuration file
├── config.yaml.example         # Example configuration
├── requirements.txt            # Python dependencies
├── README.md
├── systemd/
│   └── copilot-proxy.service   # systemd unit file
├── logs/
│   └── debug/                  # Debug log directory (per-conversation)
├── tests/                      # Unit tests
│   ├── test_rewrite.py
│   ├── test_config.py
│   ├── test_streaming.py
│   ├── test_proxy.py
│   └── test_debuglogging.py
├── proxy/                      # Application package
│   ├── __init__.py
│   ├── config.py               # YAML config loader + hot-reload
│   ├── models.py               # Pydantic config models
│   ├── middleware.py           # Timing & connection tracking
│   ├── rewrite.py              # Request body rewriting
│   ├── streaming.py            # SSE streaming passthrough + retry
│   ├── debuglogging.py         # Full request/response exchange logging
│   ├── logging.py              # Structlog request logging
│   ├── health.py               # Health & readiness endpoints
│   ├── metrics.py              # Metrics tracking
│   └── proxy.py                # FastAPI app setup
└── viewer/                     # Web-based debug log viewer
    ├── main.py
    ├── static/                 # CSS and JS assets
    └── templates/              # HTML templates
```

---

## License

MIT
