# copilot-proxy

A production-ready Python reverse proxy for GitHub Copilot that sits between VS Code and a local vLLM server. Provides transparent request forwarding with configurable request rewriting, hot-reload configuration, and comprehensive logging & metrics.

```
GitHub Copilot
        │
        ▼
 OpenAI-compatible Proxy
        │
        ▼
     vLLM Server
```

---

## Features

- **Transparent proxying** — forwards all OpenAI-compatible API calls to vLLM unchanged
- **Request rewriting** — configurable `max_tokens` clamping, temperature/top_p defaults, thinking budget injection
- **Model-specific overrides** — per-model configuration in YAML
- **Hot-reload config** — watches `config.yaml` for changes and reloads without restart
- **SSE streaming passthrough** — preserves Server-Sent Events verbatim (no parsing, no buffering)
- **Structured logging** — request/response logging with structlog
- **Metrics endpoint** — `GET /metrics` exposes request counts, latency, active connections, and error rates
- **Health & readiness checks** — `GET /health` and `GET /ready` for container orchestration

---

## Installation

### Prerequisites

- Python 3.12+
- A running vLLM server (default: `http://127.0.0.1:8000`)

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

### Configuration Reference

| Section    | Key                      | Type   | Default                 | Description                          |
| ---------- | ------------------------ | ------ | ----------------------- | ------------------------------------ |
| `listen`   | `host`                   | string | `0.0.0.0`               | Bind address                         |
| `listen`   | `port`                   | int    | `8081`                  | Bind port                            |
| `vllm`     | `url`                    | string | `http://127.0.0.1:8000` | Upstream vLLM URL                    |
| `logging`  | `requests`               | bool   | `true`                  | Log incoming requests                |
| `logging`  | `responses`              | bool   | `false`                 | Log upstream responses               |
| `logging`  | `body_preview_chars`     | int    | `400`                   | Max chars of body to log             |
| `defaults` | `max_tokens`             | int    | `4096`                  | Default max_tokens                   |
| `defaults` | `temperature`            | float  | `0.1`                   | Default temperature                  |
| `defaults` | `top_p`                  | float  | `1.0`                   | Default top_p                        |
| `rewrite`  | `clamp_max_tokens`       | bool   | `true`                  | Clamp max_tokens to configured limit |
| `rewrite`  | `remove_reasoning`       | bool   | `false`                 | Strip reasoning fields               |
| `rewrite`  | `inject_thinking_budget` | bool   | `false`                 | Inject thinking_budget field         |
| `rewrite`  | `thinking_budget`        | int    | `1024`                  | Thinking budget value                |
| `models`   | `<model_name>`           | object | —                       | Per-model overrides                  |
| `models.*` | `max_tokens`             | int    | —                       | Override max_tokens for model        |
| `models.*` | `temperature`            | float  | —                       | Override temperature for model       |
| `models.*` | `top_p`                  | float  | —                       | Override top_p for model             |

### How Request Rewriting Works

1. **max_tokens clamping**: If `clamp_max_tokens` is `true`:
   - If the request has no `max_tokens`, the effective limit is injected
   - If the request's `max_tokens` exceeds the effective limit, it is clamped down
   - The effective limit is resolved: model-specific override → default

2. **Temperature/top_p defaults**: Injected only if the client did not specify a value

3. **Thinking budget**: If `inject_thinking_budget` is `true`, a `thinking_budget` field is added when absent

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

| Method | Path       | Description                                |
| ------ | ---------- | ------------------------------------------ |
| GET    | `/health`  | Health check → `{"status": "ok"}`          |
| GET    | `/ready`   | Readiness check → probes vLLM `/v1/models` |
| GET    | `/metrics` | Metrics snapshot (JSON)                    |

### Proxied Endpoints

All other paths are forwarded to the upstream vLLM server:

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

- Verify vLLM is running at the configured URL
- Check `GET /ready` — it should return `{"status": "ready"}`
- View proxy logs: `journalctl -u copilot-proxy -f` (systemd) or console output

### Viewing metrics

```bash
curl http://localhost:8081/metrics | python -m json.tool
```

---

## Project Structure

```
copilot-proxy/
├── proxy_entry.py              # Entry point
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md
├── systemd/
│   └── copilot-proxy.service   # systemd unit file
├── logs/                       # Log directory
├── tests/                      # Unit tests
│   ├── test_rewrite.py
│   ├── test_config.py
│   ├── test_streaming.py
│   └── test_proxy.py
└── proxy/                      # Application package
    ├── __init__.py
    ├── config.py               # YAML config loader + hot-reload
    ├── models.py               # Pydantic config models
    ├── middleware.py           # Timing & connection tracking
    ├── rewrite.py              # Request body rewriting
    ├── streaming.py            # SSE streaming passthrough
    ├── logging.py              # Structlog request logging
    ├── health.py               # Health & readiness endpoints
    ├── metrics.py              # Metrics tracking
    └── proxy.py                # FastAPI app setup
```

---

## License

MIT
