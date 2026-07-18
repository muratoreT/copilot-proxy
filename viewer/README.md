# Log Viewer

A web-based viewer for browsing copilot-proxy debug logs.

## Prerequisites

- Python 3.10+
- Dependencies installed (`pip install -r requirements.txt`)

## Starting the Viewer

```bash
python -m viewer --log-dir logs/debug
```

### Options

| Flag         | Default       | Description                        |
|--------------|---------------|------------------------------------|
| `--log-dir`  | `./logs/debug` | Path to the debug log directory    |
| `--host`     | `127.0.0.1`   | Host to bind to                    |
| `--port`     | `8082`        | Port to bind to                    |

### Examples

```bash
# Default (logs/debug on port 8082)
python -m viewer

# Custom log directory and port
python -m viewer --log-dir /var/log/proxy/debug --port 9090
```

## Using the Viewer

Once started, open `http://127.0.0.1:8082` in a browser.

1. **Conversations list** — The home page shows all conversations detected in the log directory.
2. **Exchanges list** — Click a conversation to see all exchanges within it.
3. **Exchange detail** — Click an exchange to view the full request/response payload.

## Directory Structure

The viewer expects the following layout under `--log-dir`:

```
logs/debug/
├── <conversation_id_1>/
│   ├── exchange_001.json
│   ├── exchange_002.json
│   └── ...
├── <conversation_id_2>/
│   └── ...
└── ...
```

Each subdirectory represents a conversation, and each `exchange_*.json` file contains a single request/response exchange.
