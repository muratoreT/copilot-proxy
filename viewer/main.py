"""Log viewer FastAPI application."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Copilot Proxy Log Viewer")
jinja_env = Environment(
    loader=FileSystemLoader("viewer/templates"),
    autoescape=True,
)
LOG_DIR: Path = Path("./logs/debug")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ExchangeSummary(BaseModel):
    """Lightweight summary of a single exchange."""
    num: int
    request_method: str = ""
    request_timestamp: str = ""
    user_prompt: str = ""
    response_status_code: Optional[int] = None
    response_duration_ms: Optional[float] = None
    retry_count: int = 0


class ConversationInfo(BaseModel):
    """Metadata about a conversation log folder."""
    key: str
    exchange_count: int
    first_timestamp: str = ""
    last_timestamp: str = ""
    exchanges: List[ExchangeSummary] = []


# ---------------------------------------------------------------------------
# Log scanning
# ---------------------------------------------------------------------------

def _parse_exchange_file(filepath: Path) -> Dict[str, Any]:
    """Read and parse a single exchange JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to parse %s: %s", filepath, e)
        return {}


def _extract_first_user_message(data: Dict[str, Any]) -> str:
    """Extract the first user message content from an exchange, truncated."""
    body = data.get("request", {}).get("body", {})
    messages = body.get("messages", [])

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            # Strip XML tags and whitespace for a clean preview
            import re
            text = re.sub(r'<[^>]+>', '', content).strip()
            if len(text) > 100:
                text = text[:97] + "..."
            return text
        break
    return ""


def _scan_conversations(log_dir: Path) -> List[ConversationInfo]:
    """Scan log directory and build conversation index."""
    conversations: List[ConversationInfo] = []

    if not log_dir.exists():
        logger.warning("Log directory does not exist: %s", log_dir)
        return conversations

    for entry in sorted(log_dir.iterdir()):
        if not entry.is_dir():
            continue

        exchanges: List[ExchangeSummary] = []
        exchange_files = sorted(entry.glob("exchange_*.json"))

        for ef in exchange_files:
            data = _parse_exchange_file(ef)
            req = data.get("request", {})
            resp = data.get("response", {})

            exchanges.append(
                ExchangeSummary(
                    num=data.get("exchange", 0),
                    request_method=req.get("method", ""),
                    request_timestamp=req.get("timestamp", ""),
                    user_prompt=_extract_first_user_message(data),
                    response_status_code=resp.get("status_code"),
                    response_duration_ms=resp.get("duration_ms"),
                    retry_count=data.get("retry_count", 0),
                )
            )

        timestamps = [e.request_timestamp for e in exchanges if e.request_timestamp]

        conversations.append(
            ConversationInfo(
                key=entry.name,
                exchange_count=len(exchanges),
                first_timestamp=timestamps[0] if timestamps else "",
                last_timestamp=timestamps[-1] if timestamps else "",
                exchanges=exchanges,
            )
        )

    return conversations


def _load_exchange(log_dir: Path, conv_key: str, exchange_num: int) -> Dict[str, Any]:
    """Load a single exchange file by conversation key and exchange number."""
    filename = f"exchange_{exchange_num:03d}.json"
    filepath = log_dir / conv_key / filename
    return _parse_exchange_file(filepath)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _render(template_name: str, context: Dict[str, Any]) -> HTMLResponse:
    """Render a Jinja2 template to an HTML response."""
    template = jinja_env.get_template(template_name)
    html = template.render(**context)
    return HTMLResponse(html)


@app.get("/")
async def conversation_list(request: Request):
    """List all conversations."""
    conversations = _scan_conversations(LOG_DIR)
    return _render(
        "conversations.html",
        {
            "conversations": [c.model_dump(mode="json") for c in conversations],
            "log_dir": str(LOG_DIR),
        },
    )


@app.get("/conversation/{conv_key}")
async def conversation_detail(request: Request, conv_key: str):
    """Show all exchanges for a conversation."""
    conversations = _scan_conversations(LOG_DIR)
    conv = next((c for c in conversations if c.key == conv_key), None)
    if conv is None:
        return _render(
            "error.html",
            {"status": 404, "message": f"Conversation '{conv_key}' not found."},
        )
    return _render(
        "exchanges.html",
        {"conversation": conv.model_dump(mode="json")},
    )


@app.get("/conversation/{conv_key}/exchange/{exchange_num}")
async def exchange_detail(request: Request, conv_key: str, exchange_num: int):
    """Show a single exchange in full detail."""
    exchange = _load_exchange(LOG_DIR, conv_key, exchange_num)
    if not exchange:
        return _render(
            "error.html",
            {"status": 404, "message": f"Exchange {exchange_num} not found."},
        )
    return _render(
        "exchange_detail.html",
        {"conv_key": conv_key, "exchange": exchange},
    )


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="viewer/static"), name="static")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Copilot Proxy Log Viewer")
    parser.add_argument(
        "--log-dir",
        default="./logs/debug",
        help="Path to debug log directory (default: ./logs/debug)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8082, help="Port to bind to")
    args = parser.parse_args()

    global LOG_DIR
    LOG_DIR = Path(args.log_dir)

    if not LOG_DIR.exists():
        logger.error("Log directory does not exist: %s", LOG_DIR)
        return

    logger.info("Serving logs from %s", LOG_DIR)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
