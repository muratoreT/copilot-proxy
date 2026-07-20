"""Tests for proxy/debuglogging.py — debug logging module."""

import asyncio
import json
import os
import pathlib
import tempfile
import time

import pytest

from proxy.models import DebugConfig
from proxy.debuglogging import DebugLogger, ExchangeData


class TestConversationKey:
    """Test conversation key computation."""

    def test_conversation_key_consistent(self):
        """Same input produces same hash."""
        body = {"messages": [{"role": "user", "content": "Hello world"}]}
        key1 = DebugLogger.compute_conversation_key(body, "127.0.0.1")
        key2 = DebugLogger.compute_conversation_key(body, "127.0.0.1")
        assert key1 == key2

    def test_conversation_key_different_content(self):
        """Different message content produces different key."""
        body1 = {"messages": [{"role": "user", "content": "Hello"}]}
        body2 = {"messages": [{"role": "user", "content": "Goodbye"}]}
        key1 = DebugLogger.compute_conversation_key(body1, "127.0.0.1")
        key2 = DebugLogger.compute_conversation_key(body2, "127.0.0.1")
        assert key1 != key2

    def test_conversation_key_different_ip(self):
        """Different client IP produces different key."""
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        key1 = DebugLogger.compute_conversation_key(body, "127.0.0.1")
        key2 = DebugLogger.compute_conversation_key(body, "192.168.1.1")
        assert key1 != key2

    def test_conversation_key_format(self):
        """Key is 16-character hex string."""
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        key = DebugLogger.compute_conversation_key(body, "127.0.0.1")
        assert len(key) == 16
        int(key, 16)  # Should not raise — valid hex

    def test_conversation_key_with_list_content(self):
        """Handles list content (multimodal messages)."""
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First part"},
                        {"type": "text", "text": "Second part"},
                    ],
                }
            ]
        }
        key = DebugLogger.compute_conversation_key(body, "127.0.0.1")
        assert len(key) == 16


class TestDebugLogger:
    """Test DebugLogger file writing and exchange capture."""

    @pytest.mark.asyncio
    async def test_exchange_file_written(self, tmp_path):
        """Exchange JSON file is created on completion."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path))
        logger = DebugLogger(config)

        body = {"messages": [{"role": "user", "content": "Test message"}]}
        conv_key, exchange_num = await logger.start_exchange(
            body=body,
            client_ip="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
        )

        assert conv_key != ""
        assert exchange_num == 1

        await logger.complete_exchange(
            conv_key=conv_key,
            exchange_num=exchange_num,
            status_code=200,
            headers={"Content-Type": "application/json"},
            body='{"choices": [{"message": {"content": "Response"}}]}',
            duration_ms=1234.56,
        )

        # Verify file exists
        conv_dir = tmp_path / conv_key
        exchange_file = conv_dir / "exchange_001.json"
        assert exchange_file.exists()

        # Verify JSON structure
        with open(exchange_file) as f:
            data = json.load(f)

        assert "request" in data
        assert "response" in data
        assert data["request"]["method"] == "POST"
        assert data["request"]["path"] == "/v1/chat/completions"
        assert data["response"]["status_code"] == 200
        assert data["response"]["duration_ms"] == 1234.56

    @pytest.mark.asyncio
    async def test_exchange_contains_all_captured_fields(self, tmp_path):
        """Exchange file contains all 6 required captured data items."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path))
        logger = DebugLogger(config)

        body = {"messages": [{"role": "user", "content": "Test"}]}
        conv_key, exchange_num = await logger.start_exchange(
            body=body,
            client_ip="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": "Bearer test"},
        )

        await logger.complete_exchange(
            conv_key=conv_key,
            exchange_num=exchange_num,
            status_code=200,
            headers={"Content-Type": "application/json"},
            body='{"choices": []}',
            duration_ms=500.0,
        )

        exchange_file = tmp_path / conv_key / "exchange_001.json"
        with open(exchange_file) as f:
            data = json.load(f)

        # Request fields
        req = data["request"]
        assert "body" in req
        assert "headers" in req
        assert "timestamp" in req

        # Response fields
        resp = data["response"]
        assert "body" in resp
        assert "headers" in resp
        assert "status_code" in resp
        assert "duration_ms" in resp
        assert "timestamp" in resp

    @pytest.mark.asyncio
    async def test_debug_disabled_no_files(self, tmp_path):
        """When debug is disabled, no files are written."""
        config = DebugConfig(enabled=False, log_dir=str(tmp_path))
        logger = DebugLogger(config)

        body = {"messages": [{"role": "user", "content": "Test"}]}
        conv_key, exchange_num = await logger.start_exchange(
            body=body,
            client_ip="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            headers={},
        )

        assert conv_key == ""
        assert exchange_num == 0

        await logger.complete_exchange(
            conv_key=conv_key,
            exchange_num=exchange_num,
            status_code=200,
            headers={},
            body="{}",
            duration_ms=100.0,
        )

        # No conversation folder should exist
        assert tmp_path.is_dir()
        assert len(list(tmp_path.iterdir())) == 0

    @pytest.mark.asyncio
    async def test_multiple_exchanges_incremented(self, tmp_path):
        """Multiple exchanges in same conversation get incrementing numbers."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path))
        logger = DebugLogger(config)

        body = {"messages": [{"role": "user", "content": "First message"}]}

        # Exchange 1
        conv_key1, num1 = await logger.start_exchange(
            body=body, client_ip="127.0.0.1", method="POST", path="/v1/chat/completions", headers={}
        )
        await logger.complete_exchange(conv_key1, num1, 200, {}, "{}", 100.0)

        # Exchange 2
        conv_key2, num2 = await logger.start_exchange(
            body=body, client_ip="127.0.0.1", method="POST", path="/v1/chat/completions", headers={}
        )
        await logger.complete_exchange(conv_key2, num2, 200, {}, "{}", 200.0)

        # Exchange 3
        conv_key3, num3 = await logger.start_exchange(
            body=body, client_ip="127.0.0.1", method="POST", path="/v1/chat/completions", headers={}
        )
        await logger.complete_exchange(conv_key3, num3, 200, {}, "{}", 300.0)

        # Same conversation key
        assert conv_key1 == conv_key2 == conv_key3

        # Incremented numbers
        assert num1 == 1
        assert num2 == 2
        assert num3 == 3

        # Files exist
        conv_dir = tmp_path / conv_key1
        assert (conv_dir / "exchange_001.json").exists()
        assert (conv_dir / "exchange_002.json").exists()
        assert (conv_dir / "exchange_003.json").exists()

    @pytest.mark.asyncio
    async def test_response_truncation(self, tmp_path):
        """Large response is truncated at max_response_size_mb."""
        # 1 MB limit
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), max_response_size_mb=1)
        logger = DebugLogger(config)

        body = {"messages": [{"role": "user", "content": "Test"}]}
        conv_key, exchange_num = await logger.start_exchange(
            body=body,
            client_ip="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            headers={},
        )

        # Structured response body with large content field (~2 MB text)
        large_content = "x" * (2 * 1024 * 1024)
        structured_body = {
            "content": large_content,
            "reasoning": "",
            "tool_calls": [],
            "finish_reason": "stop",
        }

        await logger.complete_exchange(
            conv_key=conv_key,
            exchange_num=exchange_num,
            status_code=200,
            headers={},
            body=structured_body,
            duration_ms=100.0,
        )

        exchange_file = tmp_path / conv_key / "exchange_001.json"
        with open(exchange_file) as f:
            data = json.load(f)

        # Body should be truncated (content field shortened)
        response_body = data["response"]["body"]
        assert isinstance(response_body, dict)
        assert len(response_body["content"]) < len(large_content)

    @pytest.mark.asyncio
    async def test_conversation_folder_created(self, tmp_path):
        """Conversation folder is created on first exchange."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path))
        logger = DebugLogger(config)

        body = {"messages": [{"role": "user", "content": "Hello"}]}
        conv_key, _ = await logger.start_exchange(
            body=body,
            client_ip="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            headers={},
        )

        conv_dir = tmp_path / conv_key
        assert conv_dir.exists()
        assert conv_dir.is_dir()

    @pytest.mark.asyncio
    async def test_error_status_captured(self, tmp_path):
        """Error status codes are captured correctly."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path))
        logger = DebugLogger(config)

        body = {"messages": [{"role": "user", "content": "Test"}]}
        conv_key, exchange_num = await logger.start_exchange(
            body=body,
            client_ip="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            headers={},
        )

        await logger.complete_exchange(
            conv_key=conv_key,
            exchange_num=exchange_num,
            status_code=500,
            headers={"Content-Type": "application/json"},
            body='{"error": "Internal server error"}',
            duration_ms=50.0,
        )

        exchange_file = tmp_path / conv_key / "exchange_001.json"
        with open(exchange_file) as f:
            data = json.load(f)

        assert data["response"]["status_code"] == 500


class TestCleanup:
    """Test DebugLogger cleanup of old conversation folders."""

    @pytest.mark.asyncio
    async def test_cleanup_deletes_old_folders(self, tmp_path):
        """Folders older than retention_days are deleted."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), retention_days=7)
        debug_logger = DebugLogger(config)

        # Create a conversation folder with an old exchange file
        conv_dir = tmp_path / "abc1234567890abc"
        conv_dir.mkdir()
        exchange_file = conv_dir / "exchange_001.json"
        exchange_file.write_text(json.dumps({"exchange": 1}))

        # Backdate the file by 10 days
        old_time = time.time() - (10 * 86400)
        os.utime(exchange_file, (old_time, old_time))

        # Run cleanup
        deleted, freed = await debug_logger.cleanup_old_logs()

        assert deleted == 1
        assert freed > 0
        assert not conv_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent_folders(self, tmp_path):
        """Folders within retention window are NOT deleted."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), retention_days=7)
        debug_logger = DebugLogger(config)

        # Create a conversation folder with a recent exchange file
        conv_dir = tmp_path / "def1234567890def"
        conv_dir.mkdir()
        exchange_file = conv_dir / "exchange_001.json"
        exchange_file.write_text(json.dumps({"exchange": 1}))

        # File is recent (just created)
        deleted, freed = await debug_logger.cleanup_old_logs()

        assert deleted == 0
        assert freed == 0
        assert conv_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_deletes_empty_folders(self, tmp_path):
        """Empty conversation folders are treated as old and deleted."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), retention_days=7)
        debug_logger = DebugLogger(config)

        # Create an empty conversation folder
        conv_dir = tmp_path / "empty1234567890a"
        conv_dir.mkdir()

        deleted, freed = await debug_logger.cleanup_old_logs()

        assert deleted == 1
        assert not conv_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_uses_newest_file_mtime(self, tmp_path):
        """Only the newest file's mtime determines folder age."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), retention_days=7)
        debug_logger = DebugLogger(config)

        conv_dir = tmp_path / "mixed1234567890ab"
        conv_dir.mkdir()

        # Old file
        old_file = conv_dir / "exchange_001.json"
        old_file.write_text(json.dumps({"exchange": 1}))
        old_time = time.time() - (10 * 86400)
        os.utime(old_file, (old_time, old_time))

        # Recent file
        recent_file = conv_dir / "exchange_002.json"
        recent_file.write_text(json.dumps({"exchange": 2}))

        # Folder should survive because newest file is recent
        deleted, freed = await debug_logger.cleanup_old_logs()

        assert deleted == 0
        assert conv_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_multiple_folders(self, tmp_path):
        """Cleanup correctly handles multiple folders with mixed ages."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), retention_days=7)
        debug_logger = DebugLogger(config)

        # Old folder
        old_dir = tmp_path / "old0000000000000a"
        old_dir.mkdir()
        (old_dir / "exchange_001.json").write_text(json.dumps({"exchange": 1}))
        old_time = time.time() - (10 * 86400)
        os.utime(old_dir / "exchange_001.json", (old_time, old_time))

        # Recent folder
        recent_dir = tmp_path / "recent00000000000b"
        recent_dir.mkdir()
        (recent_dir / "exchange_001.json").write_text(json.dumps({"exchange": 1}))

        deleted, freed = await debug_logger.cleanup_old_logs()

        assert deleted == 1
        assert not old_dir.exists()
        assert recent_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_skips_non_directories(self, tmp_path):
        """Non-directory entries in log_dir are ignored."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), retention_days=7)
        debug_logger = DebugLogger(config)

        # Create a file directly in log_dir (not a folder)
        stray_file = tmp_path / "readme.txt"
        stray_file.write_text("do not delete")

        deleted, freed = await debug_logger.cleanup_old_logs()

        assert deleted == 0
        assert stray_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_loop_starts_and_stops(self, tmp_path):
        """Cleanup loop starts and stops gracefully."""
        config = DebugConfig(
            enabled=True,
            log_dir=str(tmp_path),
            retention_days=7,
            cleanup_interval_hours=24,
        )
        debug_logger = DebugLogger(config)

        await debug_logger.start_cleanup_loop()
        assert hasattr(debug_logger, "_cleanup_task")
        assert debug_logger._cleanup_task is not None

        await debug_logger.stop_cleanup_loop()

    @pytest.mark.asyncio
    async def test_cleanup_disabled_does_not_start(self, tmp_path):
        """When debug is disabled, cleanup loop does not start."""
        config = DebugConfig(
            enabled=False,
            log_dir=str(tmp_path),
            retention_days=7,
            cleanup_interval_hours=24,
        )
        debug_logger = DebugLogger(config)

        await debug_logger.start_cleanup_loop()
        assert not hasattr(debug_logger, "_cleanup_task")

    @pytest.mark.asyncio
    async def test_cleanup_zero_retention_deletes_all(self, tmp_path):
        """retention_days=0 deletes all existing folders immediately."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), retention_days=0)
        debug_logger = DebugLogger(config)

        conv_dir = tmp_path / "abc1234567890abc"
        conv_dir.mkdir()
        exchange_file = conv_dir / "exchange_001.json"
        exchange_file.write_text(json.dumps({"exchange": 1}))

        # Backdate by 1 second so it's strictly before cutoff (now)
        old_time = time.time() - 1
        os.utime(exchange_file, (old_time, old_time))

        deleted, freed = await debug_logger.cleanup_old_logs()

        assert deleted == 1
        assert not conv_dir.exists()
