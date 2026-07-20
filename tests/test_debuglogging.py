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


class TestConversationEviction:
    """Test LRU eviction of in-memory conversation keys."""

    @pytest.mark.asyncio
    async def test_eviction_removes_oldest_conversation(self, tmp_path):
        """Oldest (least recently used) conversation is evicted when limit is reached."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), max_conversations=3)
        debug_logger = DebugLogger(config)

        bodies = [
            {"messages": [{"role": "user", "content": f"Message {i}"}]}
            for i in range(4)
        ]

        # Add 4 conversations — exceeds limit of 3
        keys = []
        for body in bodies:
            conv_key, _ = await debug_logger.start_exchange(
                body=body,
                client_ip="127.0.0.1",
                method="POST",
                path="/v1/chat/completions",
                headers={},
            )
            keys.append(conv_key)

        # Should have exactly 3 conversations in memory
        assert len(debug_logger._exchanges) == 3
        assert len(debug_logger._counters) == 3

        # Oldest key should have been evicted
        assert keys[0] not in debug_logger._exchanges
        assert keys[1] in debug_logger._exchanges
        assert keys[2] in debug_logger._exchanges
        assert keys[3] in debug_logger._exchanges

    @pytest.mark.asyncio
    async def test_eviction_respects_lru_order(self, tmp_path):
        """Recently accessed conversations are kept, even if they're not the newest."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), max_conversations=2)
        debug_logger = DebugLogger(config)

        body_a = {"messages": [{"role": "user", "content": "Conversation A"}]}
        body_b = {"messages": [{"role": "user", "content": "Conversation B"}]}
        body_c = {"messages": [{"role": "user", "content": "Conversation C"}]}

        # Add A and B
        key_a, _ = await debug_logger.start_exchange(
            body=body_a, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )
        key_b, _ = await debug_logger.start_exchange(
            body=body_b, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )

        # Access A again to mark it as recently used
        _, _ = await debug_logger.start_exchange(
            body=body_a, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )

        # Add C — should evict B (least recently used), not A
        key_c, _ = await debug_logger.start_exchange(
            body=body_c, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )

        assert len(debug_logger._exchanges) == 2
        assert key_a in debug_logger._exchanges  # Recently accessed
        assert key_b not in debug_logger._exchanges  # Evicted (LRU)
        assert key_c in debug_logger._exchanges  # Newest

    @pytest.mark.asyncio
    async def test_eviction_cleans_up_locks(self, tmp_path):
        """Evicted conversations have their locks cleaned up too."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), max_conversations=2)
        debug_logger = DebugLogger(config)

        body_a = {"messages": [{"role": "user", "content": "A"}]}
        body_b = {"messages": [{"role": "user", "content": "B"}]}
        body_c = {"messages": [{"role": "user", "content": "C"}]}

        key_a, _ = await debug_logger.start_exchange(
            body=body_a, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )
        key_b, _ = await debug_logger.start_exchange(
            body=body_b, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )

        # Both locks exist
        assert key_a in debug_logger._locks
        assert key_b in debug_logger._locks

        # Add C — evicts A
        _, _ = await debug_logger.start_exchange(
            body=body_c, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )

        # A's lock should be cleaned up
        assert key_a not in debug_logger._locks

    @pytest.mark.asyncio
    async def test_no_eviction_under_limit(self, tmp_path):
        """Conversations under the limit are never evicted."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), max_conversations=10)
        debug_logger = DebugLogger(config)

        bodies = [
            {"messages": [{"role": "user", "content": f"Msg {i}"}]}
            for i in range(5)
        ]

        keys = []
        for body in bodies:
            conv_key, _ = await debug_logger.start_exchange(
                body=body, client_ip="127.0.0.1", method="POST",
                path="/v1/chat/completions", headers={},
            )
            keys.append(conv_key)

        # All 5 should still be present
        assert len(debug_logger._exchanges) == 5
        for key in keys:
            assert key in debug_logger._exchanges

    @pytest.mark.asyncio
    async def test_eviction_at_exact_limit(self, tmp_path):
        """Adding a conversation at exactly the limit triggers eviction."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), max_conversations=2)
        debug_logger = DebugLogger(config)

        body_a = {"messages": [{"role": "user", "content": "A"}]}
        body_b = {"messages": [{"role": "user", "content": "B"}]}

        key_a, _ = await debug_logger.start_exchange(
            body=body_a, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )
        key_b, _ = await debug_logger.start_exchange(
            body=body_b, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )

        # At limit — both present
        assert len(debug_logger._exchanges) == 2

        # Reuse A (touching it) — no new key, no eviction
        _, _ = await debug_logger.start_exchange(
            body=body_a, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )
        assert len(debug_logger._exchanges) == 2
        assert key_a in debug_logger._exchanges
        assert key_b in debug_logger._exchanges

    @pytest.mark.asyncio
    async def test_eviction_prevents_unbounded_growth(self, tmp_path):
        """Memory stays bounded even with many unique conversations."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), max_conversations=50)
        debug_logger = DebugLogger(config)

        # Simulate 200 unique conversations
        for i in range(200):
            body = {"messages": [{"role": "user", "content": f"Unique message {i}"}]}
            await debug_logger.start_exchange(
                body=body, client_ip="127.0.0.1", method="POST",
                path="/v1/chat/completions", headers={},
            )

        # Should never exceed max_conversations
        assert len(debug_logger._exchanges) == 50
        assert len(debug_logger._counters) == 50

    @pytest.mark.asyncio
    async def test_complete_exchange_touches_conversation(self, tmp_path):
        """Completing an exchange marks the conversation as recently used."""
        config = DebugConfig(enabled=True, log_dir=str(tmp_path), max_conversations=2)
        debug_logger = DebugLogger(config)

        body_a = {"messages": [{"role": "user", "content": "A"}]}
        body_b = {"messages": [{"role": "user", "content": "B"}]}
        body_c = {"messages": [{"role": "user", "content": "C"}]}

        # Add A and B
        key_a, num_a = await debug_logger.start_exchange(
            body=body_a, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )
        key_b, num_b = await debug_logger.start_exchange(
            body=body_b, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )

        # Complete A — should touch it
        await debug_logger.complete_exchange(
            conv_key=key_a, exchange_num=num_a, status_code=200,
            headers={}, body="{}", duration_ms=100.0,
        )

        # Add C — should evict B (now LRU), not A (just completed)
        _, _ = await debug_logger.start_exchange(
            body=body_c, client_ip="127.0.0.1", method="POST",
            path="/v1/chat/completions", headers={},
        )

        assert key_a in debug_logger._exchanges
        assert key_b not in debug_logger._exchanges
