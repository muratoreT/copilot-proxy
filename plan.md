
## Plan: Hybrid Streaming Retry Mode

**TL;DR:** Add `streaming_mode: hybrid` to `StreamingRetryConfig`. Buffers chunks only for a short timeout window (default 1s). If meaningful content arrives within the window, flush and continue streaming immediately. If timeout fires with no content, discard buffer and retry upstream. Near-instant streaming for normal responses, still catches empty/thinking-only completions.

---

### Steps

**Phase 1 — Config schema** (models.py)
1. Add `streaming_mode: Literal["buffered", "hybrid"]` — default `"hybrid"`
2. Add `empty_detection_timeout_ms: int` — default 1000, range 200–10000

**Phase 2 — Helpers** (streaming.py)
3. Add `_chunk_has_content(chunk) -> bool` — per-chunk content detection
4. Add `_chunk_has_thinking(chunk) -> bool` — per-chunk thinking detection

**Phase 3 — Core logic** (streaming.py)
5. Refactor `generate()` in `_handle_streaming_response()`:
   - `"buffered"` → existing logic unchanged
   - `"hybrid"` → buffer during timeout window, flush on content detected, retry on timeout+empty, passthrough after window

**Phase 4 — Config example** (config.yaml.example)
6. Add new fields to example config

**Phase 5 — Tests**
7. Add `TestHybridStreaming*` test classes to test_streaming.py
8. Add config validation test to test_config.py

**Phase 6 — Docs** (README.md)
9. Document new fields, hybrid behavior, and trade-offs

---

### Decisions Recap

| #   | Decision                    | Choice                                 |
| --- | --------------------------- | -------------------------------------- |
| 1   | Mode selection              | Explicit `streaming_mode` field        |
| 2   | Timeout behavior            | Let original stream drain silently     |
| 3   | Already-sent chunks         | Buffer until timeout, flush or discard |
| 4   | Timeout value               | 1000ms default                         |
| 5   | Thinking-only in hybrid     | Two-phase buffer during timeout        |
| 6   | Default mode                | `"hybrid"`                             |
| 7   | Thinking-only after timeout | Accept limitation                      |

---

### Verification
1. `pytest tests/ -v` — full suite passes
2. config.yaml.example loads without errors
3. Manual: tokens appear within ~1s for normal streams
4. Manual: retry triggers for empty completions (check logs)

Ready for your review — any changes before implementation?