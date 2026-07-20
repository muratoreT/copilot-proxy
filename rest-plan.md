## REMAINING ISSUES (not yet fixed)

### 1. Global Mutable State (Module-Level Singletons)

**Severity: High**

**Location:** metrics.py (~line 118), config.py (lines 14-17), proxy.py (lines 22-26)

**Problem:** Multiple modules expose mutable global state — `metrics = MetricsTracker()`, `_current_config`/`_config_lock`/`_config_path`, and `_http_client`/`_config_watcher`/`_config`/`_debug_logger`.

**Why it Matters:** Makes testing difficult (requires patching module-level variables), creates hidden cross-module dependencies, and prevents running multiple proxy instances in the same process.

**Recommendation:** Inject dependencies via FastAPI `app.state` or `Depends()`. Store `metrics` as `app.state.metrics`. Access config through lifespan-managed state rather than module globals.

---

### 5. `TimingMiddleware` — Missing `call_next` Type Annotation

**Severity: Low**

**Location:** middleware.py, line 23

**Problem:** `call_next` parameter has no type annotation.

**Recommendation:** Annotate as `CallNextAsync` from `starlette.types`.

---

### 7. config.py — `get_config()` Returns a Reference, Not a Copy

**Severity: Medium**

**Location:** config.py, `get_config()`

**Problem:** Returns the live `_current_config` object. Callers can mutate it, corrupting shared state.

**Recommendation:** Return `copy.copy(_current_config)` or make `ProxyConfig` frozen.

---

### 9. main.py — Re-scanning on Every Request

**Severity: Medium**

**Location:** main.py — all three route handlers

**Problem:** Every route calls `_scan_conversations(LOG_DIR)` reading all exchange JSON files. O(N) file reads per HTTP request.

**Recommendation:** Cache scan results with a TTL (e.g., 30 seconds).

---

### 10. main.py — `import re` Inside Function

**Severity: Low**

**Location:** main.py, `_extract_first_user_message()`, line ~82

**Recommendation:** Move `import re` to module level.

---

### 11. main.py — Mutable Module-Level `LOG_DIR`

**Severity: Medium**

**Location:** main.py, line 23 and `main()` line ~168

**Problem:** `LOG_DIR` reassigned via `global` in `main()`. Fragile — default path used if `main()` never called.

**Recommendation:** Pass `log_dir` via `Depends()` or a closure.

---

### 12. proxy.py — Hardcoded `"config.yaml"` Path

**Severity: Medium**

**Location:** proxy.py (lifespan), proxy_entry.py (line 21)

**Problem:** Config path hardcoded in two places.

**Recommendation:** Accept as CLI argument or environment variable, pass to `create_app()`.

---

### 13. proxy.py — `setup_logging()` Called Inside `create_app()`

**Severity: Medium**

**Location:** proxy.py, `create_app()`, line ~85

**Problem:** Global logging configuration called from a factory function. Side effect in a factory.

**Recommendation:** Call `setup_logging()` in proxy_entry.py before `create_app()`.

---

### 15. rewrite.py — `rewrite_request` Always Does `copy.deepcopy`

**Severity: Low**

**Location:** rewrite.py, line ~149

**Problem:** Deep copy called unconditionally even when no rewriting is configured.

**Recommendation:** Check config flags first, then copy only if needed.

---

### 17. proxy.py — Response Latency Logged as 0

**Severity: Medium**

**Location:** proxy.py, `proxy_request()`, line ~222

**Problem:** `log_response()` called with `latency_ms=0`. The parameter is never populated.

**Recommendation:** Calculate latency in the route handler or remove the parameter.

---

### 20. Test Coverage Gaps

**Severity: Medium**

**Problem:** No tests for struct_logging.py, metrics.py, middleware.py, `ConfigWatcher`, streaming retry end-to-end, or proxy_entry.py signal handling.

**Recommendation:** Add unit tests for untested modules and integration tests for retry flow.
