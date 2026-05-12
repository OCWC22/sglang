# SGLang LMCache MP Mode Evidence v0.1

Date: 2026-05-12
Scope: `/Users/chen/Projects/sglang` only

## Status

Implemented targeted SGLang-side launch/mode evidence and minimal lifecycle observability for LMCache MP compatibility. This does not claim live H100 support or clean MP acceptance.

## Changes

- Hardened LMCache mode labels in `python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py`:
  - Added explicit `enable_lmcache="true|false"` label to SGLang LMCache metrics.
  - Kept `lmcache_mp_host` and `lmcache_mp_port` labels.
  - Tied `lmcache_mode="mp"` to actual `LMCacheMPLayerwiseConnector` selection, not merely host/port request.
  - Fallback to `LMCacheLayerwiseConnector` now reports `lmcache_mode="embedded"`.
- Added LMCache lifecycle observability counters in `python/sglang/srt/observability/metrics_collector.py`:
  - `sglang:lmcache_release_calls_total{reason=...}`
  - `sglang:lmcache_abort_calls_total{reason=...}`
  - `sglang:lmcache_errors_total{operation=...}`
- Added best-effort LMCache request cleanup hooks in `LMCRadixCache`:
  - finish path releases with `reason="finish"` after store.
  - non-insert release path releases with `reason="abort"`.
  - reset path releases with `reason="reset"`.
  - store/load errors increment error counters; store errors also attempt lifecycle release with `reason="error"`.
  - connector cleanup method probing is defensive across candidate LMCache connector APIs.
- Strengthened focused helper tests in `test/registered/unit/mem_cache/test_lmcache_mp_helpers.py` for mode labels, endpoint labels, and abort lifecycle counters.

## Tests / checks run

Passed:

```bash
python3 -m pytest test/registered/unit/mem_cache/test_lmcache_mp_helpers.py -q
# 5 passed, 1 warning

python3 -m py_compile \
  python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py \
  python/sglang/srt/observability/metrics_collector.py \
  test/registered/unit/mem_cache/test_lmcache_mp_helpers.py

uv run ruff check \
  python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py \
  python/sglang/srt/observability/metrics_collector.py \
  test/registered/unit/mem_cache/test_lmcache_mp_helpers.py
# All checks passed!
```

Blocked locally:

```bash
python -m pytest ...
# blocked: `python` executable unavailable

PYTHONPATH=python python3 -m pytest test/registered/unit/server_args/test_server_args.py::TestLMCacheArgs ...
# blocked: local Python env missing `orjson`

python3 -m ruff check ...
# blocked: local Python env missing `ruff`
```

## Remaining required work not completed in this pass

The following are still too broad for this targeted pass and require dedicated scheduler/LMCache integration tests:

1. Exact abort/preempt/timeout cleanup wiring across all scheduler paths, including queued abort, running abort, waiting timeout, and decode retraction/preemption.
2. Verification that LMCache daemon state has no leaked sessions/read locks/load markers after abort/preempt/timeout loops.
3. Deferred cold-L2 lookup support so LMCache MP lookup/prefetch can return pending without blocking scheduler ticks.
4. Live H100 artifact validation with SGLang `/metrics`, LMCache MP `/metrics`, status before/after, KV events, and InferGuard report.

## Support statement

Current support evidence remains source-backed/partial. Do not report `clean_mp_accepted` or live H100 compatibility from these changes alone.
