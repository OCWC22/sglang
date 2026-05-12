# SGLang + LMCache + InferGuard H100 Validation v0.2

## Goal
Run patched local SGLang + patched local LMCache + patched InferGuard through the Modal/H100 validation path and produce artifacts that prove whether MP observability is complete.

## Inputs
- SGLang repo: `/Users/chen/Projects/sglang`
- LMCache repo: `/Users/chen/Projects/LMCache`
- InferGuard repo: `/Users/chen/Projects/inferguard`
- SGLang validation script: `/Users/chen/Projects/sglang/scripts/ocwc22_lmcache_mp_h100_modal.py`

## Required Test
Launch LMCache MP server and SGLang with LMCache MP, send real requests to a small model, scrape SGLang and LMCache metrics, then run InferGuard against captured metrics.

## Required Artifacts
- Modal app/run URL
- artifact directory
- SGLang metrics file
- LMCache metrics file
- SGLang logs
- LMCache logs
- InferGuard JSON report
- request receipt

## Acceptance
Accepted only if InferGuard reports:
- expected mode `mp`
- detected mode `mp`
- SGLang MP launch/connector evidence present
- required MP family breakdown complete:
  - storage_manager
  - lookup_tokens
  - l1_counters
  - l1_memory
- no mode mismatch
- no missing required MP families

## If Blocked
Return exact failing command, log tail, and artifact path. Do not call it 100% if blocked or mixed.
