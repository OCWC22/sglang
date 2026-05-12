# OCWC-22 SGLang + LMCache MP H100 Telemetry Implementation Spec v0.1

## Objective
Fork/maintain the local OCWC-22 SGLang branch and implement real SGLang-side LMCache multiprocess telemetry, then validate end-to-end with InferGuard on Modal H100 so telemetry is observed between SGLang and LMCache.

## Repos
- SGLang local checkout: `/Users/chen/Projects/sglang`
- InferGuard local checkout: `/Users/chen/Projects/inferguard`
- LMCache local checkout, if needed for source comparison: `/Users/chen/Projects/LMCache`

## Required behavior
1. Work in the SGLang checkout on a dedicated OCWC-22 branch/fork lane, preserving unrelated untracked files.
2. Do not merely update InferGuard fixtures. Make SGLang emit or expose telemetry relevant to LMCache MP KV transfer/allocation path.
3. Use source-backed contracts from current upstream SGLang/LMCache MP support:
   - SGLang PR #24089: `--enable-lmcache`, `--lmcache-mp-host`, `--lmcache-mp-port` and MP connector selection.
   - LMCache PR #3166: SGLang MP adapter and LMCache MP path.
4. Add tests in SGLang for CLI/connector/metrics behavior before implementation where practical.
5. Produce an H100 validation path that launches LMCache MP server and SGLang on Modal H100, generates requests, captures SGLang metrics/logs plus LMCache metrics/logs, and runs InferGuard acceptance against those artifacts.
6. Update InferGuard only if the SGLang telemetry contract requires parser/report changes. Keep InferGuard claims evidence-gated.

## Evidence gates
- `source_backed`: implementation cites source files/PR contracts.
- `fixture_tested`: local SGLang and InferGuard tests pass on synthetic/unit paths.
- `measured`: only after Modal H100 artifacts exist and InferGuard accepts them.

## Non-claims
- Do not claim production support unless tests and H100 artifacts prove the path.
- Do not claim performance improvement. This is observability/telemetry validation only.
- Do not claim upstream merged support unless remote source shows it merged.

## Expected artifacts
- SGLang commit hash and branch name.
- InferGuard commit hash if changed.
- Modal H100 job/run ID or URL.
- Raw artifact directory paths for SGLang logs/metrics, LMCache logs/metrics, request replay, environment receipt.
- InferGuard report path showing accepted SGLang + LMCache MP telemetry.
- Clear status: blocked/fixture_tested/measured.

## Guardrails
- Preserve unrelated dirty files: `/Users/chen/Projects/sglang/modal_sglang_pd_test.py`, `/Users/chen/Projects/sglang/python/uv.lock`, and any existing InferGuard dirty files not in scope.
- Commit/push only explicitly scoped files.
- If H100 execution is blocked by credentials, branch availability, dependency failure, or upstream instability, record exact blocker and leave runnable scripts/prompts in repo.
