"""Modal H100 validation for SGLang + LMCache multiprocess telemetry.

Run from the SGLang checkout after `modal setup`:

    modal run scripts/ocwc22_lmcache_mp_h100_modal.py \
      --model-path Qwen/Qwen2.5-0.5B-Instruct \
      --inferguard-dir /root/inferguard

The job launches an LMCache MP server and SGLang with
`--enable-lmcache --lmcache-mp-host --lmcache-mp-port`, sends a small OpenAI
chat-completions workload, captures SGLang and LMCache metrics/logs, writes an
environment receipt, and invokes InferGuard acceptance when available in the
image/container path.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import time
from datetime import datetime, timezone

import modal

APP_NAME = "ocwc22-sglang-lmcache-mp-h100-telemetry"
ARTIFACT_DIR = pathlib.Path("/artifacts/ocwc22_lmcache_mp")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "protobuf-compiler", "libnuma1")
    .pip_install("requests", "openai", "lmcache")
    .add_local_dir(
        "/Users/chen/Projects/sglang",
        remote_path="/root/sglang",
        copy=True,
        ignore=[".git", ".git/**", "__pycache__", "**/__pycache__/**", ".venv", ".venv/**"],
    )
    .add_local_dir(
        "/Users/chen/Projects/LMCache",
        remote_path="/root/LMCache",
        copy=True,
        ignore=[".git", ".git/**", "__pycache__", "**/__pycache__/**", ".venv", ".venv/**"],
    )
    .add_local_dir(
        "/Users/chen/Projects/inferguard/src",
        remote_path="/root/inferguard/src",
        copy=True,
        ignore=["__pycache__", "**/__pycache__/**"],
    )
    .add_local_file(
        "/Users/chen/Projects/inferguard/pyproject.toml",
        remote_path="/root/inferguard/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        "/Users/chen/Projects/inferguard/README.md",
        remote_path="/root/inferguard/README.md",
        copy=True,
    )
    .add_local_file(
        "/Users/chen/Projects/inferguard/LICENSE",
        remote_path="/root/inferguard/LICENSE",
        copy=True,
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0+local TORCH_CUDA_ARCH_LIST=9.0 CC=gcc CXX=g++ PATH=/root/.cargo/bin:$PATH pip install -e /root/LMCache",
        "PATH=/root/.cargo/bin:$PATH pip install -e /root/sglang/python",
        "pip install -e /root/inferguard",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="H100", timeout=60 * 60, volumes={"/artifacts": modal.Volume.from_name("ocwc22-lmcache-mp-artifacts", create_if_missing=True)})
def run(model_path: str, inferguard_dir: str = "/root/inferguard") -> dict:
    import requests

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "LMCACHE_LOG_LEVEL": "INFO",
            "PYTHONUNBUFFERED": "1",
        }
    )

    lmcache_log_path = out_dir / "lmcache.log"
    sglang_log_path = out_dir / "sglang.log"
    lmcache_log = open(lmcache_log_path, "w")
    sglang_log = open(sglang_log_path, "w")
    lmcache_http_base_url = "http://127.0.0.1:9090"
    lmcache_metrics_url = f"{lmcache_http_base_url}/metrics"

    def _tail(path: pathlib.Path, limit: int = 4000) -> str:
        if not path.exists():
            return "<missing>"
        return path.read_text(errors="replace")[-limit:]

    def _wait_http_ok(name: str, url: str, timeout_s: int, log_path: pathlib.Path) -> None:
        deadline = time.time() + timeout_s
        last_error = ""
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=5)
                if r.ok:
                    return
                last_error = f"HTTP {r.status_code}: {r.text[:500]}"
            except Exception as exc:
                last_error = repr(exc)
            time.sleep(2)
        raise RuntimeError(
            f"{name} did not become ready at {url} within {timeout_s}s; last_error={last_error}. "
            f"Log tail:\n{_tail(log_path)}"
        )

    lmcache_cmd = [
        "lmcache",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        "6555",
        "--prometheus-port",
        "9090",
        "--http-host",
        "127.0.0.1",
        "--http-port",
        "9090",
        "--l1-size-gb",
        "16",
        "--eviction-policy",
        "LRU",
    ]
    sglang_cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--enable-metrics",
        "--enable-lmcache",
        "--lmcache-mp-host",
        "127.0.0.1",
        "--lmcache-mp-port",
        "6555",
    ]

    procs = []
    try:
        print(f"artifact_dir={out_dir}", flush=True)
        print(f"starting LMCache: {' '.join(lmcache_cmd)}", flush=True)
        procs.append(subprocess.Popen(lmcache_cmd, stdout=lmcache_log, stderr=subprocess.STDOUT, env=env, cwd="/root/sglang"))
        time.sleep(10)
        print(f"starting SGLang: {' '.join(sglang_cmd)}", flush=True)
        procs.append(subprocess.Popen(sglang_cmd, stdout=sglang_log, stderr=subprocess.STDOUT, env=env, cwd="/root/sglang"))

        deadline = time.time() + 600
        last_progress = 0.0
        while time.time() < deadline:
            for name, proc, log_file, log_path in (
                ("LMCache", procs[0], lmcache_log, lmcache_log_path),
                ("SGLang", procs[1], sglang_log, sglang_log_path),
            ):
                rc = proc.poll()
                if rc is not None:
                    log_file.flush()
                    raise RuntimeError(f"{name} exited before SGLang health was ready (rc={rc}). Log tail:\n{_tail(log_path)}")
            try:
                if requests.get("http://127.0.0.1:30000/health", timeout=5).ok:
                    break
            except Exception:
                pass
            if time.time() - last_progress > 30:
                lmcache_log.flush()
                sglang_log.flush()
                print(
                    f"waiting for SGLang /health; log_bytes: lmcache={lmcache_log_path.stat().st_size} sglang={sglang_log_path.stat().st_size}",
                    flush=True,
                )
                last_progress = time.time()
            time.sleep(5)
        else:
            lmcache_log.flush()
            sglang_log.flush()
            raise RuntimeError(
                "SGLang health endpoint did not become ready within 600s. "
                f"sglang_log_tail:\n{_tail(sglang_log_path)}"
            )

        lmcache_log.flush()
        _wait_http_ok("LMCache Prometheus metrics endpoint", lmcache_metrics_url, 120, lmcache_log_path)

        shared_prefix = " ".join(
            f"Shared cache validation sentence {i:04d}: LMCache MP should store and retrieve this repeated prefix."
            for i in range(420)
        )
        payloads = [
            {
                "model": model_path,
                "messages": [{"role": "user", "content": f"{shared_prefix}\n\nQuestion A: answer with the word alpha."}],
                "max_tokens": 16,
            },
            {
                "model": model_path,
                "messages": [{"role": "user", "content": f"{shared_prefix}\n\nQuestion B: answer with the word beta."}],
                "max_tokens": 16,
            },
            {
                "model": model_path,
                "messages": [{"role": "user", "content": f"{shared_prefix}\n\nQuestion C: answer with the word gamma."}],
                "max_tokens": 16,
            },
        ]
        responses = []
        for idx, payload in enumerate(payloads):
            r = requests.post("http://127.0.0.1:30000/v1/chat/completions", json=payload, timeout=180)
            responses.append({"request_index": idx, "status_code": r.status_code, "body": r.text[:4000]})
            r.raise_for_status()
            time.sleep(5)

        time.sleep(15)
        (out_dir / "request_replay.json").write_text(json.dumps(responses, indent=2))
        (out_dir / "sglang_metrics.prom").write_text(requests.get("http://127.0.0.1:30000/metrics", timeout=30).text)

        lmcache_metrics_text = ""
        for scrape_idx in range(6):
            lmcache_metrics_response = requests.get(lmcache_metrics_url, timeout=30)
            if not lmcache_metrics_response.ok:
                raise RuntimeError(
                    f"LMCache metrics scrape failed: HTTP {lmcache_metrics_response.status_code}. "
                    f"Body: {lmcache_metrics_response.text[:1000]}\nLMCache log tail:\n{_tail(lmcache_log_path)}"
                )
            lmcache_metrics_text = lmcache_metrics_response.text
            (out_dir / f"lmcache_metrics_scrape_{scrape_idx}.prom").write_text(lmcache_metrics_text)
            time.sleep(5)
        (out_dir / "lmcache_metrics.prom").write_text(lmcache_metrics_text)

        lmcache_mp_samples = []
        for line in lmcache_metrics_text.splitlines():
            if line.startswith("#") or not line.startswith("lmcache_mp_"):
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            name = parts[0].split("{", 1)[0]
            try:
                value = float(parts[1])
            except ValueError:
                continue
            lmcache_mp_samples.append({"name": name, "sample": parts[0], "value": value, "nonzero": value != 0.0})
        lmcache_mp_samples.sort(key=lambda row: (not row["nonzero"], row["name"], row["sample"]))
        (out_dir / "lmcache_mp_metric_samples.json").write_text(json.dumps(lmcache_mp_samples, indent=2))
        print("lmcache_mp_metric_names=" + json.dumps(sorted({row["name"] for row in lmcache_mp_samples})), flush=True)
        print(
            "lmcache_mp_nonzero_samples="
            + json.dumps([row for row in lmcache_mp_samples if row["nonzero"]][:200]),
            flush=True,
        )
        (out_dir / "environment.json").write_text(json.dumps({"model_path": model_path, "lmcache_cmd": lmcache_cmd, "sglang_cmd": sglang_cmd, "run_id": run_id, "lmcache_http_base_url": lmcache_http_base_url, "lmcache_metrics_url": lmcache_metrics_url, "workload": {"shared_prefix_sentences": 420, "requests": len(payloads), "radix_cache_disabled": False}}, indent=2))

        inferguard_packet_dir = out_dir / "inferguard_lmcache_packet"
        inferguard_report = inferguard_packet_dir / "observability_coverage_report.json"
        cmd = [
            "inferguard",
            "collect-lmcache",
            "--output-dir",
            str(inferguard_packet_dir),
            "--engine-metrics-file",
            str(out_dir / "sglang_metrics.prom"),
            "--lmcache-metrics-file",
            str(out_dir / "lmcache_metrics.prom"),
            "--lmcache-http-base-url",
            lmcache_http_base_url,
            "--engine-log-file",
            str(sglang_log_path),
            "--lmcache-log-file",
            str(lmcache_log_path),
            "--expected-engine",
            "sglang",
            "--expect-mode",
            "mp",
            "--mp-prometheus-port",
            "9090",
            "--json",
        ]
        subprocess.run(cmd, cwd=inferguard_dir, check=True, env=env)

        return {"status": "measured", "artifact_dir": str(out_dir), "inferguard_report": str(inferguard_report), "inferguard_packet_dir": str(inferguard_packet_dir)}
    finally:
        for proc in reversed(procs):
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        lmcache_log.close()
        sglang_log.close()


@app.local_entrypoint()
def main(model_path: str = "Qwen/Qwen2.5-0.5B-Instruct", inferguard_dir: str = "/root/inferguard"):
    print(run.remote(model_path, inferguard_dir))
