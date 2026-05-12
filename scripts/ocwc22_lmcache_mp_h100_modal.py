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
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "protobuf-compiler")
    .pip_install("requests", "openai", "lmcache")
    .add_local_dir(
        "/Users/chen/Projects/sglang",
        remote_path="/root/sglang",
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
    env.update({"LMCACHE_LOG_LEVEL": "INFO", "PYTHONUNBUFFERED": "1"})

    lmcache_log = open(out_dir / "lmcache.log", "w")
    sglang_log = open(out_dir / "sglang.log", "w")

    lmcache_cmd = [
        "lmcache",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        "6555",
        "--prometheus-port",
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
        procs.append(subprocess.Popen(lmcache_cmd, stdout=lmcache_log, stderr=subprocess.STDOUT, env=env, cwd="/root/sglang"))
        time.sleep(10)
        procs.append(subprocess.Popen(sglang_cmd, stdout=sglang_log, stderr=subprocess.STDOUT, env=env, cwd="/root/sglang"))

        deadline = time.time() + 1800
        while time.time() < deadline:
            try:
                if requests.get("http://127.0.0.1:30000/health", timeout=5).ok:
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise RuntimeError("SGLang health endpoint did not become ready")

        payloads = [
            {"model": model_path, "messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 16},
            {"model": model_path, "messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 16},
        ]
        responses = []
        for payload in payloads:
            r = requests.post("http://127.0.0.1:30000/v1/chat/completions", json=payload, timeout=120)
            responses.append({"status_code": r.status_code, "body": r.text[:4000]})
            r.raise_for_status()

        (out_dir / "request_replay.json").write_text(json.dumps(responses, indent=2))
        (out_dir / "sglang_metrics.prom").write_text(requests.get("http://127.0.0.1:30000/metrics", timeout=30).text)
        (out_dir / "lmcache_metrics.prom").write_text(requests.get("http://127.0.0.1:9090/metrics", timeout=30).text)
        (out_dir / "environment.json").write_text(json.dumps({"model_path": model_path, "lmcache_cmd": lmcache_cmd, "sglang_cmd": sglang_cmd, "run_id": run_id}, indent=2))

        inferguard_report = out_dir / "inferguard_observability_coverage.json"
        cmd = [
            "python",
            "-m",
            "inferguard",
            "observability-coverage",
            "--engine-metrics-file",
            str(out_dir / "sglang_metrics.prom"),
            "--lmcache-metrics-file",
            str(out_dir / "lmcache_metrics.prom"),
            "--expected-engine",
            "sglang",
            "--expect-lmcache-mode",
            "mp",
            "--output",
            str(inferguard_report),
            "--json",
        ]
        subprocess.run(cmd, cwd=inferguard_dir, check=True, env=env)

        return {"status": "measured", "artifact_dir": str(out_dir), "inferguard_report": str(inferguard_report)}
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
