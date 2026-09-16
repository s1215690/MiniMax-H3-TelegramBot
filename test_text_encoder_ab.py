"""A/B the MiniMax-H3 text encoder: INT8 ConvRot (24.55GB) vs Heretic NVFP4 (14.61GB).

Both are the same uncensored/heretic model; the NVFP4 file is a re-quantization of
the INT8 one, 40% smaller, and is small enough to sit inside a single 20GB card
instead of being streamed in by comfy-aimdo.

Method: for each encoder, restart ComfyUI (so nothing is cached), run once to
measure the cold path that includes the encoder load, then run again to measure
the warm path. The generator config is the Bot's production one - two-stage
latent upscaling OFF, as Start-MiniMax-H3-Telegram.cmd sets it.

The point of the exercise is the hardware note from the quantizer: NVFP4 is native
on Blackwell and *emulated* on Ampere, and this machine's RTX 3080 reports nvfp4
under "emulated ops". So the question is whether the emulated path is fast enough
to be worth the 10GB saving.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMFY = "http://127.0.0.1:8191"
LOG = Path(r"E:\MiniMax-H3-Telegram\runtime\bot\comfyui.log")

INT8 = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
NVFP4 = "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"

# Bot production settings: no two-stage latent upscale.
os.environ["MINIMAX_H3_LATENT_UPSCALE"] = "0"

# Config is overridable so the same harness can measure both a trivial job (where
# fixed per-run overheads dominate) and a realistic one (where sampling does).
GEN_CONFIG = os.environ.get("AB_CONFIG", "448 256 4 2").split()


def load_bot():
    spec = importlib.util.spec_from_file_location("h3bot_ab", HERE / "MiniMax-H3-Telegram-Bot.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3bot_ab"] = mod
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def submit(mod, clip_name: str, tag: str) -> str:
    mod.CLIP_NAME = clip_name
    cfg = mod.parse_config(GEN_CONFIG)
    wf = mod.build_workflow(cfg, "A cat sitting on a windowsill in the afternoon sun.", f"AB_{tag}")
    assert wf["3"]["inputs"]["clip_name"] == clip_name, wf["3"]["inputs"]
    req = urllib.request.Request(
        f"{COMFY}/prompt",
        data=json.dumps({"prompt": wf, "client_id": f"dsh-ab-{tag}"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        body = json.loads(r.read().decode())
    if body.get("node_errors"):
        raise SystemExit(f"node_errors: {body['node_errors']}")
    return body["prompt_id"]


def gpu_peak(seconds: float, interval: float = 1.0) -> tuple[int, int]:
    """Poll nvidia-smi for `seconds`; return (peak single-GPU MiB, peak sum MiB)."""
    import subprocess
    import threading

    state = {"peak": 0, "peak_sum": 0, "stop": False, "samples": 0}

    def poll():
        while not state["stop"]:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                vals = [int(v.strip()) for v in out.splitlines() if v.strip().isdigit()]
                if vals:
                    state["peak"] = max(state["peak"], max(vals))
                    state["peak_sum"] = max(state["peak_sum"], sum(vals))
                    state["samples"] += 1
            except Exception:
                pass
            time.sleep(interval)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    time.sleep(seconds)
    state["stop"] = True
    t.join(timeout=5)
    return state["peak"], state["peak_sum"]


def wait_for(prompt_id: str, timeout_s: int = 900):
    """Wait for the job, sampling GPU memory for its whole duration."""
    import subprocess
    import threading

    state = {"peak": 0, "peak_sum": 0, "stop": False, "samples": 0}

    def poll():
        while not state["stop"]:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                vals = [int(v.strip()) for v in out.splitlines() if v.strip().isdigit()]
                if vals:
                    state["peak"] = max(state["peak"], max(vals))
                    state["peak_sum"] = max(state["peak_sum"], sum(vals))
                    state["samples"] += 1
            except Exception:
                pass
            time.sleep(1.5)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    start = time.time()
    result = ("TIMEOUT", None)
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(f"{COMFY}/history", timeout=15) as r:
                hist = json.loads(r.read().decode())
        except Exception:
            result = ("CRASHED", None)
            break
        if prompt_id in hist:
            rec = hist[prompt_id]
            result = (rec.get("status", {}).get("status_str", "?"), time.time() - start)
            break
        time.sleep(8)
    state["stop"] = True
    t.join(timeout=5)
    return result[0], result[1], state["peak"], state["peak_sum"], state["samples"]


def log_tail_marker() -> int:
    return LOG.stat().st_size if LOG.exists() else 0


def log_since(marker: int) -> str:
    if not LOG.exists():
        return ""
    with LOG.open("rb") as f:
        f.seek(marker)
        return f.read().decode("utf-8", errors="replace")


def staged_size(text: str) -> str:
    for line in reversed(text.splitlines()):
        if "MiniMaxH3TEModel_" in line and "Staged" in line:
            return line.strip()
        if "nvfp4" in line.lower() and ("emulated" in line.lower() or "Native ops" in line):
            return line.strip()
    return ""


def restart_comfy(mod):
    try:
        mod.stop_comfyui_process()
    except Exception:
        pass
    time.sleep(5)
    msg = mod.start_comfyui_process(mod.DEFAULT_COMFYUI_VRAM_MODE)
    deadline = time.time() + 420
    while time.time() < deadline:
        if mod.comfyui_is_online():
            return True
        time.sleep(5)
    return False


def main() -> int:
    mod = load_bot()
    print(f"latent_upscale enabled = {mod.LATENT_UPSCALE_ENABLED} (production: False)")
    print(f"ComfyUI = {mod.COMFYUI_DIR}")
    print()

    results = {}
    for label, clip in (("INT8 24.55GB", INT8), ("NVFP4 14.61GB", NVFP4)):
        print("=" * 72)
        print(f"{label}  ({clip})")
        print("=" * 72)
        if not restart_comfy(mod):
            print("  ComfyUI restart FAILED"); continue

        for phase in ("cold", "warm"):
            marker = log_tail_marker()
            try:
                pid = submit(mod, clip, f"{label.split()[0]}_{phase}")
            except SystemExit as e:
                print(f"  [{phase}] submit failed: {e}"); break
            status, secs, peak, peak_sum, samples = wait_for(pid)
            tail = log_since(marker)
            staged = staged_size(tail)
            if secs:
                print(f"  [{phase}] {status}  {secs:.1f}s  "
                      f"peak GPU0 {peak} MiB / sum {peak_sum} MiB  ({samples} samples)")
            else:
                print(f"  [{phase}] {status}")
            if staged:
                print(f"          {staged[:150]}")
            results[(label, phase)] = secs

    print()
    print("=" * 72)
    print("SUMMARY (448x256, 4 steps, 2s, latent_upscale OFF)")
    print("=" * 72)
    for phase in ("cold", "warm"):
        a = results.get(("INT8 24.55GB", phase))
        b = results.get(("NVFP4 14.61GB", phase))
        fa = f"{a:6.1f}s" if a else "   n/a"
        fb = f"{b:6.1f}s" if b else "   n/a"
        ratio = f"  ({a/b:.2f}x)" if a and b else ""
        print(f"  {phase:<5}  INT8 {fa}   NVFP4 {fb}{ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
