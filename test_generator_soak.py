"""Reliability soak for the in-Bot script generator.

The original failure was intermittent: reasoning and the answer share one token
budget, and an example-based system prompt contained a self-contradictory
timeline that the model sometimes copied verbatim. Both are fixed, but a single
green run proves nothing, so this asks for several scripts in a row and reports
the real success rate.
"""

from __future__ import annotations

import importlib.util
import io
import contextlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

IDEAS = [
    (60.0, "下雨的東京街頭，一個女生錯過末班車"),
    (30.0, "貓在窗邊發呆，午後陽光"),
    (45.0, "深夜便利商店，店員獨自整理貨架"),
    (60.0, "an old fisherman mending nets at dawn on a quiet harbour"),
    (20.0, "霓虹燈下的機車騎士穿過雨夜城市"),
    (90.0, "太空站裡太空人看著地球緩緩轉動"),
]


def load_bot():
    spec = importlib.util.spec_from_file_location(
        "h3bot", HERE / "MiniMax-H3-Telegram-Bot.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3bot"] = mod
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = load_bot()
    if not mod.llama_is_online():
        print("llama.cpp is not online; cannot soak test.")
        return 2

    print(f"max_tokens={mod.SCRIPT_GEN_MAX_TOKENS} "
          f"retry={mod.SCRIPT_GEN_RETRY_TOKENS} "
          f"attempts={mod.SCRIPT_GEN_ATTEMPTS}")
    print("=" * 72)

    ok = 0
    first_try = 0
    times: list[float] = []
    for seconds, idea in IDEAS:
        t0 = time.time()
        try:
            script, log = mod.generate_h3_script(idea, seconds, mod.INPUT_MODE_TEXT)
        except Exception as exc:
            dt = time.time() - t0
            times.append(dt)
            print(f"[FAIL] {seconds:>5.0f}s  {dt:5.1f}s  {idea[:34]}")
            print(f"       {type(exc).__name__}: {str(exc).splitlines()[0][:150]}")
            continue
        dt = time.time() - t0
        times.append(dt)
        plan = mod.build_long_video_plan(script, seconds)
        detected = mod.detect_prompt_total_seconds(script)
        attempts = len(log)
        ok += 1
        if attempts == 1:
            first_try += 1
        flag = "OK  " if attempts == 1 else f"FIX{attempts}"
        print(f"[{flag}] {seconds:>5.0f}s  {dt:5.1f}s  shots={len(plan.shots):>2}  "
              f"len={len(script):>5}  detected={detected:g}  {idea[:30]}")
        for line in log:
            print(f"       {line}")
        (HERE / f"soak_{int(seconds)}_{abs(hash(idea)) % 10000:04d}.txt").write_text(
            script, encoding="utf-8"
        )

    print("=" * 72)
    print(f"success           : {ok}/{len(IDEAS)}")
    print(f"first-attempt pass: {first_try}/{len(IDEAS)}")
    if times:
        print(f"latency           : min {min(times):.1f}s  "
              f"avg {sum(times)/len(times):.1f}s  max {max(times):.1f}s")
    print("=" * 72)
    return 0 if ok == len(IDEAS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
