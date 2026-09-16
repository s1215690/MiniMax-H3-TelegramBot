"""Test the VRAM guard that prevents starting the LLM on top of ComfyUI.

Real hazard this covers: ComfyUI keeps its models resident after a run. On this
2 x 20GB box the ~35GB llama-server then cannot fit, and starting it does not
produce a working LLM - it produces an OOM. The guard must therefore refuse with
an explanation, from every entry point, while never blocking the legitimate case
where ComfyUI is closed or idle.
"""

from __future__ import annotations

import importlib.util
import io
import contextlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_bot():
    spec = importlib.util.spec_from_file_location(
        "h3bot", HERE / "MiniMax-H3-Telegram-Bot.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3bot"] = mod
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def check(cond: bool, label: str) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return cond


def main() -> int:
    mod = load_bot()
    ok = True
    orig_online = mod.comfyui_is_online
    orig_snap = mod.gpu_vram_snapshot
    orig_llama_online = mod.llama_is_online

    # Two 20GB cards, ComfyUI holding 18.5GB on one of them (the state right
    # after a director / Bot generation).
    BUSY = [(1186, 20480), (18492, 20480)]
    IDLE = [(700, 20480), (200, 20480)]

    def with_state(comfy_online: bool, snapshot):
        mod.comfyui_is_online = lambda: comfy_online
        mod.gpu_vram_snapshot = lambda: snapshot

    print("=" * 70)
    print("1. ComfyUI 佔著顯存時必須拒絕")
    print("=" * 70)
    with_state(True, BUSY)
    blocker = mod.llm_start_blocker()
    total = sum(t for _, t in BUSY)
    free = sum(t - u for u, t in BUSY)
    ok &= check(bool(blocker), "有回報阻擋原因")
    ok &= check("OOM" in blocker, "訊息說明會 OOM")
    ok &= check(f"{free:,}" in blocker, f"列出實際可用顯存（{free:,} MB）")
    ok &= check(f"{total:,}" in blocker, f"列出總顯存（{total:,} MB）")
    ok &= check("關閉 ComfyUI" in blocker, "給出可執行的解法")
    print()
    for line in blocker.splitlines():
        print(f"        {line}")

    print()
    print("=" * 70)
    print("2. ComfyUI 沒開時必須放行")
    print("=" * 70)
    with_state(False, BUSY)
    ok &= check(mod.llm_start_blocker() == "", "ComfyUI 離線 -> 放行")

    print()
    print("=" * 70)
    print("3. ComfyUI 開著但沒佔顯存時必須放行（閒置不該被擋）")
    print("=" * 70)
    with_state(True, IDLE)
    ok &= check(mod.llm_start_blocker() == "", "ComfyUI 閒置 -> 放行")

    print()
    print("=" * 70)
    print("4. 探測不到顯存時不可誤擋")
    print("=" * 70)
    with_state(True, None)
    ok &= check(mod.llm_start_blocker() == "", "nvidia-smi 無資料 -> 放行（不誤擋）")

    print()
    print("=" * 70)
    print("5. 設定值不合理時不能讓 LLM 永久無法啟動")
    print("=" * 70)
    original_req = mod.LLM_START_MIN_FREE_MB
    mod.LLM_START_MIN_FREE_MB = 999_999_999  # absurd requirement
    with_state(True, IDLE)
    ok &= check(mod.llm_start_blocker() == "",
                "需求上限被夾在總顯存的 85%，不會因設定錯誤永久卡死")
    mod.LLM_START_MIN_FREE_MB = original_req

    print()
    print("=" * 70)
    print("6. 開關：MINIMAX_LLM_START_GUARD=0 可停用")
    print("=" * 70)
    original_flag = mod.LLM_START_GUARD
    mod.LLM_START_GUARD = False
    with_state(True, BUSY)
    ok &= check(mod.llm_start_blocker() == "", "停用後不再阻擋")
    mod.LLM_START_GUARD = original_flag

    print()
    print("=" * 70)
    print("7. start_llama_process() 本身會擋（覆蓋所有入口）")
    print("=" * 70)
    with_state(True, BUSY)
    mod.llama_is_online = lambda: False
    try:
        mod.start_llama_process()
        ok &= check(False, "應該要拋出 BotError")
    except mod.BotError as exc:
        ok &= check("OOM" in str(exc), "由 start_llama_process 拋出阻擋訊息")
    finally:
        mod.comfyui_is_online = orig_online
        mod.gpu_vram_snapshot = orig_snap
        mod.llama_is_online = orig_llama_online

    print()
    print("=" * 70)
    print("8. 對照：真實機器現況（需 ComfyUI 在線才有意義）")
    print("=" * 70)
    snap = mod.gpu_vram_snapshot()
    if snap:
        total = sum(t for _, t in snap)
        free = sum(t - u for u, t in snap)
        print(f"        {len(snap)} 張卡，總 {total:,} MB，可用 {free:,} MB")
        print(f"        ComfyUI 在線：{mod.comfyui_is_online()}")
        print(f"        目前判定：{'阻擋' if mod.llm_start_blocker() else '放行'}")
    else:
        print("        nvidia-smi 無資料")

    print()
    print("=" * 70)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
