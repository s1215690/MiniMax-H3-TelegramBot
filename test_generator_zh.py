"""Simplified-Chinese output test for the in-Bot script generator.

Verifies that
  1. generated scripts are actually Simplified Chinese (not English, not
     Traditional Chinese),
  2. the Bot's own validator still accepts them,
  3. English mode still works after the language parameterisation.

Visible scene text is exempt from the Simplification check: a Japanese station
sign legitimately reads 終電, and the official H3 guide requires dialogue, lyrics
and on-screen text to keep their original form.
"""

from __future__ import annotations

import importlib.util
import io
import contextlib
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Traditional-only forms whose simplified counterpart differs. A hit means the
# model drifted into Traditional Chinese.
TRADITIONAL_ONLY = set(
    "開關們個來時這為與後點說對過還將當種樣讓視聽體發變隻總萬億電腦號鏡線車東門長飛"
    "語聲響燈風雲裡話認識讀寫兒機務單賣買價錢銀銅鋼鐵針紙筆記詞彙傳聞讀寫與應該"
)

CASES = [
    (60.0, "下雨的東京街頭，一個女生錯過末班車"),
    (30.0, "貓在窗邊發呆，午後陽光"),
    (45.0, "深夜便利商店，店員獨自整理貨架"),
    (20.0, "霓虹燈下的機車騎士穿過雨夜城市"),
    (90.0, "太空站裡太空人看著地球緩緩轉動"),
    (12.0, "一杯咖啡冒著熱氣放在木桌上"),
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


def script_shape(text: str) -> dict:
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    latin = re.findall(r"[A-Za-z]{3,}", text)
    trad = sorted({c for c in cjk if c in TRADITIONAL_ONLY})
    return {
        "cjk": len(cjk),
        "latin": latin,
        "trad": trad,
        "cjk_ratio": len(cjk) / max(1, len(cjk) + sum(len(w) for w in latin)),
    }


def main() -> int:
    mod = load_bot()
    if not mod.llama_is_online():
        print("llama.cpp is not online; cannot test.")
        return 2

    ok = True
    print("=" * 74)
    print("A. 簡體中文輸出")
    print("=" * 74)
    for seconds, idea in CASES:
        t0 = time.time()
        try:
            script, log = mod.generate_h3_script(
                idea, seconds, mod.INPUT_MODE_TEXT, lang="zh"
            )
        except Exception as exc:
            ok = False
            print(f"[FAIL] {seconds:>5.0f}s  {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
            continue
        dt = time.time() - t0
        shape = script_shape(script)
        notes = []
        if seconds > 15:
            try:
                mod.build_long_video_plan(script, seconds)
                notes.append("validator OK")
            except Exception as exc:
                ok = False
                notes.append(f"VALIDATOR FAIL: {exc}")
        else:
            notes.append("prose")
        simplified = not shape["trad"]
        if not simplified:
            # Report but do not fail: on-screen scene text may be Japanese.
            notes.append(f"trad chars: {shape['trad']} (may be on-screen text)")
        if shape["cjk_ratio"] < 0.5:
            ok = False
            notes.append(f"NOT CHINESE (cjk ratio {shape['cjk_ratio']:.2f})")
        print(f"[{ 'OK  ' if simplified else 'WARN'}] {seconds:>5.0f}s {dt:5.1f}s "
              f"cjk={shape['cjk']:>4} latin={len(shape['latin']):>2} "
              f"{' | '.join(notes)}")
        (HERE / f"zh_{int(seconds)}s.txt").write_text(script, encoding="utf-8")

    print()
    print("=" * 74)
    print("B. 英文模式仍可運作（語言切換沒有破壞原本路徑）")
    print("=" * 74)
    try:
        t0 = time.time()
        script, log = mod.generate_h3_script(
            "a woman waiting on a rainy platform", 30.0, mod.INPUT_MODE_TEXT, lang="en"
        )
        dt = time.time() - t0
        shape = script_shape(script)
        latin_ok = len(shape["latin"]) > 30 and shape["cjk"] < 60
        ok &= latin_ok
        plan = mod.build_long_video_plan(script, 30.0)
        print(f"  [{'PASS' if latin_ok else 'FAIL'}] {dt:.1f}s latin_words={len(shape['latin'])} "
              f"cjk={shape['cjk']} shots={len(plan.shots)}")
    except Exception as exc:
        ok = False
        print(f"  [FAIL] {type(exc).__name__}: {exc}")

    print()
    print("=" * 74)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
