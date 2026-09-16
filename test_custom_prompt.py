"""Test the operator-editable custom instruction file.

Covers the behaviour the feature promises:
  • a missing file is a harmless no-op,
  • `#` lines are comments and never reach the model,
  • content is injected into the system prompt for BOTH long and short scripts,
  • braces in the custom text cannot crash str.format(),
  • edits are picked up without restarting the Bot (hot reload),
  • oversized files are truncated rather than breaking the request.
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
    path = mod.SCRIPT_GEN_PROMPT_FILE
    backup = None
    if path.is_file():
        backup = path.read_text(encoding="utf-8")

    try:
        print("=" * 68)
        print("1. 檔案不存在時是無害的 no-op")
        print("=" * 68)
        path.unlink(missing_ok=True)
        text, status = mod.load_custom_script_instructions()
        ok &= check(text == "", "回傳空字串")
        ok &= check("不存在" in status, f"狀態說明：{status}")
        msgs = mod.build_script_messages("測試", 60.0, mod.INPUT_MODE_TEXT, "zh")
        ok &= check("CUSTOM INSTRUCTIONS" not in msgs[0]["content"],
                    "沒有自訂指令時不注入區塊")

        print()
        print("=" * 68)
        print("2. 註解被剝離，內容被保留")
        print("=" * 68)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# 這行是註解，不該送出\n"
            "運鏡一律緩慢。\n"
            "\n"
            "  # 縮排的註解也不該送出\n"
            "不要出現浮水印。\n",
            encoding="utf-8",
        )
        text, status = mod.load_custom_script_instructions()
        ok &= check("註解" not in text, "註解文字沒有被載入")
        ok &= check("運鏡一律緩慢。" in text and "不要出現浮水印。" in text,
                    "兩條真正的規則都有載入")
        ok &= check("已載入" in status, f"狀態：{status}")

        print()
        print("=" * 68)
        print("3. 注入到長片與短片系統提示（且結構規則仍為權威）")
        print("=" * 68)
        for label, seconds in (("長片", 60.0), ("短片", 10.0)):
            sysmsg = mod.build_script_messages(
                "測試", seconds, mod.INPUT_MODE_TEXT, "zh"
            )[0]["content"]
            ok &= check("CUSTOM INSTRUCTIONS" in sysmsg, f"{label}：有注入區塊")
            ok &= check("運鏡一律緩慢。" in sysmsg, f"{label}：自訂文字在提示內")
            ok &= check("never override the STRUCTURE" in sysmsg,
                        f"{label}：明確聲明不得覆蓋結構規則")
            if seconds > 15:
                ok &= check("Do NOT invent your own timings" in sysmsg,
                            f"{label}：時間軸規則仍在")
                # Derive the expected heading: scene lengths are varied now.
                plan = mod.script_timeline_skeleton(seconds, "zh")
                first = f"{plan[0][0]}（{plan[0][1]:g}-{plan[0][2]:g}秒）："
                ok &= check(first in sysmsg, f"{label}：骨架仍在（{plan[0][0]}）")

        print()
        print("=" * 68)
        print("4. 大括號不會讓 str.format() 崩潰")
        print("=" * 68)
        path.write_text("務必保留 {這個} 與 {那個} 的花括號寫法。\n", encoding="utf-8")
        try:
            sysmsg = mod.build_script_messages(
                "測試", 60.0, mod.INPUT_MODE_TEXT, "zh"
            )[0]["content"]
            ok &= check("{這個}" in sysmsg, "花括號原樣保留")
        except Exception as exc:
            ok &= check(False, f"不該崩潰，卻拋出 {type(exc).__name__}: {exc}")

        print()
        print("=" * 68)
        print("5. 熱重載：改檔案後不需重啟即生效")
        print("=" * 68)
        path.write_text("第一版規則。\n", encoding="utf-8")
        first = mod.build_script_messages("測試", 60.0, mod.INPUT_MODE_TEXT, "zh")[0]["content"]
        path.write_text("第二版規則。\n", encoding="utf-8")
        second = mod.build_script_messages("測試", 60.0, mod.INPUT_MODE_TEXT, "zh")[0]["content"]
        ok &= check("第一版規則。" in first and "第二版規則。" not in first, "第一次讀到第一版")
        ok &= check("第二版規則。" in second and "第一版規則。" not in second,
                    "改檔後立即讀到第二版（無需重啟）")

        print()
        print("=" * 68)
        print("6. 過長檔案會被截斷，不會撐爆 context")
        print("=" * 68)
        path.write_text("規則。" * (mod.SCRIPT_GEN_PROMPT_MAX_CHARS), encoding="utf-8")
        text, status = mod.load_custom_script_instructions()
        ok &= check(len(text) <= mod.SCRIPT_GEN_PROMPT_MAX_CHARS + 40,
                    f"截斷後長度 {len(text)} <= 上限 {mod.SCRIPT_GEN_PROMPT_MAX_CHARS}")
        ok &= check("截斷" in status, f"狀態：{status}")

        print()
        print("=" * 68)
        print("7. 全部是註解時等同沒有自訂指令")
        print("=" * 68)
        path.write_text("# 只有註解\n# 還是註解\n", encoding="utf-8")
        text, status = mod.load_custom_script_instructions()
        ok &= check(text == "", "回傳空字串")
        ok &= check("註解" in status, f"狀態：{status}")
        sysmsg = mod.build_script_messages("測試", 60.0, mod.INPUT_MODE_TEXT, "zh")[0]["content"]
        ok &= check("CUSTOM INSTRUCTIONS" not in sysmsg, "不注入空區塊")

        print()
        print("=" * 68)
        print("8. BOM（Windows 編輯器常見）不會污染內容")
        print("=" * 68)
        path.write_text("帶 BOM 的規則。\n", encoding="utf-8-sig")
        text, status = mod.load_custom_script_instructions()
        ok &= check(text.startswith("帶 BOM"), f"開頭乾淨：{text[:12]!r}")
        ok &= check("\ufeff" not in text, "沒有殘留 BOM 字元")
    finally:
        if backup is not None:
            path.write_text(backup, encoding="utf-8")
        else:
            path.unlink(missing_ok=True)

    print()
    print("=" * 68)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
