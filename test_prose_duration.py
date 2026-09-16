"""Regression test for the reported failure:

    長片時間軸格式錯誤：超過 15 秒的長片必須提供時間軸 ...

Cause: a short (<= 15s) script is plain prose with no timeline, so its duration
cannot be detected from the text. Accepting it left a stale longer `total_seconds`
in place, which sent prose down the long-video path where build_long_video_plan()
rejects it. The script looked perfectly fine in chat but could never be generated.

This test drives the real `accept_script_draft` code path on a throwaway bot
instance (no network, no threads) and asserts the duration is reconciled.
"""

from __future__ import annotations

import importlib.util
import io
import contextlib
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROSE = (
    "A young woman with long straight black hair stands under a showerhead in a "
    "small white-tiled changing room. She tilts her head back and lets the water "
    "run over her face, then pushes her hair back with both hands. The camera "
    "slowly pushes in toward her face. The sound is the steady hiss of running "
    "water and a soft echo in the small enclosed space."
)

LONG_SCRIPT = (
    "A woman in a grey coat waits on a rainy platform at night. "
    "Muted cinematic style. A slow piano plays throughout.\n\n"
    "開頭（0-20秒）：\nShe stands under the shelter. Rain falls. "
    "The camera pushes in slowly.\n\n"
    "第一幕（20-40秒）：\nA train arrives with a whistle and she steps back. "
    "Brakes screech.\n\n"
    "結尾（40-60秒）：\nThe train leaves without her. The piano fades.\n"
)


def load_bot():
    spec = importlib.util.spec_from_file_location(
        "h3bot", HERE / "MiniMax-H3-Telegram-Bot.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3bot"] = mod
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def make_bot(mod):
    """A bot instance with just enough state for accept_script_draft()."""
    bot = object.__new__(mod.TelegramMenuBot)
    bot.settings = mod.parse_config(["864", "480", "8", "15"])
    bot.total_seconds = 60.0
    bot.input_mode = mod.INPUT_MODE_TEXT
    bot.prompt = ""
    bot.job = None
    bot.lock = threading.RLock()
    bot.allowed_chat_id = "0"
    bot.menu_section = mod.MENU_MAIN
    bot._shutdown_pending = False
    bot.shutdown_after_generation = False
    bot.script_seconds = 15.0
    bot.script_draft = None
    bot.script_idea = ""
    bot.script_busy = False
    bot.awaiting_prompt = False
    # Capture Telegram traffic instead of sending it.
    bot.sent: list[str] = []
    bot.telegram = type("T", (), {
        "send_message": lambda self, chat_id, text, reply_markup=None: bot.sent.append(text),
        "edit_message_text": lambda self, c, m, t, reply_markup=None: bot.sent.append(t),
        "send_chat_action": lambda self, c, a="typing": None,
    })()
    # save_settings() would touch the real state file; stub it out.
    bot.save_settings = lambda: None
    # show_menu() renders the full control panel, which needs a lot of unrelated
    # state. This test only cares about duration reconciliation, so the notice
    # is captured instead.
    notices: list[str] = []
    bot.notices = notices

    def fake_show_menu(chat_id, message_id=None, notice="", **kwargs):
        if notice:
            notices.append(notice)

    bot.show_menu = fake_show_menu
    return bot


def check(cond: bool, label: str) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return cond


def main() -> int:
    mod = load_bot()
    ok = True

    # ---------------------------------------------------------------- baseline
    print("=" * 68)
    print("0. 先確認這就是原始故障（散文 + 60 秒 = 被拒）")
    print("=" * 68)
    try:
        mod.build_long_video_plan(PROSE, 60.0)
        ok &= check(False, "應該要被拒絕")
    except mod.BotError as exc:
        ok &= check("必須提供時間軸" in str(exc), "重現原始錯誤訊息")

    # ------------------------------------------------------------ prose accept
    print()
    print("=" * 68)
    print("1. 短片散文腳本：採用後片長必須自動對齊（原本會卡住）")
    print("=" * 68)
    bot = make_bot(mod)
    bot.script_draft = PROSE
    bot.script_seconds = 15.0
    bot.accept_script_draft("0", None, run_now=False)
    ok &= check(abs(bot.total_seconds - 15.0) < 1e-6,
                f"total_seconds 由 60 調整為 {bot.total_seconds:g}")
    ok &= check(bot.prompt == PROSE, "提示詞已寫入")
    ok &= check(any("調整為" in t for t in bot.notices), "有明確告知片長被調整")
    # And the short path must no longer be forced through the long validator.
    ok &= check(bot.total_seconds <= mod.MAX_SEGMENT_SECONDS,
                "片長已落在短片範圍，不會再走長片驗證")

    print()
    print("=" * 68)
    print("2. 各種短片秒數都要對齊")
    print("=" * 68)
    for secs in (5.0, 10.0, 12.0, 15.0):
        b = make_bot(mod)
        b.script_draft = PROSE
        b.script_seconds = secs
        b.accept_script_draft("0", None, run_now=False)
        ok &= check(abs(b.total_seconds - secs) < 1e-6,
                    f"script_seconds={secs:g} -> total_seconds={b.total_seconds:g}")

    # ------------------------------------------------------------- long accept
    print()
    print("=" * 68)
    print("3. 長片腳本：仍由時間軸決定片長（不能被短片邏輯干擾）")
    print("=" * 68)
    b = make_bot(mod)
    b.total_seconds = 15.0            # deliberately wrong setting
    b.script_draft = LONG_SCRIPT
    b.script_seconds = 60.0
    b.accept_script_draft("0", None, run_now=False)
    ok &= check(abs(b.total_seconds - 60.0) < 1e-6,
                f"total_seconds 由時間軸校正為 {b.total_seconds:g}")
    plan = mod.build_long_video_plan(b.prompt, b.total_seconds)
    ok &= check(len(plan.shots) >= 2, f"可產生 {len(plan.shots)} 個鏡頭")

    print()
    print("=" * 68)
    print("4. 邊界：script_seconds 缺失時不可亂改片長")
    print("=" * 68)
    b = make_bot(mod)
    b.script_draft = PROSE
    b.script_seconds = 0.0            # unknown -> must not clamp to nonsense
    b.accept_script_draft("0", None, run_now=False)
    ok &= check(abs(b.total_seconds - 60.0) < 1e-6,
                f"未知秒數時維持原片長 {b.total_seconds:g}")

    print()
    print("=" * 68)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
