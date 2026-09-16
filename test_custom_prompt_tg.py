"""Test editing the custom instructions entirely from Telegram.

Drives the real bot methods (request_custom_prompt / handle_custom_prompt_text /
callback actions) against a fake Telegram client, so the assertions cover the
actual code path rather than the file helpers alone.
"""

from __future__ import annotations

import importlib.util
import io
import contextlib
import sys
import threading
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


class FakeTelegram:
    """Records every outbound call and every inline keyboard."""

    def __init__(self):
        self.messages: list[str] = []
        self.markups: list[dict] = []

    def send_message(self, chat_id, text, reply_markup=None):
        self.messages.append(text)
        if reply_markup:
            self.markups.append(reply_markup)

    def edit_message_text(self, chat_id, message_id, text, reply_markup=None):
        self.messages.append(text)
        if reply_markup:
            self.markups.append(reply_markup)

    def send_chat_action(self, chat_id, action="typing"):
        pass


def make_bot(mod):
    bot = object.__new__(mod.TelegramMenuBot)
    bot.settings = mod.parse_config(["864", "480", "8", "15"])
    bot.total_seconds = 30.0
    bot.input_mode = mod.INPUT_MODE_TEXT
    bot.prompt = ""
    bot.job = None
    bot.lock = threading.RLock()
    bot.allowed_chat_id = "0"
    bot.menu_section = mod.MENU_MAIN
    bot._shutdown_pending = False
    bot.shutdown_after_generation = False
    bot.script_lang = "zh"
    bot.script_busy = False
    bot.script_draft = None
    bot.script_idea = ""
    bot.script_seconds = 30.0
    bot.awaiting_custom_prompt = ""
    bot.awaiting_script_idea = False
    bot.awaiting_prompt = False
    bot.awaiting_duration = False
    bot.awaiting_queue_prompt = False
    bot.telegram = FakeTelegram()
    return bot


def check(cond: bool, label: str) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return cond


def main() -> int:
    mod = load_bot()
    ok = True
    path = mod.SCRIPT_GEN_PROMPT_FILE
    backup, prev = None, None
    if path.is_file():
        backup = path.read_text(encoding="utf-8")
    if mod.SCRIPT_GEN_PROMPT_BACKUP.is_file():
        prev = mod.SCRIPT_GEN_PROMPT_BACKUP.read_text(encoding="utf-8")

    try:
        path.unlink(missing_ok=True)
        mod.SCRIPT_GEN_PROMPT_BACKUP.unlink(missing_ok=True)

        print("=" * 70)
        print("1. 面板顯示自訂指令並提供編輯按鈕")
        print("=" * 70)
        bot = make_bot(mod)
        bot.show_script_prompt_file("0")
        callbacks = [b["callback_data"] for m in bot.telegram.markups
                     for row in m.get("inline_keyboard", []) for b in row]
        ok &= check("script_file:edit" in callbacks, "有「編輯」按鈕")
        ok &= check("script_file:append" in callbacks, "有「追加」按鈕")
        ok &= check("script_file:clear" in callbacks, "有「清空」按鈕")
        ok &= check("script_file:undo" in callbacks, "有「還原上一版」按鈕")
        ok &= check(any("目前沒有任何自訂指令" in m for m in bot.telegram.messages),
                    "顯示目前是空的")

        print()
        print("=" * 70)
        print("2. 在 TG 輸入內容即儲存，且真的會送給模型")
        print("=" * 70)
        bot.request_custom_prompt("0", "replace")
        ok &= check(bot.awaiting_custom_prompt == "replace", "進入等待輸入狀態")
        bot.handle_custom_prompt_text("0", "運鏡一律緩慢。\n不要出現浮水印。")
        ok &= check(bot.awaiting_custom_prompt == "", "輸入後離開等待狀態")
        saved, status = mod.load_custom_script_instructions()
        ok &= check("運鏡一律緩慢。" in saved and "不要出現浮水印。" in saved,
                    f"檔案內容正確：{saved!r}")
        sysmsg = mod.build_script_messages("測試", 60.0, mod.INPUT_MODE_TEXT, "zh")[0]["content"]
        ok &= check("運鏡一律緩慢。" in sysmsg, "已進入系統提示")
        ok &= check("CUSTOM INSTRUCTIONS" in sysmsg, "注入區塊存在")
        ok &= check(any("立即生效" in m for m in bot.telegram.messages),
                    "有回報「立即生效」")

        print()
        print("=" * 70)
        print("3. 追加模式會接在後面，不覆蓋原有規則")
        print("=" * 70)
        bot.request_custom_prompt("0", "append")
        bot.handle_custom_prompt_text("0", "每個場景都要有明確光源。")
        saved, _ = mod.load_custom_script_instructions()
        ok &= check("運鏡一律緩慢。" in saved, "原規則仍在")
        ok &= check("每個場景都要有明確光源。" in saved, "新規則已追加")
        ok &= check(saved.index("運鏡一律緩慢。") < saved.index("每個場景都要有明確光源。"),
                    "順序是新規則在後")

        print()
        print("=" * 70)
        print("4. 整段取代會蓋掉舊內容")
        print("=" * 70)
        bot.request_custom_prompt("0", "replace")
        bot.handle_custom_prompt_text("0", "只用冷色調。")
        saved, _ = mod.load_custom_script_instructions()
        ok &= check(saved.strip() == "只用冷色調。", f"只剩新內容：{saved!r}")

        print()
        print("=" * 70)
        print("5. 還原上一版")
        print("=" * 70)
        ok2, note = mod.restore_custom_script_instructions()
        saved, _ = mod.load_custom_script_instructions()
        ok &= check(ok2, f"還原成功：{note}")
        ok &= check("每個場景都要有明確光源。" in saved, "回到上一版內容")

        print()
        print("=" * 70)
        print("6. 清空")
        print("=" * 70)
        ok3, note = mod.save_custom_script_instructions("", "由 Telegram 清空")
        saved, status = mod.load_custom_script_instructions()
        ok &= check(ok3 and saved == "", f"已清空（{note}）")
        sysmsg = mod.build_script_messages("測試", 60.0, mod.INPUT_MODE_TEXT, "zh")[0]["content"]
        ok &= check("CUSTOM INSTRUCTIONS" not in sysmsg, "清空後不再注入區塊")

        print()
        print("=" * 70)
        print("7. 空訊息不破壞既有內容")
        print("=" * 70)
        mod.save_custom_script_instructions("保留我。", "test")
        bot.request_custom_prompt("0", "replace")
        bot.handle_custom_prompt_text("0", "   ")
        saved, _ = mod.load_custom_script_instructions()
        ok &= check("保留我。" in saved, "空白輸入被忽略，原內容保留")
        ok &= check(any("已取消" in m for m in bot.telegram.messages), "有提示已取消")

        print()
        print("=" * 70)
        print("8. 註解與超長內容的防護仍在")
        print("=" * 70)
        ok4, _ = mod.save_custom_script_instructions(
            "# 這是註解\n真正的規則。", "test"
        )
        saved, _ = mod.load_custom_script_instructions()
        ok &= check("這是註解" not in saved and "真正的規則。" in saved,
                    "使用者寫的 # 註解不會送出")
        huge = "規則。" * (mod.SCRIPT_GEN_PROMPT_MAX_CHARS)
        ok5, _ = mod.save_custom_script_instructions(huge, "test")
        saved, status = mod.load_custom_script_instructions()
        ok &= check(len(saved) <= mod.SCRIPT_GEN_PROMPT_MAX_CHARS + 40,
                    f"超長內容被截斷（{len(saved)} 字元）")

        print()
        print("=" * 70)
        print("9. 跨訊息狀態互不干擾")
        print("=" * 70)
        bot2 = make_bot(mod)
        bot2.request_custom_prompt("0", "replace")
        ok &= check(bot2.awaiting_custom_prompt == "replace", "進入自訂指令輸入")
        ok &= check(bot2.awaiting_script_idea is False, "腳本生成輸入未被誤觸")
        ok &= check(bot2.awaiting_prompt is False, "一般提示詞輸入未被誤觸")
    finally:
        if backup is not None:
            path.write_text(backup, encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
        if prev is not None:
            mod.SCRIPT_GEN_PROMPT_BACKUP.write_text(prev, encoding="utf-8")
        else:
            mod.SCRIPT_GEN_PROMPT_BACKUP.unlink(missing_ok=True)

    print()
    print("=" * 70)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
