"""Test repeated editing of a generated script draft.

The requirement is that a draft can be edited a second, third and Nth time, and
that any of those rounds can be undone. This drives the real bot methods against
a fake Telegram client and a stubbed LLM, so it checks the actual code path.
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
    bot.script_idea = "測試"
    bot.script_seconds = 30.0
    bot.script_history = []
    bot.script_last_action = ""
    bot.awaiting_script_edit = ""
    bot.awaiting_script_refine = False
    bot.script_refine_instruction = ""
    bot.awaiting_script_idea = False
    bot.awaiting_custom_prompt = ""
    bot.awaiting_prompt = False
    bot.awaiting_duration = False
    bot.awaiting_queue_prompt = False
    bot.telegram = FakeTelegram()
    bot.save_settings = lambda: None
    bot.show_menu = lambda *a, **k: None
    # Never touch the real LLM from these tests.
    bot._ensure_llm_ready = lambda chat_id: None
    return bot


def valid_script(mod, total=30.0, tag="A"):
    plan = mod.script_timeline_skeleton(total, "zh")
    body = f"{tag}：一個女生站在雨中的車站月台。冷色調電影感。鋼琴配樂貫穿。\n\n"
    for label, start, end in plan:
        body += (
            f"{label}（{start:g}-{end:g}秒）：\n"
            f"{tag} 雨不斷落下，鏡頭緩慢推近。她低頭看著水面。\n\n"
        )
    return body.strip()


def check(cond: bool, label: str) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return cond


class SyncThread:
    """Runs the worker inline instead of on a real thread.

    handle_script_refine_text starts a background thread itself, so calling the
    worker again from the test would run each round twice and race for the
    stubbed LLM replies. Making the thread synchronous exercises the real code
    path exactly once.
    """

    def __init__(self, target=None, args=(), name=None, daemon=None):
        self.target = target
        self.args = args

    def start(self):
        if self.target:
            self.target(*self.args)


class NoopThread:
    """Records the spawn but never runs it."""

    spawned: list = []

    def __init__(self, target=None, args=(), name=None, daemon=None):
        NoopThread.spawned.append((target, args))

    def start(self):
        pass


class patched_threads:
    """Swap the bot's threading.Thread for a controlled stand-in.

    The Bot imports the same `threading` module object as this test, so patching
    Thread here is exactly what its worker-spawning code will look up.
    """

    def __init__(self, fake):
        self.fake = fake

    def __enter__(self):
        self.original = threading.Thread
        threading.Thread = self.fake
        return self

    def __exit__(self, *exc):
        threading.Thread = self.original
        return False


def main() -> int:
    mod = load_bot()
    ok = True
    original_chat = mod.llama_chat

    print("=" * 72)
    print("1. 草稿面板提供多輪編輯按鈕")
    print("=" * 72)
    bot = make_bot(mod)
    bot.script_draft = valid_script(mod, 30.0)
    bot.show_script_draft("0", bot.script_draft)
    cbs = [b["callback_data"] for m in bot.telegram.markups
           for row in m.get("inline_keyboard", []) for b in row]
    for cb, name in (("script:refine", "指令修改(AI)"), ("script:edit", "手動編輯"),
                     ("script:append", "追加內容"), ("script:regen", "重新生成")):
        ok &= check(cb in cbs, f"有「{name}」按鈕")
    ok &= check("script:undo" not in cbs, "第一版時不顯示「回到上一版」")
    ok &= check(any("格式檢查通過" in m for m in bot.telegram.messages),
                "顯示格式檢查通過")

    print()
    print("=" * 72)
    print("2. 手動編輯：第二次、第三次、第四次…")
    print("=" * 72)
    bot = make_bot(mod)
    bot.script_draft = valid_script(mod, 30.0, "v1")
    for i in range(2, 6):
        bot.request_script_edit("0", "replace")
        ok &= check(bot.awaiting_script_edit == "replace", f"第 {i} 次進入編輯狀態")
        bot.handle_script_edit_text("0", valid_script(mod, 30.0, f"v{i}"))
        ok &= check(f"v{i}：" in bot.script_draft, f"第 {i} 次編輯已套用")
    ok &= check(len(bot.script_history) == 4, f"累積 4 個歷史版本（實際 {len(bot.script_history)}）")
    ok &= check(any("第 5 版" in m for m in bot.telegram.messages), "版本號遞增到第 5 版")

    print()
    print("=" * 72)
    print("3. 連續退回多個版本")
    print("=" * 72)
    for want in ("v4", "v3", "v2", "v1"):
        bot.undo_script_draft("0")
        ok &= check(f"{want}：" in bot.script_draft, f"退回後是 {want}")
    ok &= check(bot.script_history == [], "退到底後歷史清空")
    before = bot.script_draft
    bot.undo_script_draft("0")
    ok &= check(bot.script_draft == before, "沒有更早版本時不會損壞草稿")
    ok &= check(any("沒有更早的版本" in m for m in bot.telegram.messages), "有提示無更早版本")

    print()
    print("=" * 72)
    print("4. 追加內容：保留原有並接在後面")
    print("=" * 72)
    bot = make_bot(mod)
    bot.script_draft = valid_script(mod, 30.0, "base")
    bot.request_script_edit("0", "append")
    bot.handle_script_edit_text("0", "她輕輕嘆了一口氣。")
    ok &= check("base：" in bot.script_draft, "原內容仍在")
    ok &= check(bot.script_draft.rstrip().endswith("她輕輕嘆了一口氣。"), "新內容接在最後")

    print()
    print("=" * 72)
    print("5. 手動改壞格式：接受但明確警告")
    print("=" * 72)
    bot = make_bot(mod)
    bot.script_draft = valid_script(mod, 30.0, "good")
    bot.request_script_edit("0", "replace")
    bot.handle_script_edit_text("0", "這是一段沒有時間軸的散文，長度超過十五秒就會被拒絕。")
    ok &= check(bot.script_draft.startswith("這是一段沒有時間軸"), "手動內容已保留")
    ok &= check(any("格式檢查未通過" in m for m in bot.telegram.messages),
                "明確標示格式未通過")
    ok &= check(any("會被拒絕" in m for m in bot.telegram.messages), "說明直接生成會被拒絕")
    bot.undo_script_draft("0")
    ok &= check("good：" in bot.script_draft, "可退回改壞之前的版本")

    print()
    print("=" * 72)
    print("6. AI 指令修改：可重複多輪")
    print("=" * 72)
    bot = make_bot(mod)
    bot.script_draft = valid_script(mod, 30.0, "v1")
    replies = [valid_script(mod, 30.0, f"ai{i}") for i in range(2, 7)]
    seen: list[str] = []

    def fake_chat(messages, **kwargs):
        # The draft and the instruction must both reach the model.
        joined = " ".join(m.get("content", "") for m in messages)
        seen.append(joined)
        return replies.pop(0)

    mod.llama_chat = fake_chat
    try:
        with patched_threads(SyncThread):
            for i in range(2, 7):
                bot.handle_script_refine_text("0", f"指令{i}")
                ok &= check(f"ai{i}：" in bot.script_draft, f"第 {i} 版由 AI 產生")
        ok &= check(len(bot.script_history) == 5, f"5 輪修改留下 5 個歷史版本（{len(bot.script_history)}）")
        ok &= check(all("指令" in s for s in seen[:5]), "每次都有把修改指令送給模型")
        ok &= check(all("--- CURRENT ---" in s for s in seen[:5]), "每次都有把現有草稿送給模型")
    finally:
        mod.llama_chat = original_chat

    print()
    print("=" * 72)
    print("7. AI 修改失敗時，草稿不能被破壞")
    print("=" * 72)
    bot = make_bot(mod)
    good = valid_script(mod, 30.0, "keepme")
    bot.script_draft = good
    mod.llama_chat = lambda messages, **kw: (_ for _ in ()).throw(
        mod.BotError("模擬失敗")
    )
    try:
        with patched_threads(SyncThread):
            bot.handle_script_refine_text("0", "改壞它")
        ok &= check(bot.script_draft == good, "失敗後草稿維持原樣")
        ok &= check(bot.script_history == [], "失敗不留歷史垃圾")
        ok &= check(not bot.script_busy, "失敗後解除忙碌狀態")
        ok &= check(any("修改失敗" in m for m in bot.telegram.messages), "有回報失敗")
    finally:
        mod.llama_chat = original_chat

    print()
    print("=" * 72)
    print("8. AI 產出格式錯誤時會自我修正（不直接失敗）")
    print("=" * 72)
    bot = make_bot(mod)
    bot.script_draft = valid_script(mod, 30.0, "v1")
    calls = {"n": 0}

    def flaky(messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "開頭（0-20秒）：\n她站著。\n\n結尾（5-30秒）：\n重疊的時間軸。"
        return valid_script(mod, 30.0, "fixed")

    mod.llama_chat = flaky
    try:
        with patched_threads(SyncThread):
            bot.handle_script_refine_text("0", "修正")
        ok &= check(calls["n"] == 2, f"重試一次後成功（呼叫 {calls['n']} 次）")
        ok &= check("fixed：" in bot.script_draft, "採用修復後的版本")
    finally:
        mod.llama_chat = original_chat

    print()
    print("=" * 72)
    print("9. 新想法會清空歷史，避免退回到上一個不相關的腳本")
    print("=" * 72)
    bot = make_bot(mod)
    bot.script_draft = valid_script(mod, 30.0, "old")
    bot.script_history = ["x", "y", "z"]
    with patched_threads(NoopThread):
        bot.handle_script_idea("0", "30秒 全新的故事")
    ok &= check(bot.script_history == [], "歷史已清空")
    ok &= check(bot.script_draft is None, "草稿已清空")
    ok &= check(bot.script_idea == "全新的故事", "新想法已記錄")

    print()
    print("=" * 72)
    print("10. 輸入狀態互不干擾")
    print("=" * 72)
    bot = make_bot(mod)
    bot.script_draft = valid_script(mod, 30.0)
    bot.request_script_edit("0", "replace")
    ok &= check(bot.awaiting_script_edit == "replace" and not bot.awaiting_script_refine,
                "進入編輯狀態時不會同時等待 AI 指令")
    bot.request_script_refine("0")
    ok &= check(bot.awaiting_script_refine and bot.awaiting_script_edit == "",
                "進入 AI 指令狀態時清掉編輯狀態")
    ok &= check(not bot.awaiting_script_idea and not bot.awaiting_custom_prompt,
                "未誤觸腳本生成／自訂指令的輸入狀態")

    print()
    print("=" * 72)
    print("11. 邊界：沒草稿時不可進入編輯")
    print("=" * 72)
    bot = make_bot(mod)
    bot.script_draft = None
    bot.request_script_edit("0", "replace")
    ok &= check(bot.awaiting_script_edit == "", "沒有草稿時不進入編輯狀態")
    ok &= check(any("目前沒有草稿" in m for m in bot.telegram.messages), "有提示先產生腳本")
    bot.handle_script_refine_text("0", "改一下")
    ok &= check(not bot.script_busy, "沒有草稿時不會啟動 AI 修改")

    print()
    print("=" * 72)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
