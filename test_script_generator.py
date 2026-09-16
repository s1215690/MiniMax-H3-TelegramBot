"""End-to-end test of the in-Bot script generator.

Covers three things the Telegram flow depends on:
  1. the real generation path against the live llama.cpp server;
  2. the self-repair loop, driven deterministically by injecting a broken
     first reply so the retry logic is exercised without luck;
  3. duration parsing / detection so "60秒 ..." sets the right video length.
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

    # ---------------------------------------------------------------- repair
    print("=" * 64)
    print("1. 自修復回圈（注入壞輸出，確定性驗證）")
    print("=" * 64)
    # Gaps and overlaps: 0-20 then 5-30 then 28-60 -> must be rejected, then fixed.
    broken = (
        "A woman waits on a platform in the rain.\n\n"
        "開頭（0-20秒）：\nShe stands still. Rain falls.\n\n"
        "第一幕（5-30秒）：\nA train passes.\n\n"
        "結尾（28-60秒）：\nShe leaves.\n"
    )
    good = (
        "A woman in a grey coat waits on a rainy platform at night. "
        "Muted cinematic style. A slow piano plays throughout.\n\n"
        "開頭（0-20秒）：\nShe stands under the shelter, water dripping from the roof. "
        "The camera slowly pushes in. Rain and a distant announcement.\n\n"
        "第一幕（20-40秒）：\nA train arrives with a long whistle and she steps back. "
        "The camera tracks left. Brakes screech and doors hiss.\n\n"
        "結尾（40-60秒）：\nThe train leaves without her and she watches it go. "
        "The camera holds wide. The piano fades under the rain.\n"
    )
    calls: list[int] = []
    replies = [broken, good]

    def fake_chat(messages, **kwargs):
        calls.append(len(messages))
        # Second call must carry the validator error back to the model.
        if len(calls) == 2:
            joined = " ".join(m.get("content", "") for m in messages)
            check("重疊" in joined or "沒有內容" in joined,
                  "第二次呼叫有把驗證錯誤回饋給模型")
        return replies.pop(0) if replies else good

    original = mod.llama_chat
    mod.llama_chat = fake_chat
    try:
        script, log = mod.generate_h3_script("測試", 60.0, mod.INPUT_MODE_TEXT)
    finally:
        mod.llama_chat = original

    ok &= check(len(calls) == 2, f"第一次被拒絕、第二次成功（共 {len(calls)} 次呼叫）")
    # The returned script is normalised (trailing whitespace stripped), so the
    # comparison is made on stripped text; the wording must match exactly.
    ok &= check(script.strip() == good.strip(), "回傳的是修復後的版本")
    for line in log:
        print(f"        {line}")

    # --------------------------------------------------------------- failure
    print()
    print("=" * 64)
    print("2. 永遠壞掉時要乾淨地失敗（不能無限重試）")
    print("=" * 64)
    mod.llama_chat = lambda messages, **kw: broken
    try:
        mod.generate_h3_script("測試", 60.0, mod.INPUT_MODE_TEXT, attempts=3)
        ok &= check(False, "應該要拋出 BotError")
    except mod.BotError as exc:
        ok &= check("3 次" in str(exc), "重試 3 次後放棄並說明原因")
        print(f"        訊息：{str(exc).splitlines()[0]}")
    finally:
        mod.llama_chat = original

    # ------------------------------------------------------------ short form
    print()
    print("=" * 64)
    print("3. 短片（<=15 秒）走散文格式，不做時間軸驗證")
    print("=" * 64)
    short_reply = (
        "A grey tabby cat sits on a sunny windowsill, its tail flicking slowly. "
        "Warm afternoon light rakes across the wooden floor. The camera drifts "
        "closer as the cat blinks and settles. Soft room tone and a faint purr."
    )
    mod.llama_chat = lambda messages, **kw: short_reply
    try:
        s, l = mod.generate_h3_script("貓", 10.0, mod.INPUT_MODE_TEXT)
        ok &= check(s == short_reply, "短片直接採用散文輸出")
        ok &= check("散文" in " ".join(l), "記錄為散文格式")
    finally:
        mod.llama_chat = original

    # --------------------------------------------------------- duration parse
    print()
    print("=" * 64)
    print("5. 秒數解析：各種自然寫法都要抓到，且不可誤判")
    print("=" * 64)
    parse_cases = [
        ("30秒 下雨的車站", 30.0), ("30 秒 下雨的車站", 30.0), ("30s 下雨的車站", 30.0),
        ("30s rain station", 30.0), ("30秒，下雨的車站", 30.0), ("30秒：下雨的車站", 30.0),
        ("下雨的車站 30秒", 30.0), ("下雨的車站30秒", 30.0), ("30秒。下雨的車站", 30.0),
        ("[30秒] 下雨的車站", 30.0), ("（30秒）下雨的車站", 30.0),
        ("30 sec rain station", 30.0), ("30 seconds rain station", 30.0),
        ("1分鐘 下雨", 60.0), ("2分鐘 賽博", 120.0),
        ("1分30秒 下雨", 90.0), ("1分30秒", 90.0), ("半分鐘 下雨", 30.0),
        # These are the phrasings the old `\b` guard silently dropped, causing
        # the Bot to fall back to the previous length (asked 30s, got 15s).
        ("30秒的影片 下雨的車站", 30.0), ("請給我30秒的影片 下雨", 30.0),
        ("30秒版 下雨", 30.0), ("30秒的短片 下雨", 30.0),
        ("时长30秒 下雨", 30.0), ("時長30秒 下雨", 30.0), ("要30秒 下雨的車站", 30.0),
        ("90秒 下雨", 90.0), ("5秒 貓", 5.0), ("1800秒 長片", 1800.0),
        # Must NOT be read as a duration.
        ("30something ideas", None), ("下雨的車站", None), ("3個女孩在跳舞", None),
        ("第2幕的內容", None), ("30 天後的世界", None), ("1900秒 太長", None),
    ]
    parse_bad = []
    for text, want in parse_cases:
        got, _ = mod.parse_idea_duration(text)
        if got != want:
            parse_bad.append((text, want, got))
    ok &= check(not parse_bad, f"{len(parse_cases)} 種寫法全部正確（失敗 {len(parse_bad)}）")
    for text, want, got in parse_bad:
        print(f"        {text!r}: want {want}, got {got}")
    litter = [t for t in ("[30秒] 下雨", "（30秒）下雨", "【30秒】下雨")
              if any(c in mod.parse_idea_duration(t)[1] for c in "[]（）【】")]
    ok &= check(not litter, "移除秒數後不會留下空的括號殘渣")

    # ------------------------------------------------------------- messages
    print()
    print("=" * 64)
    print("6. 提示詞組裝：長片注入正確骨架、短片散文、各模式註記")
    print("=" * 64)
    long_msgs = mod.build_script_messages("測試想法", 60.0, mod.INPUT_MODE_TEXT)
    short_msgs = mod.build_script_messages("測試想法", 10.0, mod.INPUT_MODE_TEXT)
    sys_long = long_msgs[0]["content"]
    # Headings are language dependent: Simplified 开头/结尾 by default, and the
    # Traditional 開頭/結尾 form in English mode. Both must be copyable.
    head_zh, tail_zh = mod._SCRIPT_GEN_HEAD_LABELS[mod.SCRIPT_LANG_ZH]
    # Scene durations come from the varied rhythm generator, so expected headings
    # are derived from it instead of being hard-coded.
    plan_zh = mod.script_timeline_skeleton(60.0, "zh")
    first_zh = f"{plan_zh[0][0]}（{plan_zh[0][1]:g}-{plan_zh[0][2]:g}秒）："
    last_zh = f"{plan_zh[-1][0]}（{plan_zh[-1][1]:g}-{plan_zh[-1][2]:g}秒）："
    ok &= check(first_zh in sys_long and last_zh in sys_long,
                f"60 秒骨架標題已注入系統提示（{plan_zh[0][0]} … {plan_zh[-1][0]}）")
    ok &= check("（0-5秒）" not in sys_long,
                "舊的錯誤範例（0-5秒 / 50-60秒）已移除")
    ok &= check("Do NOT invent your own timings" in sys_long,
                "明確要求照抄標題、不得自行計時")
    ok &= check("LANGUAGE" in sys_long, "有注入語言區塊")
    zh_msgs = mod.build_script_messages("測試", 60.0, mod.INPUT_MODE_TEXT, "zh")
    en_msgs = mod.build_script_messages("測試", 60.0, mod.INPUT_MODE_TEXT, "en")
    zh_sys, en_sys = zh_msgs[0]["content"], en_msgs[0]["content"]
    head_en, tail_en = mod._SCRIPT_GEN_HEAD_LABELS[mod.SCRIPT_LANG_EN]
    plan_en = mod.script_timeline_skeleton(60.0, "en")
    first_en = f"{plan_en[0][0]}（{plan_en[0][1]:g}-{plan_en[0][2]:g}秒）："
    ok &= check("SIMPLIFIED CHINESE" in zh_sys and "Traditional Chinese" in zh_sys,
                "zh 模式明確要求簡體、禁用繁體")
    ok &= check("English" in en_sys and "SIMPLIFIED CHINESE" not in en_sys,
                "en 模式要求英文輸出")
    ok &= check(first_en in en_sys and f"{head_zh}（" not in en_sys,
                f"en 模式用繁體標題（{head_en}）")
    ok &= check("total duration" in long_msgs[1]["content"].lower(), "長片使用者訊息帶總秒數")
    ok &= check("paragraph" in short_msgs[0]["content"].lower(), "短片系統提示要求單段散文")
    # Assert on real heading syntax rather than the mere presence of a bracket:
    # the custom instruction block may legitimately contain full-width parens.
    ok &= check(
        not mod._SCRIPT_HEADING_OR_RANGE_RE.search(short_msgs[0]["content"])
        and not mod.TIMELINE_HEADER_RE.search(short_msgs[0]["content"]),
        "短片提示不含時間軸標題",
    )
    img_msgs = mod.build_script_messages("測試", 60.0, mod.INPUT_MODE_IMAGE)
    ok &= check("MODE NOTE" in img_msgs[0]["content"] and "first frame" in img_msgs[0]["content"],
                "I2VA 模式註記有注入")
    ref_msgs = mod.build_script_messages("測試", 60.0, mod.INPUT_MODE_REF2VA)
    ok &= check("setting, background, light" in ref_msgs[0]["content"],
                "Ref2VA 註記要求明寫場景")

    # ----------------------------------------------------------- skeleton
    print()
    print("=" * 64)
    print("6. 時間軸骨架：連續、無缺口、涵蓋全程、每幕 2–15 秒")
    print("=" * 64)
    bad = []
    for t in (16, 20, 30, 45, 60, 90, 120, 300, 900, 1800):
        plan = mod.script_timeline_skeleton(float(t))
        cursor = 0.0
        why = ""
        for label, s, e in plan:
            if abs(s - cursor) > 1e-6:
                why = f"{label} 起點 {s} != {cursor}"
                break
            if not (2 - 1e-9 <= e - s <= 15 + 1e-9):
                why = f"{label} 長度 {e - s}"
                break
            cursor = e
        if not why and abs(cursor - t) > 1e-6:
            why = f"結尾 {cursor} != {t}"
        if why:
            bad.append((t, why))
    ok &= check(not bad, f"10 種片長骨架全部合法（失敗 {len(bad)}）")
    for t, why in bad:
        print(f"        T={t}: {why}")

    print()
    print("=" * 64)
    print("6b. 長短鏡頭交替：幕長不可全部相同")
    print("=" * 64)
    flat = []
    for t in (30.0, 60.0, 120.0, 300.0):
        lens = [round(e - s, 3) for _, s, e in mod.script_timeline_skeleton(t)]
        if len(set(lens)) < 3:
            flat.append((t, lens))
    ok &= check(not flat, f"4 種片長都有 3 種以上不同幕長（失敗 {len(flat)}）")
    for t, lens in flat:
        print(f"        T={t:g}: {lens}")
    lens60 = [round(e - s, 3) for _, s, e in mod.script_timeline_skeleton(60.0)]
    spread = max(lens60) - min(lens60)
    ok &= check(spread >= 5.0, f"60 秒的長短差距 {spread:g} 秒 >= 5 秒（{lens60}）")
    uniform = [round(e - s, 3) for _, s, e in mod.script_timeline_skeleton(60.0, varied=False)]
    ok &= check(len(set(uniform)) == 1, f"varied=False 時仍是等長（{uniform[:3]}…）")
    # ------------------------------------------------- empty reply recovery
    print()
    print("=" * 64)
    print("7. 思考吃光預算時會自動加大預算重試（不直接失敗）")
    print("=" * 64)
    budgets: list[int] = []

    def fake_once(messages, max_tokens, temperature, timeout):
        budgets.append(max_tokens)
        if len(budgets) == 1:
            return "", "length", max_tokens          # thinking ate the budget
        return "開頭（0-10秒）：\nShe waits. Rain falls.", "stop", 100

    original_once = mod._llama_chat_once
    mod._llama_chat_once = fake_once
    try:
        text = mod.llama_chat([{"role": "user", "content": "x"}])
        ok &= check(text.startswith("開頭"), "第二次拿到正文")
        ok &= check(budgets == [mod.SCRIPT_GEN_MAX_TOKENS, mod.SCRIPT_GEN_RETRY_TOKENS],
                    f"預算由 {mod.SCRIPT_GEN_MAX_TOKENS} 提升到 {mod.SCRIPT_GEN_RETRY_TOKENS}")
    except Exception as exc:
        ok &= check(False, f"應該要重試成功，卻拋出 {type(exc).__name__}: {exc}")
    finally:
        mod._llama_chat_once = original_once

    budgets.clear()
    mod._llama_chat_once = lambda m, mt, t, to: ("", "length", mt)
    try:
        mod.llama_chat([{"role": "user", "content": "x"}])
        ok &= check(False, "預算用盡時應該明確報錯")
    except mod.BotError as exc:
        ok &= check("思考" in str(exc) and "token" in str(exc),
                    "錯誤訊息說明是思考吃光預算")
    finally:
        mod._llama_chat_once = original_once

    # --------------------------------------------------------- wrapper strip
    print()
    print("=" * 64)
    print("8. 清掉模型多餘的 markdown 圍欄與開場白")
    print("=" * 64)
    fenced = "```\n開頭（0-5秒）：\nShe stands.\n```"
    ok &= check(mod._strip_script_wrappers(fenced).startswith("開頭"),
                "移除 markdown 圍欄")
    chatty = "Here is your script:\n\n開頭（0-5秒）：\nShe stands."
    ok &= check(mod._strip_script_wrappers(chatty).startswith("開頭"),
                "移除開場白")

    # ------------------------------------------------------- live generation
    print()
    print("=" * 64)
    print("9. 對真實 llama.cpp 生成一次（30 秒）")
    print("=" * 64)
    if not mod.llama_is_online():
        print("  [SKIP] llama 不在線")
    else:
        import time
        t0 = time.time()
        try:
            script, log = mod.generate_h3_script(
                "夜市的霓虹燈下，一個女孩吃著章魚燒", 30.0, mod.INPUT_MODE_TEXT
            )
            dt = time.time() - t0
            plan = mod.build_long_video_plan(script, 30.0)
            ok &= check(True, f"{dt:.1f}s，通過驗證，{len(plan.shots)} 個鏡頭")
            det = mod.detect_prompt_total_seconds(script)
            ok &= check(det == 30.0, f"偵測總片長 = {det}")
            (HERE / "gen_test_30s.txt").write_text(script, encoding="utf-8")
            print(f"        已存到 gen_test_30s.txt（{len(script)} 字元）")
        except Exception as exc:
            ok &= check(False, f"生成失敗：{type(exc).__name__}: {exc}")

    print()
    print("=" * 64)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
