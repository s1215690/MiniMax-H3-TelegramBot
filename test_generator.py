"""De-risk the in-Bot script generator before wiring it into the Bot.

Asks the live llama.cpp server for a full 60 second H3 script using the same
system prompt the Bot would use, then validates the reply with the Bot's OWN
parser. Reports latency and token counts so the Telegram flow can be sized
honestly (timeout, progress message, retry budget).
"""

from __future__ import annotations

import importlib.util
import io
import contextlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA = "http://127.0.0.1:19092/v1/chat/completions"

SYSTEM = """You are a prompt writer for the MiniMax H3 video model.

Write ONE long-form script. Output the script only - no commentary, no markdown
fences, no preamble.

STRUCTURE (strict):
1. The text BEFORE the first timeline heading is the GLOBAL block. It must state,
   once, in English: the main character's visible appearance and clothing, the
   setting and lighting, the visual style, and the music that runs through the
   whole film. The character description is written once here only.
2. Then one heading per scene, in this EXACT form:
   開頭（0-5秒）：
   scene text
   Use 第一幕（5-15秒）：, 第二幕（15-25秒）： ... and end with 結尾（50-60秒）：.
3. The ranges MUST start at 0, be CONTIGUOUS (each scene starts where the last
   ended - no gaps, no overlaps), and the final range MUST end exactly at the
   requested total duration. Never write a range longer than 15 seconds.
4. Put a blank line between every scene.

SCENE TEXT RULES:
- English only for scene and camera descriptions.
- Describe what happens in chronological order, with concrete visible actions
  (never "she dances" - always "she raises her left hand and turns half a
  circle, her skirt lifting").
- Include the camera as natural prose ("the camera slowly pushes in").
- End every scene with one short sound sentence (ambience, physical sounds,
  music, dialogue). Audio is generated jointly with the picture.
- Describe ONLY what happens in that scene. Never describe future events.
- Never write media tags like <Picture 1>, <Video 1> or <Audio 1>.
"""

USER = (
    "Total duration: 60 seconds.\n"
    "Idea: A woman waits on a rainy train platform at night and lets a train "
    "leave without her.\n"
    "Write the full script now."
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


def main() -> int:
    bot = load_bot()
    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "max_tokens": 3000,
        "temperature": 0.8,
        "top_p": 0.95,
        "stream": False,
    }
    req = urllib.request.Request(
        LLAMA,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print("requesting a 60 second script from llama.cpp ...")
    start = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - start

    choice = data["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    usage = data.get("usage", {})
    ct = usage.get("completion_tokens", 0)

    print(f"latency      : {elapsed:.1f}s")
    print(f"tokens       : {ct} completion, {usage.get('prompt_tokens',0)} prompt")
    if elapsed > 0 and ct:
        print(f"throughput   : {ct/elapsed:.1f} tok/s")
    print(f"finish       : {choice.get('finish_reason')}")
    print(f"chars        : {len(text)}")
    print()

    out = HERE / "gen_test_60s.txt"
    out.write_text(text, encoding="utf-8")
    print(f"--- raw output saved to {out.name} ---")
    print(text[:600])
    print("... [truncated] ...")
    print()

    print("=" * 60)
    print("VALIDATION with the Bot's own parser")
    print("=" * 60)
    tl = bot.parse_timeline_prompt(text)
    print(f"  timeline scenes : {len(tl.scenes) if tl else 0}")
    if tl:
        for s in tl.scenes:
            print(f"     {s.start_seconds:6.1f} - {s.end_seconds:6.1f}  {s.label}")
        print(f"  global block    : {len(tl.global_text)} chars")
    try:
        plan = bot.build_long_video_plan(text, 60.0)
        print(f"  RESULT: PASS -> {len(plan.shots)} shots")
        return 0
    except Exception as exc:
        print(f"  RESULT: FAIL -> {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
