#!/usr/bin/env python3
"""H3 prompt formatter / validator for the MiniMax H3 Telegram Bot.

Why this exists
---------------
The Telegram Bot is strict about long-form ("長片") prompts. It will refuse a
script outright when the timeline has a gap, an overlap, or when a scene cannot
be split into >= 2 second shots. Dropping a community tool's output into the Bot
usually fails for one of two reasons:

  1. **Over-specified scripts.** A draft writes `SEGMENT 1 (0-20s):` and then
     nests `Shot 1 (0-7s):`, `Shot 2 (7-14s):` inside it. The Bot's timeline
     parser is FLAT -- every heading becomes a top-level scene -- so the nested
     shots collide with their own parent (`0-7` overlaps `0-20`).

  2. **Official MiniMax format.** Skills, ComfyUI node packs and the hosted
     rewriter emit `integrated_multimodal_description:` / `[Shot N]` /
     `<Picture N>` sections. The Bot understands none of that.

This tool repairs both, then validates using the Bot's OWN parser, so a PASS
here means the Bot will accept the prompt.

Usage
-----
    python h3_prompt_format.py check <file>              # diagnose only
    python h3_prompt_format.py fix   <file>              # repair, write .bot.txt
    python h3_prompt_format.py fix   <file> -o out.txt
    python h3_prompt_format.py fix   <file> --total 60   # override duration
    python h3_prompt_format.py official <file>           # official -> Bot format

Every mode prints a report. `fix` / `official` exit non-zero if the result still
fails the Bot's validator.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import contextlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOT_PATH = HERE / "MiniMax-H3-Telegram-Bot.py"

# Headings that only subdivide an already-timed parent block. These are the ones
# that must be flattened away, because the Bot splits scenes into shots itself.
_SUB_HEADING_WORD_RE = re.compile(r"SHOT|鏡頭|镜头|CUT|BEAT", re.IGNORECASE)


# --------------------------------------------------------------------------
# Bot parser loader
# --------------------------------------------------------------------------
def load_bot():
    """Import the Bot module so validation uses its exact rules.

    The Bot script is not a library, so it is loaded by path. It must be
    registered in sys.modules before exec_module() because its frozen
    dataclasses resolve their module namespace there.
    """
    if not BOT_PATH.is_file():
        raise SystemExit(f"找不到 Bot 主程式：{BOT_PATH}")
    spec = importlib.util.spec_from_file_location("h3bot", BOT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3bot"] = mod
    # Importing runs module-level setup; swallow anything it prints.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Detection helpers
# --------------------------------------------------------------------------
_OFFICIAL_FIELD_RE = re.compile(
    r"(?im)^[ \t]*(integrated_multimodal_description|overall_soundscape|"
    r"non_diegetic_music|subject_definitions|summary|retention_analysis|"
    r"detailed_description)[ \t]*[:：]"
)
_MEDIA_TAG_RE = re.compile(
    r"<[ \t]*(Picture|Video|Audio|Image)[ \t]*[0-9]+[ \t]*>", re.IGNORECASE
)
_SHOT_MARKER_RE = re.compile(r"\[[ \t]*(?:Shot|镜头|鏡頭)[ \t]*([0-9]+)[^\]]*\]", re.IGNORECASE)
# `[Shot 1: 00:00.000 – 00:02.800]` style explicit timings.
_SHOT_MARKER_TIME_RE = re.compile(
    r"\[[ \t]*(?:Shot|镜头|鏡頭)[ \t]*([0-9]+)[ \t]*[:：][ \t]*"
    r"(?P<start>[0-9]{1,2}:[0-9]{2}(?:\.[0-9]+)?)"
    r"[ \t]*(?:-|‐|‑|‒|–|—|−|~|～|至|到)[ \t]*"
    r"(?P<end>[0-9]{1,2}:[0-9]{2}(?:\.[0-9]+)?)[ \t]*\]",
    re.IGNORECASE,
)


def looks_official(text: str) -> bool:
    return bool(_OFFICIAL_FIELD_RE.search(text))


def _tidy(text: str) -> str:
    """Clean up the scars left behind by removing media tags and field names.

    Deleting `<Picture 1>` from mid-sentence leaves artifacts such as a doubled
    space, a space before a comma, or a stranded comma (`platform, keeps ...`).
    Only whitespace and punctuation runs are touched -- no wording is changed.
    """
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r",[ \t]*(?=[,.;])", "", text)
    text = re.sub(r"\(\s*\)", "", text)          # emptied parentheses
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _mmss(value: str) -> float:
    """Parse `MM:SS.mmm` or `HH:MM:SS.mmm` into seconds."""
    parts = value.split(":")
    total = 0.0
    for part in parts:
        total = total * 60.0 + float(part)
    return total


# --------------------------------------------------------------------------
# Repair: flatten nested sub-headings
# --------------------------------------------------------------------------
def flatten_nested_headings(text: str, bot) -> tuple[str, list[str]]:
    """Flatten headings that subdivide an already-timed parent block.

    The prose under each nested heading is deliberately KEPT. Because the
    surviving parent scene absorbs every line beneath it, the parent ends up
    holding all of its shots' descriptions in the original order -- which is
    exactly what the Bot wants, since it does its own shot splitting.

    Crucially, drafts very often put the description ON THE SAME LINE as the
    heading (`Shot 1 (0-7s): Wide establishing shot. The camera...`). Deleting
    such a line would destroy the prompt, so an inline description is preserved
    as its own line and only a genuinely empty heading is removed.

    Returns the repaired text plus a log of what changed.
    """
    normalized = bot.normalize_inline_headers(text)
    matches = list(bot.TIMELINE_HEADER_RE.finditer(normalized))
    if not matches:
        return text, []

    spans = []
    for m in matches:
        spans.append((m, float(m.group("start")), float(m.group("end"))))

    drop: set[int] = set()
    log: list[str] = []
    for i, (m, start, end) in enumerate(spans):
        label = m.group("label").strip()
        if not _SUB_HEADING_WORD_RE.search(label):
            continue  # only shot/cut style headings are ever flattened away
        # A heading is nested when some EARLIER heading strictly contains its
        # range. Every earlier heading is examined: an intervening sibling has
        # already closed by now, but the enclosing parent may still be open.
        for j in range(i - 1, -1, -1):
            pm, pstart, pend = spans[j]
            if pstart <= start and end <= pend and (pstart, pend) != (start, end):
                drop.add(i)
                log.append(
                    f"移除嵌套子標題 {label!r}（{start:g}-{end:g}s，"
                    f"被 {pm.group('label').strip()!r} {pstart:g}-{pend:g}s 包含）"
                )
                break

    if not drop:
        return text, []

    lines = normalized.split("\n")
    replacements: dict[int, str | None] = {}
    for i in sorted(drop):
        m = spans[i][0]
        line_no = normalized.count("\n", 0, m.start())
        if line_no in replacements:
            continue  # two headings on one line: handled by the first
        inline = m.group("inline").strip()
        replacements[line_no] = inline or None

    kept_inline = sum(1 for v in replacements.values() if v)
    out: list[str] = []
    for idx, line in enumerate(lines):
        if idx in replacements:
            new = replacements[idx]
            if new:
                out.append(new)  # keep the prose that shared the heading line
            continue  # drop the heading itself
        out.append(line)
    if kept_inline:
        log.append(f"其中 {kept_inline} 行的行內正文已保留（未連同標題一起刪除）")
    return "\n".join(out), log


# --------------------------------------------------------------------------
# Repair: force a contiguous timeline
# --------------------------------------------------------------------------
def linearize(text: str, bot, total: float | None) -> tuple[str, list[str]]:
    """Rebuild a gap/overlap-free timeline from whatever headings exist.

    Used when a script is over-timed in ways flattening cannot fix (for example
    a flat list of 4-second shots that the user wants spread over 60 seconds).
    Each heading becomes one scene; if the declared ranges are already a clean
    contiguous run they are preserved, otherwise the scenes are re-timed evenly
    and each is split so no scene exceeds the Bot's 15 second ceiling.
    """
    normalized = bot.normalize_inline_headers(text)
    matches = list(bot.TIMELINE_HEADER_RE.finditer(normalized))
    if not matches:
        return text, []

    preamble = normalized[: matches[0].start()].strip()
    scenes: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
        inline = m.group("inline").strip()
        body = normalized[m.end():body_end].strip()
        action = "\n".join(p for p in (inline, body) if p).strip()
        label = m.group("label").strip() or f"場景 {i + 1}"
        scenes.append((label, action))

    if not scenes:
        return text, []

    declared = [(float(m.group("start")), float(m.group("end"))) for m in matches]
    clean = _is_contiguous(declared, total)
    if clean:
        return text, []

    log = ["時間軸不連續，改為依序平均分配"]
    count = len(scenes)
    # Keep every scene under the Bot's per-scene ceiling.
    max_scene = bot.MAX_SEGMENT_SECONDS
    per = min(max_scene, total / count) if total else max_scene
    per = max(per, bot.MIN_TOTAL_SECONDS)

    blocks: list[str] = []
    if preamble:
        blocks.append(preamble)
    cursor = 0.0
    for i, (label, action) in enumerate(scenes):
        start = cursor
        end = total if i == count - 1 else round(cursor + per, 3)
        if end - start < bot.MIN_TOTAL_SECONDS:
            end = start + bot.MIN_TOTAL_SECONDS
        blocks.append(f"{label}（{start:g}-{end:g}秒）：\n{action}")
        cursor = end
    return "\n\n".join(blocks) + "\n", log


def _is_contiguous(ranges: list[tuple[float, float]], total: float | None, tol: float = 0.25) -> bool:
    if not ranges:
        return False
    cursor = 0.0
    for start, end in ranges:
        if abs(start - cursor) > tol:
            return False
        if end <= start:
            return False
        cursor = end
    if total is not None and abs(cursor - total) > tol:
        return False
    return True


# --------------------------------------------------------------------------
# Official -> Bot conversion
# --------------------------------------------------------------------------
def convert_official(text: str, total: float) -> tuple[str, list[str]]:
    """Rewrite official MiniMax H3 structured prose into the Bot's timeline.

    Official format carries media tags (`<Picture 1>`) and an alignment
    instruction line, both of which the Bot manages itself and rejects when
    hand-written. The soundscape and music fields belong in the Bot's GLOBAL
    block (the text before the first timeline heading).
    """
    log: list[str] = []
    text = text.replace("\r\n", "\n")

    tags = sorted(set(_MEDIA_TAG_RE.findall(text)))
    if tags:
        log.append("移除手寫媒體標籤：" + ", ".join(tags))
    body = _MEDIA_TAG_RE.sub("", text)

    # Drop the keyframe alignment instruction (the Bot's own instructions own it).
    body = re.sub(
        r"(?im)^[ \t]*(?:For the target video[^\n]*|"
        r"How the reference pictures align[^\n]*)",
        "",
        body,
    )

    def field(name: str) -> str:
        m = re.search(
            rf"(?ims)^[ \t]*{name}[ \t]*[:：][ \t]*(.*?)(?=^[ \t]*[a-z_]+[ \t]*[:：]|\Z)",
            body,
        )
        return m.group(1).strip() if m else ""

    integrated = field("integrated_multimodal_description")
    soundscape = field("overall_soundscape")
    music = field("non_diegetic_music")
    # Reference-mode sections: keep only the descriptive ones as global context.
    subject_defs = field("subject_definitions")
    detail = field("detailed_description")

    if not integrated:
        integrated = detail or body
        log.append("找不到 integrated_multimodal_description，改用全文作為動作描述")

    # Split the description into shots.
    markers = list(_SHOT_MARKER_TIME_RE.finditer(integrated))
    if markers:
        shots = []
        for i, m in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(integrated)
            body_i = integrated[m.end():end].strip()
            shots.append((_mmss(m.group("start")), _mmss(m.group("end")), body_i))
        log.append(f"讀取 {len(shots)} 個帶明確時間的 [Shot N]")
    else:
        flat = list(_SHOT_MARKER_RE.finditer(integrated))
        if flat:
            chunks = []
            for i, m in enumerate(flat):
                end = flat[i + 1].start() if i + 1 < len(flat) else len(integrated)
                chunk = integrated[m.end():end].strip()
                if chunk:
                    chunks.append(chunk)
            log.append(f"讀取 {len(chunks)} 個未標時間的 [Shot N]，平均分配 {total:g} 秒")
        else:
            chunks = [c.strip() for c in re.split(r"\n[ \t]*\n", integrated) if c.strip()]
            log.append(f"無 [Shot N] 標記，按空行切成 {len(chunks)} 幕")
        step = total / max(1, len(chunks))
        shots = [
            (round(i * step, 3), round(min((i + 1) * step, total), 3), c)
            for i, c in enumerate(chunks)
        ]

    if not shots:
        shots = [(0.0, total, integrated.strip())]

    # ---- assemble global block -------------------------------------------
    global_parts: list[str] = []
    if subject_defs:
        global_parts.append(subject_defs)
    if detail and detail not in global_parts:
        global_parts.append(detail)
    if soundscape:
        global_parts.append("Audio: " + soundscape)
    if music:
        global_parts.append("Music: " + music)

    blocks: list[str] = []
    if global_parts:
        blocks.append("\n\n".join(global_parts))
    for i, (start, end, action) in enumerate(shots):
        label = "開頭" if i == 0 else ("結尾" if i == len(shots) - 1 else f"第{i}幕")
        blocks.append(f"{label}（{start:g}-{end:g}秒）：\n{action}")
    return _tidy("\n\n".join(blocks)) + "\n", log


# --------------------------------------------------------------------------
# Validation against the Bot
# --------------------------------------------------------------------------
def strip_heading_labels(text: str, bot) -> str:
    """Remove every timeline heading's label/range, keeping only the prose.

    Used as the baseline for the retention check: heading labels are *supposed*
    to disappear, so counting them as "lost content" would flag every correct
    repair. What must survive is the descriptive prose, including prose that
    shared a line with its heading.
    """
    normalized = bot.normalize_inline_headers(text)
    out = normalized
    for m in reversed(list(bot.TIMELINE_HEADER_RE.finditer(normalized))):
        whole = m.group(0)
        inline = m.group("inline")
        prefix = whole[: len(whole) - len(inline)] if inline else whole
        out = out[: m.start()] + inline + out[m.end():]
    return out


def retention_baseline(text: str, bot, mode: str) -> str:
    """Reduce the input to just the content that a rewrite must preserve.

    Both rewrite paths deliberately discard scaffolding: `fix` removes heading
    labels, `official` removes field names, the keyframe instruction line,
    `[Shot N]` markers and media tags. Measuring retention against the untouched
    source would flag every correct rewrite, so those are stripped first.
    """
    if mode == "official":
        base = _MEDIA_TAG_RE.sub(" ", text)
        base = re.sub(
            r"(?im)^[ \t]*(?:For the target video[^\n]*|"
            r"How the reference pictures align[^\n]*)",
            " ",
            base,
        )
        base = _SHOT_MARKER_RE.sub(" ", base)
        base = re.sub(r"(?im)^[ \t]*[a-z_]+[ \t]*[:：]", " ", base)
        return base
    return strip_heading_labels(text, bot)


def content_retention(before: str, after: str) -> tuple[float, list[str]]:
    """Measure how much of the original wording survived a rewrite.

    Flattening headings is supposed to remove *structure*, never *content*.
    This compares the multiset of word tokens so a regression that silently
    drops a description (for example by deleting a line whose prose shared the
    heading line) is caught immediately instead of producing a valid-looking
    but empty prompt.
    """
    token_re = re.compile(r"[A-Za-z0-9']+|[\u4e00-\u9fff]", re.UNICODE)

    def bag(s: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in token_re.findall(s.lower()):
            out[t] = out.get(t, 0) + 1
        return out

    b, a = bag(before), bag(after)
    total = sum(b.values())
    if not total:
        return 1.0, []
    missing = []
    for tok, n in b.items():
        lost = n - a.get(tok, 0)
        if lost > 0:
            missing.append((lost, tok))
    missing.sort(reverse=True)
    lost_total = sum(n for n, _ in missing)
    return 1.0 - lost_total / total, [t for _, t in missing[:12]]


def validate(bot, text: str, total: float | None) -> tuple[bool, str, list]:
    detected = bot.detect_prompt_total_seconds(text)
    use_total = total or detected
    if use_total is None:
        return False, "無法判斷總片長；請用 --total 指定。", []
    try:
        plan = bot.build_long_video_plan(text, float(use_total))
    except Exception as exc:  # BotError and friends
        return False, f"{type(exc).__name__}: {exc}", []
    return True, f"總片長 {use_total:g} 秒，切分為 {len(plan.shots)} 個鏡頭", list(plan.shots)


def describe(bot, text: str) -> None:
    tl = bot.parse_timeline_prompt(text)
    sg = bot.parse_segmented_prompt(text)
    print(f"  官方格式特徵        : {'是' if looks_official(text) else '否'}")
    print(f"  時間軸標題數        : {len(tl.scenes) if tl else 0}")
    print(f"  GLOBAL/SEGMENT 區塊 : {sorted(sg.segments) if sg else '無'}")
    print(f"  自動偵測總片長      : {bot.detect_prompt_total_seconds(text)}")
    if tl and tl.global_text:
        print(f"  全局文字長度        : {len(tl.global_text)} 字元")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"  已寫入：{path}  ({len(text.encode('utf-8'))} bytes)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把提示詞整理成 MiniMax H3 Telegram Bot 接受的格式，並用 Bot 自己的解析器驗證。"
    )
    ap.add_argument("mode", choices=["check", "fix", "official"])
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="輸出檔案（預設 <input>.bot.txt）")
    ap.add_argument("--total", type=float, help="總片長秒數；預設自動偵測")
    args = ap.parse_args(argv)

    if not args.input.is_file():
        print(f"找不到檔案：{args.input}", file=sys.stderr)
        return 2

    text = args.input.read_text(encoding="utf-8")
    bot = load_bot()

    print("=" * 68)
    print(f"輸入：{args.input}")
    print("=" * 68)
    describe(bot, text)

    ok, msg, shots = validate(bot, text, args.total)
    print(f"  Bot 驗證            : {'通過' if ok else '拒絕'}")
    print(f"    {msg}")
    for s in shots:
        print(f"      {s.start_seconds:7.2f} - {s.end_seconds:7.2f}  ({s.end_seconds - s.start_seconds:.2f}s)  {s.label}")

    if args.mode == "check":
        if not ok:
            print()
            print("建議：執行 `fix` 修復，或 `official` 轉換官方格式。")
        return 0 if ok else 1

    if args.mode == "official":
        total = args.total or bot.detect_prompt_total_seconds(text) or 60.0
        print()
        print("--- 轉換官方格式 ---")
        out, log = convert_official(text, float(total))
    else:
        print()
        print("--- 修復 ---")
        out, log = flatten_nested_headings(text, bot)
        if not log:
            print("  沒有發現嵌套子標題。")
        for line in log:
            print("  " + line)
        ok2, msg2, _ = validate(bot, out, args.total)
        if not ok2:
            print(f"  攤平後仍未通過（{msg2}），改為重建時間軸…")
            total = args.total or bot.detect_prompt_total_seconds(out) or 60.0
            out, log2 = linearize(out, bot, float(total))
            for line in log2:
                print("  " + line)

    for line in log if args.mode == "official" else []:
        print("  " + line)

    print()
    print("--- 驗證輸出 ---")
    ok3, msg3, shots3 = validate(bot, out, args.total)
    print(f"  {'通過' if ok3 else '拒絕'}：{msg3}")
    for s in shots3:
        print(f"      {s.start_seconds:7.2f} - {s.end_seconds:7.2f}  ({s.end_seconds - s.start_seconds:.2f}s)  {s.label}")

    kept, sample = content_retention(retention_baseline(text, bot, args.mode), out)
    flag = "OK" if kept >= 0.99 else "警告"
    print(f"  正文保全度          : {kept:.1%}  [{flag}]  （已排除刻意移除的標題字）")
    if kept < 0.99:
        print(f"    遺失字詞樣本：{', '.join(sample)}")
    if args.mode == "fix" and kept < 0.99:
        print("    ⚠️ 修復過程疑似刪掉了正文，請勿直接使用此輸出。")
        ok3 = False

    dest = args.out or args.input.with_suffix(args.input.suffix + ".bot.txt")
    _write(dest, out)
    return 0 if ok3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
