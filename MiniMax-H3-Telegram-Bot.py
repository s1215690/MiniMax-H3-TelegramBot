#!/usr/bin/env python3
"""Telegram controller for the local MiniMax H3 Turbo ComfyUI workflow.

The bot accepts generation parameters and a prompt from one authorized chat,
submits an API-format workflow to ComfyUI, then sends the synchronized MP4
back to Telegram. Secrets are intentionally read only from environment
variables and are never written to this workspace.
"""

from __future__ import annotations

import hashlib
import json
import asyncio
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


COMFY_URL = os.environ.get("MINIMAX_COMFY_URL", "http://127.0.0.1:8191").rstrip("/")
OUTPUT_DIR = Path(
    os.environ.get(
        "MINIMAX_COMFY_OUTPUT",
        r"E:\MiniMax-H3-Telegram\output",
    )
)
INPUT_DIR = Path(
    os.environ.get(
        "MINIMAX_COMFY_INPUT",
        str(OUTPUT_DIR.parent / "input"),
    )
)
T8_API_TEMPLATE = Path(
    os.environ.get(
        "MINIMAX_T8_API_TEMPLATE",
        str(Path(__file__).resolve().parent / "dual_clock_multirate_api.json"),
    )
)
SEEDVR2_API_TEMPLATE = Path(
    os.environ.get(
        "MINIMAX_SEEDVR2_API_TEMPLATE",
        str(Path(__file__).resolve().parent / "seedvr2_3b_int8_upscale_video_api.json"),
    )
)
COMFYUI_DIR = Path(
    os.environ.get("MINIMAX_COMFY_DIR", r"E:\Comfy\ComfyUI\ComfyUI-Turbo")
)
COMFYUI_BASE_DIR = Path(
    os.environ.get("MINIMAX_COMFY_BASE_DIR", r"E:\Comfy\ComfyUI\ComfyUI")
)
COMFYUI_PYTHON = Path(
    os.environ.get(
        "MINIMAX_COMFY_PYTHON",
        r"E:\Comfy\ComfyUI\ComfyUI\.venv\Scripts\python.exe",
    )
)
COMFYUI_PORT = int(os.environ.get("MINIMAX_COMFY_PORT", "8191"))
try:
    COMFY_IDLE_SHUTDOWN_SECONDS = max(
        0.0,
        float(os.environ.get("MINIMAX_COMFY_IDLE_SHUTDOWN_SECONDS", "300")),
    )
except ValueError:
    COMFY_IDLE_SHUTDOWN_SECONDS = 300.0
COMFY_IDLE_CHECK_INTERVAL_SECONDS = 15.0
COMFYUI_LOG = Path(
    os.environ.get(
        "MINIMAX_COMFY_LOG",
        r"E:\MiniMax-H3-Telegram\runtime\bot\comfyui.log",
    )
)
COMFYUI_STATE_DIR = Path(
    os.environ.get(
        "MINIMAX_COMFY_STATE_DIR",
        r"E:\MiniMax-H3-Telegram\runtime\comfyui",
    )
)
COMFYUI_USER_DIR = COMFYUI_STATE_DIR / "user"
COMFYUI_DATABASE = COMFYUI_STATE_DIR / "comfyui.db"
DEFAULT_COMFYUI_VRAM_MODE = "lowvram"
# NOTE (2026-09-15): a full dual-GPU investigation was run and then reverted.
# Both RTX 3080s are visible to ComfyUI just fine, but every generation path
# aborts: comfy-aimdo's multi-device hostbuf copy kills the process inside its
# native DLL ("hostbuf_read_file_slice: device copy failed", uncatchable from
# Python), and H3VM - which does correctly detect the pair and split the model -
# either access-violates in ModelPatcher.deepclone_multigpu() or stalls with the
# second GPU idle. Details, evidence and the conditions that would make a retry
# worthwhile are in 雙卡調查報告.md. ComfyUI runs single-GPU by design here.
FFMPEG_PATH = os.environ.get("MINIMAX_FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
FFPROBE_PATH = os.environ.get("MINIMAX_FFPROBE", shutil.which("ffprobe") or "ffprobe")
NVIDIA_SMI_PATH = os.environ.get(
    "MINIMAX_NVIDIA_SMI",
    shutil.which("nvidia-smi")
    or r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)
# --- Local LLM (llama-server) control -------------------------------------
# Preset parameters captured from the running Qwen3.8-27B server (2026-09-05),
# so the Telegram panel can start/stop the model with the exact same settings
# as the local Llama dashboard.
#
# 2026-09-11: re-captured from the live EfficientThink + MTP-draft server so the
# Bot brings back the SAME model the user actually runs, instead of the older
# UD-Q6_K_XL preset. The two differed in build (10448 vs 10830), model, mmproj,
# context window, and speculative-decoding strategy.
LLAMA_URL = os.environ.get("MINIMAX_LLAMA_URL", "http://127.0.0.1:19092").rstrip("/")
LLAMA_PORT = int(os.environ.get("MINIMAX_LLAMA_PORT", "19092"))
LLAMA_HOST = os.environ.get("MINIMAX_LLAMA_HOST", "127.0.0.1")
# llama.cpp now ships a thin llama-server.exe launcher plus llama-server-impl.dll,
# so a ~9 KB executable here is expected, not a broken download.
LLAMA_EXE = Path(
    os.environ.get(
        "MINIMAX_LLAMA_EXE", r"D:\llama.cpp-b10830\llama-server.exe"
    )
)
LLAMA_MODEL = Path(
    os.environ.get(
        "MINIMAX_LLAMA_MODEL",
        r"D:\models\qwen38-efficientthink-q6\Qwen3.8-27B-EfficientThink-SimPO-Q6_K.gguf",
    )
)
LLAMA_MMPROJ = Path(
    os.environ.get(
        "MINIMAX_LLAMA_MMPROJ",
        r"D:\models\qwen38-efficientthink-q6\mmproj-Qwen3.8-27B-Q8_0.gguf",
    )
)
# MTP draft head for speculative decoding. `--spec-type draft-mtp` is useless
# without it, so it is validated next to the main model in start_llama_process().
LLAMA_DRAFT_MODEL = Path(
    os.environ.get(
        "MINIMAX_LLAMA_DRAFT_MODEL",
        r"D:\models\qwen38-mtp\mtp-Qwen3.8-27B-Q4_0.gguf",
    )
)
LLAMA_SLOT_CACHE = Path(
    os.environ.get("MINIMAX_LLAMA_SLOT_CACHE", r"D:\llama.cpp\slot-cache")
)
LLAMA_LOG = Path(
    os.environ.get(
        "MINIMAX_LLAMA_LOG",
        r"E:\MiniMax-H3-Telegram\runtime\bot\llama-server.log",
    )
)
# Machine-specific placement for this 2 x RTX 3080 (20 GB) box. The main model is
# split across both cards and the small MTP draft head lives on the second one.
LLAMA_CTX = int(os.environ.get("MINIMAX_LLAMA_CTX", "230000"))
LLAMA_DEVICE = os.environ.get("MINIMAX_LLAMA_DEVICE", "CUDA0,CUDA1")
LLAMA_DRAFT_DEVICE = os.environ.get("MINIMAX_LLAMA_DRAFT_DEVICE", "CUDA1")
LLAMA_TENSOR_SPLIT = os.environ.get("MINIMAX_LLAMA_TENSOR_SPLIT", "1,1")
# The bot stops the local LLM before every job to free VRAM. By default it puts
# the LLM back once the job is done, so Hermes/DSH local models work again.
# MINIMAX_LLM_RESTART_AFTER_JOB=0 leaves it down until started by hand.
RESTART_LLM_AFTER_GENERATION = os.environ.get(
    "MINIMAX_LLM_RESTART_AFTER_JOB", "1"
).strip().lower() not in {"0", "false", "off", "no"}
LLAMA_PRESET_ARGS: list[str] = [
    "-m", str(LLAMA_MODEL),
    "--jinja",
    "--metrics",
    "-c", str(LLAMA_CTX),
    "--parallel", "1",
    "-ngl", "99",
    "--host", LLAMA_HOST,
    "--port", str(LLAMA_PORT),
    "--slot-save-path", str(LLAMA_SLOT_CACHE),
    "--temp", "1",
    "--top-p", "0.95",
    "--top-k", "20",
    # Speculative decoding. The MTP draft head must be named explicitly: with
    # `--spec-type draft-mtp` alone llama.cpp has no draft model to run, so the
    # acceleration is silently lost. The ngram-mod fallback the old preset used
    # is NOT part of the live EfficientThink setup and has been dropped.
    "--spec-type", "draft-mtp",
    "--spec-draft-model", str(LLAMA_DRAFT_MODEL),
    "--spec-draft-n-max", "3",
    "--spec-draft-ngl", "99",
    "--spec-draft-device", LLAMA_DRAFT_DEVICE,
    "--mmproj", str(LLAMA_MMPROJ),
    "--image-min-tokens", "1024",
    "--reasoning", "on",
    "--reasoning-effort", "low",
    "--flash-attn", "on",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--split-mode", "tensor",
    # Pin GPU placement explicitly so a driver enumeration change cannot quietly
    # move the model or the draft head onto the wrong card.
    "--device", LLAMA_DEVICE,
    "--tensor-split", LLAMA_TENSOR_SPLIT,
    "--fit", "off",
    "--no-warmup",
]
SHUTDOWN_DELAY_SECONDS = 60
MAX_TELEGRAM_IMAGE_BYTES = 20 * 1024 * 1024
# Telegram Bot API currently accepts at most 50 MB for a bot-uploaded video.
# Keep a margin for multipart/form-data headers and the request boundary.
TELEGRAM_MAX_VIDEO_BYTES = 50_000_000
TELEGRAM_SAFE_VIDEO_BYTES = 48_000_000
TELEGRAM_AUDIO_BITRATE_KBPS = 128
MAX_TELEGRAM_PROMPT_BYTES = 512 * 1024
PROMPT_FILE_EXTENSIONS = {".txt", ".text"}
PROMPT_HELP_TEXT = """📝 MiniMax H3 提示詞精華

【通用原則】
• 按時間順序詳細描述，動作寫具體：不寫「跳舞」，寫「舉起左手轉了半圈，裙擺揚起」
• 四要素：主體（外貌／服裝）→ 場景（地點／光線／天氣）→ 動作 → 運鏡（自然散文，如 "the camera slowly pushes in"）
• 聲音一起寫：音樂、環境音、對白、音效（音畫聯合生成，不寫聲音會隨機）
• 動作描述建議用英文；不要寫「接下來會發生」；不要手寫 <Picture N> 等媒體標籤（Bot 自動管理）

【短片 T2V】
[主體] + [場景] + [按秒推進的動作] + [運鏡] + [聲音]

【圖片生視頻 I2V】
caption 只寫：接下來做什麼動作 + 運鏡 + 聲音。不要重寫圖片已經看得見的外觀。

【FL2VA】
只寫首幀 → 尾幀的中間過程，開頭結尾姿勢不用重複。

【Ref2VA】
素材是參考不是重播：一句「keeping the exact appearance of the reference」+ 新動作。
Bot 會自動加入「只鎖人物外貌、不帶走參考圖的場景/背景/光線/機位」規則，
避免整段跳成參考圖的場景；場景、背景、光線請寫在提示詞裡。

【長片時間軸】
標題格式：場景名（開始秒-結束秒）：例如 開頭（0-5秒）：
• 必須由 0 秒開始、連續寫到腳本最大秒數；缺口、重疊都會被拒絕
• 每一幕 2–15 秒（Bot 自動拆成 ≤8 秒鏡頭）
• 第一個時間標題之前的文字 = 全局設定（人物外貌／場景／風格／音樂，只寫一次，每鏡自動附上）
• Bot 會自動讀取時間軸的最大結束秒數，不需要每次手動選片長

【長片 GLOBAL／SEGMENT】
GLOBAL: 人物外貌、場景、光線、風格、音樂
SEGMENT 1: 只寫第 1 段動作（2–15 秒）
SEGMENT 2: ……（編號必須從 1 連續）

【聲音速查】
音樂：曲風＋樂器＋節奏；環境音：地點＋聲源；對白：直接寫句子；全片統一音樂寫進 GLOBAL；要無聲須明寫 "completely silent"
對白語言：Bot 預設自動鎖普通話；要粵語／其他口音，在提示詞明寫（例如「對白用粵語」）

完整範例見專案資料夾 MiniMax-H3-Prompt-Template.md。"""

SAGE_ATTENTION_ENABLED = os.environ.get("MINIMAX_SAGE_ATTENTION", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
CLIP_NAME = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
UNET_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
LORA_NAME = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
# ---- model profiles for the stock (non-YUPI) path --------------------------
# classic: the split FL2VA / Ref2VA checkpoints + the turbo LoRA (original
#          behaviour, untouched).
# fused:   MATLOWAI's single fused checkpoint — pruned fl2va with a rank-1024
#          (ref2va - fl2va) delta, lightx2v turbo-8 @1.0 and Mystic v2.0 @0.7
#          all baked into the weights. One file serves T2VA/I2VA/FL2VA/Ref2VA,
#          needs NO LoRA loader (and therefore no ~21GB unpatch backup), and is
#          built to run at 4 steps. H3 SLA sparse attention is inserted when
#          the pack is installed.
H3_PROFILE_CLASSIC = "classic"
H3_PROFILE_FUSED = "fused"
H3_PROFILES = (H3_PROFILE_CLASSIC, H3_PROFILE_FUSED)
H3_PROFILE_DEFAULT = H3_PROFILE_FUSED
FUSED_UNET_NAME = os.environ.get(
    "MINIMAX_H3_FUSED_UNET",
    "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors",
).strip()
FUSED_PROFILE_STEPS = 6


def _fused_profile_steps() -> int:
    """Sampler steps for the fused bake (4 = the step count it was built for).

    Set MINIMAX_H3_FUSED_STEPS to override (e.g. 6) when comparing speed vs
    quality without editing this file.
    """
    try:
        value = int(os.environ.get("MINIMAX_H3_FUSED_STEPS", "").strip())
    except ValueError:
        value = 0
    return value if value > 0 else 4


FUSED_PROFILE_STEPS = _fused_profile_steps()
SLA_NODE_CLASS = "H3SLAAttention"
SLA_NODE_ID = "900"
OUTPUT_PREFIX = "MiniMaxH3/Telegram_Turbo"
# ---- YUPI workflow (isolated NSFW Ref2VA path) -----------------------------
# A separate, self-contained generation path that does NOT touch the stock
# Turbo pipeline. It loads yupi_nsfw_api.json (Ref2VA + AfterMidnight NSFW
# LoRA, euler/beta) and is triggered by the "YUPI工作流" button.
YUPI_API_TEMPLATE = Path(__file__).resolve().parent / "yupi_nsfw_api.json"
YUPI_OUTPUT_PREFIX = "MiniMaxH3/YUPI_NSFW"
YUPI_BUTTON = "🌙 YUPI工作流（6步）"
YUPI_TASK_TYPE = "yupi"
YUPI_LORA_NAME = os.environ.get(
    "MINIMAX_YUPI_LORA",
    "Vagina_minimax-h3_epoch20.safetensors",
).strip()
# Anatomy adapters (stills-trained) sit under the action adapter. Trigger
# tokens live in the prompt: 'pussy' for HMPussy, 'hmmotion' for HMNSFW.
YUPI_ACTION_LORA_NAME = os.environ.get(
    "MINIMAX_YUPI_ACTION_LORA",
    "HMNSFW-AIO-V2.5.safetensors",
).strip()
# ---- YUPI_FAST variant (FastH3 6-step distill, still euler/beta) -----------
# YUPI now runs this variant by default (it replaced the 20-step YUPI graph).
# It loads yupi_fast_api.json, which chains anatomy -> action -> FastH3 and
# runs the scheduler at 6 steps. The 20-step graph (yupi_nsfw_api.json) and the
# stock Turbo pipeline are both untouched.
YUPI_FAST_API_TEMPLATE = Path(__file__).resolve().parent / "yupi_fast_api.json"
YUPI_FAST_OUTPUT_PREFIX = "MiniMaxH3/YUPI_FAST"
YUPI_FAST_LORA_NAME = os.environ.get(
    "MINIMAX_YUPI_FAST_LORA",
    "fasth3_6step.safetensors",
).strip()
STATE_PATH = Path(
    os.environ.get(
        "MINIMAX_TELEGRAM_STATE",
        r"E:\MiniMax-H3-Telegram\runtime\bot\settings.json",
    )
)
IMAGE_DIR = STATE_PATH.parent / "input_images"
REFERENCE_DIR = STATE_PATH.parent / "reference_media"
MAX_SEGMENT_SECONDS = 15.0
MAX_SHOT_SECONDS = 8.0
SHOT_TRANSITION_SECONDS = 0.35
SEEDVR2_UNET_NAME = "seedvr2_3b_int8_convrot.safetensors"
SEEDVR2_VAE_NAME = "seedvr2_ema_vae_fp16.safetensors"
SEEDVR2_FHD_LONG_EDGE = 1920
SEEDVR2_2K_LONG_EDGE = 2560
SEEDVR2_SPLIT_SECONDS = 8.0
TIMELINE_TOLERANCE_SECONDS = 0.25
MIN_TOTAL_SECONDS = 2.0
MAX_TOTAL_SECONDS = 30.0 * 60.0
CONTINUATION_DIR = STATE_PATH.parent / "continuation_frames"
LONG_CHECKPOINT_DIR = STATE_PATH.parent / "long_checkpoints"
QUEUE_STATE_PATH = STATE_PATH.parent / "story_queue.json"
LONG_CHECKPOINT_VERSION = 1
QUEUE_STATE_VERSION = 1
MAX_HISTORY_ITEMS = 30
MAX_QUEUE_ITEMS = 30
BOT_LOG = STATE_PATH.parent / "bot.log"
LONG_CONTINUITY_MODE = os.environ.get(
    "MINIMAX_H3_LONG_CONTINUITY", "motion_context"
).strip().lower()
# User-facing long-video continuation modes: pin the previous shot's AV latent
# (motion_context) or fall back to a plain tail-frame handoff (tail_frame).
LONG_CONTINUITY_MODES = ("motion_context", "tail_frame")
MOTION_CONTEXT_LENGTH = 22
# The pinned *audio* window is a separate setting from the pinned *video*
# window. The H3 Motion Context README recommends exactly 24 frames: that is
# one full second of sound and lands exactly on the model's 40 Hz audio grid
# (a video frame is 5/3 of an audio step), so the window is pinned at
# precisely the requested width. Off-grid values (e.g. 22) are widened to the
# nearest whole step by the node, which is what the old code shipped by
# mistake. 48 pins the last two whole seconds.
MOTION_CONTEXT_AUDIO_LENGTH = 24
MOTION_CONTEXT_EXTRA_SECONDS = MOTION_CONTEXT_LENGTH / 24.0
MODEL_H3 = "h3"
INPUT_MODE_TEXT = "text"
INPUT_MODE_IMAGE = "image"
INPUT_MODE_FL2VA = "fl2va"
INPUT_MODE_REF2VA = "ref2va"
# YUPI is a real input mode, not an action button: selecting it only stages the
# mode (the user uploads a reference image and then presses 🚀 生成影片).
INPUT_MODE_YUPI = "yupi"
INPUT_MODES = {
    INPUT_MODE_TEXT,
    INPUT_MODE_IMAGE,
    INPUT_MODE_FL2VA,
    INPUT_MODE_REF2VA,
    INPUT_MODE_YUPI,
}
MENU_MAIN = "main"
MENU_INPUT = "input"
MENU_SETTINGS = "settings"
MENU_MODE = "mode"
MENU_DURATION = "duration"
MENU_QUALITY = "quality"
MENU_JOB = "job"
MENU_SYSTEM = "system"
MENU_HISTORY = "history"
CONTROL_PANEL_BUTTON = "🎛️ 面板"
MENU_SECTIONS = {
    MENU_MAIN,
    MENU_INPUT,
    MENU_SETTINGS,
    MENU_MODE,
    MENU_DURATION,
    MENU_QUALITY,
    MENU_JOB,
    MENU_SYSTEM,
    MENU_HISTORY,
}
# Ref2VA reference model. Default is the community hybrid
# (fl2va base + ref2va late-block adaln, smhfacct b25-49 int8): it keeps
# ref2va's reference conditioning while restoring fl2va-level visual/audio
# fidelity, which is the drop-in fix for ref2va's "greasy/blurry/inconsistent"
# output. The env var can still point back to either original checkpoint
# (pruned ref2va or the hybrid b20-49/b15-49 variants) if needed.
REF2VA_UNET_NAME = os.environ.get(
    "MINIMAX_H3_REF2VA_MODEL",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
)
# Latent upscaler weights (LBH-123-AI 3D fp16) for the two-stage
# low-res -> latent-upscale -> re-sample path; scanned from
# models/latent_upscale_models by the Comfyui_Minimax_h3_latent_Upscaler node.
LATENT_UPSCALER_MODEL = os.environ.get(
    "MINIMAX_H3_LATENT_UPSCALER",
    "minimax_h3_latent_upscaler_3d_fp16.safetensors",
)
# Two-stage latent upscaling (YZ_金鱼's trick from BV18qtp6WEdi): generate at
# half resolution, upscale the video latent in latent space (no VAE
# decode/encode, so it is fast and keeps the audio latent intact), then
# re-sample at full resolution with a partial denoise. This removes the
# "greasy/blurry" look and adds detail. Motion-context long videos carry a
# context AV latent that must stay 1:1 with the context, so they skip this
# path automatically. Set the env var to 0/false to disable.
LATENT_UPSCALE_ENABLED = os.environ.get(
    "MINIMAX_H3_LATENT_UPSCALE", "1"
) not in ("0", "false", "False", "off", "no")
MAX_REF2VA_IMAGES = 9
MAX_REF2VA_VIDEOS = 3
MAX_REF2VA_AUDIOS = 3


def normalize_model_mode(value: Any) -> str:
    # The LTX 2.3 branch was removed; only the MiniMax H3 Turbo workflow remains.
    return MODEL_H3


def normalize_task_type(value: Any) -> str:
    """Keep the isolated YUPI path distinguishable from the stock H3 path."""
    return (
        YUPI_TASK_TYPE
        if str(value or "").strip().lower() == YUPI_TASK_TYPE
        else normalize_model_mode(value)
    )


def normalize_input_mode(value: Any) -> str:
    mode = str(value or INPUT_MODE_TEXT).strip().lower()
    return mode if mode in INPUT_MODES else INPUT_MODE_TEXT


def is_ref2va_like(value: Any) -> bool:
    """True for modes that stage Ref2VA-style reference media (Ref2VA, YUPI)."""
    return normalize_input_mode(value) in {INPUT_MODE_REF2VA, INPUT_MODE_YUPI}


def normalize_menu_section(value: Any) -> str:
    section = str(value or MENU_MAIN).strip().lower()
    return section if section in MENU_SECTIONS else MENU_MAIN


def control_panel_reply_markup() -> dict[str, Any]:
    """Keep a one-tap control-panel shortcut beside Telegram's input box."""
    return {
        "keyboard": [[{"text": CONTROL_PANEL_BUTTON}]],
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
        "input_field_placeholder": "輸入提示詞，或按 🎛️ 面板",
    }


class BotError(RuntimeError):
    pass


_log_lock = threading.Lock()


def bot_log(message: str) -> None:
    """Append a timestamped line to the bot's persistent log file."""
    try:
        with _log_lock:
            with BOT_LOG.open("a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


def http_error_detail(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    return f"HTTP Error {exc.code}: {body[:1200]}" if body else str(exc)


def decode_prompt_text(data: bytes) -> str:
    """Decode a Telegram text file without losing multiline prompt structure."""
    if not data:
        raise BotError("TXT 檔案是空白的，請先加入提示詞內容。")

    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030", "big5")
    for encoding in encodings:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" in text:
            continue
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if normalized:
            return normalized
    raise BotError("無法讀取 TXT 編碼，請另存為 UTF-8 後再上傳。")


def run_hidden_command(
    command: list[str], timeout: float = 10
) -> subprocess.CompletedProcess[str]:
    """Run a local diagnostic/control command without opening a console window."""
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "check": False,
    }
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(command, **kwargs)


def temperature_report() -> str:
    """Read available Windows/NVIDIA temperature and VRAM information."""
    lines = ["🌡 電腦溫度／顯卡狀態"]
    smi_path = Path(NVIDIA_SMI_PATH)
    if smi_path.is_file() or shutil.which(NVIDIA_SMI_PATH):
        try:
            result = run_hidden_command(
                [
                    NVIDIA_SMI_PATH,
                    "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                timeout=8,
            )
            if result.returncode == 0:
                rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                for index, row in enumerate(rows, start=1):
                    fields = [field.strip() for field in row.split(",")]
                    if len(fields) >= 5:
                        name, temperature, utilization, memory_used, memory_total = fields[:5]
                        lines.append(
                            f"GPU {index}：{name}｜{temperature}°C｜"
                            f"GPU {utilization}%｜VRAM {memory_used}/{memory_total} MiB"
                        )
            else:
                detail = (result.stderr or result.stdout).strip().splitlines()
                lines.append(f"GPU：讀取失敗（{detail[-1][:180] if detail else 'nvidia-smi error'}）")
        except (OSError, subprocess.TimeoutExpired) as exc:
            lines.append(f"GPU：讀取失敗（{exc}）")
    else:
        lines.append("GPU：找不到 nvidia-smi")

    cpu_query = (
        "try { $values = @(Get-CimInstance -Namespace root/wmi "
        "-ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop | "
        "ForEach-Object { [math]::Round(($_.CurrentTemperature / 10) - 273.15, 1) }); "
        "if ($values.Count -gt 0) { $values -join ',' } else { exit 1 } "
        "} catch { exit 1 }"
    )
    try:
        cpu_result = run_hidden_command(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                cpu_query,
            ],
            timeout=8,
        )
        cpu_values = [value.strip() for value in cpu_result.stdout.split(",") if value.strip()]
        if cpu_result.returncode == 0 and cpu_values:
            lines.append(f"CPU：{'、'.join(value + '°C' for value in cpu_values[:4])}")
        else:
            lines.append("CPU：Windows 未提供可讀取的溫度感測器")
    except (OSError, subprocess.TimeoutExpired):
        lines.append("CPU：無法讀取溫度感測器")

    lines.append(f"讀取時間：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def schedule_windows_shutdown() -> None:
    if os.name != "nt":
        raise BotError("自動關機只支援 Windows。")
    result = run_hidden_command(
        ["shutdown.exe", "/s", "/t", str(SHUTDOWN_DELAY_SECONDS)],
        timeout=10,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise BotError(f"排程關機失敗：{detail[-500:] or 'shutdown.exe error'}")


def cancel_windows_shutdown() -> None:
    if os.name != "nt":
        raise BotError("取消自動關機只支援 Windows。")
    result = run_hidden_command(["shutdown.exe", "/a"], timeout=10)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise BotError(f"取消關機失敗：{detail[-500:] or 'shutdown.exe error'}")


def validate_total_seconds(seconds: float) -> float:
    """Validate a Telegram total duration and keep a stable saved value."""
    if not math.isfinite(seconds):
        raise BotError("總片長必須是有效數字，例如 37 或 600。")
    if seconds < MIN_TOTAL_SECONDS or seconds > MAX_TOTAL_SECONDS:
        raise BotError("總片長必須介乎 2 至 1800 秒（30 分鐘）。")
    return round(float(seconds), 3)


@dataclass(frozen=True)
class GenerationConfig:
    width: int
    height: int
    steps: int
    requested_seconds: float
    length: int

    @property
    def actual_seconds(self) -> float:
        return self.length / 24.0


@dataclass
class JobState:
    chat_id: str
    config: GenerationConfig
    prompt: str
    started_at: float
    prompt_id: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    pause_requested: threading.Event = field(default_factory=threading.Event)
    resume_event: threading.Event = field(default_factory=threading.Event)
    preview_in_progress: threading.Event = field(default_factory=threading.Event)
    output_prefix: str = OUTPUT_PREFIX
    segment_index: int = 1
    segment_total: int = 1
    total_seconds: float = 0.0
    shot_plan: tuple[ShotSpec, ...] = field(default_factory=tuple)
    story_global_text: str = ""
    segment_start_seconds: float = 0.0
    segment_end_seconds: float = 0.0
    input_image_path: Optional[Path] = None
    last_image_path: Optional[Path] = None
    reference_image_paths: list[Path] = field(default_factory=list)
    reference_video_paths: list[Path] = field(default_factory=list)
    reference_audio_paths: list[Path] = field(default_factory=list)
    continuation_video_path: Optional[Path] = None
    comfy_image_name: Optional[str] = None
    comfy_last_image_name: Optional[str] = None
    comfy_reference_image_names: list[str] = field(default_factory=list)
    comfy_reference_video_names: list[str] = field(default_factory=list)
    comfy_reference_audio_names: list[str] = field(default_factory=list)
    continuation_image_path: Optional[Path] = None
    audio_reference_name: Optional[str] = None
    workflow_reports: list[str] = field(default_factory=list)
    progress_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    progress_percent: float = 0.0
    progress_node_id: Optional[str] = None
    progress_node_state: str = "queued"
    progress_node_value: float = 0.0
    progress_node_max: float = 1.0
    progress_node_index: int = 0
    progress_node_total: int = 0
    progress_queue_remaining: Optional[int] = None
    progress_phase: str = "queued"
    progress_tracker: Any = field(default=None, repr=False, compare=False)
    task_type: str = "h3"
    generation_mode: str = INPUT_MODE_TEXT
    # True when this job is the YUPI_FAST variant (FastH3 6-step distill chain).
    # Kept separate from task_type so every existing YUPI task_type check still
    # applies unchanged; only the template/prefix/name-hint differ.
    yupi_fast: bool = False
    upscale_source_path: Optional[Path] = None
    upscale_target_width: int = 0
    upscale_target_height: int = 0
    # Long-video fields are persisted in LONG_CHECKPOINT_DIR after every shot.
    # They let a new Bot process continue at the first unfinished shot.
    base_config: Optional[GenerationConfig] = None
    long_base_prefix: Optional[str] = None
    checkpoint_path: Optional[Path] = None
    resume_from_segment: int = 1
    completed_video_paths: list[Path] = field(default_factory=list)
    initial_context_video_path: Optional[Path] = None
    initial_context_latent_path: Optional[str] = None
    resume_motion_context: Optional[bool] = None
    long_resolution: Optional[tuple[int, int]] = None
    resolution_fallbacks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PendingUpscale:
    token: str
    chat_id: str
    source_path: Path
    source_width: int
    source_height: int
    duration_seconds: float
    shutdown_after_choice: bool = False


@dataclass(frozen=True)
class QueuedStory:
    """A snapshot of one story waiting for sequential generation."""

    item_id: str
    prompt: str
    config: GenerationConfig
    total_seconds: float
    input_image_path: Optional[Path] = None
    last_image_path: Optional[Path] = None
    reference_image_paths: tuple[Path, ...] = field(default_factory=tuple)
    reference_video_paths: tuple[Path, ...] = field(default_factory=tuple)
    reference_audio_paths: tuple[Path, ...] = field(default_factory=tuple)
    generation_mode: str = INPUT_MODE_TEXT
    model_mode: str = MODEL_H3
    created_at: float = field(default_factory=time.time)


class ComfyProgressTracker:
    """Listen to ComfyUI's WebSocket progress events for one prompt."""

    def __init__(self, job: JobState, client_id: str = "telegram-turbo-bot"):
        self.job = job
        self.client_id = str(client_id or "telegram-turbo-bot")
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run,
            name="minimax-comfy-progress",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)

    def _run(self) -> None:
        try:
            asyncio.run(self._listen())
        except Exception as exc:
            # Progress is best-effort; the normal history poll remains authoritative.
            print(f"Comfy progress tracker error: {exc}", flush=True)

    async def _listen(self) -> None:
        try:
            import aiohttp
        except ImportError as exc:
            print(f"Comfy progress tracker unavailable: {exc}", flush=True)
            return

        ws_base = COMFY_URL.replace("https://", "wss://", 1).replace(
            "http://", "ws://", 1
        )
        ws_url = f"{ws_base}/ws?clientId={quote(self.client_id, safe='')}"
        timeout = aiohttp.ClientTimeout(total=None, connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.ws_connect(
                    ws_url,
                    heartbeat=30,
                    autoping=True,
                ) as websocket:
                    while not self.stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(
                                websocket.receive(), timeout=1.0
                            )
                        except asyncio.TimeoutError:
                            continue
                        if message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                payload = json.loads(message.data)
                            except (TypeError, json.JSONDecodeError):
                                continue
                            if isinstance(payload, dict):
                                self._handle_message(payload)
                        elif message.type in {
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.ERROR,
                        }:
                            break
            except Exception as exc:
                print(f"Comfy progress WebSocket unavailable: {exc}", flush=True)

    def _is_current_prompt(self, data: dict[str, Any]) -> bool:
        prompt_id = data.get("prompt_id")
        return prompt_id is None or str(prompt_id) == str(self.job.prompt_id)

    def _handle_message(self, payload: dict[str, Any]) -> None:
        message_type = str(payload.get("type", ""))
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            return

        if message_type == "status":
            exec_info = (data.get("status") or {}).get("exec_info") or {}
            queue_remaining = exec_info.get("queue_remaining")
            if queue_remaining is not None:
                with self.job.progress_lock:
                    self.job.progress_queue_remaining = int(queue_remaining)
            return

        if not self._is_current_prompt(data):
            return

        if message_type == "executing":
            node_id = data.get("node")
            with self.job.progress_lock:
                if node_id is None:
                    self.job.progress_phase = "finishing"
                    self.job.progress_node_state = "finishing"
                else:
                    self.job.progress_phase = "running"
                    self.job.progress_node_id = str(node_id)
                    self.job.progress_node_state = "running"
            return

        if message_type == "progress":
            self._update_step_progress(data)
            return

        if message_type == "progress_state":
            self._update_node_progress(data.get("nodes") or {})
            return

        if message_type == "execution_success":
            with self.job.progress_lock:
                self.job.progress_percent = 100.0
                self.job.progress_phase = "completed"
                self.job.progress_node_state = "finished"
            return

        if message_type == "execution_error":
            with self.job.progress_lock:
                self.job.progress_phase = "error"
                self.job.progress_node_state = "error"

    def _update_step_progress(self, data: dict[str, Any]) -> None:
        try:
            value = float(data.get("value", 0))
            maximum = max(float(data.get("max", 1)), 1.0)
        except (TypeError, ValueError):
            return
        percent = max(0.0, min(100.0, value / maximum * 100.0))
        with self.job.progress_lock:
            self.job.progress_percent = percent
            self.job.progress_phase = "sampling"
            self.job.progress_node_state = "running"
            self.job.progress_node_value = value
            self.job.progress_node_max = maximum
            if data.get("node") is not None:
                self.job.progress_node_id = str(data["node"])

    def _update_node_progress(self, nodes: Any) -> None:
        if not isinstance(nodes, dict):
            return
        valid_nodes = [node for node in nodes.values() if isinstance(node, dict)]
        if not valid_nodes:
            return

        finished = 0
        running_node: Optional[tuple[str, dict[str, Any]]] = None
        running_fraction = 0.0
        for key, node in nodes.items():
            if not isinstance(node, dict):
                continue
            state = str(node.get("state", "pending"))
            if state == "finished":
                finished += 1
            elif state in {"running", "executing"} and running_node is None:
                running_node = (str(key), node)

        if running_node is not None:
            key, node = running_node
            try:
                value = float(node.get("value", 0))
                maximum = max(float(node.get("max", 1)), 1.0)
                running_fraction = max(0.0, min(1.0, value / maximum))
            except (TypeError, ValueError):
                value, maximum = 0.0, 1.0
            node_id = node.get("display_node_id") or node.get("node_id") or key
            node_state = "running"
        else:
            value, maximum = 0.0, 1.0
            node_id = None
            node_state = "finished" if finished == len(valid_nodes) else "pending"

        total = len(valid_nodes)
        percent = ((finished + running_fraction) / total) * 100.0
        with self.job.progress_lock:
            self.job.progress_percent = max(0.0, min(100.0, percent))
            self.job.progress_phase = "running"
            self.job.progress_node_id = str(node_id) if node_id is not None else None
            self.job.progress_node_state = node_state
            self.job.progress_node_value = value
            self.job.progress_node_max = maximum
            self.job.progress_node_index = min(total, finished + (1 if running_node else 0))
            self.job.progress_node_total = total


def valid_length(seconds: float) -> int:
    """Return the next valid H3 frame count on the 17n+5 grid at 24fps."""
    if seconds < 2 or seconds > 15:
        raise BotError("秒數目前只允許 2 到 15 秒。")
    target_frames = max(5, math.ceil(seconds * 24.0))
    if target_frames <= 5:
        return 5
    n = math.ceil((target_frames - 5) / 17)
    return 17 * n + 5


def split_story_queue_prompts(text: str) -> list[str]:
    """Split a queue submission without breaking timeline text."""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    separator = re.compile(
        r"(?im)^\s*(?:-{3,}|={3,}|(?:story|prompt|故事)\s*\d+\s*[:：]?)\s*$"
    )
    prompts = [part.strip() for part in separator.split(normalized)]
    return [prompt for prompt in prompts if prompt]


def parse_config(parts: list[str]) -> GenerationConfig:
    if len(parts) != 4:
        raise BotError("格式：/gen 寬度 高度 steps 秒數\n例如：/gen 1344 768 4 5")
    try:
        width, height, steps = (int(parts[0]), int(parts[1]), int(parts[2]))
        seconds = float(parts[3])
    except ValueError as exc:
        raise BotError("寬度、高度、steps 和秒數都要是數字。") from exc

    if width < 32 or height < 32 or width > 1344 or height > 768:
        raise BotError("解析度範圍是 32 至 1344×768。")
    if width % 32 or height % 32:
        raise BotError("寬度和高度必須是 32 的倍數，例如 1344×768、768×768。")
    if width * height > 1344 * 768:
        raise BotError("解析度太高，先不要超過 1344×768。")
    if steps < 4 or steps > 20:
        raise BotError("steps 目前允許 4 至 20，8-step Turbo LoRA 建議 8，Full 建議 20。")

    length = valid_length(seconds)
    return GenerationConfig(width, height, steps, seconds, length)


def megapixel_label(width: int, height: int) -> str:
    """Return the one-decimal megapixel label used by H3 size references."""
    return f"{width * height / 1_000_000:.1f} MP"


def resolution_label(width: int, height: int) -> str:
    return f"{megapixel_label(width, height)} · {width}×{height}"


RESOLUTION_LADDER = (
    (448, 256),
    (512, 288),
    (608, 352),
    (736, 416),
    (864, 480),
    (960, 544),
    (1152, 640),
    (1280, 736),
    (1344, 768),
)


def next_lower_resolution(width: int, height: int) -> Optional[tuple[int, int]]:
    """Return the next lower configured resolution by pixel area."""
    current_area = int(width) * int(height)
    candidates = [
        resolution
        for resolution in RESOLUTION_LADDER
        if resolution[0] * resolution[1] < current_area
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda resolution: resolution[0] * resolution[1])


def is_cuda_oom_error(error: BaseException) -> bool:
    """Recognize ComfyUI/PyTorch OOM messages without hiding other failures."""
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "out of memory",
            "cuda out of memory",
            "allocation on device",
            "cublas_status_alloc_failed",
            "not enough memory",
        )
    )


def probe_video_info(video_path: Path) -> tuple[float, int, int]:
    """Read duration and dimensions for importing older generated videos."""
    result = run_hidden_command(
        [
            FFPROBE_PATH,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=width,height",
            "-select_streams",
            "v:0",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ],
        timeout=30,
    )
    if result.returncode != 0:
        raise BotError(f"無法讀取影片資訊：{video_path}")
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    duration = 0.0
    width = 0
    height = 0
    for value in values:
        if "x" in value:
            dimensions = value.split("x", 1)
            try:
                width = int(dimensions[0])
                height = int(dimensions[1])
            except (TypeError, ValueError):
                pass
        else:
            try:
                duration = float(value)
            except (TypeError, ValueError):
                pass
    if duration <= 0 or width <= 0 or height <= 0:
        raise BotError(f"影片資訊不完整：{video_path}")
    return duration, width, height


def extract_last_frame(video_path: Path, output_path: Path) -> Path:
    """Extract a single PNG used to anchor the next long-video segment."""
    if not video_path.is_file():
        raise BotError(f"找不到上一段影片：{video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-sseof",
        "-0.05",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-y",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BotError(f"抽取上一段最後畫面失敗：{exc}") from exc
    if result.returncode != 0 or not output_path.is_file():
        details = (result.stderr or "").strip()
        raise BotError(f"抽取上一段最後畫面失敗：{details[-800:]}")
    return output_path


@dataclass(frozen=True)
class SegmentedPrompt:
    global_text: str
    segments: dict[int, str]


@dataclass(frozen=True)
class TimelineScene:
    start_seconds: float
    end_seconds: float
    label: str
    action: str

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class TimelinePrompt:
    global_text: str
    scenes: tuple[TimelineScene, ...]


@dataclass(frozen=True)
class ShotSpec:
    start_seconds: float
    end_seconds: float
    label: str
    action: str

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class LongVideoPlan:
    global_text: str
    shots: tuple[ShotSpec, ...]
    source_format: str


SEGMENT_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(GLOBAL|SEGMENT[ \t]+([1-9][0-9]*))[ \t]*[:：][ \t]*(.*)$"
)
TIMELINE_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(?P<label>[^\n:：()（）]{0,40}?)[ \t]*"
    r"[（(][ \t]*(?P<start>[0-9]+(?:\.[0-9]+)?)[ \t]*"
    # LLMs often emit typographic/non-breaking hyphens (for example `0‑15`)
    # when copying a timeline from formatted text. Treat all common dash
    # variants as the same time-range separator.
    r"(?:-|‐|‑|‒|–|—|−|~|～|至|到)[ \t]*(?P<end>[0-9]+(?:\.[0-9]+)?)[ \t]*"
    r"(?:秒|s|sec|seconds?)?[ \t]*[）)][ \t]*[:：]?[ \t]*(?P<inline>.*)$"
)
SHARED_TAIL_SEPARATOR_RE = re.compile(
    r"(?m)^[ \t]*(?:-{3,}|─{3,}|={3,})[ \t]*$"
)
# A heading may carry its time range WITHOUT parentheses, e.g.
# `Segment 3 5-8 秒：`, `Act 2 20-40 s:` or `第一幕 0-20 秒：`. Both header
# regexes above require the parenthesised form (or `SEGMENT n:`), so this third
# spelling needs its own pattern. It is deliberately anchored on a heading word
# plus a time unit, so ordinary prose ("走了 2-3 步") is never taken for a scene.
# The English spelling carries its own index (`Segment 3`) while the CJK one
# already contains it (`第三幕`), and the filler between heading and range may
# not contain digits — otherwise `第二幕 20-40 秒` would eat the leading `2`.
_HEADING_WORD = (
    r"(?:(?:SEGMENT|SCENE|SHOT|ACT|PART)[ \t]*[0-9]*"
    r"|第[ \t]*[0-9０-９一二三四五六七八九十百千]+[ \t]*(?:幕|段|場景|场景|部分))"
)
_TIME_PAIR_CORE = (
    r"[0-9]+(?:\.[0-9]+)?[ \t]*(?:-|‐|‑|‒|–|—|−|~|～|至|到)[ \t]*"
    r"[0-9]+(?:\.[0-9]+)?[ \t]*(?:秒|s|sec|seconds?)"
)
_BARE_HEADER_CORE = (
    rf"{_HEADING_WORD}[ \t]*[^\n:：()（）0-9]{{0,24}}?[ \t]*{_TIME_PAIR_CORE}"
)
BARE_TIMELINE_HEADER_RE = re.compile(
    rf"(?im)^[ \t]*(?P<label>{_HEADING_WORD}[ \t]*[^\n:：()（）0-9]{{0,24}}?)[ \t]*"
    rf"(?P<start>[0-9]+(?:\.[0-9]+)?)[ \t]*"
    rf"(?:-|‐|‑|‒|–|—|−|~|～|至|到)[ \t]*(?P<end>[0-9]+(?:\.[0-9]+)?)[ \t]*"
    rf"(?:秒|s|sec|seconds?)[ \t]*[:：]?[ \t]*(?P<inline>.*)$"
)
# The simplest spelling of all: the line IS just a time range, no heading word.
#   `0-2 秒：…`   `2-5s: …`   `5-8秒 …`   `10-16：…`
# Accepts 秒 / s / sec / seconds, with or without a colon after a unit. A bare
# `2-3 个人` (no unit, no colon) is still prose and never matches.
_RANGE_START_CORE = (
    r"[0-9]+(?:\.[0-9]+)?[ \t]*(?:-|‐|‑|‒|–|—|−|~|～|至|到)[ \t]*"
    r"[0-9]+(?:\.[0-9]+)?[ \t]*"
)
NAKED_RANGE_HEADER_RE = re.compile(
    rf"(?im)^[ \t]*(?P<start>[0-9]+(?:\.[0-9]+)?)[ \t]*"
    rf"(?:-|‐|‑|‒|–|—|−|~|～|至|到)[ \t]*(?P<end>[0-9]+(?:\.[0-9]+)?)[ \t]*"
    rf"(?:(?:秒|s|sec|seconds?)[ \t]*[:：]?|[:：])[ \t]*(?P<inline>.*)$"
)
# Telegram users routinely paste a whole script as ONE paragraph, which the
# anchored header regexes above cannot see (`^` never matches mid-line). This
# finds headings that are not already at the start of a line so they can be
# broken onto their own line before parsing. Every branch sits INSIDE the
# lookahead so sub() only ever replaces the leading whitespace, never the
# heading text itself.
INLINE_HEADER_BREAK_RE = re.compile(
    r"(?im)(?<!\n)(?<![0-9])[ \t]*(?="
    r"(?:GLOBAL[ \t]*[:：])"
    r"|(?:SEGMENT[ \t]*[1-9][0-9]*[ \t]*(?:[（(][^（()）]{0,24}[）)]|[：:]))"
    r"|(?:第[ \t]*[0-9０-９一二三四五六七八九十百千]+[ \t]*"
    r"(?:幕|段|場景|场景|部分)[ \t]*[（(])"
    rf"|(?:{_BARE_HEADER_CORE})"
    rf"|(?:{_RANGE_START_CORE}(?:秒|s|sec|seconds?)[ \t]*[:：])"
    r")"
)


def _parenthesise_bare_range(match: "re.Match[str]") -> str:
    """Rewrite `Segment 3 5-8 秒：…` as `Segment 3（5-8秒）：…`."""
    label = match.group("label").strip()
    inline = match.group("inline").strip()
    head = f"{label}（{match.group('start')}-{match.group('end')}秒）："
    return f"{head}{inline}" if inline else head


def _parenthesise_naked_range(match: "re.Match[str]") -> str:
    """Rewrite a bare `5-8 秒：…` line as `（5-8秒）：…`.

    The timeline parser names an unlabelled scene automatically, so no heading
    word is needed at all.
    """
    inline = match.group("inline").strip()
    head = f"（{match.group('start')}-{match.group('end')}秒）："
    return f"{head}{inline}" if inline else head


def normalize_inline_headers(prompt: str) -> str:
    """Put GLOBAL / SEGMENT / act headings on their own line before parsing.

    Without this, a one-paragraph prompt never matches the anchored header
    regexes and long videos fail with "必須提供時間軸" even though the script
    is correctly structured. Existing line breaks are left untouched. Headings
    whose range has no parentheses are rewritten into the parenthesised form so
    every downstream parser sees one shape.
    """
    if not prompt:
        return prompt
    prompt = INLINE_HEADER_BREAK_RE.sub("\n", prompt)
    prompt = BARE_TIMELINE_HEADER_RE.sub(_parenthesise_bare_range, prompt)
    return NAKED_RANGE_HEADER_RE.sub(_parenthesise_naked_range, prompt)

# A prompt can state its total duration in GLOBAL text when it uses the
# SEGMENT format instead of explicit time ranges.  Keep these expressions
# deliberately narrow so ages, years, and ordinary numbers are not mistaken
# for the requested video length.
PROMPT_DURATION_HINT_RES = (
    re.compile(
        r"(?i)\b(?:total\s+)?(?:duration|length|runtime|video\s+length)"
        r"[ \t]*[:：=]?[ \t]*(?P<seconds>[0-9]+(?:\.[0-9]+)?)"
        r"[ \t]*(?:seconds?|secs?|sec|s)\b"
    ),
    re.compile(
        r"(?i)(?:總片長|總時長|總長度|影片長度|視頻長度|片長|時長|全片長度)"
        r"[ \t]*[:：=]?[ \t]*(?P<seconds>[0-9]+(?:\.[0-9]+)?)"
        r"[ \t]*(?:秒|秒鐘|s)"
    ),
    re.compile(
        r"(?i)(?P<seconds>[0-9]+(?:\.[0-9]+)?)[ \t]*"
        r"(?:秒|秒鐘|seconds?|secs?|sec|s)[ \t]*"
        r"(?:長片|長視頻|影片|視頻|video)"
    ),
)


def detect_prompt_total_seconds(prompt: str) -> Optional[float]:
    """Return the script's declared total duration, if it has one.

    Explicit timeline headings are authoritative: their largest end time is
    the script duration.  SEGMENT prompts may not carry ranges, so fall back
    to a clearly labelled GLOBAL duration such as ``Total duration 60
    seconds`` or ``總片長：60秒``.
    """
    normalized = str(prompt or "").replace("\u00a0", " ").strip()
    if not normalized:
        return None

    try:
        timeline = parse_timeline_prompt(normalized)
    except BotError:
        # Leave malformed prompts for build_long_video_plan() to report when
        # the user presses Generate; detection should not hide that error.
        timeline = None
    if timeline is not None and timeline.scenes:
        candidate = max(scene.end_seconds for scene in timeline.scenes)
        if MIN_TOTAL_SECONDS <= candidate <= MAX_TOTAL_SECONDS:
            return round(candidate, 3)

    for pattern in PROMPT_DURATION_HINT_RES:
        match = pattern.search(normalized)
        if not match:
            continue
        try:
            candidate = float(match.group("seconds"))
        except (TypeError, ValueError):
            continue
        if MIN_TOTAL_SECONDS <= candidate <= MAX_TOTAL_SECONDS:
            return round(candidate, 3)
    return None


def parse_segmented_prompt(prompt: str) -> Optional[SegmentedPrompt]:
    """Parse GLOBAL/SEGMENT headings while preserving multiline prompt text."""
    prompt = normalize_inline_headers(prompt)
    matches = list(SEGMENT_HEADER_RE.finditer(prompt))
    if not any(match.group(2) for match in matches):
        return None

    global_parts: list[str] = []
    segments: dict[int, str] = {}
    preamble = prompt[: matches[0].start()].strip()
    if preamble:
        global_parts.append(preamble)

    for position, match in enumerate(matches):
        body_end = matches[position + 1].start() if position + 1 < len(matches) else len(prompt)
        inline_text = match.group(3).strip()
        body_text = prompt[match.end() : body_end].strip()
        text = "\n".join(part for part in (inline_text, body_text) if part).strip()

        segment_number = match.group(2)
        if segment_number is None:
            if text:
                global_parts.append(text)
            continue

        number = int(segment_number)
        if number in segments:
            raise BotError(f"分段提示詞重複了 SEGMENT {number}。")
        if not text:
            raise BotError(f"SEGMENT {number} 沒有任何提示詞內容。")
        segments[number] = text

    # A shared style block is often placed after the final SEGMENT behind a
    # separator. Treat it as GLOBAL text instead of assigning it only to the
    # final segment.
    if global_parts and segments:
        final_number = max(segments)
        tail_parts = SHARED_TAIL_SEPARATOR_RE.split(segments[final_number], maxsplit=1)
        if len(tail_parts) == 2 and tail_parts[1].strip():
            segments[final_number] = tail_parts[0].strip()
            global_parts.append(tail_parts[1].strip())
            if not segments[final_number]:
                raise BotError(f"SEGMENT {final_number} 沒有任何提示詞內容。")

    return SegmentedPrompt(
        global_text="\n\n".join(global_parts).strip(),
        segments=segments,
    )


def parse_timeline_prompt(prompt: str) -> Optional[TimelinePrompt]:
    """Parse headings such as `第一幕（5-15秒）：` into ordered scenes."""
    prompt = normalize_inline_headers(prompt)
    matches = list(TIMELINE_HEADER_RE.finditer(prompt))
    if not matches:
        return None

    preamble = prompt[: matches[0].start()].strip()
    scenes: list[TimelineScene] = []
    for position, match in enumerate(matches):
        body_end = matches[position + 1].start() if position + 1 < len(matches) else len(prompt)
        inline_text = match.group("inline").strip()
        body_text = prompt[match.end() : body_end].strip()
        action = "\n".join(part for part in (inline_text, body_text) if part).strip()
        label = match.group("label").strip(" \t【】[]") or f"場景 {position + 1}"
        start_seconds = float(match.group("start"))
        end_seconds = float(match.group("end"))
        if end_seconds <= start_seconds:
            raise BotError(
                f"時間軸「{label}」的結束時間必須大於開始時間："
                f"{start_seconds:g}-{end_seconds:g} 秒。"
            )
        if not action:
            raise BotError(f"時間軸「{label}」沒有任何畫面或動作內容。")
        scenes.append(
            TimelineScene(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                label=label,
                action=action,
            )
        )
    return TimelinePrompt(global_text=preamble, scenes=tuple(scenes))


def split_action_units(text: str) -> list[str]:
    """Split a scene into ordered visual beats without reordering its text."""
    units: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pieces = re.findall(r".+?(?:[。！？!?；;.]+|$)", line)
        units.extend(piece.strip() for piece in pieces if piece.strip())
    return units or [text.strip()]


def split_scene_into_shots(scene: TimelineScene) -> list[ShotSpec]:
    """Split one timeline scene into 5–8 second generation shots."""
    part_count = max(1, math.ceil(scene.duration / MAX_SHOT_SECONDS))
    part_duration = scene.duration / part_count
    if part_duration < MIN_TOTAL_SECONDS:
        raise BotError(
            f"時間軸「{scene.label}」切分後每個鏡頭少於 2 秒；"
            "請合併過短場景或延長時間。"
        )

    units = split_action_units(scene.action)
    shots: list[ShotSpec] = []
    for index in range(part_count):
        start_seconds = scene.start_seconds + part_duration * index
        end_seconds = (
            scene.end_seconds
            if index == part_count - 1
            else scene.start_seconds + part_duration * (index + 1)
        )
        if len(units) >= part_count:
            unit_start = math.floor(index * len(units) / part_count)
            unit_end = math.floor((index + 1) * len(units) / part_count)
            unit_end = max(unit_start + 1, unit_end)
            action = "\n".join(units[unit_start:unit_end]).strip()
        else:
            action = units[min(index, len(units) - 1)]
        shots.append(
            ShotSpec(
                start_seconds=round(start_seconds, 3),
                end_seconds=round(end_seconds, 3),
                label=scene.label,
                action=action,
            )
        )
    return shots


def validate_timeline_coverage(
    scenes: tuple[TimelineScene, ...], total_seconds: float
) -> None:
    cursor = 0.0
    for scene in scenes:
        if scene.start_seconds > cursor + TIMELINE_TOLERANCE_SECONDS:
            raise BotError(
                f"時間軸在 {cursor:g}-{scene.start_seconds:g} 秒沒有內容。"
            )
        if scene.start_seconds < cursor - TIMELINE_TOLERANCE_SECONDS:
            raise BotError(
                f"時間軸「{scene.label}」與上一幕重疊："
                f"{scene.start_seconds:g} 秒早於 {cursor:g} 秒。"
            )
        cursor = scene.end_seconds
    if cursor < total_seconds - TIMELINE_TOLERANCE_SECONDS:
        raise BotError(
            f"時間軸只寫到 {cursor:g} 秒，但目前總片長是 {total_seconds:g} 秒；"
            f"請補上 {cursor:g}-{total_seconds:g} 秒的結尾。"
        )
    if cursor > total_seconds + TIMELINE_TOLERANCE_SECONDS:
        raise BotError(
            f"時間軸寫到 {cursor:g} 秒，超過目前設定的 {total_seconds:g} 秒。"
        )


def build_long_video_plan(prompt: str, total_seconds: float) -> LongVideoPlan:
    """Build an explicit short-shot plan; never replay one long prompt blindly."""
    timeline = parse_timeline_prompt(prompt)
    if timeline is not None:
        validate_timeline_coverage(timeline.scenes, total_seconds)
        shots = tuple(
            shot
            for scene in timeline.scenes
            for shot in split_scene_into_shots(scene)
        )
        return LongVideoPlan(timeline.global_text, shots, "timeline")

    segmented = parse_segmented_prompt(prompt)
    if segmented is not None:
        segment_numbers = sorted(segmented.segments)
        if not segment_numbers or segment_numbers[0] != 1:
            raise BotError("SEGMENT numbering must start at SEGMENT 1.")
        for expected_number, actual_number in enumerate(segment_numbers, start=1):
            if actual_number != expected_number:
                raise BotError(
                    "SEGMENT numbering must be consecutive; "
                    f"expected SEGMENT {expected_number}, got SEGMENT {actual_number}."
                )

        # The number of SEGMENT blocks controls the story beats. Do not derive
        # it from total_seconds: a 120-second video may intentionally have 8,
        # 12, or more story beats. Each beat is then split into <= 8-second
        # generation shots by split_scene_into_shots().
        segment_total = len(segment_numbers)
        segment_duration = total_seconds / segment_total
        if segment_duration < MIN_TOTAL_SECONDS:
            raise BotError(
                f"{segment_total} SEGMENT blocks are too many for "
                f"{total_seconds:g} seconds; each SEGMENT must be at least "
                f"{MIN_TOTAL_SECONDS:g} seconds."
            )
        # A SEGMENT is a story beat, not necessarily one model clip. Longer
        # beats are split into <=8-second shots below, just like explicit
        # timeline scenes. This lets a natural three-beat 60-second script
        # (three 20-second SEGMENT blocks) work without forcing users to add
        # artificial SEGMENT headings.

        shots: list[ShotSpec] = []
        for position, number in enumerate(segment_numbers):
            start_seconds = round(total_seconds * position / segment_total, 3)
            end_seconds = (
                round(total_seconds * (position + 1) / segment_total, 3)
                if position < segment_total - 1
                else round(total_seconds, 3)
            )
            scene = TimelineScene(
                start_seconds,
                end_seconds,
                f"SEGMENT {number}",
                segmented.segments[number],
            )
            shots.extend(split_scene_into_shots(scene))
        return LongVideoPlan(segmented.global_text, tuple(shots), "segments")

    raise BotError(
        "超過 15 秒的長片必須提供時間軸，例如「第一幕（0-8秒）：……」；"
        "也可以使用 GLOBAL／SEGMENT 1／SEGMENT 2 格式。"
    )


def ref2va_reference_rule(job: JobState) -> str:
    """Force the model to treat the reference media as an *identity anchor only*.

    Ref2VA has no continuation frame and no other structural prompt: the user's
    text is the sole instruction. Without an explicit appearance-only rule the
    model reads the reference image as "reproduce this whole picture" and copies
    the reference's scene, framing, background and props into the output, which
    produces a hard cut the moment the reference's environment stops matching the
    prompt's. This clause scopes the reference to character identity and costume
    only, and tells the model to take the setting purely from the prompt.
    """
    if job.generation_mode != INPUT_MODE_REF2VA:
        return ""
    # Reference images are only attached to the workflow on the first segment;
    # later Ref2VA segments switch to a tail-frame handoff (no reference media in
    # the graph), so the anchor wording would be wrong there.
    if job.segment_index != 1:
        return ""
    return (
        "REFERENCE MEDIA = IDENTITY ANCHOR ONLY. Take the reference image(s) as a "
        "strict character-appearance anchor: use them to lock face, identity, "
        "hairstyle, skin, build, costume and accessories — and ONLY those. Do NOT "
        "copy, reuse or blend the reference's scene, location, background, set "
        "dressing, props, framing, camera angle, lighting or time of day into the "
        "video. The setting, environment, background, lighting, framing and camera "
        "must come entirely from this prompt, not from the reference. Never open on "
        "or cut to the reference's environment; keep the character in the scene "
        "described here with a natural, continuous transition."
    )


def continuity_instruction(job: JobState, motion_context: bool = False) -> str:
    """Per-shot continuity rules appended to the long-video prompt.

    Two hand-off regimes exist and they need different wording:

      * tail-frame (stable): the previous shot's final frame is pinned as this
        shot's first frame, but the model still has to "imagine" the transition
        and the audio is regenerated from scratch.
      * motion context: the previous shot's last `MOTION_CONTEXT_LENGTH` frames
        AND the last whole second of its audio are pinned onto the head of this
        shot and trimmed out after sampling. The model therefore already *sees*
        the closing frames and the closing audio — the prompt must tell it to
        CONTINUE them, not restart, re-frame or re-describe them.

    The motion-context wording bakes in the traps documented in the
    ComfyUI-H3-Motion-Context README: the model renders contradictory casts as
    a *union* (not a swap), a held framing with nothing in it renders as a
    literal freeze, and the pinned head is short of the sampled length so the
    opening ~2s should hold the incoming framing with small life before the new
    action.
    """
    if job.segment_index == 1:
        text = (
            "This is the opening segment. Perform only the CURRENT SEGMENT action. "
            "Do not advance into later segments."
        )
        if motion_context:
            text += (
                " The opening frames and the final second of audio from the previous "
                "source are already pinned to the start of this clip. Do not re-stage "
                "or replay them: keep the same cast, framing and motion direction, and "
                "let the action flow out of them."
            )
        if not motion_context and job.audio_reference_name:
            text += (
                " Keep the same music bed, tempo, instrumentation and ambience as "
                "<Audio 1>, while generating new audio for this segment."
            )
        return text

    if motion_context:
        text = (
            "Continue directly from the pinned opening frames. The closing frames of "
            "the previous shot and the last whole second of its audio are already "
            "baked into the head of this clip and will be trimmed out of the "
            "delivered file, so do not restart, re-frame or re-describe them. For "
            "the first ~2 seconds keep the same character identity, framing and "
            "motion direction with only small natural life (a breath, a weight "
            "shift, an eyeline change) so the held shot never reads as a freeze; "
            "then carry into the CURRENT SEGMENT action. Keep exactly the same cast "
            "as the pinned frames — do not add or remove any person or object, "
            "because the model renders a contradictory cast as a union, not a "
            "swap. Do not replay any earlier action."
        )
        # The pinned audio is the real tail of the soundtrack; the rule is to
        # keep that bed going, not to reference a separate asset.
        text += (
            " The final second of the soundtrack is already pinned; keep the same "
            "music bed, tempo, instrumentation and ambience flowing and generate "
            "only the new audio for the remainder of this shot."
        )
        return text

    # Stable tail-frame hand-off (no latent continuation).
    text = (
        "Directly continue from the supplied first frame. During the first second, "
        "keep the same character identity, pose, framing and motion direction with "
        "only small natural movement; then perform only the CURRENT SEGMENT action. "
        "Do not restart the scene or replay any earlier action."
    )
    if job.audio_reference_name:
        text += (
            " Keep the same music bed, tempo, instrumentation and ambience as "
            "<Audio 1>, while generating new audio for this segment."
        )
    return text


def segment_prompt(job: JobState, motion_context: bool = False) -> str:
    """Give each long-video segment a focused time window and continuity rule.

    For Ref2VA jobs the reference media is an identity anchor only, so an
    appearance-only rule is prepended to stop the model copying the reference's
    scene/background into the output (the cause of the hard scene cut).
    """
    base = _segment_prompt_core(job, motion_context)
    rule = ref2va_reference_rule(job)
    return f"{rule}\n\n{base}" if rule else base


# H3 renders picture and sound jointly, so a GLOBAL block that mentions
# dialogue lines makes EVERY shot try to speak. Shots that carry no line then
# invent words, and with Motion Context pinning the previous shot's audio the
# babble continues from shot to shot — which is exactly the "someone muttering
# in an alien language" defect. This marker also tells
# apply_speech_language_rule() to skip the spoken-language clause.
NO_DIALOGUE_MARKER = "NO SPOKEN DIALOGUE IN THIS SEGMENT"
_QUOTED_SPEECH_RE = re.compile(
    # A complete quoted line, or an opening quote left unterminated because the
    # per-shot splitter cut the sentence at the full stop inside the quote
    # (`她低声说："……怎么现在才回来呀。` — the closing mark lands in the
    # next chunk). Both spellings mean this shot does speak.
    r"[\"“「『][^\"“”「」『』\n]{1,300}[\"“”」』]"
    r"|[\"“「『][^\"“”「」『』\n]{2,300}"
)


def segment_dialogue_rule(action_text: str) -> str:
    """Return a silence clause when this shot's own text has no spoken line."""
    if _QUOTED_SPEECH_RE.search(action_text or ""):
        return ""
    return (
        f"{NO_DIALOGUE_MARKER}: nobody speaks, whispers or murmurs. This "
        "overrides any dialogue line, spoken-language rule or pinned audio "
        "stated earlier — do not continue or echo any speech from the pinned "
        "context. The audio is room tone, foley, cloth, footsteps and "
        "breathing only: no words and no vocalisations in any language."
    )


def _segment_prompt_core(job: JobState, motion_context: bool = False) -> str:
    """Core per-segment prompt assembly (no Ref2VA reference scoping)."""
    if job.shot_plan:
        shot = job.shot_plan[job.segment_index - 1]
        continuity = continuity_instruction(job, motion_context)
        blocks = []
        if job.story_global_text:
            blocks.append(f"GLOBAL CONTINUITY RULES:\n{job.story_global_text}")
        blocks.extend(
            [
                (
                    f"LONG VIDEO SHOT {job.segment_index}/{job.segment_total}; "
                    f"story time window {shot.start_seconds:g}-{shot.end_seconds:g} seconds.\n"
                    f"{continuity}\n"
                    "Preserve the exact same character identity, face, hairstyle, costume, "
                    "props, location continuity, lighting direction and camera language. "
                    "This shot must begin from the supplied continuation frame and must not "
                    "repeat any earlier action."
                ),
                f"CURRENT SHOT ACTION — {shot.label}:\n{shot.action}",
            ]
        )
        silence = segment_dialogue_rule(shot.action)
        if silence:
            blocks.append(silence)
        return "\n\n".join(blocks)

    parsed = parse_segmented_prompt(job.prompt)
    start_seconds = (job.segment_index - 1) * MAX_SEGMENT_SECONDS
    end_seconds = min(job.segment_index * MAX_SEGMENT_SECONDS, job.total_seconds)
    continuity = continuity_instruction(job, motion_context)

    if parsed is not None:
        current = parsed.segments.get(job.segment_index)
        if current is None:
            raise BotError(f"分段提示詞缺少 SEGMENT {job.segment_index}。")
        blocks = []
        if parsed.global_text:
            blocks.append(f"GLOBAL CONTINUITY RULES:\n{parsed.global_text}")
        blocks.extend(
            [
                (
                    f"LONG VIDEO SEGMENT {job.segment_index}/{job.segment_total}; "
                    f"story time window {start_seconds:g}-{end_seconds:g} seconds.\n"
                    f"{continuity}\n"
                    "Preserve the same characters, costumes, location, lighting and "
                    "camera direction across the cut."
                ),
                f"CURRENT SEGMENT ACTION:\n{current}",
            ]
        )
        silence = segment_dialogue_rule(current)
        if silence:
            blocks.append(silence)
        return "\n\n".join(blocks)

    if job.segment_total <= 1:
        return job.prompt
    return (
        f"{job.prompt}\n\n"
        f"LONG VIDEO SEGMENT {job.segment_index}/{job.segment_total}; "
        f"story time window {start_seconds:g}-{end_seconds:g} seconds.\n"
        f"{continuity}\n"
        "Advance the story to this time window only. Preserve the same characters, costumes, "
        "location, lighting, camera language and motion direction. Smoothly carry over the "
        "last pose and momentum from the previous segment. Do not replay earlier events."
    )


def _workflow_node_input(
    workflow: dict[str, Any], node_id: str, name: str, default: Any = ""
) -> Any:
    node = workflow.get(str(node_id), {})
    if not isinstance(node, dict):
        return default
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        return default
    value = inputs.get(name, default)
    return default if isinstance(value, (list, dict)) else value


# Fixed node ids used by the official-core graph: keep dynamic LoadImage /
# LoadVideo / LoadAudio nodes off these so the two can never collide.
_RESERVED_NODE_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13,
    15, 16, 17, 18, 19, 20, 21, 22, 23,
    101, 102, 103, 104, 105, 106, 107,
)


def _next_workflow_node_id(workflow: dict[str, Any]) -> str:
    """Return an unused numeric node id without overwriting template nodes."""
    numeric_ids = [int(node_id) for node_id in workflow if str(node_id).isdigit()]
    next_id = max(numeric_ids + list(_RESERVED_NODE_IDS), default=0) + 1
    while str(next_id) in workflow:
        next_id += 1
    return str(next_id)


def _model_filename(value: Any) -> str:
    text = str(value or "").strip()
    return Path(text.replace("\\", "/")).name or "未設定"


def workflow_usage_report(workflow: dict[str, Any], vram_mode: str) -> str:
    """Describe the actual models and acceleration nodes in the submitted graph."""
    class_types = {
        str(node.get("class_type"))
        for node in workflow.values()
        if isinstance(node, dict) and node.get("class_type")
    }
    # Official core graph: node 7 = KSamplerSelect (sampler name), node 13 =
    # BasicScheduler (steps/scheduler). Two-stage refine adds node 104.
    sampler_name = str(
        _workflow_node_input(workflow, "7", "sampler_name", "res_multistep")
    )
    steps = (
        f"{_workflow_node_input(workflow, '13', 'steps', '未設定')} steps / "
        f"scheduler {_workflow_node_input(workflow, '13', 'scheduler', 'simple')}"
    )
    if "104" in workflow:
        steps += (
            f" → 精修 {_workflow_node_input(workflow, '104', 'steps', '未設定')} steps / "
            f"denoise {_workflow_node_input(workflow, '104', 'denoise', '0.5')}"
        )

    acceleration_labels = {
        "MiniMaxH3MotionContext": "Motion Context",
        "MiniMaxH3MotionContextLoadLatent": "Motion Context latent",
        "MiniMaxH3MemoryEfficientSageAttentionPatch": "Memory-efficient SageAttention",
        "PathchSageAttentionKJ": "SageAttention KJ",
        "ApplyMiniMaxH3FirstBlockCache": "First Block Cache",
        "SpectrumApplyMiniMaxH3": "Spectrum H3",
        "MiniMaxLowVRAMAttention": "LowVRAM Attention",
        "MiniMaxChunkFeedForward": "Chunk FeedForward",
    }
    acceleration = []
    if SAGE_ATTENTION_ENABLED:
        acceleration.append("SageAttention")
    if "LoraLoaderBypassModelOnly" in class_types or "LoraLoaderModelOnly" in class_types:
        acceleration.append("Turbo LoRA")
    if sampler_name == "res_multistep":
        acceleration.append("res_multistep (官方)")
    if "H3AVLatentJoin" in class_types:
        acceleration.append("兩段式 latent 放大")
    for class_type, label in acceleration_labels.items():
        if class_type in class_types and label not in acceleration:
            acceleration.append(label)

    try:
        vram_label = comfyui_vram_mode_label(vram_mode)
    except NameError:
        vram_label = vram_mode
    return "\n".join(
        [
            f"任務節點：{str((workflow.get('6') or {}).get('class_type', '未設定'))}",
            f"採樣：{sampler_name} | 步數：{steps}",
            f"主模型：{_model_filename(_workflow_node_input(workflow, '4', 'unet_name'))}",
            f"CLIP：{_model_filename(_workflow_node_input(workflow, '3', 'clip_name'))}",
            f"Turbo LoRA：{_model_filename(_workflow_node_input(workflow, '5', 'lora_name'))}",
            f"加速組件：{'、'.join(acceleration) if acceleration else '無額外加速節點'}",
            f"顯存模式：{vram_label}",
        ]
    )


def format_elapsed(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    minutes, remainder = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} 分 {remainder:02d} 秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 小時 {minutes:02d} 分 {remainder:02d} 秒"


def completion_report(
    job: JobState,
    elapsed_seconds: float,
    duration_seconds: Optional[float] = None,
    config: Optional[GenerationConfig] = None,
    partial: bool = False,
) -> str:
    display_config = config or job.config
    duration = (
        float(duration_seconds)
        if duration_seconds is not None
        else display_config.actual_seconds
    )
    lines = [
        "📊 長片部分結果資訊" if partial else "📊 本次生成資訊",
        f"⏱ 總耗時：{format_elapsed(elapsed_seconds)}",
        f"🎞 影片：{display_config.width}×{display_config.height} / {duration:.2f} 秒",
    ]
    reports = job.workflow_reports or ["未記錄工作流資訊"]
    for index, report in enumerate(reports, start=1):
        if len(reports) > 1:
            lines.append(f"\n工作流配置 {index}：")
        else:
            lines.append("\n實際使用配置：")
        lines.append(report)
    if job.resolution_fallbacks:
        lines.append("\n顯存自動降級記錄：")
        lines.extend(f"- {fallback}" for fallback in job.resolution_fallbacks)
    return "\n".join(lines)


def apply_speech_language_rule(prompt: str) -> str:
    """Pin a default spoken language so H3 does not sample random accents.

    H3 generates audio and video jointly. The model has no way to know the
    requested accent from a Chinese prompt alone, so it samples from its
    whole Chinese distribution: one clip comes out standard Mandarin, the next
    Cantonese, the next somewhere in between. If the prompt contains Chinese
    text but names no spoken language, pin the default to standard Mandarin
    (普通話) so the accent is consistent inside and across clips. When the
    user explicitly names a language or accent, the prompt is left untouched
    and the user's choice wins.
    """
    if not any("\u4e00" <= ch <= "\u9fff" for ch in prompt):
        return prompt
    if NO_DIALOGUE_MARKER in prompt:
        # This shot was explicitly marked silent; a spoken-language clause here
        # would only invite the model to invent dialogue.
        return prompt
    lowered = prompt.lower()
    explicit = (
        "普通话" in prompt
        or "普通話" in prompt
        or "粤语" in prompt
        or "粵語" in prompt
        or "广东话" in prompt
        or "廣東話" in prompt
        or "広東話" in prompt
        or "mandarin" in lowered
        or "cantonese" in lowered
    )
    if explicit:
        return prompt
    return (
        f"{prompt}\n\n"
        "All spoken dialogue is standard Mandarin Chinese (普通話), the same "
        "accent for every line and every character; do not use Cantonese or "
        "any regional accent."
    )


def build_workflow(
    config: GenerationConfig,
    prompt: str,
    output_prefix: str = OUTPUT_PREFIX,
    image_name: Optional[str] = None,
    last_image_name: Optional[str] = None,
    reference_image_names: Optional[list[str]] = None,
    reference_video_names: Optional[list[str]] = None,
    reference_audio_names: Optional[list[str]] = None,
    audio_reference_name: Optional[str] = None,
    generation_mode: str = INPUT_MODE_TEXT,
    motion_context: bool = False,
    context_video_name: Optional[str] = None,
    context_latent_path: Optional[str] = None,
    load_latent_clip_index: int = 0,
    save_latent_prefix: Optional[str] = None,
    save_latent_clip_index: Optional[int] = None,
    latent_upscale: Optional[bool] = None,
    h3_profile: str = H3_PROFILE_DEFAULT,
) -> dict[str, Any]:
    """Build the stock-core MiniMax H3 graph (no T8 nodes).

    The conditioning node is the official core ``MiniMaxH3ReferenceToVideo``
    (ref2va) or ``MiniMaxH3ImageToVideo`` (t2va/i2va/fl2va), sampling runs
    through ``res_multistep`` + a custom guider/scheduler, and decode uses the
    core ``VAEDecode`` (video) + ``VAEDecodeAudio`` (audio) + ``CreateVideo``
    + ``SaveVideo``. This matches the author's (YZ_金鱼) official-node
    workflow. The two-stage latent-upscale path keeps the LBH 3D upscaler but
    splits/joins the joint AV latent with the bridge nodes
    (``H3AVLatentSeparate`` / ``H3AVLatentJoin``) instead of T8's. Long-video
    continuation keeps the (non-T8) ``MiniMaxH3MotionContext`` nodes.
    """
    if latent_upscale is None:
        latent_upscale = LATENT_UPSCALE_ENABLED
    if not T8_API_TEMPLATE.is_file():
        raise BotError(f"找不到官方核心工作流模板：{T8_API_TEMPLATE}")
    with T8_API_TEMPLATE.open("r", encoding="utf-8") as handle:
        workflow = json.load(handle)

    workflow["1"]["inputs"]["vae_name"] = VIDEO_VAE
    workflow["2"]["inputs"]["vae_name"] = AUDIO_VAE
    workflow["3"]["inputs"]["clip_name"] = CLIP_NAME
    mode = normalize_input_mode(generation_mode)
    profile = str(h3_profile or H3_PROFILE_DEFAULT).strip().lower()
    if profile not in H3_PROFILES:
        profile = H3_PROFILE_DEFAULT
    fused = profile == H3_PROFILE_FUSED
    if fused:
        # The fused file covers every conditioning mode, so the FL2VA/Ref2VA
        # split and the separate turbo LoRA both go away (and with them the
        # ~21GB of pristine-weight backups ComfyUI keeps for a patched model).
        workflow["4"]["inputs"]["unet_name"] = FUSED_UNET_NAME
        workflow.pop("5", None)
        model_source: list[Any] = ["4", 0]
    else:
        workflow["4"]["inputs"]["unet_name"] = (
            REF2VA_UNET_NAME if mode == INPUT_MODE_REF2VA else UNET_NAME
        )
        # Official Lightning mode: standard model-only LoRA loading (matches
        # YZ_金鱼's R2V template). The 4-step Turbo LoRA is trained for the
        # pruned INT8 ConvRot model and loads without bypass.
        workflow["5"]["class_type"] = "LoraLoaderModelOnly"
        workflow["5"]["inputs"]["lora_name"] = LORA_NAME
        workflow["5"]["inputs"]["strength_model"] = 1.0
        model_source = ["5", 0]
    effective_steps = FUSED_PROFILE_STEPS if fused else max(4, config.steps)
    if fused and h3_sla_available():
        # SLA block-sparse attention: ~40% faster and audibly cleaner than
        # dense at 4 steps. Omitted when the pack is missing so the fused
        # profile still runs (just slower and with a duller soundtrack).
        workflow[SLA_NODE_ID] = {
            "inputs": {
                "model": model_source,
                "sparsity_ratio": 0.9,
                "block_size": "64",
                "min_seq_len": 8192,
                "dense_last_steps": 0,
                "protect_audio": True,
                "enabled": True,
            },
            "class_type": SLA_NODE_CLASS,
            "_meta": {"title": "H3 SLA sparse attention (fused profile)"},
        }
        model_source = [SLA_NODE_ID, 0]
    prompt = apply_speech_language_rule(prompt)

    reference_image_names = list(reference_image_names or [])
    reference_video_names = list(reference_video_names or [])
    reference_audio_names = list(reference_audio_names or [])
    if len(reference_image_names) > MAX_REF2VA_IMAGES:
        raise BotError(f"Ref2VA 最多支援 {MAX_REF2VA_IMAGES} 張參考圖。")
    if len(reference_video_names) > MAX_REF2VA_VIDEOS:
        raise BotError(f"Ref2VA 最多支援 {MAX_REF2VA_VIDEOS} 段參考影片。")
    if len(reference_audio_names) > MAX_REF2VA_AUDIOS:
        raise BotError(f"Ref2VA 最多支援 {MAX_REF2VA_AUDIOS} 段參考音訊。")

    # A standalone audio reference (previous segment's soundtrack) maps onto
    # the official R2V ``ref_audios``; the I2V core node has no audio
    # reference input, so there it is dropped and audio is generated natively
    # (the motion-context path carries the soundtrack via the context latent).
    use_ref2va = mode == INPUT_MODE_REF2VA and (
        reference_image_names or reference_video_names or reference_audio_names
    )
    if use_ref2va and audio_reference_name and not motion_context:
        reference_audio_names = list(reference_audio_names) + [audio_reference_name]
        if len(reference_audio_names) > MAX_REF2VA_AUDIOS:
            reference_audio_names = reference_audio_names[:MAX_REF2VA_AUDIOS]
        audio_reference_name = None

    if mode == INPUT_MODE_REF2VA and not use_ref2va:
        raise BotError("Ref2VA 至少需要一張參考圖片、一段參考影片或一段參考音訊。")

    # Motion context and Ref2VA reference media are mutually exclusive (the
    # motion-context nodes own the continuation). Clear the reference lists
    # BEFORE any reference nodes are created so no orphaned LoadImage /
    # LoadVideo / LoadAudio nodes are left in the graph; the conditioning node
    # then falls back to the I2V core node.
    if motion_context and (
        reference_image_names or reference_video_names or reference_audio_names
    ):
        bot_log(
            "build_workflow: motion context active; clearing Ref2VA reference "
            "media to avoid node id conflicts"
        )
        reference_image_names = []
        reference_video_names = []
        reference_audio_names = []
        use_ref2va = False

    # ---- conditioning node (id 6) ---------------------------------------
    # Built as a full dict; reference media and keyframes are wired onto it
    # below. The node id (6) and the dict identity both stay stable so the
    # stage-2 deep copy preserves every reference on the refine pass.
    cond: dict[str, Any] = {
        "prompt": prompt.strip(),
        "width": config.width,
        "height": config.height,
        "length": config.length,
        "clip": ["3", 0],
        "vae": ["1", 0],
    }
    if use_ref2va:
        cond["audio_vae"] = ["2", 0]
        cond["ref_image_size"] = "match"
        workflow["6"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": cond,
            "_meta": {"title": "H3 Reference to Video (official core)"},
        }
    else:
        workflow["6"] = {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": cond,
            "_meta": {"title": "H3 Image to Video (official core)"},
        }

    # ---- sampler chain (fixed ids) --------------------------------------
    # id 7 = KSamplerSelect(res_multistep), 8 = RandomNoise (template),
    # 9 = BasicGuider, 10 = SamplerCustomAdvanced, 13 = BasicScheduler.
    workflow["7"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "res_multistep"},
        "_meta": {"title": "res_multistep sampler (official core)"},
    }
    workflow["9"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": model_source, "conditioning": ["6", 0]},
        "_meta": {"title": "Basic guider"},
    }
    workflow["10"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["8", 0],
            "guider": ["9", 0],
            "sampler": ["7", 0],
            "sigmas": ["13", 0],
            "latent_image": ["6", 1],
        },
        "_meta": {"title": "Sample (official core)"},
    }
    workflow["13"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "scheduler": "simple",
            "steps": effective_steps,
            "denoise": 1.0,
            "model": model_source,
        },
        "_meta": {"title": "Scheduler (official core)"},
    }

    # ---- reference media (R2V node only) + keyframes (I2V node only) ----
    # Wired after the sampler chain so _next_workflow_node_id sees the fixed
    # ids already reserved and never hands one of them out. (Ref2VA reference
    # media was already cleared above when motion context is active.)
    if image_name and not motion_context and not use_ref2va:
        image_node_id = _next_workflow_node_id(workflow)
        workflow[image_node_id] = {
            "inputs": {"image": image_name},
            "class_type": "LoadImage",
            "_meta": {"title": "Telegram input image"},
        }
        cond["first_frame"] = [image_node_id, 0]
    if last_image_name and not motion_context and mode in {
        INPUT_MODE_FL2VA,
        INPUT_MODE_REF2VA,
    }:
        last_image_node_id = _next_workflow_node_id(workflow)
        workflow[last_image_node_id] = {
            "inputs": {"image": last_image_name},
            "class_type": "LoadImage",
            "_meta": {
                "title": (
                    "Telegram Ref2VA tail frame"
                    if mode == INPUT_MODE_REF2VA
                    else "Telegram FL2VA last frame"
                )
            },
        }
        cond["last_frame"] = [last_image_node_id, 0]

    for index, reference_name in enumerate(reference_image_names):
        ref_node_id = _next_workflow_node_id(workflow)
        workflow[ref_node_id] = {
            "inputs": {"image": reference_name},
            "class_type": "LoadImage",
            "_meta": {"title": f"Telegram Ref2VA image {index + 1}"},
        }
        cond[f"ref_images.ref_image_{index}"] = [ref_node_id, 0]
    for index, reference_name in enumerate(reference_video_names):
        video_node_id = _next_workflow_node_id(workflow)
        workflow[video_node_id] = {
            "inputs": {"file": reference_name},
            "class_type": "LoadVideo",
            "_meta": {"title": f"Telegram Ref2VA video {index + 1}"},
        }
        components_node_id = _next_workflow_node_id(workflow)
        workflow[components_node_id] = {
            "inputs": {"video": [video_node_id, 0]},
            "class_type": "GetVideoComponents",
            "_meta": {"title": f"Ref2VA video {index + 1} frames and audio"},
        }
        cond[f"ref_videos.ref_video_{index}"] = [components_node_id, 0]
        cond[f"ref_video_audios.ref_video_audio_{index}"] = [components_node_id, 1]
    for index, reference_name in enumerate(reference_audio_names):
        audio_node_id = _next_workflow_node_id(workflow)
        workflow[audio_node_id] = {
            "inputs": {"audio": reference_name},
            "class_type": "LoadAudio",
            "_meta": {"title": f"Telegram Ref2VA audio {index + 1}"},
        }
        cond[f"ref_audios.ref_audio_{index}"] = [audio_node_id, 0]

    # ---- two-stage latent upscaling (YZ_金鱼) --------------------------
    latent_final = "10"  # sampler output that feeds decode
    if latent_upscale and not motion_context and (
        config.width >= 256 and config.height >= 256
    ):
        import copy

        # Stage 1 generates at half resolution (32-px grid), the video latent
        # is upscaled in latent space (audio carried through unchanged), then
        # stage 2 re-samples at full resolution with a partial denoise. The
        # stage-2 conditioning deep-copies node 6 so every reference and
        # keyframe is preserved on the refine pass.
        _lw = max(32, (config.width // 2) // 32 * 32)
        _lh = max(32, (config.height // 2) // 32 * 32)
        cond["width"] = _lw
        cond["height"] = _lh

        _stage2 = copy.deepcopy(cond)
        _stage2["width"] = config.width
        _stage2["height"] = config.height
        workflow["106"] = {
            "class_type": workflow["6"]["class_type"],
            "inputs": _stage2,
            "_meta": {"title": "Stage 2 full-resolution conditioning"},
        }
        workflow["101"] = {
            "class_type": "H3AVLatentSeparate",
            "inputs": {"av_latent": ["10", 0]},
        }
        workflow["102"] = {
            "class_type": "MinimaxH3LatentUpscaler3D",
            "inputs": {
                "latent": ["101", 0],
                "model_name": LATENT_UPSCALER_MODEL,
                "mode": "target dimensions",
                "mode.width": config.width,
                "mode.height": config.height,
                "align": 32,
                "enable_temporal_chunking": True,
                "force_unload": True,
                "device": "cuda",
                "precision": "fp16",
            },
        }
        workflow["103"] = {
            "class_type": "H3AVLatentJoin",
            "inputs": {
                "video_latent": ["102", 0],
                "audio_latent": ["101", 1],
            },
        }
        workflow["107"] = {
            "class_type": "BasicGuider",
            "inputs": {"model": model_source, "conditioning": ["106", 0]},
        }
        workflow["104"] = {
            "class_type": "BasicScheduler",
            "inputs": {
                "scheduler": "simple",
                "steps": effective_steps,
                "denoise": 0.5,
                "model": model_source,
            },
        }
        workflow["105"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["107", 0],
                "sampler": ["7", 0],
                "sigmas": ["104", 0],
                "latent_image": ["103", 0],
            },
        }
        latent_final = "105"

    # ---- motion context (non-T8) ----------------------------------------
    if motion_context:
        if not context_video_name or not context_latent_path:
            raise BotError("Motion Context 需要上一段影片和上一段 AV latent。")
        workflow["15"] = {
            "inputs": {"file": context_video_name},
            "class_type": "LoadVideo",
            "_meta": {"title": "Previous segment for H3 Motion Context"},
        }
        workflow["16"] = {
            "inputs": {"video": ["15", 0]},
            "class_type": "GetVideoComponents",
            "_meta": {"title": "Previous segment frames and audio"},
        }
        workflow["17"] = {
            "inputs": {
                "latent_path": context_latent_path,
                "clip_index": max(0, int(load_latent_clip_index)),
            },
            "class_type": "MiniMaxH3MotionContextLoadLatent",
            "_meta": {"title": "Previous H3 AV latent"},
        }
        workflow["18"] = {
            "inputs": {
                "conditioning": ["6", 0],
                "vae": ["1", 0],
                "latent": ["6", 1],
                "context_frames": ["16", 0],
                "context_length": str(MOTION_CONTEXT_LENGTH),
                "audio_context_length": MOTION_CONTEXT_AUDIO_LENGTH,
                "context_latent": ["17", 0],
            },
            "class_type": "MiniMaxH3MotionContext",
            "_meta": {"title": "Experimental H3 AV latent continuation"},
        }
        # Motion context pins the continuation onto the head; the guider then
        # samples with the context-adjusted conditioning.
        workflow["9"]["inputs"]["conditioning"] = ["18", 0]

    # ---- decode + mux (official core) -----------------------------------
    # id 21 = VAEDecode (video), 22 = VAEDecodeAudio (audio), 23 = CreateVideo,
    # 12 = SaveVideo. Motion context trims the pinned head (id 19) before mux.
    workflow["21"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [latent_final, 0], "vae": ["1", 0]},
        "_meta": {"title": "Decode H3 video (official core)"},
    }
    workflow["22"] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": [latent_final, 0], "vae": ["2", 0]},
        "_meta": {"title": "Decode H3 audio (official core)"},
    }
    video_src = ["21", 0]
    audio_src = ["22", 0]
    if motion_context:
        workflow["19"] = {
            "inputs": {
                "images": ["21", 0],
                "audio": ["22", 0],
                "trim_frames": ["18", 1],
                "fps": 24.0,
                "match_tail": True,
            },
            "class_type": "MiniMaxH3MotionContextTrim",
            "_meta": {"title": "Trim duplicated context audio and frames"},
        }
        video_src = ["19", 0]
        audio_src = ["19", 1]
    workflow["23"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": video_src,
            "audio": audio_src,
            "fps": 24.0,
            "bit_depth": 8,
        },
        "_meta": {"title": "Mux H3 video + audio (official core)"},
    }
    workflow["12"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["23", 0],
            "filename_prefix": output_prefix,
            "format": "mp4",
            "codec": "auto",
        },
        "_meta": {"title": "Save synchronized MP4 (official core)"},
    }

    if save_latent_prefix:
        workflow["20"] = {
            "inputs": {
                "latent": [latent_final, 0],
                "filename_prefix": save_latent_prefix,
                "clip_index": int(save_latent_clip_index or 0),
            },
            "class_type": "MiniMaxH3MotionContextSaveLatent",
            "_meta": {"title": "Save H3 AV latent for next segment"},
        }

    workflow["8"]["inputs"]["noise_seed"] = secrets.randbits(63)
    return workflow


def round_video_dimension(value: float) -> int:
    """Round a SeedVR2 target to a safe 32-pixel alignment."""
    return max(32, int(round(value / 32.0) * 32))


def upscale_dimensions(width: int, height: int, longer_edge: int) -> tuple[int, int]:
    """Preserve the source aspect ratio while choosing a SeedVR2 long edge."""
    scale = float(longer_edge) / max(width, height)
    return (
        round_video_dimension(width * scale),
        round_video_dimension(height * scale),
    )


def build_seedvr2_workflow(
    input_video_name: str,
    target_long_edge: int,
    output_prefix: str,
    split_latent: bool = False,
) -> dict[str, Any]:
    """Build the native ComfyUI SeedVR2 3B INT8 video-upscale graph."""
    if not SEEDVR2_API_TEMPLATE.is_file():
        raise BotError(f"找不到 SeedVR2 API 工作流模板：{SEEDVR2_API_TEMPLATE}")
    with SEEDVR2_API_TEMPLATE.open("r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    workflow["1"]["inputs"]["file"] = input_video_name
    resize_inputs = workflow["3"]["inputs"]
    resize_inputs["resize_type"] = "scale longer dimension"
    resize_inputs["resize_type.longer_size"] = int(target_long_edge)
    resize_inputs["scale_method"] = "lanczos"
    workflow["5"]["inputs"]["vae_name"] = SEEDVR2_VAE_NAME
    workflow["7"]["inputs"]["unet_name"] = SEEDVR2_UNET_NAME
    workflow["10"]["inputs"]["seed"] = secrets.randbits(63)
    workflow["14"]["inputs"]["filename_prefix"] = output_prefix
    if not split_latent:
        workflow["8"]["inputs"]["vae_conditioning"] = ["6", 0]
        workflow["10"]["inputs"]["latent_image"] = ["6", 0]
        workflow["12"]["inputs"]["samples"] = ["10", 0]
    return workflow


def seedvr2_usage_report(target_long_edge: int) -> str:
    acceleration = ["SeedVR2 3B INT8", "1-step", "tiled VAE", "automatic temporal chunks"]
    if SAGE_ATTENTION_ENABLED:
        acceleration.append("SageAttention")
    return "\n".join(
        [
            f"放大目標：長邊 {target_long_edge}px",
            f"SeedVR2 VAE：{SEEDVR2_VAE_NAME}",
            f"SeedVR2 模型：{SEEDVR2_UNET_NAME}",
            "放大配置：" + "、".join(acceleration),
            "ComfyUI 顯存模式：lowvram",
        ]
    )


def json_request(
    url: str,
    payload: Optional[dict[str, Any]] = None,
    timeout: float = 45.0,
    headers: Optional[dict[str, str]] = None,
) -> Any:
    data = None
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = Request(
        url, data=data, headers=request_headers, method="POST" if data else "GET"
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raise BotError(f"連線失敗：{http_error_detail(exc)}") from exc
    except (URLError, TimeoutError) as exc:
        raise BotError(f"連線失敗：{exc}") from exc
    except OSError as exc:
        # A connection reset/abort can surface while reading the body, and those
        # are plain OSError subclasses that neither URLError nor TimeoutError
        # covers. Callers rely on BotError to distinguish "the service said no"
        # from a crash, so an uncaught OSError here used to escape all the way
        # up and abort multi-step cleanup (notably stopping ComfyUI to restart
        # the LLM). Normalise it here so every caller is safe.
        raise BotError(f"連線中斷：{exc}") from exc
    if not raw:
        # ComfyUI's interrupt and a few control endpoints legitimately return
        # an empty 2xx body. Treat that as a successful empty response.
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise BotError("服務回傳了無法解析的資料。") from exc


def comfy_post(path: str, payload: Optional[dict[str, Any]] = None) -> Any:
    return json_request(f"{COMFY_URL}{path}", payload)


def yupi_lora_name() -> str:
    """Verify the YUPI LoRA is visible to the running ComfyUI instance."""
    try:
        object_info = json_request(f"{COMFY_URL}/object_info", timeout=15.0)
    except BotError as exc:
        raise BotError(f"無法檢查 YUPI LoRA：{exc}") from exc
    node_info = object_info.get("LoraLoaderModelOnly", {})
    choices = (
        node_info.get("input", {})
        .get("required", {})
        .get("lora_name", [[]])[0]
    )
    if YUPI_LORA_NAME in choices:
        return YUPI_LORA_NAME
    raise BotError(
        "YUPI LoRA 尚未被 ComfyUI 載入："
        f"{YUPI_LORA_NAME}。請確認檔案位於 ComfyUI\\models\\loras，"
        "然後重啟 ComfyUI。"
    )


def yupi_action_lora_name() -> str:
    """Verify the HMNSFW action adapter is visible to ComfyUI."""
    try:
        object_info = json_request(f"{COMFY_URL}/object_info", timeout=15.0)
    except BotError as exc:
        raise BotError(f"無法檢查 HMNSFW LoRA：{exc}") from exc
    node_info = object_info.get("LoraLoaderModelOnly", {})
    choices = (
        node_info.get("input", {})
        .get("required", {})
        .get("lora_name", [[]])[0]
    )
    if YUPI_ACTION_LORA_NAME in choices:
        return YUPI_ACTION_LORA_NAME
    raise BotError(
        "HMNSFW 動作 LoRA 尚未被 ComfyUI 載入："
        f"{YUPI_ACTION_LORA_NAME}。請從 "
        "https://huggingface.co/Hearmeman/minimax-h3-loras 下載後放到 "
        "ComfyUI\\models\\loras，再重啟 ComfyUI。"
    )


def yupi_fast_lora_name() -> str:
    """Verify the FastH3 6-step distill LoRA is visible to ComfyUI."""
    try:
        object_info = json_request(f"{COMFY_URL}/object_info", timeout=15.0)
    except BotError as exc:
        raise BotError(f"無法檢查 FastH3 LoRA：{exc}") from exc
    node_info = object_info.get("LoraLoaderModelOnly", {})
    choices = (
        node_info.get("input", {})
        .get("required", {})
        .get("lora_name", [[]])[0]
    )
    if YUPI_FAST_LORA_NAME in choices:
        return YUPI_FAST_LORA_NAME
    raise BotError(
        "FastH3 LoRA 尚未被 ComfyUI 載入："
        f"{YUPI_FAST_LORA_NAME}。請先用 ComfyUI-FastH3-Lora-Converter 轉檔，"
        "把產出的 .safetensors 放到 ComfyUI\\models\\loras，再重啟 ComfyUI。"
    )


def load_yupi_workflow(fast: bool = False) -> dict[str, Any]:
    """Load the isolated YUPI API workflow template (or its FAST variant)."""
    template = YUPI_FAST_API_TEMPLATE if fast else YUPI_API_TEMPLATE
    try:
        with template.open("r", encoding="utf-8") as handle:
            workflow = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BotError(f"找不到或無法讀取 YUPI 工作流：{exc}") from exc
    if not isinstance(workflow, dict):
        raise BotError("YUPI 工作流格式錯誤：頂層必須是物件。")
    return workflow


def yupi_generation_config(workflow: dict[str, Any]) -> GenerationConfig:
    """Build progress metadata from the actual YUPI workflow settings."""
    try:
        inputs = workflow["6"]["inputs"]
        width = int(inputs["width"])
        height = int(inputs["height"])
        length = int(inputs["length"])
        steps = int(workflow["13"]["inputs"]["steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BotError("YUPI 工作流缺少有效的解析度、影格數或 steps 設定。") from exc
    if width < 32 or height < 32 or length < 1 or steps < 1:
        raise BotError("YUPI 工作流的解析度、影格數或 steps 設定無效。")
    return GenerationConfig(
        width=width,
        height=height,
        steps=steps,
        requested_seconds=length / 24.0,
        length=length,
    )


def configure_yupi_workflow(
    workflow: dict[str, Any],
    config: GenerationConfig,
    prompt: str,
    reference_name: str,
    output_prefix: str,
) -> tuple[dict[str, Any], str]:
    """Inject one YUPI shot's settings into a fresh API workflow."""
    try:
        lora_name = yupi_lora_name()
        conditioning_inputs = workflow["6"]["inputs"]
        conditioning_inputs["prompt"] = prompt
        conditioning_inputs["width"] = config.width
        conditioning_inputs["height"] = config.height
        conditioning_inputs["length"] = config.length
        conditioning_inputs["ref_images"]["ref_image_0"] = ["14", 0]
        workflow["14"]["inputs"]["image"] = reference_name
        workflow["5"]["inputs"]["lora_name"] = lora_name
        if "16" in workflow:
            # Anatomy -> action: the HMNSFW adapter sits above the stills-
            # trained anatomy file (node 16 -> node 5 -> node 4).
            action_lora = yupi_action_lora_name()
            workflow["16"]["inputs"]["lora_name"] = action_lora
            lora_name = f"{lora_name} + {action_lora}"
        if "15" in workflow:
            # YUPI_FAST: the FastH3 6-step distill LoRA is chained last
            # (node 15 -> node 16 -> node 5 -> node 4). Validate it is present.
            fast_lora = yupi_fast_lora_name()
            workflow["15"]["inputs"]["lora_name"] = fast_lora
            lora_name = f"{lora_name} + {fast_lora}"
        workflow["8"]["inputs"]["noise_seed"] = secrets.randbits(63)
        workflow["13"]["inputs"]["steps"] = config.steps
        workflow["12"]["inputs"]["filename_prefix"] = output_prefix
    except (KeyError, TypeError, ValueError) as exc:
        raise BotError("YUPI 工作流缺少必要的節點或輸入欄位。") from exc
    return workflow, lora_name


def attach_yupi_motion_context(
    workflow: dict[str, Any],
    context_video_name: Optional[str],
    context_latent_path: Optional[str],
    load_latent_clip_index: int = 0,
) -> dict[str, Any]:
    """Inject the H3 Motion Context chain into an isolated YUPI API graph.

    The stock graph reserves ids 15-20 for Motion Context, but the YUPI_FAST
    graph already uses id 15 for the FastH3 LoRA, so this uses 31-35 instead.
    Wiring mirrors build_workflow(): the guider's conditioning is re-pointed at
    the context node and CreateVideo takes the trimmed output rather than the
    raw decode. The latent save node is added separately by
    attach_yupi_save_latent(), because shot 1 needs to save without loading.
    """
    if not context_video_name or not context_latent_path:
        raise BotError("YUPI Motion Context 需要上一段影片和上一段 AV latent。")
    try:
        workflow["31"] = {
            "inputs": {"file": context_video_name},
            "class_type": "LoadVideo",
            "_meta": {"title": "YUPI: previous segment for Motion Context"},
        }
        workflow["32"] = {
            "inputs": {"video": ["31", 0]},
            "class_type": "GetVideoComponents",
            "_meta": {"title": "YUPI: previous segment frames and audio"},
        }
        workflow["33"] = {
            "inputs": {
                "latent_path": context_latent_path,
                "clip_index": max(0, int(load_latent_clip_index)),
            },
            "class_type": "MiniMaxH3MotionContextLoadLatent",
            "_meta": {"title": "YUPI: previous H3 AV latent"},
        }
        workflow["34"] = {
            "inputs": {
                "conditioning": ["6", 0],
                "vae": ["1", 0],
                "latent": ["6", 1],
                "context_frames": ["32", 0],
                "context_length": str(MOTION_CONTEXT_LENGTH),
                "audio_context_length": MOTION_CONTEXT_AUDIO_LENGTH,
                "context_latent": ["33", 0],
            },
            "class_type": "MiniMaxH3MotionContext",
            "_meta": {"title": "YUPI: AV latent continuation"},
        }
        workflow["9"]["inputs"]["conditioning"] = ["34", 0]
        workflow["35"] = {
            "inputs": {
                "images": ["21", 0],
                "audio": ["22", 0],
                "trim_frames": ["34", 1],
                "fps": 24.0,
                "match_tail": True,
            },
            "class_type": "MiniMaxH3MotionContextTrim",
            "_meta": {"title": "YUPI: trim duplicated context audio and frames"},
        }
        workflow["23"]["inputs"]["images"] = ["35", 0]
        workflow["23"]["inputs"]["audio"] = ["35", 1]
    except (KeyError, TypeError, ValueError) as exc:
        raise BotError(f"YUPI Motion Context 接線失敗：{exc}") from exc
    return workflow


def attach_yupi_save_latent(
    workflow: dict[str, Any],
    save_latent_prefix: Optional[str],
    save_latent_clip_index: Optional[int] = None,
) -> dict[str, Any]:
    """Add the latent-save node so the NEXT YUPI shot can continue from it.

    This must be attached on EVERY shot of a motion-context run, including the
    first one — the first shot has no context to load, but it still has to
    write the latent that shot 2 reads. build_workflow() does the same: its
    save node is outside the ``if motion_context`` block.
    """
    if not save_latent_prefix:
        return workflow
    workflow["36"] = {
        "inputs": {
            "latent": ["10", 0],
            "filename_prefix": save_latent_prefix,
            "clip_index": int(save_latent_clip_index or 0),
        },
        "class_type": "MiniMaxH3MotionContextSaveLatent",
        "_meta": {"title": "YUPI: save H3 AV latent for next segment"},
    }
    return workflow


def unload_comfy_models() -> None:
    """Release the previously loaded model so a new job starts clean."""
    try:
        comfy_post("/free", {"unload_models": True, "free_memory": True})
    except BotError:
        # ComfyUI may be stopped; the normal model loader will handle that
        # case when the selected workflow is submitted.
        pass


def motion_context_nodes_available() -> bool:
    """Check that the experimental AV-latent continuation nodes are loaded."""
    required = {
        "MiniMaxH3MotionContext",
        "MiniMaxH3MotionContextTrim",
        "MiniMaxH3MotionContextSaveLatent",
        "MiniMaxH3MotionContextLoadLatent",
    }
    try:
        object_info = json_request(f"{COMFY_URL}/object_info", timeout=15.0)
        return required.issubset(object_info.keys())
    except Exception:
        return False


_sla_available_cache: Optional[bool] = None
_sla_check_failed_at: float = 0.0


def h3_sla_available() -> bool:
    """True when the H3 SLA sparse-attention node pack is loaded.

    The fused profile inserts the node only when it exists, so a missing pack
    degrades to dense attention instead of failing the whole generation.

    A positive result is cached for the life of the process. A failure is NOT:
    the first check often runs before ComfyUI has finished starting (the Bot
    launches ComfyUI on demand), and caching that miss permanently disabled SLA
    for every later run in the same process. Failures now retry after a short
    cooldown instead.
    """
    global _sla_available_cache, _sla_check_failed_at
    if _sla_available_cache:
        return True
    now = time.time()
    if now - _sla_check_failed_at < 60.0:
        return False
    try:
        info = json_request(
            f"{COMFY_URL}/object_info/{SLA_NODE_CLASS}", timeout=10.0
        )
        if isinstance(info, dict) and SLA_NODE_CLASS in info:
            _sla_available_cache = True
            return True
    except Exception:
        pass
    _sla_check_failed_at = now
    return False


def is_motion_context_layout_error(error: BaseException) -> bool:
    """Recognize the pack's old-layout refusal, including wrapped BotErrors."""
    seen: set[int] = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if "h3_motion_context" in message and "older h3 layout" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def comfy_model_candidates(model_name: str) -> tuple[Path, ...]:
    """Return the local ComfyUI model paths used by this installation."""
    name = str(model_name or "").replace("/", "\\").strip("\\")
    if not name:
        return tuple()
    return tuple(
        dict.fromkeys(
            (
                COMFYUI_BASE_DIR / "models" / "diffusion_models" / name,
                COMFYUI_DIR / "models" / "diffusion_models" / name,
                Path(r"E:\Comfy\ComfyUI\ComfyUI\models\diffusion_models") / name,
            )
        )
    )


def comfy_model_available(model_name: str) -> bool:
    return any(path.is_file() for path in comfy_model_candidates(model_name))


def require_ref2va_model() -> None:
    if comfy_model_available(REF2VA_UNET_NAME):
        return
    raise BotError(
        "Ref2VA 主模型尚未安裝："
        f"{REF2VA_UNET_NAME}。請放到 ComfyUI\\models\\diffusion_models，"
        "再按一次生成。"
    )


def upload_image_to_comfy(image_path: Path) -> str:
    """Upload a Telegram image to ComfyUI input and return its LoadImage name."""
    if not image_path.is_file():
        raise BotError(f"找不到輸入圖片：{image_path}")
    boundary = f"----MiniMaxH3Image{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("ascii")
    remote_name = f"telegram_{uuid.uuid4().hex}{image_path.suffix.lower() or '.jpg'}"
    chunks: list[bytes] = []
    for name, value in {
        "type": "input",
        "subfolder": "TelegramInputs",
        "overwrite": "true",
    }.items():
        chunks.extend(
            [
                b"--" + boundary_bytes + b"\r\n",
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                    "utf-8"
                ),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            b"--" + boundary_bytes + b"\r\n",
            (
                f'Content-Disposition: form-data; name="image"; '
                f'filename="{remote_name}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: application/octet-stream\r\n\r\n",
            image_path.read_bytes(),
            b"\r\n--" + boundary_bytes + b"--\r\n",
        ]
    )
    request = Request(
        f"{COMFY_URL}/upload/image",
        data=b"".join(chunks),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise BotError(f"圖片上傳到 ComfyUI 失敗：{http_error_detail(exc)}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise BotError(f"圖片上傳到 ComfyUI 失敗：{exc}") from exc
    if not result.get("name"):
        raise BotError(f"ComfyUI 沒有回傳圖片名稱：{result}")
    subfolder = str(result.get("subfolder", "")).strip("/\\")
    name = str(result["name"])
    return f"{subfolder}/{name}" if subfolder else name


def upload_audio_to_comfy(audio_path: Path) -> str:
    """Upload an MP4/WAV reference that ComfyUI's LoadAudio can read."""
    if not audio_path.is_file():
        raise BotError(f"找不到音訊參考檔：{audio_path}")
    boundary = f"----MiniMaxH3Audio{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("ascii")
    remote_name = f"telegram_audio_{uuid.uuid4().hex}{audio_path.suffix.lower() or '.wav'}"
    chunks: list[bytes] = []
    for name, value in {
        "type": "input",
        "subfolder": "TelegramAudio",
        "overwrite": "true",
    }.items():
        chunks.extend(
            [
                b"--" + boundary_bytes + b"\r\n",
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            b"--" + boundary_bytes + b"\r\n",
            (
                f'Content-Disposition: form-data; name="image"; '
                f'filename="{remote_name}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: application/octet-stream\r\n\r\n",
            audio_path.read_bytes(),
            b"\r\n--" + boundary_bytes + b"--\r\n",
        ]
    )
    request = Request(
        f"{COMFY_URL}/upload/image",
        data=b"".join(chunks),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise BotError(f"音訊參考上傳到 ComfyUI 失敗：{http_error_detail(exc)}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise BotError(f"音訊參考上傳到 ComfyUI 失敗：{exc}") from exc
    if not result.get("name"):
        raise BotError(f"ComfyUI 沒有回傳音訊名稱：{result}")
    subfolder = str(result.get("subfolder", "")).strip("/\\")
    name = str(result["name"])
    return f"{subfolder}/{name}" if subfolder else name


def upload_video_to_comfy(video_path: Path) -> str:
    """Upload a previous MP4 so LoadVideo can expose its frame batch."""
    return upload_audio_to_comfy(video_path)


_comfy_start_lock = threading.Lock()
_comfy_process: Optional[subprocess.Popen] = None
_llama_start_lock = threading.Lock()
_llama_process: Optional[subprocess.Popen] = None


def normalize_comfyui_vram_mode(mode: Optional[str]) -> str:
    aliases = {
        "low": "lowvram",
        "lowvram": "lowvram",
        "quick": "lowvram",
        "turbo": "lowvram",
    }
    return aliases.get(str(mode or "").strip().lower(), DEFAULT_COMFYUI_VRAM_MODE)


def comfyui_vram_mode_label(mode: Optional[str]) -> str:
    return "Turbo（--lowvram）"


def comfyui_is_online() -> bool:
    try:
        json_request(f"{COMFY_URL}/system_stats", timeout=4)
        return True
    except (BotError, OSError):
        # OSError is listed for the same reason as llama_is_online(): a server
        # shutting down can reset the socket, and that must read as "offline"
        # rather than escaping as a crash.
        return False


def comfyui_has_pending_work() -> bool:
    """Return whether ComfyUI currently has a running or pending queue item."""
    try:
        queue = json_request(f"{COMFY_URL}/queue", timeout=4)
    except BotError:
        return False
    if not isinstance(queue, dict):
        return False
    return bool(queue.get("queue_running") or queue.get("queue_pending"))


def start_comfyui_process(vram_mode: Optional[str] = None) -> str:
    """Start this user's local Turbo ComfyUI only when its API is offline."""
    global _comfy_process
    vram_mode = normalize_comfyui_vram_mode(vram_mode)
    if comfyui_is_online():
        return f"ComfyUI 已經在運行（{COMFY_URL}）。"

    with _comfy_start_lock:
        if comfyui_is_online():
            return f"ComfyUI 已經在運行（{COMFY_URL}）。"
        if _comfy_process is not None and _comfy_process.poll() is None:
            return f"ComfyUI 正在啟動中（PID {_comfy_process.pid}）。"
        if not COMFYUI_DIR.is_dir():
            raise BotError(f"找不到 ComfyUI 資料夾：{COMFYUI_DIR}")
        if not COMFYUI_BASE_DIR.is_dir():
            raise BotError(f"找不到 ComfyUI base-directory：{COMFYUI_BASE_DIR}")
        if not COMFYUI_PYTHON.is_file():
            raise BotError(f"找不到 ComfyUI Python：{COMFYUI_PYTHON}")

        COMFYUI_LOG.parent.mkdir(parents=True, exist_ok=True)
        COMFYUI_USER_DIR.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{COMFYUI_DATABASE.as_posix()}"
        memory_flags = ["--lowvram"]
        if SAGE_ATTENTION_ENABLED:
            memory_flags.append("--use-sage-attention")
        command = [
            str(COMFYUI_PYTHON),
            "main.py",
            "--base-directory",
            str(COMFYUI_BASE_DIR),
            "--listen",
            "127.0.0.1",
            "--port",
            str(COMFYUI_PORT),
            *memory_flags,
            "--user-directory",
            str(COMFYUI_USER_DIR),
            "--database-url",
            database_url,
            "--output-directory",
            str(OUTPUT_DIR),
            "--input-directory",
            str(INPUT_DIR),
            "--disable-auto-launch",
        ]
        # Multi-GPU fix: comfy-aimdo (DynamicVRAM) crashes with
        # "hostbuf_read_file_slice: device copy failed" when more than one GPU is
        # initialized. Force a single device (default cuda:0) so DynamicVRAM works.
        # Override with MINIMAX_COMFY_CUDA_VISIBLE_DEVICES (e.g. "0,1" once aimdo
        # upstream fixes multi-GPU hostbuf).
        comfy_env = os.environ.copy()
        # Force UTF-8 for the child's stdio. stdout is redirected to the log
        # file above (see COMFYUI_LOG), so without this the child encodes it
        # with the system locale (cp950 here). A single emoji in a custom
        # node's print then raises UnicodeEncodeError, which cascades into a
        # logging error and can kill ComfyUI's prompt worker (comfyui.log
        # 2026-09-05: MinimaxH3 latent upscaler node).
        comfy_env["PYTHONIOENCODING"] = "utf-8"
        # Auto-detect GPU usage and pick the least utilized GPU
        # No default: an unset var must not crash startup (see 2026-09-05
        # "NoneType has no attribute strip" -- env var unset, .strip() on None).
        visible = (
            os.environ.get("MINIMAX_COMFY_CUDA_VISIBLE_DEVICES") or ""
        ).strip()
        if visible:
            comfy_env["CUDA_VISIBLE_DEVICES"] = visible
        else:
            # Pick a GPU for ComfyUI. Preference order:
            #   1. a card that is NOT driving a display (never put a 20GB video
            #      model on the user's desktop card while a free card exists),
            #   2. most free VRAM.
            # The old "most free" rule could still land on the display card when
            # the freed VRAM happened to be a few MiB higher there, which is
            # exactly what this avoids. A non-display card is only skipped when
            # it has less than COMFY_GPU_MIN_FREE_MB free (e.g. while the local
            # LLM is holding it), in which case the freest card wins.
            min_free = int(os.environ.get("MINIMAX_COMFY_GPU_MIN_FREE_MB", "6000"))
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total,display_attached",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                rows: list[tuple[int, int, bool]] = []
                for i, line in enumerate(result.stdout.strip().split("\n")):
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) < 2:
                        continue
                    try:
                        used, total = int(parts[0]), int(parts[1])
                    except ValueError:
                        continue
                    attached = len(parts) > 2 and parts[2].lower() in {"yes", "enabled"}
                    rows.append((i, total - used, attached))
                if not rows:
                    raise RuntimeError("nvidia-smi returned no usable GPU rows")
                preferred = [
                    row for row in rows if not row[2] and row[1] >= min_free
                ]
                best_gpu, best_free, best_attached = max(
                    preferred or rows, key=lambda row: row[1]
                )
                comfy_env["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
                note = "display card, no free card above the floor" if best_attached else "non-display card"
                bot_log(
                    f"Auto-selected GPU {best_gpu} ({best_free} MiB free, {note})"
                )
            except Exception as e:
                bot_log(f"GPU auto-detection failed: {e}, using GPU 0")
                comfy_env["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = None
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            with COMFYUI_LOG.open("ab") as log_file:
                _comfy_process = subprocess.Popen(
                    command,
                    cwd=str(COMFYUI_DIR),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=comfy_env,
                    startupinfo=startupinfo,
                    creationflags=creationflags,
                )
        except OSError as exc:
            raise BotError(f"啟動 ComfyUI 失敗：{exc}") from exc

    return (
        f"已啟動 ComfyUI，正在載入中（PID {_comfy_process.pid}）。\n"
        f"日誌：{COMFYUI_LOG}"
    )


def _running_comfy_process_ids() -> set[int]:
    """Find only the configured ComfyUI server processes on Windows."""
    pids: set[int] = set()
    if _comfy_process is not None and _comfy_process.poll() is None:
        pids.add(int(_comfy_process.pid))
    if os.name != "nt":
        return pids

    query = (
        "$items = Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -and "
        "$_.CommandLine -match 'main\\.py' -and "
        f"$_.CommandLine -match '--port\\s+{COMFYUI_PORT}(\\s|$)' "
        "}; $items | Select-Object -ExpandProperty ProcessId"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", query],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return pids
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid > 0 and pid != os.getpid():
            pids.add(pid)
    return pids


def stop_comfyui_process() -> str:
    """Interrupt and stop the configured local ComfyUI server."""
    global _comfy_process
    try:
        comfy_post("/interrupt", {})
    except BotError:
        pass

    with _comfy_start_lock:
        pids = _running_comfy_process_ids()
        if not pids:
            _comfy_process = None
            return "ComfyUI 目前已關閉。"

        failures: list[str] = []
        for pid in sorted(pids):
            try:
                if os.name == "nt":
                    result = subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=20,
                        check=False,
                    )
                    if result.returncode != 0 and "not found" not in (
                        result.stdout + result.stderr
                    ).lower():
                        failures.append(f"PID {pid}")
                else:
                    os.kill(pid, 15)
            except (OSError, subprocess.TimeoutExpired):
                failures.append(f"PID {pid}")
        _comfy_process = None

    deadline = time.time() + 20
    while time.time() < deadline and comfyui_is_online():
        time.sleep(0.25)
    if comfyui_is_online():
        return "已發出關閉 ComfyUI 的要求，但 8191 埠仍在回應。"
    if failures:
        return f"ComfyUI 關閉不完整，請檢查：{', '.join(failures)}"
    return "ComfyUI 已關閉。"


def restart_comfyui_process(vram_mode: Optional[str] = None) -> str:
    """Stop and start the configured local Turbo ComfyUI server."""
    stop_message = stop_comfyui_process()
    start_message = start_comfyui_process(vram_mode)
    return f"{stop_message}\n{start_message}"


def _llama_preset_value(flag: str) -> str:
    """Return the value that follows `flag` in the preset argument list."""
    try:
        return str(LLAMA_PRESET_ARGS[LLAMA_PRESET_ARGS.index(flag) + 1])
    except (ValueError, IndexError):
        return ""


def llama_is_online() -> bool:
    """Return whether the local llama-server /health endpoint answers.

    A server that is starting up or shutting down can reset the socket
    mid-request; those surface as raw OSError subclasses (e.g.
    ConnectionResetError) rather than the BotError that json_request usually
    wraps, so treat any network-level failure as "not online".
    """
    try:
        json_request(f"{LLAMA_URL}/health", timeout=4)
        return True
    except (BotError, OSError):
        return False


def llama_server_responding() -> bool:
    """Whether the server process answers at all, even while loading weights.

    /health returns 503 with `{"error":{"message":"Loading model"}}` while the
    ~22GB of weights are being read, which takes about a minute. During that
    window llama_is_online() is False, so a request arriving right after a
    generation used to be reported as "the LLM is not running" and then
    "starting it" - both wrong and confusing, since the Bot had just started it
    itself. This separates "still loading" from "not started".
    """
    request = Request(f"{LLAMA_URL}/health", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=4):
            return True
    except HTTPError:
        # Any HTTP status means something is listening and serving.
        return True
    except (URLError, TimeoutError, OSError):
        return False


# --- guard: never start the LLM on top of a VRAM-hungry ComfyUI -------------
# ComfyUI keeps its models resident after a run, so on this 2 x 20GB box the
# ~35GB llama-server cannot fit alongside them. Starting it anyway does not
# produce a working LLM, it produces an OOM, so the start is refused with an
# explanation instead. Set MINIMAX_LLM_START_GUARD=0 to disable, or tune the
# required headroom with MINIMAX_LLM_START_MIN_FREE_MB.
LLM_START_GUARD = os.environ.get(
    "MINIMAX_LLM_START_GUARD", "1"
).strip().lower() not in {"0", "false", "off", "no"}
LLM_START_MIN_FREE_MB = int(
    os.environ.get("MINIMAX_LLM_START_MIN_FREE_MB", "34000")
)


def gpu_vram_snapshot() -> Optional[list[tuple[int, int]]]:
    """Per-GPU (used_mb, total_mb) from nvidia-smi, or None when unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    snapshot: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        if "," not in line:
            continue
        try:
            used, total = (int(part.strip()) for part in line.split(",", 1))
        except ValueError:
            continue
        snapshot.append((used, total))
    return snapshot or None


def llm_start_blocker() -> str:
    """Explain why starting the LLM now would OOM, or return "" when it fits.

    ComfyUI reports only the single device the Bot exposed to it, so the check
    is based on the whole-machine picture from nvidia-smi rather than on
    ComfyUI's own /system_stats.
    """
    if not LLM_START_GUARD:
        return ""
    if not comfyui_is_online():
        return ""
    snapshot = gpu_vram_snapshot()
    if snapshot is None:
        return ""  # Cannot tell; do not block on a missing probe.
    total_vram = sum(total for _, total in snapshot)
    free = sum(total - used for used, total in snapshot)
    # Never demand more than the machine physically has, so a bad config value
    # cannot make the LLM permanently unstartable.
    required = min(LLM_START_MIN_FREE_MB, int(total_vram * 0.85))
    if free >= required:
        return ""
    per_gpu = "、".join(f"GPU{i} 剩 {total - used:,}MB" for i, (used, total) in enumerate(snapshot))
    return (
        f"ComfyUI 正在佔用顯存，現在啟動本機 LLM 會直接 OOM。\n\n"
        f"可用顯存：{free:,} MB（共 {total_vram:,} MB）\n"
        f"啟動 LLM 約需：{required:,} MB\n"
        f"{per_gpu}\n\n"
        "ComfyUI 跑完不會自動釋放模型，請先擇一：\n"
        "  • 按面板的「🛑 關閉 ComfyUI」釋放顯存，再重試\n"
        "  • 或先在 ComfyUI 裡卸載模型（Free model and node cache）\n\n"
        "（可用 MINIMAX_LLM_START_GUARD=0 停用這道檢查，但不建議。）"
    )


def _running_llama_process_ids() -> set[int]:
    """Find only the configured local llama-server processes on Windows."""
    pids: set[int] = set()
    if _llama_process is not None and _llama_process.poll() is None:
        pids.add(int(_llama_process.pid))
    if os.name != "nt":
        return pids

    query = (
        "$items = Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -and "
        "$_.CommandLine -match 'llama-server\\.exe' -and "
        f"$_.CommandLine -match '--port\\s+{LLAMA_PORT}(\\s|$)' "
        "}; $items | Select-Object -ExpandProperty ProcessId"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", query],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return pids
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid > 0 and pid != os.getpid():
            pids.add(pid)
    return pids


def _llama_gpu_line() -> str:
    """Best-effort single GPU usage line (name, temp, util, VRAM) via nvidia-smi."""
    smi = Path(NVIDIA_SMI_PATH)
    if not (smi.is_file() or shutil.which(NVIDIA_SMI_PATH)):
        return "GPU：找不到 nvidia-smi"
    try:
        result = run_hidden_command(
            [
                NVIDIA_SMI_PATH,
                "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=8,
        )
        if result.returncode == 0:
            rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            for row in rows:
                fields = [field.strip() for field in row.split(",")]
                if len(fields) >= 5:
                    name, temp, util, used, total = fields[:5]
                    return f"GPU：{name}｜{temp}°C｜{util}%｜VRAM {used}/{total} MiB"
            return "GPU：nvidia-smi 無資料"
        detail = (result.stderr or result.stdout).strip().splitlines()
        return f"GPU：讀取失敗（{detail[-1][:160] if detail else 'nvidia-smi error'}）"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"GPU：讀取失敗（{exc}）"


def start_llama_process() -> str:
    """Start the local llama-server with the preset parameters when offline."""
    global _llama_process
    if llama_is_online():
        return f"本地 LLM 已經在運行（{LLAMA_URL}）。"
    if not LLAMA_EXE.is_file():
        raise BotError(f"找不到 llama-server：{LLAMA_EXE}")
    if not LLAMA_MODEL.is_file():
        raise BotError(f"找不到模型檔案：{LLAMA_MODEL}")
    if not LLAMA_MMPROJ.is_file():
        raise BotError(f"找不到 mmproj 檔案：{LLAMA_MMPROJ}")
    if not LLAMA_DRAFT_MODEL.is_file():
        raise BotError(f"找不到 MTP 草稿模型：{LLAMA_DRAFT_MODEL}")

    # VRAM guard, placed here so EVERY entry point is covered in one place:
    # the panel buttons, /llm_start, /llm_restart, restart_llama_process() and
    # the script generator. restart_llm_after_job() stops ComfyUI before calling
    # this, so it is unaffected. Checked outside the start lock so a refusal
    # cannot be mistaken for "already starting".
    blocker = llm_start_blocker()
    if blocker:
        raise BotError(blocker)

    with _llama_start_lock:
        if llama_is_online():
            return f"本地 LLM 已經在運行（{LLAMA_URL}）。"
        if _llama_process is not None and _llama_process.poll() is None:
            return f"本地 LLM 正在啟動中（PID {_llama_process.pid}）。"
        LLAMA_LOG.parent.mkdir(parents=True, exist_ok=True)
        command = [str(LLAMA_EXE), *LLAMA_PRESET_ARGS]
        try:
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = None
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            with LLAMA_LOG.open("ab") as log_file:
                _llama_process = subprocess.Popen(
                    command,
                    cwd=str(LLAMA_EXE.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    startupinfo=startupinfo,
                    creationflags=creationflags,
                )
        except OSError as exc:
            raise BotError(f"啟動 LLM 失敗：{exc}") from exc

    warning = ""
    if comfyui_is_online():
        warning = (
            "\n⚠️ 注意：ComfyUI 目前也在運行，兩者共用 GPU 顯存，"
            "可能出現顯存不足；若要生成影片請先關閉 LLM。"
        )
    return (
        f"已啟動本地 LLM，正在載入模型中（PID {_llama_process.pid}）。\n"
        f"位址：{LLAMA_URL}\n"
        f"日誌：{LLAMA_LOG}{warning}"
    )


def stop_llama_process() -> str:
    """Interrupt and stop the configured local llama-server."""
    global _llama_process
    with _llama_start_lock:
        pids = _running_llama_process_ids()
        if not pids:
            _llama_process = None
            return "本地 LLM 目前已關閉。"

        failures: list[str] = []
        for pid in sorted(pids):
            try:
                if os.name == "nt":
                    result = subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=20,
                        check=False,
                    )
                    if result.returncode != 0 and "not found" not in (
                        result.stdout + result.stderr
                    ).lower():
                        failures.append(f"PID {pid}")
                else:
                    os.kill(pid, 15)
            except (OSError, subprocess.TimeoutExpired):
                failures.append(f"PID {pid}")
        _llama_process = None

    deadline = time.time() + 20
    while time.time() < deadline and llama_is_online():
        time.sleep(0.25)
    if llama_is_online():
        return f"已發出關閉 LLM 的要求，但 {LLAMA_PORT} 埠仍在回應。"
    if failures:
        return f"本地 LLM 關閉不完整，請檢查：{', '.join(failures)}"
    return "本地 LLM 已關閉。"


def restart_llama_process() -> str:
    """Stop and start the configured local llama-server."""
    stop_message = stop_llama_process()
    start_message = start_llama_process()
    return f"{stop_message}\n{start_message}"


def restart_bot_process() -> None:
    """Schedule a detached helper that stops this Bot and relaunches it hidden.

    The current process keeps polling for a few more seconds so the "正在重啟"
    confirmation can be delivered first; the helper then force-stops every
    process whose command line references this Bot script and starts a fresh
    copy through the same VBS launcher used by the Start/Restart .cmd files.
    """
    vbs_path = Path(__file__).resolve().parent / "Start-MiniMax-H3-Telegram.vbs"
    if not vbs_path.is_file():
        raise BotError(f"找不到 Bot 啟動器：{vbs_path}")
    pattern = "MiniMax-H3-Telegram-Bot.py"
    vbs_arg = str(vbs_path).replace("'", "''")
    script = (
        "Start-Sleep -Seconds 4; "
        "$self=$PID; "
        f"$targets=@(Get-CimInstance Win32_Process | Where-Object {{ "
        f"$_.ProcessId -ne $self -and $_.CommandLine -like '*{pattern}*' }}); "
        "if ($targets.Count -gt 0) { "
        "foreach ($p in $targets) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } "
        "Start-Sleep -Seconds 1 "
        "}; "
        f"wscript.exe '//nologo' '{vbs_arg}'"
    )
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-WindowStyle",
        "Hidden",
        "-Command",
        script,
    ]
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    subprocess.Popen(command, **kwargs)


# --- tail-frame reference handoff -------------------------------------------
# After a generation finishes, the last two frames are sent back to Telegram and
# can be promoted to the new reference images with one tap. That is how a story
# is chained: the tail of one segment becomes the look-lock for the next one.
TAIL_REF_OFFER_ENABLED = os.environ.get(
    "MINIMAX_TAIL_REF_OFFER", "1"
).strip().lower() not in {"0", "false", "off", "no"}
TAIL_FRAME_DIR = Path(
    os.environ.get("MINIMAX_TAIL_FRAME_DIR", str(STATE_PATH.parent / "tail_frames"))
)


def extract_tail_frames(
    video_path: Path, count: int = 2, spacing: float = 0.5
) -> tuple[str, list[Path]]:
    """Save the last `count` frames of a video as JPEGs.

    Frames are sampled `spacing` seconds apart ending on the video's final
    frame (the very last two frames are usually identical, which makes the
    second one useless as a reference image). Identical outputs are dropped.

    Returns (token, frames), frames ordered oldest-to-newest. The token is
    embedded in the Telegram callback data so the button handler can find the
    frames on disk later without keeping per-message state in memory.
    """
    if not video_path.is_file():
        return "", []
    try:
        duration, _, _ = probe_video_info(video_path)
    except Exception:
        return "", []
    if duration <= 0:
        return "", []

    token = uuid.uuid4().hex[:10]
    try:
        TAIL_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return "", []
    frames: list[Path] = []
    seen_hashes: set[str] = set()
    for index in range(count, 0, -1):
        # `-sseof` seeks from the end of the file; seeking to `duration` with
        # `-ss` lands past the last video frame (the audio track is usually a
        # bit longer than the video one) and yields nothing at all.
        offset = -(0.1 + (index - 1) * spacing)
        destination = TAIL_FRAME_DIR / f"tail_{token}_{count - index + 1}.jpg"
        command = [
            FFMPEG_PATH,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-sseof",
            f"{offset:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(destination),
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if not destination.is_file() or destination.stat().st_size == 0:
            continue
        digest = hashlib.md5(destination.read_bytes()).hexdigest()
        if digest in seen_hashes:
            destination.unlink(missing_ok=True)
            continue
        seen_hashes.add(digest)
        frames.append(destination)
    return token, frames


def tail_reference_frames(token: str) -> list[Path]:
    """Frames previously saved by extract_tail_frames for a callback token."""
    if not token:
        return []
    try:
        return sorted(TAIL_FRAME_DIR.glob(f"tail_{token}_*.jpg"))
    except OSError:
        return []


def promote_reference_frames(frames: list[Path], count: int = 2) -> list[Path]:
    """Replace the reference images with the given frames (tail-frame handoff).

    Deletes the old ref_image_* files and copies up to `count` frames in as
    ref_image_01.jpg / ref_image_02.jpg. Shared by the ✅ offer button and the
    🔗 auto-chain.
    """
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for old in REFERENCE_DIR.glob("ref_image_*"):
        if old.is_file():
            old.unlink(missing_ok=True)
    new_paths: list[Path] = []
    for index, frame in enumerate(frames[:count], start=1):
        destination = REFERENCE_DIR / f"ref_image_{index:02d}.jpg"
        shutil.copyfile(frame, destination)
        new_paths.append(destination)
    return new_paths


def chain_plan_durations(total: float, cap: float = 0.0) -> list[float]:
    """Split a total duration into 2-15s clips using the varied rhythm.

    Reuses the script generator's scene rhythm (the same irregular lengths the
    timeline headings use) so an auto-chain feels hand-cut rather than
    metronomic. `cap` (2-15s) hard-limits any single clip.
    """
    total = float(total)
    plan = script_timeline_skeleton(total, SCRIPT_LANG_ZH, varied=True)
    lengths = [round(end - start, 2) for _label, start, end in plan]
    if cap and cap < float(MAX_SEGMENT_SECONDS):
        split: list[float] = []
        for length in lengths:
            while length > cap + 0.01:
                half = round(length / 2, 1)
                if half < 2.0 or round(length - half, 2) < 2.0:
                    break
                split.append(half)
                length = round(length - half, 2)
            split.append(length)
        lengths = split
    return [length for length in lengths if length > 0]


def chain_merge_clips(paths: list[Path], output_path: Path) -> Path:
    """Join equal-format clips with the ffmpeg concat demuxer (stream copy).

    The 🔗 auto-chain produces short clips from the same pipeline (same codec,
    size and rate), so a stream copy is lossless and fast - no crossfade pass.
    """
    if len(paths) < 2:
        raise BotError("合併至少需要兩段影片。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_path.with_suffix(".concat.txt")
    try:
        list_file.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in paths),
            encoding="utf-8",
        )
        command = [
            FFMPEG_PATH,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BotError(f"合併失敗：{exc}") from exc
    finally:
        list_file.unlink(missing_ok=True)
    if result.returncode != 0 or not output_path.is_file():
        detail = (result.stderr or "").strip()
        raise BotError("合併失敗：" + (detail[-300:] or "ffmpeg 沒有產出檔案"))
    return output_path


# Injected into the writer's system prompt once the tail frames of a finished
# clip have been accepted as the new reference images: the next segment must
# continue that story instead of drifting into a new person or a new place.
TAIL_CONTINUITY_NOTE = """【接續規則 — 這段是上一段影片的直接延續，不是新故事】
- 參考圖是上一段影片的最後畫面：人物（臉、髮型、身材、膚色）和衣著／裸露狀態必須完全一致，不可以換人。
- 場景、環境、光線、時間、天氣都要跟上一段一致，不可以換地方、不可以加人。
- 開頭／第一幕要從上一段最後的姿勢與動作直接接下去；不可以重新開始、不可以重新介紹角色、不要 recap。
- 運鏡與視覺風格維持和上一段一致。
- 只寫接下來發生的事。"""


def multipart_request(
    url: str,
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    content_type: str = "video/mp4",
) -> Any:
    boundary = f"----MiniMaxH3Telegram{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("ascii")
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                b"--" + boundary_bytes + b"\r\n",
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            b"--" + boundary_bytes + b"\r\n",
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n--" + boundary_bytes + b"--\r\n",
        ]
    )
    request = Request(
        url,
        data=b"".join(chunks),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise BotError(f"傳送影片失敗：{http_error_detail(exc)}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise BotError(f"傳送影片失敗：{exc}") from exc
    if not result.get("ok"):
        raise BotError(result.get("description", "Telegram 傳送影片失敗。"))
    return result


def telegram_caption_with_note(caption: str, note: str) -> str:
    """Append a delivery note without exceeding Telegram's caption limit."""
    suffix = f"\n\n{note.strip()}" if note.strip() else ""
    text = str(caption or "").strip()
    combined = f"{text}{suffix}" if text else note.strip()
    if len(combined) <= 1024:
        return combined
    return combined[:1021].rstrip() + "..."


def telegram_video_target_bitrate(duration: float, factor: float = 1.0) -> int:
    """Return a conservative video bitrate in kbit/s for Telegram delivery."""
    safe_total_kbps = (
        TELEGRAM_SAFE_VIDEO_BYTES * 8 * 0.92 / max(float(duration), 1.0) / 1000.0
    )
    video_kbps = int(safe_total_kbps * float(factor)) - TELEGRAM_AUDIO_BITRATE_KBPS
    return max(256, video_kbps)


def _telegram_temp_video_path(source_path: Path, suffix: str) -> Path:
    return source_path.with_name(
        f".{source_path.stem}.telegram-{suffix}-{uuid.uuid4().hex[:8]}.mp4"
    )


def compress_video_for_telegram(video_path: Path) -> Optional[Path]:
    """Encode an oversized MP4 below Telegram's upload limit.

    The original generated file is never replaced.  The caller owns the
    returned temporary file and must remove it after sending.
    """
    duration, _, _ = probe_video_info(video_path)
    last_detail = ""
    for attempt, factor in enumerate((1.0, 0.84, 0.68, 0.54), start=1):
        output_path = _telegram_temp_video_path(video_path, f"compress{attempt}")
        video_kbps = telegram_video_target_bitrate(duration, factor)
        command = [
            FFMPEG_PATH,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            f"{video_kbps}k",
            "-maxrate",
            f"{video_kbps}k",
            "-bufsize",
            f"{video_kbps * 2}k",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            f"{TELEGRAM_AUDIO_BITRATE_KBPS}k",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            result = run_hidden_command(command, timeout=max(600.0, duration * 4.0))
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_detail = str(exc)
            result = None
        if result is not None and result.returncode == 0 and output_path.is_file():
            output_size = output_path.stat().st_size
            if output_size <= TELEGRAM_SAFE_VIDEO_BYTES:
                return output_path
            last_detail = f"attempt {attempt}: {output_size} bytes"
        elif result is not None:
            last_detail = (result.stderr or result.stdout or "ffmpeg failed").strip()
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
    bot_log(f"Telegram video compression did not fit: {video_path} ({last_detail[-500:]})")
    return None


def split_video_for_telegram(video_path: Path) -> list[Path]:
    """Split an oversized video into uploadable MP4 parts as a last resort."""
    duration, _, _ = probe_video_info(video_path)
    source_size = max(video_path.stat().st_size, 1)
    chunk_seconds = max(
        5.0,
        duration * TELEGRAM_SAFE_VIDEO_BYTES / source_size * 0.82,
    )
    parts: list[Path] = []
    start = 0.0
    part_number = 1
    try:
        while start < duration - 0.05:
            remaining = duration - start
            attempt_seconds = min(chunk_seconds, remaining)
            output_path: Optional[Path] = None
            for _ in range(7):
                candidate = _telegram_temp_video_path(video_path, f"part{part_number:03d}")
                command = [
                    FFMPEG_PATH,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start:.3f}",
                    "-i",
                    str(video_path),
                    "-t",
                    f"{attempt_seconds:.3f}",
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0?",
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-movflags",
                    "+faststart",
                    str(candidate),
                ]
                try:
                    result = run_hidden_command(
                        command,
                        timeout=max(180.0, attempt_seconds * 3.0),
                    )
                except (OSError, subprocess.TimeoutExpired):
                    result = None
                if result is not None and result.returncode == 0 and candidate.is_file():
                    if candidate.stat().st_size <= TELEGRAM_SAFE_VIDEO_BYTES:
                        output_path = candidate
                        break
                try:
                    candidate.unlink(missing_ok=True)
                except OSError:
                    pass
                attempt_seconds = max(2.0, attempt_seconds * 0.68)
            if output_path is None:
                raise BotError(f"無法把影片分段至 Telegram 上傳大小：第 {part_number} 段")
            parts.append(output_path)
            start += attempt_seconds
            part_number += 1
    except Exception:
        for part in parts:
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return parts


def build_transition_filter(
    shots: list[ShotSpec] | tuple[ShotSpec, ...],
    transition_seconds: float = SHOT_TRANSITION_SECONDS,
    output_size: Optional[tuple[int, int]] = None,
) -> tuple[str, str, str]:
    """Build a duration-preserving FFmpeg xfade/acrossfade graph."""
    if len(shots) < 2:
        raise BotError("轉場至少需要兩個鏡頭。")
    filters: list[str] = []
    for index, shot in enumerate(shots):
        trim_duration = shot.duration
        if index < len(shots) - 1:
            trim_duration += transition_seconds
        video_normalization = ""
        if output_size is not None:
            output_width, output_height = output_size
            video_normalization = (
                f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,"
                f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                "setsar=1,"
            )
        filters.append(
            f"[{index}:v]trim=duration={trim_duration:.3f},"
            f"setpts=PTS-STARTPTS,{video_normalization}fps=24,format=yuv420p[v{index}]"
        )
        filters.append(
            f"[{index}:a]atrim=duration={trim_duration:.3f},"
            f"asetpts=PTS-STARTPTS,aresample=48000,highshelf=f=5000:g=2[a{index}]"
        )

    current_video = "v0"
    current_audio = "a0"
    offset = shots[0].duration
    for index in range(1, len(shots)):
        next_video = f"vx{index}"
        next_audio = f"ax{index}"
        filters.append(
            f"[{current_video}][v{index}]xfade=transition=fade:"
            f"duration={transition_seconds:.3f}:offset={offset:.3f}[{next_video}]"
        )
        filters.append(
            f"[{current_audio}][a{index}]acrossfade=d={transition_seconds:.3f}:"
            f"c1=tri:c2=tri[{next_audio}]"
        )
        current_video = next_video
        current_audio = next_audio
        offset += shots[index].duration
    return ";".join(filters), f"[{current_video}]", f"[{current_audio}]"


def build_normalized_concat_filter(
    input_count: int,
    output_size: tuple[int, int],
) -> tuple[str, str, str]:
    """Build a re-encode fallback that also handles mixed segment resolutions."""
    output_width, output_height = output_size
    filters: list[str] = []
    concat_inputs: list[str] = []
    for index in range(input_count):
        filters.append(
            f"[{index}:v]scale={output_width}:{output_height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps=24,format=yuv420p[vn{index}]"
        )
        filters.append(f"[{index}:a]aresample=48000,highshelf=f=5000:g=2[an{index}]")
        concat_inputs.extend([f"[vn{index}]", f"[an{index}]"])
    filters.append(
        "".join(concat_inputs)
        + f"concat=n={input_count}:v=1:a=1[vout][aout]"
    )
    return ";".join(filters), "[vout]", "[aout]"


def concat_videos(
    video_paths: list[Path],
    output_path: Path,
    total_seconds: float,
    shot_plan: Optional[tuple[ShotSpec, ...]] = None,
    output_size: Optional[tuple[int, int]] = None,
) -> Path:
    """Join generated shots, preferring short audio/video crossfades."""
    if len(video_paths) < 2:
        raise BotError("長片至少需要兩段影片才能合併。")
    if shutil.which(FFMPEG_PATH) is None and not Path(FFMPEG_PATH).is_file():
        raise BotError(
            f"找不到 FFmpeg：{FFMPEG_PATH}。請安裝 FFmpeg，或設定 MINIMAX_FFMPEG。"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shot_plan and len(shot_plan) == len(video_paths):
        filter_graph, video_output, audio_output = build_transition_filter(
            shot_plan,
            output_size=output_size,
        )
        transition_command = [FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-y"]
        for video_path in video_paths:
            transition_command.extend(["-i", str(video_path)])
        transition_command.extend(
            [
                "-filter_complex",
                filter_graph,
                "-map",
                video_output,
                "-map",
                audio_output,
                "-t",
                f"{total_seconds:.3f}",
                "-r",
                "24",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        try:
            transition_result = subprocess.run(
                transition_command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"transition merge unavailable, using stream-copy fallback: {exc}", flush=True)
        else:
            if transition_result.returncode == 0 and output_path.is_file():
                return output_path
            detail = (transition_result.stderr or "").strip()
            print(
                "transition merge failed, using stream-copy fallback: "
                + detail[-800:],
                flush=True,
            )

            if output_size is not None:
                fallback_graph, fallback_video, fallback_audio = (
                    build_normalized_concat_filter(len(video_paths), output_size)
                )
                fallback_command = [FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-y"]
                for video_path in video_paths:
                    fallback_command.extend(["-i", str(video_path)])
                fallback_command.extend(
                    [
                        "-filter_complex",
                        fallback_graph,
                        "-map",
                        fallback_video,
                        "-map",
                        fallback_audio,
                        "-t",
                        f"{total_seconds:.3f}",
                        "-r",
                        "24",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-movflags",
                        "+faststart",
                        str(output_path),
                    ]
                )
                try:
                    fallback_result = subprocess.run(
                        fallback_command,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=1800,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    fallback_result = None
                    print(f"normalized concat fallback unavailable: {exc}", flush=True)
                if fallback_result is not None:
                    if fallback_result.returncode == 0 and output_path.is_file():
                        return output_path
                    fallback_detail = (fallback_result.stderr or "").strip()
                    print(
                        "normalized concat fallback failed: "
                        + fallback_detail[-800:],
                        flush=True,
                    )

    list_path = output_path.with_suffix(".concat.txt")
    lines = []
    for video_path in video_paths:
        escaped = str(video_path.resolve()).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    command = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-t",
        f"{total_seconds:.3f}",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BotError(f"合併長片時 FFmpeg 失敗：{exc}") from exc
    finally:
        try:
            list_path.unlink()
        except OSError:
            pass

    if result.returncode != 0 or not output_path.is_file():
        details = (result.stderr or "").strip()
        raise BotError(f"合併長片失敗：{details[-800:]}")
    return output_path


def trim_single_video(
    video_path: Path,
    output_path: Path,
    duration_seconds: float,
) -> Path:
    """Trim one completed shot for an early-cancel partial result."""
    if not video_path.is_file():
        raise BotError(f"找不到已完成影片：{video_path}")
    if duration_seconds <= 0:
        raise BotError("部分合成的影片長度必須大於 0 秒。")
    if shutil.which(FFMPEG_PATH) is None and not Path(FFMPEG_PATH).is_file():
        raise BotError(
            f"找不到 FFmpeg：{FFMPEG_PATH}。請安裝 FFmpeg，或設定 MINIMAX_FFMPEG。"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BotError(f"單段部分合成失敗：{exc}") from exc
    if result.returncode != 0 or not output_path.is_file():
        detail = (result.stderr or "").strip()
        raise BotError(f"單段部分合成失敗：{detail[-800:]}")
    return output_path


def merge_completed_segments(
    video_paths: list[Path],
    output_path: Path,
    total_seconds: float,
    shot_plan: Optional[tuple[ShotSpec, ...]] = None,
    output_size: Optional[tuple[int, int]] = None,
) -> Path:
    """Merge all completed shots, including a one-shot early cancellation."""
    if not video_paths:
        raise BotError("沒有已完成的分段可以合成。")
    if len(video_paths) == 1:
        return trim_single_video(video_paths[0], output_path, total_seconds)
    return concat_videos(
        video_paths,
        output_path,
        total_seconds,
        shot_plan=shot_plan,
        output_size=output_size,
    )


# --- In-Bot script generator (local llama.cpp) -----------------------------
# Turns a one-line idea plus a duration into a complete, format-legal H3 script
# without leaving Telegram. The reply is validated with the SAME parser that
# gates generation (build_long_video_plan), so a script the generator accepts is
# one the Bot will actually run; when the model breaks a rule, the exact parser
# error is fed back and it is asked to repair itself.
SCRIPT_GEN_ENABLED = os.environ.get(
    "MINIMAX_SCRIPT_GEN", "1"
).strip().lower() not in {"0", "false", "off", "no"}
SCRIPT_GEN_TIMEOUT = float(os.environ.get("MINIMAX_SCRIPT_GEN_TIMEOUT", "600"))
SCRIPT_GEN_ATTEMPTS = int(os.environ.get("MINIMAX_SCRIPT_GEN_ATTEMPTS", "3"))
SCRIPT_GEN_TEMPERATURE = float(os.environ.get("MINIMAX_SCRIPT_GEN_TEMP", "0.85"))
# Reasoning and the answer share the completion budget. 6000 leaves comfortable
# room for the observed ~500-1500 thinking tokens plus a full script, while
# keeping the model from over-thinking (a larger allowance measurably made it
# think longer for no better output). The retry budget is only used when the
# first attempt comes back empty.
SCRIPT_GEN_MAX_TOKENS = int(os.environ.get("MINIMAX_SCRIPT_GEN_MAX_TOKENS", "6000"))
SCRIPT_GEN_RETRY_TOKENS = int(
    os.environ.get("MINIMAX_SCRIPT_GEN_RETRY_TOKENS", "16000")
)
# Which LLM writes the scripts: the local llama.cpp server (default) or the
# Command Code cloud API (OpenAI-compatible chat completions). Runtime-togglable
# from Telegram (/scriptllm or the 🧠 button) and persisted in settings;
# MINIMAX_SCRIPT_LLM sets the boot default.
SCRIPT_LLM_LOCAL = "local"
SCRIPT_LLM_COMMANDCODE = "commandcode"
SCRIPT_LLM_PROVIDERS = (SCRIPT_LLM_LOCAL, SCRIPT_LLM_COMMANDCODE)
SCRIPT_LLM_LABEL = {
    SCRIPT_LLM_LOCAL: "本機",
    SCRIPT_LLM_COMMANDCODE: "雲端",
}
SCRIPT_LLM_DEFAULT = (
    os.environ.get("MINIMAX_SCRIPT_LLM", SCRIPT_LLM_LOCAL).strip().lower()
)
if SCRIPT_LLM_DEFAULT not in SCRIPT_LLM_PROVIDERS:
    SCRIPT_LLM_DEFAULT = SCRIPT_LLM_LOCAL
COMMANDCODE_BASE_URL = os.environ.get(
    "MINIMAX_COMMANDCODE_BASE_URL", "https://api.commandcode.ai/provider/v1"
).rstrip("/")
COMMANDCODE_MODEL = os.environ.get(
    "MINIMAX_COMMANDCODE_MODEL", "deepseek/deepseek-v4.1-flash"
)
# Cloudflare sits in front of the API and blocks unknown browser signatures
# (error 1010), so requests must carry a normal browser User-Agent.
COMMANDCODE_USER_AGENT = os.environ.get(
    "MINIMAX_COMMANDCODE_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
)
COMMANDCODE_API_KEY_ENV = "COMMANDCODE_API_KEY"
# The key normally comes from the environment; when it is absent there it is
# read from the Hermes .env that already holds it on this machine (override the
# path with MINIMAX_COMMANDCODE_ENV_FILE). The value is never logged.
_COMMANDCODE_ENV_FILE = Path(
    os.environ.get(
        "MINIMAX_COMMANDCODE_ENV_FILE",
        str(Path(os.environ.get("LOCALAPPDATA", "C:/")) / "hermes" / ".env"),
    )
)
_script_llm_active = SCRIPT_LLM_DEFAULT


def normalize_script_llm(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"cc", "command-code", "command_code", "cloud", "api"}:
        return SCRIPT_LLM_COMMANDCODE
    if text in SCRIPT_LLM_PROVIDERS:
        return text
    return SCRIPT_LLM_DEFAULT


def get_script_llm_provider() -> str:
    """The engine that will answer the next script-generation call."""
    return _script_llm_active


def set_script_llm_provider(value: str) -> str:
    global _script_llm_active
    _script_llm_active = normalize_script_llm(value)
    return _script_llm_active


def script_llm_display_name() -> str:
    """User-facing name of the active script engine."""
    if _script_llm_active == SCRIPT_LLM_COMMANDCODE:
        return f"Command Code（{COMMANDCODE_MODEL}）"
    return "本機 LLM"


def commandcode_api_key() -> str:
    """Resolve the Command Code API key (env first, then the Hermes .env)."""
    key = (os.environ.get(COMMANDCODE_API_KEY_ENV) or "").strip()
    if key:
        return key
    try:
        for line in _COMMANDCODE_ENV_FILE.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            line = line.strip()
            if line.startswith(COMMANDCODE_API_KEY_ENV + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


# Which script-guidance template the writer uses: the operator's adult custom
# file (default, never modified by this switch) or the built-in general template.
# Toggled from Telegram (/scripttemplate or the 📄 button); MINIMAX_SCRIPT_TEMPLATE
# sets the boot default.
SCRIPT_TEMPLATE_ADULT = "adult"
SCRIPT_TEMPLATE_GENERAL = "general"
SCRIPT_TEMPLATES = (SCRIPT_TEMPLATE_ADULT, SCRIPT_TEMPLATE_GENERAL)
SCRIPT_TEMPLATE_LABEL = {
    SCRIPT_TEMPLATE_ADULT: "成人版",
    SCRIPT_TEMPLATE_GENERAL: "一般版",
}
SCRIPT_TEMPLATE_DEFAULT = (
    os.environ.get("MINIMAX_SCRIPT_TEMPLATE", SCRIPT_TEMPLATE_ADULT).strip().lower()
)
if SCRIPT_TEMPLATE_DEFAULT not in SCRIPT_TEMPLATES:
    SCRIPT_TEMPLATE_DEFAULT = SCRIPT_TEMPLATE_ADULT
_script_template_active = SCRIPT_TEMPLATE_DEFAULT


def normalize_script_template(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"general", "normal", "sfw", "safe", "一般", "一般版", "非成人"}:
        return SCRIPT_TEMPLATE_GENERAL
    if text in {"adult", "nsfw", "成人", "成人版"}:
        return SCRIPT_TEMPLATE_ADULT
    return SCRIPT_TEMPLATE_DEFAULT


def get_script_template() -> str:
    """Which guidance template the writer will use next."""
    return _script_template_active


def set_script_template(value: str) -> str:
    global _script_template_active
    _script_template_active = normalize_script_template(value)
    return _script_template_active


def script_template_display_name() -> str:
    if _script_template_active == SCRIPT_TEMPLATE_GENERAL:
        return "一般版（非成人）"
    return "成人版"


# Output language for generated scripts. The H3 model was trained mostly on
# English, so English action/camera phrasing is the most stable; Simplified
# Chinese is fully supported and is the default here because it is what the user
# asked for. Switch at runtime with /lang or the 🌐 toggle.
SCRIPT_LANG_ZH = "zh"
SCRIPT_LANG_EN = "en"
SCRIPT_LANGS = (SCRIPT_LANG_ZH, SCRIPT_LANG_EN)
SCRIPT_LANG_DEFAULT = (
    os.environ.get("MINIMAX_SCRIPT_LANG", SCRIPT_LANG_ZH).strip().lower()
)
if SCRIPT_LANG_DEFAULT not in SCRIPT_LANGS:
    SCRIPT_LANG_DEFAULT = SCRIPT_LANG_ZH

SCRIPT_LANG_LABEL = {
    SCRIPT_LANG_ZH: "簡體中文",
    SCRIPT_LANG_EN: "English",
}
# Compact labels for the panel button row (full labels truncate on phones).
SCRIPT_LANG_BUTTON = {
    SCRIPT_LANG_ZH: "簡中",
    SCRIPT_LANG_EN: "English",
}

_SCRIPT_GEN_LANGUAGE_BLOCK = {
    SCRIPT_LANG_ZH: (
        "LANGUAGE - this applies to EVERYTHING you write:\n"
        "- Write the GLOBAL block and every scene's action, camera and sound\n"
        "  description in SIMPLIFIED CHINESE (简体中文). Use simplified characters\n"
        "  only. Never write Traditional Chinese characters.\n"
        "- Do not write English sentences. English is acceptable only for a proper\n"
        "  noun that has no common Chinese form.\n"
        "- Write camera movement as natural Chinese prose, e.g. 镜头缓慢推近 or\n"
        "  镜头横移跟随.\n"
        "- Write dialogue directly in Chinese inside quotes, e.g. 她说：“等我。”\n"
        "  The Bot locks Mandarin pronunciation automatically.\n"
        "- Keep the timeline headings exactly as given; they are already Chinese."
    ),
    SCRIPT_LANG_EN: (
        "LANGUAGE - this applies to EVERYTHING you write:\n"
        "- Write the GLOBAL block and every scene description in English.\n"
        "- Write camera movement as natural English prose, e.g. \"the camera slowly\n"
        "  pushes in\"."
    ),
}

_SCRIPT_GEN_EXAMPLES = {
    SCRIPT_LANG_ZH: {
        "vague": "跳舞",
        "concrete": "她举起左手转了半圈，裙摆扬起",
        "camera": "镜头缓慢推近",
        "short_length": "150 到 260 個中文字",
    },
    SCRIPT_LANG_EN: {
        "vague": "dances",
        "concrete": "she raises her left hand and turns half a circle, her skirt lifting",
        "camera": "the camera slowly pushes in",
        "short_length": "90 to 140 words",
    },
}

# Timeline heading labels by language. Only 开/開 and 结/結 differ; 第一幕 is
# identical in both scripts. The Bot's parser accepts either form.
_SCRIPT_GEN_HEAD_LABELS = {
    SCRIPT_LANG_ZH: ("开头", "结尾"),
    SCRIPT_LANG_EN: ("開頭", "結尾"),
}

# Operator-editable extra instructions. This file is re-read on EVERY generation,
# so editing it takes effect immediately with no Bot restart. It is appended to
# the system prompt after the built-in templates (see build_script_messages), so
# the operator's wording can override the general style guidance without ever
# touching the machine-parsed STRUCTURE / heading rules.
SCRIPT_GEN_PROMPT_FILE = Path(
    os.environ.get(
        "MINIMAX_SCRIPT_PROMPT_FILE",
        r"E:\MiniMax-H3-Telegram\runtime\bot\script_prompt.txt",
    )
)
# Generous but bounded: custom guidance is meant to be a few sentences, and an
# accidentally huge file must not crowd out the script itself.
SCRIPT_GEN_PROMPT_MAX_CHARS = int(
    os.environ.get("MINIMAX_SCRIPT_PROMPT_MAX_CHARS", "4000")
)
# Optional operator file for 一般版. When absent or empty, the built-in general
# template (_SCRIPT_GEN_GENERAL_TEMPLATE) is used instead.
SCRIPT_GEN_PROMPT_FILE_GENERAL = Path(
    os.environ.get(
        "MINIMAX_SCRIPT_PROMPT_FILE_GENERAL",
        str(SCRIPT_GEN_PROMPT_FILE.with_name("script_prompt_general.txt")),
    )
)

SCRIPT_GEN_PROMPT_TEMPLATE = """\
# ============================================================================
# 自訂指令 — 在這裡加任何你想套用到每次生成的規則。
# 存檔後「立即生效」，不需要重啟 Bot。
#
# 規則：
#   • 以 # 開頭的行是註解，不會送給模型。
#   • 其餘文字會原樣附加到系統提示的後段。
#   • 適合寫「風格偏好」；結構與時間軸由 Bot 控制，這裡改不動（也不該改）。
#
# 建議一次只加 1-3 條，方便判斷哪一條造成變化。
# ============================================================================

# --- 範例（把前面的 # 拿掉就會生效）-----------------------------------------

# 運鏡一律緩慢，不要快速甩鏡或手持晃動。
# 每個場景至少要有一個明確的燈光來源或光線變化。
# 人物服裝與髮型在全片保持一致，不要中途改變。
# 不要出現文字、字幕、商標或浮水印。

# --- 你的規則寫在下面這條線之後 ---------------------------------------------
"""


def load_custom_script_instructions(path: Optional[Path] = None) -> tuple[str, str]:
    """Read the operator's custom instruction file.

    Returns (text, status) where status is a short human-readable note. The file
    is read on every call so edits apply without restarting the Bot.

    Lines starting with `#` are treated as comments and dropped, which lets the
    shipped template document itself without those notes reaching the model.
    """
    path = path if path is not None else SCRIPT_GEN_PROMPT_FILE
    try:
        # utf-8-sig also tolerates the BOM that Windows editors add silently.
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return "", "檔案不存在"
    except OSError as exc:
        bot_log(f"script prompt file unreadable: {exc}")
        return "", f"讀取失敗：{exc}"
    except UnicodeDecodeError as exc:
        bot_log(f"script prompt file not UTF-8: {exc}")
        return "", "不是 UTF-8 編碼，請另存為 UTF-8"

    kept: list[str] = []
    for line in raw.splitlines():
        if line.lstrip().startswith("#"):
            continue
        kept.append(line)
    text = "\n".join(kept).strip()

    if not text:
        return "", "沒有內容（全部都是註解或空白）"
    if len(text) > SCRIPT_GEN_PROMPT_MAX_CHARS:
        bot_log(
            f"script prompt file truncated "
            f"({len(text)} > {SCRIPT_GEN_PROMPT_MAX_CHARS} chars)"
        )
        text = (
            text[: SCRIPT_GEN_PROMPT_MAX_CHARS].rstrip()
            + "\n[自訂指令過長，已截斷]"
        )
        return text, f"過長已截斷（上限 {SCRIPT_GEN_PROMPT_MAX_CHARS} 字元）"
    return text, f"已載入 {len(text)} 字元"


def ensure_script_prompt_file() -> str:
    """Create the template file on first use so there is something to edit."""
    path = SCRIPT_GEN_PROMPT_FILE
    if path.is_file():
        return ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SCRIPT_GEN_PROMPT_TEMPLATE, encoding="utf-8")
    except OSError as exc:
        return f"無法建立自訂指令檔：{exc}"
    return f"已建立預設自訂指令檔：{path}"


# Keep one previous revision so a mistaken edit made from Telegram can be undone
# without the user having to remember what the text used to be.
SCRIPT_GEN_PROMPT_BACKUP = SCRIPT_GEN_PROMPT_FILE.with_suffix(".prev.txt")


def _custom_prompt_header(note: str) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return (
        "# MiniMax H3 自訂指令\n"
        f"# {note}（{stamp}）\n"
        "# 以 # 開頭的行是註解，不會送給模型。\n"
        "\n"
    )


def save_custom_script_instructions(text: str, note: str) -> tuple[bool, str]:
    """Write the custom instruction file, keeping the previous revision.

    The file is what the generator reads on every request, so this takes effect
    on the very next generation with no restart. Comments are written back as a
    short header purely so the file stays self-explanatory when opened in a text
    editor later.
    """
    path = SCRIPT_GEN_PROMPT_FILE
    body = (text or "").strip()
    if len(body) > SCRIPT_GEN_PROMPT_MAX_CHARS:
        body = (
            body[: SCRIPT_GEN_PROMPT_MAX_CHARS].rstrip() + "\n[自訂指令過長，已截斷]"
        )
    payload = _custom_prompt_header(note)
    if body:
        payload += body + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            current = path.read_text(encoding="utf-8", errors="replace")
            SCRIPT_GEN_PROMPT_BACKUP.write_text(current, encoding="utf-8")
        path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        bot_log(f"custom prompt save failed: {exc}")
        return False, f"寫入失敗：{exc}"
    return True, f"已儲存 {len(body)} 字元" if body else "已清空"


def restore_custom_script_instructions() -> tuple[bool, str]:
    """Roll the custom instruction file back to the previous revision."""
    path = SCRIPT_GEN_PROMPT_FILE
    if not SCRIPT_GEN_PROMPT_BACKUP.is_file():
        return False, "沒有可還原的上一版。"
    try:
        previous = SCRIPT_GEN_PROMPT_BACKUP.read_text(encoding="utf-8")
        # Swap so the restore is itself undoable.
        if path.is_file():
            current = path.read_text(encoding="utf-8", errors="replace")
            SCRIPT_GEN_PROMPT_BACKUP.write_text(current, encoding="utf-8")
        path.write_text(previous, encoding="utf-8")
    except OSError as exc:
        bot_log(f"custom prompt restore failed: {exc}")
        return False, f"還原失敗：{exc}"
    text, _ = load_custom_script_instructions()
    return True, f"已還原上一版（{len(text)} 字元）"


def normalize_script_lang(value: Any) -> str:
    """Clamp a language request to a supported code."""
    candidate = str(value or "").strip().lower()
    if candidate in {"zh", "cn", "zh-cn", "chs", "简体", "簡體", "中文"}:
        return SCRIPT_LANG_ZH
    if candidate in {"en", "eng", "english", "英文"}:
        return SCRIPT_LANG_EN
    return SCRIPT_LANG_DEFAULT
# The hard ceiling the Bot enforces on one scene; the writer must respect it or
# the plan builder refuses the script.
SCRIPT_GEN_MAX_SCENE = 15.0

_SCRIPT_GEN_LONG_SYSTEM = """You write prompts for the MiniMax H3 video model.

Write ONE script. Output the script itself only - no commentary, no markdown
code fences, no headings other than the timeline headings described below.

{language}

STRUCTURE (strict, this is machine-parsed):
1. Everything BEFORE the first timeline heading is the GLOBAL block. State there,
   once: the main character's visible appearance and clothing, the setting and its
   lighting, the overall visual style, and the audio of the whole film - state
   explicitly that there is NO music of any kind: no melody, no singing, no
   humming, only her voice and natural sounds. Never repeat the character
   description later.
2. After the GLOBAL block, write EXACTLY these headings, in this order, copying
   every number character for character. Do not add, remove, rename, reorder or
   re-time them. Put the scene text on the line after each heading:

{skeleton}

   Do NOT invent your own timings. These ranges are already correct and
   contiguous; your only job is to write the scene text under each one.
3. Put one blank line between scenes.

SCENE TEXT RULES:
- Chronological, concrete, visibly observable actions. Never write a vague verb
  like "{vague}"; write "{concrete}".
- Include the camera as natural prose, e.g. "{camera}".
- End every scene with one short sound sentence covering ambience, physical
  sounds, her voice or dialogue - never music, melody, singing or humming.
  Picture and audio are generated jointly, so a scene with no sound description
  will get random audio.
- No music and no singing anywhere in the film: never write song lyrics, never
  describe singing, humming or a soundtrack.
- Describe ONLY what happens inside that scene. Never describe future events.
- Never write media tags such as <Picture 1>, <Video 1> or <Audio 1>.
"""

_SCRIPT_GEN_SHORT_SYSTEM = """You write prompts for the MiniMax H3 video model.

The requested clip is short, so write ONE single flowing paragraph of
{short_length}. No headings, no timeline, no markdown, no commentary.

{language}

Cover, in this order: the subject's visible appearance and clothing, the setting
and lighting, the action unfolding over the clip, the camera movement written as
natural prose ("{camera}"), and finally the sound - ambience, physical sounds,
her voice or dialogue, and never music, melody, singing or humming. Picture and
audio are generated together, so the sound must be described or it will be
random.

Be concrete and visibly observable. Never write a vague verb like "{vague}";
write "{concrete}". Never write media tags such as <Picture 1> or <Video 1>.
"""

# Built-in general-content guidance, used when the 📄 switch is on 一般版 (and no
# script_prompt_general.txt overrides it). The adult custom file is never
# modified - switching simply stops feeding it to the model.
_SCRIPT_GEN_GENERAL_TEMPLATE = """【一般內容模式（非成人）】
【身份與輸出】
你是 MiniMax H3 影片提示詞作者。只輸出腳本本身：不要說明、不要客套話、不要 markdown。
時間軸標題一律照抄 Bot 給的原文，不要改成「場景N」、不要自己改秒數（時間已保證連續）。
短片（15 秒以內）＝一段流暢散文，照下面規則寫，不套時間軸。
腳本正文一律用簡體中文書寫（面板切成 English 時改用英文）；對白用中文或日文。
想法中的每個元素（人物、動作、場景細節）都要逐字落實，不可以遺漏。

【內容範圍（一般向，非成人）】
- 題材由用戶想法決定：日常生活、旅遊、美食、運動、工藝、劇情小品、紀錄、廣告、產品展示等，任何一般向題材都可以。
- 人物衣著整齊、行為得體；不出現裸露、性內容、暴力血腥或令人不安的畫面。
- 想法如含成人內容，改寫成一般向版本；不寫色情暗示。
- 不要假設主角性別——按想法寫「她」或「他」。

【GLOBAL SETUP 必須包含（缺一不可）】
1 人物：外貌要具體（樣貌、髮型、衣著）；有參考圖時保持與參考圖完全一致。
2 場景與光線必須明寫（不寫＝模型會退回去參考圖場景）：自然日光或明亮白光，寫明時間與天氣。
3 鏡頭：以近景與中景為主、主體清晰；可以有一個交代環境的鏡頭。
4 音訊：明寫全程無音樂——The scene is completely silent except natural sounds and voices — NO music, no melody, no singing, no humming。
5 對白：見下節。

【對白（最關鍵，寫錯＝主角變旁白）】
- 唯一合法格式：She says: "對白內容"（男角用 He says:）——H3 會自動生成唇形與語氣。
- 禁止「她說：『…』」「對白：…」這類寫法，會被當成旁白讀出。
- 對白短句、自然；每幕最多 1 至 2 句；「……」不可以當對白內容——該幕沒有對白就不要加 She says 行。
- 句尾不要用「～」，改用「……」。

【聲音句】
每幕最後照寫一句短聲音句：環境音、腳步、器物聲、呼吸或說話聲；不要音樂、旋律、歌聲、歌詞。

【動作與運鏡】
- 動作具體、可觀察、按時間順序；不寫未來式。
- 運鏡寫成自然散文（the camera slowly pushes in），不用「Camera direction:」這類標籤。
- 每幕至少一個鏡頭是固定機位；開場動作（開門、入屋）放第一幕，不要寫進 GLOBAL。

【參考圖】
- Ref2VA：keeping the exact appearance of the reference ＋ 做新動作（參考圖用來鎖樣貌，不是重播）。
- I2V：不重複描述圖中已見外觀，只寫動作、鏡頭、聲音。
- 純文字模式：不要提「參考圖」三個字，直接具體描述人物外貌。

【長片接續】
- 每幕動作自然接續上一幕尾幀姿勢（Continue directly from the previous segment），不重新演一次開頭。
- GLOBAL 不寫姿勢／動作，只寫人物、場景、光線。

【2026-09 實測鐵律】
1 每幕結尾鏡頭必須回到主體面部（清晰近景或中景），下一段接尾幀才不會變臉。
2 場景每幕明寫，防止模型退回去參考圖場景。
3 全片光線、衣著、髮型保持一致。"""

_SCRIPT_GEN_MODE_NOTES = {
    INPUT_MODE_IMAGE: (
        "The user supplies a first frame image. Do NOT re-describe the subject's "
        "static appearance - the image already carries it. Describe only what "
        "happens next, the camera, and the sound."
    ),
    INPUT_MODE_FL2VA: (
        "The user supplies a first AND a last frame. Describe only the motion "
        "that carries the first frame into the last frame. Do not restate either "
        "end pose."
    ),
    INPUT_MODE_REF2VA: (
        "The user supplies reference media. Open the GLOBAL block with a single "
        "line keeping the exact appearance of the reference. Borrow only the "
        "person or style - the setting, background, light and camera must be "
        "written out explicitly in the GLOBAL block."
    ),
}

_SCRIPT_GEN_REPAIR = """The script you produced was rejected by the validator.

Validator error: {error}

Rewrite the whole script and fix exactly that problem, keeping everything that
was already valid. Copy the required headings verbatim - do not retime them:

{skeleton}

Keep the same output language as before. Output the corrected script only.
"""


def _strip_script_wrappers(text: str) -> str:
    """Remove markdown fences and stray leading chatter from a model reply."""
    body = (text or "").strip()
    fence = re.match(r"(?s)^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```\s*$", body)
    if fence:
        body = fence.group(1).strip()
    # A model that ignores "no commentary" often prefixes one line of it. The
    # first real line of a script is either a timeline heading or a sentence of
    # the GLOBAL block, so anything else short and unterminated is chatter.
    #
    # The terminator test must include CJK punctuation: a short single-line
    # GLOBAL block ends with "。" (or ！？), not ".", and testing only for "."
    # silently deleted that line - losing the character/setting description that
    # every scene depends on.
    lines = body.splitlines()
    while lines and lines[0].strip() and not _SCRIPT_HEADING_OR_RANGE_RE.search(lines[0]):
        stripped = lines[0].strip()
        if len(stripped) < 120 and not stripped.endswith(_SENTENCE_ENDINGS):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()


# Colons are deliberately excluded: chatter such as "Here is your script:" ends
# with one and must still be stripped.
_SENTENCE_ENDINGS = (".", "。", "！", "？", "…", "!", "?")


_SCRIPT_HEADING_OR_RANGE_RE = re.compile(
    r"（\s*[0-9]+(?:\.[0-9]+)?\s*[-‐‑‒–—−~～至到]\s*[0-9]+(?:\.[0-9]+)?\s*秒\s*）"
)

_CN_DIGITS = "零一二三四五六七八九"


def _cn_number(value: int) -> str:
    """Render 1-99 as Chinese numerals for scene labels; 100+ as digits.

    Labels are free text to the parser, so falling back to Arabic numerals past
    99 is safe and avoids an out-of-range glyph lookup on very long skeletons.
    """
    if value <= 0 or value > 99:
        return str(value)
    if value < 10:
        return _CN_DIGITS[value]
    if value < 20:
        return "十" + (_CN_DIGITS[value % 10] if value % 10 else "")
    tens, ones = divmod(value, 10)
    return _CN_DIGITS[tens] + "十" + (_CN_DIGITS[ones] if ones else "")


# A deliberately irregular rhythm for scene lengths. Uniform scenes (every one
# the same length) read as mechanical; the operator explicitly asked for short
# and long shots to alternate. Values stay inside the Bot's per-scene window
# (MIN_TOTAL_SECONDS..SCRIPT_GEN_MAX_SCENE) and are whole seconds so the model
# can copy the headings exactly.
_SCRIPT_GEN_RHYTHM = (4.0, 9.0, 6.0, 12.0, 7.0, 5.0, 10.0, 3.0, 8.0, 11.0)


def _script_scene_lengths(total: float, varied: bool = True) -> list[float]:
    """Split a duration into scene lengths that sum to exactly `total`.

    Every length lands in [MIN_TOTAL_SECONDS, SCRIPT_GEN_MAX_SCENE] so the Bot's
    plan builder always accepts the result. With `varied` the lengths cycle
    through an irregular rhythm, which is what produces a natural long/short
    cutting pace; without it every scene is equal.
    """
    total = float(total)
    floor = MIN_TOTAL_SECONDS
    ceiling = SCRIPT_GEN_MAX_SCENE
    if total <= ceiling:
        return [total]

    lengths: list[float] = []
    remaining = total
    index = 0
    # Bounded so a pathological duration cannot spin forever.
    for _ in range(400):
        if remaining <= ceiling:
            lengths.append(round(remaining, 2))
            remaining = 0.0
            break
        want = _SCRIPT_GEN_RHYTHM[index % len(_SCRIPT_GEN_RHYTHM)] if varied else ceiling
        want = min(want, ceiling)
        # Never leave a tail that is too short to be a legal scene: shrink this
        # scene so the remainder is either nothing or at least the floor.
        tail = remaining - want
        if 0 < tail < floor:
            want = remaining - floor
        if want < floor:
            want = floor
        if want > remaining:
            want = remaining
        lengths.append(round(want, 2))
        remaining = round(remaining - want, 2)
        index += 1
    if remaining > 1e-6:
        lengths.append(round(remaining, 2))
    return lengths


def script_timeline_skeleton(
    total: float,
    lang: str = SCRIPT_LANG_DEFAULT,
    target_scene: float = 10.0,
    varied: bool = True,
) -> list[tuple[str, float, float]]:
    """Build an exactly-contiguous heading plan for a given duration.

    The writer model is told to copy these headings verbatim rather than invent
    timings. That matters because an example-based instruction was previously
    self-contradictory (a hard-coded example ending at 50-60s while the scenes
    before it already reached 55s), and the model faithfully reproduced the
    overlap. Computing the skeleton here makes the arithmetic correct by
    construction, so contiguity is guaranteed before the model writes a word.

    Scene lengths alternate through an irregular rhythm by default (see
    _script_scene_lengths) because uniform 10s/20s blocks read as mechanical;
    set `varied=False` for equal scenes. Returns [(label, start, end), ...]
    covering 0..total with no gap or overlap.
    """
    total = float(total)
    head_label, tail_label = _SCRIPT_GEN_HEAD_LABELS.get(
        normalize_script_lang(lang), _SCRIPT_GEN_HEAD_LABELS[SCRIPT_LANG_ZH]
    )
    if total <= 0:
        return [(head_label, 0.0, max(0.0, total))]

    lengths = _script_scene_lengths(total, varied=varied)
    n = len(lengths)

    plan: list[tuple[str, float, float]] = []
    cursor = 0.0
    for index, length in enumerate(lengths):
        start = cursor
        end = total if index == n - 1 else round(cursor + length, 2)
        if index == 0:
            label = head_label
        elif index == n - 1:
            label = tail_label
        else:
            label = f"第{_cn_number(index)}幕"
        plan.append((label, round(start, 2), round(end, 2)))
        cursor = end
    return plan


def format_skeleton(plan: list[tuple[str, float, float]]) -> str:
    """Render the skeleton as the literal heading block the model must copy."""
    lines = []
    for label, start, end in plan:
        lines.append(f"   {label}（{start:g}-{end:g}秒）：")
    return "\n".join(lines)


def _script_seconds_from_text(script: str) -> Optional[float]:
    try:
        return detect_prompt_total_seconds(script)
    except (BotError, ValueError):
        return None


def parse_idea_duration(text: str) -> tuple[Optional[float], str]:
    """Split a one-line request into (seconds, idea).

    Accepts `60秒`, `60s`, `60 seconds`, `2分鐘`, `2min`, `1分30秒`, `半分鐘`,
    and a trailing or leading placement of that expression. Returns None for the
    duration when the request carries none, so the caller can fall back to the
    current setting.

    Note on the trailing guard: `\\b` cannot be used after `秒`. CJK characters
    are Unicode word characters, so there is no word boundary between `秒` and a
    following `的`, and a perfectly ordinary request such as
    `30秒的影片 下雨的車站` would silently fail to parse. Because the duration is
    then dropped, the Bot quietly falls back to the previously selected length -
    the user asks for 30 seconds and gets 15. A negative lookahead that only
    rejects a following ASCII alphanumeric (so `30something` still cannot match)
    accepts CJK punctuation and particles while keeping that protection.
    """
    raw = (text or "").strip()
    if not raw:
        return None, ""

    tail = r"(?![0-9A-Za-z])"
    # `1分30秒` is one duration, not two. It must be tried before the
    # minutes-only pattern, which would otherwise read it as a flat 1 minute and
    # leave a stray `30秒` in the idea text.
    mixed = re.compile(
        r"(?i)([0-9]+(?:\.[0-9]+)?)\s*(?:分鐘|分钟|分)\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:秒鐘|秒钟|秒)" + tail
    )
    half = re.compile(r"(?:半)\s*(?:分鐘|分钟|分)" + tail)
    minutes = re.compile(
        r"(?i)([0-9]+(?:\.[0-9]+)?)\s*(?:分鐘|分钟|分|min(?:ute)?s?|m)" + tail
    )
    seconds = re.compile(
        r"(?i)([0-9]+(?:\.[0-9]+)?)\s*(?:秒鐘|秒钟|秒|sec(?:ond)?s?|s)" + tail
    )

    found: Optional[float] = None
    match = mixed.search(raw)
    if match:
        found = float(match.group(1)) * 60.0 + float(match.group(2))
    else:
        match = half.search(raw)
        if match:
            found = 30.0
        else:
            match = minutes.search(raw)
            if match:
                found = float(match.group(1)) * 60.0
            else:
                match = seconds.search(raw)
                if match:
                    found = float(match.group(1))

    span = match.span() if match else None
    idea = raw
    if span is not None:
        idea = idea[: span[0]] + " " + idea[span[1] :]
    # Drop the punctuation and empty brackets left where the duration was, so
    # `[30秒] 下雨` and `（30秒）下雨` do not leave `[ ]` / `（ ）` in the idea.
    idea = re.sub(r"[\[【（(]\s*[\]】）)]", " ", idea)
    idea = re.sub(r"^[\s,，、:：。.！!？?\-–—]+", "", idea)
    idea = re.sub(r"\s{2,}", " ", idea).strip()
    if found is not None and not (MIN_TOTAL_SECONDS <= found <= MAX_TOTAL_SECONDS):
        found = None
    return found, idea


def _llama_chat_once(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, str, int]:
    """One chat completion. Returns (content, finish_reason, completion_tokens)."""
    if get_script_llm_provider() == SCRIPT_LLM_COMMANDCODE:
        key = commandcode_api_key()
        if not key:
            raise BotError(
                f"Command Code API key 未設定（環境變數 {COMMANDCODE_API_KEY_ENV}）。"
            )
        payload = {
            "model": COMMANDCODE_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        try:
            data = json_request(
                f"{COMMANDCODE_BASE_URL}/chat/completions",
                payload,
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {key}",
                    "User-Agent": COMMANDCODE_USER_AGENT,
                },
            )
        except BotError as exc:
            raise BotError(
                f"Command Code 沒有回應（{COMMANDCODE_BASE_URL}）：{exc}"
            ) from exc
    else:
        payload = {
            "model": "local",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "stream": False,
        }
        try:
            data = json_request(f"{LLAMA_URL}/v1/chat/completions", payload, timeout=timeout)
        except BotError as exc:
            detail = str(exc)
            if "Loading model" in detail:
                raise BotError(
                    "本機 LLM 還在載入模型（開機後約需一分鐘），請稍候再試。"
                ) from exc
            raise BotError(
                f"本機 LLM 沒有回應（{LLAMA_URL}）：{exc}\n"
                "請確認 llama-server 正在執行，或在面板按「啟動 LLM」。"
            ) from exc
    if not isinstance(data, dict):
        raise BotError(f"{script_llm_display_name()}回傳了非預期的格式。")
    choices = data.get("choices") or []
    if not choices:
        raise BotError(f"{script_llm_display_name()}沒有產生任何內容。")
    choice = choices[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "").strip()
    finish = str(choice.get("finish_reason") or "")
    usage = data.get("usage") or {}
    try:
        tokens = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        tokens = 0
    return content, finish, tokens


def llama_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = SCRIPT_GEN_MAX_TOKENS,
    temperature: float = SCRIPT_GEN_TEMPERATURE,
    timeout: float = SCRIPT_GEN_TIMEOUT,
) -> str:
    """Send a chat completion to the local llama.cpp server and return the text.

    Only `message.content` is returned. The server runs with reasoning enabled,
    so chain-of-thought arrives in a separate `reasoning_content` field and is
    deliberately discarded - it must never leak into an H3 prompt.

    Reasoning and the answer share one token budget. When the model thinks for
    too long it can spend the whole allowance and emit an empty answer with
    finish_reason `length`. That is not a hard failure, so the call is retried
    once with a larger budget before giving up. (Disabling thinking entirely is
    NOT the fix: measurements showed the model needs it to respect the timeline;
    without it, output was consistently malformed.)
    """
    budgets = [max_tokens]
    if max_tokens < SCRIPT_GEN_RETRY_TOKENS:
        budgets.append(SCRIPT_GEN_RETRY_TOKENS)

    last_finish = ""
    last_tokens = 0
    for budget in budgets:
        content, finish, tokens = _llama_chat_once(
            messages, budget, temperature, timeout
        )
        if content:
            if finish == "length":
                bot_log(
                    f"script generator: reply hit the {budget}-token ceiling "
                    f"({tokens} tokens) but still produced content"
                )
            return content
        last_finish, last_tokens = finish, tokens
        bot_log(
            f"script generator: empty content at max_tokens={budget} "
            f"(finish={finish}, {tokens} tokens)"
        )

    if last_finish == "length":
        raise BotError(
            f"{script_llm_display_name()}的思考用光了 {budgets[-1]} token 預算，還沒開始寫正文。\n"
            "請縮短想法，或把 MINIMAX_SCRIPT_GEN_MAX_TOKENS 調更高。"
        )
    raise BotError(
        f"{script_llm_display_name()}沒有回傳正文（finish_reason={last_finish or '未知'}，"
        f"{last_tokens} tokens）。請重試。"
    )


def build_script_messages(
    idea: str,
    seconds: float,
    input_mode: str,
    lang: str = SCRIPT_LANG_DEFAULT,
    continuity: str = "",
    template: str = "",
) -> list[dict[str, str]]:
    """Compose the system/user pair that asks for a complete H3 script."""
    lang = normalize_script_lang(lang)
    template = normalize_script_template(template or get_script_template())
    long_form = seconds > MAX_SEGMENT_SECONDS
    language = _SCRIPT_GEN_LANGUAGE_BLOCK[lang]
    examples = _SCRIPT_GEN_EXAMPLES[lang]
    if long_form:
        skeleton = format_skeleton(script_timeline_skeleton(seconds, lang))
        system = _SCRIPT_GEN_LONG_SYSTEM.format(
            language=language, skeleton=skeleton, **examples
        )
    else:
        system = _SCRIPT_GEN_SHORT_SYSTEM.format(language=language, **examples)
    note = _SCRIPT_GEN_MODE_NOTES.get(input_mode)
    if note:
        system = system + "\nMODE NOTE: " + note + "\n"

    # Template guidance is appended AFTER .format() on purpose: the text is
    # arbitrary operator input, and running it through str.format() would crash
    # on any stray brace. It also has to come last so it can refine style, while
    # the STRUCTURE and heading rules above stay authoritative.
    if template == SCRIPT_TEMPLATE_GENERAL:
        # 一般版: the adult custom file is deliberately NOT read. An optional
        # script_prompt_general.txt can override the built-in general template.
        custom, _status = load_custom_script_instructions(
            SCRIPT_GEN_PROMPT_FILE_GENERAL
        )
        if not custom:
            custom = _SCRIPT_GEN_GENERAL_TEMPLATE
        header = (
            "\nGENERAL MODE (operator guidance; general-audience content only - "
            "follow these rules, keep everything non-explicit, and ignore any "
            "conflicting style guidance above):\n"
        )
    else:
        custom, _status = load_custom_script_instructions()
        header = (
            "\nCUSTOM INSTRUCTIONS (from the operator; follow these for style "
            "and content - they take priority over the general style guidance "
            "above, but never override the STRUCTURE section or the exact "
            "timeline headings):\n"
        )
    if custom:
        system = system + header + custom + "\n"

    # Continuity block (set when the new reference images are the tail frames of
    # the previous clip). Appended last so the "same person, same place" contract
    # outranks every generic style rule above it.
    if continuity:
        system = system + "\n" + continuity.strip() + "\n"

    if long_form:
        user = (
            f"Total duration: {seconds:g} seconds exactly.\n"
            f"Idea: {idea}\n"
            "Write the complete script now, using exactly the headings given. "
            f"The final heading ends at {seconds:g} seconds."
        )
    else:
        user = (
            f"Clip duration: {seconds:g} seconds.\n"
            f"Idea: {idea}\n"
            "Write the prompt paragraph now."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _script_accepts(script: str, seconds: float) -> tuple[bool, str]:
    """Run the Bot's own validator over a draft.

    Returns (ok, message). When ok the message describes what passed; otherwise
    it is the exact parser error, which is what both the self-repair loop and the
    user-facing warnings need.
    """
    if seconds <= MAX_SEGMENT_SECONDS:
        if len(script) < 40:
            return False, "產出的提示詞太短。"
        return True, f"{len(script)} 字元（短片散文格式）"
    try:
        plan = build_long_video_plan(script, float(seconds))
    except (BotError, ValueError) as exc:
        return False, str(exc)
    return True, f"通過驗證，{len(plan.shots)} 個鏡頭"


_SCRIPT_GEN_REFINE_USER = """Here is the CURRENT script:

--- CURRENT ---
{draft}
--- END ---

Revise it according to this instruction:
{instruction}

Rewrite the WHOLE script with that change applied. Keep everything that is not
affected by the instruction exactly as good as it already is, and return the
complete script in the same format and the same language.{headings_note}
"""


def refine_h3_script(
    draft: str,
    instruction: str,
    seconds: float,
    input_mode: str = INPUT_MODE_TEXT,
    *,
    on_progress: Optional[Any] = None,
    attempts: int = SCRIPT_GEN_ATTEMPTS,
    lang: str = SCRIPT_LANG_DEFAULT,
    continuity: str = "",
) -> tuple[str, list[str]]:
    """Revise an existing script from a natural-language instruction.

    Used for the second, third and every later round of editing, so it must be
    repeatable without degrading the result. The same structure contract and the
    same validator apply as for a first draft, which is what keeps repeated
    LLM edits from drifting into an ungeneratable prompt.
    """
    lang = normalize_script_lang(lang)
    long_form = seconds > MAX_SEGMENT_SECONDS
    # Reuse the generation system prompt verbatim so every structural rule,
    # language rule, mode note and operator override still applies on an edit.
    system = build_script_messages(instruction, seconds, input_mode, lang, continuity)[0]["content"]
    skeleton = format_skeleton(script_timeline_skeleton(float(seconds), lang))
    headings_note = (
        "\nThe timeline headings must stay EXACTLY as they are - do not add, "
        "remove, rename or re-time any of them."
        if long_form
        else ""
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": _SCRIPT_GEN_REFINE_USER.format(
                draft=draft,
                instruction=instruction,
                headings_note=headings_note,
            ),
        },
    ]

    log: list[str] = []
    last_error = ""
    for attempt in range(1, max(1, attempts) + 1):
        if on_progress is not None:
            on_progress(attempt, max(1, attempts), last_error)
        reply = llama_chat(messages, timeout=SCRIPT_GEN_TIMEOUT)
        script = _strip_script_wrappers(reply)
        messages.append({"role": "assistant", "content": script})
        ok, message = _script_accepts(script, seconds)
        if ok:
            log.append(f"第 {attempt} 次嘗試：{message}")
            return script, log
        last_error = message
        log.append(f"第 {attempt} 次嘗試：驗證失敗 — {last_error}")
        if attempt < attempts:
            messages.append(
                {
                    "role": "user",
                    "content": _SCRIPT_GEN_REPAIR.format(
                        error=last_error,
                        skeleton=skeleton,
                    ),
                }
            )

    raise BotError(
        f"連續 {attempts} 次都無法產出符合格式的修改結果。\n最後錯誤：{last_error}"
    )


def generate_h3_script(
    idea: str,
    seconds: float,
    input_mode: str = INPUT_MODE_TEXT,
    *,
    on_progress: Optional[Any] = None,
    attempts: int = SCRIPT_GEN_ATTEMPTS,
    lang: str = SCRIPT_LANG_DEFAULT,
    continuity: str = "",
) -> tuple[str, list[str]]:
    """Generate a script and self-repair it until the Bot's validator accepts.

    Returns the script plus a human-readable log of what happened. Raises
    BotError when every attempt still fails validation, and includes the last
    parser error so the failure is diagnosable from Telegram alone.
    """
    lang = normalize_script_lang(lang)
    messages = build_script_messages(idea, seconds, input_mode, lang, continuity)
    skeleton = format_skeleton(script_timeline_skeleton(float(seconds), lang))
    log: list[str] = []
    last_error = ""

    for attempt in range(1, max(1, attempts) + 1):
        if on_progress is not None:
            on_progress(attempt, max(1, attempts), last_error)
        reply = llama_chat(messages, timeout=SCRIPT_GEN_TIMEOUT)
        script = _strip_script_wrappers(reply)
        messages.append({"role": "assistant", "content": script})

        ok, message = _script_accepts(script, seconds)
        if ok:
            log.append(f"第 {attempt} 次嘗試：{message}")
            return script, log
        last_error = message
        log.append(f"第 {attempt} 次嘗試：驗證失敗 — {last_error}")

        if attempt < attempts:
            messages.append(
                {
                    "role": "user",
                    "content": _SCRIPT_GEN_REPAIR.format(
                        error=last_error,
                        skeleton=skeleton,
                    ),
                }
            )

    raise BotError(
        f"連續 {attempts} 次都無法產出符合格式的腳本。\n最後錯誤：{last_error}"
    )


class TelegramClient:
    def __init__(self, token: str):
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.file_base_url = f"https://api.telegram.org/file/bot{token}"

    def call(self, method: str, params: Optional[dict[str, Any]] = None, timeout: float = 45.0) -> Any:
        query = urlencode(params or {}, doseq=True)
        url = f"{self.base_url}/{method}"
        if query:
            url += "?" + query
        try:
            with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise BotError(
                f"Telegram 連線失敗：{http_error_detail(exc)}"
            ) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise BotError(f"Telegram 連線失敗：{exc}") from exc
        if not result.get("ok"):
            raise BotError(result.get("description", "Telegram API 失敗。"))
        return result.get("result")

    def set_my_commands(self, commands: list[dict[str, str]]) -> None:
        self.call(
            "setMyCommands",
            {"commands": json.dumps(commands, ensure_ascii=False)},
            timeout=30,
        )

    def set_chat_menu_button(self, chat_id: str) -> None:
        self.call(
            "setChatMenuButton",
            {
                "chat_id": chat_id,
                "menu_button": json.dumps({"type": "default"}),
            },
            timeout=30,
        )

    def get_file(self, file_id: str) -> str:
        result = self.call("getFile", {"file_id": file_id}, timeout=30)
        file_path = result.get("file_path") if isinstance(result, dict) else None
        if not file_path:
            raise BotError("Telegram 沒有回傳檔案路徑。")
        return str(file_path)

    def download_bytes(
        self,
        file_path: str,
        max_bytes: int,
        kind: str = "檔案",
    ) -> bytes:
        try:
            with urlopen(
                Request(
                    f"{self.file_base_url}/{file_path}",
                    headers={"Accept": "application/octet-stream"},
                ),
                timeout=120,
            ) as response:
                data = response.read(max_bytes + 1)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise BotError(f"下載 Telegram {kind}失敗：{exc}") from exc
        if len(data) > max_bytes:
            raise BotError(f"{kind}太大，請控制在 {max_bytes / 1024:g} KB 以內。")
        return data

    def download_file(self, file_path: str, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.download_bytes(file_path, MAX_TELEGRAM_IMAGE_BYTES, "圖片")
        target_path.write_bytes(data)

    def get_updates(self, offset: Optional[int]) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "timeout": 25,
            "allowed_updates": json.dumps(["message", "callback_query"]),
        }
        if offset is not None:
            params["offset"] = offset
        result = self.call("getUpdates", params, timeout=35)
        return result or []

    def send_message(
        self,
        chat_id: str,
        text: str,
        reply_markup: Optional[dict[str, Any]] = None,
    ) -> None:
        params: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_markup is not None:
            params["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
        return self.call("sendMessage", params, timeout=30)

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        params: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            params["text"] = text
        self.call("answerCallbackQuery", params, timeout=30)

    def send_chat_action(self, chat_id: str, action: str = "typing") -> None:
        """Show Telegram's progress indicator while a slow local task runs."""
        try:
            self.call(
                "sendChatAction",
                {"chat_id": chat_id, "action": action},
                timeout=15,
            )
        except BotError:
            # Purely cosmetic; never let it break the work it is decorating.
            pass

    def edit_message_text(
        self,
        chat_id: str,
        message_id: int,
        text: str,
        reply_markup: Optional[dict[str, Any]] = None,
    ) -> None:
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        if reply_markup is not None:
            params["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
        self.call("editMessageText", params, timeout=30)

    def delete_message(self, chat_id: str, message_id: int) -> None:
        self.call(
            "deleteMessage",
            {"chat_id": chat_id, "message_id": message_id},
            timeout=30,
        )

    def send_video(self, chat_id: str, video_path: Path, caption: str) -> None:
        if not video_path.is_file():
            raise BotError(f"找不到要傳送的影片：{video_path}")

        original_size = video_path.stat().st_size
        temporary_paths: list[Path] = []
        upload_url = f"{self.base_url}/sendVideo"
        fields = {
            "chat_id": chat_id,
            "supports_streaming": "true",
        }
        try:
            if original_size <= TELEGRAM_SAFE_VIDEO_BYTES:
                multipart_request(
                    upload_url,
                    {**fields, "caption": caption},
                    "video",
                    video_path,
                )
                return

            try:
                self.send_message(
                    chat_id,
                    "影片超過 Telegram 50 MB 上傳限制，正在自動壓縮；原片會保留在電腦。",
                )
            except BotError:
                pass

            compressed_path = compress_video_for_telegram(video_path)
            if compressed_path is not None:
                temporary_paths.append(compressed_path)
                compressed_size = compressed_path.stat().st_size
                note = (
                    "原片約 "
                    f"{original_size / 1_000_000:.1f} MB，已自動壓縮至 "
                    f"{compressed_size / 1_000_000:.1f} MB。"
                )
                multipart_request(
                    upload_url,
                    {
                        **fields,
                        "caption": telegram_caption_with_note(caption, note),
                    },
                    "video",
                    compressed_path,
                )
                return

            parts = split_video_for_telegram(video_path)
            temporary_paths.extend(parts)
            try:
                self.send_message(
                    chat_id,
                    f"影片仍然太大，將自動分成 {len(parts)} 段傳送。",
                )
            except BotError:
                pass
            for index, part_path in enumerate(parts, start=1):
                part_caption = telegram_caption_with_note(
                    caption,
                    f"檔案過大，已分段傳送：第 {index}/{len(parts)} 段。",
                )
                multipart_request(
                    upload_url,
                    {**fields, "caption": part_caption},
                    "video",
                    part_path,
                )
        finally:
            for temporary_path in temporary_paths:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def send_photo(
        self,
        chat_id: str,
        photo_path: Path,
        caption: str = "",
        reply_markup: Optional[dict[str, Any]] = None,
    ) -> None:
        """Send one image with an optional inline keyboard."""
        if not photo_path.is_file():
            raise BotError(f"找不到要傳送的圖片：{photo_path}")
        fields: dict[str, str] = {"chat_id": chat_id}
        if caption:
            fields["caption"] = caption
        if reply_markup is not None:
            fields["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
        multipart_request(
            f"{self.base_url}/sendPhoto",
            fields,
            "photo",
            photo_path,
            content_type="image/jpeg",
        )

    def clear_inline_keyboard(self, chat_id: str, message_id: int) -> None:
        """Remove the inline buttons from a message (photo or text)."""
        self.call(
            "editMessageReplyMarkup",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": json.dumps({"inline_keyboard": []}),
            },
            timeout=30,
        )


class TelegramTurboBot:
    def __init__(self, token: str, allowed_chat_id: str):
        self.telegram = TelegramClient(token)
        self.allowed_chat_id = str(allowed_chat_id)
        self.offset: Optional[int] = None
        self.pending_config: Optional[GenerationConfig] = None
        self.job: Optional[JobState] = None
        self.pending_upscale: Optional[PendingUpscale] = None
        self.lock = threading.Lock()
        self.progress_message_lock = threading.Lock()
        self.progress_message_id: Optional[int] = None
        self.progress_message_chat_id: Optional[str] = None
        self.progress_message_text = ""
        self.progress_refresh_thread = threading.Thread(
            target=self._progress_refresh_loop,
            name="telegram-progress-refresh",
            daemon=True,
        )
        self.progress_refresh_thread.start()
        self._comfy_idle_since = time.time()
        self._comfy_idle_shutdown_stop = threading.Event()
        self.comfy_idle_shutdown_thread = threading.Thread(
            target=self._comfy_idle_shutdown_loop,
            name="comfyui-idle-shutdown",
            daemon=True,
        )
        self.comfy_idle_shutdown_thread.start()

    def touch_comfy_activity(self) -> None:
        """Reset the idle countdown after a Bot task or ComfyUI control action."""
        self._comfy_idle_since = time.time()

    def _comfy_idle_shutdown_loop(self) -> None:
        if COMFY_IDLE_SHUTDOWN_SECONDS <= 0:
            return
        while not self._comfy_idle_shutdown_stop.wait(
            COMFY_IDLE_CHECK_INTERVAL_SECONDS
        ):
            if not comfyui_is_online():
                self.touch_comfy_activity()
                continue
            with self.lock:
                active_job = self.job is not None
            if active_job or comfyui_has_pending_work():
                self.touch_comfy_activity()
                continue
            idle_seconds = time.time() - self._comfy_idle_since
            if idle_seconds < COMFY_IDLE_SHUTDOWN_SECONDS:
                continue
            try:
                result = stop_comfyui_process()
            except Exception as exc:
                bot_log(f"idle ComfyUI shutdown failed: {exc}")
                self.touch_comfy_activity()
                continue
            self.touch_comfy_activity()
            bot_log(
                f"ComfyUI auto-stopped after {idle_seconds:.0f}s idle: {result}"
            )
            self.send_safe(
                self.allowed_chat_id,
                "ComfyUI 閒置超過 5 分鐘，已自動關閉以釋放顯存。"
                "需要生成時按「▶️ 啟動 ComfyUI」或輸入 /comfy_start。",
            )

    def help_text(self) -> str:
        return (
            "MiniMax H3 Turbo 控制器\n\n"
            "/gen 寬度 高度 steps 秒數\n"
            "例如：/gen 1344 768 4 5\n"
            "下一則訊息貼完整提示詞即可。\n\n"
            "也可同一則訊息輸入：\n"
            "/gen 1344 768 4 5\n你的提示詞\n\n"
            "/status 查看狀態\n"
            "/make [秒數] 一句話想法 → 本機 LLM 生成完整腳本\n"
            "   例如：/make 60秒 下雨的車站，女生錯過末班車\n"
            "   也可直接打 /make 再依提示輸入\n"
            "/lang zh|en 切換腳本語言（預設簡體中文）\n"
            "/scriptllm 切換腳本 LLM（本機 / Command Code）\n"
            "/scripttemplate 切換腳本模板（成人版 / 一般版）\n"
            "/chain 5 8 5 10 或 48 [想法] 自動接力：短段生成→尾帧接續→合併（/chain off 停）\n"
            "/gpu GPU 狀態：邊張卡係屏幕卡、ComfyUI 會用邊張\n"
            "/prompt_file 查看／編輯自訂指令檔（存檔即生效，免重啟）\n"
            "/prompt_help 提示詞寫作精華\n"
            "/progress 查看即時生成進度\n"
            "/pause 暫停長片（在目前鏡頭完成後）\n"
            "/resume 或 /play 繼續長片\n"
            "/preview 預覽已完成的長片片段（生成不中斷）\n"
            "長片顯存不足時會保留已完成鏡頭，自動逐級降低解析度重試\n"
             "/resume_long 從失敗檢查點繼續長片\n"
             "/extend 秒數 [提示詞] 從上一條完整長片尾端延續\n"
             "/history 查看歷史長片並選擇 ID\n"
             "/queue 查看故事排隊\n"
             "/queue_add 加入一個或多個故事\n"
             "/queue_start 開始排隊\n"
             "/queue_clear 清空等待中的故事\n"
            "/temperature 查看 GPU／CPU 溫度\n"
            "「⚙️ 片長／解析度／steps」頁可開關「🔬 兩段式 latent 上採樣」：\n"
            "開＝半解析度底片→latent 放大→全解析度精修（較清晰、稍慢）\n"
            "關＝依所選解析度單段直出（較快）\n"
            "/cancel_shutdown 取消已排程的自動關機\n"
            "/comfy_restart 重啟 ComfyUI\n"
            "/comfy_stop 關閉 ComfyUI\n"
            "/comfy_start 啟動 ComfyUI（閒置自動關閉見 MINIMAX_COMFY_IDLE_SHUTDOWN_SECONDS）\n"
            "/bot_restart 重啟 Telegram Bot\n"
            "/cancel 取消目前生成\n"
            "/help 查看說明"
        )

    def send_safe(self, chat_id: str, text: str) -> None:
        try:
            self.telegram.send_message(
                chat_id,
                text,
                reply_markup=control_panel_reply_markup(),
            )
        except BotError as exc:
            print(f"Telegram sendMessage error: {exc}", flush=True)

    def offer_upscale(
        self,
        chat_id: str,
        video_path: Path,
        source_width: int,
        source_height: int,
        duration_seconds: float,
        shutdown_after_choice: bool = False,
    ) -> None:
        """Keep the original and expose optional SeedVR2 actions in Telegram."""
        token = uuid.uuid4().hex[:12]
        pending = PendingUpscale(
            token=token,
            chat_id=str(chat_id),
            source_path=video_path,
            source_width=int(source_width),
            source_height=int(source_height),
            duration_seconds=float(duration_seconds),
            shutdown_after_choice=shutdown_after_choice,
        )
        with self.lock:
            self.pending_upscale = pending
        markup = {
            "inline_keyboard": [
                [
                    {
                        "text": "⬆️ 放大到 1080p",
                        "callback_data": f"upscale:1080:{token}",
                    }
                ],
                [
                    {
                        "text": "⬆️ 放大到 2K",
                        "callback_data": f"upscale:2k:{token}",
                    }
                ],
                [
                    {
                        "text": "✅ 保留原片",
                        "callback_data": f"upscale:keep:{token}",
                    }
                ],
            ]
        }
        try:
            self.telegram.send_message(
                chat_id,
                "原片已回傳。要不要用 SeedVR2 3B INT8 放大？\n"
                "放大會另外建立 ComfyUI 任務，原片會保留。",
                reply_markup=markup,
            )
        except BotError as exc:
            print(f"Telegram upscale menu error: {exc}", flush=True)

    def finalize_upscale_choice(
        self, chat_id: str, pending: PendingUpscale
    ) -> None:
        """Hook for the menu bot to apply deferred long-video shutdown."""
        return

    def run_upscale_job(self, job: JobState, pending: PendingUpscale) -> None:
        """Run one optional SeedVR2 upscale after H3 has returned the source."""
        started_at = time.time()
        source_path = pending.source_path
        try:
            if source_path is None or not source_path.is_file():
                raise BotError("找不到要放大的原片，請重新生成一次。")
            self.ensure_comfyui_ready(job)
            self.send_safe(job.chat_id, "SeedVR2 放大開始：正在載入原片與模型，請稍候。")
            try:
                comfy_post("/free", {"unload_models": True, "free_memory": True})
            except BotError as exc:
                bot_log(f"ComfyUI memory release before SeedVR2 was unavailable: {exc}")
            input_video_name = upload_video_to_comfy(source_path)
            target_long_edge = max(job.upscale_target_width, job.upscale_target_height)
            output_prefix = f"MiniMaxH3/Telegram_Turbo_Upscale/{uuid.uuid4().hex[:12]}"
            workflow = build_seedvr2_workflow(
                input_video_name,
                target_long_edge,
                output_prefix,
                split_latent=pending.duration_seconds > SEEDVR2_SPLIT_SECONDS,
            )
            response = comfy_post(
                "/prompt",
                {"prompt": workflow, "client_id": "telegram-turbo-bot"},
            )
            prompt_id = response.get("prompt_id")
            if not prompt_id:
                raise BotError(f"ComfyUI 沒有回傳放大 prompt_id：{response}")
            job.prompt_id = str(prompt_id)
            with job.progress_lock:
                job.progress_percent = 0.0
                job.progress_phase = "waiting"
                job.progress_node_state = "queued"
                job.progress_queue_remaining = None
            progress_tracker = ComfyProgressTracker(job)
            job.progress_tracker = progress_tracker
            progress_tracker.start()
            self.send_safe(
                job.chat_id,
                f"已開始 SeedVR2 放大：{job.upscale_target_width}×{job.upscale_target_height}\n"
                f"Prompt ID: {prompt_id}",
            )
            try:
                history: Optional[dict[str, Any]] = None
                while True:
                    if job.cancel_event.is_set():
                        raise BotError("SeedVR2 放大已取消。")
                    try:
                        history_all = comfy_post(f"/history/{prompt_id}")
                        history = (
                            history_all.get(str(prompt_id))
                            if isinstance(history_all, dict)
                            else None
                        )
                    except BotError:
                        history = None
                    if history:
                        status = history.get("status", {})
                        status_name = status.get("status_str")
                        if status_name == "error":
                            raise BotError(self.execution_error(history))
                        if status.get("completed") or status_name == "success":
                            break
                    time.sleep(3)
            finally:
                progress_tracker.stop()
                if job.progress_tracker is progress_tracker:
                    job.progress_tracker = None

            output_path = self.find_video(
                history or {}, started_at, name_hint="Telegram_Turbo_Upscale"
            )
            if output_path is None:
                raise BotError("SeedVR2 已完成，但找不到放大後的 MP4。")
            with job.progress_lock:
                job.progress_percent = 100.0
                job.progress_phase = "uploading"
                job.progress_node_state = "finished"
            elapsed = time.time() - started_at
            caption = (
                f"SeedVR2 放大完成\n{job.upscale_target_width}×{job.upscale_target_height} | "
                f"{format_elapsed(elapsed)}"
            )
            self.telegram.send_video(job.chat_id, output_path, caption)
            self.send_safe(
                job.chat_id,
                "SeedVR2 放大完成。\n"
                f"總用時：{format_elapsed(elapsed)}\n"
                f"{seedvr2_usage_report(target_long_edge)}",
            )
            self.finalize_upscale_choice(job.chat_id, pending)
            bot_log(f"seedvr2 upscale done {output_path}")
        except Exception as exc:
            if not job.cancel_event.is_set():
                self.send_safe(job.chat_id, f"SeedVR2 放大失敗：{exc}")
            bot_log(f"seedvr2 upscale error: {exc}")
            print(f"upscale error: {exc}", flush=True)
        finally:
            self.touch_comfy_activity()
            with self.lock:
                if self.job is job:
                    self.job = None
            self.on_job_finished(job.chat_id)

    def _progress_refresh_loop(self) -> None:
        while True:
            time.sleep(3.0)
            with self.lock:
                job = self.job
            with self.progress_message_lock:
                message_id = self.progress_message_id
                message_chat_id = self.progress_message_chat_id
                previous_text = self.progress_message_text
            if job is None or message_id is None or not message_chat_id:
                continue

            text = self.progress_text()
            if text == previous_text:
                continue
            try:
                self.telegram.edit_message_text(message_chat_id, message_id, text)
            except BotError as exc:
                error_text = str(exc).lower()
                if "not modified" in error_text:
                    continue
                if "message to edit not found" in error_text:
                    with self.progress_message_lock:
                        if self.progress_message_id == message_id:
                            self.progress_message_id = None
                            self.progress_message_chat_id = None
                            self.progress_message_text = ""
                    continue
                print(f"Telegram progress refresh error: {exc}", flush=True)
                continue
            with self.progress_message_lock:
                if self.progress_message_id == message_id:
                    self.progress_message_text = text

    @staticmethod
    def progress_bar(percent: float) -> str:
        bounded = max(0.0, min(100.0, percent))
        filled = min(10, int(bounded / 10.0))
        return "█" * filled + "░" * (10 - filled)

    @staticmethod
    def eta_text(overall: float, elapsed: int) -> str:
        """Linear extrapolation of remaining time once real progress exists."""
        if overall < 5.0 or overall >= 99.5:
            return ""
        remaining = elapsed * (100.0 - overall) / max(overall, 0.001)
        if remaining < 90:
            eta_human = f"{remaining:.0f} 秒"
        elif remaining < 5400:
            eta_human = f"{remaining / 60:.0f} 分鐘"
        else:
            eta_human = f"{remaining / 3600:.1f} 小時"
        finish = time.strftime("%H:%M", time.localtime(time.time() + remaining))
        return f"預計剩餘：{eta_human}（約 {finish} 完成）"

    def progress_text(self) -> str:
        with self.lock:
            job = self.job
            pending = self.pending_config
        if job is None:
            if pending is not None:
                return "目前沒有生成中的工作，正在等待你貼上提示詞。"
            return "目前沒有生成中的工作。"

        with job.progress_lock:
            percent = job.progress_percent
            node_id = job.progress_node_id
            node_state = job.progress_node_state
            node_value = job.progress_node_value
            node_max = job.progress_node_max
            node_index = job.progress_node_index
            node_total = job.progress_node_total
            queue_remaining = job.progress_queue_remaining
            phase = job.progress_phase

        phase_labels = {
            "queued": "等待 ComfyUI 開始",
            "waiting": "等待 ComfyUI 回報進度",
            "running": "執行 ComfyUI 節點",
            "sampling": "採樣中",
            "paused": "已暫停，等待播放／繼續",
            "finishing": "正在整理影片與音訊",
            "completed": "本段生成完成",
            "merging": "正在合併長片分段",
            "uploading": "正在傳回 Telegram",
            "error": "生成錯誤",
        }
        phase_text = phase_labels.get(phase, phase)
        elapsed = max(0, int(time.time() - job.started_at))
        elapsed_text = f"{elapsed // 60}分 {elapsed % 60}秒"

        if job.segment_total > 1:
            if job.shot_plan:
                shot = job.shot_plan[job.segment_index - 1]
                completed_story_seconds = shot.start_seconds
                active_story_seconds = shot.duration * percent / 100.0
                overall = min(
                    100.0,
                    ((completed_story_seconds + active_story_seconds) / job.total_seconds)
                    * 100.0,
                )
                segment_line = (
                    f"長片：鏡頭 {job.segment_index}/{job.segment_total} "
                    f"（劇情 {shot.start_seconds:g}-{shot.end_seconds:g} 秒）\n"
                    f"總進度：{self.progress_bar(overall)} {overall:.1f}%\n"
                    f"本鏡進度：{self.progress_bar(percent)} {percent:.1f}%"
                )
            else:
                completed_segments = max(0, job.segment_index - 1)
                overall = min(
                    100.0,
                    ((completed_segments + percent / 100.0) / job.segment_total) * 100.0,
                )
                segment_line = (
                    f"長片：第 {job.segment_index}/{job.segment_total} 段\n"
                    f"總進度：{self.progress_bar(overall)} {overall:.1f}%\n"
                    f"本段進度：{self.progress_bar(percent)} {percent:.1f}%"
                )
        else:
            overall = percent
            segment_line = f"進度：{self.progress_bar(overall)} {overall:.1f}%"

        if job.task_type == YUPI_TASK_TYPE:
            node_labels = {
                "1": "載入影片 VAE",
                "2": "載入音訊 VAE",
                "3": "載入 H3 視覺編碼器",
                "4": "載入 Ref2VA 模型",
                "5": "套用 AfterMidnight LoRA",
                "6": "建立參考條件",
                "7": "設定 Euler 採樣器",
                "8": "準備噪聲",
                "9": "建立引導",
                "10": "採樣",
                "12": "儲存影片",
                "13": "設定 Beta 排程",
                "14": "載入參考圖片",
                "15": "套用 FastH3 蒸餾 LoRA",
                "16": "套用 HMNSFW 動作 LoRA",
                "21": "影片 VAE 解碼",
                "22": "音訊 VAE 解碼",
                "23": "建立同步影片",
            }
        else:
            node_labels = {
                "1": "載入影片 VAE",
                "2": "載入音訊 VAE",
                "3": "載入文字／視覺編碼器",
                "4": "載入 H3 模型",
                "5": "套用 Turbo",
                "6": "建立條件",
                "7": "設定採樣器",
                "8": "準備噪聲",
                "9": "引導",
                "10": "採樣",
                "11": "VAE 解碼",
                "12": "儲存影片",
            }
        task_labels = {
            "seedvr2": "SeedVR2 放大",
            YUPI_TASK_TYPE: (
                "🌙 YUPI（6步）"
                if job.yupi_fast
                else "🌙 YUPI Ref2VA"
            ),
        }
        lines = [
            f"📊 {task_labels.get(job.task_type, 'MiniMax H3')} 進度",
            segment_line,
            f"狀態：{phase_text}",
            f"已用時間：{elapsed_text}",
        ]
        eta_line = self.eta_text(overall, elapsed)
        if eta_line:
            lines.append(eta_line)
        if job.pause_requested.is_set():
            control_text = (
                "已暫停，等待播放／繼續"
                if phase == "paused"
                else "已收到暫停，會在目前鏡頭完成後停下"
            )
        else:
            control_text = "正常執行"
        lines.append(f"控制：{control_text}")
        if node_id is not None:
            node_name = node_labels.get(str(node_id), f"ComfyUI 節點 {node_id}")
            if node_total:
                node_name = f"{node_name}（{node_index}/{node_total}）"
            lines.append(f"目前：{node_name}｜{node_state}")
            if node_max > 1:
                lines.append(f"節點進度：{node_value:.0f}/{node_max:.0f}")
        elif phase in {"queued", "waiting", "running"}:
            lines.append("詳細節點進度尚未回報，但 ComfyUI 任務仍在處理。")
        if queue_remaining is not None:
            lines.append(f"ComfyUI 佇列剩餘：{queue_remaining}")
        if job.prompt_id:
            lines.append(f"Prompt ID：{job.prompt_id}")
        return "\n".join(lines)

    def _send_progress_message(self, chat_id: str) -> None:
        """Create the single Telegram message that the refresh thread edits."""
        text = self.progress_text()
        with self.progress_message_lock:
            old_message_id = self.progress_message_id
            old_chat_id = self.progress_message_chat_id
            self.progress_message_id = None
            self.progress_message_chat_id = None
            self.progress_message_text = ""

        if old_message_id is not None and old_chat_id:
            try:
                self.telegram.delete_message(old_chat_id, old_message_id)
            except BotError:
                pass

        try:
            result = self.telegram.send_message(chat_id, text)
            new_message_id = (
                result.get("message_id") if isinstance(result, dict) else None
            )
            if new_message_id:
                with self.progress_message_lock:
                    self.progress_message_id = int(new_message_id)
                    self.progress_message_chat_id = chat_id
                    self.progress_message_text = text
        except BotError as exc:
            self.send_safe(chat_id, f"顯示生成進度失敗：{exc}")

    def handle_message(self, message: dict[str, Any]) -> None:
        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != self.allowed_chat_id:
            return
        text = str(message.get("text", "")).strip()
        if not text:
            return

        if text.startswith("/"):
            self.handle_command(chat_id, text)
            return

        with self.lock:
            pending = self.pending_config
            self.pending_config = None
        if pending is None:
            self.send_safe(chat_id, "請先使用 /gen 寬度 高度 steps 秒數，再貼提示詞。")
            return
        self.start_generation(chat_id, pending, text)

    def handle_command(self, chat_id: str, text: str) -> None:
        lines = text.splitlines()
        first = lines[0].strip()
        parts = first.split()
        command = parts[0].split("@", 1)[0].lower()

        if command in {"/start", "/help"}:
            self.send_safe(chat_id, self.help_text())
            return
        if command == "/progress":
            self._send_progress_message(chat_id)
            return
        if command == "/status":
            with self.lock:
                job = self.job
                pending = self.pending_config
            if pending:
                self.send_safe(chat_id, "等待你貼上提示詞。")
            elif job:
                current = job.prompt_id or "正在提交到 ComfyUI"
                self.send_safe(
                    chat_id,
                    f"生成中：{current}\n{job.config.width}×{job.config.height} | "
                    f"{job.config.steps} steps | 約 {job.config.actual_seconds:.2f} 秒",
                )
            else:
                self.send_safe(chat_id, "目前沒有生成工作。")
            return
        if command == "/cancel":
            with self.lock:
                job = self.job
                self.pending_config = None
                if job:
                    job.cancel_event.set()
            if job:
                try:
                    comfy_post("/interrupt", {})
                except BotError:
                    pass
                self.send_safe(chat_id, "已要求取消目前生成。")
            else:
                self.send_safe(chat_id, "沒有正在生成的工作。")
            return
        if command != "/gen":
            self.send_safe(chat_id, "不認識這個指令，輸入 /help 查看用法。")
            return

        if len(parts) < 5:
            self.send_safe(chat_id, "格式：/gen 寬度 高度 steps 秒數\n例如：/gen 1344 768 4 5")
            return
        try:
            config = parse_config(parts[1:5])
        except BotError as exc:
            self.send_safe(chat_id, str(exc))
            return

        inline_prompt = " ".join(parts[5:]).strip()
        if len(lines) > 1:
            inline_prompt = (inline_prompt + "\n" + "\n".join(lines[1:])).strip()
        if inline_prompt:
            self.start_generation(chat_id, config, inline_prompt)
            return

        with self.lock:
            if self.job:
                self.send_safe(chat_id, "目前已有工作在生成，請先等待完成或使用 /cancel。")
                return
            self.pending_config = config
        self.send_safe(
            chat_id,
            f"設定已收取：{config.width}×{config.height} | {config.steps} steps | "
            f"約 {config.actual_seconds:.2f} 秒。\n請下一則訊息貼上完整提示詞。",
        )

    def start_generation(
        self,
        chat_id: str,
        config: GenerationConfig,
        prompt: str,
        input_image_path: Optional[Path] = None,
        last_image_path: Optional[Path] = None,
        reference_image_paths: Optional[list[Path]] = None,
        reference_video_paths: Optional[list[Path]] = None,
        reference_audio_paths: Optional[list[Path]] = None,
        generation_mode: str = INPUT_MODE_TEXT,
        task_type: str = MODEL_H3,
    ) -> bool:
        prompt = prompt.strip()
        if not prompt:
            self.send_safe(chat_id, "提示詞不可為空白。")
            return False
        mode = normalize_input_mode(generation_mode)
        if mode == INPUT_MODE_FL2VA and (
            input_image_path is None
            or not input_image_path.is_file()
            or last_image_path is None
            or not last_image_path.is_file()
        ):
            self.send_safe(chat_id, "FL2VA 需要首幀和尾幀兩張圖片。")
            return False
        if mode == INPUT_MODE_REF2VA:
            if not (
                any(path.is_file() for path in (reference_image_paths or []))
                or any(path.is_file() for path in (reference_video_paths or []))
                or any(path.is_file() for path in (reference_audio_paths or []))
            ):
                self.send_safe(chat_id, "Ref2VA 尚未收到參考素材，請先上傳圖片、影片或音訊。")
                return False
            try:
                require_ref2va_model()
            except BotError as exc:
                self.send_safe(chat_id, str(exc))
                return False
        with self.lock:
            if self.job:
                self.send_safe(chat_id, "目前已有工作在生成，請先等待完成或使用 /cancel。")
                return False
            job = JobState(
                chat_id,
                config,
                prompt,
                time.time(),
                cancel_event=threading.Event(),
                input_image_path=input_image_path,
                last_image_path=last_image_path,
                reference_image_paths=list(reference_image_paths or []),
                reference_video_paths=list(reference_video_paths or []),
                reference_audio_paths=list(reference_audio_paths or []),
                task_type=normalize_task_type(task_type),
                generation_mode=mode,
            )
            job.resume_event.set()
            self.job = job
        self.touch_comfy_activity()
        thread = threading.Thread(target=self.run_job, args=(job,), daemon=True)
        thread.start()
        return True

    def run_yupi_generation(self, chat_id: str, fast: bool = False) -> None:
        """🌙 YUPI工作流：isolated NSFW generation (Ref2VA + AfterMidnight LoRA,
        euler/beta). Loads yupi_nsfw_api.json, injects the staged prompt and a
        reference image, submits to ComfyUI, and sends the MP4 back. This path
        is separate from the stock Turbo workflow, but still uses the shared
        JobState so progress, cancel, idle-shutdown, and queue controls work.
        fast=True selects the YUPI_FAST variant (yupi_fast_api.json: the FastH3
        6-step distill LoRA chained after AfterMidnight, scheduler at 6 steps)."""
        prompt = (self.prompt or "").strip()
        if not prompt:
            self.send_safe(
                chat_id,
                "🌙 YUPI工作流：請先輸入提示詞（按「📝 提示詞」或 /prompt），"
                "再按「🚀 生成影片」。",
            )
            return
        ref_path: Optional[Path] = None
        for candidate in list(self.reference_image_paths or []) + (
            [self.image_path] if self.image_path else []
        ):
            if candidate is not None and Path(candidate).is_file():
                ref_path = Path(candidate)
                break
        if ref_path is None:
            self.send_safe(
                chat_id,
                "🌙 YUPI工作流：請先上傳一張參考圖（直接傳圖片即可），"
                "再按「🚀 生成影片」。",
            )
            return
        try:
            workflow = load_yupi_workflow(fast)
            template_config = yupi_generation_config(workflow)
            requested_total = validate_total_seconds(
                float(
                    getattr(
                        self,
                        "total_seconds",
                        template_config.actual_seconds,
                    )
                )
            )
        except BotError as exc:
            self.send_safe(chat_id, f"🌙 YUPI 工作流設定錯誤：{exc}")
            return
        # Resolution follows the panel selection (self.settings) instead of the
        # template default, so YUPI honours what the user picked. Steps stay as
        # the workflow requires: 20 for YUPI, 6 for the FastH3 distill chain
        # (running more steps than the distill was trained for degrades it).
        selected = getattr(self, "settings", None)
        width = int(getattr(selected, "width", 0) or template_config.width)
        height = int(getattr(selected, "height", 0) or template_config.height)
        if width < 32 or height < 32:
            width, height = template_config.width, template_config.height
        is_long = requested_total > MAX_SEGMENT_SECONDS
        # valid_length() only accepts a single 2-15s shot. A long request is
        # planned into <=15s shots later, so cap this base config at one shot;
        # passing e.g. 60 here would raise "秒數目前只允許 2 到 15 秒".
        base_seconds = (
            min(requested_total, MAX_SEGMENT_SECONDS) if is_long else requested_total
        )
        config = GenerationConfig(
            width=width,
            height=height,
            steps=template_config.steps,
            requested_seconds=base_seconds,
            length=valid_length(base_seconds),
        )
        if is_long:
            # The standalone YUPI graph is intentionally short (the model can
            # only process one small H3 clip safely). Reuse the long-video
            # planner so a selected 60/120/... seconds becomes real clips,
            # then merge them into one MP4 while preserving the YUPI graph.
            self.start_long_generation(
                chat_id,
                config,
                prompt,
                requested_total,
                input_image_path=ref_path,
                reference_image_paths=[ref_path],
                generation_mode=INPUT_MODE_REF2VA,
                task_type=YUPI_TASK_TYPE,
                yupi_fast=fast,
            )
            return
        workflow["6"]["inputs"]["length"] = config.length
        with self.lock:
            if self.job is not None:
                self.send_safe(
                    chat_id,
                    "目前已有工作在生成，請先完成或 /cancel 後再跑 YUPI工作流。",
                )
                return
            job = JobState(
                chat_id=chat_id,
                config=config,
                prompt=prompt,
                started_at=time.time(),
                output_prefix=(
                    YUPI_FAST_OUTPUT_PREFIX if fast else YUPI_OUTPUT_PREFIX
                ),
                total_seconds=config.actual_seconds,
                input_image_path=ref_path,
                reference_image_paths=[ref_path],
                task_type=YUPI_TASK_TYPE,
                generation_mode=INPUT_MODE_REF2VA,
                yupi_fast=fast,
            )
            self.job = job
        self.touch_comfy_activity()
        bot_log("YUPI workflow requested")
        # YUPI used to bypass JobState entirely, so the progress button had no
        # job to display. Create the live message immediately; the refresh
        # thread will replace the queued text once ComfyUI emits progress.
        self._send_progress_message(chat_id)
        thread = threading.Thread(
            target=self._yupi_worker,
            args=(chat_id, prompt, ref_path, job, workflow),
            name="yupi-generation",
            daemon=True,
        )
        thread.start()

    def _ensure_comfyui_for_yupi(self, chat_id: str) -> None:
        """Bring ComfyUI online before the isolated YUPI path touches it.
        Mirrors ensure_comfyui_ready() so YUPI works even when the idle loop
        has already shut ComfyUI down to save VRAM. Reuses the same module
        helpers; still fully separate from the stock job pipeline."""
        if llama_is_online():
            try:
                self.send_safe(
                    chat_id,
                    "🧠 先關閉本地 LLM，釋放顯存避免 OOM。\n" + stop_llama_process(),
                )
            except (BotError, OSError):
                pass
        if comfyui_is_online():
            self.touch_comfy_activity()
            return
        self.send_safe(chat_id, start_comfyui_process(self.comfyui_vram_mode()))
        self.touch_comfy_activity()
        deadline = time.time() + 180
        while time.time() < deadline:
            if comfyui_is_online():
                self.touch_comfy_activity()
                self.send_safe(chat_id, "ComfyUI 已就緒，開始 YUPI 工作流。")
                return
            time.sleep(3)
        raise BotError(f"ComfyUI 在 180 秒內沒有就緒，請查看日誌：{COMFYUI_LOG}")

    def _yupi_worker(
        self,
        chat_id: str,
        prompt: str,
        ref_path: Path,
        job: JobState,
        workflow: dict[str, Any],
    ) -> None:
        """Background worker for the isolated YUPI NSFW workflow."""
        started_at = job.started_at
        progress_tracker: Optional[ComfyProgressTracker] = None
        try:
            if job.cancel_event.is_set():
                raise BotError("生成已取消。")
            self._ensure_comfyui_for_yupi(chat_id)
            ref_name = upload_image_to_comfy(ref_path)
            workflow, lora_name = configure_yupi_workflow(
                workflow,
                job.config,
                prompt,
                ref_name,
                job.output_prefix,
            )
            self.send_safe(
                chat_id,
                ("🌙 YUPI工作流（6步）開始\n" if job.yupi_fast else "🌙 YUPI工作流（20步）開始\n")
                + f"{prompt[:120]}\n"
                + f"(Ref2VA + {lora_name}, euler/beta, {job.config.steps} steps)",
            )
            response = comfy_post(
                "/prompt",
                {"prompt": workflow, "client_id": "telegram-yupi-bot"},
            )
            prompt_id = response.get("prompt_id")
            if not prompt_id:
                raise BotError(f"ComfyUI 沒回傳 prompt_id：{response}")
            job.prompt_id = str(prompt_id)
            with job.progress_lock:
                job.progress_percent = 0.0
                job.progress_node_id = None
                job.progress_node_state = "queued"
                job.progress_node_value = 0.0
                job.progress_node_max = 1.0
                job.progress_node_index = 0
                job.progress_node_total = 0
                job.progress_queue_remaining = None
                job.progress_phase = "waiting"
            progress_tracker = ComfyProgressTracker(
                job, client_id="telegram-yupi-bot"
            )
            job.progress_tracker = progress_tracker
            progress_tracker.start()
            history: Optional[dict[str, Any]] = None
            while True:
                if job.cancel_event.is_set():
                    raise BotError("生成已取消。")
                try:
                    history_all = comfy_post(f"/history/{prompt_id}")
                    history = (
                        history_all.get(str(prompt_id))
                        if isinstance(history_all, dict)
                        else None
                    )
                except BotError:
                    history = None
                if history:
                    status = history.get("status", {})
                    if status.get("status_str") == "error":
                        raise BotError(self.execution_error(history))
                    if status.get("completed") or status.get("status_str") == "success":
                        break
                time.sleep(3)
            video_path = self.find_video(
                history or {},
                started_at,
                name_hint="YUPI_FAST" if job.yupi_fast else "YUPI_NSFW",
            )
            if video_path is None:
                raise BotError(
                    "YUPI 完成但找不到輸出 MP4，請到 ComfyUI output 資料夾查看。"
                )
            with job.progress_lock:
                job.progress_percent = 100.0
                job.progress_phase = "uploading"
                job.progress_node_state = "finished"
            self.telegram.send_video(
                chat_id,
                video_path,
                (
                    "🌙 YUPI工作流（6步）完成\n"
                    if job.yupi_fast
                    else "🌙 YUPI工作流完成\n"
                )
                + prompt[:120],
            )
            self.offer_tail_reference(chat_id, video_path)
            bot_log(f"YUPI workflow finished: {video_path}")
        except BotError as exc:
            if not job.cancel_event.is_set():
                self.send_safe(chat_id, f"🌙 YUPI 工作流失敗：{exc}")
            bot_log(f"YUPI workflow error: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any worker failure
            if not job.cancel_event.is_set():
                self.send_safe(chat_id, f"🌙 YUPI 工作流未預期錯誤：{exc}")
            bot_log(f"YUPI workflow unexpected error: {exc}")
        finally:
            if progress_tracker is not None:
                progress_tracker.stop()
            if job.progress_tracker is progress_tracker:
                job.progress_tracker = None
            self.touch_comfy_activity()
            with self.lock:
                if self.job is job:
                    self.job = None
            self.on_job_finished(chat_id)

    def ensure_comfyui_ready(self, job: JobState) -> None:
        """Hook for a subclass to start or wait for ComfyUI before queuing."""
        return

    def comfyui_vram_mode(self) -> str:
        return DEFAULT_COMFYUI_VRAM_MODE

    def on_job_finished(self, chat_id: str) -> None:
        """Hook for menu bots that have work waiting behind the current job."""
        return

    def cancel_job_for_comfy_control(self) -> bool:
        """Mark the active Telegram job cancelled before stopping ComfyUI."""
        with self.lock:
            job = self.job
            if job is not None:
                job.cancel_event.set()
                job.resume_event.set()
        return job is not None

    def restart_bot(self, chat_id: str) -> None:
        """Cancel any running job, confirm, then schedule a detached self-restart."""
        cancelled = self.cancel_job_for_comfy_control()
        if cancelled:
            try:
                comfy_post("/interrupt", {})
            except BotError:
                pass
        prefix = "目前生成已取消。\n" if cancelled else ""
        self.send_safe(
            chat_id,
            prefix + "正在重啟 Bot… 幾秒後會重新啟動，之後請再按 /start 或 /menu 確認。",
        )
        try:
            restart_bot_process()
        except BotError as exc:
            self.send_safe(chat_id, f"重啟 Bot 失敗：{exc}")
            return
        bot_log("Bot restart requested from Telegram")

    def run_yupi_segment(
        self,
        job: JobState,
        announce: bool = True,
        motion_context: bool = False,
        context_video_name: Optional[str] = None,
        context_latent_path: Optional[str] = None,
        load_latent_clip_index: int = 0,
        save_latent_prefix: Optional[str] = None,
        save_latent_clip_index: Optional[int] = None,
        **_: Any,
    ) -> Path:
        """Run one short YUPI clip for the shared long-video planner."""
        segment_started_at = time.time()
        reference_path: Optional[Path] = None
        if job.segment_index > 1 and job.continuation_image_path is not None:
            if job.continuation_image_path.is_file():
                reference_path = job.continuation_image_path
        if reference_path is None:
            candidates = ([job.input_image_path] if job.input_image_path else [])
            candidates.extend(job.reference_image_paths)
            reference_path = next(
                (path for path in candidates if path.is_file()),
                None,
            )
        if reference_path is None:
            raise BotError("YUPI 長片找不到參考圖片或上一鏡尾幀。")

        reference_name = upload_image_to_comfy(reference_path)
        workflow = load_yupi_workflow(job.yupi_fast)
        workflow, lora_name = configure_yupi_workflow(
            workflow,
            job.config,
            segment_prompt(job),
            reference_name,
            job.output_prefix,
        )
        if motion_context:
            # YUPI now honours Motion Context: the previous segment's AV latent
            # and tail frames are pinned onto this shot's head, then trimmed.
            workflow = attach_yupi_motion_context(
                workflow,
                context_video_name=context_video_name,
                context_latent_path=context_latent_path,
                load_latent_clip_index=load_latent_clip_index,
            )
        # Always write this shot's latent when the planner asked for it: the
        # first shot has no context to load, but it must still produce the file
        # that shot 2 reads (otherwise shot 2 fails with "is neither a file nor
        # a folder").
        workflow = attach_yupi_save_latent(
            workflow,
            save_latent_prefix,
            save_latent_clip_index,
        )
        response = comfy_post(
            "/prompt",
            {"prompt": workflow, "client_id": "telegram-yupi-bot"},
        )
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise BotError(f"ComfyUI 沒有回傳 YUPI prompt_id：{response}")
        job.prompt_id = str(prompt_id)
        with job.progress_lock:
            job.progress_percent = 0.0
            job.progress_node_id = None
            job.progress_node_state = "queued"
            job.progress_node_value = 0.0
            job.progress_node_max = 1.0
            job.progress_node_index = 0
            job.progress_node_total = 0
            job.progress_queue_remaining = None
            job.progress_phase = "waiting"
        progress_tracker = ComfyProgressTracker(
            job,
            client_id="telegram-yupi-bot",
        )
        job.progress_tracker = progress_tracker
        progress_tracker.start()
        if announce:
            self.send_safe(
                job.chat_id,
                ("🌙 YUPI（6步）" if job.yupi_fast else "🌙 YUPI")
                + f" 鏡頭開始：{job.config.actual_seconds:.2f} 秒 | "
                f"{lora_name}\nPrompt ID：{prompt_id}",
            )

        try:
            history: Optional[dict[str, Any]] = None
            while True:
                if job.cancel_event.is_set():
                    raise BotError("生成已取消。")
                try:
                    history_all = comfy_post(f"/history/{prompt_id}")
                    history = (
                        history_all.get(str(prompt_id))
                        if isinstance(history_all, dict)
                        else None
                    )
                except BotError:
                    history = None
                if history:
                    status = history.get("status", {})
                    status_name = status.get("status_str")
                    if status_name == "error":
                        raise BotError(self.execution_error(history))
                    if status.get("completed") or status_name == "success":
                        break
                time.sleep(3)
            video_path = self.find_video(
                history or {},
                segment_started_at,
                name_hint="YUPI_FAST" if job.yupi_fast else "YUPI_NSFW",
            )
            if video_path is None:
                raise BotError("YUPI 鏡頭完成但找不到輸出 MP4。")
            with job.progress_lock:
                job.progress_percent = 100.0
                job.progress_phase = "completed"
                job.progress_node_state = "finished"
            return video_path
        finally:
            progress_tracker.stop()
            if job.progress_tracker is progress_tracker:
                job.progress_tracker = None

    def run_segment(
        self,
        job: JobState,
        announce: bool = True,
        motion_context: bool = False,
        context_video_name: Optional[str] = None,
        context_latent_path: Optional[str] = None,
        load_latent_clip_index: int = 0,
        save_latent_prefix: Optional[str] = None,
        save_latent_clip_index: Optional[int] = None,
    ) -> Path:
        if job.task_type == YUPI_TASK_TYPE:
            return self.run_yupi_segment(
                job,
                announce=announce,
                motion_context=motion_context,
                context_video_name=context_video_name,
                context_latent_path=context_latent_path,
                load_latent_clip_index=load_latent_clip_index,
                save_latent_prefix=save_latent_prefix,
                save_latent_clip_index=save_latent_clip_index,
            )
        segment_started_at = time.time()
        image_name: Optional[str] = None
        last_image_name: Optional[str] = None
        # Ref2VA is useful for locking the opening identity, but carrying the
        # same reference images into every later shot can pull the model back
        # to the same pose.  After the first shot, switch to ordinary I2VA and
        # use only the immediately previous tail frame as its first frame.
        use_ref2va_tail_i2va = (
            not motion_context
            and job.generation_mode == INPUT_MODE_REF2VA
            and job.segment_index > 1
            and job.continuation_image_path is not None
        )
        if (
            not motion_context
            and job.generation_mode in {INPUT_MODE_IMAGE, INPUT_MODE_FL2VA}
            and job.input_image_path is not None
            and job.segment_index == 1
        ):
            if job.comfy_image_name is None:
                job.comfy_image_name = upload_image_to_comfy(job.input_image_path)
            image_name = job.comfy_image_name
            if (
                job.generation_mode == INPUT_MODE_FL2VA
                and job.last_image_path is not None
            ):
                if job.comfy_last_image_name is None:
                    job.comfy_last_image_name = upload_image_to_comfy(job.last_image_path)
                last_image_name = job.comfy_last_image_name
        elif (
            not motion_context
            and job.generation_mode in {INPUT_MODE_IMAGE, INPUT_MODE_FL2VA}
            and job.segment_index > 1
            and job.continuation_image_path is not None
        ):
            image_name = upload_image_to_comfy(job.continuation_image_path)
        elif (
            use_ref2va_tail_i2va
        ):
            image_name = upload_image_to_comfy(job.continuation_image_path)
        ref2va_reference_segment = (
            not motion_context
            and job.generation_mode == INPUT_MODE_REF2VA
            and not use_ref2va_tail_i2va
        )
        if ref2va_reference_segment:
            if not job.comfy_reference_image_names:
                job.comfy_reference_image_names = [
                    upload_image_to_comfy(path)
                    for path in job.reference_image_paths
                    if path.is_file()
                ]
            if not job.comfy_reference_video_names:
                job.comfy_reference_video_names = [
                    upload_video_to_comfy(path)
                    for path in job.reference_video_paths
                    if path.is_file()
                ]
            if not job.comfy_reference_audio_names:
                job.comfy_reference_audio_names = [
                    upload_audio_to_comfy(path)
                    for path in job.reference_audio_paths
                    if path.is_file()
                ]
            if not (
                job.comfy_reference_image_names
                or job.comfy_reference_video_names
                or job.comfy_reference_audio_names
            ):
                raise BotError("Ref2VA 參考素材不存在，請重新上傳圖片、影片或音訊。")
        workflow_mode = job.generation_mode
        if use_ref2va_tail_i2va:
            workflow_mode = INPUT_MODE_IMAGE
        if (
            workflow_mode == INPUT_MODE_FL2VA
            and job.segment_index > 1
            and not last_image_name
        ):
            workflow_mode = INPUT_MODE_IMAGE
        workflow = build_workflow(
            job.config,
            segment_prompt(job, motion_context=motion_context),
            job.output_prefix,
            image_name=image_name,
            last_image_name=last_image_name,
            reference_image_names=(
                job.comfy_reference_image_names
                if workflow_mode == INPUT_MODE_REF2VA
                else []
            ),
            reference_video_names=(
                job.comfy_reference_video_names
                if workflow_mode == INPUT_MODE_REF2VA
                else []
            ),
            reference_audio_names=(
                job.comfy_reference_audio_names
                if workflow_mode == INPUT_MODE_REF2VA
                else []
            ),
            audio_reference_name=(None if motion_context else job.audio_reference_name),
            generation_mode=workflow_mode,
            motion_context=motion_context,
            context_video_name=context_video_name,
            context_latent_path=context_latent_path,
            load_latent_clip_index=load_latent_clip_index,
            save_latent_prefix=save_latent_prefix,
            save_latent_clip_index=save_latent_clip_index,
            latent_upscale=bool(getattr(self, "latent_upscale", LATENT_UPSCALE_ENABLED)),
            h3_profile=str(
                getattr(self, "h3_profile", H3_PROFILE_DEFAULT)
            ),
        )
        usage_report = workflow_usage_report(workflow, self.comfyui_vram_mode())
        if usage_report not in job.workflow_reports:
            job.workflow_reports.append(usage_report)
        response = comfy_post(
            "/prompt",
            {"prompt": workflow, "client_id": "telegram-turbo-bot"},
        )
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise BotError(f"ComfyUI 沒有回傳 prompt_id：{response}")
        job.prompt_id = str(prompt_id)
        with job.progress_lock:
            job.progress_percent = 0.0
            job.progress_node_id = None
            job.progress_node_state = "queued"
            job.progress_node_value = 0.0
            job.progress_node_max = 1.0
            job.progress_node_index = 0
            job.progress_node_total = 0
            job.progress_queue_remaining = None
            job.progress_phase = "waiting"
        progress_tracker = ComfyProgressTracker(job)
        job.progress_tracker = progress_tracker
        progress_tracker.start()
        if announce:
            step_label = f"{job.config.steps} steps"
            self.send_safe(
                job.chat_id,
                f"已開始生成：{job.config.width}×{job.config.height} | "
                f"{step_label} | 約 {job.config.actual_seconds:.2f} 秒\n"
                f"Prompt ID: {prompt_id}",
            )

        try:
            history: Optional[dict[str, Any]] = None
            while True:
                if job.cancel_event.is_set():
                    raise BotError("生成已取消。")
                try:
                    history_all = comfy_post(f"/history/{prompt_id}")
                    history = (
                        history_all.get(str(prompt_id))
                        if isinstance(history_all, dict)
                        else None
                    )
                except BotError:
                    history = None
                if history:
                    status = history.get("status", {})
                    status_name = status.get("status_str")
                    if status_name == "error":
                        raise BotError(self.execution_error(history))
                    if status.get("completed") or status_name == "success":
                        break
                time.sleep(3)

            video_path = self.find_video(history or {}, segment_started_at)
            if video_path is None:
                raise BotError("生成完成，但找不到输出 MP4。请在 ComfyUI output 資料夾查看。")
            with job.progress_lock:
                job.progress_percent = 100.0
                job.progress_phase = "completed"
                job.progress_node_state = "finished"
            return video_path
        finally:
            progress_tracker.stop()
            if job.progress_tracker is progress_tracker:
                job.progress_tracker = None

    def _run_segment_with_oom_fallback(self, job: JobState) -> Path:
        """Run a single-shot segment, auto-downgrading resolution on CUDA OOM.

        The long-video loop already retries at a lower resolution when a
        segment hits CUDA OOM; single-shot jobs (including I2VA, whose
        first-frame VAE encode is the most VRAM-hungry mode) had no such
        fallback and failed hard.  Mirror the long-video logic here.
        """
        while True:
            try:
                return self.run_segment(job, announce=True)
            except Exception as exc:
                if job.cancel_event.is_set() or not is_cuda_oom_error(exc):
                    raise
                next_resolution = next_lower_resolution(
                    job.config.width, job.config.height
                )
                if next_resolution is None:
                    self.send_safe(
                        job.chat_id,
                        f"最低解析度 {resolution_label(job.config.width, job.config.height)} "
                        "仍然顯存不足，無法繼續。",
                    )
                    raise
                old_label = resolution_label(job.config.width, job.config.height)
                new_label = resolution_label(*next_resolution)
                job.resolution_fallbacks.append(f"{old_label} → {new_label}")
                job.config = parse_config(
                    [
                        str(next_resolution[0]),
                        str(next_resolution[1]),
                        str(job.config.steps),
                        str(job.config.requested_seconds),
                    ]
                )
                bot_log(f"single-shot OOM at {old_label}; retrying at {new_label}: {exc}")
                try:
                    comfy_post("/free", {"unload_models": True, "free_memory": True})
                except BotError as free_exc:
                    bot_log(f"OOM memory release before retry unavailable: {free_exc}")
                self.send_safe(
                    job.chat_id,
                    f"⚠️ 顯存不足，自動降級：{old_label} → {new_label}\n正在以較低解析度重試。",
                )

    def run_job(self, job: JobState) -> None:
        bot_log(
            f"job start {job.config.width}x{job.config.height} "
            f"steps={job.config.steps} {job.config.actual_seconds:.2f}s"
        )
        self.last_job_video_path = None
        try:
            self.ensure_comfyui_ready(job)
            video_path = self._run_segment_with_oom_fallback(job)
            with job.progress_lock:
                job.progress_phase = "uploading"
            model_label = "MiniMax H3 Turbo 完成"
            caption = (
                f"{model_label}\n{job.config.width}×{job.config.height} | "
                f"{job.config.steps} steps | {job.config.actual_seconds:.2f} 秒"
            )
            self.telegram.send_video(job.chat_id, video_path, caption)
            self.last_job_video_path = video_path
            if not self.chain_active():
                self.offer_tail_reference(job.chat_id, video_path)
            self.send_safe(
                job.chat_id,
                completion_report(job, time.time() - job.started_at),
            )
            self.offer_upscale(
                job.chat_id,
                video_path,
                job.config.width,
                job.config.height,
                job.config.actual_seconds,
            )
            bot_log(f"job done {video_path}")
        except Exception as exc:  # keep the long-polling bot alive after one job fails
            if not job.cancel_event.is_set():
                self.send_safe(job.chat_id, f"生成失败：{exc}")
            bot_log(f"job error: {exc}")
            print(f"generation error: {exc}", flush=True)
        finally:
            self.touch_comfy_activity()
            with self.lock:
                if self.job is job:
                    self.job = None
            self.on_job_finished(job.chat_id)

    @staticmethod
    def execution_error(history: dict[str, Any]) -> str:
        for message in history.get("status", {}).get("messages", []):
            if isinstance(message, list) and message and message[0] == "execution_error":
                details = message[1] if len(message) > 1 else {}
                return str(details.get("exception_message", "ComfyUI execution error"))
        return "ComfyUI execution error"

    @staticmethod
    def find_video(
        history: dict[str, Any],
        started_at: float,
        name_hint: str = "Telegram_Turbo",
    ) -> Optional[Path]:
        candidates: list[Path] = []
        for node_output in history.get("outputs", {}).values():
            if not isinstance(node_output, dict):
                continue
            # ComfyUI's SaveVideo node currently reports MP4 entries under
            # `images` (with `animated: true`), while other video nodes may
            # use `gifs`, `videos`, or `files`.  Accept all of them so
            # produced videos are delivered instead of being reported missing.
            for key in ("images", "gifs", "videos", "files"):
                for item in node_output.get(key, []) or []:
                    if not isinstance(item, dict) or not item.get("filename"):
                        continue
                    path = OUTPUT_DIR / str(item.get("subfolder", "")) / str(item["filename"])
                    if path.is_file():
                        candidates.append(path)

        if OUTPUT_DIR.is_dir():
            for path in OUTPUT_DIR.rglob("*.mp4"):
                try:
                    if path.stat().st_mtime >= started_at - 5 and name_hint.lower() in path.name.lower():
                        candidates.append(path)
                except OSError:
                    continue

        unique: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                unique.append(path)
        audio = [path for path in unique if path.name.lower().endswith("-audio.mp4")]
        return (audio or unique)[0] if (audio or unique) else None

    def run(self) -> None:
        self.send_safe(
            self.allowed_chat_id,
            "MiniMax H3 Turbo Telegram 控制器已啟動。輸入 /help 查看用法。",
        )
        while True:
            try:
                updates = self.telegram.get_updates(self.offset)
                for update in updates:
                    self.offset = int(update["update_id"]) + 1
                    message = update.get("message")
                    if message:
                        self.handle_message(message)
                    callback = update.get("callback_query")
                    if callback:
                        self.handle_callback(callback)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"polling error: {exc}", flush=True)
                time.sleep(5)


class TelegramMenuBot(TelegramTurboBot):
    """Button-driven Telegram UI with persistent generation settings."""

    RESOLUTIONS = RESOLUTION_LADDER
    SECONDS = (5, 10, 12, 15)
    LONG_SECONDS = (30, 60, 120, 180, 300, 600, 900, 1200, 1800)
    STEPS = (8, 12, 16)

    def __init__(self, token: str, allowed_chat_id: str):
        super().__init__(token, allowed_chat_id)
        self.settings = self.load_settings()
        self.total_seconds = self.load_saved_total_seconds()
        self.prompt = self.load_saved_prompt()
        self.input_mode = self.load_saved_mode()
        self.model_mode = self.load_saved_model_mode()
        self.image_path = self.load_saved_image_path()
        saved_media = self._load_saved_media_paths
        saved_last = saved_media("last_image_paths")
        self.last_image_path = saved_last[0] if saved_last else None
        self.reference_image_paths = saved_media("reference_image_paths")
        self.reference_video_paths = saved_media("reference_video_paths")
        self.reference_audio_paths = saved_media("reference_audio_paths")
        self.vram_mode = self.load_saved_vram_mode()
        self.latent_upscale = self.load_saved_latent_upscale()
        self.long_continuity = self.load_saved_long_continuity()
        self.h3_profile = self.load_saved_h3_profile()
        self.restart_llm_after_generation = self.load_saved_restart_llm()
        self.script_lang = self.load_saved_script_lang()
        self.script_continuity = self.load_saved_script_continuity()
        self.continuity_source_script = self.load_saved_continuity_source()
        self.script_llm = self.load_saved_script_llm()
        set_script_llm_provider(self.script_llm)
        self.script_template = self.load_saved_script_template()
        set_script_template(self.script_template)
        self.shutdown_after_generation = self.load_saved_shutdown_after_generation()
        self._shutdown_pending = False
        self.awaiting_prompt = False
        self.awaiting_duration = False
        self.awaiting_extension_duration = False
        self.awaiting_extension_prompt = False
        self.awaiting_queue_prompt = False
        self.extension_seconds: Optional[float] = None
        self.extension_checkpoint_id: Optional[str] = None
        # ✨ script generator state
        self.awaiting_script_idea = False
        self.script_busy = False
        self.script_draft: Optional[str] = None
        # 🔗 auto-chain state (short clips chained by tail-frame handoff)
        self.chain_remaining = 0
        self.chain_total = 0
        self.chain_durations: list[float] = []
        self.chain_idea = ""
        self.chain_done_paths: list[Path] = []
        self.chain_current_script = ""
        self.chain_awaiting_idea = False
        self.chain_pending_durations: list[float] = []
        self.awaiting_chain_setup = False
        self.last_job_video_path: Optional[Path] = None
        self.script_idea = ""
        self.script_seconds = float(self.total_seconds)
        self.awaiting_custom_prompt = ""
        # Multi-round draft editing: script_history holds every previous revision
        # so the user can step back after any number of manual or AI edits.
        self.script_history: list[str] = []
        self.script_last_action = ""
        self.awaiting_script_edit = ""
        self.awaiting_script_refine = False
        self.script_refine_instruction = ""
        self.story_queue: list[QueuedStory] = self.load_story_queue()
        self._queue_starting = False
        self.menu_message_id: Optional[int] = None
        self.menu_section = MENU_MAIN
        self.control_keyboard_sent = False
        # Restore the saved prompt's duration from its own script on startup,
        # so a restart cannot silently bring back the previous manual value.
        self.auto_detect_prompt_duration(self.prompt, persist=True)

    @staticmethod
    def default_settings() -> GenerationConfig:
        return parse_config(["1344", "768", "4", "5"])

    def load_settings(self) -> GenerationConfig:
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            return parse_config(
                [
                    str(saved["width"]),
                    str(saved["height"]),
                    str(saved["steps"]),
                    str(saved["seconds"]),
                ]
            )
        except (OSError, ValueError, KeyError, TypeError, BotError, json.JSONDecodeError):
            return self.default_settings()

    @staticmethod
    def load_saved_prompt() -> str:
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            prompt = saved.get("prompt", "")
            return str(prompt).strip() if prompt else ""
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return ""

    @staticmethod
    def load_saved_mode() -> str:
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            return normalize_input_mode(saved.get("input_mode", INPUT_MODE_TEXT))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return "text"

    @staticmethod
    def load_saved_model_mode() -> str:
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            return normalize_model_mode(saved.get("model_mode"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return MODEL_H3

    @staticmethod
    def load_saved_vram_mode() -> str:
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            return normalize_comfyui_vram_mode(saved.get("comfy_vram_mode"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return DEFAULT_COMFYUI_VRAM_MODE

    @staticmethod
    def load_saved_latent_upscale() -> bool:
        """Two-stage latent upscaling preference; env default on first run."""
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = saved.get("latent_upscale")
            if isinstance(value, bool):
                return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return LATENT_UPSCALE_ENABLED

    @staticmethod
    def load_saved_long_continuity() -> str:
        """Long-video continuation mode: 'motion_context' or 'tail_frame'.

        The saved preference wins; the MINIMAX_H3_LONG_CONTINUITY env var is
        only the first-run default.
        """
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = saved.get("long_continuity")
            if isinstance(value, str) and value.strip().lower() in LONG_CONTINUITY_MODES:
                return value.strip().lower()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return (
            "motion_context"
            if LONG_CONTINUITY_MODE in {"motion_context", "motion", "experimental"}
            else "tail_frame"
        )

    @staticmethod
    def load_saved_h3_profile() -> str:
        """Stock-path model profile: 'fused' (default) or 'classic' (fallback).

        The fused bake is the standard now; classic stays selectable in case a
        specific clip needs the split FL2VA/Ref2VA checkpoints + LoRA.
        """
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = saved.get("h3_profile")
            if isinstance(value, str) and value.strip().lower() in H3_PROFILES:
                return value.strip().lower()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return H3_PROFILE_FUSED

    @staticmethod
    def load_saved_restart_llm() -> bool:
        """Whether to start the local LLM again after each generation."""
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = saved.get("restart_llm_after_generation")
            if isinstance(value, bool):
                return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return RESTART_LLM_AFTER_GENERATION

    @staticmethod
    def load_saved_script_lang() -> str:
        """Output language for generated scripts (zh = Simplified Chinese)."""
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = saved.get("script_lang")
            if isinstance(value, str) and value.strip().lower() in SCRIPT_LANGS:
                return value.strip().lower()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return SCRIPT_LANG_DEFAULT

    @staticmethod
    def load_saved_script_continuity() -> str:
        """Continuity block injected into the writer prompt, when one is active."""
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = saved.get("script_continuity")
            if isinstance(value, str):
                return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return ""

    @staticmethod
    def load_saved_continuity_source() -> str:
        """The script of the last generated clip (fed to the writer on a handoff)."""
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = saved.get("continuity_source_script")
            if isinstance(value, str):
                return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return ""

    @staticmethod
    def load_saved_script_llm() -> str:
        """Which engine writes the scripts: local llama.cpp or Command Code."""
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = saved.get("script_llm")
            if isinstance(value, str):
                return normalize_script_llm(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return SCRIPT_LLM_DEFAULT

    @staticmethod
    def load_saved_script_template() -> str:
        """Which guidance template the writer uses: adult or general."""
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = saved.get("script_template")
            if isinstance(value, str):
                return normalize_script_template(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return SCRIPT_TEMPLATE_DEFAULT

    @staticmethod
    def load_saved_shutdown_after_generation() -> bool:
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = saved.get("shutdown_after_generation", False)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return value is True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def comfyui_vram_mode(self) -> str:
        return normalize_comfyui_vram_mode(self.vram_mode)

    @staticmethod
    def load_saved_image_path() -> Optional[Path]:
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = str(saved.get("image_path", "")).strip()
            path = Path(value) if value else None
            if not path or not path.is_file():
                return None
            allowed_roots = (IMAGE_DIR.resolve(), REFERENCE_DIR.resolve())
            if not any(
                path.resolve().is_relative_to(root) for root in allowed_roots
            ):
                return None
            return path
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _load_saved_media_paths(key: str) -> list[Path]:
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return []
        raw_paths = saved.get(key, [])
        if not isinstance(raw_paths, list):
            raw_paths = [raw_paths]
        result: list[Path] = []
        try:
            reference_root = REFERENCE_DIR.resolve()
        except OSError:
            reference_root = REFERENCE_DIR.absolute()
        for raw_path in raw_paths:
            candidate = Path(str(raw_path).strip()) if raw_path else None
            if candidate is None:
                continue
            try:
                candidate.resolve().relative_to(reference_root)
            except (OSError, ValueError):
                continue
            if candidate.is_file():
                result.append(candidate)
        return result

    def load_saved_total_seconds(self) -> float:
        try:
            with STATE_PATH.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            value = float(saved.get("total_seconds", saved.get("seconds", 15)))
            return validate_total_seconds(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        except BotError:
            pass
        return float(self.settings.requested_seconds)

    def load_story_queue(self) -> list[QueuedStory]:
        try:
            with QUEUE_STATE_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return []
        if not isinstance(payload, dict) or int(payload.get("version", 0)) != QUEUE_STATE_VERSION:
            return []
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            return []
        items: list[QueuedStory] = []
        seen_ids: set[str] = set()
        for raw in raw_items[:MAX_QUEUE_ITEMS]:
            if not isinstance(raw, dict):
                continue
            prompt = str(raw.get("prompt", "")).strip()
            item_id = str(raw.get("item_id", "")).strip()
            if not prompt or not re.fullmatch(r"q_[A-Za-z0-9]{6,24}", item_id):
                continue
            if item_id in seen_ids:
                continue
            try:
                total_seconds = validate_total_seconds(float(raw["total_seconds"]))
                raw_config = raw.get("config", {})
                if not isinstance(raw_config, dict):
                    continue
                config = parse_config(
                    [
                        str(raw_config["width"]),
                        str(raw_config["height"]),
                        str(raw_config["steps"]),
                        str(min(total_seconds, MAX_SEGMENT_SECONDS)),
                    ]
                )
            except (BotError, KeyError, TypeError, ValueError):
                continue
            input_image_path: Optional[Path] = None
            raw_image_path = str(raw.get("input_image_path", "")).strip()
            if raw_image_path:
                candidate = Path(raw_image_path)
                try:
                    allowed_roots = (IMAGE_DIR.resolve(), REFERENCE_DIR.resolve())
                    if not any(
                        candidate.resolve().is_relative_to(root)
                        for root in allowed_roots
                    ):
                        raise ValueError
                except (OSError, ValueError):
                    candidate = None
                if candidate is not None and candidate.is_file():
                    input_image_path = candidate
            last_paths = self._checkpoint_media_paths(raw, "last_image_paths")
            try:
                created_at = float(raw.get("created_at", time.time()))
            except (TypeError, ValueError):
                created_at = time.time()
            items.append(
                QueuedStory(
                    item_id=item_id,
                    prompt=prompt,
                    config=config,
                    total_seconds=total_seconds,
                    input_image_path=input_image_path,
                    last_image_path=last_paths[0] if last_paths else None,
                    reference_image_paths=tuple(
                        self._checkpoint_media_paths(raw, "reference_image_paths")
                    ),
                    reference_video_paths=tuple(
                        self._checkpoint_media_paths(raw, "reference_video_paths")
                    ),
                    reference_audio_paths=tuple(
                        self._checkpoint_media_paths(raw, "reference_audio_paths")
                    ),
                    generation_mode=normalize_input_mode(
                        raw.get("generation_mode", INPUT_MODE_TEXT)
                    ),
                    model_mode=normalize_model_mode(raw.get("model_mode")),
                    created_at=created_at,
                )
            )
            seen_ids.add(item_id)
        return items

    def save_story_queue(self) -> None:
        with self.lock:
            items = list(self.story_queue)
        payload = {
            "version": QUEUE_STATE_VERSION,
            "updated_at": time.time(),
            "items": [
                {
                    "item_id": item.item_id,
                    "prompt": item.prompt,
                    "total_seconds": item.total_seconds,
                    "created_at": item.created_at,
                    "config": {
                        "width": item.config.width,
                        "height": item.config.height,
                        "steps": item.config.steps,
                    },
                    "input_image_path": (
                        str(item.input_image_path)
                        if item.input_image_path is not None
                        else ""
                    ),
                    "last_image_paths": (
                        [str(item.last_image_path)]
                        if item.last_image_path is not None
                        else []
                    ),
                    "reference_image_paths": [str(path) for path in item.reference_image_paths],
                    "reference_video_paths": [str(path) for path in item.reference_video_paths],
                    "reference_audio_paths": [str(path) for path in item.reference_audio_paths],
                    "generation_mode": normalize_input_mode(item.generation_mode),
                    "model_mode": normalize_model_mode(item.model_mode),
                }
                for item in items
            ],
        }
        QUEUE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = QUEUE_STATE_PATH.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(QUEUE_STATE_PATH)

    def save_settings(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = STATE_PATH.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "width": self.settings.width,
                    "height": self.settings.height,
                    "steps": self.settings.steps,
                    "seconds": self.settings.requested_seconds,
                    "total_seconds": getattr(
                        self, "total_seconds", self.settings.requested_seconds
                    ),
                    "prompt": getattr(self, "prompt", ""),
                    "input_mode": getattr(self, "input_mode", "text"),
                    "model_mode": normalize_model_mode(
                        getattr(self, "model_mode", MODEL_H3)
                    ),
                    "comfy_vram_mode": normalize_comfyui_vram_mode(
                        getattr(self, "vram_mode", DEFAULT_COMFYUI_VRAM_MODE)
                    ),
                    "latent_upscale": bool(
                        getattr(self, "latent_upscale", LATENT_UPSCALE_ENABLED)
                    ),
                    "long_continuity": str(
                        getattr(self, "long_continuity", "motion_context")
                    ),
                    "h3_profile": str(
                        getattr(self, "h3_profile", H3_PROFILE_DEFAULT)
                    ),
                    "restart_llm_after_generation": bool(
                        getattr(
                            self,
                            "restart_llm_after_generation",
                            RESTART_LLM_AFTER_GENERATION,
                        )
                    ),
                    "script_lang": normalize_script_lang(
                        getattr(self, "script_lang", SCRIPT_LANG_DEFAULT)
                    ),
                    "script_continuity": str(
                        getattr(self, "script_continuity", "")
                    ),
                    "continuity_source_script": str(
                        getattr(self, "continuity_source_script", "")
                    ),
                    "script_llm": normalize_script_llm(
                        getattr(self, "script_llm", SCRIPT_LLM_DEFAULT)
                    ),
                    "script_template": normalize_script_template(
                        getattr(self, "script_template", SCRIPT_TEMPLATE_DEFAULT)
                    ),
                    "shutdown_after_generation": bool(
                        getattr(self, "shutdown_after_generation", False)
                    ),
                    "image_path": (
                        str(getattr(self, "image_path", ""))
                        if getattr(self, "image_path", None)
                        else ""
                    ),
                    "last_image_paths": (
                        [str(getattr(self, "last_image_path"))]
                        if getattr(self, "last_image_path", None)
                        else []
                    ),
                    "reference_image_paths": [
                        str(path)
                        for path in getattr(self, "reference_image_paths", [])
                        if path.is_file()
                    ],
                    "reference_video_paths": [
                        str(path)
                        for path in getattr(self, "reference_video_paths", [])
                        if path.is_file()
                    ],
                    "reference_audio_paths": [
                        str(path)
                        for path in getattr(self, "reference_audio_paths", [])
                        if path.is_file()
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(STATE_PATH)

    @staticmethod
    def _checkpoint_id(path: Path) -> str:
        return path.stem

    @staticmethod
    def _write_checkpoint_payload(path: Path, payload: dict[str, Any]) -> None:
        LONG_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    @staticmethod
    def _read_checkpoint_payload(path: Path) -> Optional[dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if int(payload.get("version", 0)) != LONG_CHECKPOINT_VERSION:
            return None
        return payload

    @staticmethod
    def _checkpoint_config(payload: dict[str, Any]) -> GenerationConfig:
        config = payload.get("config")
        if not isinstance(config, dict):
            raise BotError("長片檢查點缺少生成配置。")
        return parse_config(
            [
                str(config["width"]),
                str(config["height"]),
                str(config["steps"]),
                str(config.get("requested_seconds", config.get("seconds", 15))),
            ]
        )

    @staticmethod
    def _checkpoint_long_resolution(
        payload: dict[str, Any],
        base_config: GenerationConfig,
    ) -> tuple[int, int]:
        raw_resolution = payload.get("next_resolution")
        if isinstance(raw_resolution, dict):
            try:
                resolution = parse_config(
                    [
                        str(raw_resolution["width"]),
                        str(raw_resolution["height"]),
                        str(base_config.steps),
                        str(base_config.requested_seconds),
                    ]
                )
                return resolution.width, resolution.height
            except (BotError, KeyError, TypeError, ValueError):
                pass
        return base_config.width, base_config.height

    @staticmethod
    def _checkpoint_resolution_fallbacks(payload: dict[str, Any]) -> list[str]:
        raw_fallbacks = payload.get("resolution_fallbacks", [])
        if not isinstance(raw_fallbacks, list):
            return []
        return [str(item).strip() for item in raw_fallbacks[:50] if str(item).strip()]

    @staticmethod
    def _checkpoint_shots(payload: dict[str, Any]) -> tuple[ShotSpec, ...]:
        raw_shots = payload.get("shot_plan")
        if not isinstance(raw_shots, list) or not raw_shots:
            raise BotError("長片檢查點缺少鏡頭時間軸。")
        shots: list[ShotSpec] = []
        for raw in raw_shots:
            if not isinstance(raw, dict):
                raise BotError("長片檢查點的鏡頭資料無效。")
            shots.append(
                ShotSpec(
                    float(raw["start_seconds"]),
                    float(raw["end_seconds"]),
                    str(raw.get("label", "鏡頭")),
                    str(raw["action"]),
                )
            )
        return tuple(shots)

    @staticmethod
    def _checkpoint_video_paths(payload: dict[str, Any]) -> list[Path]:
        raw_paths = payload.get("completed_video_paths", [])
        if not isinstance(raw_paths, list):
            raise BotError("長片檢查點的影片清單無效。")
        paths: list[Path] = []
        for raw_path in raw_paths:
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = OUTPUT_DIR / path
            paths.append(path)
        return paths

    @staticmethod
    def _checkpoint_media_paths(payload: dict[str, Any], key: str) -> list[Path]:
        raw_paths = payload.get(key, [])
        if not isinstance(raw_paths, list):
            raw_paths = [raw_paths]
        try:
            root = REFERENCE_DIR.resolve()
        except OSError:
            root = REFERENCE_DIR.absolute()
        paths: list[Path] = []
        for raw_path in raw_paths:
            candidate = Path(str(raw_path))
            try:
                candidate.resolve().relative_to(root)
            except (OSError, ValueError):
                continue
            if candidate.is_file():
                paths.append(candidate)
        return paths

    def save_long_checkpoint(
        self,
        job: JobState,
        video_paths: list[Path],
        next_segment_index: int,
        motion_context_enabled: bool,
        latent_prefix: Optional[str],
        context_latent_path: Optional[str],
        status: str = "running",
        error: str = "",
    ) -> Path:
        """Persist enough state to restart a failed long video without replaying shots."""
        if job.checkpoint_path is None:
            base_prefix = job.long_base_prefix or job.output_prefix
            checkpoint_name = base_prefix.rsplit("/", 1)[-1]
            job.checkpoint_path = LONG_CHECKPOINT_DIR / f"{checkpoint_name}.json"
        base_config = job.base_config or job.config
        job.completed_video_paths = list(video_paths)
        job.resume_from_segment = int(next_segment_index)
        completed_count = len(video_paths)
        completed_seconds = 0.0
        if job.shot_plan:
            completed_seconds = sum(
                shot.duration for shot in job.shot_plan[:completed_count]
            )
        payload = {
            "version": LONG_CHECKPOINT_VERSION,
            "checkpoint_id": self._checkpoint_id(job.checkpoint_path),
            "chat_id": str(job.chat_id),
            "task_type": normalize_task_type(job.task_type),
            "generation_mode": normalize_input_mode(job.generation_mode),
            "yupi_fast": bool(getattr(job, "yupi_fast", False)),
            "status": status,
            "last_error": error[-4000:] if error else "",
            "created_at": float(job.started_at),
            "updated_at": time.time(),
            "prompt": job.prompt,
            "config": {
                "width": base_config.width,
                "height": base_config.height,
                "steps": base_config.steps,
                "requested_seconds": base_config.requested_seconds,
                "length": base_config.length,
            },
            "output_prefix": job.long_base_prefix or job.output_prefix,
            "total_seconds": float(job.total_seconds),
            "segment_total": int(job.segment_total),
            "next_segment_index": int(next_segment_index),
            "completed_seconds": round(completed_seconds, 3),
            "shot_plan": [
                {
                    "start_seconds": shot.start_seconds,
                    "end_seconds": shot.end_seconds,
                    "label": shot.label,
                    "action": shot.action,
                }
                for shot in job.shot_plan
            ],
            "story_global_text": job.story_global_text,
            "input_image_path": str(job.input_image_path or ""),
            "last_image_path": str(job.last_image_path or ""),
            "reference_image_paths": [str(path) for path in job.reference_image_paths],
            "reference_video_paths": [str(path) for path in job.reference_video_paths],
            "reference_audio_paths": [str(path) for path in job.reference_audio_paths],
            "completed_video_paths": [
                str(path.resolve()) for path in video_paths if path.is_file()
            ],
            "motion_context_enabled": bool(motion_context_enabled),
            "latent_prefix": latent_prefix or "",
            "last_context_latent_path": context_latent_path or "",
            "next_resolution": {
                "width": (job.long_resolution or (base_config.width, base_config.height))[0],
                "height": (job.long_resolution or (base_config.width, base_config.height))[1],
            },
            "resolution_fallbacks": list(job.resolution_fallbacks),
        }
        self._write_checkpoint_payload(job.checkpoint_path, payload)
        return job.checkpoint_path

    def mark_long_checkpoint(
        self, job: JobState, status: str, error: str = ""
    ) -> None:
        """Change only the lifecycle state after a checkpoint was saved."""
        path = job.checkpoint_path
        if path is None or not path.is_file():
            return
        payload = self._read_checkpoint_payload(path)
        if payload is None:
            return
        payload["status"] = status
        payload["last_error"] = error[-4000:] if error else ""
        payload["updated_at"] = time.time()
        try:
            self._write_checkpoint_payload(path, payload)
        except OSError as exc:
            bot_log(f"checkpoint status update failed: {exc}")

    def checkpoint_for_id(self, checkpoint_id: str) -> Optional[tuple[Path, dict[str, Any]]]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", checkpoint_id):
            return None
        path = LONG_CHECKPOINT_DIR / f"{checkpoint_id}.json"
        payload = self._read_checkpoint_payload(path)
        if payload is None or str(payload.get("chat_id", "")) != self.allowed_chat_id:
            return None
        return path, payload

    # Segment MP4 layouts: the official-core SaveVideo node saves into a
    # per-segment subfolder with a trailing underscore (the prefix is split by
    # folder_paths.get_save_image_path and nodes_video.py appends "_NNNNN_"):
    # long_xxx/segment_01/segment_01_00001_.mp4.  The retired T8 package wrote
    # files straight into the batch directory with no underscore:
    # long_xxx/segment_01_00001.mp4.  Accept both; the T8-era "-audio" sidecar
    # files never match the strict pattern below.
    _SEGMENT_VIDEO_RE = re.compile(r"segment_(\d+)_00001_?\.mp4")

    @classmethod
    def _long_segment_video_paths(cls, directory: Path) -> dict[int, Path]:
        """Map segment index -> MP4 path inside one long-video batch directory."""
        paths: dict[int, Path] = {}
        patterns = (
            "segment_*_00001*.mp4",
            "segment_*/segment_*_00001*.mp4",
        )
        for pattern in patterns:
            for path in directory.glob(pattern):
                if not path.is_file():
                    continue
                match = cls._SEGMENT_VIDEO_RE.fullmatch(path.name)
                if match is None:
                    continue
                paths.setdefault(int(match.group(1)), path)
        return paths

    def discover_legacy_checkpoint(self) -> Optional[Path]:
        """Create a one-time checkpoint for an older run that predates persistence."""
        if self.total_seconds <= MAX_SEGMENT_SECONDS or not self.prompt:
            return None
        try:
            plan = build_long_video_plan(self.prompt, self.total_seconds)
        except BotError:
            return None
        root = OUTPUT_DIR / OUTPUT_PREFIX
        if not root.is_dir():
            return None
        candidates: list[tuple[float, Path, dict[int, Path]]] = []
        for directory in root.glob("long_*"):
            if not directory.is_dir():
                continue
            segment_paths: dict[int, Path] = self._long_segment_video_paths(directory)
            contiguous: dict[int, Path] = {}
            index = 1
            while index in segment_paths:
                contiguous[index] = segment_paths[index]
                index += 1
            if not contiguous or len(contiguous) >= len(plan.shots):
                continue
            newest = max(path.stat().st_mtime for path in contiguous.values())
            candidates.append((newest, directory, contiguous))
        if not candidates:
            return None
        _, directory, contiguous = max(candidates, key=lambda item: item[0])
        checkpoint_path = LONG_CHECKPOINT_DIR / f"{directory.name}.json"
        if checkpoint_path.is_file():
            return checkpoint_path
        base_prefix = directory.relative_to(OUTPUT_DIR).as_posix()
        completed_paths = [contiguous[index] for index in sorted(contiguous)]
        last_index = len(completed_paths)
        latent_relative = (
            f"{base_prefix}/motion_context/latent_{last_index:05d}.safetensors"
        )
        latent_path = OUTPUT_DIR / Path(*latent_relative.split("/"))
        payload = {
            "version": LONG_CHECKPOINT_VERSION,
            "checkpoint_id": directory.name,
            "chat_id": self.allowed_chat_id,
            "status": "failed",
            "last_error": "由既有輸出影片建立的恢復檢查點；上一次執行沒有保存檢查點。",
            "created_at": min(path.stat().st_mtime for path in completed_paths),
            "updated_at": time.time(),
            "prompt": self.prompt,
            "config": {
                "width": self.settings.width,
                "height": self.settings.height,
                "steps": self.settings.steps,
                "requested_seconds": self.settings.requested_seconds,
                "length": self.settings.length,
            },
            "output_prefix": base_prefix,
            "total_seconds": float(self.total_seconds),
            "segment_total": len(plan.shots),
            "next_segment_index": last_index + 1,
            "completed_seconds": round(
                sum(shot.duration for shot in plan.shots[:last_index]), 3
            ),
            "shot_plan": [
                {
                    "start_seconds": shot.start_seconds,
                    "end_seconds": shot.end_seconds,
                    "label": shot.label,
                    "action": shot.action,
                }
                for shot in plan.shots
            ],
            "story_global_text": plan.global_text,
            "input_image_path": "",
            "completed_video_paths": [str(path.resolve()) for path in completed_paths],
            "motion_context_enabled": latent_path.is_file(),
            "latent_prefix": f"{base_prefix}/motion_context/latent",
            "last_context_latent_path": (
                latent_relative if latent_path.is_file() else ""
            ),
        }
        try:
            self._write_checkpoint_payload(checkpoint_path, payload)
        except OSError as exc:
            bot_log(f"legacy checkpoint creation failed: {exc}")
            return None
        bot_log(f"legacy checkpoint discovered {checkpoint_path}")
        return checkpoint_path

    def discover_legacy_history(self) -> None:
        """Register completed older output folders as extendable history items."""
        if getattr(self, "_legacy_history_discovered", False):
            return
        self._legacy_history_discovered = True
        root = OUTPUT_DIR / OUTPUT_PREFIX
        if not root.is_dir():
            return
        for directory in root.glob("long_*"):
            if not directory.is_dir():
                continue
            checkpoint_path = LONG_CHECKPOINT_DIR / f"{directory.name}.json"
            if checkpoint_path.is_file():
                continue
            full_video = directory / f"{directory.name}.mp4"
            segment_map: dict[int, Path] = self._long_segment_video_paths(directory)
            segment_paths: list[Path] = [segment_map[i] for i in sorted(segment_map)]
            if not full_video.is_file() or not segment_paths:
                continue
            try:
                total_seconds, width, height = probe_video_info(full_video)
                if not MIN_TOTAL_SECONDS <= total_seconds <= MAX_TOTAL_SECONDS:
                    continue
                segment_durations: list[float] = []
                for path in segment_paths:
                    try:
                        duration, _, _ = probe_video_info(path)
                    except BotError:
                        duration = total_seconds / len(segment_paths)
                    segment_durations.append(max(0.1, duration))
                duration_sum = sum(segment_durations)
                if duration_sum <= 0:
                    continue
                config_seconds = min(MAX_SEGMENT_SECONDS, max(MIN_TOTAL_SECONDS, total_seconds))
                config = parse_config(
                    [str(width), str(height), str(self.settings.steps), str(config_seconds)]
                )
            except (BotError, OSError, ValueError):
                continue

            shots: list[dict[str, Any]] = []
            cursor = 0.0
            for index, duration in enumerate(segment_durations, start=1):
                end = min(total_seconds, cursor + duration)
                shots.append(
                    {
                        "start_seconds": round(cursor, 3),
                        "end_seconds": round(end, 3),
                        "label": f"HISTORICAL {index}",
                        "action": (
                            "Previously generated historical shot. Preserve its character, "
                            "location, lighting and camera language before continuing."
                        ),
                    }
                )
                cursor = end
            if shots:
                shots[-1]["end_seconds"] = round(total_seconds, 3)

            latent_candidates = [
                path
                for path in directory.rglob("latent_*.safetensors")
                if re.fullmatch(r"latent_\d+\.safetensors", path.name)
            ]
            latent_candidates.sort(
                key=lambda path: int(re.search(r"latent_(\d+)", path.name).group(1))
            )
            last_latent = latent_candidates[-1] if latent_candidates else None
            base_prefix = directory.relative_to(OUTPUT_DIR).as_posix()
            try:
                relative_latent = (
                    last_latent.relative_to(OUTPUT_DIR).as_posix()
                    if last_latent is not None
                    else ""
                )
            except ValueError:
                relative_latent = ""
            try:
                created_at = min(path.stat().st_mtime for path in segment_paths)
                updated_at = full_video.stat().st_mtime
            except OSError:
                created_at = time.time()
                updated_at = created_at
            payload = {
                "version": LONG_CHECKPOINT_VERSION,
                "checkpoint_id": directory.name,
                "chat_id": self.allowed_chat_id,
                "status": "completed",
                "last_error": "",
                "created_at": created_at,
                "updated_at": updated_at,
                "prompt": "",
                "config": {
                    "width": config.width,
                    "height": config.height,
                    "steps": config.steps,
                    "requested_seconds": config.requested_seconds,
                    "length": config.length,
                },
                "output_prefix": base_prefix,
                "total_seconds": round(total_seconds, 3),
                "segment_total": len(segment_paths),
                "next_segment_index": len(segment_paths) + 1,
                "completed_seconds": round(total_seconds, 3),
                "shot_plan": shots,
                "story_global_text": (
                    "This is a completed historical video imported from the local output folder. "
                    "Use the supplied previous video and AV latent as the continuity source; "
                    "preserve the same subject, setting, lighting and camera language."
                ),
                "input_image_path": "",
                "completed_video_paths": [str(path.resolve()) for path in segment_paths],
                "motion_context_enabled": bool(relative_latent),
                "latent_prefix": (
                    f"{base_prefix}/motion_context/latent" if relative_latent else ""
                ),
                "last_context_latent_path": relative_latent,
            }
            try:
                self._write_checkpoint_payload(checkpoint_path, payload)
            except OSError as exc:
                bot_log(f"legacy history import failed {directory.name}: {exc}")
                continue
            bot_log(f"legacy history imported {checkpoint_path}")

    def long_checkpoint_records(self) -> list[tuple[Path, dict[str, Any]]]:
        self.discover_legacy_history()
        records: list[tuple[float, Path, dict[str, Any]]] = []
        if LONG_CHECKPOINT_DIR.is_dir():
            for path in LONG_CHECKPOINT_DIR.glob("*.json"):
                payload = self._read_checkpoint_payload(path)
                if payload is None or str(payload.get("chat_id", "")) != self.allowed_chat_id:
                    continue
                try:
                    updated = float(payload.get("updated_at", path.stat().st_mtime))
                except (OSError, TypeError, ValueError):
                    updated = path.stat().st_mtime
                records.append((updated, path, payload))
        if not records:
            legacy_path = self.discover_legacy_checkpoint()
            if legacy_path is not None:
                payload = self._read_checkpoint_payload(legacy_path)
                if payload is not None:
                    records.append((float(payload.get("updated_at", time.time())), legacy_path, payload))
        records.sort(key=lambda item: item[0], reverse=True)
        return [(path, payload) for _, path, payload in records]

    def latest_long_checkpoint(self) -> Optional[tuple[Path, dict[str, Any]]]:
        records = self.long_checkpoint_records()
        return records[0] if records else None

    @staticmethod
    def checkpoint_status_text(payload: dict[str, Any]) -> str:
        status = str(payload.get("status", "unknown"))
        next_index = int(payload.get("next_segment_index", 1))
        total = int(payload.get("segment_total", 0))
        if next_index > total:
            return "已完成" if status == "completed" else status
        return f"{status}，下次第 {next_index}/{total} 段"

    def history_text(self, records: list[tuple[Path, dict[str, Any]]]) -> str:
        lines = [
            "📚 歷史長片列表",
            "選擇一個 ID 後，可以延續已完成影片，或恢復未完成的影片。",
            "",
        ]
        if not records:
            lines.append("目前沒有可用的歷史長片 checkpoint。")
            return "\n".join(lines)
        for index, (path, payload) in enumerate(records[:MAX_HISTORY_ITEMS], start=1):
            checkpoint_id = self._checkpoint_id(path)
            total = float(payload.get("total_seconds", 0.0))
            status = self.checkpoint_status_text(payload)
            lines.append(f"{index}. {checkpoint_id} | {total:g} 秒 | {status}")
        if len(records) > MAX_HISTORY_ITEMS:
            lines.append(f"\n只顯示最近 {MAX_HISTORY_ITEMS} 項。")
        return "\n".join(lines)

    def history_markup(self, records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
        rows: list[list[dict[str, str]]] = []
        for path, payload in records[:MAX_HISTORY_ITEMS]:
            checkpoint_id = self._checkpoint_id(path)
            total = float(payload.get("total_seconds", 0.0))
            status = self.checkpoint_status_text(payload)
            icon = "📼" if int(payload.get("next_segment_index", 1)) > int(
                payload.get("segment_total", 0)
            ) else "🔁"
            rows.append(
                [
                    {
                        "text": f"{icon} {checkpoint_id} · {total:g}s",
                        "callback_data": f"history_select:{checkpoint_id}",
                    }
                ]
            )
        rows.append([{"text": "↩️ 返回控制面板", "callback_data": "history_back"}])
        return {"inline_keyboard": rows}

    def show_history(self, chat_id: str, message_id: Optional[int] = None) -> None:
        records = self.long_checkpoint_records()
        text = self.history_text(records)
        markup = self.history_markup(records)
        if message_id is not None:
            try:
                self.telegram.edit_message_text(chat_id, message_id, text, reply_markup=markup)
                self.menu_message_id = int(message_id)
                return
            except BotError:
                self.menu_message_id = None
        try:
            result = self.telegram.send_message(chat_id, text, reply_markup=markup)
            if isinstance(result, dict) and result.get("message_id"):
                self.menu_message_id = int(result["message_id"])
        except BotError as exc:
            self.send_safe(chat_id, f"歷史長片列表更新失敗：{exc}")

    def show_checkpoint_detail(
        self, chat_id: str, checkpoint_id: str, message_id: Optional[int] = None
    ) -> None:
        record = self.checkpoint_for_id(checkpoint_id)
        if record is None:
            self.send_safe(chat_id, "找不到這個歷史長片 ID，請重新開啟 /history。")
            return
        path, payload = record
        next_index = int(payload.get("next_segment_index", 1))
        total_segments = int(payload.get("segment_total", 0))
        total_seconds = float(payload.get("total_seconds", 0.0))
        video_paths = self._checkpoint_video_paths(payload)
        config_payload = payload.get("config")
        if not isinstance(config_payload, dict):
            config_payload = {}
        lines = [
            "📼 歷史長片詳情",
            f"ID：{self._checkpoint_id(path)}",
            f"片長：{total_seconds:g} 秒",
            f"分段：{max(0, next_index - 1)}/{total_segments}",
            f"解析度：{config_payload.get('width', '?')}×{config_payload.get('height', '?')}",
            f"目前狀態：{self.checkpoint_status_text(payload)}",
            f"可用影片分段：{len(video_paths)}",
        ]
        if payload.get("last_context_latent_path"):
            lines.append("連貫資料：影片尾端 + Motion Context latent")
        else:
            lines.append("連貫資料：影片尾幀／音訊參考")
        rows: list[list[dict[str, str]]] = []
        if next_index <= total_segments:
            rows.append(
                [
                    {
                        "text": f"🔁 從第 {next_index} 段繼續",
                        "callback_data": f"long_resume:{checkpoint_id}",
                    }
                ]
            )
        else:
            rows.append(
                [
                    {
                        "text": "📼 從這條影片延續新故事",
                        "callback_data": f"long_extend:{checkpoint_id}",
                    }
                ]
            )
        rows.append([{"text": "↩️ 返回歷史列表", "callback_data": "history"}])
        markup = {"inline_keyboard": rows}
        text = "\n".join(lines)
        if message_id is not None:
            try:
                self.telegram.edit_message_text(chat_id, message_id, text, reply_markup=markup)
                self.menu_message_id = int(message_id)
                return
            except BotError:
                self.menu_message_id = None
        try:
            result = self.telegram.send_message(chat_id, text, reply_markup=markup)
            if isinstance(result, dict) and result.get("message_id"):
                self.menu_message_id = int(result["message_id"])
        except BotError as exc:
            self.send_safe(chat_id, f"歷史長片詳情更新失敗：{exc}")

    def checkpoint_markup(self, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        checkpoint_id = self._checkpoint_id(path)
        next_index = int(payload.get("next_segment_index", 1))
        total = int(payload.get("segment_total", 0))
        status = str(payload.get("status", "failed"))
        rows: list[list[dict[str, str]]] = []
        if next_index <= total:
            rows.append(
                [
                    {
                        "text": f"🔁 從第 {next_index} 鏡繼續",
                        "callback_data": f"long_resume:{checkpoint_id}",
                    }
                ]
            )
        if next_index > total and status in {
            "completed",
            "failed",
            "running",
            "cancelled",
        }:
            rows.append(
                [
                    {
                        "text": "📼 延續上一條長片",
                        "callback_data": f"long_extend:{checkpoint_id}",
                    }
                ]
            )
        return {"inline_keyboard": rows}

    def send_checkpoint_actions(
        self, chat_id: str, path: Path, payload: dict[str, Any], notice: str = ""
    ) -> None:
        next_index = int(payload.get("next_segment_index", 1))
        total = int(payload.get("segment_total", 0))
        completed = max(0, next_index - 1)
        if next_index <= total:
            action_text = (
                f"{notice}\n已保留長片前 {completed}/{total} 鏡。"
                f"可以從第 {next_index} 鏡繼續，不會重做前面的鏡頭。"
            ).strip()
        else:
            action_text = f"{notice}\n這條長片已完成，可以從尾部繼續新增內容。".strip()
        try:
            self.telegram.send_message(
                chat_id,
                action_text,
                reply_markup=self.checkpoint_markup(path, payload),
            )
        except BotError as exc:
            bot_log(f"checkpoint action message failed: {exc}")

    def queue_text(self) -> str:
        with self.lock:
            items = list(self.story_queue)
            active_job = self.job
            pending_upscale = self.pending_upscale
        lines = [
            "🧾 故事生成排隊",
            f"等待中的故事：{len(items)}/{MAX_QUEUE_ITEMS}",
        ]
        if active_job is not None:
            lines.append("目前有一個故事正在生成，完成後會自動開始下一個。")
        elif pending_upscale is not None:
            lines.append("目前正在等你選擇上一條影片是否放大；選擇後才會開始下一個。")
        elif items:
            lines.append("按「▶️ 開始排隊」即可開始第一個故事。")
        else:
            lines.append("排隊是空的。可以一次貼上多個故事，使用獨立一行的 --- 分隔。")
        for index, item in enumerate(items, start=1):
            preview = " ".join(item.prompt.split()).replace("---", "-")
            if len(preview) > 90:
                preview = preview[:87] + "..."
            lines.append(
                f"{index}. {item.item_id} | {item.total_seconds:g}s | "
                f"{item.config.width}×{item.config.height} | {preview}"
            )
        return "\n".join(lines)

    def queue_markup(self) -> dict[str, Any]:
        with self.lock:
            items = list(self.story_queue)
        rows: list[list[dict[str, str]]] = []
        for index, item in enumerate(items, start=1):
            rows.append(
                [
                    {
                        "text": f"❌ 移除第 {index} 個 ({item.item_id})",
                        "callback_data": f"queue_remove:{item.item_id}",
                    }
                ]
            )
        rows.extend(
            [
                [
                    {"text": "➕ 加入故事", "callback_data": "queue_add"},
                    {"text": "▶️ 開始排隊", "callback_data": "queue_start"},
                ],
                [
                    {"text": "🗑 清空排隊", "callback_data": "queue_clear"},
                    {"text": "↩️ 返回控制面板", "callback_data": "history_back"},
                ],
            ]
        )
        return {"inline_keyboard": rows}

    def show_queue(self, chat_id: str, message_id: Optional[int] = None) -> None:
        text = self.queue_text()
        markup = self.queue_markup()
        if message_id is not None:
            try:
                self.telegram.edit_message_text(chat_id, message_id, text, reply_markup=markup)
                self.menu_message_id = int(message_id)
                return
            except BotError:
                self.menu_message_id = None
        try:
            result = self.telegram.send_message(chat_id, text, reply_markup=markup)
            if isinstance(result, dict) and result.get("message_id"):
                self.menu_message_id = int(result["message_id"])
        except BotError as exc:
            self.send_safe(chat_id, f"排隊列表更新失敗：{exc}")

    def request_queue_prompt(self, chat_id: str) -> None:
        self.awaiting_queue_prompt = True
        self.awaiting_prompt = False
        self.awaiting_duration = False
        self.awaiting_extension_duration = False
        self.awaiting_extension_prompt = False
        self.telegram.send_message(
            chat_id,
            "請貼上要排隊的故事提示詞。\n"
            "一次輸入多個故事時，請用獨立一行的 --- 分隔；每個故事會自動讀取自己的腳本片長。",
            reply_markup={
                "force_reply": True,
                "input_field_placeholder": "故事 1...\n---\n故事 2...",
            },
        )

    def enqueue_story_prompts(self, chat_id: str, text: str) -> None:
        prompts = split_story_queue_prompts(text)
        if not prompts:
            self.send_safe(chat_id, "提示詞不可為空白。")
            return
        with self.lock:
            available = max(0, MAX_QUEUE_ITEMS - len(self.story_queue))
        if available <= 0:
            self.send_safe(chat_id, f"排隊已達上限 {MAX_QUEUE_ITEMS} 個，請先移除或清空。")
            return
        prompts = prompts[:available]
        config = self.effective_config()
        total_seconds = self.total_seconds
        input_image_path = (
            self.image_path
            if self.input_mode == "image" and self.image_path and self.image_path.is_file()
            else None
        )
        if self.input_mode in {INPUT_MODE_IMAGE, INPUT_MODE_FL2VA}:
            input_image_path = (
                self.image_path if self.image_path and self.image_path.is_file() else None
            )
        last_image_path = (
            self.last_image_path
            if self.input_mode == INPUT_MODE_FL2VA
            and self.last_image_path
            and self.last_image_path.is_file()
            else None
        )
        reference_image_paths = tuple(
            path for path in self.reference_image_paths if path.is_file()
        )
        reference_video_paths = tuple(
            path for path in self.reference_video_paths if path.is_file()
        )
        reference_audio_paths = tuple(
            path for path in self.reference_audio_paths if path.is_file()
        )
        new_items: list[QueuedStory] = []
        for prompt in prompts:
            prompt_total = detect_prompt_total_seconds(prompt) or total_seconds
            prompt_config = parse_config(
                [
                    str(config.width),
                    str(config.height),
                    str(config.steps),
                    str(min(prompt_total, MAX_SEGMENT_SECONDS)),
                ]
            )
            new_items.append(
                QueuedStory(
                    item_id=f"q_{secrets.token_hex(4)}",
                    prompt=prompt,
                    config=prompt_config,
                    total_seconds=prompt_total,
                    input_image_path=input_image_path,
                    last_image_path=last_image_path,
                    reference_image_paths=reference_image_paths,
                    reference_video_paths=reference_video_paths,
                    reference_audio_paths=reference_audio_paths,
                    generation_mode=self.input_mode,
                    model_mode=self.model_mode,
                )
            )
        with self.lock:
            self.story_queue.extend(new_items)
        self.save_story_queue()
        self.send_safe(
            chat_id,
            f"已加入 {len(new_items)} 個故事；目前排隊 {len(self.story_queue)} 個。"
            + (f"（最多只能再加入 {available} 個，本次已截取前 {available} 個。）" if len(split_story_queue_prompts(text)) > available else ""),
        )
        self.awaiting_queue_prompt = False
        self.start_next_queued_story(chat_id)
        self.show_queue(chat_id)

    def start_next_queued_story(self, chat_id: str) -> bool:
        with self.lock:
            if (
                self.job is not None
                or self.pending_upscale is not None
                or self._queue_starting
                or not self.story_queue
            ):
                return False
            item = self.story_queue.pop(0)
            self._queue_starting = True
        self.save_story_queue()
        started = False
        try:
            self.prompt = item.prompt
            self.settings = item.config
            self.total_seconds = item.total_seconds
            self.model_mode = normalize_model_mode(item.model_mode)
            self.save_settings()
            if item.total_seconds > MAX_SEGMENT_SECONDS:
                started = self.start_long_generation(
                    chat_id,
                    item.config,
                    item.prompt,
                    item.total_seconds,
                    input_image_path=item.input_image_path,
                    last_image_path=item.last_image_path,
                    reference_image_paths=list(item.reference_image_paths),
                    reference_video_paths=list(item.reference_video_paths),
                    reference_audio_paths=list(item.reference_audio_paths),
                    generation_mode=item.generation_mode,
                )
            else:
                started = self.start_generation(
                    chat_id,
                    item.config,
                    item.prompt,
                    input_image_path=item.input_image_path,
                    last_image_path=item.last_image_path,
                    reference_image_paths=list(item.reference_image_paths),
                    reference_video_paths=list(item.reference_video_paths),
                    reference_audio_paths=list(item.reference_audio_paths),
                    generation_mode=item.generation_mode,
                )
        except Exception as exc:
            self.send_safe(chat_id, f"排隊故事啟動失敗：{exc}")
        finally:
            with self.lock:
                self._queue_starting = False
        if not started:
            with self.lock:
                self.story_queue.insert(0, item)
            self.save_story_queue()
            return False
        with self.lock:
            remaining = len(self.story_queue)
        self.send_safe(
            chat_id,
            f"▶️ 排隊故事 {item.item_id} 已開始。剩餘 {remaining} 個，完成後自動接續。",
        )
        return True

    def clear_story_queue(self, chat_id: str, message_id: Optional[int] = None) -> None:
        with self.lock:
            removed = len(self.story_queue)
            self.story_queue.clear()
        self.save_story_queue()
        self.show_queue(chat_id, message_id)
        self.send_safe(chat_id, f"已清空排隊，移除 {removed} 個等待中的故事。")

    def remove_queued_story(
        self, chat_id: str, item_id: str, message_id: Optional[int] = None
    ) -> None:
        with self.lock:
            before = len(self.story_queue)
            self.story_queue = [item for item in self.story_queue if item.item_id != item_id]
            removed = before - len(self.story_queue)
        self.save_story_queue()
        self.show_queue(chat_id, message_id)
        self.send_safe(chat_id, f"已移除 {item_id}。" if removed else "找不到這個排隊項目。")

    def on_job_finished(self, chat_id: str) -> None:
        try:
            if self.chain_active():
                # An auto-chain run owns the next step; don't wake or stop the
                # LLM/ComfyUI between clips.
                bot_log("on_job_finished: auto-chain running, LLM restart skipped")
                return
            if self.start_next_queued_story(chat_id):
                # A queued story is already running; don't start the LLM only to
                # stop it again for the next job.
                bot_log("on_job_finished: queued story started, LLM restart skipped")
                return
            self.restart_llm_after_job(chat_id)
        except Exception as exc:  # noqa: BLE001 - never kill the job thread
            bot_log(f"on_job_finished failed: {type(exc).__name__}: {exc}")

    def restart_llm_after_job(self, chat_id: str) -> None:
        """Start the local LLM again once a generation has finished.

        The bot stops the LLM before every job to free VRAM, so this puts it
        back for Hermes / the DSH local models. ComfyUI is shut down FIRST:
        the LLM needs ~20GB and cannot share the card with a loaded H3 model,
        so starting the LLM while ComfyUI still holds its model would OOM.
        Toggle with the 🧠 button in the system menu
        (or MINIMAX_LLM_RESTART_AFTER_JOB=0 for the default).
        """
        if get_script_llm_provider() == SCRIPT_LLM_COMMANDCODE:
            # Scripts come from the cloud engine now, so the Bot has no reason to
            # wake the local LLM back up after a job. Start it manually (🧠 啟動
            # LLM / /llm_start) when Hermes or the DSH models need it.
            bot_log("restart_llm_after_job: skipped (script engine = Command Code)")
            return
        enabled = bool(getattr(self, "restart_llm_after_generation", True))
        try:
            llama_up = llama_is_online()
            comfy_up = comfyui_is_online()
        except OSError as exc:
            # Probe failures must never block the restore below.
            bot_log(f"restart_llm_after_job: probe failed {type(exc).__name__}: {exc}")
            llama_up, comfy_up = False, True
        bot_log(
            f"restart_llm_after_job: enabled={enabled} llama_online={llama_up} "
            f"comfy_online={comfy_up}"
        )
        if not enabled:
            return
        if llama_up:
            return
        notes: list[str] = []
        # Free the GPU before the LLM loads. stop_comfyui_process() kills the
        # server and waits for port 8191 to go quiet, so the VRAM is really
        # released by the time we return.
        #
        # This step is best-effort on purpose: the entire point of this method is
        # to bring the LLM back, so a ComfyUI hiccup (including a raw socket
        # error such as ConnectionResetError) must not abort it. OSError is
        # caught alongside BotError for exactly that reason.
        if comfy_up:
            try:
                notes.append(stop_comfyui_process())
            except (BotError, OSError) as exc:
                notes.append(f"⚠️ 關閉 ComfyUI 失敗：{exc}")
        try:
            notes.append(start_llama_process())
        except (BotError, OSError) as exc:
            notes.append(f"⚠️ 啟動 LLM 失敗：{exc}")
        summary = "\n".join(note for note in notes if note)
        if not summary:
            bot_log("restart_llm_after_job: nothing to report (empty summary)")
            return
        bot_log("LLM restart after job: " + summary.replace("\n", " | "))
        try:
            self.send_safe(chat_id, "🧠 生成完成，還原本地 LLM：\n" + summary)
        except (BotError, OSError) as exc:
            # A failed status message must not look like a failed restart.
            bot_log(f"restart_llm_after_job: report failed {exc}")

    def update_settings(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        seconds: Optional[float] = None,
    ) -> None:
        current = self.settings
        self.settings = parse_config(
            [
                str(width if width is not None else current.width),
                str(height if height is not None else current.height),
                str(steps if steps is not None else current.steps),
                str(seconds if seconds is not None else current.requested_seconds),
            ]
        )
        self.save_settings()

    @staticmethod
    def selected(label: str, active: bool) -> str:
        return ("✅ " if active else "") + label

    @staticmethod
    def duration_label(seconds: float) -> str:
        if seconds >= 60 and seconds % 60 == 0:
            return f"{int(seconds // 60)} 分鐘"
        return f"{seconds:g} 秒"

    def menu_markup(self, section: Optional[str] = None) -> dict[str, Any]:
        """Build a compact, sectioned inline menu.

        The old menu rendered every control at once.  Keeping the callbacks but
        grouping them here makes the main panel usable on a phone while leaving
        all existing generation and system actions available one level down.
        """
        section = normalize_menu_section(
            section or getattr(self, "menu_section", MENU_MAIN)
        )
        current = self.settings
        mode_row = [
            {
                "text": self.selected("📝 T2VA 文字", self.input_mode == INPUT_MODE_TEXT),
                "callback_data": "mode:text",
            },
            {
                "text": self.selected("🖼 I2VA 圖片", self.input_mode == INPUT_MODE_IMAGE),
                "callback_data": "mode:image",
            },
        ]
        reference_mode_row = [
            {
                "text": self.selected(
                    "🎬 FL2VA 首尾幀", self.input_mode == INPUT_MODE_FL2VA
                ),
                "callback_data": "mode:fl2va",
            },
            {
                "text": self.selected(
                    "📚 Ref2VA 參考", self.input_mode == INPUT_MODE_REF2VA
                ),
                "callback_data": "mode:ref2va",
            },
        ]
        # YUPI工作流：a real mode now (Ref2VA + AfterMidnight + FastH3 6-step).
        # Selecting it only stages the mode; generation starts from
        # 🚀 生成影片 like every other mode.
        yupi_row = [
            {
                "text": self.selected(
                    YUPI_BUTTON, self.input_mode == INPUT_MODE_YUPI
                ),
                "callback_data": "mode:yupi",
            },
        ]
        resolution_row = [
            {
                "text": self.selected(
                    resolution_label(width, height),
                    (width, height) == (current.width, current.height),
                ),
                "callback_data": f"res:{width}x{height}",
            }
            for width, height in self.RESOLUTIONS
        ]
        short_seconds_row = [
            {
                "text": self.selected(
                    self.duration_label(seconds), abs(self.total_seconds - seconds) < 0.001
                ),
                "callback_data": f"sec:{seconds}",
            }
            for seconds in self.SECONDS
        ]
        long_seconds_row = [
            {
                "text": self.selected(
                    self.duration_label(seconds), abs(self.total_seconds - seconds) < 0.001
                ),
                "callback_data": f"sec:{seconds}",
            }
            for seconds in self.LONG_SECONDS
        ]
        steps_row = [
            {
                "text": self.selected(f"{steps} steps", current.steps == steps),
                "callback_data": f"steps:{steps}",
            }
            for steps in self.STEPS
        ]
        with self.lock:
            active_job = self.job

        if active_job is not None and active_job.pause_requested.is_set():
            pause_button = {"text": "⏸ 暫停中", "callback_data": "job_pause"}
        else:
            pause_button = {"text": "⏸ 暫停", "callback_data": "job_pause"}
        job_control_row = [
            {"text": "⛔ 中止", "callback_data": "job_abort"},
            pause_button,
            {"text": "▶️ 播放／繼續", "callback_data": "job_resume"},
        ]
        checkpoint_rows: list[list[dict[str, str]]] = []
        if section in {MENU_MAIN, MENU_HISTORY} and active_job is None:
            checkpoint_record = self.latest_long_checkpoint()
            if checkpoint_record is not None:
                checkpoint_path, checkpoint_payload = checkpoint_record
                checkpoint_rows = self.checkpoint_markup(
                    checkpoint_path, checkpoint_payload
                ).get("inline_keyboard", [])

        def back_row(callback: str = "menu:main") -> list[list[dict[str, str]]]:
            label = "↩️ 返回生成設定" if callback == "menu:settings" else "↩️ 返回主選單"
            return [[{"text": label, "callback_data": callback}]]

        if section == MENU_MAIN:
            rows = [
                mode_row,
                reference_mode_row,
                yupi_row,
                [
                    {"text": "✍️ 輸入／更換提示詞", "callback_data": "prompt"},
                    {"text": "🧹 清除提示詞", "callback_data": "clear"},
                ],
                [{"text": "✨ 一句話生成腳本", "callback_data": "script:new"}],
                [
                    {
                        "text": "🌐 語言："
                        + SCRIPT_LANG_BUTTON[
                            normalize_script_lang(
                                getattr(self, "script_lang", SCRIPT_LANG_DEFAULT)
                            )
                        ],
                        "callback_data": "script_lang:toggle",
                    },
                    {
                        "text": "📄 模板："
                        + SCRIPT_TEMPLATE_LABEL[
                            normalize_script_template(
                                getattr(self, "script_template", SCRIPT_TEMPLATE_DEFAULT)
                            )
                        ],
                        "callback_data": "script_template:toggle",
                    },
                ],
                [
                    {
                        "text": "🧠 LLM："
                        + SCRIPT_LLM_LABEL[
                            normalize_script_llm(
                                getattr(self, "script_llm", SCRIPT_LLM_DEFAULT)
                            )
                        ],
                        "callback_data": "script_llm:toggle",
                    },
                    {"text": "📝 自訂指令", "callback_data": "script_file"},
                ],
                [{"text": "🗑 清除上傳素材", "callback_data": "clear_image"}],
            ]
            if is_ref2va_like(self.input_mode):
                rows.append(
                    [{"text": "✅ 完成參考素材上傳", "callback_data": "media_done"}]
                )
            rows.extend(
                [
                    [
                        {"text": "🚀 生成影片", "callback_data": "generate"},
                        {"text": "♻️ 讀取上次設定", "callback_data": "last"},
                    ],
                    [
                        {"text": "🔗 自動接力", "callback_data": "chain:start"},
                    ],
                    [
                        {
                            "text": "⚙️ 片長／解析度／steps",
                            "callback_data": "menu:settings",
                        }
                    ],
                ]
            )
            if active_job is not None:
                rows.append(job_control_row)
                if active_job.segment_total > 1:
                    rows.append(
                        [{"text": "🎬 預覽已完成片段", "callback_data": "job_preview"}]
                    )
            else:
                rows.append([{"text": "目前沒有進行中的任務", "callback_data": "noop"}])
            rows.append(
                [
                    {"text": "📚 歷史長片", "callback_data": "history"},
                    {"text": "🧾 故事排隊", "callback_data": "queue_view"},
                ]
            )
            rows.extend(checkpoint_rows)
            rows.extend(
                [
                    [
                        {
                            "text": "🖥️ 電腦／ComfyUI／LLM／Bot",
                            "callback_data": "menu:system",
                        }
                    ],
                ]
            )
            shutdown_label = (
                "🛑 取消即將關機"
                if self._shutdown_pending
                else self.selected("🔌 長片完成後關機", self.shutdown_after_generation)
            )
            rows.append(
                [
                    {
                        "text": shutdown_label,
                        "callback_data": (
                            "shutdown_cancel"
                            if self._shutdown_pending
                            else "shutdown_toggle"
                        ),
                    }
                ]
            )
            rows.append([{"text": "📊 查看／刷新生成進度", "callback_data": "progress"}])
        elif section == MENU_INPUT:
            rows = [
                [
                    {"text": "✍️ 輸入／更換提示詞", "callback_data": "prompt"},
                    {"text": "🧹 清除提示詞", "callback_data": "clear"},
                ],
                [{"text": "✨ 一句話生成腳本", "callback_data": "script:new"}],
                [
                    {
                        "text": "🌐 語言："
                        + SCRIPT_LANG_BUTTON[
                            normalize_script_lang(
                                getattr(self, "script_lang", SCRIPT_LANG_DEFAULT)
                            )
                        ],
                        "callback_data": "script_lang:toggle",
                    },
                    {
                        "text": "📄 模板："
                        + SCRIPT_TEMPLATE_LABEL[
                            normalize_script_template(
                                getattr(self, "script_template", SCRIPT_TEMPLATE_DEFAULT)
                            )
                        ],
                        "callback_data": "script_template:toggle",
                    },
                ],
                [
                    {
                        "text": "🧠 LLM："
                        + SCRIPT_LLM_LABEL[
                            normalize_script_llm(
                                getattr(self, "script_llm", SCRIPT_LLM_DEFAULT)
                            )
                        ],
                        "callback_data": "script_llm:toggle",
                    },
                    {"text": "📝 自訂指令", "callback_data": "script_file"},
                ],
                [{"text": "🗑 清除上傳素材", "callback_data": "clear_image"}],
            ]
            if is_ref2va_like(self.input_mode):
                rows.append(
                    [{"text": "✅ 完成參考素材上傳", "callback_data": "media_done"}]
                )
            rows.append([{"text": "🎛️ 前往生成模式", "callback_data": "menu:mode"}])
            rows.extend(back_row())
        elif section == MENU_SETTINGS:
            two_pass_label = self.selected(
                "🔬 兩段式 latent 上採樣",
                bool(getattr(self, "latent_upscale", LATENT_UPSCALE_ENABLED)),
            )
            continuity_label = (
                "🔗 長片接續：Motion Context ✅"
                if getattr(self, "long_continuity", "motion_context")
                == "motion_context"
                else "🔗 長片接續：尾幀接續 ✅"
            )
            profile_label = (
                "🧩 模型：融合加速（6步+SLA）✅"
                if getattr(self, "h3_profile", H3_PROFILE_DEFAULT)
                == H3_PROFILE_FUSED
                else "🧩 模型：經典（FL2VA/Ref2VA+LoRA）✅"
            )
            rows = [
                [{"text": "⏱️ 片長／秒數（按下選擇）", "callback_data": "noop"}],
                short_seconds_row[:2],
                short_seconds_row[2:],
                [{"text": "長片：自動分段", "callback_data": "noop"}],
                long_seconds_row[:3],
                long_seconds_row[3:6],
                long_seconds_row[6:],
                [{"text": "✏️ 自定義秒數", "callback_data": "sec_custom"}],
                [{"text": "🖼️ 解析度／MP（按下選擇）", "callback_data": "noop"}],
                resolution_row[:3],
                resolution_row[3:6],
                resolution_row[6:],
                [{"text": "⚙️ 步數（按下選擇）", "callback_data": "noop"}],
                steps_row,
                [{"text": two_pass_label, "callback_data": "twopass_toggle"}],
                [{"text": continuity_label, "callback_data": "continuity_toggle"}],
                [{"text": profile_label, "callback_data": "h3_profile_toggle"}],
            ]
            rows.extend(back_row())
        elif section == MENU_MODE:
            rows = [mode_row, reference_mode_row, yupi_row]
            rows.extend(back_row("menu:settings"))
        elif section == MENU_DURATION:
            rows = [
                [{"text": "短片：5／10／12／15 秒", "callback_data": "noop"}],
                short_seconds_row[:2],
                short_seconds_row[2:],
                [{"text": "長片：自動分段", "callback_data": "noop"}],
                long_seconds_row[:3],
                long_seconds_row[3:6],
                long_seconds_row[6:],
                [{"text": "✏️ 自定義秒數", "callback_data": "sec_custom"}],
            ]
            rows.extend(back_row("menu:settings"))
        elif section == MENU_QUALITY:
            rows = [
                [{"text": "🖼️ 解析度／MP", "callback_data": "noop"}],
                resolution_row[:3],
                resolution_row[3:6],
                resolution_row[6:],
                [{"text": "⚙️ 步數", "callback_data": "noop"}],
                steps_row,
            ]
            rows.extend(back_row("menu:settings"))
        elif section == MENU_JOB:
            rows = [[{"text": "📊 查看／刷新生成進度", "callback_data": "progress"}]]
            if active_job is not None:
                rows.append(job_control_row)
                if active_job.segment_total > 1:
                    rows.append(
                        [{"text": "🎬 預覽已完成片段", "callback_data": "job_preview"}]
                    )
            else:
                rows.append([{"text": "目前沒有進行中的任務", "callback_data": "noop"}])
            rows.extend(back_row())
        elif section == MENU_SYSTEM:
            shutdown_label = (
                "🛑 取消即將關機"
                if self._shutdown_pending
                else self.selected(
                    "🔌 長片完成後關機", self.shutdown_after_generation
                )
            )
            rows = [
                [{"text": "🌡 查看電腦溫度", "callback_data": "temperature"}],
                [
                    {"text": "▶️ 啟動 ComfyUI", "callback_data": "comfy_start"},
                    {"text": "📡 ComfyUI 狀態", "callback_data": "comfy_status"},
                ],
                [
                    {"text": "🔄 重啟 ComfyUI", "callback_data": "comfy_restart"},
                    {"text": "⏹ 關閉 ComfyUI", "callback_data": "comfy_stop"},
                ],
                [
                    {"text": "▶️ 啟動 LLM", "callback_data": "llama_start"},
                    {"text": "📡 LLM 狀態", "callback_data": "llama_status"},
                ],
                [
                    {"text": "🔄 重啟 LLM", "callback_data": "llama_restart"},
                    {"text": "⏹ 關閉 LLM", "callback_data": "llama_stop"},
                ],
                [
                    {
                        "text": self.selected(
                            "🧠 生成後自動啟動 LLM",
                            bool(
                                getattr(
                                    self,
                                    "restart_llm_after_generation",
                                    RESTART_LLM_AFTER_GENERATION,
                                )
                            ),
                        ),
                        "callback_data": "llm_after_job_toggle",
                    }
                ],
                [{"text": "🔄 重啟 Bot", "callback_data": "bot_restart"}],
                [{"text": shutdown_label, "callback_data": "shutdown_toggle"}],
            ]
            if self._shutdown_pending:
                rows[-1][0]["callback_data"] = "shutdown_cancel"
            rows.extend(back_row())
        elif section == MENU_HISTORY:
            rows = [
                [
                    {"text": "📚 歷史長片", "callback_data": "history"},
                    {"text": "🧾 故事排隊", "callback_data": "queue_view"},
                ],
                *checkpoint_rows,
            ]
            rows.extend(back_row())
        else:
            rows = []
            rows.extend(back_row())
        return {"inline_keyboard": rows}

    def effective_config(self) -> GenerationConfig:
        segment_seconds = min(self.total_seconds, MAX_SEGMENT_SECONDS)
        return parse_config(
            [
                str(self.settings.width),
                str(self.settings.height),
                str(self.settings.steps),
                str(segment_seconds),
            ]
        )

    def auto_detect_prompt_duration(
        self, prompt: str, *, chat_id: Optional[str] = None, persist: bool = False
    ) -> Optional[float]:
        """Use the script's maximum timestamp as the total video duration."""
        detected = detect_prompt_total_seconds(prompt)
        if detected is None:
            return None
        changed = abs(float(getattr(self, "total_seconds", 0.0)) - detected) > 0.001
        self.total_seconds = detected
        current = self.settings
        self.settings = parse_config(
            [
                str(current.width),
                str(current.height),
                str(current.steps),
                str(min(detected, MAX_SEGMENT_SECONDS)),
            ]
        )
        if persist:
            self.save_settings()
        if chat_id is not None and changed:
            self.send_safe(
                chat_id,
                f"已按腳本自動設定總片長：{self.duration_label(detected)}。",
            )
        return detected

    def set_total_seconds(self, seconds: float) -> None:
        if not math.isfinite(seconds):
            raise BotError("總片長必須是有效數字，範圍為 2 至 1800 秒。")
        self.total_seconds = validate_total_seconds(seconds)
        self.update_settings(seconds=min(self.total_seconds, MAX_SEGMENT_SECONDS))

    def wait_for_resume(self, job: JobState) -> bool:
        """Pause long-video work safely between generated shots."""
        if job.segment_total <= 1 or not job.pause_requested.is_set():
            return not job.cancel_event.is_set()
        with job.progress_lock:
            job.progress_phase = "paused"
            job.progress_node_state = "paused"
        self.send_safe(
            job.chat_id,
            "目前長片已暫停，會保留已完成分段；按「▶️ 播放／繼續」生成下一段。",
        )
        while job.pause_requested.is_set() and not job.cancel_event.is_set():
            job.resume_event.wait(1.0)
        if job.cancel_event.is_set():
            return False
        with job.progress_lock:
            job.progress_phase = "waiting"
            job.progress_node_state = "resumed"
        self.send_safe(job.chat_id, "已繼續長片生成。")
        return True

    def abort_current_job(
        self, chat_id: str, message_id: Optional[int] = None
    ) -> None:
        with self.lock:
            job = self.job
            prompt_id = job.prompt_id if job is not None else None
            if job is not None:
                job.cancel_event.set()
                job.resume_event.set()
        if job is None:
            self.show_menu(chat_id, message_id, "目前沒有生成中的任務")
            return
        if prompt_id:
            try:
                comfy_post("/interrupt", {"prompt_id": prompt_id})
            except BotError as exc:
                self.send_safe(chat_id, f"已標記中止，但 ComfyUI 中止請求失敗：{exc}")
        self.send_safe(chat_id, "已中止目前生成任務，未完成分段不會繼續。")
        self.show_menu(chat_id, message_id)

    def pause_current_job(
        self, chat_id: str, message_id: Optional[int] = None
    ) -> None:
        with self.lock:
            job = self.job
            if job is not None and job.segment_total > 1:
                if job.pause_requested.is_set():
                    already_paused = True
                else:
                    already_paused = False
                    job.pause_requested.set()
                    job.resume_event.clear()
            else:
                already_paused = False
        if job is None:
            self.show_menu(chat_id, message_id, "目前沒有生成中的任務")
            return
        if job.segment_total <= 1:
            self.send_safe(
                chat_id,
                "單段影片不能安全凍結採樣；只有長片可以在每段完成後暫停。需要停止請按「中止」。",
            )
        elif already_paused:
            self.send_safe(chat_id, "長片已在暫停流程中，會在目前鏡頭完成後停下。")
        else:
            self.send_safe(
                chat_id,
                "已收到暫停要求；目前短鏡頭完成後會暫停，不會丟失已完成鏡頭。",
            )
        self.show_menu(chat_id, message_id)

    def resume_current_job(
        self, chat_id: str, message_id: Optional[int] = None
    ) -> None:
        with self.lock:
            job = self.job
            if job is not None and job.pause_requested.is_set():
                job.pause_requested.clear()
                job.resume_event.set()
                was_paused = True
            else:
                was_paused = False
        if job is None:
            record = self.latest_long_checkpoint()
            if record is not None:
                checkpoint_path, checkpoint_payload = record
                if int(checkpoint_payload.get("next_segment_index", 1)) <= int(
                    checkpoint_payload.get("segment_total", 0)
                ):
                    self.resume_long_checkpoint(
                        chat_id,
                        self._checkpoint_id(checkpoint_path),
                        message_id,
                    )
                    return
            self.show_menu(chat_id, message_id, "目前沒有可繼續的生成任務")
            return
        if job.segment_total <= 1:
            self.send_safe(chat_id, "單段影片沒有暫停狀態。")
        elif was_paused:
            self.send_safe(chat_id, "已播放／繼續；下一段會繼續生成。")
        else:
            self.send_safe(chat_id, "目前任務沒有暫停，會繼續生成。")
        self.show_menu(chat_id, message_id)

    def request_video_preview(
        self, chat_id: str, message_id: Optional[int] = None
    ) -> None:
        """Merge the finished shots so far and send them as an early preview.

        Runs in its own thread so the long job keeps generating the remaining
        shots while FFmpeg merges and Telegram uploads the completed part.
        """
        with self.lock:
            job = self.job
            if job is None:
                self.show_menu(chat_id, message_id, "目前沒有生成中的任務")
                return
            if job.segment_total <= 1:
                self.send_safe(chat_id, "單段影片沒有可預覽的已完成分段。")
                return
            if job.cancel_event.is_set():
                self.send_safe(
                    chat_id,
                    "長片正在中止流程中，稍候會收到已合成的部分影片。",
                )
                return
            with job.progress_lock:
                phase = job.progress_phase
            if phase in {"merging", "uploading"}:
                self.send_safe(
                    chat_id,
                    "所有鏡頭都已完成，最終影片正在合併，稍候就會收到完整版。",
                )
                return
            if job.preview_in_progress.is_set():
                self.send_safe(
                    chat_id,
                    "🎬 預覽正在合成中，請稍候片刻，長片生成不受影響。",
                )
                return
            snapshot = list(job.completed_video_paths)
            if not snapshot:
                self.send_safe(
                    chat_id,
                    "第一個鏡頭還沒完成，暫時沒有可預覽的內容；"
                    "完成後再按一次即可。",
                )
                return
            segment_total = job.segment_total
            shot_plan = job.shot_plan
            total_seconds = job.total_seconds
            base_prefix = job.long_base_prefix or job.output_prefix
            base_config = job.base_config or job.config
            job.preview_in_progress.set()
        completed_count = len(snapshot)
        self.send_safe(
            chat_id,
            f"🎬 已收到預覽要求：正在合成已完成的前 {completed_count}/{segment_total} 段，"
            "長片生成不會中斷。",
        )
        threading.Thread(
            target=self._run_video_preview,
            args=(
                job,
                snapshot,
                base_prefix,
                base_config,
                completed_count,
                shot_plan,
                total_seconds,
            ),
            name="minimax-long-preview",
            daemon=True,
        ).start()
        self.show_menu(chat_id, message_id)

    def _run_video_preview(
        self,
        job: JobState,
        snapshot: list[Path],
        base_prefix: str,
        base_config: GenerationConfig,
        completed_count: int,
        shot_plan: tuple[ShotSpec, ...],
        total_seconds: float,
    ) -> None:
        """Background preview worker: merge a snapshot and send it to Telegram."""
        try:
            completed_shots = (
                shot_plan[:completed_count]
                if len(shot_plan) >= completed_count
                else tuple()
            )
            if completed_shots:
                completed_seconds = min(
                    total_seconds,
                    sum(shot.duration for shot in completed_shots),
                )
            else:
                completed_seconds = min(
                    total_seconds,
                    total_seconds * completed_count / max(job.segment_total, 1),
                )
            batch_name = base_prefix.rsplit("/", 1)[-1]
            output_path = (
                OUTPUT_DIR
                / base_prefix
                / f"{batch_name}_preview_{completed_count:02d}_"
                f"{uuid.uuid4().hex[:8]}.mp4"
            )
            merge_completed_segments(
                snapshot,
                output_path,
                completed_seconds,
                shot_plan=completed_shots or None,
                output_size=(base_config.width, base_config.height),
            )
            caption = (
                "🎬 提前預覽（長片仍在繼續生成）\n"
                f"已完成 {completed_count}/{job.segment_total} 段，"
                f"約 {completed_seconds:.2f} 秒 | "
                f"{base_config.width}×{base_config.height}"
            )
            self.telegram.send_video(job.chat_id, output_path, caption)
            bot_log(
                f"long preview sent {output_path} "
                f"({completed_count}/{job.segment_total})"
            )
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
        except Exception as exc:
            bot_log(f"long preview error: {exc}")
            self.send_safe(job.chat_id, f"🎬 預覽合成失敗：{exc}")
        finally:
            job.preview_in_progress.clear()

    def start_selected_generation(self, chat_id: str, prompt: str) -> bool:
        self.auto_detect_prompt_duration(prompt, chat_id=chat_id, persist=True)
        if self.input_mode == INPUT_MODE_YUPI:
            # YUPI has its own graph (Ref2VA + AfterMidnight + FastH3 6-step).
            # It only runs from 🚀 生成影片, like every other mode.
            has_reference = any(
                path.is_file() for path in self.reference_image_paths
            ) or (self.image_path is not None and self.image_path.is_file())
            if not has_reference:
                raise BotError("YUPI 需要一張參考圖：請先上傳圖片，再按「🚀 生成影片」。")
            self.run_yupi_generation(chat_id, fast=True)
            return True
        config = self.effective_config()
        input_image_path = (
            self.image_path
            if self.input_mode in {INPUT_MODE_IMAGE, INPUT_MODE_FL2VA}
            and self.image_path
            and self.image_path.is_file()
            else None
        )
        last_image_path = (
            self.last_image_path
            if self.input_mode == INPUT_MODE_FL2VA
            and self.last_image_path
            and self.last_image_path.is_file()
            else None
        )
        reference_image_paths = [
            path for path in self.reference_image_paths if path.is_file()
        ]
        reference_video_paths = [
            path for path in self.reference_video_paths if path.is_file()
        ]
        reference_audio_paths = [
            path for path in self.reference_audio_paths if path.is_file()
        ]
        if self.input_mode == INPUT_MODE_FL2VA and (
            input_image_path is None or last_image_path is None
        ):
            raise BotError("FL2VA 需要首幀和尾幀兩張圖片。")
        if self.input_mode == INPUT_MODE_REF2VA:
            require_ref2va_model()
            if not (
                reference_image_paths
                or reference_video_paths
                or reference_audio_paths
            ):
                raise BotError(
                    "Ref2VA 尚未收到參考素材，請先上傳圖片、影片或音訊。"
                )
        if self.total_seconds > MAX_SEGMENT_SECONDS:
            return self.start_long_generation(
                chat_id,
                config,
                prompt,
                self.total_seconds,
                input_image_path=input_image_path,
                last_image_path=last_image_path,
                reference_image_paths=reference_image_paths,
                reference_video_paths=reference_video_paths,
                reference_audio_paths=reference_audio_paths,
                generation_mode=self.input_mode,
            )
        return self.start_generation(
            chat_id,
            config,
            prompt,
            input_image_path=input_image_path,
            last_image_path=last_image_path,
            reference_image_paths=reference_image_paths,
            reference_video_paths=reference_video_paths,
            reference_audio_paths=reference_audio_paths,
            generation_mode=self.input_mode,
        )

    def resume_long_checkpoint(
        self,
        chat_id: str,
        checkpoint_id: Optional[str] = None,
        message_id: Optional[int] = None,
    ) -> None:
        """Resume the first unfinished shot from a persistent long-job checkpoint."""
        record = (
            self.checkpoint_for_id(checkpoint_id)
            if checkpoint_id
            else self.latest_long_checkpoint()
        )
        if record is None:
            self.send_safe(chat_id, "目前找不到可恢復的長片檢查點。")
            self.show_menu(chat_id, message_id)
            return
        path, payload = record
        next_index = int(payload.get("next_segment_index", 1))
        total = int(payload.get("segment_total", 0))
        if next_index > total:
            self.send_safe(chat_id, "這條長片已完成；請使用「📼 延續上一條長片」新增內容。")
            self.show_menu(chat_id, message_id)
            return
        with self.lock:
            if self.job is not None:
                self.send_safe(chat_id, "目前已有工作在生成，請先等待完成或按「⛔ 中止」。")
                return
        try:
            config = self._checkpoint_config(payload)
            shots = self._checkpoint_shots(payload)
            video_paths = self._checkpoint_video_paths(payload)
            generation_mode = normalize_input_mode(
                payload.get("generation_mode", INPUT_MODE_TEXT)
            )
            reference_image_paths = self._checkpoint_media_paths(
                payload, "reference_image_paths"
            )
            reference_video_paths = self._checkpoint_media_paths(
                payload, "reference_video_paths"
            )
            reference_audio_paths = self._checkpoint_media_paths(
                payload, "reference_audio_paths"
            )
            if generation_mode == INPUT_MODE_REF2VA:
                require_ref2va_model()
            if len(video_paths) != next_index - 1 or not video_paths:
                raise BotError("檢查點的已完成鏡頭數量不一致。")
            missing = [str(video_path) for video_path in video_paths if not video_path.is_file()]
            if missing:
                raise BotError(f"找不到已完成鏡頭：{missing[-1]}")
            last_latent = str(payload.get("last_context_latent_path", "")).strip() or None
            motion_context = bool(payload.get("motion_context_enabled")) and bool(last_latent)
            long_resolution = self._checkpoint_long_resolution(payload, config)
            resolution_fallbacks = self._checkpoint_resolution_fallbacks(payload)
            job = JobState(
                chat_id=chat_id,
                config=config,
                prompt=str(payload.get("prompt", "")),
                started_at=time.time(),
                task_type=normalize_task_type(payload.get("task_type", MODEL_H3)),
                yupi_fast=bool(payload.get("yupi_fast", False)),
                output_prefix=str(payload["output_prefix"]),
                long_base_prefix=str(payload["output_prefix"]),
                checkpoint_path=path,
                base_config=config,
                segment_total=len(shots),
                total_seconds=float(payload["total_seconds"]),
                shot_plan=shots,
                story_global_text=str(payload.get("story_global_text", "")),
                generation_mode=generation_mode,
                reference_image_paths=reference_image_paths,
                reference_video_paths=reference_video_paths,
                reference_audio_paths=reference_audio_paths,
                resume_from_segment=next_index,
                completed_video_paths=list(video_paths),
                initial_context_video_path=video_paths[-1],
                initial_context_latent_path=last_latent if motion_context else None,
                resume_motion_context=motion_context,
                long_resolution=long_resolution,
                resolution_fallbacks=resolution_fallbacks,
            )
            job.resume_event.set()
        except (BotError, KeyError, TypeError, ValueError) as exc:
            self.send_safe(chat_id, f"無法恢復長片：{exc}")
            self.show_menu(chat_id, message_id)
            return
        with self.lock:
            self.job = job
        self.touch_comfy_activity()
        self.send_safe(
            chat_id,
            f"已恢復長片檢查點：保留前 {next_index - 1}/{total} 鏡，"
            f"接下來從第 {next_index} 鏡繼續。\n"
            "會先要求 ComfyUI 釋放暫存顯存，再重新上傳上一鏡作為接續來源。",
        )
        thread = threading.Thread(
            target=self.run_long_job,
            args=(job,),
            name="minimax-long-resume",
            daemon=True,
        )
        thread.start()
        self.show_menu(chat_id, message_id, "已送出長片恢復工作")

    def request_extension_duration(
        self,
        chat_id: str,
        checkpoint_id: Optional[str] = None,
        message_id: Optional[int] = None,
    ) -> None:
        record = (
            self.checkpoint_for_id(checkpoint_id)
            if checkpoint_id
            else self.latest_long_checkpoint()
        )
        if record is None:
            self.send_safe(chat_id, "目前找不到可以延續的完整長片。")
            return
        path, payload = record
        if str(payload.get("status", "")) != "completed" and int(
            payload.get("next_segment_index", 1)
        ) <= int(payload.get("segment_total", 0)):
            self.send_checkpoint_actions(chat_id, path, payload, "這條長片還未全部完成")
            return
        self.extension_checkpoint_id = self._checkpoint_id(path)
        self.awaiting_extension_duration = True
        self.awaiting_extension_prompt = False
        self.awaiting_duration = False
        self.awaiting_prompt = False
        self.telegram.send_message(
            chat_id,
            "請輸入要在原片尾端新增的秒數（2 至 1800），例如 15、30 或 60。",
            reply_markup={
                "force_reply": True,
                "input_field_placeholder": "例如 30",
            },
        )
        self.show_menu(chat_id, message_id)

    def request_extension_prompt(self, chat_id: str) -> None:
        if self.extension_seconds is None or not self.extension_checkpoint_id:
            self.request_extension_duration(chat_id, self.extension_checkpoint_id)
            return
        self.awaiting_extension_duration = False
        self.awaiting_extension_prompt = True
        self.awaiting_prompt = False
        self.awaiting_duration = False
        self.telegram.send_message(
            chat_id,
            "請貼上尾端延續的提示詞。若新增超過 15 秒，請用時間軸或 SEGMENT 1／SEGMENT 2 分段。",
            reply_markup={
                "force_reply": True,
                "input_field_placeholder": "例如：她走出門口，鏡頭繼續向前推進",
            },
        )

    def start_extension_generation(
        self,
        chat_id: str,
        checkpoint_id: str,
        extra_seconds: float,
        prompt: str,
    ) -> None:
        record = self.checkpoint_for_id(checkpoint_id)
        if record is None:
            self.send_safe(chat_id, "找不到原長片檢查點，請重新按 /extend。")
            return
        path, payload = record
        if str(payload.get("status", "")) != "completed" and int(
            payload.get("next_segment_index", 1)
        ) <= int(payload.get("segment_total", 0)):
            self.send_checkpoint_actions(chat_id, path, payload, "原長片尚未完成")
            return
        try:
            extra_seconds = validate_total_seconds(extra_seconds)
            old_total = float(payload["total_seconds"])
            total_seconds = old_total + extra_seconds
            if total_seconds > MAX_TOTAL_SECONDS:
                raise BotError(f"延續後總片長不可超過 {MAX_TOTAL_SECONDS:g} 秒。")
            old_shots = self._checkpoint_shots(payload)
            old_paths = self._checkpoint_video_paths(payload)
            if len(old_paths) != len(old_shots):
                raise BotError("原長片的鏡頭檔案不完整，不能直接延續。")
            missing = [str(video_path) for video_path in old_paths if not video_path.is_file()]
            if missing:
                raise BotError(f"找不到原長片影片：{missing[-1]}")
            prompt = prompt.strip()
            if not prompt:
                raise BotError("延續提示詞不可為空白。")
            try:
                extension_plan = build_long_video_plan(prompt, extra_seconds)
            except BotError:
                if extra_seconds > MAX_SEGMENT_SECONDS:
                    raise
                extension_plan = LongVideoPlan(
                    "",
                    tuple(
                        split_scene_into_shots(
                            TimelineScene(0.0, extra_seconds, "延續", prompt)
                        )
                    ),
                    "extension",
                )
            offset_shots = tuple(
                ShotSpec(
                    round(shot.start_seconds + old_total, 3),
                    round(shot.end_seconds + old_total, 3),
                    shot.label,
                    shot.action,
                )
                for shot in extension_plan.shots
            )
            combined_shots = old_shots + offset_shots
            combined_global = "\n\n".join(
                part
                for part in (
                    str(payload.get("story_global_text", "")).strip(),
                    extension_plan.global_text.strip(),
                )
                if part
            )
            config = self._checkpoint_config(payload)
            long_resolution = self._checkpoint_long_resolution(payload, config)
            resolution_fallbacks = self._checkpoint_resolution_fallbacks(payload)
            generation_mode = normalize_input_mode(
                payload.get("generation_mode", INPUT_MODE_TEXT)
            )
            reference_image_paths = self._checkpoint_media_paths(
                payload, "reference_image_paths"
            )
            reference_video_paths = self._checkpoint_media_paths(
                payload, "reference_video_paths"
            )
            reference_audio_paths = self._checkpoint_media_paths(
                payload, "reference_audio_paths"
            )
            if generation_mode == INPUT_MODE_REF2VA:
                require_ref2va_model()
            last_latent = str(payload.get("last_context_latent_path", "")).strip() or None
            motion_context = bool(payload.get("motion_context_enabled")) and bool(last_latent)
            extension_prefix = (
                f"{payload['output_prefix']}/extension_{uuid.uuid4().hex[:12]}"
            )
            job = JobState(
                chat_id=chat_id,
                config=config,
                prompt=prompt,
                started_at=time.time(),
                task_type=normalize_task_type(payload.get("task_type", MODEL_H3)),
                yupi_fast=bool(payload.get("yupi_fast", False)),
                output_prefix=extension_prefix,
                long_base_prefix=extension_prefix,
                base_config=config,
                checkpoint_path=LONG_CHECKPOINT_DIR / f"{Path(extension_prefix).name}.json",
                segment_total=len(combined_shots),
                total_seconds=total_seconds,
                shot_plan=combined_shots,
                story_global_text=combined_global,
                generation_mode=generation_mode,
                reference_image_paths=reference_image_paths,
                reference_video_paths=reference_video_paths,
                reference_audio_paths=reference_audio_paths,
                resume_from_segment=len(old_paths) + 1,
                completed_video_paths=list(old_paths),
                initial_context_video_path=old_paths[-1],
                initial_context_latent_path=last_latent if motion_context else None,
                resume_motion_context=motion_context,
                long_resolution=long_resolution,
                resolution_fallbacks=resolution_fallbacks,
            )
            job.resume_event.set()
            self.save_long_checkpoint(
                job,
                old_paths,
                len(old_paths) + 1,
                motion_context,
                f"{extension_prefix}/motion_context/latent" if motion_context else None,
                last_latent,
                status="running",
            )
        except (BotError, KeyError, TypeError, ValueError) as exc:
            self.send_safe(chat_id, f"無法延續長片：{exc}")
            return
        with self.lock:
            if self.job is not None:
                self.send_safe(chat_id, "目前已有工作在生成，請先等待完成或按「⛔ 中止」。")
                return
            self.job = job
        self.touch_comfy_activity()
        self.send_safe(
            chat_id,
            f"已從原片尾端延續 {extra_seconds:g} 秒；原片 {old_total:g} 秒會保留，"
            f"完成後回傳合併的新影片（總長約 {total_seconds:g} 秒）。",
        )
        thread = threading.Thread(
            target=self.run_long_job,
            args=(job,),
            name="minimax-long-extension",
            daemon=True,
        )
        thread.start()
        self.show_menu(chat_id, notice="已送出長片延續工作")

    def start_long_generation(
        self,
        chat_id: str,
        config: GenerationConfig,
        prompt: str,
        total_seconds: float,
        input_image_path: Optional[Path] = None,
        last_image_path: Optional[Path] = None,
        reference_image_paths: Optional[list[Path]] = None,
        reference_video_paths: Optional[list[Path]] = None,
        reference_audio_paths: Optional[list[Path]] = None,
        generation_mode: str = INPUT_MODE_TEXT,
        task_type: str = MODEL_H3,
        yupi_fast: bool = False,
    ) -> bool:
        prompt = prompt.strip()
        if not prompt:
            self.send_safe(chat_id, "提示詞不可為空白。")
            return False
        total_seconds = validate_total_seconds(total_seconds)
        mode = normalize_input_mode(generation_mode)
        if mode == INPUT_MODE_FL2VA and (
            input_image_path is None
            or not input_image_path.is_file()
            or last_image_path is None
            or not last_image_path.is_file()
        ):
            self.send_safe(chat_id, "FL2VA 長片需要首幀和尾幀兩張圖片。")
            return False
        if mode == INPUT_MODE_REF2VA:
            if not (
                any(path.is_file() for path in (reference_image_paths or []))
                or any(path.is_file() for path in (reference_video_paths or []))
                or any(path.is_file() for path in (reference_audio_paths or []))
            ):
                self.send_safe(chat_id, "Ref2VA 尚未收到參考素材，請先上傳圖片、影片或音訊。")
                return False
            try:
                require_ref2va_model()
            except BotError as exc:
                self.send_safe(chat_id, str(exc))
                return False
        try:
            plan = build_long_video_plan(prompt, total_seconds)
        except BotError as exc:
            # The most common cause is a prompt/length mismatch: a short prose
            # script (or a hand-written paragraph) left selected while the video
            # length is still a long value. Name that explicitly, because the
            # raw parser error does not tell the user which control to change.
            hint = ""
            if (
                parse_timeline_prompt(prompt) is None
                and parse_segmented_prompt(prompt) is None
            ):
                hint = (
                    "\n\n目前的提示詞是一段散文、沒有任何時間軸。"
                    "如果它其實是短片腳本，請把片長改成 15 秒以內；"
                    "如果要 15 秒以上的長片，請改用時間軸格式"
                    "（按「✨ 一句話生成腳本」會自動產生正確的時間軸）。"
                )
            self.send_safe(chat_id, f"長片時間軸格式錯誤：{exc}{hint}")
            return False
        segment_total = len(plan.shots)
        if segment_total < 2:
            self.send_safe(chat_id, "長片時間軸至少需要兩個鏡頭。")
            return False
        if yupi_fast:
            output_root = YUPI_FAST_OUTPUT_PREFIX
        elif task_type == YUPI_TASK_TYPE:
            output_root = YUPI_OUTPUT_PREFIX
        else:
            output_root = OUTPUT_PREFIX
        batch_prefix = f"{output_root}/long_{uuid.uuid4().hex[:12]}"
        with self.lock:
            if self.job:
                self.send_safe(chat_id, "目前已有工作在生成，請先等待完成或使用 /cancel。")
                return False
            job = JobState(
                chat_id,
                config,
                prompt,
                time.time(),
                cancel_event=threading.Event(),
                output_prefix=batch_prefix,
                segment_total=segment_total,
                total_seconds=total_seconds,
                shot_plan=plan.shots,
                story_global_text=plan.global_text,
                input_image_path=input_image_path,
                last_image_path=last_image_path,
                reference_image_paths=list(reference_image_paths or []),
                reference_video_paths=list(reference_video_paths or []),
                reference_audio_paths=list(reference_audio_paths or []),
                task_type=normalize_task_type(task_type),
                generation_mode=mode,
                yupi_fast=yupi_fast,
                base_config=config,
                long_base_prefix=batch_prefix,
                checkpoint_path=LONG_CHECKPOINT_DIR
                / f"{Path(batch_prefix).name}.json",
                long_resolution=(config.width, config.height),
            )
            job.resume_event.set()
            self.job = job
        self.touch_comfy_activity()
        self.save_long_checkpoint(
            job,
            [],
            1,
            False,
            None,
            None,
            status="running",
        )
        format_text = "自然時間軸" if plan.source_format == "timeline" else "SEGMENT 分段"
        self.send_safe(
            chat_id,
            f"已解析{format_text}：共 {segment_total} 個連續鏡頭，"
            f"每鏡頭最多 {MAX_SHOT_SECONDS:g} 秒。\n"
            "後續鏡頭會使用上一鏡尾幀；每鏡重新生成原生音訊，避免重複上一鏡對白。\n"
            "若你另外上傳參考音訊，仍會按參考音訊模式生成。",
        )
        if task_type == YUPI_TASK_TYPE:
            self._send_progress_message(chat_id)
        thread = threading.Thread(target=self.run_long_job, args=(job,), daemon=True)
        thread.start()
        return True

    def send_partial_long_result(
        self,
        job: JobState,
        video_paths: list[Path],
        base_prefix: str,
    ) -> Optional[Path]:
        """Merge and send completed shots after a long job is cancelled."""
        if not video_paths:
            self.send_safe(job.chat_id, "長片已中止，尚未完成任何分段，沒有影片可以合成。")
            return None

        completed_count = len(video_paths)
        completed_shots = (
            job.shot_plan[:completed_count]
            if len(job.shot_plan) >= completed_count
            else tuple()
        )
        if completed_shots:
            completed_seconds = min(
                job.total_seconds,
                sum(shot.duration for shot in completed_shots),
            )
        else:
            completed_seconds = min(
                job.total_seconds,
                job.total_seconds * completed_count / max(job.segment_total, 1),
            )
        batch_name = base_prefix.rsplit("/", 1)[-1]
        output_path = (
            OUTPUT_DIR
            / base_prefix
            / f"{batch_name}_partial_{completed_count:02d}.mp4"
        )
        with job.progress_lock:
            job.progress_phase = "merging"
            job.progress_percent = min(
                99.0,
                completed_count / max(job.segment_total, 1) * 100.0,
            )
            job.progress_node_id = None
            job.progress_node_state = "merging completed segments"
        self.send_safe(
            job.chat_id,
            f"已中止，正在合成已完成的 {completed_count}/{job.segment_total} 段，"
            f"約 {completed_seconds:.2f} 秒。",
        )
        try:
            merge_config = job.base_config or job.config
            merge_completed_segments(
                video_paths,
                output_path,
                completed_seconds,
                shot_plan=completed_shots or None,
                output_size=(merge_config.width, merge_config.height),
            )
            with job.progress_lock:
                job.progress_phase = "uploading"
                job.progress_percent = 100.0
            model_label = (
                "🌙 YUPI"
                if job.task_type == YUPI_TASK_TYPE
                else "MiniMax H3 Turbo"
            )
            caption = (
                f"{model_label} 長片已提早中止，已合成部分結果\n"
                f"{completed_seconds:.2f} 秒 | {job.config.width}×{job.config.height} | "
                f"{job.config.steps} steps | {completed_count}/{job.segment_total} 段"
            )
            self.telegram.send_video(job.chat_id, output_path, caption)
            self.send_safe(
                job.chat_id,
                completion_report(
                    job,
                    time.time() - job.started_at,
                    duration_seconds=completed_seconds,
                    partial=True,
                ),
            )
            self.offer_upscale(
                job.chat_id,
                output_path,
                job.config.width,
                job.config.height,
                completed_seconds,
            )
            bot_log(f"partial long job sent {output_path}")
            return output_path
        except Exception as exc:
            bot_log(f"partial long job merge error: {exc}")
            self.send_safe(job.chat_id, f"已中止，但部分影片合成失敗：{exc}")
            return None

    def run_long_job(self, job: JobState) -> None:
        bot_log(
            f"long job start {job.total_seconds:.0f}s "
            f"{job.segment_total} segments {job.config.width}x{job.config.height} "
            f"steps={job.config.steps}"
        )
        partial_reported = False
        video_paths: list[Path] = list(job.completed_video_paths)
        base_prefix = job.long_base_prefix or job.output_prefix
        motion_context_legacy_layout = False

        def report_partial() -> None:
            nonlocal partial_reported
            if partial_reported:
                return
            partial_reported = True
            self.mark_long_checkpoint(job, "cancelled")
            self.send_partial_long_result(job, video_paths, base_prefix)

        def switch_to_tail_frame_continuation(reason: str) -> None:
            """Disable Motion Context and prepare the current segment to retry."""
            nonlocal motion_context_enabled
            nonlocal motion_context_legacy_layout
            nonlocal context_video_name, context_latent_path, latent_prefix
            motion_context_enabled = False
            motion_context_legacy_layout = True
            context_video_name = None
            context_latent_path = None
            latent_prefix = None
            job.audio_reference_name = None

            if job.continuation_image_path and job.continuation_image_path.is_file():
                return
            source_video = (
                video_paths[-1]
                if video_paths
                else job.initial_context_video_path
            )
            if source_video is None or not source_video.is_file():
                bot_log(
                    "long job: Motion Context fallback has no local previous "
                    f"video for a tail frame ({reason})"
                )
                return
            continuation_path = (
                CONTINUATION_DIR
                / f"{uuid.uuid4().hex}_layout_fallback.png"
            )
            try:
                job.continuation_image_path = extract_last_frame(
                    source_video, continuation_path
                )
            except Exception as exc:
                bot_log(f"long job: tail-frame fallback extraction failed: {exc}")

        try:
            self.ensure_comfyui_ready(job)
            base_config = job.base_config or job.config
            if job.resume_from_segment > 1 or job.initial_context_video_path is not None:
                try:
                    comfy_post("/free", {"unload_models": True, "free_memory": True})
                except BotError as exc:
                    bot_log(f"ComfyUI memory release before long resume unavailable: {exc}")
            if job.resume_motion_context is None:
                motion_context_enabled = (
                    getattr(self, "long_continuity", "motion_context")
                    == "motion_context"
                    and motion_context_nodes_available()
                )
            else:
                motion_context_enabled = bool(job.resume_motion_context)
                if motion_context_enabled and not motion_context_nodes_available():
                    motion_context_enabled = False
            # The installed pack can patch older PackedLayout constructors.
            # A frame_count parameter alone cannot establish incompatibility.
            # Let the node's runtime self-test decide; the existing execution
            # error handler still falls back for packs that reject the layout.
            if (
                job.generation_mode == INPUT_MODE_REF2VA
                and job.task_type != YUPI_TASK_TYPE
            ):
                if motion_context_enabled:
                    bot_log(
                        "Ref2VA long job: disabling AV Motion Context; using "
                        "Ref2VA for shot 1 and I2VA tail continuation afterwards"
                    )
                # Ref2VA is used only for the opening shot.  Later shots are
                # assembled by run_segment as I2VA from the previous tail
                # frame, so the reference pose cannot reset every segment.
                # YUPI is exempt: it keeps Motion Context and pins the previous
                # AV latent onto each shot's head instead.
                motion_context_enabled = False
            if getattr(self, "long_continuity", "motion_context") != "motion_context":
                self.send_safe(
                    job.chat_id,
                    "長片接續：目前使用「尾幀接續」——第 2 鏡起只接上一鏡尾幀，"
                    "每鏡重新生成原生音訊。",
                )
            if getattr(self, "long_continuity", "motion_context") == "motion_context":
                if job.task_type == YUPI_TASK_TYPE:
                    if motion_context_enabled:
                        self.send_safe(
                            job.chat_id,
                            "YUPI 長片：第 1 鏡用 AfterMidnight Ref2VA 錨定角色，"
                            "第 2 鏡起接續上一鏡的 AV latent 與尾幀"
                            "（Motion Context），最後合併為完整影片。",
                        )
                    else:
                        self.send_safe(
                            job.chat_id,
                            "YUPI 長片會以 AfterMidnight Ref2VA 逐鏡生成；"
                            "第 2 鏡起接續上一鏡尾幀，最後合併為完整影片。"
                            "（Motion Context 節點未就緒）",
                        )
                elif job.generation_mode == INPUT_MODE_REF2VA:
                    self.send_safe(
                        job.chat_id,
                        "已啟用長片接續：第 1 鏡使用 Ref2VA 多參考圖；第 2 鏡起改用 I2VA，"
                        "只接續上一鏡尾幀，避免每段重複參考姿勢。",
                    )
                elif motion_context_enabled:
                    self.send_safe(
                        job.chat_id,
                        "已啟用實驗性 H3 Motion Context：後續鏡頭會接續上一段的尾幀、"
                        "影像 latent 和音訊 latent。",
                    )
                elif motion_context_legacy_layout:
                    self.send_safe(
                        job.chat_id,
                        "目前 ComfyUI 使用舊版 H3 layout，與已安裝的 Motion Context pack 不相容；"
                        "這次自動改用穩定的尾幀接續，每鏡頭重新生成原生音訊。",
                    )
                else:
                    self.send_safe(
                        job.chat_id,
                        "Motion Context 節點未就緒，這次先使用穩定的尾幀接續；每鏡頭改用原生音訊。",
                    )
            context_video_name: Optional[str] = None
            context_latent_path: Optional[str] = None
            latent_prefix = (
                f"{base_prefix}/motion_context/latent"
                if motion_context_enabled
                else None
            )
            if job.initial_context_video_path is not None:
                if not job.initial_context_video_path.is_file():
                    raise BotError(
                        f"找不到接續來源影片：{job.initial_context_video_path}"
                    )
                if motion_context_enabled:
                    context_video_name = upload_video_to_comfy(
                        job.initial_context_video_path
                    )
                    context_latent_path = job.initial_context_latent_path
                    if not context_latent_path:
                        motion_context_enabled = False
                        latent_prefix = None
                        context_video_name = None
                        self.send_safe(
                            job.chat_id,
                            "上一鏡沒有可用的 AV latent，這次改用尾幀接續；後續鏡頭使用原生音訊。",
                        )
                if not motion_context_enabled:
                    # Keep only the previous video's visual tail frame. Do not
                    # feed its complete audio into the next shot: that can
                    # replay the previous dialogue or music. Explicit audio
                    # references uploaded by the user remain independent.
                    job.audio_reference_name = None
                    continuation_path = (
                        CONTINUATION_DIR
                        / f"{uuid.uuid4().hex}_resume_segment.png"
                    )
                    job.continuation_image_path = extract_last_frame(
                        job.initial_context_video_path, continuation_path
                    )
            if job.input_image_path is not None:
                self.send_safe(
                    job.chat_id,
                    "圖片長片會把圖片用作第一鏡首幀，後續鏡頭使用上一鏡尾幀接續。",
                )
            start_index = max(1, int(job.resume_from_segment))
            current_resolution = job.long_resolution or (
                base_config.width,
                base_config.height,
            )
            job.long_resolution = current_resolution
            for index in range(start_index, job.segment_total + 1):
                if not self.wait_for_resume(job):
                    report_partial()
                    return
                if job.cancel_event.is_set():
                    report_partial()
                    return
                job.segment_index = index
                shot = job.shot_plan[index - 1]
                job.segment_start_seconds = shot.start_seconds
                job.segment_end_seconds = shot.end_seconds
                generation_seconds = shot.duration
                if index < job.segment_total:
                    generation_seconds += SHOT_TRANSITION_SECONDS
                use_motion_context = motion_context_enabled and (
                    index > 1 or job.initial_context_video_path is not None
                )
                if use_motion_context:
                    # Motion Context pins a 22-frame head which is trimmed from
                    # the decoded result. Generate that head plus the requested
                    # shot duration so the delivered shot keeps its timeline.
                    generation_seconds += MOTION_CONTEXT_EXTRA_SECONDS
                while True:
                    current_width, current_height = current_resolution
                    job.long_resolution = current_resolution
                    job.config = parse_config(
                        [
                            str(current_width),
                            str(current_height),
                            str(base_config.steps),
                            str(generation_seconds),
                        ]
                    )
                    job.output_prefix = f"{base_prefix}/segment_{index:02d}"
                    self.send_safe(
                        job.chat_id,
                        f"長片鏡頭 {index}/{job.segment_total} 開始生成："
                        f"劇情 {shot.start_seconds:g}-{shot.end_seconds:g} 秒 | "
                        f"解析度 {resolution_label(current_width, current_height)} | "
                        f"模型約 {job.config.actual_seconds:.2f} 秒。",
                    )
                    try:
                        video_path = self.run_segment(
                            job,
                            announce=False,
                            motion_context=use_motion_context,
                            context_video_name=context_video_name,
                            context_latent_path=context_latent_path,
                            load_latent_clip_index=(index - 1 if use_motion_context else 0),
                            save_latent_prefix=latent_prefix,
                            save_latent_clip_index=index if latent_prefix else None,
                        )
                        break
                    except Exception as exc:
                        if (
                            use_motion_context
                            and is_motion_context_layout_error(exc)
                        ):
                            switch_to_tail_frame_continuation(str(exc))
                            generation_seconds = shot.duration
                            if index < job.segment_total:
                                generation_seconds += SHOT_TRANSITION_SECONDS
                            use_motion_context = False
                            self.save_long_checkpoint(
                                job,
                                video_paths,
                                index,
                                False,
                                None,
                                None,
                                status="running",
                                error=str(exc),
                            )
                            self.send_safe(
                                job.chat_id,
                                "⚠️ Motion Context 與目前 ComfyUI H3 layout 不相容；"
                                "已自動切換為尾幀接續，正在重試本鏡，前面已完成的鏡頭保留。",
                            )
                            continue
                        if job.cancel_event.is_set() or not is_cuda_oom_error(exc):
                            raise
                        next_resolution = next_lower_resolution(
                            current_width,
                            current_height,
                        )
                        if next_resolution is None:
                            self.send_safe(
                                job.chat_id,
                                f"長片鏡頭 {index} 在最低解析度 "
                                f"{resolution_label(current_width, current_height)} "
                                "仍然顯存不足，無法繼續。",
                            )
                            raise
                        old_label = resolution_label(current_width, current_height)
                        new_label = resolution_label(*next_resolution)
                        fallback_note = (
                            f"鏡頭 {index}：{old_label} → {new_label}"
                        )
                        job.resolution_fallbacks.append(fallback_note)
                        current_resolution = next_resolution
                        job.long_resolution = current_resolution
                        bot_log(
                            f"long segment {index} OOM at {old_label}; "
                            f"retrying at {new_label}: {exc}"
                        )
                        try:
                            comfy_post(
                                "/free",
                                {"unload_models": True, "free_memory": True},
                            )
                        except BotError as free_exc:
                            bot_log(
                                f"OOM memory release before retry unavailable: {free_exc}"
                            )
                        self.save_long_checkpoint(
                            job,
                            video_paths,
                            index,
                            motion_context_enabled,
                            latent_prefix,
                            context_latent_path,
                            status="running",
                            error=str(exc),
                        )
                        self.send_safe(
                            job.chat_id,
                            f"⚠️ 長片鏡頭 {index} 顯存不足，前 {index - 1} 段保留不變。\n"
                            f"自動降級：{old_label} → {new_label}\n"
                            "正在從本鏡重新生成，不會由第一段開始。",
                        )
                video_paths.append(video_path)
                job.completed_video_paths = list(video_paths)
                next_context_latent_path = (
                    f"{latent_prefix}_{index:05d}.safetensors"
                    if latent_prefix
                    else None
                )
                if index < job.segment_total:
                    if motion_context_enabled:
                        # Persist before uploading the next context. If upload
                        # or the next sample fails, the MP4 and latent are still
                        # enough to restart from this exact boundary.
                        context_latent_path = next_context_latent_path
                        self.save_long_checkpoint(
                            job,
                            video_paths,
                            index + 1,
                            motion_context_enabled,
                            latent_prefix,
                            context_latent_path,
                        )
                        context_video_name = upload_video_to_comfy(video_path)
                        if job.task_type == YUPI_TASK_TYPE:
                            # YUPI's graph always attaches a reference image
                            # (ref_image_0 is wired to its LoadImage node). Use
                            # the immediate previous tail frame rather than the
                            # original character reference, so the pose cannot
                            # reset every shot now that Motion Context carries
                            # the latent continuity.
                            job.audio_reference_name = None
                            previous_frame = job.continuation_image_path
                            continuation_path = (
                                CONTINUATION_DIR
                                / f"{uuid.uuid4().hex}_segment_{index:03d}.png"
                            )
                            job.continuation_image_path = extract_last_frame(
                                video_path, continuation_path
                            )
                            if previous_frame and previous_frame != continuation_path:
                                try:
                                    previous_frame.unlink()
                                except OSError:
                                    pass
                    else:
                        # Keep visual continuity from the immediate previous
                        # segment, but generate native audio for this shot.
                        # Reusing the complete previous audio can replay its
                        # dialogue or music in every subsequent shot.
                        job.audio_reference_name = None
                        previous_frame = job.continuation_image_path
                        continuation_path = (
                            CONTINUATION_DIR
                            / f"{uuid.uuid4().hex}_segment_{index:03d}.png"
                        )
                        job.continuation_image_path = extract_last_frame(
                            video_path, continuation_path
                        )
                        if previous_frame and previous_frame != continuation_path:
                            try:
                                previous_frame.unlink()
                            except OSError:
                                pass
                        self.save_long_checkpoint(
                            job,
                            video_paths,
                            index + 1,
                            motion_context_enabled,
                            latent_prefix,
                            None,
                        )
                else:
                    self.save_long_checkpoint(
                        job,
                        video_paths,
                        index + 1,
                        motion_context_enabled,
                        latent_prefix,
                        next_context_latent_path,
                    )
                self.send_safe(job.chat_id, f"長片鏡頭 {index}/{job.segment_total} 完成。")
                bot_log(f"segment {index}/{job.segment_total} done {video_path}")

            if job.cancel_event.is_set():
                report_partial()
                return
            batch_name = base_prefix.rsplit("/", 1)[-1]
            output_path = OUTPUT_DIR / base_prefix / f"{batch_name}.mp4"
            with job.progress_lock:
                job.progress_phase = "merging"
                job.progress_percent = 100.0
                job.progress_node_id = None
                job.progress_node_state = "merging"
            concat_videos(
                video_paths,
                output_path,
                job.total_seconds,
                shot_plan=job.shot_plan,
                output_size=(base_config.width, base_config.height),
            )
            self.mark_long_checkpoint(job, "completed")
            with job.progress_lock:
                job.progress_phase = "uploading"
            model_label = (
                "🌙 YUPI"
                if job.task_type == YUPI_TASK_TYPE
                else "MiniMax H3 Turbo"
            )
            caption = (
                f"{model_label} 長片完成\n{job.total_seconds:.0f} 秒 | "
                f"{base_config.width}×{base_config.height} | {base_config.steps} steps | "
                f"{job.segment_total} 鏡頭合併"
            )
            self.telegram.send_video(job.chat_id, output_path, caption)
            self.offer_tail_reference(job.chat_id, output_path)
            self.send_safe(
                job.chat_id,
                completion_report(
                    job,
                    time.time() - job.started_at,
                    duration_seconds=job.total_seconds,
                    config=base_config,
                ),
            )
            self.offer_upscale(
                job.chat_id,
                output_path,
                base_config.width,
                base_config.height,
                job.total_seconds,
                shutdown_after_choice=True,
            )
            bot_log(f"long job done {output_path}")
        except Exception as exc:
            if job.cancel_event.is_set():
                report_partial()
            else:
                self.send_safe(job.chat_id, f"長片生成失敗：{exc}")
                self.mark_long_checkpoint(job, "failed", str(exc))
                if job.checkpoint_path:
                    payload = self._read_checkpoint_payload(job.checkpoint_path)
                    if payload is not None:
                        self.send_checkpoint_actions(
                            job.chat_id,
                            job.checkpoint_path,
                            payload,
                            "已保存恢復檢查點",
                        )
            bot_log(f"long job error: {exc}")
            print(f"long generation error: {exc}", flush=True)
        finally:
            self.touch_comfy_activity()
            if job.continuation_image_path:
                try:
                    job.continuation_image_path.unlink()
                except OSError:
                    pass
            with self.lock:
                if self.job is job:
                    self.job = None
            self.on_job_finished(job.chat_id)

    def menu_text(self, notice: str = "") -> str:
        section = normalize_menu_section(
            getattr(self, "menu_section", MENU_MAIN)
        )
        section_titles = {
            MENU_MAIN: "主選單",
            MENU_INPUT: "提示詞／上傳素材",
            MENU_SETTINGS: "生成參數",
            MENU_MODE: "生成模式",
            MENU_DURATION: "片長／秒數",
            MENU_QUALITY: "解析度／步數",
            MENU_JOB: "當前任務",
            MENU_SYSTEM: "ComfyUI／LLM／系統",
            MENU_HISTORY: "歷史／長片／排隊",
        }
        current = self.settings
        prompt_status = f"已輸入（{len(self.prompt)} 字）" if self.prompt else "尚未輸入"
        mode_text = {
            INPUT_MODE_TEXT: "T2VA 文字生視頻",
            INPUT_MODE_IMAGE: "I2VA 圖片生視頻",
            INPUT_MODE_FL2VA: "FL2VA 首尾幀生視頻",
            INPUT_MODE_REF2VA: "Ref2VA 參考素材生視頻",
            INPUT_MODE_YUPI: "🌙 YUPI工作流（Ref2VA + FastH3 6步）",
        }.get(self.input_mode, "T2VA 文字生視頻")
        image_status = "已收到" if self.image_path and self.image_path.is_file() else "未收到"
        media_status = (
            f"首幀：{'已上傳' if self.image_path and self.image_path.is_file() else '未上傳'}；"
            f"尾幀：{'已上傳' if self.last_image_path and self.last_image_path.is_file() else '未上傳'}"
            if self.input_mode == INPUT_MODE_FL2VA
            else (
                f"參考圖 {len(self.reference_image_paths)} 張／"
                f"參考片 {len(self.reference_video_paths)} 段／"
                f"參考音訊 {len(self.reference_audio_paths)} 段"
                if is_ref2va_like(self.input_mode)
                else image_status
            )
        )
        prefix = f"{notice}\n\n" if notice else ""
        if self.total_seconds > MAX_SEGMENT_SECONDS:
            duration_text = f"長片 {self.total_seconds:.0f} 秒"
        else:
            effective = self.effective_config()
            duration_text = f"短片 {effective.actual_seconds:.2f} 秒"
        if self._shutdown_pending:
            shutdown_text = "完成後關機：倒數中（可取消）"
        elif self.shutdown_after_generation:
            shutdown_text = "完成後關機：已開啟（只對長片生效）"
        else:
            shutdown_text = "完成後關機：關閉"
        with self.lock:
            active_job = self.job
            queue_count = len(self.story_queue)
            pending_upscale = self.pending_upscale is not None
        if active_job is None:
            job_text = "當前任務：無"
        elif active_job.pause_requested.is_set():
            job_text = "當前任務：已暫停／等待播放"
        elif active_job.segment_total > 1:
            job_text = (
                f"當前任務：長片第 {active_job.segment_index}/"
                f"{active_job.segment_total} 段"
            )
        else:
            job_text = "當前任務：生成中"
        queue_text = f"故事排隊：{queue_count} 個等待中"
        if pending_upscale and queue_count:
            queue_text += "（等待放大選擇後接續）"
        if COMFY_IDLE_SHUTDOWN_SECONDS > 0:
            idle_shutdown_text = f"閒置 {COMFY_IDLE_SHUTDOWN_SECONDS / 60:g} 分鐘關閉"
        else:
            idle_shutdown_text = "閒置自動關閉：關閉"
        section_hints = {
            MENU_MAIN: "模式、提示詞、任務和系統按鈕直接顯示；只有片長、解析度和 steps 收納在生成參數。",
            MENU_INPUT: "可直接發圖片、影片、音訊或 TXT；Ref2VA 素材完成後按確認。",
            MENU_SETTINGS: "這裡集中調整片長、解析度、steps、長片接續方式；其他功能仍在主選單。",
            MENU_MODE: "T2VA／I2VA／FL2VA／Ref2VA／YUPI 會在生成時使用對應接線。",
            MENU_DURATION: "超過 15 秒會按提示詞時間軸自動分段。",
            MENU_QUALITY: "解析度越高越清晰，也越容易需要更多顯存。",
            MENU_JOB: "生成中的任務可以查看進度、暫停、繼續或中止。",
            MENU_SYSTEM: "這裡管理 ComfyUI、本地 LLM、溫度、顯存模式和自動關機。",
            MENU_HISTORY: "可以恢復失敗鏡頭、延續長片或管理故事排隊。",
        }
        two_pass_text = (
            "兩段式上採：開啟（半解析度→放大→精修）"
            if bool(getattr(self, "latent_upscale", LATENT_UPSCALE_ENABLED))
            else "兩段式上採：關閉（單段直出）"
        )
        fused_active = (
            getattr(self, "h3_profile", H3_PROFILE_FUSED) == H3_PROFILE_FUSED
        )
        profile_text = (
            f"融合加速（{FUSED_PROFILE_STEPS} steps，SLA {'開' if h3_sla_available() else '關'}）"
            if fused_active
            else f"經典（{current.steps} steps，turbo LoRA）"
        )
        menu = (
            f"{prefix}🎬 MiniMax H3 Turbo 控制面板\n"
            f"目前頁面：{section_titles.get(section, '主選單')}\n\n"
            f"模式：{mode_text}\n"
            f"模型：{profile_text}\n"
            f"參數：{resolution_label(current.width, current.height)} | "
            f"{FUSED_PROFILE_STEPS if fused_active else current.steps} steps | "
            f"{duration_text}\n"
            f"素材：{media_status}\n"
            f"提示詞：{prompt_status}\n"
            f"任務：{job_text}\n"
            f"排隊：{queue_text}\n"
            f"顯存：{comfyui_vram_mode_label(self.comfyui_vram_mode())}\n"
            f"{two_pass_text}\n"
            f"{shutdown_text}；{idle_shutdown_text}\n\n"
            f"{section_hints.get(section, section_hints[MENU_MAIN])}"
        )
        return menu

    def finalize_upscale_choice(
        self, chat_id: str, pending: PendingUpscale
    ) -> None:
        if not pending.shutdown_after_choice or not self.shutdown_after_generation:
            return
        with self.lock:
            queued_count = len(self.story_queue)
        if queued_count:
            self.send_safe(
                chat_id,
                f"排隊還有 {queued_count} 個故事，已跳過這次自動關機；全部完成後再自行關閉 ComfyUI。",
            )
            return
        with self.lock:
            if self._shutdown_pending:
                return
            self._shutdown_pending = True
        try:
            schedule_windows_shutdown()
        except BotError as exc:
            with self.lock:
                self._shutdown_pending = False
            self.send_safe(chat_id, f"放大後安排關機失敗：{exc}")
            return
        self.send_safe(
            chat_id,
            f"已安排放大後 {SHUTDOWN_DELAY_SECONDS} 秒關機；如要取消請按選單按鈕。",
        )

    def schedule_shutdown_if_enabled(self, job: JobState) -> None:
        with self.lock:
            should_schedule = (
                not job.cancel_event.is_set()
                and job.segment_total > 1
                and self.shutdown_after_generation
                and not self._shutdown_pending
            )
            if should_schedule:
                self._shutdown_pending = True
        if not should_schedule:
            return
        try:
            schedule_windows_shutdown()
        except BotError as exc:
            with self.lock:
                self._shutdown_pending = False
            self.send_safe(job.chat_id, f"長片已完成，但排程關機失敗：{exc}")
            return
        self.send_safe(
            job.chat_id,
            f"長片已傳送完成，電腦將在 {SHUTDOWN_DELAY_SECONDS} 秒後關機。"
            "如需取消，請按面板的「取消即將關機」或輸入 /cancel_shutdown。",
        )

    def cancel_scheduled_shutdown(
        self, chat_id: str, message_id: Optional[int] = None
    ) -> None:
        with self.lock:
            pending = self._shutdown_pending
        if pending:
            try:
                cancel_windows_shutdown()
            except BotError as exc:
                self.send_safe(chat_id, str(exc))
                return
            with self.lock:
                self._shutdown_pending = False
        self.shutdown_after_generation = False
        self.save_settings()
        self.show_menu(chat_id, message_id, "自動關機已取消")

    def show_progress(
        self,
        chat_id: str,
        message_id: Optional[int] = None,
    ) -> None:
        """Show one auto-refreshing progress message at the chat bottom."""
        self._send_progress_message(chat_id)
        self.show_menu(chat_id, message_id)

    def show_menu(
        self,
        chat_id: str,
        message_id: Optional[int] = None,
        notice: str = "",
        force_new: bool = False,
        section: Optional[str] = None,
    ) -> None:
        if section is not None:
            self.menu_section = normalize_menu_section(section)
        elif force_new:
            self.menu_section = MENU_MAIN
        text = self.menu_text(notice)
        markup = self.menu_markup(self.menu_section)
        target_message_id = None if force_new else (message_id or self.menu_message_id)
        try:
            if target_message_id is None:
                result = self.telegram.send_message(chat_id, text, reply_markup=markup)
                if isinstance(result, dict) and result.get("message_id"):
                    self.menu_message_id = int(result["message_id"])
                    self.ensure_control_panel_shortcut(chat_id)
            else:
                self.telegram.edit_message_text(
                    chat_id, target_message_id, text, reply_markup=markup
                )
                self.menu_message_id = target_message_id
        except BotError as exc:
            if target_message_id is not None and "not modified" in str(exc).lower():
                return
            if target_message_id is not None:
                self.menu_message_id = None
                try:
                    result = self.telegram.send_message(
                        chat_id, text, reply_markup=markup
                    )
                    if isinstance(result, dict) and result.get("message_id"):
                        self.menu_message_id = int(result["message_id"])
                        self.ensure_control_panel_shortcut(chat_id)
                    return
                except BotError:
                    pass
            self.send_safe(chat_id, f"選單更新失敗：{exc}")

    def ensure_control_panel_shortcut(self, chat_id: str) -> None:
        """Install a persistent bottom keyboard instead of pinning messages."""
        if self.control_keyboard_sent:
            return
        try:
            self.telegram.send_message(
                chat_id,
                "控制面板快捷入口已啟用；新訊息很多時，按下方按鈕即可返回面板。",
                reply_markup=control_panel_reply_markup(),
            )
            self.control_keyboard_sent = True
        except BotError as exc:
            bot_log(f"control panel shortcut unavailable: {exc}")

    def request_duration(self, chat_id: str) -> None:
        self.awaiting_duration = True
        self.awaiting_prompt = False
        self.awaiting_queue_prompt = False
        self.awaiting_script_idea = False
        self.telegram.send_message(
            chat_id,
            "請輸入總片長秒數（2 至 1800），例如 37、180、600 或 1800。",
            reply_markup={
                "force_reply": True,
                "input_field_placeholder": "例如 600",
            },
        )

    def request_prompt(self, chat_id: str, note: str = "") -> None:
        self.awaiting_duration = False
        self.awaiting_prompt = True
        self.awaiting_queue_prompt = False
        self.awaiting_script_idea = False
        text = (
            "請下一則訊息貼上提示詞，可以是多行文字。Bot 會自動讀取腳本時間軸最大秒數，"
            "完成後回到面板按「生成影片」。"
        )
        if note:
            text = note + "\n\n" + text
        self.telegram.send_message(
            chat_id,
            text,
            reply_markup={
                "force_reply": True,
                "input_field_placeholder": "貼上影片提示詞",
            },
        )

    # --- ✨ script generator ------------------------------------------------
    def request_script_idea(self, chat_id: str, note: str = "") -> None:
        """Ask for a one-line idea; the duration may be embedded in the line."""
        if not SCRIPT_GEN_ENABLED:
            self.send_safe(
                chat_id,
                "腳本生成器已停用（MINIMAX_SCRIPT_GEN=0）。",
            )
            return
        self.awaiting_script_idea = True
        self.awaiting_prompt = False
        self.awaiting_duration = False
        self.awaiting_queue_prompt = False
        text = (
            "✨ 一句話生成完整腳本\n\n"
            "直接描述你想拍的畫面，開頭或結尾加上秒數即可：\n"
            "  • 60秒 下雨的東京街頭，一個女生錯過末班車\n"
            "  • 30秒 貓在窗邊發呆，午後陽光\n"
            "  • 2分鐘 賽博龐克機車追逐\n\n"
            f"沒寫秒數就用目前的 {self.duration_label(self.total_seconds)}。\n"
            f"腳本 LLM：{script_llm_display_name()}（輸入 /scriptllm 可切換）\n"
            f"腳本模板：{script_template_display_name()}（輸入 /scripttemplate 可切換）\n"
            f"腳本語言：{SCRIPT_LANG_LABEL[normalize_script_lang(getattr(self, 'script_lang', SCRIPT_LANG_DEFAULT))]}"
            "（輸入 /lang 可切換）\n"
            "生成約需 20–60 秒，完成後可一鍵採用或重新生成。"
        )
        if note:
            text = note + "\n\n" + text
        self.telegram.send_message(
            chat_id,
            text,
            reply_markup={
                "force_reply": True,
                "input_field_placeholder": "例如：60秒 下雨的車站，女生錯過末班車",
            },
        )

    def set_script_lang(self, chat_id: str, value: str, message_id: Optional[int] = None) -> None:
        """Switch the generated-script output language and persist it."""
        self.script_lang = normalize_script_lang(value)
        self.save_settings()
        label = SCRIPT_LANG_LABEL[self.script_lang]
        extra = (
            "\n（H3 以英文語料為主，英文運鏡描述通常最穩定；簡中完全可用。）"
            if self.script_lang == SCRIPT_LANG_ZH
            else "\n（英文是 H3 訓練語料的主要語言，動態與運鏡描述最穩定。）"
        )
        notice = f"腳本語言已切換為：{label}{extra}"
        if message_id is not None:
            self.show_menu(chat_id, message_id, notice)
        else:
            self.show_menu(chat_id, notice=notice)

    def switch_script_llm(self, chat_id: str, value: str, message_id: Optional[int] = None) -> None:
        """Switch which LLM writes the scripts and persist it."""
        self.script_llm = normalize_script_llm(value)
        set_script_llm_provider(self.script_llm)
        self.save_settings()
        if self.script_llm == SCRIPT_LLM_COMMANDCODE:
            extra = (
                f"\n雲端引擎：Command Code，模型 {COMMANDCODE_MODEL}"
                "（不佔本機顯存、不需要啟動 llama-server）。"
            )
            if not commandcode_api_key():
                extra += (
                    f"\n⚠️ 未找到 API key：請設定環境變數 {COMMANDCODE_API_KEY_ENV}。"
                )
        else:
            extra = (
                "\n本機引擎：llama.cpp（127.0.0.1:19092）；"
                "生成腳本前 Bot 會自動確保它已啟動。"
            )
        notice = f"🧠 腳本 LLM 已切換為：{script_llm_display_name()}{extra}"
        if message_id is not None:
            self.show_menu(chat_id, message_id, notice)
        else:
            self.show_menu(chat_id, notice=notice)

    def switch_script_template(self, chat_id: str, value: str, message_id: Optional[int] = None) -> None:
        """Switch the script-guidance template (成人版 / 一般版) and persist it."""
        self.script_template = normalize_script_template(value)
        set_script_template(self.script_template)
        self.save_settings()
        if self.script_template == SCRIPT_TEMPLATE_GENERAL:
            extra = (
                "\n一般版：使用內建非成人模板，唔會讀 script_prompt.txt。"
                "\n（想自訂一般版規則：建立 script_prompt_general.txt，內容會取代內建模板。）"
            )
        else:
            extra = "\n成人版：繼續使用你的自訂指令檔 script_prompt.txt（原檔未改動）。"
        notice = f"📄 腳本模板已切換為：{script_template_display_name()}{extra}"
        if message_id is not None:
            self.show_menu(chat_id, message_id, notice)
        else:
            self.show_menu(chat_id, notice=notice)

    def show_script_prompt_file(self, chat_id: str, message_id: Optional[int] = None) -> None:
        """Show the custom-instruction file: path, status and its live content."""
        created = ensure_script_prompt_file()
        custom, status = load_custom_script_instructions()
        lines = [
            "📝 自訂指令（會附加到每次生成的系統提示後段）",
            "",
            f"狀態：{status}",
            "",
            "以 # 開頭的行是註解，不會送出。",
            "改完立即生效，不必重啟 Bot。",
        ]
        if created:
            lines += ["", created]
        if custom:
            lines += ["", "── 目前實際送出的內容 ──", custom]
        else:
            lines += ["", "目前沒有任何自訂指令。"]

        markup = {
            "inline_keyboard": [
                [{"text": "✏️ 編輯（整段取代）", "callback_data": "script_file:edit"}],
                [{"text": "➕ 追加一條規則", "callback_data": "script_file:append"}],
                [
                    {"text": "🗑 清空", "callback_data": "script_file:clear"},
                    {"text": "↩️ 還原上一版", "callback_data": "script_file:undo"},
                ],
                [{"text": "🔄 重新整理", "callback_data": "script_file"}],
            ]
        }
        text = "\n".join(lines)
        try:
            if message_id is not None:
                self.telegram.edit_message_text(chat_id, message_id, text, reply_markup=markup)
            else:
                self.telegram.send_message(chat_id, text, reply_markup=markup)
        except BotError:
            self.send_long_text(chat_id, text)

    def request_custom_prompt(self, chat_id: str, mode: str) -> None:
        """Ask for custom instruction text typed straight into Telegram."""
        self.awaiting_custom_prompt = mode  # "replace" | "append"
        self.awaiting_prompt = False
        self.awaiting_script_idea = False
        self.awaiting_duration = False
        self.awaiting_queue_prompt = False
        current, _ = load_custom_script_instructions()
        if mode == "append":
            hint = (
                "請輸入要「追加」的規則，一則訊息可以寫多行。\n"
                f"目前已有 {len(current)} 字元，新的會接在後面。"
                if current
                else "目前沒有內容，這則會成為第一條規則。"
            )
            placeholder = "例如：運鏡一律緩慢，不要手持晃動"
        else:
            hint = (
                "請輸入「完整」的自訂指令內容（會整段取代現有內容）。\n"
                "一則訊息可以寫多行；不需要寫 # 註解。\n"
                f"目前內容 {len(current)} 字元，送出後可用「↩️ 還原上一版」復原。"
            )
            placeholder = "例如：運鏡一律緩慢。不要出現浮水印。"
        self.telegram.send_message(
            chat_id,
            f"📝 {hint}\n\n隨時可用 /cancel 取消。",
            reply_markup={"force_reply": True, "input_field_placeholder": placeholder},
        )

    def handle_custom_prompt_text(self, chat_id: str, text: str) -> None:
        """Apply text typed in Telegram to the custom instruction file."""
        mode = str(getattr(self, "awaiting_custom_prompt", "") or "")
        self.awaiting_custom_prompt = ""

        body = (text or "").strip()
        if not body:
            self.send_safe(chat_id, "內容是空的，已取消。")
            self.show_script_prompt_file(chat_id)
            return

        if mode == "append":
            current, _ = load_custom_script_instructions()
            combined = f"{current}\n{body}" if current else body
            ok, note = save_custom_script_instructions(combined, "由 Telegram 追加")
        else:
            ok, note = save_custom_script_instructions(body, "由 Telegram 編輯")

        if not ok:
            self.send_safe(chat_id, f"⚠️ {note}")
            return

        # Prove the file is genuinely in effect by showing what will be sent.
        saved, status = load_custom_script_instructions()
        self.send_long_text(
            chat_id,
            f"✅ {note}（{status}）\n\n"
            "── 之後每次生成都會送出 ──\n"
            f"{saved}\n\n"
            "立即生效，不需要重啟。",
        )
        self.show_script_prompt_file(chat_id)

    def handle_script_idea(
        self, chat_id: str, text: str, seconds_override: Optional[float] = None
    ) -> None:
        """Kick off script generation in the background."""
        if getattr(self, "chain_awaiting_idea", False):
            # The text is the first-clip idea for a pending auto-chain run.
            self.chain_awaiting_idea = False
            self.awaiting_script_idea = False
            self.start_chain(
                chat_id,
                list(self.chain_pending_durations or []),
                text.strip(),
            )
            return
        if self.script_busy:
            self.send_safe(chat_id, "上一個腳本還在生成中，請稍候。")
            return
        seconds, idea = parse_idea_duration(text)
        if seconds_override is not None:
            seconds = float(seconds_override)
        elif seconds is None:
            seconds = float(self.total_seconds)
        if not idea:
            self.send_safe(chat_id, "請描述你想拍的畫面，例如：60秒 下雨的車站，女生錯過末班車")
            return

        self.awaiting_script_idea = False
        self.script_idea = idea
        self.script_seconds = float(seconds)
        self.script_busy = True
        # A brand-new idea starts a fresh editing session; dropping the history
        # prevents "undo" from jumping back into an unrelated script.
        self.script_draft = None
        self.script_history = []

        if get_script_llm_provider() == SCRIPT_LLM_COMMANDCODE:
            lead = f"✨ 正在用 Command Code（{COMMANDCODE_MODEL}）生成"
            wait = "約 5–30 秒"
        else:
            lead = "✨ 正在用本機 LLM 生成"
            wait = "首次約需 20–60 秒"
        self.send_safe(
            chat_id,
            f"{lead} {self.duration_label(seconds)} 腳本…\n"
            f"想法：{idea}\n\n{wait}，請稍候。",
        )
        thread = threading.Thread(
            target=self._script_worker,
            args=(chat_id,),
            name="h3-script-generator",
            daemon=True,
        )
        thread.start()

    def _ensure_llm_ready(self, chat_id: str) -> None:
        """Make sure the local LLM can answer, waiting out a model load.

        Right after a generation the Bot restarts the LLM itself and it spends
        about a minute reading ~22GB of weights (503 "Loading model"). Saying
        "not started, starting it" during that window is both wrong and alarming,
        so the two states are reported separately.
        """
        if get_script_llm_provider() == SCRIPT_LLM_COMMANDCODE:
            # Cloud engine selected: there is nothing local to start or wait
            # for, and the llama.cpp process must not be woken up for this.
            return
        if llama_is_online():
            return
        # NOTE: the VRAM guard against a model-holding ComfyUI lives inside
        # start_llama_process(), so it covers this path and every other entry
        # point from one place.
        if llama_server_responding():
            self.send_safe(
                chat_id,
                "🧠 本機 LLM 正在載入模型（剛生成完會自動重啟），請稍候…",
            )
        else:
            self.send_safe(chat_id, "🧠 本機 LLM 未啟動，正在啟動…")
            try:
                self.send_safe(chat_id, start_llama_process())
            except (BotError, OSError) as exc:
                raise BotError(f"無法啟動本機 LLM：{exc}") from exc
        deadline = time.time() + 300
        while time.time() < deadline:
            if llama_is_online():
                break
            time.sleep(3)
        else:
            raise BotError("本機 LLM 在 300 秒內沒有就緒。")
        self.send_safe(chat_id, "🧠 本機 LLM 已就緒。")

    def _script_progress(self, chat_id: str):
        """Callback shared by generation and refinement attempts."""

        def progress(attempt: int, total: int, last_error: str) -> None:
            if attempt > 1:
                self.send_safe(
                    chat_id,
                    f"🔁 第 {attempt}/{total} 次嘗試（上次：{last_error[:120]}）",
                )
            self.telegram.send_chat_action(chat_id, "typing")

        return progress

    def _script_lang(self) -> str:
        return normalize_script_lang(
            getattr(self, "script_lang", SCRIPT_LANG_DEFAULT)
        )

    def _draft_seconds(self) -> float:
        """Duration to validate the current draft against.

        A hand-edited timeline states its own length, which wins; otherwise fall
        back to the length the draft was written for.
        """
        detected = detect_prompt_total_seconds(self.script_draft or "")
        if detected is not None:
            return float(detected)
        return float(getattr(self, "script_seconds", 0) or self.total_seconds)

    def _set_script_draft(self, script: str, label: str) -> None:
        """Replace the draft, keeping the previous revision for undo."""
        previous = self.script_draft
        if previous and previous != script:
            self.script_history.append(previous)
            # Bounded so a long editing session cannot grow without limit.
            if len(self.script_history) > 30:
                del self.script_history[0]
        self.script_draft = script
        self.script_last_action = label

    def _script_worker(self, chat_id: str) -> None:
        """Generate, self-repair and report a script without blocking polling."""
        progress = self._script_progress(chat_id)
        try:
            self._ensure_llm_ready(chat_id)
            script, log = generate_h3_script(
                self.script_idea,
                self.script_seconds,
                self.input_mode,
                on_progress=progress,
                lang=self._script_lang(),
                continuity=self.script_continuity,
            )
        except (BotError, OSError) as exc:
            self.script_busy = False
            bot_log(f"script generator failed: {exc}")
            self.send_safe(chat_id, f"❌ 腳本生成失敗：{exc}")
            return

        self.script_busy = False
        # Route through _set_script_draft so a regenerate is also undoable.
        self._set_script_draft(script, "重新生成")
        bot_log("script generator: " + " | ".join(log))
        self.show_script_draft(chat_id, script, "\n".join(log))

    def send_long_text(self, chat_id: str, text: str) -> None:
        """Send text longer than Telegram's 4096 character message limit."""
        limit = 3500
        remaining = text
        while remaining:
            if len(remaining) <= limit:
                chunk, remaining = remaining, ""
            else:
                split = remaining.rfind("\n\n", 0, limit)
                if split < limit // 2:
                    split = remaining.rfind("\n", 0, limit)
                if split < limit // 2:
                    split = limit
                chunk, remaining = remaining[:split], remaining[split:].lstrip("\n")
            self.send_safe(chat_id, chunk)

    def show_script_draft(self, chat_id: str, script: str, log: str = "") -> None:
        """Present the draft with accept, repeated-edit and discard actions."""
        seconds = self._draft_seconds()
        detected = detect_prompt_total_seconds(script)
        revision = len(self.script_history)
        title = f"✨ 腳本草稿（第 {revision + 1} 版，{self.duration_label(detected or self.script_seconds)}）"
        if log:
            title += f"\n{log}"
        title += "\n" + "─" * 18
        self.send_long_text(chat_id, f"{title}\n\n{script}")

        ok, message = _script_accepts(script, seconds)
        mode_note = {
            INPUT_MODE_IMAGE: "🖼 I2VA：已依「不重述圖片外觀」規則生成",
            INPUT_MODE_FL2VA: "🎬 FL2VA：只描述首尾幀之間的中間過程",
            INPUT_MODE_REF2VA: "📚 Ref2VA：已鎖人物外貌、場景寫在提示詞內",
        }.get(self.input_mode, "📝 T2VA：完整時間軸")

        status = f"✅ 格式檢查通過：{message}" if ok else f"⚠️ 格式檢查未通過：{message}"
        rows: list[list[dict[str, str]]] = [
            [{"text": "✅ 採用並生成影片", "callback_data": "script:accept_run"}],
            [{"text": "📥 只採用（回面板）", "callback_data": "script:accept"}],
            [
                {"text": "🤖 指令修改（AI）", "callback_data": "script:refine"},
                {"text": "✏️ 手動編輯", "callback_data": "script:edit"},
            ],
            [
                {"text": "➕ 追加內容", "callback_data": "script:append"},
                {"text": "🔄 重新生成", "callback_data": "script:regen"},
            ],
        ]
        if self.script_history:
            rows.append(
                [
                    {
                        "text": f"↩️ 回到上一版（共 {len(self.script_history)} 版可退）",
                        "callback_data": "script:undo",
                    }
                ]
            )
        rows.append([{"text": "❌ 放棄", "callback_data": "script:cancel"}])

        hint = (
            "可以一直改下去：\n"
            "  • 🤖 指令修改 — 用一句話叫 AI 改（例如「運鏡再慢一點」）\n"
            "  • ✏️ 手動編輯 — 直接貼上你要的完整內容\n"
            "  • ➕ 追加內容 — 在現有內容後面補一段\n"
            "每改一次都會自動保留上一版，可隨時退回。"
        )
        if not ok:
            hint += (
                "\n\n⚠️ 目前格式未通過，直接生成會被拒絕。"
                "長片必須是連續、無缺口的時間軸；可手動修正或按上一版退回。"
            )
        self.telegram.send_message(
            chat_id,
            f"{mode_note}\n字數 {len(script)}｜{status}",
            reply_markup={"inline_keyboard": rows},
        )
        self.send_safe(chat_id, hint)

    def request_script_edit(self, chat_id: str, mode: str) -> None:
        """Ask for replacement or additional script text typed in Telegram."""
        if not self.script_draft:
            self.send_safe(chat_id, "目前沒有草稿，請先按「✨ 生成腳本」。")
            return
        self.awaiting_script_edit = mode  # "replace" | "append"
        self.awaiting_script_refine = False
        self.awaiting_script_idea = False
        self.awaiting_custom_prompt = ""
        if mode == "append":
            hint = (
                "請輸入要「追加」到草稿後面的內容，一則訊息可多行。\n"
                "現有內容會保留，新的接在後面。"
            )
            placeholder = "例如：結尾再加一個鏡頭，雨停了"
        else:
            hint = (
                "請貼上「完整」的腳本內容（會整段取代目前草稿）。\n"
                "一則訊息可多行；送出後可用「↩️ 回到上一版」復原。"
            )
            placeholder = "貼上完整腳本"
        self.telegram.send_message(
            chat_id,
            f"✏️ {hint}\n\n隨時可用 /cancel 取消。",
            reply_markup={"force_reply": True, "input_field_placeholder": placeholder},
        )

    def handle_script_edit_text(self, chat_id: str, text: str) -> None:
        """Apply hand-typed script text and re-check it against the validator."""
        mode = str(getattr(self, "awaiting_script_edit", "") or "")
        self.awaiting_script_edit = ""
        body = (text or "").strip()
        if not body:
            self.send_safe(chat_id, "內容是空的，已取消。")
            self.show_script_draft(chat_id, self.script_draft or "")
            return

        draft = self.script_draft or ""
        if mode == "append":
            new = f"{draft}\n{body}".strip() if draft else body
            label = "手動追加"
        else:
            new = body
            label = "手動編輯"

        self._set_script_draft(new, label)
        # A hand edit is accepted even when it breaks the format: the user may be
        # fixing a long timeline in several steps, and blocking the save would
        # make that impossible. The draft view states the result plainly instead.
        self.show_script_draft(chat_id, new, f"{label}後重新檢查：")

    def request_script_refine(self, chat_id: str) -> None:
        """Ask for a one-line instruction that the LLM applies to the draft."""
        if not self.script_draft:
            self.send_safe(chat_id, "目前沒有草稿，請先按「✨ 生成腳本」。")
            return
        self.awaiting_script_refine = True
        self.awaiting_script_edit = ""
        self.awaiting_script_idea = False
        self.awaiting_custom_prompt = ""
        self.telegram.send_message(
            chat_id,
            "🤖 請用一句話說明要怎麼改（AI 會重寫整份腳本）：\n\n"
            "  • 運鏡再慢一點，多用推近\n"
            "  • 把時間改到清晨，光線更冷\n"
            "  • 第二幕太長，動作拆細一點\n"
            "  • 結尾改成她轉頭看鏡頭\n\n"
            f"要改幾次都可以。隨時可用 /cancel 取消。",
            reply_markup={
                "force_reply": True,
                "input_field_placeholder": "例如：運鏡再慢一點",
            },
        )

    def handle_script_refine_text(self, chat_id: str, instruction: str) -> None:
        """Run one LLM refinement round in the background."""
        if self.script_busy:
            self.send_safe(chat_id, "上一次修改還在進行中，請稍候。")
            return
        if not self.script_draft:
            self.send_safe(chat_id, "目前沒有草稿，請先按「✨ 生成腳本」。")
            return
        body = (instruction or "").strip()
        if not body:
            self.send_safe(chat_id, "沒有收到修改指令，已取消。")
            return

        self.awaiting_script_refine = False
        self.script_refine_instruction = body
        self.script_busy = True
        self.send_safe(chat_id, f"🤖 正在依指令修改腳本…\n「{body}」\n\n約需 20–60 秒。")
        threading.Thread(
            target=self._script_refine_worker,
            args=(chat_id,),
            name="h3-script-refiner",
            daemon=True,
        ).start()

    def _script_refine_worker(self, chat_id: str) -> None:
        """Apply one refinement, validating before it replaces the draft."""
        progress = self._script_progress(chat_id)
        draft = self.script_draft or ""
        try:
            self._ensure_llm_ready(chat_id)
            script, log = refine_h3_script(
                draft,
                self.script_refine_instruction,
                self._draft_seconds(),
                self.input_mode,
                on_progress=progress,
                lang=self._script_lang(),
                continuity=self.script_continuity,
            )
        except (BotError, OSError) as exc:
            self.script_busy = False
            bot_log(f"script refine failed: {exc}")
            self.send_safe(
                chat_id,
                f"❌ 修改失敗：{exc}\n\n草稿維持原樣，可以換個說法再試。",
            )
            return

        self.script_busy = False
        # Only a validated result replaces the draft, so a failed refinement can
        # never destroy a good script the user already had.
        self._set_script_draft(script, "指令修改")
        bot_log("script refine: " + " | ".join(log))
        self.show_script_draft(
            chat_id, script, f"指令：{self.script_refine_instruction}\n" + "\n".join(log)
        )

    def undo_script_draft(self, chat_id: str) -> None:
        """Step back to the previous draft revision."""
        if not self.script_history:
            self.send_safe(chat_id, "沒有更早的版本可以退回。")
            return
        previous = self.script_history.pop()
        self.script_draft = previous
        self.show_script_draft(
            chat_id,
            previous,
            f"已退回上一版（還有 {len(self.script_history)} 版可退）",
        )

    def accept_script_draft(self, chat_id: str, message_id: Optional[int], run_now: bool) -> None:
        """Make the draft the active prompt, optionally starting generation."""
        script = self.script_draft
        if not script:
            self.send_safe(chat_id, "草稿已過期，請重新按「✨ 生成腳本」。")
            return
        self.prompt = script
        self.continuity_source_script = script
        if self.chain_active():
            # First clip of an armed auto-chain: the adopted script is the
            # continuity source for the clips that follow.
            self.chain_current_script = script
        self.awaiting_prompt = False
        detected = self.auto_detect_prompt_duration(script, persist=True)
        adjusted_from: Optional[float] = None
        if detected is None:
            # A short (<= 15s) script is plain prose with no timeline, so its
            # duration cannot be recovered from the text. Apply the duration the
            # script was actually written for. Without this, a leftover longer
            # setting sends prose down the long-video path, where
            # build_long_video_plan() rejects it with "必須提供時間軸" - the
            # script looks fine in chat but can never be generated.
            target = min(
                float(getattr(self, "script_seconds", 0) or 0), MAX_SEGMENT_SECONDS
            )
            if target >= MIN_TOTAL_SECONDS:
                previous = float(getattr(self, "total_seconds", 0) or 0)
                try:
                    self.set_total_seconds(target)
                except (BotError, ValueError):
                    pass
                else:
                    detected = self.total_seconds
                    if abs(previous - detected) > 0.001:
                        adjusted_from = previous
        self.save_settings()
        if detected is None:
            note = "已採用腳本"
        elif adjusted_from is not None:
            note = (
                f"已採用腳本，片長由 {self.duration_label(adjusted_from)} "
                f"調整為 {self.duration_label(detected)}"
                "（短片散文腳本沒有時間軸，需用短片片長生成）"
            )
        else:
            note = f"已採用腳本，片長 {self.duration_label(detected)}"
        if not run_now:
            self.show_menu(chat_id, notice=note + "，按「🚀 生成影片」開始。")
            return

        if message_id is not None:
            self.show_menu(chat_id, message_id, note + "，開始生成。")
        else:
            self.show_menu(chat_id, note=note + "，開始生成。")
        try:
            self.start_selected_generation(chat_id, script)
        except (BotError, ValueError) as exc:
            self.send_safe(chat_id, f"生成失敗：{exc}")

    def regenerate_script(self, chat_id: str) -> None:
        if self.script_busy:
            self.send_safe(chat_id, "上一版還在生成中，請稍候。")
            return
        if not self.script_idea:
            self.request_script_idea(chat_id)
            return
        self.script_busy = True
        self.send_safe(chat_id, "🔄 正在重新生成腳本…")
        threading.Thread(
            target=self._script_worker,
            args=(chat_id,),
            name="h3-script-generator",
            daemon=True,
        ).start()

    def comfy_status_text(self) -> str:
        if comfyui_is_online():
            return f"ComfyUI 正常運行中：{COMFY_URL}"
        if _comfy_process is not None and _comfy_process.poll() is None:
            return f"ComfyUI 程序已啟動，仍在載入：PID {_comfy_process.pid}"
        return f"ComfyUI 目前未運行：{COMFY_URL}"

    def llama_status_text(self) -> str:
        """Build the local LLM status report for the Telegram panel."""
        lines = ["🧠 本地 LLM 狀態（Qwen3.8-27B）"]
        pids = _running_llama_process_ids()
        pid_text = "、".join(str(pid) for pid in sorted(pids)) if pids else "—"
        if llama_is_online():
            lines.append("狀態：運行中（API 已就緒）")
            try:
                models = json_request(f"{LLAMA_URL}/v1/models", timeout=4)
                data = models.get("data", []) if isinstance(models, dict) else []
                if data and isinstance(data[0], dict) and data[0].get("id"):
                    lines.append(f"載入模型：{data[0]['id']}")
            except (BotError, OSError):
                pass
        elif _llama_process is not None and _llama_process.poll() is None:
            lines.append("狀態：啟動中（模型載入中）")
        else:
            lines.append("狀態：未運行")
        lines.append(f"位址：{LLAMA_URL}（{LLAMA_HOST}:{LLAMA_PORT}）")
        lines.append(f"PID：{pid_text}")
        lines.append(f"模型檔：{LLAMA_MODEL.name}")
        context = _llama_preset_value("-c")
        if context:
            lines.append(f"上下文：{context} tokens")
        lines.append(_llama_gpu_line())
        lines.append(f"日誌：{LLAMA_LOG}")
        return "\n".join(lines)

    def ensure_comfyui_ready(self, job: JobState) -> None:
        # Free GPU VRAM before ComfyUI loads its model. If the local LLM is
        # still running it holds ~18GB and would push ComfyUI into OOM, so
        # close it first. This is a no-op (and stays silent) when the LLM is
        # already off; a failure here must never break the generation itself.
        if llama_is_online():
            try:
                self.send_safe(
                    job.chat_id,
                    "🧠 先關閉本地 LLM，釋放顯存避免 OOM。\n" + stop_llama_process(),
                )
            except (BotError, OSError):
                pass
        if comfyui_is_online():
            self.touch_comfy_activity()
            return
        self.send_safe(job.chat_id, start_comfyui_process(self.comfyui_vram_mode()))
        self.touch_comfy_activity()
        deadline = time.time() + 180
        while time.time() < deadline:
            if job.cancel_event.is_set():
                return
            if comfyui_is_online():
                self.touch_comfy_activity()
                self.send_safe(job.chat_id, "ComfyUI 已就緒，開始送出影片工作。")
                return
            time.sleep(3)
        raise BotError(f"ComfyUI 在 180 秒內沒有就緒，請查看日誌：{COMFYUI_LOG}")

    @staticmethod
    def image_file_id(message: dict[str, Any]) -> Optional[str]:
        photos = message.get("photo")
        if isinstance(photos, list) and photos:
            largest = photos[-1]
            if isinstance(largest, dict) and largest.get("file_id"):
                return str(largest["file_id"])
        document = message.get("document")
        if isinstance(document, dict) and document.get("file_id"):
            mime_type = str(document.get("mime_type", "")).lower()
            file_name = str(document.get("file_name", "")).lower()
            if mime_type.startswith("image/") or Path(file_name).suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".bmp",
            }:
                return str(document["file_id"])
        return None

    @staticmethod
    def video_file_id(message: dict[str, Any]) -> Optional[str]:
        video = message.get("video")
        if isinstance(video, dict) and video.get("file_id"):
            return str(video["file_id"])
        document = message.get("document")
        if isinstance(document, dict) and document.get("file_id"):
            mime_type = str(document.get("mime_type", "")).lower()
            suffix = Path(str(document.get("file_name", ""))).suffix.lower()
            if mime_type.startswith("video/") or suffix in {".mp4", ".mov", ".webm", ".mkv"}:
                return str(document["file_id"])
        return None

    @staticmethod
    def audio_file_id(message: dict[str, Any]) -> Optional[str]:
        for key in ("audio", "voice"):
            media = message.get(key)
            if isinstance(media, dict) and media.get("file_id"):
                return str(media["file_id"])
        document = message.get("document")
        if isinstance(document, dict) and document.get("file_id"):
            mime_type = str(document.get("mime_type", "")).lower()
            suffix = Path(str(document.get("file_name", ""))).suffix.lower()
            if mime_type.startswith("audio/") or suffix in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
                return str(document["file_id"])
        return None

    @staticmethod
    def prompt_file_info(
        message: dict[str, Any],
    ) -> Optional[tuple[str, str, Optional[int]]]:
        """Return file id, name and size for a supported TXT document."""
        document = message.get("document")
        if not isinstance(document, dict) or not document.get("file_id"):
            return None
        file_name = str(document.get("file_name", "prompt.txt")).strip() or "prompt.txt"
        mime_type = str(document.get("mime_type", "")).lower()
        suffix = Path(file_name).suffix.lower()
        if suffix not in PROMPT_FILE_EXTENSIONS and mime_type not in {
            "text/plain",
            "text/markdown",
        }:
            return None
        raw_size = document.get("file_size")
        try:
            file_size = int(raw_size) if raw_size is not None else None
        except (TypeError, ValueError):
            file_size = None
        return str(document["file_id"]), file_name, file_size

    def handle_image_message(self, message: dict[str, Any], chat_id: str) -> None:
        file_id = self.image_file_id(message)
        if not file_id:
            return
        try:
            remote_path = self.telegram.get_file(file_id)
            suffix = Path(remote_path).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                suffix = ".jpg"
            caption = str(message.get("caption", "")).strip()
            if self.input_mode == INPUT_MODE_FL2VA:
                REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
                # A complete pair starts a fresh FL2VA pair when the user sends
                # another image; an incomplete pair treats the next image as
                # its tail frame.
                is_first = self.image_path is None or self.last_image_path is not None
                prefix = "fl2va_first" if is_first else "fl2va_last"
                old_patterns = [f"{prefix}.*"]
                if is_first:
                    old_patterns.append("fl2va_last.*")
                for pattern in old_patterns:
                    for old_path in REFERENCE_DIR.glob(pattern):
                        try:
                            old_path.unlink()
                        except OSError:
                            pass
                target_path = REFERENCE_DIR / f"{prefix}{suffix}"
                self.telegram.download_file(remote_path, target_path)
                if is_first:
                    self.image_path = target_path
                    self.last_image_path = None
                    notice = "FL2VA 首幀已收到，請再上傳尾幀圖片。"
                else:
                    self.last_image_path = target_path
                    notice = "FL2VA 首幀和尾幀都已收到；現在輸入提示詞即可生成。"
                if caption:
                    self.prompt = caption
                    self.auto_detect_prompt_duration(caption)
                self.awaiting_prompt = False
                self.awaiting_duration = False
                self.save_settings()
                self.show_menu(chat_id, notice=notice)
                return
            if is_ref2va_like(self.input_mode):
                mode_label = (
                    "YUPI" if self.input_mode == INPUT_MODE_YUPI else "Ref2VA"
                )
                if len(self.reference_image_paths) >= MAX_REF2VA_IMAGES:
                    self.send_safe(
                        chat_id,
                        f"{mode_label} 最多支援 {MAX_REF2VA_IMAGES} 張參考圖。",
                    )
                    return
                REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
                if not self.reference_image_paths:
                    # A first uploaded image starts a new reference set, so the
                    # previous story's continuity contract no longer applies.
                    self.script_continuity = ""
                target_path = (
                    REFERENCE_DIR
                    / f"ref_image_{len(self.reference_image_paths) + 1:02d}{suffix}"
                )
                self.telegram.download_file(remote_path, target_path)
                self.reference_image_paths.append(target_path)
                if caption:
                    self.prompt = caption
                    self.auto_detect_prompt_duration(caption)
                self.awaiting_prompt = False
                self.awaiting_duration = False
                self.save_settings()
                self.show_menu(
                    chat_id,
                    notice=(
                        f"{mode_label} 已收到第 {len(self.reference_image_paths)} 張參考圖；"
                        "完成後按「✅ 完成參考素材上傳」。"
                    ),
                )
                return
            IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            for old_path in IMAGE_DIR.glob("current_input.*"):
                try:
                    old_path.unlink()
                except OSError:
                    pass
            target_path = IMAGE_DIR / f"current_input{suffix}"
            self.telegram.download_file(remote_path, target_path)
            self.image_path = target_path
            self.input_mode = "image"
            self.awaiting_prompt = False
            self.awaiting_duration = False
            if caption:
                self.prompt = caption
                self.auto_detect_prompt_duration(caption)
            self.save_settings()
            if caption:
                self.show_menu(chat_id, notice="图片和提示詞已收到")
            else:
                self.show_menu(chat_id, notice="图片已收到；现在输入提示詞")
        except BotError as exc:
            self.send_safe(chat_id, f"处理图片失败：{exc}")

    def handle_reference_video_message(
        self, message: dict[str, Any], chat_id: str
    ) -> None:
        file_id = self.video_file_id(message)
        if not file_id:
            return
        if self.input_mode != INPUT_MODE_REF2VA:
            self.send_safe(chat_id, "參考影片目前只在 Ref2VA 模式使用，請先按 Ref2VA。")
            return
        if len(self.reference_video_paths) >= MAX_REF2VA_VIDEOS:
            self.send_safe(chat_id, f"Ref2VA 最多支援 {MAX_REF2VA_VIDEOS} 段參考影片。")
            return
        try:
            remote_path = self.telegram.get_file(file_id)
            suffix = Path(remote_path).suffix.lower()
            if suffix not in {".mp4", ".mov", ".webm", ".mkv"}:
                suffix = ".mp4"
            REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
            target_path = (
                REFERENCE_DIR
                / f"ref_video_{len(self.reference_video_paths) + 1:02d}{suffix}"
            )
            data = self.telegram.download_bytes(
                remote_path, MAX_TELEGRAM_IMAGE_BYTES, "參考影片"
            )
            target_path.write_bytes(data)
            self.reference_video_paths.append(target_path)
            caption = str(message.get("caption", "")).strip()
            if caption:
                self.prompt = caption
                self.auto_detect_prompt_duration(caption)
            self.save_settings()
            self.show_menu(
                chat_id,
                notice=(
                    f"Ref2VA 已收到第 {len(self.reference_video_paths)} 段參考影片；"
                    "可繼續上傳，完成後按「完成參考素材上傳」。"
                ),
            )
        except BotError as exc:
            self.send_safe(chat_id, f"參考影片下載失敗：{exc}")

    def handle_reference_audio_message(
        self, message: dict[str, Any], chat_id: str
    ) -> None:
        file_id = self.audio_file_id(message)
        if not file_id:
            return
        if self.input_mode != INPUT_MODE_REF2VA:
            self.send_safe(chat_id, "參考音訊目前只在 Ref2VA 模式使用，請先按 Ref2VA。")
            return
        if len(self.reference_audio_paths) >= MAX_REF2VA_AUDIOS:
            self.send_safe(chat_id, f"Ref2VA 最多支援 {MAX_REF2VA_AUDIOS} 段參考音訊。")
            return
        try:
            remote_path = self.telegram.get_file(file_id)
            suffix = Path(remote_path).suffix.lower()
            if suffix not in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
                suffix = ".ogg"
            REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
            target_path = (
                REFERENCE_DIR
                / f"ref_audio_{len(self.reference_audio_paths) + 1:02d}{suffix}"
            )
            data = self.telegram.download_bytes(
                remote_path, MAX_TELEGRAM_IMAGE_BYTES, "參考音訊"
            )
            target_path.write_bytes(data)
            self.reference_audio_paths.append(target_path)
            self.save_settings()
            self.show_menu(
                chat_id,
                notice=(
                    f"Ref2VA 已收到第 {len(self.reference_audio_paths)} 段參考音訊；"
                    "可繼續上傳，完成後按「完成參考素材上傳」。"
                ),
            )
        except BotError as exc:
            self.send_safe(chat_id, f"參考音訊下載失敗：{exc}")

    def handle_prompt_file_message(self, message: dict[str, Any], chat_id: str) -> None:
        file_info = self.prompt_file_info(message)
        if file_info is None:
            return
        file_id, file_name, file_size = file_info
        if file_size is not None and file_size > MAX_TELEGRAM_PROMPT_BYTES:
            self.send_safe(
                chat_id,
                f"TXT 檔案太大，請控制在 {MAX_TELEGRAM_PROMPT_BYTES / 1024:g} KB 以內。",
            )
            return
        try:
            remote_path = self.telegram.get_file(file_id)
            data = self.telegram.download_bytes(
                remote_path,
                MAX_TELEGRAM_PROMPT_BYTES,
                "TXT 提示詞",
            )
            prompt = decode_prompt_text(data)
        except BotError as exc:
            self.send_safe(chat_id, f"讀取 TXT 提示詞失敗：{exc}")
            return

        if self.awaiting_extension_prompt:
            checkpoint_id = self.extension_checkpoint_id
            extension_seconds = self.extension_seconds
            self.awaiting_extension_prompt = False
            if checkpoint_id and extension_seconds is not None:
                self.start_extension_generation(
                    chat_id,
                    checkpoint_id,
                    extension_seconds,
                    prompt,
                )
            else:
                self.send_safe(chat_id, "延續設定已過期，請重新按 /extend。")
            return

        if self.awaiting_queue_prompt:
            self.enqueue_story_prompts(chat_id, prompt)
            return

        self.prompt = prompt
        self.awaiting_prompt = False
        self.awaiting_duration = False
        self.awaiting_extension_duration = False
        self.awaiting_extension_prompt = False
        self.awaiting_queue_prompt = False
        detected = self.auto_detect_prompt_duration(prompt)
        self.save_settings()
        duration_note = (
            f"；已自動設定片長 {self.duration_label(detected)}"
            if detected is not None
            else ""
        )
        self.show_menu(
            chat_id,
            notice=f"已讀取 {file_name}，提示詞已更新（{len(prompt)} 字）{duration_note}",
        )

    def handle_message(self, message: dict[str, Any]) -> None:
        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != self.allowed_chat_id:
            return
        if self.video_file_id(message):
            self.handle_reference_video_message(message, chat_id)
            return
        if self.audio_file_id(message):
            self.handle_reference_audio_message(message, chat_id)
            return
        if self.image_file_id(message):
            self.handle_image_message(message, chat_id)
            return
        if self.prompt_file_info(message):
            self.handle_prompt_file_message(message, chat_id)
            return
        if isinstance(message.get("document"), dict):
            self.send_safe(chat_id, "目前只支援上傳 .txt 或 .text 提示詞檔案。")
            return
        text = str(message.get("text", "")).strip()
        if not text:
            return
        if text == CONTROL_PANEL_BUTTON:
            # Editing the old inline panel does not scroll Telegram back to it.
            # Send a fresh panel at the current chat position instead.
            self.show_menu(chat_id, force_new=True, section=MENU_MAIN)
            return
        if self.awaiting_script_edit and text.lower() != "/cancel":
            if text.startswith("/"):
                self.handle_command(chat_id, text)
                return
            self.handle_script_edit_text(chat_id, text)
            return
        if self.awaiting_script_refine and text.lower() != "/cancel":
            if text.startswith("/"):
                self.handle_command(chat_id, text)
                return
            self.handle_script_refine_text(chat_id, text)
            return
        if self.awaiting_custom_prompt and text.lower() != "/cancel":
            if text.startswith("/"):
                self.handle_command(chat_id, text)
                return
            self.handle_custom_prompt_text(chat_id, text)
            return
        if self.awaiting_chain_setup and text.lower() != "/cancel":
            if text.startswith("/"):
                self.handle_command(chat_id, text)
                return
            self.handle_chain_setup(chat_id, text)
            return
        if self.awaiting_script_idea and text.lower() != "/cancel":
            if text.startswith("/"):
                self.handle_command(chat_id, text)
                return
            self.handle_script_idea(chat_id, text)
            return
        if self.awaiting_extension_duration and text.lower() != "/cancel":
            if text.startswith("/"):
                self.handle_command(chat_id, text)
                return
            try:
                self.extension_seconds = validate_total_seconds(float(text))
                self.request_extension_prompt(chat_id)
            except (BotError, ValueError) as exc:
                self.send_safe(chat_id, str(exc))
            return
        if self.awaiting_extension_prompt and text.lower() != "/cancel":
            if text.startswith("/"):
                self.handle_command(chat_id, text)
                return
            checkpoint_id = self.extension_checkpoint_id
            extension_seconds = self.extension_seconds
            self.awaiting_extension_prompt = False
            if checkpoint_id and extension_seconds is not None:
                self.start_extension_generation(
                    chat_id,
                    checkpoint_id,
                    extension_seconds,
                    text,
                )
            else:
                self.send_safe(chat_id, "延續設定已過期，請重新按 /extend。")
            return
        if self.awaiting_queue_prompt and text.lower() != "/cancel":
            if text.startswith("/"):
                self.handle_command(chat_id, text)
                return
            self.enqueue_story_prompts(chat_id, text)
            return
        if self.awaiting_duration and text.lower() != "/cancel":
            if text.startswith("/"):
                self.handle_command(chat_id, text)
                return
            try:
                self.set_total_seconds(float(text))
                self.awaiting_duration = False
                self.show_menu(chat_id, notice="自定義總片長已更新")
            except (BotError, ValueError) as exc:
                self.send_safe(chat_id, str(exc))
            return
        if self.awaiting_prompt and text.lower() != "/cancel":
            if text.startswith("/"):
                self.handle_command(chat_id, text)
                return
            self.prompt = text
            self.continuity_source_script = text
            self.awaiting_prompt = False
            detected = self.auto_detect_prompt_duration(text)
            self.save_settings()
            duration_note = (
                f"；已自動設定片長 {self.duration_label(detected)}"
                if detected is not None
                else ""
            )
            self.show_menu(chat_id, notice=f"提示詞已更新{duration_note}")
            return
        if text.startswith("/"):
            self.handle_command(chat_id, text)
        else:
            self.show_menu(chat_id)

    def handle_upscale_callback(
        self, chat_id: str, message_id: Optional[int], data: str
    ) -> None:
        parts = data.split(":", 2)
        if len(parts) != 3:
            self.send_safe(chat_id, "放大選項無效，請重新生成影片。")
            return
        choice, token = parts[1], parts[2]
        with self.lock:
            pending = self.pending_upscale
            active_job = self.job
        if pending is None or pending.token != token or pending.chat_id != chat_id:
            self.send_safe(chat_id, "這個放大選項已過期，請重新生成影片。")
            return
        if active_job is not None:
            self.send_safe(chat_id, "目前仍有任務執行中，請等它完成後再放大。")
            return
        if not pending.source_path.is_file():
            with self.lock:
                self.pending_upscale = None
            self.send_safe(chat_id, "原片已不在輸出目錄，請重新生成影片。")
            return
        if choice == "keep":
            with self.lock:
                self.pending_upscale = None
            self.send_safe(chat_id, "已保留原片，不進行放大。")
            self.finalize_upscale_choice(chat_id, pending)
            self.start_next_queued_story(chat_id)
            self.show_menu(chat_id, message_id)
            return
        if choice == "1080":
            target_long_edge = SEEDVR2_FHD_LONG_EDGE
            label = "1080p"
        elif choice == "2k":
            target_long_edge = SEEDVR2_2K_LONG_EDGE
            label = "2K"
        else:
            self.send_safe(chat_id, "未知的放大尺寸。")
            return
        target_width, target_height = upscale_dimensions(
            pending.source_width,
            pending.source_height,
            target_long_edge,
        )
        preview_seconds = min(
            MAX_SEGMENT_SECONDS,
            max(MIN_TOTAL_SECONDS, pending.duration_seconds),
        )
        config = GenerationConfig(
            pending.source_width,
            pending.source_height,
            1,
            preview_seconds,
            valid_length(preview_seconds),
        )
        job = JobState(
            chat_id=chat_id,
            config=config,
            prompt=f"SeedVR2 {label} video upscale",
            started_at=time.time(),
            output_prefix=f"MiniMaxH3/Telegram_Turbo_Upscale/{token}",
            total_seconds=pending.duration_seconds,
            task_type="seedvr2",
            upscale_source_path=pending.source_path,
            upscale_target_width=target_width,
            upscale_target_height=target_height,
        )
        job.resume_event.set()
        with self.lock:
            self.pending_upscale = None
            self.job = job
        self.send_safe(
            chat_id,
            f"已選擇 SeedVR2 {label}：目標約 {target_width}×{target_height}。\n"
            "原片會保留，放大期間可按「中止」或輸入 /cancel。",
        )
        thread = threading.Thread(
            target=self.run_upscale_job,
            args=(job, pending),
            name="seedvr2-upscale",
            daemon=True,
        )
        thread.start()
        self.show_menu(chat_id, message_id)

    def clear_uploaded_media(self) -> None:
        paths = []
        if self.image_path:
            paths.append(self.image_path)
        if self.last_image_path:
            paths.append(self.last_image_path)
        paths.extend(self.reference_image_paths)
        paths.extend(self.reference_video_paths)
        paths.extend(self.reference_audio_paths)
        for path in dict.fromkeys(paths):
            try:
                path.unlink()
            except OSError:
                pass
        self.image_path = None
        self.last_image_path = None
        self.reference_image_paths = []
        self.reference_video_paths = []
        self.reference_audio_paths = []
        self.script_continuity = ""
        self.save_settings()

    # --- tail-frame reference handoff --------------------------------------
    def offer_tail_reference(self, chat_id: str, video_path: Path) -> None:
        """Send the finished video's last two frames and offer them as new refs.

        Best-effort: this runs after a successful delivery, so every failure
        path only logs and the delivery itself is never affected.
        """
        if not TAIL_REF_OFFER_ENABLED:
            return
        try:
            token, frames = extract_tail_frames(video_path, 2)
        except Exception as exc:  # noqa: BLE001 - cosmetic feature
            bot_log(f"tail reference extraction failed: {exc}")
            return
        if not frames:
            bot_log(f"tail reference: no frames extracted from {video_path}")
            return
        try:
            keyboard = {
                "inline_keyboard": [
                    [
                        {
                            "text": "✅ 清空參考圖，用這兩帧做新參考圖",
                            "callback_data": f"tailref:yes:{token}",
                        }
                    ],
                    [
                        {
                            "text": "❌ 保留原本參考圖",
                            "callback_data": f"tailref:no:{token}",
                        }
                    ],
                ]
            }
            if len(frames) > 1:
                self.telegram.send_photo(
                    chat_id,
                    frames[0],
                    "📸 片段最後兩帧（可以直接變成下一段嘅參考圖）",
                )
                self.telegram.send_photo(
                    chat_id,
                    frames[-1],
                    "最後一帧。",
                    reply_markup=keyboard,
                )
            else:
                self.telegram.send_photo(
                    chat_id,
                    frames[0],
                    "📸 片段最後一帧（可以直接變成下一段嘅參考圖）",
                    reply_markup=keyboard,
                )
            bot_log(f"tail reference offered: token={token} frames={len(frames)}")
        except BotError as exc:
            bot_log(f"tail reference offer failed: {exc}")

    def handle_tailref_callback(
        self, chat_id: str, message_id: Optional[int], data: str
    ) -> None:
        """Handle the ✅/❌ buttons that follow a tail-frame offer."""
        parts = data.split(":")
        action = parts[1] if len(parts) > 1 else ""
        token = parts[2] if len(parts) > 2 else ""

        def drop_keyboard() -> None:
            if message_id is None:
                return
            try:
                self.telegram.clear_inline_keyboard(chat_id, message_id)
            except BotError:
                pass

        if action != "yes":
            drop_keyboard()
            self.send_safe(chat_id, "❌ 保留原本參考圖，冇改動。")
            return

        frames = tail_reference_frames(token)
        if not frames:
            drop_keyboard()
            self.send_safe(chat_id, "⚠️ 搵唔到嗰兩帧（可能已被清理），今次冇改動。")
            return

        try:
            new_paths = promote_reference_frames(frames)
        except OSError as exc:
            drop_keyboard()
            self.send_safe(chat_id, f"⚠️ 更新參考圖失敗：{exc}")
            return

        self.reference_image_paths = new_paths
        source = (
            (getattr(self, "continuity_source_script", "") or "").strip()
            or (getattr(self, "prompt", "") or "").strip()
        )
        if len(source) > 4000:
            source = source[:4000].rstrip() + "\n……（劇本太長，已截斷）"
        self.script_continuity = TAIL_CONTINUITY_NOTE
        if source:
            self.script_continuity += (
                "\n\n【上一段劇本（最新一條片用嘅；只寫之後發生嘅事）】\n"
                + source
                + "\n【上一段劇本完】"
            )
        self.save_settings()
        bot_log(f"tail reference promoted: {[str(path) for path in new_paths]}")

        drop_keyboard()
        note = (
            f"✅ 已清空舊參考圖，改用呢 {len(new_paths)} 帧（片段尾帧）做新參考圖。\n"
            + (
                "🔗 已記住上一段劇本：跟住發一句「想法」，本地 LLM 會讀住上一段幫你寫延續嘅"
                "新腳本（同一人物、同一場景、由尾帧動作接落去，唔會重複演過嘅嘢）。\n"
                if source
                else "🔗 新一段會當作「上一幕嘅延續」生成（同一人物、同一場景、由尾帧動作接落去）。\n"
            )
            + "（想自己貼完整提示詞：面板「✍️ 輸入／更換提示詞」。）"
        )
        if (
            normalize_input_mode(getattr(self, "input_mode", INPUT_MODE_TEXT))
            != INPUT_MODE_REF2VA
        ):
            note += "\n（提醒：目前唔係 Ref2VA 模式，參考圖要切去 Ref2VA 先會用到。）"
        self.request_script_idea(chat_id, note=note)

    # --- 🔗 auto-chain (short clips chained by tail-frame handoff) ----------
    def show_gpu_status(self, chat_id: str) -> None:
        """Report per-GPU memory/display state and the card ComfyUI will use."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,display_attached",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=8,
            )
            lines = ["🖥 GPU 狀態"]
            rows: list[tuple[int, int, bool]] = []
            for line in result.stdout.strip().split("\n"):
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 5:
                    continue
                try:
                    index, used, total = int(parts[0]), int(parts[2]), int(parts[3])
                except ValueError:
                    continue
                attached = parts[4].lower() in {"yes", "enabled"}
                free = total - used
                rows.append((index, free, attached))
                tag = "（屏幕卡）" if attached else ""
                lines.append(
                    f"GPU {index} {parts[1]}{tag}：{used} / {total} MiB（空閒 {free}）"
                )
            if rows:
                min_free = int(
                    os.environ.get("MINIMAX_COMFY_GPU_MIN_FREE_MB", "6000")
                )
                preferred = [r for r in rows if not r[2] and r[1] >= min_free]
                pick = max(preferred or rows, key=lambda r: r[1])
                why = "非屏幕卡" if not pick[2] else "冇可用非屏幕卡，揀最空閒嗰張"
                lines.append(f"ComfyUI 生成會用：GPU {pick[0]}（{why}）")
            lines.append(
                "（想強制指定：設 MINIMAX_COMFY_CUDA_VISIBLE_DEVICES=1 並重啟 ComfyUI）"
            )
            self.send_safe(chat_id, "\n".join(lines))
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            self.send_safe(chat_id, f"讀取 GPU 狀態失敗：{exc}")

    def chain_active(self) -> bool:
        """True while an auto-chain run still has clips left to generate."""
        return bool(getattr(self, "chain_remaining", 0))

    def request_chain_setup(self, chat_id: str) -> None:
        """Ask for the one-line chain setup: 段數 每段秒數 [想法]."""
        self.awaiting_chain_setup = True
        self.awaiting_prompt = False
        self.awaiting_script_idea = False
        self.awaiting_duration = False
        self.awaiting_queue_prompt = False
        self.telegram.send_message(
            chat_id,
            "🔗 自動接力（短段生成 → 尾帧接續）\n\n"
            "每段長度可以自己排，或者畀個總長由 bot 自動用「長短交替」節奏分段。\n"
            "之後 bot 會：生成第 1 段 → 自動抽尾帧做新參考圖 → LLM 讀住上一段寫下一段 → 再生成…\n"
            "（每段都會即刻傳返 Telegram；最後自動合併成一段）\n\n"
            "格式（三選一）：\n"
            "  • 時長清單：5 8 5 10 → 4 段：5/8/5/10 秒\n"
            "  • 總秒數：48 → 自動分段（例如 4、9、6、12、7、5、10、7 秒）\n"
            "  • 總秒數＋每段上限：48 6 → 總 48 秒、每段唔超過 6 秒\n"
            "（後面直接寫想法）\n\n"
            "例：5 8 5 10 女生在浴室沖涼，之後慢慢走出客廳",
            reply_markup={
                "force_reply": True,
                "input_field_placeholder": "例：5 8 5 10 女生在浴室沖涼…",
            },
        )

    def handle_chain_setup(self, chat_id: str, text: str) -> None:
        """Parse "<durations|total> [cap] [idea]" and start (or ask for the idea).

        Accepted duration forms:
          5 8 5 10      - explicit per-clip seconds (2-15 each)
          48            - total seconds, split with the varied rhythm
          48 6          - total seconds with a 6s per-clip cap
        """
        self.awaiting_chain_setup = False
        numbers: list[float] = []
        words: list[str] = []
        for token in text.strip().replace(",", " ").replace("，", " ").split():
            if not words and re.fullmatch(r"\d+(?:\.\d+)?", token):
                numbers.append(float(token))
            else:
                words.append(token)
        idea = " ".join(words).strip()
        max_clip = float(MAX_SEGMENT_SECONDS)
        durations: list[float] = []
        if not numbers:
            self.send_safe(
                chat_id,
                "格式：時長清單（5 8 5 10）或總秒數（48）＋想法，"
                "例如：5 8 5 10 女生在浴室沖涼",
            )
            return
        error = ""
        if len(numbers) == 1:
            # A single number is the TOTAL length: split with the varied rhythm.
            total = numbers[0]
            if not 2 <= total <= 1800:
                error = "總秒數請在 2–1800 之間。"
            else:
                durations = chain_plan_durations(total)
        elif all(n <= max_clip for n in numbers):
            # Every number is a per-clip length: 5 8 5 10 -> four clips.
            bad = [n for n in numbers if not 2 <= n <= max_clip]
            if bad:
                error = f"每段秒數請在 2–{max_clip:g} 秒之間（收到 {bad}）。"
            else:
                durations = list(numbers)
        else:
            # total [cap]: 48 6 -> 48 seconds total, no clip over 6s.
            total = numbers[0]
            cap = numbers[1] if len(numbers) > 1 else 0.0
            if total <= max_clip:
                error = f"每段秒數請在 2–{max_clip:g} 秒之間（收到 {numbers}）。"
            elif not 2 <= total <= 1800:
                error = "總秒數請在 2–1800 之間。"
            elif cap and not 2 <= cap <= max_clip:
                error = f"每段上限請在 2–{max_clip:g} 秒之間。"
            else:
                durations = chain_plan_durations(total, cap)
        if error:
            self.send_safe(chat_id, error)
            return
        if not durations:
            self.send_safe(
                chat_id,
                "睇唔明時長：可以寫清單（5 8 5 10）、總秒數（48）或總秒數＋上限（48 6）。",
            )
            return
        if len(durations) > 50:
            self.send_safe(chat_id, "段數太多（>50），請加大每段秒數。")
            return
        plan_text = "、".join(f"{d:g}" for d in durations)
        self.chain_pending_durations = durations
        if idea:
            self.start_chain(chat_id, durations, idea)
            return
        self.chain_awaiting_idea = True
        self.request_script_idea(
            chat_id,
            note=(
                f"🔗 自動接力已設定：{len(durations)} 段（{plan_text} 秒，總 "
                f"{sum(durations):g} 秒）\n"
                "請發一句「第 1 段」嘅想法；之後每段會自動接住上一段。"
            ),
        )

    def start_chain(self, chat_id: str, durations: list[float], idea: str) -> None:
        """Kick off the auto-chain: one clip per duration, each continued from the last."""
        if self.chain_active():
            self.send_safe(chat_id, "已有一個接力進行中（/chain off 可以停）。")
            return
        refs = [
            path
            for path in getattr(self, "reference_image_paths", [])
            if path is not None and Path(path).is_file()
        ]
        if not refs:
            self.send_safe(
                chat_id,
                "🔗 自動接力需要參考圖（每段尾帧會做下一段嘅新參考圖）："
                "請先上傳一張參考圖再試。",
            )
            return
        if not SCRIPT_GEN_ENABLED:
            self.send_safe(
                chat_id, "腳本生成器已停用（MINIMAX_SCRIPT_GEN=0），無法接力。"
            )
            return
        if self.script_busy:
            self.send_safe(chat_id, "上一個腳本還在生成中，請稍候再開始接力。")
            return
        with self.lock:
            if self.job is not None:
                self.send_safe(
                    chat_id, "目前有生成工作，請等它完成（或 /cancel）再開始接力。"
                )
                return
        clean = [float(d) for d in durations if float(d) > 0]
        if not clean:
            self.send_safe(chat_id, "接力時長清單是空的。")
            return
        self.chain_durations = clean
        self.chain_total = len(clean)
        self.chain_remaining = len(clean)
        self.chain_idea = idea.strip()
        self.chain_done_paths = []
        self.chain_current_script = ""
        self.input_mode = INPUT_MODE_REF2VA
        self.save_settings()
        plan_text = "、".join(f"{d:g}" for d in clean)
        self.send_safe(
            chat_id,
            "🔗 自動接力開始\n"
            f"段數：{self.chain_total} 段（{plan_text} 秒；總約 {sum(clean):g} 秒）\n"
            f"腳本引擎：{script_llm_display_name()}｜模板：{script_template_display_name()}\n"
            f"想法：{self.chain_idea}\n\n"
            "第 1 段：LLM 生成中——完成後請審閱，按「✅ 採用並生成」就開始接力，"
            "之後每段自動接落去。\n（要停：/chain off）",
        )
        threading.Thread(
            target=self._chain_worker,
            args=(chat_id,),
            name="h3-chain",
            daemon=True,
        ).start()
        # The first clip goes through the normal draft review instead of being
        # auto-submitted: the operator can adopt, edit or regenerate it, and the
        # worker above simply waits for the clip that gets submitted.
        self.handle_script_idea(
            chat_id, self.chain_idea, seconds_override=float(clean[0])
        )

    def chain_submit(self, chat_id: str, script: str, seconds: float) -> bool:
        """Submit one chain clip as a Ref2VA single-shot job."""
        config = parse_config(
            [
                str(self.settings.width),
                str(self.settings.height),
                str(self.settings.steps),
                str(seconds),
            ]
        )
        self.last_job_video_path = None
        return self.start_generation(
            chat_id,
            config,
            script,
            reference_image_paths=[
                path for path in self.reference_image_paths if path.is_file()
            ],
            reference_video_paths=[
                path for path in self.reference_video_paths if path.is_file()
            ],
            reference_audio_paths=[
                path for path in self.reference_audio_paths if path.is_file()
            ],
            generation_mode=INPUT_MODE_REF2VA,
        )

    def chain_wait_for_job(self, timeout: float = 5400.0) -> bool:
        """Block until the submitted clip finishes. False on stop/timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.chain_active():
                return False
            with self.lock:
                if self.job is not None:
                    break
            time.sleep(1.0)
        else:
            return False
        while time.time() < deadline:
            with self.lock:
                if self.job is None:
                    return True
            time.sleep(2.0)
        return False

    def _chain_worker(self, chat_id: str) -> None:
        """One clip at a time; each clip's tail frames become the next refs."""
        try:
            try:
                self._ensure_llm_ready(chat_id)
            except (BotError, OSError) as exc:
                self.send_safe(chat_id, f"🔗 接力中止：{exc}")
                return
            total = len(self.chain_durations)
            self.chain_total = total
            for index, clip_seconds in enumerate(list(self.chain_durations), start=1):
                if not self.chain_active():
                    self.send_safe(
                        chat_id,
                        f"🔗 接力已停止（完成 {len(self.chain_done_paths)}/{total} 段）。",
                    )
                    return
                if index > 1:
                    idea = (
                        f"{self.chain_idea}\n（這是第 {index}/{total} 段、片長 {clip_seconds:g} 秒："
                        "直接延續上一段的結尾動作，不要重複已經演過的內容，也不要重新介紹角色）"
                    )
                    self.send_safe(
                        chat_id,
                        f"🔗 第 {index}/{total} 段（{clip_seconds:g} 秒）：LLM 正在續寫腳本…",
                    )
                    try:
                        script, log = generate_h3_script(
                            idea,
                            float(clip_seconds),
                            INPUT_MODE_REF2VA,
                            on_progress=None,
                            lang=self._script_lang(),
                            continuity=self.script_continuity,
                        )
                    except (BotError, OSError) as exc:
                        self.send_safe(
                            chat_id,
                            f"🔗 接力中止：第 {index}/{total} 段腳本生成失敗：{exc}",
                        )
                        return
                    self.chain_current_script = script
                    self.send_safe(
                        chat_id,
                        f"🔗 第 {index}/{total} 段腳本完成（{'；'.join(log) or '通過驗證'}），"
                        "生成中…",
                    )
                    if not self.chain_submit(chat_id, script, float(clip_seconds)):
                        self.send_safe(chat_id, "🔗 接力中止：無法送出生成工作。")
                        return
                # Clip 1 arrives here already submitted by the operator's own
                # adoption of the LLM draft; every clip then waits the same way.
                if not self.chain_wait_for_job():
                    self.send_safe(chat_id, "🔗 接力中止：工作被取消或超時。")
                    return
                video = getattr(self, "last_job_video_path", None)
                if video is None or not Path(video).is_file():
                    self.send_safe(
                        chat_id, f"🔗 接力中止：第 {index}/{total} 段沒有產出影片。"
                    )
                    return
                self.chain_done_paths.append(Path(video))
                self.chain_remaining = total - index
                _token, frames = extract_tail_frames(Path(video), 2)
                if not frames:
                    self.send_safe(chat_id, "🔗 接力中止：抽不到尾帧（無法接續）。")
                    return
                try:
                    self.reference_image_paths = promote_reference_frames(frames)
                except OSError as exc:
                    self.send_safe(chat_id, f"🔗 接力中止：更新參考圖失敗：{exc}")
                    return
                clip_script = self.chain_current_script or ""
                source = clip_script[:4000]
                if len(clip_script) > 4000:
                    source += "\n……（劇本太長，已截斷）"
                self.continuity_source_script = clip_script
                self.script_continuity = (
                    TAIL_CONTINUITY_NOTE
                    + "\n\n【上一段劇本（最新一條片用嘅；只寫之後發生嘅事）】\n"
                    + source
                    + "\n【上一段劇本完】"
                )
                self.save_settings()
                if index < total:
                    self.send_safe(
                        chat_id,
                        f"🔗 第 {index}/{total} 段完成 → 尾帧已成為新參考圖，"
                        f"準備第 {index + 1} 段…",
                    )
                else:
                    self.send_safe(chat_id, f"🔗 全部 {total} 段完成，正在合併…")
            self.chain_finish(chat_id)
        except Exception as exc:  # noqa: BLE001 - the chain must never kill the bot
            bot_log(f"chain worker failed: {type(exc).__name__}: {exc}")
            self.send_safe(chat_id, f"🔗 接力發生錯誤：{exc}")
        finally:
            self.chain_remaining = 0
            self.chain_total = 0
            try:
                self.save_settings()
            except (BotError, OSError, ValueError):
                pass

    def chain_finish(self, chat_id: str) -> None:
        """Merge the finished clips and send the combined video."""
        paths = [path for path in self.chain_done_paths if path.is_file()]
        if len(paths) < 2:
            return
        try:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            merged = chain_merge_clips(paths, OUTPUT_DIR / f"chain_{stamp}.mp4")
        except (BotError, OSError) as exc:
            self.send_safe(chat_id, f"（合併成品失敗；上面各段仍可正常使用：{exc}）")
            return
        try:
            self.telegram.send_video(
                chat_id, merged, f"🔗 接力完成：{len(paths)} 段合併成一段"
            )
        except (BotError, OSError) as exc:
            self.send_safe(chat_id, f"（合併成品傳送失敗：{exc}）")

    def handle_callback(self, callback: dict[str, Any]) -> None:
        query_id = str(callback.get("id", ""))
        message = callback.get("message") or {}
        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != self.allowed_chat_id:
            return
        try:
            self.telegram.answer_callback_query(query_id)
        except BotError:
            pass

        data = str(callback.get("data", ""))
        message_id = message.get("message_id")
        try:
            if data.startswith("tailref:"):
                self.handle_tailref_callback(chat_id, message_id, data)
                return
            if data.startswith("upscale:"):
                self.handle_upscale_callback(chat_id, message_id, data)
                return
            if data == "noop":
                return
            if data == "script_file":
                self.show_script_prompt_file(chat_id, message_id)
                return
            if data.startswith("script_file:"):
                action = data.removeprefix("script_file:")
                if action == "edit":
                    self.request_custom_prompt(chat_id, "replace")
                elif action == "append":
                    self.request_custom_prompt(chat_id, "append")
                elif action == "clear":
                    ok, note = save_custom_script_instructions("", "由 Telegram 清空")
                    self.send_safe(
                        chat_id,
                        f"🗑 {note}" if ok else f"⚠️ {note}",
                    )
                    self.show_script_prompt_file(chat_id, message_id)
                elif action == "undo":
                    ok, note = restore_custom_script_instructions()
                    self.send_safe(
                        chat_id,
                        f"↩️ {note}" if ok else f"⚠️ {note}",
                    )
                    self.show_script_prompt_file(chat_id, message_id)
                return
            if data.startswith("script_lang:"):
                action = data.removeprefix("script_lang:")
                if action == "toggle":
                    current = normalize_script_lang(
                        getattr(self, "script_lang", SCRIPT_LANG_DEFAULT)
                    )
                    target = (
                        SCRIPT_LANG_EN
                        if current == SCRIPT_LANG_ZH
                        else SCRIPT_LANG_ZH
                    )
                else:
                    target = normalize_script_lang(action)
                self.set_script_lang(chat_id, target, message_id)
                return
            if data.startswith("script_template:"):
                action = data.removeprefix("script_template:")
                current = normalize_script_template(
                    getattr(self, "script_template", SCRIPT_TEMPLATE_DEFAULT)
                )
                if action in {"", "toggle"}:
                    target = (
                        SCRIPT_TEMPLATE_GENERAL
                        if current == SCRIPT_TEMPLATE_ADULT
                        else SCRIPT_TEMPLATE_ADULT
                    )
                else:
                    target = normalize_script_template(action)
                self.switch_script_template(chat_id, target, message_id)
                return
            if data.startswith("script_llm:"):
                action = data.removeprefix("script_llm:")
                current = normalize_script_llm(
                    getattr(self, "script_llm", SCRIPT_LLM_DEFAULT)
                )
                if action in {"", "toggle"}:
                    target = (
                        SCRIPT_LLM_COMMANDCODE
                        if current == SCRIPT_LLM_LOCAL
                        else SCRIPT_LLM_LOCAL
                    )
                else:
                    target = normalize_script_llm(action)
                self.switch_script_llm(chat_id, target, message_id)
                return
            if data.startswith("script:"):
                action = data.removeprefix("script:")
                if action == "new":
                    self.request_script_idea(chat_id)
                elif action == "accept":
                    self.accept_script_draft(chat_id, message_id, run_now=False)
                elif action == "accept_run":
                    self.accept_script_draft(chat_id, message_id, run_now=True)
                elif action == "regen":
                    self.regenerate_script(chat_id)
                elif action == "cancel":
                    self.script_draft = None
                    self.script_history = []
                    self.script_busy = False
                    self.awaiting_script_edit = ""
                    self.awaiting_script_refine = False
                    if self.chain_active() and self.job is None:
                        # An armed chain is waiting for this draft: dropping it
                        # stops the run before the first clip was ever submitted.
                        self.chain_remaining = 0
                        self.send_safe(chat_id, "🔗 接力已取消（草稿已放棄）。")
                    self.show_menu(chat_id, message_id, "已放棄腳本草稿。")
                elif action == "edit":
                    self.request_script_edit(chat_id, "replace")
                elif action == "append":
                    self.request_script_edit(chat_id, "append")
                elif action == "refine":
                    self.request_script_refine(chat_id)
                elif action == "undo":
                    self.undo_script_draft(chat_id)
                return
            if data.startswith("menu:"):
                self.menu_section = normalize_menu_section(data.removeprefix("menu:"))
                self.show_menu(chat_id, message_id, section=self.menu_section)
                return
            if data == "progress":
                self.show_progress(chat_id, message_id)
                return
            if data.startswith("long_resume:"):
                self.resume_long_checkpoint(
                    chat_id,
                    data.removeprefix("long_resume:"),
                    message_id,
                )
                return
            if data.startswith("long_extend:"):
                self.request_extension_duration(
                    chat_id,
                    data.removeprefix("long_extend:"),
                    message_id,
                )
                return
            if data == "history":
                self.show_history(chat_id, message_id)
                return
            if data == "history_back":
                self.show_menu(chat_id, message_id, section=MENU_MAIN)
                return
            if data.startswith("history_select:"):
                self.show_checkpoint_detail(
                    chat_id,
                    data.removeprefix("history_select:"),
                    message_id,
                )
                return
            if data == "queue_view":
                self.show_queue(chat_id, message_id)
                return
            if data == "queue_add":
                self.request_queue_prompt(chat_id)
                return
            if data == "queue_start":
                started = self.start_next_queued_story(chat_id)
                if not started:
                    self.send_safe(chat_id, "目前有任務、放大選擇，或排隊是空的；請稍後再試。")
                self.show_queue(chat_id, message_id)
                return
            if data == "queue_clear":
                self.clear_story_queue(chat_id, message_id)
                return
            if data.startswith("queue_remove:"):
                self.remove_queued_story(
                    chat_id,
                    data.removeprefix("queue_remove:"),
                    message_id,
                )
                return
            if data == "job_abort":
                self.abort_current_job(chat_id, message_id)
                return
            if data == "job_pause":
                self.pause_current_job(chat_id, message_id)
                return
            if data == "job_resume":
                self.resume_current_job(chat_id, message_id)
                return
            if data == "job_preview":
                self.request_video_preview(chat_id, message_id)
                return
            if data.startswith("model:"):
                selected_model = normalize_model_mode(data.removeprefix("model:"))
                self.model_mode = selected_model
                self.save_settings()
                self.show_menu(chat_id, message_id, "目前使用：MiniMax H3 Turbo")
                return
            if data in {"mode:text", "mode:image"}:
                self.input_mode = (
                    INPUT_MODE_TEXT if data == "mode:text" else INPUT_MODE_IMAGE
                )
                self.save_settings()
                self.show_menu(chat_id, message_id, "生成模式已切換。")
                return
            if data == "mode:fl2va":
                self.input_mode = INPUT_MODE_FL2VA
                self.save_settings()
                self.show_menu(
                    chat_id,
                    message_id,
                    "已選 FL2VA：請依次上傳首幀圖片、尾幀圖片，再輸入提示詞。",
                )
                return
            if data == "mode:ref2va":
                self.input_mode = INPUT_MODE_REF2VA
                self.save_settings()
                self.show_menu(
                    chat_id,
                    message_id,
                    "已選 Ref2VA：可連續上傳參考圖片／影片／音訊，完成後按按鈕。",
                )
                return
            if data == "mode:yupi":
                # 🌙 YUPI工作流：a real mode now. Selecting it only stages the
                # mode; the user uploads a reference image, types a prompt, then
                # presses 🚀 生成影片 — same flow as every other mode.
                self.input_mode = INPUT_MODE_YUPI
                self.save_settings()
                self.show_menu(
                    chat_id,
                    message_id,
                    "已選 YUPI工作流（Ref2VA + AfterMidnight + FastH3 6步）："
                    "上傳參考圖、輸入提示詞，再按「🚀 生成影片」。",
                )
                return
            if data == "media_done":
                if self.input_mode == INPUT_MODE_YUPI:
                    if not any(
                        path.is_file() for path in self.reference_image_paths
                    ) and not (
                        self.image_path is not None and self.image_path.is_file()
                    ):
                        self.send_safe(chat_id, "YUPI 尚未收到參考圖片。")
                        return
                elif self.input_mode == INPUT_MODE_REF2VA and not (
                    self.reference_image_paths
                    or self.reference_video_paths
                    or self.reference_audio_paths
                ):
                    self.send_safe(chat_id, "Ref2VA 尚未收到任何參考素材。")
                    return
                self.show_menu(chat_id, message_id, "參考素材已完成；現在輸入提示詞或按生成。")
                return
            if data.startswith("res:"):
                width, height = data.removeprefix("res:").split("x", 1)
                self.update_settings(width=int(width), height=int(height))
                self.show_menu(chat_id, message_id, "解析度已更新")
                return
            if data == "sec_custom":
                self.request_duration(chat_id)
                return
            if data.startswith("sec:"):
                self.set_total_seconds(float(data.removeprefix("sec:")))
                self.show_menu(chat_id, message_id, "總片長已更新；超過 15 秒會自動分段合併")
                return
            if data.startswith("steps:"):
                self.update_settings(steps=int(data.removeprefix("steps:")))
                self.show_menu(chat_id, message_id, "steps 已更新")
                return
            if data == "last":
                self.settings = self.load_settings()
                self.total_seconds = self.load_saved_total_seconds()
                self.prompt = self.load_saved_prompt()
                self.input_mode = self.load_saved_mode()
                self.model_mode = self.load_saved_model_mode()
                self.image_path = self.load_saved_image_path()
                saved_media = self._load_saved_media_paths
                saved_last = saved_media("last_image_paths")
                self.last_image_path = saved_last[0] if saved_last else None
                self.reference_image_paths = saved_media("reference_image_paths")
                self.reference_video_paths = saved_media("reference_video_paths")
                self.reference_audio_paths = saved_media("reference_audio_paths")
                self.shutdown_after_generation = self.load_saved_shutdown_after_generation()
                self.show_menu(chat_id, message_id, "已讀取上次設定")
                return
            if data == "prompt":
                self.request_prompt(chat_id)
                return
            if data == "temperature":
                self.send_safe(chat_id, temperature_report())
                return
            if data == "shutdown_toggle":
                if self.total_seconds <= MAX_SEGMENT_SECONDS:
                    self.send_safe(
                        chat_id,
                        "自動關機只對超過 15 秒的長片生效，請先選擇 30 秒或更長片長。",
                    )
                    return
                self.shutdown_after_generation = not self.shutdown_after_generation
                self.save_settings()
                notice = (
                    "已開啟：長片完成並傳送後會在 60 秒後關機。"
                    if self.shutdown_after_generation
                    else "已關閉：長片完成後不會自動關機。"
                )
                self.show_menu(chat_id, message_id, notice)
                return
            if data == "shutdown_cancel":
                self.cancel_scheduled_shutdown(chat_id, message_id)
                return
            if data == "clear":
                self.prompt = ""
                self.awaiting_prompt = False
                self.save_settings()
                self.show_menu(chat_id, message_id, "提示詞已清除")
                return
            if data == "clear_image":
                self.clear_uploaded_media()
                self.show_menu(chat_id, message_id, "已清除目前模式的全部上傳素材。")
                return
            if data == "twopass_toggle":
                self.latent_upscale = not bool(
                    getattr(self, "latent_upscale", LATENT_UPSCALE_ENABLED)
                )
                self.save_settings()
                if self.latent_upscale:
                    note = (
                        "兩段式 latent 上採樣：已開啟。\n"
                        "先生成半解析度底片 → latent 空間放大 → 全解析度重採樣精修。\n"
                        "畫質更清晰、細節更多，耗時略增；長片續段不受影響。"
                    )
                else:
                    note = (
                        "兩段式 latent 上採樣：已關閉。\n"
                        "現在依所選解析度單段直出，速度較快。"
                    )
                self.show_menu(chat_id, message_id, note)
                return
            if data == "continuity_toggle":
                current = getattr(self, "long_continuity", "motion_context")
                self.long_continuity = (
                    "tail_frame" if current == "motion_context" else "motion_context"
                )
                self.save_settings()
                if self.long_continuity == "motion_context":
                    note = (
                        "長片接續：已切換為 Motion Context。\n"
                        "第 2 鏡起接續上一鏡的 AV latent 與尾幀，動作與音訊更連貫。\n"
                        "需要 4 個 Motion Context 節點；若缺少會自動退回尾幀接續。"
                    )
                else:
                    note = (
                        "長片接續：已切換為尾幀接續。\n"
                        "第 2 鏡起只接上一鏡尾幀，每鏡重新生成原生音訊（較穩定）。"
                    )
                self.show_menu(chat_id, message_id, note)
                return
            if data == "h3_profile_toggle":
                current = getattr(self, "h3_profile", H3_PROFILE_DEFAULT)
                self.h3_profile = (
                    H3_PROFILE_FUSED
                    if current != H3_PROFILE_FUSED
                    else H3_PROFILE_CLASSIC
                )
                self.save_settings()
                if self.h3_profile == H3_PROFILE_FUSED:
                    sla_note = (
                        "SLA 稀疏注意力：已偵測到，會自動插入（約快 40%、音訊高頻更好）。"
                        if h3_sla_available()
                        else "⚠️ 未偵測到 H3 SLA 節點，這次會用稠密注意力（較慢、音訊較悶）。"
                    )
                    note = (
                        "模型：已切換為融合加速。\n"
                        "單一 21GB 檔同時支援 T2VA／I2VA／FL2VA／Ref2VA，"
                        "turbo 與 Mystic 已烤進權重（不掛 LoRA），固定 6 步。\n"
                        f"{sla_note}\n"
                        "YUPI 工作流不受影響。"
                    )
                else:
                    note = (
                        "模型：已切換為經典。\n"
                        "FL2VA／Ref2VA 分開載入 + turbo LoRA，步數吃面板設定。"
                    )
                self.show_menu(chat_id, message_id, note)
                return
            if data.startswith("vram:"):
                mode = normalize_comfyui_vram_mode(data.removeprefix("vram:"))
                self.vram_mode = mode
                self.save_settings()
                cancelled = self.cancel_job_for_comfy_control()
                result = restart_comfyui_process(mode)
                self.touch_comfy_activity()
                prefix = "已取消目前生成工作。\n" if cancelled else ""
                self.send_safe(
                    chat_id,
                    prefix
                    + f"已切換到{comfyui_vram_mode_label(mode)}，正在重啟 ComfyUI。\n"
                    + result,
                )
                self.show_menu(chat_id, message_id)
                return
            if data == "comfy_start":
                result = start_comfyui_process(self.comfyui_vram_mode())
                self.touch_comfy_activity()
                self.send_safe(chat_id, result)
                return
            if data == "comfy_status":
                self.send_safe(chat_id, self.comfy_status_text())
                return
            if data == "comfy_restart":
                cancelled = self.cancel_job_for_comfy_control()
                result = restart_comfyui_process(self.comfyui_vram_mode())
                self.touch_comfy_activity()
                prefix = "目前生成已取消。\n" if cancelled else ""
                self.send_safe(chat_id, prefix + result)
                return
            if data == "comfy_stop":
                cancelled = self.cancel_job_for_comfy_control()
                result = stop_comfyui_process()
                prefix = "目前生成已取消。\n" if cancelled else ""
                self.send_safe(chat_id, prefix + result)
                return
            if data == "llama_start":
                try:
                    self.send_safe(chat_id, start_llama_process())
                except BotError as exc:
                    self.send_safe(chat_id, str(exc))
                return
            if data == "llama_status":
                self.send_safe(chat_id, self.llama_status_text())
                return
            if data == "llama_restart":
                try:
                    self.send_safe(chat_id, restart_llama_process())
                except BotError as exc:
                    self.send_safe(chat_id, str(exc))
                return
            if data == "llama_stop":
                try:
                    self.send_safe(chat_id, stop_llama_process())
                except BotError as exc:
                    self.send_safe(chat_id, str(exc))
                return
            if data == "llm_after_job_toggle":
                self.restart_llm_after_generation = not bool(
                    getattr(
                        self,
                        "restart_llm_after_generation",
                        RESTART_LLM_AFTER_GENERATION,
                    )
                )
                self.save_settings()
                if self.restart_llm_after_generation:
                    note = (
                        "生成後自動啟動 LLM：已開啟。\n"
                        "每個生成任務完成後，Bot 會先關閉 ComfyUI 釋放顯存，"
                        "再自動啟動本地 LLM（Hermes／DSH 的本機模型就能繼續用）。"
                    )
                else:
                    note = (
                        "生成後自動啟動 LLM：已關閉。\n"
                        "生成完成後 LLM 保持關閉、ComfyUI 也維持原狀，"
                        "需要時再按「▶️ 啟動 LLM」。"
                    )
                self.show_menu(chat_id, message_id, note)
                return
            if data == "bot_restart":
                self.restart_bot(chat_id)
                return
            if data == "chain:start":
                self.request_chain_setup(chat_id)
                return
            if data == "generate":
                if self.awaiting_duration:
                    self.send_safe(chat_id, "請先輸入自定義總片長秒數，或使用 /cancel 取消。")
                elif self.awaiting_prompt:
                    self.send_safe(chat_id, "請先貼上提示詞，或使用 /cancel。")
                elif self.input_mode == INPUT_MODE_IMAGE and not self.image_path:
                    self.send_safe(chat_id, "請先上傳圖片。")
                elif self.input_mode == INPUT_MODE_FL2VA and (
                    not self.image_path or not self.last_image_path
                ):
                    self.send_safe(
                        chat_id,
                        "FL2VA 需要兩張圖片：請先上傳首幀，再上傳尾幀。",
                    )
                elif self.input_mode == INPUT_MODE_YUPI and not (
                    any(path.is_file() for path in self.reference_image_paths)
                    or (self.image_path is not None and self.image_path.is_file())
                ):
                    self.send_safe(
                        chat_id,
                        "YUPI 需要一張參考圖：請先上傳圖片。",
                    )
                elif self.input_mode == INPUT_MODE_REF2VA and not (
                    self.reference_image_paths
                    or self.reference_video_paths
                    or self.reference_audio_paths
                ):
                    self.send_safe(
                        chat_id,
                        "Ref2VA 需要至少一份參考圖片、影片或音訊。",
                    )
                elif not self.prompt:
                    self.request_prompt(chat_id)
                else:
                    self.start_selected_generation(chat_id, self.prompt)
                    self.show_menu(chat_id, message_id, "已送出生成工作")
        except (BotError, ValueError) as exc:
            self.send_safe(chat_id, str(exc))

    def handle_command(self, chat_id: str, text: str) -> None:
        lines = text.splitlines()
        parts = lines[0].strip().split()
        command = parts[0].split("@", 1)[0].lower()

        if command in {"/start", "/menu", "/help"}:
            try:
                self.show_menu(chat_id, force_new=True, section=MENU_MAIN)
            except (BotError, ValueError) as exc:
                self.send_safe(chat_id, str(exc))
            return
        if command == "/prompt":
            self.request_prompt(chat_id)
            return
        if command in {"/make", "/script"}:
            # One-line idea (+ optional duration) -> full H3 script via the local
            # LLM, entirely inside Telegram.
            remainder = text.split(None, 1)
            if len(remainder) > 1 and remainder[1].strip():
                self.awaiting_script_idea = False
                self.handle_script_idea(chat_id, remainder[1].strip())
            else:
                self.request_script_idea(chat_id)
            return
        if command in {"/gpu", "/gpuinfo"}:
            self.show_gpu_status(chat_id)
            return
        if command == "/chain":
            rest = text.split(None, 1)
            if len(rest) > 1 and rest[1].strip():
                if rest[1].strip().lower() in {"off", "stop", "cancel", "停"}:
                    was = self.chain_active()
                    self.chain_remaining = 0
                    self.send_safe(
                        chat_id,
                        "🔗 接力已停止（目前嗰段仍然會完成，之後唔會再接落去）。"
                        if was
                        else "目前冇接力進行中。",
                    )
                else:
                    self.handle_chain_setup(chat_id, rest[1].strip())
            else:
                self.request_chain_setup(chat_id)
            return
        if command in {"/scripttemplate", "/template"}:
            if len(parts) > 1:
                self.switch_script_template(chat_id, parts[1])
            else:
                self.send_safe(
                    chat_id,
                    f"📄 目前腳本模板：{script_template_display_name()}\n\n"
                    "/scripttemplate adult — 成人版（用你的 script_prompt.txt）\n"
                    "/scripttemplate general — 一般版（內建非成人模板）\n\n"
                    "也可以在面板按「📄 模板」切換。",
                )
            return
        if command in {"/scriptllm", "/script_llm"}:
            if len(parts) > 1:
                self.switch_script_llm(chat_id, parts[1])
            else:
                current = normalize_script_llm(
                    getattr(self, "script_llm", SCRIPT_LLM_DEFAULT)
                )
                self.send_safe(
                    chat_id,
                    f"🧠 目前腳本 LLM：{script_llm_display_name()}\n\n"
                    "/scriptllm local — 本機 llama.cpp（127.0.0.1:19092）\n"
                    f"/scriptllm cc — Command Code 雲端（{COMMANDCODE_MODEL}）\n\n"
                    "也可以在面板按「🧠 LLM」切換。",
                )
            return
        if command in {"/lang", "/language"}:
            if len(parts) >= 2:
                self.set_script_lang(chat_id, parts[1])
            else:
                current = normalize_script_lang(
                    getattr(self, "script_lang", SCRIPT_LANG_DEFAULT)
                )
                self.send_safe(
                    chat_id,
                    f"🌐 目前腳本語言：{SCRIPT_LANG_LABEL[current]}\n\n"
                    "/lang zh — 簡體中文\n"
                    "/lang en — English\n\n"
                    "也可以在面板按「🌐 腳本語言」切換。",
                )
            return
        if command in {"/prompt_file", "/custom"}:
            # Custom instruction file; re-read on every generation, so editing it
            # needs no Bot restart.
            self.show_script_prompt_file(chat_id)
            return
        if command == "/prompt_help":
            self.send_safe(chat_id, PROMPT_HELP_TEXT)
            return
        if command in {"/model", "/h3"}:
            if command in {"/h3"}:
                selected_model = MODEL_H3
            elif len(parts) >= 2:
                selected_model = normalize_model_mode(parts[1])
            else:
                self.show_menu(chat_id)
                return
            self.model_mode = selected_model
            self.save_settings()
            self.show_menu(chat_id, notice="目前使用：MiniMax H3 Turbo")
            return
        if command in {"/fl2va", "/first_last"}:
            self.input_mode = INPUT_MODE_FL2VA
            self.save_settings()
            self.send_safe(
                chat_id,
                "已選 FL2VA：請依次上傳首幀圖片、尾幀圖片，再輸入提示詞。",
            )
            return
        if command in {"/ref2va", "/reference"}:
            self.input_mode = INPUT_MODE_REF2VA
            self.save_settings()
            self.send_safe(
                chat_id,
                "已選 Ref2VA：連續上傳參考圖片／影片／音訊，完成後按選單按鈕。",
            )
            return
        if command == "/image":
            self.input_mode = "image"
            self.save_settings()
            self.send_safe(chat_id, "已切换到图片生视频；请发送一张图片。")
            return
        if command == "/text":
            self.input_mode = "text"
            self.save_settings()
            self.send_safe(chat_id, "已切换到文字生视频。")
            return
        if command in {"/comfy_status", "/comfy"}:
            self.send_safe(chat_id, self.comfy_status_text())
            return
        if command == "/comfy_start":
            try:
                result = start_comfyui_process(self.comfyui_vram_mode())
                self.touch_comfy_activity()
                self.send_safe(chat_id, result)
            except BotError as exc:
                self.send_safe(chat_id, str(exc))
            return
        if command == "/comfy_restart":
            try:
                cancelled = self.cancel_job_for_comfy_control()
                result = restart_comfyui_process(self.comfyui_vram_mode())
                self.touch_comfy_activity()
                prefix = "目前生成已取消。\n" if cancelled else ""
                self.send_safe(chat_id, prefix + result)
            except BotError as exc:
                self.send_safe(chat_id, str(exc))
            return
        if command == "/comfy_stop":
            try:
                cancelled = self.cancel_job_for_comfy_control()
                result = stop_comfyui_process()
                prefix = "目前生成已取消。\n" if cancelled else ""
                self.send_safe(chat_id, prefix + result)
            except BotError as exc:
                self.send_safe(chat_id, str(exc))
            return
        if command in {"/llm_status", "/llm", "/llama_status"}:
            self.send_safe(chat_id, self.llama_status_text())
            return
        if command in {"/llm_start", "/llama_start"}:
            try:
                self.send_safe(chat_id, start_llama_process())
            except BotError as exc:
                self.send_safe(chat_id, str(exc))
            return
        if command in {"/llm_restart", "/llama_restart"}:
            try:
                self.send_safe(chat_id, restart_llama_process())
            except BotError as exc:
                self.send_safe(chat_id, str(exc))
            return
        if command in {"/llm_stop", "/llama_stop"}:
            try:
                self.send_safe(chat_id, stop_llama_process())
            except BotError as exc:
                self.send_safe(chat_id, str(exc))
            return
        if command == "/bot_restart":
            try:
                self.restart_bot(chat_id)
            except BotError as exc:
                self.send_safe(chat_id, str(exc))
            return
        if command == "/progress":
            self.show_progress(chat_id)
            return
        if command == "/history":
            self.show_history(chat_id)
            return
        if command in {"/queue", "/queue_view"}:
            inline_prompt = " ".join(parts[1:]).strip()
            if len(lines) > 1:
                inline_prompt = (inline_prompt + "\n" + "\n".join(lines[1:])).strip()
            if inline_prompt:
                self.enqueue_story_prompts(chat_id, inline_prompt)
            else:
                self.show_queue(chat_id)
            return
        if command in {"/queue_add", "/add_story"}:
            inline_prompt = " ".join(parts[1:]).strip()
            if len(lines) > 1:
                inline_prompt = (inline_prompt + "\n" + "\n".join(lines[1:])).strip()
            if inline_prompt:
                self.enqueue_story_prompts(chat_id, inline_prompt)
            else:
                self.request_queue_prompt(chat_id)
            return
        if command == "/queue_start":
            started = self.start_next_queued_story(chat_id)
            if not started:
                self.send_safe(chat_id, "目前有任務、放大選擇，或排隊是空的；請稍後再試。")
            self.show_queue(chat_id)
            return
        if command == "/queue_clear":
            self.clear_story_queue(chat_id)
            return
        if command in {"/resume_long", "/resume_checkpoint"}:
            checkpoint_id = parts[1] if len(parts) > 1 else None
            self.resume_long_checkpoint(chat_id, checkpoint_id)
            return
        if command == "/extend":
            if len(parts) < 2:
                self.request_extension_duration(chat_id)
                return
            checkpoint_id: Optional[str] = None
            seconds_index = 1
            if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", parts[1]):
                checkpoint_id = parts[1]
                seconds_index = 2
                if len(parts) < 3:
                    self.request_extension_duration(chat_id, checkpoint_id)
                    return
            try:
                extra_seconds = validate_total_seconds(float(parts[seconds_index]))
            except (BotError, ValueError) as exc:
                self.send_safe(chat_id, str(exc))
                return
            record = (
                self.checkpoint_for_id(checkpoint_id)
                if checkpoint_id
                else self.latest_long_checkpoint()
            )
            if record is None:
                self.send_safe(chat_id, "目前找不到可以延續的完整長片。")
                return
            self.extension_checkpoint_id = self._checkpoint_id(record[0])
            self.extension_seconds = extra_seconds
            inline_prompt = " ".join(parts[seconds_index + 1 :]).strip()
            if len(lines) > 1:
                inline_prompt = (inline_prompt + "\n" + "\n".join(lines[1:])).strip()
            if inline_prompt:
                self.start_extension_generation(
                    chat_id,
                    self.extension_checkpoint_id,
                    extra_seconds,
                    inline_prompt,
                )
            else:
                self.request_extension_prompt(chat_id)
            return
        if command == "/pause":
            self.pause_current_job(chat_id)
            return
        if command in {"/resume", "/play"}:
            self.resume_current_job(chat_id)
            return
        if command == "/preview":
            self.request_video_preview(chat_id)
            return
        if command in {"/temperature", "/temp"}:
            self.send_safe(chat_id, temperature_report())
            return
        if command == "/cancel_shutdown":
            self.cancel_scheduled_shutdown(chat_id)
            return
        if command == "/status":
            with self.lock:
                job = self.job
                queue_count = len(self.story_queue)
            if job:
                self.send_safe(chat_id, self.progress_text())
            elif self.awaiting_queue_prompt:
                self.send_safe(chat_id, "等待你貼上要排隊的故事提示詞。")
            elif self.awaiting_prompt:
                self.send_safe(chat_id, "等待你貼上提示詞。")
            elif queue_count:
                self.show_queue(chat_id)
            else:
                self.show_menu(chat_id)
            return
        if command == "/cancel":
            self.awaiting_prompt = False
            self.awaiting_duration = False
            self.awaiting_extension_duration = False
            self.awaiting_extension_prompt = False
            self.awaiting_queue_prompt = False
            self.awaiting_script_idea = False
            self.awaiting_custom_prompt = ""
            self.awaiting_script_edit = ""
            self.awaiting_script_refine = False
            self.extension_seconds = None
            self.extension_checkpoint_id = None
            with self.lock:
                has_job = self.job is not None
            if has_job:
                self.abort_current_job(chat_id)
            elif self._shutdown_pending:
                self.cancel_scheduled_shutdown(chat_id)
            else:
                self.show_menu(chat_id, notice="已取消輸入")
            return
        if command in {"/duration", "/seconds"}:
            if len(parts) < 2:
                self.request_duration(chat_id)
                return
            try:
                self.set_total_seconds(float(parts[1]))
                self.awaiting_duration = False
                self.show_menu(chat_id, notice="自定義總片長已更新")
            except (BotError, ValueError) as exc:
                self.send_safe(chat_id, str(exc))
            return
        if command == "/long":
            if len(parts) < 5:
                self.send_safe(chat_id, "格式：/long 寬度 高度 steps 總秒數（30、60 或 120）")
                return
            try:
                total_seconds = validate_total_seconds(float(parts[4]))
                if not (MIN_TOTAL_SECONDS <= total_seconds <= MAX_TOTAL_SECONDS):
                    raise BotError("長片總秒數只支援 30、60 或 120 秒。")
                config = parse_config([parts[1], parts[2], parts[3], "15"])
                self.settings = config
                self.total_seconds = total_seconds
                self.save_settings()
            except (BotError, ValueError) as exc:
                self.send_safe(chat_id, str(exc))
                return
            inline_prompt = " ".join(parts[5:]).strip()
            if len(lines) > 1:
                inline_prompt = (inline_prompt + "\n" + "\n".join(lines[1:])).strip()
            if inline_prompt:
                self.prompt = inline_prompt
                self.save_settings()
                self.start_selected_generation(chat_id, self.prompt)
            else:
                self.request_prompt(chat_id)
            return
        if command == "/gen":
            if len(parts) < 5:
                self.send_safe(chat_id, "也可以使用按鈕；格式：/gen 寬度 高度 steps 秒數")
                return
            try:
                total_seconds = validate_total_seconds(float(parts[4]))
                self.settings = parse_config(
                    [parts[1], parts[2], parts[3], str(min(total_seconds, MAX_SEGMENT_SECONDS))]
                )
                self.total_seconds = total_seconds
                self.save_settings()
            except (BotError, ValueError) as exc:
                self.send_safe(chat_id, str(exc))
                return
            inline_prompt = " ".join(parts[5:]).strip()
            if len(lines) > 1:
                inline_prompt = (inline_prompt + "\n" + "\n".join(lines[1:])).strip()
            if inline_prompt:
                self.prompt = inline_prompt
                self.save_settings()
                self.start_selected_generation(chat_id, self.prompt)
            else:
                self.request_prompt(chat_id)
            return
        self.send_safe(chat_id, "輸入 /menu 開啟按鈕控制面板。")

    def configure_telegram_menu(self) -> None:
        commands = [
            {"command": "menu", "description": "開啟控制面板"},
            {"command": "progress", "description": "查看生成進度"},
            {"command": "prompt", "description": "輸入提示詞或上傳 TXT"},
            {"command": "model", "description": "查看目前使用的 MiniMax H3 Turbo"},
            {"command": "image", "description": "切換圖生視頻"},
            {"command": "text", "description": "切換文生視頻"},
            {"command": "fl2va", "description": "FL2VA 首尾幀模式"},
            {"command": "ref2va", "description": "Ref2VA 參考素材模式"},
            {"command": "duration", "description": "設定秒數"},
            {"command": "status", "description": "查看目前狀態"},
            {"command": "cancel", "description": "中止目前生成"},
            {"command": "pause", "description": "暫停長片"},
            {"command": "resume", "description": "繼續長片"},
            {"command": "preview", "description": "預覽已完成的長片片段"},
            {"command": "resume_long", "description": "從檢查點續做長片"},
            {"command": "extend", "description": "延續上一條長片"},
            {"command": "history", "description": "查看歷史長片 ID"},
            {"command": "queue", "description": "查看故事排隊"},
            {"command": "queue_add", "description": "加入一個或多個故事"},
            {"command": "queue_start", "description": "開始故事排隊"},
            {"command": "queue_clear", "description": "清空故事排隊"},
            {"command": "temperature", "description": "查看電腦溫度"},
            {"command": "comfy_status", "description": "查看 ComfyUI 狀態"},
            {"command": "comfy_start", "description": "啟動 ComfyUI"},
            {"command": "comfy_restart", "description": "重啟 ComfyUI"},
            {"command": "comfy_stop", "description": "關閉 ComfyUI"},
            {"command": "llm_status", "description": "查看本地 LLM 狀態"},
            {"command": "llm_start", "description": "以預設參數啟動本地 LLM"},
            {"command": "llm_restart", "description": "重啟本地 LLM"},
            {"command": "llm_stop", "description": "關閉本地 LLM"},
            {"command": "bot_restart", "description": "重啟 Telegram Bot"},
            {"command": "help", "description": "查看說明"},
        ]
        try:
            # Keep slash commands working when typed manually, but do not fill
            # Telegram's native Menu button with a second large control panel.
            self.telegram.set_my_commands([])
            self.telegram.set_chat_menu_button(self.allowed_chat_id)
        except BotError as exc:
            bot_log(f"Telegram menu setup failed: {exc}")
            print(f"Telegram menu setup failed: {exc}", flush=True)

    def run(self) -> None:
        self.configure_telegram_menu()
        self.show_menu(self.allowed_chat_id, notice="Turbo Telegram 控制器已啟動")
        bot_log(f"bot started pid={os.getpid()}")
        last_heartbeat = 0.0
        while True:
            try:
                updates = self.telegram.get_updates(self.offset)
                for update in updates:
                    self.offset = int(update["update_id"]) + 1
                    if update.get("callback_query"):
                        self.handle_callback(update["callback_query"])
                    elif update.get("message"):
                        self.handle_message(update["message"])
                now = time.time()
                if now - last_heartbeat >= 60.0:
                    last_heartbeat = now
                    with self.lock:
                        job = self.job
                    job_desc = (
                        f"seg {job.segment_index}/{job.segment_total} {job.prompt_id}"
                        if job is not None
                        else "idle"
                    )
                    bot_log(f"heartbeat ok pid={os.getpid()} job={job_desc}")
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                bot_log(f"polling error: {exc}")
                print(f"polling error: {exc}", flush=True)
                time.sleep(5)


def check_installation() -> None:
    config = GenerationConfig(1344, 768, 4, 5, valid_length(5))
    workflow = build_workflow(config, "A short bright test scene with clear motion and synchronized sound.")
    print("workflow=ok")
    print(f"template={T8_API_TEMPLATE}")
    print(f"comfy={COMFY_URL}")
    print(f"comfy_base={COMFYUI_BASE_DIR}")
    print(f"output={OUTPUT_DIR}")
    print(f"length={workflow['6']['inputs']['length']} frames ({config.actual_seconds:.2f}s)")
    print(f"task node={workflow['6']['class_type']}")
    print(
        f"sampler={workflow['7']['inputs']['sampler_name']} "
        f"steps={workflow['13']['inputs']['steps']} "
        f"scheduler={workflow['13']['inputs']['scheduler']}"
    )


class SingleInstanceGuard:
    """Prevent multiple hidden Bot processes from polling the same token."""

    ERROR_ALREADY_EXISTS = 183

    def __init__(self) -> None:
        self.handle: Any = None
        self.kernel32: Any = None

    def acquire(self) -> bool:
        if os.name != "nt":
            return True
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_wchar_p,
        ]
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_bool
        handle = kernel32.CreateMutexW(
            None,
            True,
            "Local\\MiniMaxH3TelegramBotSingleInstance",
        )
        if not handle:
            return False
        if ctypes.get_last_error() == self.ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self.kernel32 = kernel32
        self.handle = handle
        return True

    def release(self) -> None:
        if self.handle is not None and self.kernel32 is not None:
            self.kernel32.CloseHandle(self.handle)
            self.handle = None


def main() -> int:
    if "--check" in sys.argv:
        try:
            check_installation()
            return 0
        except Exception as exc:
            print(f"check failed: {exc}", file=sys.stderr)
            return 1

    token = os.environ.get("MINIMAX_TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("MINIMAX_TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print(
            "Missing MINIMAX_TELEGRAM_BOT_TOKEN or MINIMAX_TELEGRAM_CHAT_ID. "
            "Run Configure-MiniMax-H3-Telegram.cmd first.",
            file=sys.stderr,
        )
        return 2
    guard = SingleInstanceGuard()
    if not guard.acquire():
        print("MiniMax H3 Telegram Bot is already running; exiting.", flush=True)
        return 0
    try:
        TelegramMenuBot(token, chat_id).run()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        guard.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
