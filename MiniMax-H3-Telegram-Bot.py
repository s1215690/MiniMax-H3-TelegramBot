#!/usr/bin/env python3
"""Telegram controller for the local MiniMax H3 Turbo ComfyUI workflow.

The bot accepts generation parameters and a prompt from one authorized chat,
submits an API-format workflow to ComfyUI, then sends the synchronized MP4
back to Telegram. Secrets are intentionally read only from environment
variables and are never written to this workspace.
"""

from __future__ import annotations

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
from urllib.parse import urlencode
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
FFMPEG_PATH = os.environ.get("MINIMAX_FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
FFPROBE_PATH = os.environ.get("MINIMAX_FFPROBE", shutil.which("ffprobe") or "ffprobe")
NVIDIA_SMI_PATH = os.environ.get(
    "MINIMAX_NVIDIA_SMI",
    shutil.which("nvidia-smi")
    or r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)
SHUTDOWN_DELAY_SECONDS = 60
MAX_TELEGRAM_IMAGE_BYTES = 20 * 1024 * 1024
# Telegram Bot API currently accepts at most 50 MB for a bot-uploaded video.
# Keep a margin for multipart/form-data headers and the request boundary.
TELEGRAM_MAX_VIDEO_BYTES = 50_000_000
TELEGRAM_SAFE_VIDEO_BYTES = 48_000_000
TELEGRAM_AUDIO_BITRATE_KBPS = 128
MAX_TELEGRAM_PROMPT_BYTES = 512 * 1024
PROMPT_FILE_EXTENSIONS = {".txt", ".text"}
SAGE_ATTENTION_ENABLED = os.environ.get("MINIMAX_SAGE_ATTENTION", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
CLIP_NAME = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
UNET_NAME = "minimax_h3_fl2va_int8_convrot.safetensors"
LORA_NAME = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
OUTPUT_PREFIX = "MiniMaxH3/Telegram_Turbo"
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
SHOT_TRANSITION_SECONDS = 0.12
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
MOTION_CONTEXT_LENGTH = 22
MOTION_CONTEXT_EXTRA_SECONDS = MOTION_CONTEXT_LENGTH / 24.0
MODEL_H3 = "h3"
INPUT_MODE_TEXT = "text"
INPUT_MODE_IMAGE = "image"
INPUT_MODE_FL2VA = "fl2va"
INPUT_MODE_REF2VA = "ref2va"
INPUT_MODES = {
    INPUT_MODE_TEXT,
    INPUT_MODE_IMAGE,
    INPUT_MODE_FL2VA,
    INPUT_MODE_REF2VA,
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
REF2VA_UNET_NAME = os.environ.get(
    "MINIMAX_H3_REF2VA_MODEL",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
)
MAX_REF2VA_IMAGES = 9
MAX_REF2VA_VIDEOS = 3
MAX_REF2VA_AUDIOS = 3


def normalize_model_mode(value: Any) -> str:
    # Only the MiniMax H3 Turbo workflow is installed.
    return MODEL_H3


def normalize_input_mode(value: Any) -> str:
    mode = str(value or INPUT_MODE_TEXT).strip().lower()
    return mode if mode in INPUT_MODES else INPUT_MODE_TEXT


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

    def __init__(self, job: JobState):
        self.job = job
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
        ws_url = f"{ws_base}/ws?clientId=telegram-turbo-bot"
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
        raise BotError("格式：/gen 寬度 高度 steps 秒數\n例如：/gen 864 480 12 15")
    try:
        width, height, steps = (int(parts[0]), int(parts[1]), int(parts[2]))
        seconds = float(parts[3])
    except ValueError as exc:
        raise BotError("寬度、高度、steps 和秒數都要是數字。") from exc

    if width < 32 or height < 32 or width > 1344 or height > 768:
        raise BotError("解析度範圍是 32 至 1344×768。")
    if width % 32 or height % 32:
        raise BotError("寬度和高度必須是 32 的倍數，例如 608×352、864×480。")
    if width * height > 1344 * 768:
        raise BotError("解析度太高，先不要超過 1344×768。")
    if steps < 4 or steps > 20:
        raise BotError("steps 目前允許 4 至 20，建議 8 或 12。")

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
    r"(?:-|–|—|~|～|至|到)[ \t]*(?P<end>[0-9]+(?:\.[0-9]+)?)[ \t]*"
    r"(?:秒|s|sec|seconds?)?[ \t]*[）)][ \t]*[:：]?[ \t]*(?P<inline>.*)$"
)
SHARED_TAIL_SEPARATOR_RE = re.compile(
    r"(?m)^[ \t]*(?:-{3,}|─{3,}|={3,})[ \t]*$"
)


def parse_segmented_prompt(prompt: str) -> Optional[SegmentedPrompt]:
    """Parse GLOBAL/SEGMENT headings while preserving multiline prompt text."""
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
        if segment_duration > MAX_SEGMENT_SECONDS:
            raise BotError(
                f"{segment_total} SEGMENT blocks are too few for "
                f"{total_seconds:g} seconds; each SEGMENT can be at most "
                f"{MAX_SEGMENT_SECONDS:g} seconds. Add more SEGMENT blocks."
            )

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


def continuity_instruction(job: JobState) -> str:
    audio_rule = (
        " Keep the same music bed, tempo, instrumentation and ambience as <Audio 1>, "
        "while generating new audio for this segment."
        if job.audio_reference_name
        else ""
    )
    if job.segment_index == 1:
        return (
            "This is the opening segment. Perform only the CURRENT SEGMENT action. "
            "Do not advance into later segments."
            + audio_rule
        )
    return (
        "Directly continue from the supplied first frame. During the first second, "
        "keep the same character identity, pose, framing and motion direction with "
        "only small natural movement; then perform only the CURRENT SEGMENT action. "
        "Do not restart the scene or replay any earlier action."
        + audio_rule
    )


def segment_prompt(job: JobState) -> str:
    """Give each long-video segment a focused time window and continuity rule."""
    if job.shot_plan:
        shot = job.shot_plan[job.segment_index - 1]
        continuity = continuity_instruction(job)
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
        return "\n\n".join(blocks)

    parsed = parse_segmented_prompt(job.prompt)
    start_seconds = (job.segment_index - 1) * MAX_SEGMENT_SECONDS
    end_seconds = min(job.segment_index * MAX_SEGMENT_SECONDS, job.total_seconds)
    continuity = continuity_instruction(job)

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


def _next_workflow_node_id(workflow: dict[str, Any]) -> str:
    """Return an unused numeric node id without overwriting template nodes."""
    numeric_ids = [int(node_id) for node_id in workflow if str(node_id).isdigit()]
    next_id = max(numeric_ids, default=0) + 1
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
    sampler_type = str(workflow.get("7", {}).get("class_type", "未設定"))
    if sampler_type == "MiniMaxH3MultiRateSamplerEXPT8":
        steps = (
            f"影片 {_workflow_node_input(workflow, '7', 'video_steps')} / "
            f"音訊 {_workflow_node_input(workflow, '7', 'audio_steps')}"
        )
    elif sampler_type == "MiniMaxH3TurboSampler":
        steps = (
            f"{_workflow_node_input(workflow, '13', 'steps')} steps / "
            f"scheduler {_workflow_node_input(workflow, '13', 'scheduler', 'simple')}"
        )
    else:
        steps = str(_workflow_node_input(workflow, "7", "steps", "未設定"))

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
    if sampler_type == "MiniMaxH3MultiRateSamplerEXPT8":
        acceleration.append("MultiRate EXPT8")
    elif sampler_type == "MiniMaxH3TurboSampler":
        acceleration.append("Author Turbo Sampler")
    for class_type, label in acceleration_labels.items():
        if class_type in class_types and label not in acceleration:
            acceleration.append(label)

    try:
        vram_label = comfyui_vram_mode_label(vram_mode)
    except NameError:
        vram_label = vram_mode
    return "\n".join(
        [
            f"模式：{_workflow_node_input(workflow, '6', 'task_type', '未設定')} / "
            f"音訊：{_workflow_node_input(workflow, '6', 'audio_mode', '未設定')}",
            f"採樣：{sampler_type} | 步數：{steps}",
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
    save_latent_prefix: Optional[str] = None,
    save_latent_clip_index: Optional[int] = None,
) -> dict[str, Any]:
    if not T8_API_TEMPLATE.is_file():
        raise BotError(f"找不到 T8 API 工作流模板：{T8_API_TEMPLATE}")
    with T8_API_TEMPLATE.open("r", encoding="utf-8") as handle:
        workflow = json.load(handle)

    workflow["1"]["inputs"]["vae_name"] = VIDEO_VAE
    workflow["2"]["inputs"]["vae_name"] = AUDIO_VAE
    workflow["3"]["inputs"]["clip_name"] = CLIP_NAME
    mode = normalize_input_mode(generation_mode)
    workflow["4"]["inputs"]["unet_name"] = (
        REF2VA_UNET_NAME if mode == INPUT_MODE_REF2VA else UNET_NAME
    )

    # INT8 ConvRot uses the bypass model-only loader for the Turbo LoRA.
    workflow["5"]["class_type"] = "LoraLoaderBypassModelOnly"
    workflow["5"]["inputs"]["lora_name"] = LORA_NAME
    workflow["5"]["inputs"]["strength_model"] = 1.0

    conditioning = workflow["6"]["inputs"]
    reference_image_names = list(reference_image_names or [])
    reference_video_names = list(reference_video_names or [])
    reference_audio_names = list(reference_audio_names or [])
    if len(reference_image_names) > MAX_REF2VA_IMAGES:
        raise BotError(f"Ref2VA 最多支援 {MAX_REF2VA_IMAGES} 張參考圖。")
    if len(reference_video_names) > MAX_REF2VA_VIDEOS:
        raise BotError(f"Ref2VA 最多支援 {MAX_REF2VA_VIDEOS} 段參考影片。")
    if len(reference_audio_names) > MAX_REF2VA_AUDIOS:
        raise BotError(f"Ref2VA 最多支援 {MAX_REF2VA_AUDIOS} 段參考音訊。")
    if mode == INPUT_MODE_REF2VA and not (
        reference_image_names or reference_video_names or reference_audio_names
    ):
        raise BotError("Ref2VA 至少需要一張參考圖片、一段參考影片或一段參考音訊。")
    prompt_enhancer = workflow.get("30", {}).get("class_type") == "MiniMaxH3PromptEnhancer"
    if not prompt_enhancer:
        conditioning["prompt"] = prompt.strip()
    conditioning["width"] = config.width
    conditioning["height"] = config.height
    conditioning["length"] = config.length
    if motion_context:
        if not context_video_name or not context_latent_path:
            raise BotError(
                "Motion Context 需要上一段影片和上一段 AV latent。"
            )
        # The Motion Context node supplies the continuation keyframes and the
        # previous AV latent. Do not also attach the old single-frame/reference
        # inputs: that would turn this back into the stable reference path.
        conditioning["task_type"] = "T2VA"
        conditioning["audio_mode"] = "native"
        conditioning["add_source_as_reference"] = False
        conditioning["prompt_primary_audio_ordinal"] = 0
    elif mode == INPUT_MODE_REF2VA:
        conditioning["task_type"] = "Ref2VA"
        conditioning["audio_mode"] = (
            "reference_only"
            if audio_reference_name or reference_audio_names
            else "native"
        )
        conditioning["add_source_as_reference"] = bool(
            audio_reference_name or reference_audio_names
        )
        conditioning["prompt_primary_audio_ordinal"] = (
            1 if audio_reference_name or reference_audio_names else 0
        )
    elif mode == INPUT_MODE_FL2VA:
        conditioning["task_type"] = "auto" if audio_reference_name else "FL2VA"
        conditioning["audio_mode"] = "reference_only" if audio_reference_name else "native"
        conditioning["add_source_as_reference"] = bool(audio_reference_name)
        conditioning["prompt_primary_audio_ordinal"] = 1 if audio_reference_name else 0
    else:
        # Audio references plus a continuation first frame require T8 Auto,
        # which resolves to Hybrid; explicit I2VA rejects reference media.
        conditioning["task_type"] = (
            "auto"
            if audio_reference_name
            else ("I2VA" if image_name else "T2VA")
        )
        conditioning["audio_mode"] = "reference_only" if audio_reference_name else "native"
        conditioning["add_source_as_reference"] = bool(audio_reference_name)
        conditioning["prompt_primary_audio_ordinal"] = 1 if audio_reference_name else 0
    if prompt_enhancer:
        workflow["30"]["inputs"]["manual_prompt"] = prompt.strip()
        workflow["30"]["inputs"]["mode_report"] = (
            f"H3 task_type={conditioning['task_type']}; "
            f"audio_mode={conditioning['audio_mode']}; "
            "preserve the user's requested content and timing."
        )
    if image_name and not motion_context and mode != INPUT_MODE_REF2VA:
        image_node_id = _next_workflow_node_id(workflow)
        workflow[image_node_id] = {
            "inputs": {"image": image_name},
            "class_type": "LoadImage",
            "_meta": {"title": "Telegram input image"},
        }
        conditioning["first_frame"] = [image_node_id, 0]
        if prompt_enhancer:
            workflow["30"]["inputs"]["image"] = [image_node_id, 0]
    if last_image_name and not motion_context and mode == INPUT_MODE_FL2VA:
        last_image_node_id = _next_workflow_node_id(workflow)
        workflow[last_image_node_id] = {
            "inputs": {"image": last_image_name},
            "class_type": "LoadImage",
            "_meta": {"title": "Telegram FL2VA last frame"},
        }
        conditioning["last_frame"] = [last_image_node_id, 0]
    for index, reference_name in enumerate(reference_image_names):
        ref_node_id = _next_workflow_node_id(workflow)
        workflow[ref_node_id] = {
            "inputs": {"image": reference_name},
            "class_type": "LoadImage",
            "_meta": {"title": f"Telegram Ref2VA image {index + 1}"},
        }
        conditioning[f"ref_images.ref_image_{index}"] = [ref_node_id, 0]
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
        conditioning[f"ref_videos.ref_video_{index}"] = [components_node_id, 0]
        conditioning[f"ref_video_audios.ref_video_audio_{index}"] = [
            components_node_id,
            1,
        ]
    for index, reference_name in enumerate(reference_audio_names):
        audio_node_id = _next_workflow_node_id(workflow)
        workflow[audio_node_id] = {
            "inputs": {"audio": reference_name},
            "class_type": "LoadAudio",
            "_meta": {"title": f"Telegram Ref2VA audio {index + 1}"},
        }
        conditioning[f"ref_audios.ref_audio_{index}"] = [audio_node_id, 0]
    if audio_reference_name and not motion_context:
        audio_node_id = _next_workflow_node_id(workflow)
        workflow[audio_node_id] = {
            "inputs": {"audio": audio_reference_name},
            "class_type": "LoadAudio",
            "_meta": {"title": "Previous segment audio reference"},
        }
        conditioning["drive_audio"] = [audio_node_id, 0]

    if motion_context:
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
                "clip_index": 0,
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
                "context_length": MOTION_CONTEXT_LENGTH,
                "encode_mode": "video",
                "anchor_mode": "head",
                "crop": "disabled",
                "audio_context_length": MOTION_CONTEXT_LENGTH,
                "audio_mode": "timeline",
                "context_latent": ["17", 0],
            },
            "class_type": "MiniMaxH3MotionContext",
            "_meta": {"title": "Experimental H3 AV latent continuation"},
        }
        workflow["9"]["inputs"]["conditioning"] = ["18", 0]
        workflow["19"] = {
            "inputs": {
                "images": ["11", 0],
                "audio": ["11", 1],
                "trim_frames": ["18", 1],
                "fps": 24.0,
                "match_tail": True,
            },
            "class_type": "MiniMaxH3MotionContextTrim",
            "_meta": {"title": "Trim duplicated context audio and frames"},
        }
        workflow["12"]["inputs"]["images"] = ["19", 0]
        workflow["12"]["inputs"]["audio"] = ["19", 1]

    if save_latent_prefix:
        workflow["20"] = {
            "inputs": {
                "latent": ["10", 0],
                "filename_prefix": save_latent_prefix,
                "clip_index": int(save_latent_clip_index or 0),
            },
            "class_type": "MiniMaxH3MotionContextSaveLatent",
            "_meta": {"title": "Save H3 AV latent for next segment"},
        }

    sampler_inputs = workflow["7"]["inputs"]
    sampler_type = workflow["7"].get("class_type")
    if sampler_type == "MiniMaxH3MultiRateSamplerEXPT8":
        sampler_inputs["video_steps"] = min(4, config.steps)
        sampler_inputs["audio_steps"] = config.steps
    elif sampler_type == "MiniMaxH3TurboSampler":
        scheduler_inputs = workflow.setdefault("13", {}).setdefault("inputs", {})
        scheduler_inputs["scheduler"] = "simple"
        scheduler_inputs["steps"] = config.steps
        scheduler_inputs["denoise"] = 1.0
    else:
        sampler_inputs["steps"] = config.steps
    if "shift_video" in sampler_inputs:
        sampler_inputs["shift_video"] = 12.0
    if "shift_audio" in sampler_inputs:
        sampler_inputs["shift_audio"] = 3.0
    workflow["8"]["inputs"]["noise_seed"] = secrets.randbits(63)
    workflow["12"]["inputs"]["filename_prefix"] = output_prefix
    return workflow


def ltx_workflow_usage_report(vram_mode: str) -> str:
    try:
        vram_label = comfyui_vram_mode_label(vram_mode)
    except NameError:
        vram_label = vram_mode
    return "\n".join(
        [
            "模型：LTX 2.3 PinkCherry NSFW v1.8",
            "主模型：PinkCherry_FineTune_Q5_K_M_v18_LTX23.gguf",
            "文字編碼器：Gemma 3 12B Heretic v2 INT4 + LTX projection",
            "加速：LTX 2.3 distilled LoRA（工作流內置原生雙階段採樣）",
            "解碼：LTX23 video VAE + audio VAE；目前不使用 H3 Turbo LoRA",
            f"顯存模式：{vram_label}",
        ]
    )


def ltx_workflow_usage_report(vram_mode: str) -> str:
    """Report the actual LTX model/LoRA selected by the isolated branch."""
    try:
        vram_label = comfyui_vram_mode_label(vram_mode)
    except NameError:
        vram_label = vram_mode
    if ltx_author_model_ready():
        model_line = f"主模型：{LTX_AUTHOR_MODEL_NAME}"
        lora_line = f"LoRA：{LTX_AUTHOR_LORA_NAME}（strength 0.6）"
        graph_line = "作者 v1.8 graph：Chunk Feed-Forward + Preview Override + NAG"
    else:
        model_line = "主模型：PinkCherry_FineTune_Q5_K_M_v18_LTX23.gguf"
        lora_line = "LoRA：dynamic distilled LTX 2.3（strength 0.5）"
        graph_line = "簡化 LTX graph（作者 INT8 尚未完成下載）"
    return "\n".join(
        [
            "LTX 2.3 PinkCherry v1.8",
            model_line,
            "Text encoder：Gemma 3 12B Heretic v2 INT4 + LTX projection",
            lora_line,
            graph_line,
            "VAE：LTX23 video VAE + audio VAE；SaveVideo H264 CRF 8",
            f"VRAM：{vram_label}",
        ]
    )


def apply_ltx_author_graph(workflow: dict[str, Any]) -> dict[str, Any]:
    """Upgrade only the LTX branch to the author's standard-checkpoint graph.

    The original MiniMax H3 graph is built by ``build_workflow`` and is not
    touched here.  The old Q5/GGUF LTX graph remains the fallback until the
    separate author INT8 checkpoint has been fully downloaded.
    """
    loader_id = next(
        node_id for node_id, node in workflow.items()
        if node.get("class_type") == "UnetLoaderGGUF"
    )
    lora_id = next(
        node_id for node_id, node in workflow.items()
        if node.get("class_type") == "LoraLoaderModelOnly"
    )
    clip_id = next(
        node_id for node_id, node in workflow.items()
        if node.get("class_type") == "DualCLIPLoaderGGUF"
    )
    video_vae_id = next(
        node_id for node_id, node in workflow.items()
        if node.get("class_type") == "VAELoader"
        and "video_vae" in str(node.get("inputs", {}).get("vae_name", ""))
    )
    guider_ids = [
        node_id for node_id, node in workflow.items()
        if node.get("class_type") == "CFGGuider"
    ]
    sigma_nodes = [
        node_id for node_id, node in workflow.items()
        if node.get("class_type") == "ManualSigmas"
    ]
    workflow[loader_id] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": LTX_AUTHOR_MODEL_NAME},
    }
    workflow[lora_id]["inputs"]["lora_name"] = LTX_AUTHOR_LORA_NAME
    workflow[lora_id]["inputs"]["strength_model"] = 0.6
    low_sigma_id = next(
        node_id for node_id in sigma_nodes
        if str(workflow[node_id]["inputs"].get("sigmas", "")).strip().startswith("1.0")
    )
    high_sigma_id = next(node_id for node_id in sigma_nodes if node_id != low_sigma_id)
    low_sampler_id = workflow[low_sigma_id]
    workflow[low_sigma_id]["inputs"]["sigmas"] = (
        "1.0, 0.998, 0.995, 0.99, 0.982, 0.97, 0.94, 0.89, "
        "0.82, 0.73, 0.62, 0.50, 0.38, 0.27, 0.18, 0.11, "
        "0.06, 0.03, 0.01, 0.0"
    )
    workflow[high_sigma_id]["inputs"]["sigmas"] = "0.85, 0.7250, 0.4219, 0.0"
    for sampler in workflow.values():
        if sampler.get("class_type") != "SamplerCustomAdvanced":
            continue
        sigma_ref = sampler.get("inputs", {}).get("sigmas")
        if not isinstance(sigma_ref, list):
            continue
        sampler_select_id = sampler["inputs"].get("sampler", [None, 0])[0]
        if sampler_select_id not in workflow:
            continue
        if sigma_ref[0] == low_sigma_id:
            workflow[sampler_select_id]["inputs"]["sampler_name"] = "euler_ancestral_cfg_pp"
        elif sigma_ref[0] == high_sigma_id:
            workflow[sampler_select_id]["inputs"]["sampler_name"] = "euler_cfg_pp"

    # These are the quality/denoise-control nodes present in the author's
    # v1.8 workflow.  They run on the standard checkpoint, not on the old
    # dynamic GGUF loader (which cannot evaluate NAG connectors safely).
    workflow["1001"] = {
        "class_type": "LTXVChunkFeedForward",
        "inputs": {
            "model": [lora_id, 0],
            "chunks": 2,
            "dim_threshold": 4096,
        },
    }
    workflow["1002"] = {
        "class_type": "LTX2SamplingPreviewOverride",
        "inputs": {
            "model": ["1001", 0],
            "vae": [video_vae_id, 0],
            "preview_rate": 8,
        },
    }
    workflow["1003"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [clip_id, 0],
            "text": (
                "logos, voice over, narration, off camera speech, watermarks, "
                "poor anatomy, low detail, slow motion, slow, boring"
            ),
        },
    }
    workflow["1004"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [clip_id, 0],
            "text": (
                "logos, voice over, narration, off camera speech, watermarks, "
                "poor anatomy, low detail"
            ),
        },
    }
    workflow["1005"] = {
        "class_type": "LTX2_NAG",
        "inputs": {
            "model": ["1002", 0],
            "nag_cond_video": ["1003", 0],
            "nag_cond_audio": ["1004", 0],
            "nag_scale": 11.0,
            "nag_alpha": 0.25,
            "nag_tau": 2.5,
            "inplace": True,
        },
    }
    for guider_id in guider_ids:
        workflow[guider_id]["inputs"]["model"] = ["1005", 0]

    for node in workflow.values():
        if not isinstance(node, dict) or node.get("class_type") != "SaveVideo":
            continue
        inputs = node.setdefault("inputs", {})
        inputs.update(
            {
                "format": "mp4",
                "codec": "h264",
                "encoding": "re-encode",
                "crf": 8,
            }
        )
    return workflow


def build_ltx_workflow(
    config: GenerationConfig,
    prompt: str,
    output_prefix: str,
    image_name: Optional[str] = None,
) -> dict[str, Any]:
    """Build an isolated LTX 2.3 API graph without touching the H3 graph."""

    template_path = LTX_I2V_API_TEMPLATE if image_name else LTX_T2V_API_TEMPLATE
    if not template_path.is_file():
        raise BotError(f"找不到 LTX 2.3 API 工作流模板：{template_path}")
    with template_path.open("r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    if not isinstance(workflow, dict):
        raise BotError("LTX 2.3 API 工作流格式無效。")

    # The converted native template exposes these four values as PrimitiveInt
    # nodes.  Identify them by their template defaults so the Bot remains
    # compatible with both T2V and I2V node IDs.
    primitive_nodes = [
        node
        for node in workflow.values()
        if isinstance(node, dict) and node.get("class_type") == "PrimitiveInt"
    ]

    def set_primitive(default: int, value: int) -> None:
        for node in primitive_nodes:
            inputs = node.setdefault("inputs", {})
            if inputs.get("value") == default:
                inputs["value"] = int(value)
                return
        raise BotError(f"LTX 工作流缺少 PrimitiveInt 參數（預設值 {default}）。")

    set_primitive(512, config.width)
    set_primitive(288, config.height)
    set_primitive(5, max(2, int(round(config.actual_seconds))))
    set_primitive(24, 24)

    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        if class_type == "CLIPTextEncode" and inputs.get("text") == "replace_with_prompt":
            inputs["text"] = prompt.strip()
        elif class_type == "ResizeImageMaskNode":
            # ComfyUI API v3 DynamicCombo inputs use a selector and dotted
            # child names; a nested dict is silently discarded by the API.
            inputs["resize_type"] = "scale dimensions"
            inputs["resize_type.width"] = int(config.width)
            inputs["resize_type.height"] = int(config.height)
            inputs["resize_type.crop"] = "center"
            inputs["scale_method"] = "lanczos"
        elif class_type == "ResizeImagesByLongerEdge":
            inputs["longer_edge"] = max(int(config.width), int(config.height))
        elif class_type == "VAEDecodeTiled":
            # Small temporal tiles can show up as frame-to-frame snow and
            # brightness jumps.  A 10-second 864x480 validation run fits on
            # the 10GB card without temporal tiling, so keep short clips in
            # one decode window.  Longer clips use large 240-frame windows;
            # this keeps memory bounded while reducing the number of seams.
            inputs["tile_size"] = min(512, max(256, max(config.width, config.height)))
            inputs["overlap"] = 64
            # A nominal 10-second request is encoded as about 10.04 seconds
            # at 24 fps, so include that small frame-rounding margin.
            if float(config.actual_seconds) <= 10.5:
                inputs["temporal_size"] = 4096
                inputs["temporal_overlap"] = 8
            else:
                inputs["temporal_size"] = 240
                inputs["temporal_overlap"] = 16
        elif class_type == "RandomNoise":
            inputs["noise_seed"] = secrets.randbits(63)
        elif class_type == "LoadImage" and image_name:
            inputs["image"] = image_name
        elif class_type == "SaveVideo":
            inputs["filename_prefix"] = (
                output_prefix if output_prefix != OUTPUT_PREFIX else "LTX23/Telegram"
            )

    if ltx_author_model_ready():
        workflow = apply_ltx_author_graph(workflow)
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


def json_request(url: str, payload: Optional[dict[str, Any]] = None, timeout: float = 45.0) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raise BotError(f"連線失敗：{http_error_detail(exc)}") from exc
    except (URLError, TimeoutError) as exc:
        raise BotError(f"連線失敗：{exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise BotError("服務回傳了無法解析的資料。") from exc


def comfy_post(path: str, payload: Optional[dict[str, Any]] = None) -> Any:
    return json_request(f"{COMFY_URL}{path}", payload)


def unload_comfy_models() -> None:
    """Release a previous model before switching between H3 and LTX."""
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
    except BotError:
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


def multipart_request(url: str, fields: dict[str, str], file_field: str, file_path: Path) -> Any:
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
            b"Content-Type: video/mp4\r\n\r\n",
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
            f"asetpts=PTS-STARTPTS,aresample=48000[a{index}]"
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
        filters.append(f"[{index}:a]aresample=48000[an{index}]")
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
            "例如：/gen 864 480 12 15\n"
            "下一則訊息貼完整提示詞即可。\n\n"
            "也可同一則訊息輸入：\n"
            "/gen 864 480 12 15\n你的提示詞\n\n"
            "/status 查看狀態\n"
            "/progress 查看即時生成進度\n"
            "/pause 暫停長片（在目前鏡頭完成後）\n"
            "/resume 或 /play 繼續長片\n"
            "長片顯存不足時會保留已完成鏡頭，自動逐級降低解析度重試\n"
             "/resume_long 從失敗檢查點繼續長片\n"
             "/extend 秒數 [提示詞] 從上一條完整長片尾端延續\n"
             "/history 查看歷史長片並選擇 ID\n"
             "/queue 查看故事排隊\n"
             "/queue_add 加入一個或多個故事\n"
             "/queue_start 開始排隊\n"
             "/queue_clear 清空等待中的故事\n"
            "/temperature 查看 GPU／CPU 溫度\n"
            "/cancel_shutdown 取消已排程的自動關機\n"
            "/comfy_restart 重啟 ComfyUI\n"
            "/comfy_stop 關閉 ComfyUI\n"
            "/comfy_start 啟動 ComfyUI（閒置 5 分鐘會自動關閉）\n"
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
        lines = [
            f"📊 {'SeedVR2 放大' if job.task_type == 'seedvr2' else 'MiniMax H3'} 進度",
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
            self.send_safe(chat_id, "格式：/gen 寬度 高度 steps 秒數\n例如：/gen 864 480 12 15")
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
                task_type=normalize_model_mode(task_type),
                generation_mode=mode,
            )
            job.resume_event.set()
            self.job = job
        self.touch_comfy_activity()
        thread = threading.Thread(target=self.run_job, args=(job,), daemon=True)
        thread.start()
        return True

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

    def run_segment(
        self,
        job: JobState,
        announce: bool = True,
        motion_context: bool = False,
        context_video_name: Optional[str] = None,
        context_latent_path: Optional[str] = None,
        save_latent_prefix: Optional[str] = None,
        save_latent_clip_index: Optional[int] = None,
    ) -> Path:
        segment_started_at = time.time()
        image_name: Optional[str] = None
        last_image_name: Optional[str] = None
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
        if not motion_context and job.generation_mode == INPUT_MODE_REF2VA:
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
        if (
            workflow_mode == INPUT_MODE_FL2VA
            and job.segment_index > 1
            and not last_image_name
        ):
            workflow_mode = INPUT_MODE_IMAGE
        workflow = build_workflow(
            job.config,
            segment_prompt(job),
            job.output_prefix,
            image_name=image_name,
            last_image_name=last_image_name,
            reference_image_names=job.comfy_reference_image_names,
            reference_video_names=job.comfy_reference_video_names,
            reference_audio_names=job.comfy_reference_audio_names,
            audio_reference_name=(None if motion_context else job.audio_reference_name),
            generation_mode=workflow_mode,
            motion_context=motion_context,
            context_video_name=context_video_name,
            context_latent_path=context_latent_path,
            save_latent_prefix=save_latent_prefix,
            save_latent_clip_index=save_latent_clip_index,
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

    def run_job(self, job: JobState) -> None:
        bot_log(
            f"job start {job.config.width}x{job.config.height} "
            f"steps={job.config.steps} {job.config.actual_seconds:.2f}s"
        )
        try:
            self.ensure_comfyui_ready(job)
            video_path = self.run_segment(job, announce=True)
            with job.progress_lock:
                job.progress_phase = "uploading"
            model_label = "MiniMax H3 Turbo 完成"
            caption = (
                f"{model_label}\n{job.config.width}×{job.config.height} | "
                f"{job.config.steps} steps | {job.config.actual_seconds:.2f} 秒"
            )
            self.telegram.send_video(job.chat_id, video_path, caption)
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
            # use `gifs`, `videos`, or `files`.  Accept all of them so LTX
            # outputs are delivered instead of being reported as missing.
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
    STEPS = (4, 8, 12)

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
        self.shutdown_after_generation = self.load_saved_shutdown_after_generation()
        self._shutdown_pending = False
        self.awaiting_prompt = False
        self.awaiting_duration = False
        self.awaiting_extension_duration = False
        self.awaiting_extension_prompt = False
        self.awaiting_queue_prompt = False
        self.extension_seconds: Optional[float] = None
        self.extension_checkpoint_id: Optional[str] = None
        self.story_queue: list[QueuedStory] = self.load_story_queue()
        self._queue_starting = False
        self.menu_message_id: Optional[int] = None
        self.menu_section = MENU_MAIN
        self.control_keyboard_sent = False

    @staticmethod
    def default_settings() -> GenerationConfig:
        return parse_config(["864", "480", "12", "15"])

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
            "task_type": normalize_model_mode(job.task_type),
            "generation_mode": normalize_input_mode(job.generation_mode),
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
            segment_paths: dict[int, Path] = {}
            for path in directory.glob("segment_*_00001.mp4"):
                match = re.fullmatch(r"segment_(\d+)_00001\.mp4", path.name)
                if match:
                    segment_paths[int(match.group(1))] = path
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
            segment_paths: list[Path] = []
            for path in directory.glob("segment_*_00001.mp4"):
                if re.fullmatch(r"segment_\d+_00001\.mp4", path.name):
                    segment_paths.append(path)
            segment_paths.sort(key=lambda path: int(re.search(r"segment_(\d+)_", path.name).group(1)))
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
            "一次輸入多個故事時，請用獨立一行的 --- 分隔；每個故事會使用目前的解析度、steps 和片長設定。",
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
        new_items = [
            QueuedStory(
                item_id=f"q_{secrets.token_hex(4)}",
                prompt=prompt,
                config=config,
                total_seconds=total_seconds,
                input_image_path=input_image_path,
                last_image_path=last_image_path,
                reference_image_paths=reference_image_paths,
                reference_video_paths=reference_video_paths,
                reference_audio_paths=reference_audio_paths,
                generation_mode=self.input_mode,
                model_mode=self.model_mode,
            )
            for prompt in prompts
        ]
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
        self.start_next_queued_story(chat_id)

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
                [
                    {"text": "✍️ 輸入／更換提示詞", "callback_data": "prompt"},
                    {"text": "🧹 清除提示詞", "callback_data": "clear"},
                ],
                [{"text": "🗑 清除上傳素材", "callback_data": "clear_image"}],
            ]
            if self.input_mode == INPUT_MODE_REF2VA:
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
                        {
                            "text": "⚙️ 片長／解析度／steps",
                            "callback_data": "menu:settings",
                        }
                    ],
                ]
            )
            if active_job is not None:
                rows.append(job_control_row)
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
                            "text": "🖥️ 電腦／ComfyUI／Bot",
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
                [{"text": "🗑 清除上傳素材", "callback_data": "clear_image"}],
            ]
            if self.input_mode == INPUT_MODE_REF2VA:
                rows.append(
                    [{"text": "✅ 完成參考素材上傳", "callback_data": "media_done"}]
                )
            rows.append([{"text": "🎛️ 前往生成模式", "callback_data": "menu:mode"}])
            rows.extend(back_row())
        elif section == MENU_SETTINGS:
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
            ]
            rows.extend(back_row())
        elif section == MENU_MODE:
            rows = [mode_row, reference_mode_row]
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

    def start_selected_generation(self, chat_id: str, prompt: str) -> bool:
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
                task_type=normalize_model_mode(payload.get("task_type", MODEL_H3)),
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
                task_type=normalize_model_mode(payload.get("task_type", MODEL_H3)),
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
            self.send_safe(chat_id, f"長片時間軸格式錯誤：{exc}")
            return False
        segment_total = len(plan.shots)
        if segment_total < 2:
            self.send_safe(chat_id, "長片時間軸至少需要兩個鏡頭。")
            return False
        batch_prefix = f"{OUTPUT_PREFIX}/long_{uuid.uuid4().hex[:12]}"
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
                task_type=normalize_model_mode(task_type),
                generation_mode=mode,
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
            caption = (
                "MiniMax H3 Turbo 長片已提早中止，已合成部分結果\n"
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

        def report_partial() -> None:
            nonlocal partial_reported
            if partial_reported:
                return
            partial_reported = True
            self.mark_long_checkpoint(job, "cancelled")
            self.send_partial_long_result(job, video_paths, base_prefix)

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
                    job.generation_mode != INPUT_MODE_REF2VA
                    and
                    LONG_CONTINUITY_MODE in {"motion_context", "motion", "experimental"}
                    and motion_context_nodes_available()
                )
            else:
                motion_context_enabled = bool(job.resume_motion_context)
                if motion_context_enabled and not motion_context_nodes_available():
                    motion_context_enabled = False
            if LONG_CONTINUITY_MODE in {"motion_context", "motion", "experimental"}:
                if motion_context_enabled:
                    self.send_safe(
                        job.chat_id,
                        "已啟用實驗性 H3 Motion Context：後續鏡頭會接續上一段的尾幀、"
                        "影像 latent 和音訊 latent。",
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
                            save_latent_prefix=latent_prefix,
                            save_latent_clip_index=index if latent_prefix else None,
                        )
                        break
                    except Exception as exc:
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
            caption = (
                f"MiniMax H3 Turbo 長片完成\n{job.total_seconds:.0f} 秒 | "
                f"{base_config.width}×{base_config.height} | {base_config.steps} steps | "
                f"{job.segment_total} 鏡頭合併"
            )
            self.telegram.send_video(job.chat_id, output_path, caption)
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
            MENU_SYSTEM: "ComfyUI／系統",
            MENU_HISTORY: "歷史／長片／排隊",
        }
        current = self.settings
        prompt_status = f"已輸入（{len(self.prompt)} 字）" if self.prompt else "尚未輸入"
        mode_text = {
            INPUT_MODE_TEXT: "T2VA 文字生視頻",
            INPUT_MODE_IMAGE: "I2VA 圖片生視頻",
            INPUT_MODE_FL2VA: "FL2VA 首尾幀生視頻",
            INPUT_MODE_REF2VA: "Ref2VA 參考素材生視頻",
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
                if self.input_mode == INPUT_MODE_REF2VA
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
            MENU_SETTINGS: "這裡集中調整片長、解析度和 steps；其他功能仍在主選單。",
            MENU_MODE: "T2VA／I2VA／FL2VA／Ref2VA 會在生成時使用對應接線。",
            MENU_DURATION: "超過 15 秒會按提示詞時間軸自動分段。",
            MENU_QUALITY: "解析度越高越清晰，也越容易需要更多顯存。",
            MENU_JOB: "生成中的任務可以查看進度、暫停、繼續或中止。",
            MENU_SYSTEM: "這裡管理 ComfyUI、溫度、顯存模式和自動關機。",
            MENU_HISTORY: "可以恢復失敗鏡頭、延續長片或管理故事排隊。",
        }
        menu = (
            f"{prefix}🎬 MiniMax H3 Turbo 控制面板\n"
            f"目前頁面：{section_titles.get(section, '主選單')}\n\n"
            f"模式：{mode_text}\n"
            f"參數：{resolution_label(current.width, current.height)} | "
            f"{current.steps} steps | {duration_text}\n"
            f"素材：{media_status}\n"
            f"提示詞：{prompt_status}\n"
            f"任務：{job_text}\n"
            f"排隊：{queue_text}\n"
            f"顯存：{comfyui_vram_mode_label(self.comfyui_vram_mode())}\n"
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
            new_message_id = result.get("message_id") if isinstance(result, dict) else None
            if new_message_id:
                with self.progress_message_lock:
                    self.progress_message_id = int(new_message_id)
                    self.progress_message_chat_id = chat_id
                    self.progress_message_text = text
        except BotError as exc:
            self.send_safe(chat_id, f"顯示生成進度失敗：{exc}")
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
        self.telegram.send_message(
            chat_id,
            "請輸入總片長秒數（2 至 1800），例如 37、180、600 或 1800。",
            reply_markup={
                "force_reply": True,
                "input_field_placeholder": "例如 600",
            },
        )

    def request_prompt(self, chat_id: str) -> None:
        self.awaiting_duration = False
        self.awaiting_prompt = True
        self.awaiting_queue_prompt = False
        self.telegram.send_message(
            chat_id,
            "請下一則訊息貼上提示詞，可以是多行文字。完成後回到面板按「生成影片」。",
            reply_markup={
                "force_reply": True,
                "input_field_placeholder": "貼上影片提示詞",
            },
        )

    def comfy_status_text(self) -> str:
        if comfyui_is_online():
            return f"ComfyUI 正常運行中：{COMFY_URL}"
        if _comfy_process is not None and _comfy_process.poll() is None:
            return f"ComfyUI 程序已啟動，仍在載入：PID {_comfy_process.pid}"
        return f"ComfyUI 目前未運行：{COMFY_URL}"

    def ensure_comfyui_ready(self, job: JobState) -> None:
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
                self.awaiting_prompt = False
                self.awaiting_duration = False
                self.save_settings()
                self.show_menu(chat_id, notice=notice)
                return
            if self.input_mode == INPUT_MODE_REF2VA:
                if len(self.reference_image_paths) >= MAX_REF2VA_IMAGES:
                    self.send_safe(chat_id, f"Ref2VA 最多支援 {MAX_REF2VA_IMAGES} 張參考圖。")
                    return
                REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
                target_path = (
                    REFERENCE_DIR
                    / f"ref_image_{len(self.reference_image_paths) + 1:02d}{suffix}"
                )
                self.telegram.download_file(remote_path, target_path)
                self.reference_image_paths.append(target_path)
                if caption:
                    self.prompt = caption
                self.awaiting_prompt = False
                self.awaiting_duration = False
                self.save_settings()
                self.show_menu(
                    chat_id,
                    notice=(
                        f"Ref2VA 已收到第 {len(self.reference_image_paths)} 張參考圖；"
                        "可繼續上傳，完成後按「完成參考素材上傳」。"
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
        self.save_settings()
        self.show_menu(chat_id, notice=f"已讀取 {file_name}，提示詞已更新（{len(prompt)} 字）")

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
            self.awaiting_prompt = False
            self.save_settings()
            self.show_menu(chat_id, notice="提示詞已更新")
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
        self.save_settings()

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
            if data.startswith("upscale:"):
                self.handle_upscale_callback(chat_id, message_id, data)
                return
            if data == "noop":
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
            if data == "media_done":
                if self.input_mode == INPUT_MODE_REF2VA and not (
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
            if data == "bot_restart":
                self.restart_bot(chat_id)
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
    config = GenerationConfig(864, 480, 12, 15, valid_length(15))
    workflow = build_workflow(config, "A short bright test scene with clear motion and synchronized sound.")
    print("workflow=ok")
    print(f"template={T8_API_TEMPLATE}")
    print(f"comfy={COMFY_URL}")
    print(f"comfy_base={COMFYUI_BASE_DIR}")
    print(f"output={OUTPUT_DIR}")
    print(f"length={workflow['6']['inputs']['length']} frames ({config.actual_seconds:.2f}s)")
    sampler_inputs = workflow["7"]["inputs"]
    sampler_type = workflow["7"].get("class_type")
    if sampler_type == "MiniMaxH3MultiRateSamplerEXPT8":
        print(
            f"steps={sampler_inputs['video_steps']} video / "
            f"{sampler_inputs['audio_steps']} audio"
        )
    elif sampler_type == "MiniMaxH3TurboSampler":
        print(
            f"steps={workflow['13']['inputs']['steps']} "
            f"scheduler={workflow['13']['inputs']['scheduler']}"
        )
    else:
        print(f"steps={sampler_inputs['steps']}")


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
