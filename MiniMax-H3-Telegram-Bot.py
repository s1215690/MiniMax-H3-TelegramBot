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


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_APP_DATA = Path(
    os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
)
COMFY_ROOT = Path(
    os.environ.get("MINIMAX_COMFY_ROOT", str(Path.home() / "ComfyUI"))
)
COMFYUI_BASE_DIR = Path(
    os.environ.get("MINIMAX_COMFY_BASE_DIR", str(COMFY_ROOT / "ComfyUI"))
)
COMFYUI_DIR = Path(
    os.environ.get("MINIMAX_COMFY_DIR", str(COMFY_ROOT / "ComfyUI-Turbo"))
)
COMFYUI_PYTHON = Path(
    os.environ.get(
        "MINIMAX_COMFY_PYTHON",
        str(COMFYUI_BASE_DIR / ".venv" / "Scripts" / "python.exe"),
    )
)
COMFY_URL = os.environ.get("MINIMAX_COMFY_URL", "http://127.0.0.1:8191").rstrip("/")
OUTPUT_DIR = Path(
    os.environ.get("MINIMAX_COMFY_OUTPUT", str(LOCAL_APP_DATA / "ComfyUI" / "output"))
)
INPUT_DIR = Path(os.environ.get("MINIMAX_COMFY_INPUT", str(OUTPUT_DIR.parent / "input")))
T8_API_TEMPLATE = Path(
    os.environ.get(
        "MINIMAX_T8_API_TEMPLATE",
        str(PROJECT_DIR / "workflow" / "dual_clock_multirate_api.json"),
    )
)
SEEDVR2_API_TEMPLATE = Path(
    os.environ.get(
        "MINIMAX_SEEDVR2_API_TEMPLATE",
        str(PROJECT_DIR / "workflow" / "seedvr2_3b_int8_upscale_video_api.json"),
    )
)
COMFYUI_PORT = int(os.environ.get("MINIMAX_COMFY_PORT", "8191"))
COMFYUI_LOG = Path(
    os.environ.get(
        "MINIMAX_COMFY_LOG",
        str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "MiniMax-H3-Telegram" / "comfyui.log"),
    )
)
COMFYUI_STATE_DIR = Path(
    os.environ.get(
        "MINIMAX_COMFY_STATE_DIR",
        str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "MiniMax-H3-Turbo-ComfyUI"),
    )
)
COMFYUI_USER_DIR = COMFYUI_STATE_DIR / "user"
COMFYUI_DATABASE = COMFYUI_STATE_DIR / "comfyui.db"
DEFAULT_COMFYUI_VRAM_MODE = "lowvram"
FFMPEG_PATH = os.environ.get("MINIMAX_FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
NVIDIA_SMI_PATH = os.environ.get(
    "MINIMAX_NVIDIA_SMI",
    shutil.which("nvidia-smi")
    or r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)
SHUTDOWN_DELAY_SECONDS = 60
MAX_TELEGRAM_IMAGE_BYTES = 20 * 1024 * 1024
SAGE_ATTENTION_ENABLED = os.environ.get("MINIMAX_SAGE_ATTENTION", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

VIDEO_VAE = os.environ.get("MINIMAX_VIDEO_VAE", "minimax_h3_video_vae_fp16.safetensors")
AUDIO_VAE = os.environ.get("MINIMAX_AUDIO_VAE", "minimax_h3_audio_vae_fp32.safetensors")
CLIP_NAME = os.environ.get(
    "MINIMAX_CLIP", "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
)
UNET_NAME = os.environ.get("MINIMAX_UNET", "minimax_h3_fl2va_int8_convrot.safetensors")
LORA_NAME = os.environ.get(
    "MINIMAX_LORA", "minimax_h3_turbo_v4_step600_comfyui_T8-convert.safetensors"
)
OUTPUT_PREFIX = "MiniMaxH3/Telegram_Turbo"
STATE_PATH = Path(
    os.environ.get(
        "MINIMAX_TELEGRAM_STATE",
        str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "MiniMax-H3-Telegram" / "settings.json"),
    )
)
IMAGE_DIR = STATE_PATH.parent / "input_images"
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
LONG_CONTINUITY_MODE = os.environ.get(
    "MINIMAX_H3_LONG_CONTINUITY", "motion_context"
).strip().lower()
MOTION_CONTEXT_LENGTH = 22
MOTION_CONTEXT_EXTRA_SECONDS = MOTION_CONTEXT_LENGTH / 24.0


class BotError(RuntimeError):
    pass


def http_error_detail(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    return f"HTTP Error {exc.code}: {body[:1200]}" if body else str(exc)


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
    comfy_image_name: Optional[str] = None
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
    upscale_source_path: Optional[Path] = None
    upscale_target_width: int = 0
    upscale_target_height: int = 0


@dataclass(frozen=True)
class PendingUpscale:
    token: str
    chat_id: str
    source_path: Path
    source_width: int
    source_height: int
    duration_seconds: float
    shutdown_after_choice: bool = False


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
        segment_total = math.ceil(total_seconds / MAX_SEGMENT_SECONDS)
        missing = [
            number
            for number in range(1, segment_total + 1)
            if number not in segmented.segments
        ]
        if missing:
            missing_text = "、".join(f"SEGMENT {number}" for number in missing)
            raise BotError(
                f"這條影片需要 {segment_total} 個 SEGMENT，但缺少 {missing_text}。"
            )
        shots: list[ShotSpec] = []
        for number in range(1, segment_total + 1):
            start_seconds = (number - 1) * MAX_SEGMENT_SECONDS
            end_seconds = min(number * MAX_SEGMENT_SECONDS, total_seconds)
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
    return "\n".join(lines)


def build_workflow(
    config: GenerationConfig,
    prompt: str,
    output_prefix: str = OUTPUT_PREFIX,
    image_name: Optional[str] = None,
    audio_reference_name: Optional[str] = None,
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
    workflow["4"]["inputs"]["unet_name"] = UNET_NAME

    # INT8 ConvRot uses the bypass model-only loader for the Turbo LoRA.
    workflow["5"]["class_type"] = "LoraLoaderBypassModelOnly"
    workflow["5"]["inputs"]["lora_name"] = LORA_NAME
    workflow["5"]["inputs"]["strength_model"] = 1.0

    conditioning = workflow["6"]["inputs"]
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
        conditioning["task_type"] = "T2VA"
        conditioning["audio_mode"] = "native"
        conditioning["add_source_as_reference"] = False
        conditioning["prompt_primary_audio_ordinal"] = 0
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
    if image_name and not motion_context:
        workflow["13"] = {
            "inputs": {"image": image_name},
            "class_type": "LoadImage",
            "_meta": {"title": "Telegram input image"},
        }
        conditioning["first_frame"] = ["13", 0]
        if prompt_enhancer:
            workflow["30"]["inputs"]["image"] = ["13", 0]
    if audio_reference_name and not motion_context:
        workflow["14"] = {
            "inputs": {"audio": audio_reference_name},
            "class_type": "LoadAudio",
            "_meta": {"title": "Previous segment audio reference"},
        }
        conditioning["drive_audio"] = ["14", 0]

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
    if workflow["7"].get("class_type") == "MiniMaxH3MultiRateSamplerEXPT8":
        sampler_inputs["video_steps"] = min(4, config.steps)
        sampler_inputs["audio_steps"] = config.steps
    else:
        sampler_inputs["steps"] = config.steps
    sampler_inputs["shift_video"] = 12.0
    sampler_inputs["shift_audio"] = 3.0
    workflow["8"]["inputs"]["noise_seed"] = secrets.randbits(63)
    workflow["12"]["inputs"]["filename_prefix"] = output_prefix
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


def build_transition_filter(
    shots: list[ShotSpec] | tuple[ShotSpec, ...],
    transition_seconds: float = SHOT_TRANSITION_SECONDS,
) -> tuple[str, str, str]:
    """Build a duration-preserving FFmpeg xfade/acrossfade graph."""
    if len(shots) < 2:
        raise BotError("轉場至少需要兩個鏡頭。")
    filters: list[str] = []
    for index, shot in enumerate(shots):
        trim_duration = shot.duration
        if index < len(shots) - 1:
            trim_duration += transition_seconds
        filters.append(
            f"[{index}:v]trim=duration={trim_duration:.3f},"
            f"setpts=PTS-STARTPTS,fps=24,format=yuv420p[v{index}]"
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


def concat_videos(
    video_paths: list[Path],
    output_path: Path,
    total_seconds: float,
    shot_plan: Optional[tuple[ShotSpec, ...]] = None,
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
        filter_graph, video_output, audio_output = build_transition_filter(shot_plan)
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
) -> Path:
    """Merge all completed shots, including a one-shot early cancellation."""
    if not video_paths:
        raise BotError("沒有已完成的分段可以合成。")
    if len(video_paths) == 1:
        return trim_single_video(video_paths[0], output_path, total_seconds)
    return concat_videos(video_paths, output_path, total_seconds, shot_plan=shot_plan)


def upload_video_to_comfy(video_path: Path) -> str:
    """Upload a previous MP4 so LoadVideo can expose its frame batch."""
    return upload_audio_to_comfy(video_path)


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
                "menu_button": json.dumps({"type": "commands"}),
            },
            timeout=30,
        )

    def get_file(self, file_id: str) -> str:
        result = self.call("getFile", {"file_id": file_id}, timeout=30)
        file_path = result.get("file_path") if isinstance(result, dict) else None
        if not file_path:
            raise BotError("Telegram 沒有回傳圖片檔案路徑。")
        return str(file_path)

    def download_file(self, file_path: str, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(
                Request(
                    f"{self.file_base_url}/{file_path}",
                    headers={"Accept": "application/octet-stream"},
                ),
                timeout=120,
            ) as response:
                data = response.read(MAX_TELEGRAM_IMAGE_BYTES + 1)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise BotError(f"下載 Telegram 圖片失敗：{exc}") from exc
        if len(data) > MAX_TELEGRAM_IMAGE_BYTES:
            raise BotError("圖片太大，請控制在 20 MB 以內。")
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
        self.call("sendMessage", params, timeout=30)

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
        multipart_request(
            f"{self.base_url}/sendVideo",
            {"chat_id": chat_id, "caption": caption, "supports_streaming": "true"},
            "video",
            video_path,
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
            "/temperature 查看 GPU／CPU 溫度\n"
            "/cancel_shutdown 取消已排程的自動關機\n"
            "/comfy_restart 重啟 ComfyUI\n"
            "/comfy_stop 關閉 ComfyUI\n"
            "/cancel 取消目前生成\n"
            "/help 查看說明"
        )

    def send_safe(self, chat_id: str, text: str) -> None:
        try:
            self.telegram.send_message(chat_id, text)
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
                print(f"ComfyUI memory release before SeedVR2 was unavailable: {exc}", flush=True)
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
            print(f"seedvr2 upscale done: {output_path}", flush=True)
        except Exception as exc:
            if not job.cancel_event.is_set():
                self.send_safe(job.chat_id, f"SeedVR2 放大失敗：{exc}")
            print(f"seedvr2 upscale error: {exc}", flush=True)
            print(f"upscale error: {exc}", flush=True)
        finally:
            with self.lock:
                if self.job is job:
                    self.job = None

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
    ) -> None:
        prompt = prompt.strip()
        if not prompt:
            self.send_safe(chat_id, "提示詞不可為空白。")
            return
        with self.lock:
            if self.job:
                self.send_safe(chat_id, "目前已有工作在生成，請先等待完成或使用 /cancel。")
                return
            job = JobState(
                chat_id,
                config,
                prompt,
                time.time(),
                cancel_event=threading.Event(),
                input_image_path=input_image_path,
            )
            job.resume_event.set()
            self.job = job
        thread = threading.Thread(target=self.run_job, args=(job,), daemon=True)
        thread.start()

    def ensure_comfyui_ready(self, job: JobState) -> None:
        """Hook for a subclass to start or wait for ComfyUI before queuing."""
        return

    def comfyui_vram_mode(self) -> str:
        return DEFAULT_COMFYUI_VRAM_MODE

    def cancel_job_for_comfy_control(self) -> bool:
        """Mark the active Telegram job cancelled before stopping ComfyUI."""
        with self.lock:
            job = self.job
            if job is not None:
                job.cancel_event.set()
                job.resume_event.set()
        return job is not None

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
        if not motion_context and job.input_image_path is not None and job.segment_index == 1:
            if job.comfy_image_name is None:
                job.comfy_image_name = upload_image_to_comfy(job.input_image_path)
            image_name = job.comfy_image_name
        elif (
            not motion_context
            and job.segment_index > 1
            and job.continuation_image_path is not None
        ):
            image_name = upload_image_to_comfy(job.continuation_image_path)
        workflow = build_workflow(
            job.config,
            segment_prompt(job),
            job.output_prefix,
            image_name=image_name,
            audio_reference_name=(None if motion_context else job.audio_reference_name),
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
            self.send_safe(
                job.chat_id,
                f"已開始生成：{job.config.width}×{job.config.height} | "
                f"{job.config.steps} steps | 約 {job.config.actual_seconds:.2f} 秒\n"
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
        try:
            self.ensure_comfyui_ready(job)
            video_path = self.run_segment(job, announce=True)
            with job.progress_lock:
                job.progress_phase = "uploading"
            caption = (
                f"MiniMax H3 Turbo 完成\n{job.config.width}×{job.config.height} | "
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
        except Exception as exc:  # keep the long-polling bot alive after one job fails
            if not job.cancel_event.is_set():
                self.send_safe(job.chat_id, f"生成失败：{exc}")
            print(f"generation error: {exc}", flush=True)
        finally:
            with self.lock:
                if self.job is job:
                    self.job = None

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
            for key in ("gifs", "videos", "files"):
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

    RESOLUTIONS = ((608, 352), (736, 416), (864, 480), (960, 544))
    SECONDS = (5, 10, 12, 15)
    LONG_SECONDS = (30, 60, 120, 180, 300, 600, 900, 1200, 1800)
    STEPS = (4, 8, 12)

    def __init__(self, token: str, allowed_chat_id: str):
        super().__init__(token, allowed_chat_id)
        self.settings = self.load_settings()
        self.total_seconds = self.load_saved_total_seconds()
        self.prompt = self.load_saved_prompt()
        self.input_mode = self.load_saved_mode()
        self.image_path = self.load_saved_image_path()
        self.vram_mode = self.load_saved_vram_mode()
        self.shutdown_after_generation = self.load_saved_shutdown_after_generation()
        self._shutdown_pending = False
        self.awaiting_prompt = False
        self.awaiting_duration = False
        self.menu_message_id: Optional[int] = None

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
            mode = str(saved.get("input_mode", "text"))
            return mode if mode in {"text", "image"} else "text"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return "text"

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
            try:
                path.resolve().relative_to(IMAGE_DIR.resolve())
            except ValueError:
                return None
            return path
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

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
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(STATE_PATH)

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

    def menu_markup(self) -> dict[str, Any]:
        current = self.settings
        mode_row = [
            {
                "text": self.selected("📝 文字生视频", self.input_mode == "text"),
                "callback_data": "mode:text",
            },
            {
                "text": self.selected("🖼 图片生视频", self.input_mode == "image"),
                "callback_data": "mode:image",
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
        custom_seconds_row = [
            {"text": "✏️ 自定義秒數", "callback_data": "sec_custom"}
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
        if self._shutdown_pending:
            shutdown_row = [
                {"text": "🛑 取消即將關機", "callback_data": "shutdown_cancel"}
            ]
        elif self.total_seconds > MAX_SEGMENT_SECONDS:
            shutdown_row = [
                {
                    "text": self.selected(
                        "🔌 長片完成後關機", self.shutdown_after_generation
                    ),
                    "callback_data": "shutdown_toggle",
                }
            ]
        else:
            shutdown_row = [
                {
                    "text": "🔌 完成後關機（只限長片）",
                    "callback_data": "shutdown_toggle",
                }
            ]
        return {
            "inline_keyboard": [
                [{"text": "🎬 生成模式（选择一种）", "callback_data": "noop"}],
                mode_row,
                [{"text": "⏱ 總片長（短片）", "callback_data": "noop"}],
                short_seconds_row,
                [{"text": "🎞 總片長（長片，會自動分段）", "callback_data": "noop"}],
                long_seconds_row[:4],
                long_seconds_row[4:8],
                long_seconds_row[8:],
                custom_seconds_row,
                [{"text": "🖼 解析度／MP（按下選擇）", "callback_data": "noop"}],
                resolution_row,
                [{"text": "⚙️ 步數（按下選擇）", "callback_data": "noop"}],
                steps_row,
                [
                    {"text": "✍️ 輸入／更換提示詞", "callback_data": "prompt"},
                    {"text": "🧹 清除提示詞", "callback_data": "clear"},
                    {"text": "🗑 清除图片", "callback_data": "clear_image"},
                ],
                [
                    {"text": "🚀 生成影片", "callback_data": "generate"},
                    {"text": "♻️ 讀取上次設定", "callback_data": "last"},
                ],
                job_control_row,
                shutdown_row,
                [{"text": "🌡 查看電腦溫度", "callback_data": "temperature"}],
                [
                    {"text": "▶️ 啟動 ComfyUI", "callback_data": "comfy_start"},
                    {"text": "📡 ComfyUI 狀態", "callback_data": "comfy_status"},
                ],
                [
                    {"text": "🔄 重啟 ComfyUI", "callback_data": "comfy_restart"},
                    {"text": "⏹ 關閉 ComfyUI", "callback_data": "comfy_stop"},
                ],
                [{"text": "📊 查看／刷新生成進度", "callback_data": "progress"}],
            ]
        }

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
            self.show_menu(chat_id, message_id, "目前沒有生成中的任務")
            return
        if job.segment_total <= 1:
            self.send_safe(chat_id, "單段影片沒有暫停狀態。")
        elif was_paused:
            self.send_safe(chat_id, "已播放／繼續；下一段會繼續生成。")
        else:
            self.send_safe(chat_id, "目前任務沒有暫停，會繼續生成。")
        self.show_menu(chat_id, message_id)

    def start_selected_generation(self, chat_id: str, prompt: str) -> None:
        config = self.effective_config()
        input_image_path = (
            self.image_path
            if self.input_mode == "image" and self.image_path and self.image_path.is_file()
            else None
        )
        if self.total_seconds > MAX_SEGMENT_SECONDS:
            self.start_long_generation(
                chat_id,
                config,
                prompt,
                self.total_seconds,
                input_image_path=input_image_path,
            )
        else:
            self.start_generation(chat_id, config, prompt, input_image_path=input_image_path)

    def start_long_generation(
        self,
        chat_id: str,
        config: GenerationConfig,
        prompt: str,
        total_seconds: float,
        input_image_path: Optional[Path] = None,
    ) -> None:
        prompt = prompt.strip()
        if not prompt:
            self.send_safe(chat_id, "提示詞不可為空白。")
            return
        total_seconds = validate_total_seconds(total_seconds)
        try:
            plan = build_long_video_plan(prompt, total_seconds)
        except BotError as exc:
            self.send_safe(chat_id, f"長片時間軸格式錯誤：{exc}")
            return
        segment_total = len(plan.shots)
        if segment_total < 2:
            self.send_safe(chat_id, "長片時間軸至少需要兩個鏡頭。")
            return
        batch_prefix = f"{OUTPUT_PREFIX}/long_{uuid.uuid4().hex[:12]}"
        with self.lock:
            if self.job:
                self.send_safe(chat_id, "目前已有工作在生成，請先等待完成或使用 /cancel。")
                return
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
            )
            job.resume_event.set()
            self.job = job
        format_text = "自然時間軸" if plan.source_format == "timeline" else "SEGMENT 分段"
        self.send_safe(
            chat_id,
            f"已解析{format_text}：共 {segment_total} 個連續鏡頭，"
            f"每鏡頭最多 {MAX_SHOT_SECONDS:g} 秒。\n"
            "後續鏡頭會使用上一鏡尾幀和第一鏡音訊風格，合併時加入短音畫轉場。",
        )
        thread = threading.Thread(target=self.run_long_job, args=(job,), daemon=True)
        thread.start()

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
            merge_completed_segments(
                video_paths,
                output_path,
                completed_seconds,
                shot_plan=completed_shots or None,
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
            print(f"partial long job sent: {output_path}", flush=True)
            return output_path
        except Exception as exc:
            print(f"partial long job merge error: {exc}", flush=True)
            self.send_safe(job.chat_id, f"已中止，但部分影片合成失敗：{exc}")
            return None

    def run_long_job(self, job: JobState) -> None:
        partial_reported = False
        video_paths: list[Path] = []
        base_prefix = job.output_prefix

        def report_partial() -> None:
            nonlocal partial_reported
            if partial_reported:
                return
            partial_reported = True
            self.send_partial_long_result(job, video_paths, base_prefix)

        try:
            self.ensure_comfyui_ready(job)
            base_config = job.config
            motion_context_enabled = (
                LONG_CONTINUITY_MODE in {"motion_context", "motion", "experimental"}
                and motion_context_nodes_available()
            )
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
                        "Motion Context 節點未就緒，這次先使用穩定的尾幀＋音訊參考模式。",
                    )
            context_video_name: Optional[str] = None
            context_latent_path: Optional[str] = None
            latent_prefix = (
                f"{base_prefix}/motion_context/latent"
                if motion_context_enabled
                else None
            )
            if job.input_image_path is not None:
                self.send_safe(
                    job.chat_id,
                    "圖片長片會把圖片用作第一鏡首幀，後續鏡頭使用上一鏡尾幀接續。",
                )
            for index in range(1, job.segment_total + 1):
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
                use_motion_context = motion_context_enabled and index > 1
                if use_motion_context:
                    # Motion Context pins a 22-frame head which is trimmed from
                    # the decoded result. Generate that head plus the requested
                    # shot duration so the delivered shot keeps its timeline.
                    generation_seconds += MOTION_CONTEXT_EXTRA_SECONDS
                job.config = parse_config(
                    [
                        str(base_config.width),
                        str(base_config.height),
                        str(base_config.steps),
                        str(generation_seconds),
                    ]
                )
                job.output_prefix = f"{base_prefix}/segment_{index:02d}"
                self.send_safe(
                    job.chat_id,
                    f"長片鏡頭 {index}/{job.segment_total} 開始生成："
                    f"劇情 {shot.start_seconds:g}-{shot.end_seconds:g} 秒 | "
                    f"模型約 {job.config.actual_seconds:.2f} 秒。",
                )
                video_path = self.run_segment(
                    job,
                    announce=False,
                    motion_context=use_motion_context,
                    context_video_name=context_video_name,
                    context_latent_path=context_latent_path,
                    save_latent_prefix=latent_prefix,
                    save_latent_clip_index=index if latent_prefix else None,
                )
                video_paths.append(video_path)
                if index < job.segment_total:
                    if motion_context_enabled:
                        # The next graph reads the previous MP4 and loads this
                        # segment's paired AV latent.
                        context_video_name = upload_video_to_comfy(video_path)
                        context_latent_path = (
                            f"{latent_prefix}_{index:05d}.safetensors"
                        )
                    else:
                        # Keep the immediate previous segment as the stable
                        # reference instead of always reusing segment 1.
                        job.audio_reference_name = upload_audio_to_comfy(video_path)
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
                self.send_safe(job.chat_id, f"長片鏡頭 {index}/{job.segment_total} 完成。")

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
            )
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
        except Exception as exc:
            if job.cancel_event.is_set():
                report_partial()
            else:
                self.send_safe(job.chat_id, f"長片生成失敗：{exc}")
            print(f"long generation error: {exc}", flush=True)
        finally:
            if job.continuation_image_path:
                try:
                    job.continuation_image_path.unlink()
                except OSError:
                    pass
            with self.lock:
                if self.job is job:
                    self.job = None

    def menu_text(self, notice: str = "") -> str:
        current = self.settings
        prompt_status = f"已輸入（{len(self.prompt)} 字）" if self.prompt else "尚未輸入"
        mode_text = "圖片生視頻" if self.input_mode == "image" else "文字生視頻"
        image_status = "已收到" if self.image_path and self.image_path.is_file() else "未收到"
        prefix = f"{notice}\n\n" if notice else ""
        if self.total_seconds > MAX_SEGMENT_SECONDS:
            duration_text = (
                f"總片長：約 {self.total_seconds:.0f} 秒"
                f"（按提示詞時間軸拆成最多 {MAX_SHOT_SECONDS:g} 秒鏡頭）"
            )
        else:
            effective = self.effective_config()
            duration_text = (
                f"總片長：約 {effective.actual_seconds:.2f} 秒（{effective.length} frames）"
            )
        if self._shutdown_pending:
            shutdown_text = "完成後關機：倒數中（可取消）"
        elif self.shutdown_after_generation:
            shutdown_text = "完成後關機：已開啟（只對長片生效）"
        else:
            shutdown_text = "完成後關機：關閉"
        with self.lock:
            active_job = self.job
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
        menu = (
            f"{prefix}MiniMax H3 Turbo 控制面板\n\n"
            f"模式：{mode_text}\n"
            f"輸入圖片：{image_status}\n"
            f"解析度：{resolution_label(current.width, current.height)}\n"
            f"步數：{current.steps}\n"
                f"{duration_text}\n"
                f"ComfyUI 顯存模式：{comfyui_vram_mode_label(self.comfyui_vram_mode())}\n"
            f"{shutdown_text}\n"
            f"{job_text}\n"
            f"提示詞：{prompt_status}\n\n"
            "圖片模式：先發圖片，再輸入提示詞；文字模式：直接輸入提示詞。\n"
            "最後按「生成影片」。\n"
            "長片會解析時間軸、短鏡頭接力生成後加入轉場合併；設定和提示詞會自動保存。"
        )
        return menu

    def finalize_upscale_choice(
        self, chat_id: str, pending: PendingUpscale
    ) -> None:
        if not pending.shutdown_after_choice or not self.shutdown_after_generation:
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
    ) -> None:
        text = self.menu_text(notice)
        markup = self.menu_markup()
        target_message_id = None if force_new else (message_id or self.menu_message_id)
        try:
            if target_message_id is None:
                result = self.telegram.send_message(chat_id, text, reply_markup=markup)
                if isinstance(result, dict) and result.get("message_id"):
                    self.menu_message_id = int(result["message_id"])
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
                    return
                except BotError:
                    pass
            self.send_safe(chat_id, f"選單更新失敗：{exc}")

    def request_duration(self, chat_id: str) -> None:
        self.awaiting_duration = True
        self.awaiting_prompt = False
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
            return
        self.send_safe(job.chat_id, start_comfyui_process(self.comfyui_vram_mode()))
        deadline = time.time() + 180
        while time.time() < deadline:
            if job.cancel_event.is_set():
                return
            if comfyui_is_online():
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

    def handle_image_message(self, message: dict[str, Any], chat_id: str) -> None:
        file_id = self.image_file_id(message)
        if not file_id:
            return
        try:
            remote_path = self.telegram.get_file(file_id)
            suffix = Path(remote_path).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                suffix = ".jpg"
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
            caption = str(message.get("caption", "")).strip()
            if caption:
                self.prompt = caption
            self.save_settings()
            if caption:
                self.show_menu(chat_id, notice="图片和提示詞已收到")
            else:
                self.show_menu(chat_id, notice="图片已收到；现在输入提示詞")
        except BotError as exc:
            self.send_safe(chat_id, f"处理图片失败：{exc}")

    def handle_message(self, message: dict[str, Any]) -> None:
        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != self.allowed_chat_id:
            return
        if self.image_file_id(message):
            self.handle_image_message(message, chat_id)
            return
        text = str(message.get("text", "")).strip()
        if not text:
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
            if data == "progress":
                self.show_progress(chat_id, message_id)
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
            if data == "mode:text":
                self.input_mode = "text"
                self.save_settings()
                self.show_menu(chat_id, message_id, "已切换到文字生视频")
                return
            if data == "mode:image":
                self.input_mode = "image"
                self.save_settings()
                self.show_menu(chat_id, message_id, "请发送一张图片")
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
                self.image_path = self.load_saved_image_path()
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
                if self.image_path and self.image_path.is_file():
                    try:
                        self.image_path.unlink()
                    except OSError:
                        pass
                self.image_path = None
                self.save_settings()
                self.show_menu(chat_id, message_id, "输入图片已清除")
                return
            if data.startswith("vram:"):
                mode = normalize_comfyui_vram_mode(data.removeprefix("vram:"))
                self.vram_mode = mode
                self.save_settings()
                cancelled = self.cancel_job_for_comfy_control()
                result = restart_comfyui_process(mode)
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
                self.send_safe(chat_id, start_comfyui_process(self.comfyui_vram_mode()))
                return
            if data == "comfy_status":
                self.send_safe(chat_id, self.comfy_status_text())
                return
            if data == "comfy_restart":
                cancelled = self.cancel_job_for_comfy_control()
                result = restart_comfyui_process(self.comfyui_vram_mode())
                prefix = "目前生成已取消。\n" if cancelled else ""
                self.send_safe(chat_id, prefix + result)
                return
            if data == "comfy_stop":
                cancelled = self.cancel_job_for_comfy_control()
                result = stop_comfyui_process()
                prefix = "目前生成已取消。\n" if cancelled else ""
                self.send_safe(chat_id, prefix + result)
                return
            if data == "generate":
                if self.awaiting_duration:
                    self.send_safe(chat_id, "請先輸入自定義總片長秒數，或使用 /cancel 取消。")
                elif self.awaiting_prompt:
                    self.send_safe(chat_id, "請先貼上提示詞，或使用 /cancel。")
                elif self.input_mode == "image" and not self.image_path:
                    self.send_safe(chat_id, "圖片模式還沒有圖片；請先發送一張圖片。")
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
            self.show_menu(chat_id, force_new=True)
            return
        if command == "/prompt":
            self.request_prompt(chat_id)
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
                self.send_safe(chat_id, start_comfyui_process(self.comfyui_vram_mode()))
            except BotError as exc:
                self.send_safe(chat_id, str(exc))
            return
        if command == "/comfy_restart":
            try:
                cancelled = self.cancel_job_for_comfy_control()
                result = restart_comfyui_process(self.comfyui_vram_mode())
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
        if command == "/progress":
            self.send_safe(chat_id, self.progress_text())
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
            if job:
                self.send_safe(chat_id, self.progress_text())
            elif self.awaiting_prompt:
                self.send_safe(chat_id, "等待你貼上提示詞。")
            else:
                self.show_menu(chat_id)
            return
        if command == "/cancel":
            self.awaiting_prompt = False
            self.awaiting_duration = False
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
            {"command": "prompt", "description": "輸入提示詞"},
            {"command": "image", "description": "切換圖生視頻"},
            {"command": "text", "description": "切換文生視頻"},
            {"command": "duration", "description": "設定秒數"},
            {"command": "status", "description": "查看目前狀態"},
            {"command": "cancel", "description": "中止目前生成"},
            {"command": "pause", "description": "暫停長片"},
            {"command": "resume", "description": "繼續長片"},
            {"command": "temperature", "description": "查看電腦溫度"},
            {"command": "comfy_status", "description": "查看 ComfyUI 狀態"},
            {"command": "comfy_start", "description": "啟動 ComfyUI"},
            {"command": "comfy_restart", "description": "重啟 ComfyUI"},
            {"command": "comfy_stop", "description": "關閉 ComfyUI"},
            {"command": "bot_restart", "description": "重啟 Telegram Bot"},
            {"command": "help", "description": "查看說明"},
        ]
        try:
            self.telegram.set_my_commands(commands)
            self.telegram.set_chat_menu_button(self.allowed_chat_id)
        except BotError as exc:
            print(f"Telegram menu setup failed: {exc}", flush=True)

    def run(self) -> None:
        self.configure_telegram_menu()
        self.show_menu(self.allowed_chat_id, notice="Turbo Telegram 控制器已啟動")
        while True:
            try:
                updates = self.telegram.get_updates(self.offset)
                for update in updates:
                    self.offset = int(update["update_id"]) + 1
                    if update.get("callback_query"):
                        self.handle_callback(update["callback_query"])
                    elif update.get("message"):
                        self.handle_message(update["message"])
            except KeyboardInterrupt:
                raise
            except Exception as exc:
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
    if workflow["7"].get("class_type") == "MiniMaxH3MultiRateSamplerEXPT8":
        print(
            f"steps={sampler_inputs['video_steps']} video / "
            f"{sampler_inputs['audio_steps']} audio"
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
