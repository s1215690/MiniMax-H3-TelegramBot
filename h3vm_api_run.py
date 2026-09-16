"""Build an API-format prompt for the H3VM dual-GPU test.

ComfyUI's HTTP /prompt endpoint wants API format, which is normally produced by
the browser frontend. The browser here is unreliable, so this converts the
director's UI-format workflow directly:

  * widget inputs are identified from /object_info (a widget is INT/FLOAT/STRING/
    BOOLEAN, or a COMBO whose options are a list). Everything else is a link.
  * the director node keeps TWO stores of its widget values (positional
    widgets_values and widgets_values_named) and ComfyUI applies the positional
    array by index during configure, which shifts once the seed's
    control_after_generate companion exists. The named map is therefore treated
    as authoritative, exactly as the plugin's own restore path does.
  * H3VM Core is spliced between the UNET loader and the director.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

COMFY = "http://127.0.0.1:8191"
WORKFLOW = Path(
    r"E:\Comfy\ComfyUI\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director"
    r"\example_workflows\minimax_h3_director_t2v.json"
)
CLIP_NAME = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"

# 双卡极速 (DUAL_SYNC_ACCEL) takes core_adapter's deepclone_multigpu path, which
# access-violates on this machine. 双卡扩容 (DUAL_CAPACITY) takes the other
# branch (core_adapter.py:308) and never calls deepclone_multigpu.
import os

GPU_MODE = os.environ.get("H3VM_GPU_MODE", "双卡扩容")

WIDGET_SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN"}

# Marks the positional slot consumed by seed's control_after_generate widget.
CONTROL_SENTINEL = "\x00control_after_generate"


def object_info() -> dict:
    with urllib.request.urlopen(f"{COMFY}/object_info", timeout=120) as resp:
        return json.loads(resp.read().decode())


def is_widget(spec: list) -> bool:
    """True when an /object_info input spec is a widget rather than a link."""
    if not spec:
        return False
    kind = spec[0]
    if isinstance(kind, str):
        return kind in WIDGET_SCALARS
    if isinstance(kind, list):
        # COMBO when the options are plain values; a list-of-node-class-names
        # (multi-type link) has a nested list as its first element.
        return not (kind and isinstance(kind[0], list))
    return False


def widget_names(info: dict, class_type: str) -> list[str]:
    node = info.get(class_type)
    if not node:
        return []
    names = []
    for section in ("required", "optional"):
        for name, spec in (node.get("input", {}).get(section) or {}).items():
            if is_widget(spec):
                names.append(name)
    return names


def widget_layout(node: dict) -> list[str]:
    """Ordered widget names from the node's own serialized inputs.

    A widget slot is the one that carries a `widget` key; link slots do not, so
    this needs no guessing from /object_info (which does not even list custom
    widget types such as BDGROUP, and mis-aligns the positional array).

    The frontend additionally stores a control_after_generate companion right
    after an INT seed widget. It occupies a positional widgets_values slot with
    no matching input entry, which is exactly what shifted `width`, `cfg` and
    `shift_audio` by one in the first attempt. A sentinel keeps them aligned.
    """
    layout: list[str] = []
    for slot in node.get("inputs") or []:
        if "widget" not in slot:
            continue
        layout.append(slot["name"])
        if slot["name"] == "seed":
            layout.append(CONTROL_SENTINEL)
    return layout


def build() -> dict:
    info = object_info()
    wf = json.loads(WORKFLOW.read_text(encoding="utf-8-sig"))

    # link_id -> (origin_node_id, origin_slot)
    links = {}
    for link in wf.get("links", []):
        # [id, origin_id, origin_slot, target_id, target_slot, type]
        links[link[0]] = (str(link[1]), link[2])

    prompt: dict[str, dict] = {}
    director_id = None

    for node in wf["nodes"]:
        class_type = node.get("type")
        if class_type not in info:
            continue  # frontend-only node (MarkdownNote etc.)
        node_id = str(node["id"])
        inputs: dict = {}

        # Linked inputs first, so a widget name can never shadow a real link.
        for slot in node.get("inputs") or []:
            link_id = slot.get("link")
            name = slot.get("name")
            if link_id is None or name is None:
                continue
            if link_id in links:
                inputs[name] = list(links[link_id])

        # Widgets: prefer the named map (authoritative when present), otherwise
        # walk the node's own widget layout against the positional array.
        named = node.get("widgets_values_named") or {}
        values = node.get("widgets_values") or []
        for index, name in enumerate(widget_layout(node)):
            if name is CONTROL_SENTINEL or name in inputs:
                continue
            if name in named:
                inputs[name] = named[name]
            elif index < len(values):
                inputs[name] = values[index]

        prompt[node_id] = {"class_type": class_type, "inputs": inputs}
        if class_type == "MiniMaxH3Director":
            director_id = node_id

    # The user's CLIP, not the one the example names.
    for node in prompt.values():
        if node["class_type"] == "CLIPLoader":
            node["inputs"]["clip_name"] = CLIP_NAME

    # Splice H3VM Core between the UNET loader and the director.
    unet_id = next(k for k, v in prompt.items() if v["class_type"] == "UNETLoader")
    prompt[unet_id]["inputs"]["unet_name"] = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    prompt["900"] = {
        "class_type": "H3VMCoreEngine",
        "inputs": {
            "ui_language": "中文",
            "model": [unet_id, 0],
            "multi_gpu_enabled": True,
            "dual_vae_enabled": True,
            "gpu_mode": GPU_MODE,
            "mode4_predictor": "SPECTRAL｜频谱极速",
            "gpu_participation": "全｜100%",
            "custom_participation": 100,
            "expected_steps_hint": 20,
            "telemetry": True,
        },
    }
    prompt[director_id]["inputs"]["model"] = ["900", 0]
    prompt[director_id]["inputs"]["steps"] = 20

    return prompt


if __name__ == "__main__":
    import sys

    p = build()
    d = next(k for k, v in p.items() if v["class_type"] == "MiniMaxH3Director")
    c = next((k for k, v in p.items() if v["class_type"] == "H3VMCoreEngine"), None)
    print(f"nodes={len(p)} director={d} core={c}")
    print(f"director steps={p[d]['inputs'].get('steps')} model={p[d]['inputs'].get('model')}")
    print(f"core={p[c]['inputs'] if c else None}")
    if "--submit" in sys.argv:
        req = urllib.request.Request(
            f"{COMFY}/prompt",
            data=json.dumps({"prompt": p, "client_id": "dsh-h3vm-api"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                body = json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            print("HTTP", exc.code)
            print(exc.read().decode()[:4000])
            raise SystemExit(1)
        print("SUBMITTED", body.get("prompt_id"), body.get("node_errors"))
