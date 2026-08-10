"""Build API-format LTX 2.3 workflows from ComfyUI's subgraph templates.

The ComfyUI template files are frontend/UI workflows.  Telegram submits the
API prompt format, so this small converter expands the LTX 2.3 T2V and I2V
subgraphs and keeps only the native generation path used by the Bot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_DIR = Path(
    r"E:\Comfy\ComfyUI\ComfyUI\.venv\Lib\site-packages"
    r"\comfyui_workflow_templates_json\templates"
)

# Model names are deliberately kept in one place.  The Bot replaces these
# values only for LTX jobs; MiniMax H3 uses its own separate workflow.
LTX_MAIN = "PinkCherry_FineTune_Q5_K_M_v18_LTX23.gguf"
LTX_GEMMA = "gemma-3-12b-it-heretic-v2_int4.safetensors"
LTX_PROJECTION = "ltx-2.3_text_projection_bf16.safetensors"
LTX_VIDEO_VAE = "LTX23_video_vae_bf16.safetensors"
LTX_AUDIO_VAE = "LTX23_audio_vae_bf16.safetensors"
LTX_DISTILL_LORA = (
    "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"
)
LTX_UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

_UNSET = object()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _external_values(mode: str) -> dict[int, Any]:
    """Values for the subgraph's -10 input node, indexed by input slot."""

    if mode == "i2v":
        # first_frame, prompt, prompt_enhance, width, height, duration, fps,
        # seed, checkpoint, distilled_lora, text_encoder, upscaler, lora.
        return {
            0: _UNSET,
            1: "replace_with_prompt",
            2: False,
            3: 512,
            4: 288,
            5: 5,
            6: 24,
            7: 0,
            8: LTX_MAIN,
            9: LTX_DISTILL_LORA,
            10: LTX_GEMMA,
            11: LTX_UPSCALER,
            12: "",
        }
    # prompt, prompt_enhance, width, height, duration, fps, seed,
    # checkpoint, distilled_lora, text_encoder, upscaler, lora.
    return {
        0: "replace_with_prompt",
        1: False,
        2: 512,
        3: 288,
        4: 5,
        5: 24,
        6: 0,
        7: LTX_MAIN,
        8: LTX_DISTILL_LORA,
        9: LTX_GEMMA,
        10: LTX_UPSCALER,
        11: "",
    }


def _link_value(
    link: dict[str, Any],
    external_values: dict[int, Any],
) -> Any:
    origin_id = int(link.get("origin_id", 0))
    if origin_id == -10:
        return external_values.get(int(link.get("origin_slot", 0)), _UNSET)
    if origin_id < 0:
        return _UNSET
    return [str(origin_id), int(link.get("origin_slot", 0))]


def _set_common_hidden_inputs(node_type: str, widgets: list[Any], inputs: dict[str, Any]) -> None:
    """Expose advanced/hidden widgets which are omitted from UI input sockets."""

    if node_type == "RandomNoise" and widgets:
        inputs.setdefault("noise_seed", widgets[0])
    elif node_type == "KSamplerSelect" and widgets:
        inputs.setdefault("sampler_name", str(widgets[0]))
    elif node_type == "ManualSigmas" and widgets:
        inputs.setdefault("sigmas", str(widgets[0]))
    elif node_type == "CFGGuider" and widgets:
        inputs.setdefault("cfg", float(widgets[0]))
    elif node_type == "ComfyMathExpression" and widgets:
        inputs.setdefault("expression", str(widgets[0]))
    elif node_type == "PrimitiveBoolean" and widgets:
        inputs.setdefault("value", bool(widgets[0]))
    elif node_type == "ResizeImagesByLongerEdge" and widgets:
        inputs.setdefault("longer_edge", int(widgets[0]))
    elif node_type == "LoraLoaderModelOnly" and len(widgets) > 1:
        inputs.setdefault("strength_model", widgets[1])
    elif node_type == "LTXVImgToVideoInplace" and widgets:
        inputs.setdefault("strength", widgets[0])
    elif node_type == "EmptyLTXVLatentVideo":
        inputs.setdefault("batch_size", int(widgets[3] if len(widgets) > 3 else 1))
    elif node_type == "LTXVEmptyLatentAudio":
        inputs.setdefault("batch_size", int(widgets[2] if len(widgets) > 2 else 1))
    elif node_type == "EmptyImage":
        # These are frontend-only widgets in the native template, but the
        # API validator requires all four EmptyImage inputs explicitly.
        values = list(widgets) + [512, 512, 1, 0]
        inputs.setdefault("width", int(values[0]))
        inputs.setdefault("height", int(values[1]))
        inputs.setdefault("batch_size", int(values[2]))
        inputs.setdefault("color", int(values[3]))
    elif node_type == "LTXVPreprocess":
        # The native graph stores this compression setting as a widget only.
        inputs.setdefault("img_compression", int(widgets[0] if widgets else 18))
    elif node_type == "VAEDecodeTiled":
        values = list(widgets) + [768, 64, 4096, 4]
        inputs.setdefault("tile_size", int(values[0]))
        inputs.setdefault("overlap", int(values[1]))
        inputs.setdefault("temporal_size", int(values[2]))
        inputs.setdefault("temporal_overlap", int(values[3]))
    elif node_type == "LTXAVTextEncoderLoader":
        inputs.setdefault("device", "default")


def _convert_subgraph(template: dict[str, Any], mode: str) -> dict[str, Any]:
    definitions = template.get("definitions") or {}
    subgraphs = definitions.get("subgraphs") or []
    if not subgraphs:
        raise ValueError("LTX template has no subgraph")
    subgraph = subgraphs[0]
    nodes = subgraph.get("nodes") or []
    links = {int(link["id"]): link for link in subgraph.get("links") or []}
    external_values = _external_values(mode)
    prompt: dict[str, Any] = {}

    for node in nodes:
        node_id = int(node["id"])
        if node_id < 0:
            continue
        node_type = str(node["type"])
        inputs: dict[str, Any] = {}
        widgets = list(node.get("widgets_values") or [])
        widget_index = 0

        for socket in node.get("inputs") or []:
            name = str(socket.get("name", ""))
            link_id = socket.get("link")
            if link_id is not None:
                link = links.get(int(link_id))
                if link is not None:
                    value = _link_value(link, external_values)
                    if value is not _UNSET:
                        inputs[name] = value
                # A linked widget still consumes its visible widget value in
                # the frontend representation; advance for hidden alignment.
                if socket.get("widget") is not None and widget_index < len(widgets):
                    widget_index += 1
                continue
            if socket.get("widget") is not None and widget_index < len(widgets):
                inputs[name] = widgets[widget_index]
                widget_index += 1

        _set_common_hidden_inputs(node_type, widgets, inputs)
        prompt[str(node_id)] = {
            "class_type": node_type,
            "inputs": inputs,
        }

    # Swap the official full checkpoint/text loader for the Q5 GGUF model and
    # the Heretic Gemma + LTX projection pair.
    main_loader_id: Optional[str] = None
    video_vae_id: Optional[str] = None
    for node_id, node in prompt.items():
        class_type = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if class_type == "CheckpointLoaderSimple":
            node["class_type"] = "UnetLoaderGGUF"
            node["inputs"] = {"unet_name": LTX_MAIN}
            main_loader_id = str(node_id)
        elif class_type == "LTXAVTextEncoderLoader":
            node["class_type"] = "DualCLIPLoaderGGUF"
            node["inputs"] = {
                "clip_name1": LTX_GEMMA,
                "clip_name2": LTX_PROJECTION,
                "type": "ltxv",
            }
        elif class_type == "LTXVAudioVAELoader":
            node["class_type"] = "VAELoaderKJ"
            node["inputs"] = {
                "vae_name": LTX_AUDIO_VAE,
                "device": "cpu",
                "weight_dtype": "bf16",
            }
        elif class_type == "VAELoader":
            inputs["vae_name"] = LTX_VIDEO_VAE
        elif class_type == "LoraLoaderModelOnly":
            inputs["lora_name"] = LTX_DISTILL_LORA
            inputs["strength_model"] = 0.5
        elif class_type == "LatentUpscaleModelLoader":
            inputs["model_name"] = LTX_UPSCALER

    # The official checkpoint exposes its video VAE as output 2.  A GGUF
    # UNET loader only returns MODEL, so replace that Reroute with an explicit
    # video VAE loader and rewrite every old checkpoint-VAE reference.
    if main_loader_id is not None:
        for node_id, node in prompt.items():
            if node.get("class_type") != "Reroute":
                continue
            if node.get("inputs", {}).get("") == [main_loader_id, 2]:
                node["class_type"] = "VAELoader"
                node["inputs"] = {"vae_name": LTX_VIDEO_VAE}
                video_vae_id = str(node_id)
                break
    if video_vae_id is None:
        video_vae_id = "902"
        prompt[video_vae_id] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": LTX_VIDEO_VAE},
        }
    for node in prompt.values():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        if not isinstance(inputs, dict):
            continue
        for key, value in list(inputs.items()):
            if value == [main_loader_id, 2]:
                inputs[key] = [video_vae_id, 0]

    # Bypass the prompt-enhancer path and feed the actual Telegram prompt into
    # the positive encoder.  The negative encoder is kept deterministic.
    for node in prompt.values():
        if node.get("class_type") == "CLIPTextEncode":
            inputs = node.setdefault("inputs", {})
            if "clip" in inputs and "text" not in inputs:
                inputs["text"] = "blurry, low quality, still frame, frames, watermark, overlay, titles, subtitles"
    clip_nodes = [
        node for node in prompt.values() if node.get("class_type") == "CLIPTextEncode"
    ]
    if clip_nodes:
        # The first CLIPTextEncode is positive in both current templates.
        clip_nodes[0].setdefault("inputs", {})["text"] = "replace_with_prompt"

    # Dynamic resize input: the frontend stores this as a nested combo.
    for node in prompt.values():
        if node.get("class_type") == "ResizeImageMaskNode":
            node["inputs"] = {
                "input": node.get("inputs", {}).get("input"),
                # ComfyUI API v3 represents DynamicCombo values with a
                # selector plus dotted child input names, not a nested dict.
                "resize_type": "scale dimensions",
                "resize_type.width": 512,
                "resize_type.height": 288,
                "resize_type.crop": "center",
                "scale_method": "lanczos",
            }
        node["inputs"] = {
            key: value
            for key, value in node["inputs"].items()
            if value is not None
        }

    # Add an output node.  The subgraph exposes VIDEO through -20; locate its
    # producer instead of relying on hard-coded node IDs between templates.
    output_links = subgraph.get("outputs") or []
    output_link_ids = set()
    for output in output_links:
        output_link_ids.update(int(link_id) for link_id in output.get("linkIds") or [])
    producer_id = None
    producer_slot = 0
    for link_id in output_link_ids:
        link = links.get(link_id)
        if link is not None:
            producer_id = int(link["origin_id"])
            producer_slot = int(link.get("origin_slot", 0))
            break
    if producer_id is None:
        raise ValueError("Could not find LTX video output producer")
    prompt["900"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": [str(producer_id), producer_slot],
            "filename_prefix": "MiniMaxH3/LTX23/Telegram",
            "format": "auto",
            "codec": "auto",
        },
    }

    # Image-to-video gets an explicit LoadImage node.  T2V has no first-frame
    # external socket and therefore does not receive this node.
    if mode == "i2v":
        first_frame_link = None
        for link in links.values():
            if int(link.get("origin_id", 0)) == -10 and int(link.get("origin_slot", -1)) == 0:
                first_frame_link = link
                break
        if first_frame_link is None:
            raise ValueError("I2V template has no first-frame input")
        target_id = str(first_frame_link["target_id"])
        target_slot = int(first_frame_link["target_slot"])
        target = prompt[target_id]
        target_inputs = target.setdefault("inputs", {})
        target_input_name = str((nodes and next(
            socket["name"]
            for ui_node in nodes
            if int(ui_node["id"]) == int(target_id)
            for socket in ui_node.get("inputs") or []
            if socket.get("link") == first_frame_link["id"]
        )) or "input")
        target_inputs[target_input_name] = ["901", 0]
        prompt["901"] = {
            "class_type": "LoadImage",
            "inputs": {"image": "replace_with_image"},
        }

    # Keep only nodes needed by SaveVideo.  This drops the optional prompt
    # enhancer, its Gemma LoRA, preview nodes, and other disconnected UI-only
    # helpers without relying on template-specific numeric IDs.
    required: set[str] = set()
    pending = ["900"]
    while pending:
        node_id = pending.pop()
        if node_id in required or node_id not in prompt:
            continue
        required.add(node_id)
        inputs = prompt[node_id].get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if isinstance(value, list) and len(value) >= 1:
                source_id = str(value[0])
                if source_id in prompt:
                    pending.append(source_id)
    prompt = {node_id: node for node_id, node in prompt.items() if node_id in required}

    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode, name in (("t2v", "video_ltx2_3_t2v.json"), ("i2v", "video_ltx2_3_i2v.json")):
        template = _read_json(args.template_dir / name)
        workflow = _convert_subgraph(template, mode)
        output = args.output_dir / f"ltx23_{mode}_api.json"
        output.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {output} ({len(workflow)} nodes)")


if __name__ == "__main__":
    main()
