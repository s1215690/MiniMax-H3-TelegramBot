"""H3 VRAM Master runtime package.

Importing this package is intentionally side-effect free.  ComfyUI scans custom
nodes at startup, so merely installing H3VM must not change global ComfyUI,
comfy-kitchen, CLIPLoader, VAELoader, or ModelPatcher behavior.  Runtime patches
are installed lazily only when an H3VM execution path is actually used.
"""
from __future__ import annotations

from threading import RLock

_RUNTIME_LOCK = RLock()
_RUNTIME_ACTIVE = False
_RUNTIME_ACTIVATING = False


def runtime_active() -> bool:
    """Return whether the H3VM execution runtime has been activated."""
    return bool(_RUNTIME_ACTIVE)


def activate_runtime() -> bool:
    """Lazily install H3VM execution-only runtime patches.

    This function is idempotent and is called by H3VM execution nodes/builders,
    never during ordinary ComfyUI custom-node discovery.  Standard ComfyUI
    CLIPLoader and VAELoader are deliberately not patched here.
    """
    global _RUNTIME_ACTIVE, _RUNTIME_ACTIVATING
    if _RUNTIME_ACTIVE:
        return True

    with _RUNTIME_LOCK:
        if _RUNTIME_ACTIVE:
            return True
        if _RUNTIME_ACTIVATING:
            return False
        _RUNTIME_ACTIVATING = True
        try:
            # Install H3VM-private device/runtime overlays in the same validated
            # order as v0.21.1, but only after an H3VM path is actually used.
            from .gpu_preflight import install_gpu_preflight_patch
            install_gpu_preflight_patch()

            from .comfy_kitchen_multigpu import install_comfy_kitchen_multigpu_dlpack_guard
            install_comfy_kitchen_multigpu_dlpack_guard()

            from .segmented_relay import install_segmented_relay_patch
            install_segmented_relay_patch()

            from .attention_relay_packet import install_attention_relay_packet_patch
            install_attention_relay_packet_patch()

            from .vram_master_fusion import install_vram_master_fusion
            install_vram_master_fusion()

            from .capacity_quantized_qkv import install_capacity_quantized_qkv_patch
            install_capacity_quantized_qkv_patch()

            _RUNTIME_ACTIVE = True
            print(
                "[H3VM] runtime activated lazily | standard ComfyUI CLIP/VAE loaders untouched",
                flush=True,
            )
            return True
        finally:
            _RUNTIME_ACTIVATING = False
