from __future__ import annotations

"""Loader integration for the H3 VRAM Master Exact-SP Preview backend.

The backend is user-selectable from the standalone Master Loader.  Runtime
capability differences are reported as telemetry instead of artificial launch
parameter gates.
"""

import gc
import logging
import sys


LOG = logging.getLogger("H3VM")
GIB = 1024 ** 3


def _resolve_exact_sp_transport(requested, pair, comfy_args):
    """Resolve a conservative physical transport for the Exact-SP exchange fabric.

    ``auto`` means: native D2D when bidirectional peer access really exists;
    Windows no-P2P stays on the locally validated pageable host relay; other
    host-relay platforms may use H3VM's bounded pinned runway only when ComfyUI
    itself has disabled global pinned memory.
    """
    value = str(requested or "auto").strip().lower()
    if value != "auto":
        return value
    if pair.bidirectional_p2p:
        return "direct_d2d"
    if sys.platform == "win32":
        return "neutral_pageable"
    if bool(getattr(comfy_args, "disable_pinned_memory", False)):
        return "neutral_pinned"
    return "neutral_pageable"


def build_h3_exact_sp_lab(*, unet_name: str,
                          primary_device: str = "gpu:0",
                          secondary_device: str = "gpu:1",
                          expected_steps: int = 4,
                          primary_fraction=None,
                          primary_runtime_reserve_gb: float = 5.0,
                          secondary_runtime_reserve_gb: float = 5.0,
                          primary_hot_cache_gb: float = 2.0,
                          secondary_hot_cache_gb: float = 2.0,
                          trim_interval: int = 1,
                          one_ahead_prefetch: bool = False,
                          transport_mode: str = "auto",
                          safe_profile: bool = True,
                          telemetry: bool = True):
    """Build the 50-block Exact-SP Preview runtime.

    The main H3 graph retains only fixed/front/back modules.  Every transformer
    block becomes a zero-weight proxy into paired DynamicVRAM packet islands.
    Both ranks carry full token-local weights, sharing one immutable CPU backing;
    only QKV is materialized as complementary head-row shards.
    """
    import torch
    import comfy.model_management as mm
    import comfy.patcher_extension
    from comfy.cli_args import args
    from comfy.ldm.modules.attention import optimized_attention
    from .compute_planner import build_runtime_plan, probe_cuda_pair
    from .exact_sp_block import ExactSPBlockRuntime, build_exact_sp_block_pair
    from .exact_sp_exchange import ExactSPExchangeFabric
    from .exact_sp_streaming import ExactSPBlockProxyFactory, ExactSPStreamingRuntime
    from .global_memory import GlobalMemorySpace
    from .loader_base import _common_preflight, _h3vm_island_patcher_class, _load_private_h3
    from .snapshot_islands import make_island_root

    primary, secondary = _common_preflight(primary_device, secondary_device)
    compiler_disabled = bool(getattr(args, "disable_comfy_compiler", False))
    pinned_disabled = bool(getattr(args, "disable_pinned_memory", False))
    attention_name = getattr(optimized_attention, "__name__", type(optimized_attention).__name__)
    if not compiler_disabled:
        LOG.warning("H3VM Exact-SP running with Comfy compiler/AIMDO enabled; dynamic two-device packets remain runtime-managed")
    if bool(safe_profile) and not pinned_disabled:
        LOG.warning("H3VM Exact-SP safe profile requested while global pinned memory is enabled; using resolved pageable transport")
    LOG.info("H3VM Exact-SP attention backend=%s (user/ComfyUI selection preserved)", attention_name)

    pair = probe_cuda_pair(primary, secondary)
    plan = build_runtime_plan(
        pair,
        backend="EXACT_SP",
        manual_ratio=primary_fraction,
        total_blocks=50,
        total_heads=56,
    )
    patcher, dm = _load_private_h3(str(unet_name), primary, bool(safe_profile))
    original_blocks = list(dm.blocks)
    if len(original_blocks) != 50:
        raise RuntimeError(
            f"Exact-SP Preview is validated only for the 50-block H3 graph; got "
            f"{len(original_blocks)} blocks"
        )
    heads = int(original_blocks[0].attn.heads)
    if heads != 56:
        raise RuntimeError(
            f"Exact-SP Preview is validated only for the 56-head H3 graph; got {heads} heads"
        )
    if any(int(block.attn.heads) != heads for block in original_blocks):
        raise RuntimeError("Exact-SP Preview requires uniform H3 attention head counts")
    head_split = int(plan.attention_primary_heads)

    primary_packets = {}
    secondary_packets = {}
    for index, block in enumerate(original_blocks):
        primary_packet, secondary_packet = build_exact_sp_block_pair(
            block, primary_heads=head_split
        )
        primary_packet.h3vm_index = int(index)
        secondary_packet.h3vm_index = int(index)
        primary_packets[int(index)] = primary_packet
        secondary_packets[int(index)] = secondary_packet
        if index == 0 or (index + 1) % 10 == 0:
            LOG.info("H3VM Exact-SP packetized blocks=%d/50", index + 1)

    resolved_transport = _resolve_exact_sp_transport(transport_mode, pair, args)
    allow_pinned = resolved_transport == "neutral_pinned"
    space = GlobalMemorySpace(
        primary,
        secondary,
        transport_mode=resolved_transport,
        benchmark_mb=16,
        benchmark_repeats=1,
        allow_explicit_pinned=allow_pinned,
        pinned_ring_mb=32,
        pinned_ring_slots=2,
        host_only=resolved_transport.startswith("neutral_"),
    )
    exchange = ExactSPExchangeFabric(
        primary, secondary, transport_engine=space.transport
    )
    block_runtime = ExactSPBlockRuntime(primary, secondary, exchange=exchange)
    runtime = ExactSPStreamingRuntime(
        primary,
        secondary,
        primary_packets,
        secondary_packets,
        block_runtime=block_runtime,
        space=space,
        expected_steps=max(1, int(expected_steps)),
        primary_fraction=float(plan.primary_fraction),
        primary_runtime_reserve_gb=float(primary_runtime_reserve_gb),
        secondary_runtime_reserve_gb=float(secondary_runtime_reserve_gb),
        primary_hot_cache_gb=float(primary_hot_cache_gb),
        secondary_hot_cache_gb=float(secondary_hot_cache_gb),
        trim_interval=max(1, int(trim_interval)),
        one_ahead_prefetch=bool(one_ahead_prefetch),
        telemetry=bool(telemetry),
    )

    primary_root = make_island_root(
        patcher.model, dm, primary_packets, len(original_blocks),
        "exact-sp-primary-packets",
    )
    secondary_root = make_island_root(
        patcher.model, dm, secondary_packets, len(original_blocks),
        "exact-sp-secondary-packets",
    )
    for index in range(len(original_blocks)):
        dm.blocks[index] = ExactSPBlockProxyFactory.make(runtime, index)

    Patcher = _h3vm_island_patcher_class(patcher)
    primary_patcher = Patcher(
        primary_root,
        load_device=primary,
        offload_device=torch.device("cpu"),
        size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )
    secondary_patcher = Patcher(
        secondary_root,
        load_device=secondary,
        offload_device=torch.device("cpu"),
        size=0,
        weight_inplace_update=getattr(patcher, "weight_inplace_update", False),
    )
    for item in (patcher, primary_patcher, secondary_patcher):
        item.size = 0
        item.cached_patcher_init = None
    runtime.bind_patchers(primary_patcher, secondary_patcher)

    # Dropping the source block shells releases their full QKV tensors.  The two
    # complementary packet shards together retain exactly one QKV row set, while
    # token-local packet Parameters share the original CPU storage.
    del original_blocks
    gc.collect()

    fixed_bytes = int(mm.module_size(patcher.model))
    primary_logical = int(mm.module_size(primary_root))
    secondary_logical = int(mm.module_size(secondary_root))
    patcher.set_attachments("h3vm_exact_sp_runtime", runtime)
    patcher.set_attachments("h3vm_exact_sp_primary", primary_patcher)
    patcher.set_attachments("h3vm_exact_sp_secondary", secondary_patcher)
    patcher.set_attachments("h3vm_global_space", space)
    patcher.set_attachments("h3vm_exact_sp_config", {
        "architecture": "EXACT_SP_2RANK_TOKEN_HEAD_PARALLEL",
        "public": True,
        "stability": "preview",
        "exact": True,
        "blocks": 50,
        "primary_fraction": float(plan.primary_fraction),
        "head_counts": [head_split, heads - head_split],
        "transport": str(space.transport.selected_mode),
        "transport_requested": str(transport_mode),
        "planner": plan.summary(),
        "primary_runtime_reserve_gb": float(primary_runtime_reserve_gb),
        "secondary_runtime_reserve_gb": float(secondary_runtime_reserve_gb),
        "primary_hot_cache_gb": float(primary_hot_cache_gb),
        "secondary_hot_cache_gb": float(secondary_hot_cache_gb),
        "trim_interval": max(1, int(trim_interval)),
        "one_ahead_prefetch": bool(one_ahead_prefetch),
        "attention_kernel": str(attention_name),
        "comfy_compiler": "disabled" if compiler_disabled else "enabled",
    })

    def sample_wrapper(executor, *wrapper_args, **wrapper_kwargs):
        runtime.sampling_begin()
        try:
            return executor(*wrapper_args, **wrapper_kwargs)
        finally:
            runtime.sampling_end()

    patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        "h3vm_exact_sp_preview_lifecycle",
        sample_wrapper,
    )
    LOG.info(
        "H3VM EXACT-SP PREVIEW READY | blocks=50 heads=%d/%d compute=%.0f/%.0f "
        "transport=%s | fixed=%.2fGiB packet_logical=%.2f+%.2fGiB",
        head_split,
        heads - head_split,
        plan.primary_fraction * 100.0,
        plan.secondary_fraction * 100.0,
        space.transport.selected_mode,
        fixed_bytes / GIB,
        primary_logical / GIB,
        secondary_logical / GIB,
    )
    print(
        f"[H3VM EXACT-SP PREVIEW READY] 50 blocks | heads={head_split}/{heads-head_split} | "
        f"compute={plan.primary_fraction*100:.0f}/{plan.secondary_fraction*100:.0f} | "
        f"transport={space.transport.selected_mode} | attention={attention_name}",
        flush=True,
    )
    return patcher


# Product-facing alias; keep the historical function name for compatibility.
build_h3_exact_sp_preview = build_h3_exact_sp_lab
