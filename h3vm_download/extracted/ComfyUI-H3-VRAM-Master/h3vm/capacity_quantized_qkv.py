from __future__ import annotations

"""Quantized-QKV fast path for H3VM Capacity.

The existing Capacity implementation is the correctness fallback. This overlay
only replaces QKV *projection slicing*: when the source weight is a supported
QuantizedTensor it slices quantized rows/scales first and executes the smaller
projection directly. Any incompatibility falls back to the proven dense
materialize/slice path for that call.
"""

import logging
import os
import time

LOG = logging.getLogger("H3VM")
_PATCH_MARKER = "_h3vm_capacity_quantized_qkv_v1"


def _looks_quantized(weight) -> bool:
    return type(weight).__name__ == "QuantizedTensor" and hasattr(weight, "_qdata")


def _active_bypass_hook(linear):
    """Return Comfy's live bypass hook when ``forward`` was instance-injected."""
    forward = getattr(linear, "__dict__", {}).get("forward")
    hook = getattr(forward, "__self__", None)
    if (
        hook is not None
        and type(hook).__name__ == "BypassForwardHook"
        and getattr(hook, "module", None) is linear
        and getattr(hook, "original_forward", None) is not None
    ):
        return hook
    return None


def _apply_simple_lora_slice(linear, shard, x, row_index):
    """Run a row-sharded base projection plus an exact simple-LoRA delta.

    H3 Turbo uses a ``LoRAAdapter`` subclass with ordinary 2-D up/down matrices.
    Other bypass adapter geometries deliberately raise so the caller can use the
    correctness-first full-output fallback.
    """
    import torch
    import torch.nn.functional as F

    base = shard(x)
    hook = _active_bypass_hook(linear)
    if hook is None:
        return base

    adapter = hook.adapter
    if not any(cls.__name__ == "LoRAAdapter" for cls in type(adapter).__mro__):
        raise RuntimeError(f"unsupported QKV bypass adapter {type(adapter).__name__}")
    weights = getattr(adapter, "weights", ())
    if len(weights) < 6:
        raise RuntimeError("unsupported QKV LoRA weight tuple")
    up, down, alpha, mid, dora_scale, reshape = weights[:6]
    if (
        not isinstance(up, torch.Tensor)
        or not isinstance(down, torch.Tensor)
        or up.ndim != 2
        or down.ndim != 2
        or mid is not None
        or dora_scale is not None
        or reshape is not None
        or bool(getattr(adapter, "is_conv", False))
    ):
        raise RuntimeError("unsupported complex QKV LoRA geometry")

    selected_up = up.index_select(0, row_index.to(up.device)).to(device=x.device, dtype=x.dtype)
    down = down.to(device=x.device, dtype=x.dtype)
    rank = int(down.shape[0])
    scale = (float(alpha) / rank if alpha is not None else 1.0) * float(
        getattr(adapter, "multiplier", 1.0)
    )
    delta = F.linear(F.linear(x, down), selected_up)
    return base.add_(delta, alpha=scale)


def project_qkv_slice(fabric, linear, x, heads: int, head_dim: int,
                      start_head: int, end_head: int):
    """Return (projected_qkv, path_name) for one contiguous head range."""
    weight = getattr(linear, "weight", None)
    if _looks_quantized(weight):
        try:
            if "forward" in getattr(linear, "__dict__", {}) and _active_bypass_hook(linear) is None:
                raise RuntimeError("unsupported instance QKV forward; preserving full output")
            from .quant_shard import qkv_head_row_indices, shard_qkv_heads
            row_index = qkv_head_row_indices(
                int(heads), int(head_dim), int(start_head), int(end_head),
                device=weight.device,
            )
            shard = shard_qkv_heads(
                linear, int(heads), int(head_dim), int(start_head), int(end_head)
            )
            try:
                projected = _apply_simple_lora_slice(linear, shard, x, row_index)
                expected_out = 3 * (int(end_head) - int(start_head)) * int(head_dim)
                if int(projected.shape[-1]) != expected_out:
                    raise RuntimeError(
                        "quantized QKV shard output mismatch "
                        f"{tuple(projected.shape)} expected_last={expected_out}"
                    )
                return projected, "quantized_native"
            finally:
                del shard
        except Exception as exc:
            reasons = getattr(fabric, "_qkv_fallback_reasons", None)
            if reasons is None:
                reasons = fabric._qkv_fallback_reasons = {}
            reason = type(exc).__name__ + ": " + str(exc)
            reasons[reason] = reasons.get(reason, 0) + 1
            # Fail soft per call. Capacity is an exact/capacity backend and must
            # prefer known-correct execution over a brittle optimization.
            if not getattr(fabric, "_qkv_native_warned", False):
                fabric._qkv_native_warned = True
                LOG.warning(
                    "H3VM CAPACITY quantized QKV native slice fallback | %s", exc
                )

    if not _looks_quantized(weight):
        reasons = getattr(fabric, "_qkv_fallback_reasons", None)
        if reasons is None:
            reasons = fabric._qkv_fallback_reasons = {}
        reasons["weight_not_quantized"] = reasons.get("weight_not_quantized", 0) + 1
    bypass = _active_bypass_hook(linear)
    if bypass is not None or "forward" in getattr(linear, "__dict__", {}):
        from .quant_shard import qkv_head_row_indices

        full = linear(x)
        index = qkv_head_row_indices(
            int(heads), int(head_dim), int(start_head), int(end_head),
            device=full.device,
        )
        return full.index_select(-1, index), "bypass_output_fallback"

    import torch.nn.functional as F
    import comfy.ops

    rw, rb, roff = fabric._dense_linear_weight(linear, x)
    try:
        sw, sb = fabric._qkv_weight_slice(
            rw, rb, int(heads), int(head_dim), int(start_head), int(end_head)
        )
        out = F.linear(x, sw, sb)
        del sw, sb
        return out, "dense_fallback"
    finally:
        comfy.ops.uncast_bias_weight(linear, rw, rb, roff)


def _execute_capacity_attention_vram_master(self, index: int, attn, h, rope_freqs,
                                            transformer_options):
    import torch
    import comfy.model_management as mm

    index = int(index)
    owner = self.primary
    helper = self.secondary
    if h.device != owner:
        raise RuntimeError(f"H3VM Capacity attention requires root hidden on {owner}, got {h.device}")
    helper_attn = self.helper_attn_by_block.get(index)
    if helper_attn is None:
        raise RuntimeError(f"H3VM Capacity missing helper attention packet for block {index}")

    self._sample_free()
    caller_stream = torch.cuda.current_stream(owner)
    root_stream = self.compute_streams[owner]
    helper_stream = self.compute_streams[helper]
    root_stream.wait_stream(caller_stream)
    (r0, r1), (h0i, h1i), counts = self._head_ranges(attn)

    re0 = torch.cuda.Event(enable_timing=True)
    re1 = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(owner), torch.cuda.stream(root_stream), mm.cuda_device_context(owner):
        re0.record(root_stream)
        rqkv, root_path = project_qkv_slice(
            self, attn.qkv_proj, h, attn.heads, attn.head_dim, r0, r1
        )
        root_heads = self._qkv_norm_rope_attention(
            attn, rqkv, rope_freqs, transformer_options, r1-r0
        )
        del rqkv
        re1.record(root_stream)

    t_stage = time.perf_counter()
    remote_h, remote_ready, _ = self._stage_to_helper(h.contiguous(), owner, helper, caller_stream)
    helper_rope = self._ensure_helper_rope(rope_freqs, owner, helper, caller_stream)
    self._attn_stage_ms += (time.perf_counter() - t_stage) * 1000.0

    he0 = torch.cuda.Event(enable_timing=True)
    he1 = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(helper), torch.cuda.stream(helper_stream), mm.cuda_device_context(helper):
        helper_stream.wait_event(remote_ready)
        he0.record(helper_stream)
        hqkv, helper_path = project_qkv_slice(
            self, helper_attn.qkv_proj, remote_h,
            helper_attn.heads, helper_attn.head_dim, h0i, h1i
        )
        helper_heads = self._qkv_norm_rope_attention(
            helper_attn, hqkv, helper_rope, transformer_options, h1i-h0i
        )
        del hqkv, remote_h
        he1.record(helper_stream)

    t_ret = time.perf_counter()
    returned, returned_ready, _ = self._return_to_owner(helper_heads, owner, helper, he1)
    self._attn_return_ms += (time.perf_counter() - t_ret) * 1000.0

    caller_stream.wait_event(re1)
    caller_stream.wait_event(returned_ready)
    out = self._chunked_outproj(attn, root_heads, returned, h.shape[0])
    out.record_stream(caller_stream)
    root_heads.record_stream(caller_stream)
    returned.record_stream(caller_stream)

    try:
        self._attn_root_ms += float(re0.elapsed_time(re1))
        self._attn_helper_ms += float(he0.elapsed_time(he1))
    except Exception:
        pass
    self._attn_calls += 1
    self._qkv_native_calls = int(getattr(self, "_qkv_native_calls", 0)) + int(root_path == "quantized_native") + int(helper_path == "quantized_native")
    self._qkv_dense_fallback_calls = int(getattr(self, "_qkv_dense_fallback_calls", 0)) + int(root_path != "quantized_native") + int(helper_path != "quantized_native")

    if self._attn_calls <= 2 or self._attn_calls % 10 == 0:
        LOG.info(
            "H3VM CAPACITY ATTN #%d block=%d heads=%s seq=%d | QKV=%s/%s native=%d fallback=%d | "
            "stage_acc=%.1fms return_acc=%.1fms root_acc=%.1fms helper_acc=%.1fms | outproj_chunk=%d",
            self._attn_calls, index, counts, int(h.shape[0]), root_path, helper_path,
            self._qkv_native_calls, self._qkv_dense_fallback_calls,
            self._attn_stage_ms, self._attn_return_ms,
            self._attn_root_ms, self._attn_helper_ms,
            self.outproj_chunk_rows,
        )
    self._sample_free()
    return out


def install_capacity_quantized_qkv_patch() -> bool:
    """Install the lab fast path unless explicitly disabled."""
    if os.environ.get("H3VM_DISABLE_CAPACITY_QUANT_SHARD", "0") == "1":
        return False

    from .capacity_mode import CapacityFabric
    current = CapacityFabric.execute_capacity_attention
    if getattr(current, _PATCH_MARKER, False):
        return True

    CapacityFabric._h3vm_original_execute_capacity_attention = current
    setattr(_execute_capacity_attention_vram_master, _PATCH_MARKER, True)
    CapacityFabric.execute_capacity_attention = _execute_capacity_attention_vram_master
    LOG.info(
        "H3VM VRAM Master Capacity quantized-QKV patch installed | native slice + dense fail-safe"
    )
    return True
