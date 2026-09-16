from __future__ import annotations

"""Correctness-first two-GPU execution for one MiniMax H3 DiT block.

This module is the model-facing layer above :mod:`exact_sp_exchange`.  It keeps
the block state sequence-sharded between calls, gathers only normalized hidden
rows for head-local attention, redistributes attention heads back to token
ownership, and then runs OutProj/MLP locally on each token shard.

The Master Loader exposes this backend as Exact-SP Preview. Real-model single-block,
50-block parity, and a short end-to-end lifecycle smoke pass; public Core/prebuilt
MODEL integration remains intentionally gated until packetization and LoRA replay
receive their own parity coverage.
"""

import copy
from dataclasses import dataclass

import torch

from .exact_sp_exchange import ExactSPExchangeFabric
from .mlp_token_parallel import _clone_linear_shared
from .post_attention_island import _clone_norm_shared
from .quant_shard import shard_qkv_heads
from .sequence_partition import SequencePartition


def build_exact_sp_sequence_partition(total: int, primary_fraction: float = 0.5):
    """Match planner ownership, including primary ownership of an odd remainder."""
    if abs(float(primary_fraction) - 0.5) <= 1e-12:
        return SequencePartition.balanced(int(total), parts=2)
    return SequencePartition.dual_ratio(int(total), float(primary_fraction))


def _clone_adaln_shared(src):
    result = copy.copy(src)
    result._parameters = dict(getattr(src, "_parameters", {}))
    result._buffers = dict(getattr(src, "_buffers", {}))
    result._modules = dict(getattr(src, "_modules", {}))
    result.linear = _clone_linear_shared(src.linear)
    return result


def _has_instance_forward(module) -> bool:
    return "forward" in getattr(module, "__dict__", {})


class ExactSPAttentionShard(torch.nn.Module):
    def __init__(self, src_attn, start_head: int, end_head: int):
        super().__init__()
        self.total_heads = int(src_attn.heads)
        self.head_dim = int(src_attn.head_dim)
        self.head_start = int(start_head)
        self.head_end = int(end_head)
        self.heads = self.head_end - self.head_start
        self.qkv_proj = shard_qkv_heads(
            src_attn.qkv_proj,
            self.total_heads,
            self.head_dim,
            self.head_start,
            self.head_end,
        )
        self.q_norm = _clone_norm_shared(src_attn.q_norm)
        self.k_norm = _clone_norm_shared(src_attn.k_norm)
        self.out_proj = _clone_linear_shared(src_attn.out_proj)

    def forward(self, *args, **kwargs):
        raise RuntimeError("Exact-SP attention shards are driven by ExactSPBlockRuntime")


class ExactSPMLP(torch.nn.Module):
    def __init__(self, src_mlp):
        super().__init__()
        self.fc1 = _clone_linear_shared(src_mlp.fc1)
        self.fc2 = _clone_linear_shared(src_mlp.fc2)

    def forward(self, x):
        # Production ComfyUI uses the fused/compatible helper.  Keep a tiny
        # mathematically equivalent fallback so the Exact-SP block contract can
        # be tested outside a full ComfyUI checkout instead of silently dropping
        # the most important new backend from CI.
        try:
            import comfy.ops
        except ModuleNotFoundError:
            import torch.nn.functional as F
            a, b = self.fc1(x).chunk(2, dim=-1)
            return self.fc2(F.silu(a) * b)
        return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")


class ExactSPBlockPacket(torch.nn.Module):
    def __init__(self, block, start_head: int, end_head: int):
        super().__init__()
        self.norm1 = _clone_norm_shared(block.norm1)
        self.norm2 = _clone_norm_shared(block.norm2)
        self.attn = ExactSPAttentionShard(block.attn, start_head, end_head)
        self.mlp = ExactSPMLP(block.mlp)
        self.adaln_proj = _clone_adaln_shared(block.adaln_proj)

    def forward(self, *args, **kwargs):
        raise RuntimeError("Exact-SP block packets are driven by ExactSPBlockRuntime")


def _clone_block_packet(block, start_head: int, end_head: int):
    if _has_instance_forward(block.attn.qkv_proj):
        raise RuntimeError(
            "Exact-SP Preview does not yet accept an instance-patched QKV "
            "projection (for example Turbo/LoRA); use the base model parity gate"
        )
    return ExactSPBlockPacket(block, start_head, end_head)


def build_exact_sp_block_pair(block, primary_heads: int):
    """Create two CPU-backed compute packets with disjoint QKV head rows.

    Norm, OutProj, MLP and AdaLN Parameters are separate shells sharing the
    immutable CPU storage of ``block``.  QKV is the only materialized shard:
    rank 0 stores ``[0, primary_heads)`` and rank 1 stores the remaining rows.
    """
    heads = int(block.attn.heads)
    split = int(primary_heads)
    if not 0 < split < heads:
        raise ValueError(f"primary_heads must be within 1..{heads - 1}")
    return (
        _clone_block_packet(block, 0, split),
        _clone_block_packet(block, split, heads),
    )


@dataclass
class ExactSPBlockState:
    primary: object
    secondary: object
    partition: SequencePartition


@dataclass
class ExactSPBlockStats:
    block_calls: int = 0
    ingress_calls: int = 0
    egress_calls: int = 0

    def as_dict(self):
        return {
            "block_calls": int(self.block_calls),
            "ingress_calls": int(self.ingress_calls),
            "egress_calls": int(self.egress_calls),
        }


def _move_segment_payload(segment, device):
    import torch

    return tuple(
        value.to(device) if torch.is_tensor(value) and value.device != device else value
        for value in segment
    )


def _mod_value(values, row, reference):
    import torch

    if torch.is_tensor(row) and row.device != values.device:
        row = row.to(values.device)
    return values[row].to(device=reference.device, dtype=reference.dtype)


def _mod_scale_shift_local(hidden, shift, scale, segments):
    for start, stop, row in segments:
        view = hidden[int(start):int(stop)]
        view.mul_(1.0 + _mod_value(scale, row, view)).add_(
            _mod_value(shift, row, view)
        )
    return hidden


def _mod_gate_local(state, gate, update, segments):
    for start, stop, row in segments:
        target = state[int(start):int(stop)]
        target.addcmul_(
            update[int(start):int(stop)],
            _mod_value(gate, row, target),
        )
    return state


class ExactSPBlockRuntime:
    """Execute H3 blocks while preserving two-rank sequence ownership."""

    def __init__(self, primary, secondary, *, exchange: ExactSPExchangeFabric,
                 trace_stages: bool = False):
        import torch

        self.primary = torch.device(primary)
        self.secondary = torch.device(secondary)
        self.exchange = exchange
        if self.exchange.primary != self.primary or self.exchange.secondary != self.secondary:
            raise ValueError("Exact-SP exchange devices do not match block runtime devices")
        self.stats = ExactSPBlockStats()
        self.trace_stages = bool(trace_stages)
        self.last_trace = {}
        self._secondary_rope = None
        self._secondary_rope_signature = None

    def _trace_pair(self, name, primary, secondary):
        if not self.trace_stages:
            return
        import torch

        self.last_trace[str(name)] = torch.cat(
            (primary.detach().float().cpu(), secondary.detach().float().cpu()), dim=0
        )

    @staticmethod
    def _device_context(device):
        import contextlib
        import torch

        if torch.device(device).type != "cuda":
            return contextlib.nullcontext()
        import comfy.model_management as mm
        return mm.cuda_device_context(device)

    @staticmethod
    def _validate_packets(primary_packet, secondary_packet):
        p = primary_packet.attn
        s = secondary_packet.attn
        if int(p.total_heads) != int(s.total_heads) or int(p.head_dim) != int(s.head_dim):
            raise ValueError("Exact-SP attention packet geometry mismatch")
        if int(p.head_start) != 0 or int(p.head_end) != int(s.head_start):
            raise ValueError("Exact-SP head shards must be contiguous and primary-first")
        if int(s.head_end) != int(p.total_heads):
            raise ValueError("Exact-SP head shards do not cover every attention head")

    @staticmethod
    def _validate_state(state: ExactSPBlockState, primary, secondary):
        if state.partition.parts != 2:
            raise ValueError("Exact-SP block runtime requires a two-way partition")
        if state.primary.device != primary or state.secondary.device != secondary:
            raise ValueError("Exact-SP state tensors are not on their owning devices")
        if int(state.primary.shape[0]) != state.partition.size(0):
            raise ValueError("primary state row count does not match partition")
        if int(state.secondary.shape[0]) != state.partition.size(1):
            raise ValueError("secondary state row count does not match partition")

    def _rope_for_secondary(self, rope_freqs):
        if rope_freqs is None:
            return None
        signature = (
            int(rope_freqs.data_ptr()), tuple(rope_freqs.shape),
            str(rope_freqs.dtype), str(rope_freqs.device),
        )
        if self._secondary_rope is None or signature != self._secondary_rope_signature:
            self._secondary_rope = self.exchange.move_tensor(
                rope_freqs.contiguous(), self.secondary
            )
            self._secondary_rope_signature = signature
        return self._secondary_rope

    @staticmethod
    def _attention_heads(attn, hidden, rope_freqs, transformer_options):
        import torch

        seq = int(hidden.shape[0])
        heads = int(attn.heads)
        head_dim = int(attn.head_dim)
        inner = heads * head_dim
        qkv = attn.qkv_proj(hidden)
        if int(qkv.shape[-1]) != 3 * inner:
            raise RuntimeError(
                f"Exact-SP QKV shard output={qkv.shape[-1]} expected={3 * inner}"
            )
        q, k, v = qkv.split(inner, dim=-1)
        v = v.view(seq, heads, head_dim)
        if rope_freqs is None:
            q = attn.q_norm(q.view(seq, heads, head_dim))
            k = attn.k_norm(k.view(seq, heads, head_dim))
        else:
            import comfy.model_management as mm
            import comfy.quant_ops

            q = q.view(1, seq, heads, head_dim)
            k = k.view(1, seq, heads, head_dim)
            qw = mm.cast_to(attn.q_norm.weight, device=hidden.device)
            kw = mm.cast_to(attn.k_norm.weight, device=hidden.device)
            rot = int(rope_freqs.shape[-3]) * 2
            if mm.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw,
                    epsilon=attn.q_norm.eps, rot_dim=rot,
                )
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw,
                    epsilon=attn.q_norm.eps, rot_dim=rot,
                )
            q, k = q[0], k[0]

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        if hidden.device.type == "cuda":
            from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

            return optimized_attention(
                AttentionTensorContainer(q),
                AttentionTensorContainer(k),
                AttentionTensorContainer(v),
                heads, mask=None, skip_reshape=True,
                skip_output_reshape=True,
                transformer_options=transformer_options,
            )
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    @staticmethod
    def _outproj(attn, heads):
        rows = int(heads.shape[-2])
        flat = heads.squeeze(0).transpose(0, 1).reshape(rows, -1)
        return attn.out_proj(flat)

    def split_input(self, full_state, partition: SequencePartition) -> ExactSPBlockState:
        if full_state.device != self.primary:
            raise ValueError(f"Exact-SP full input must start on {self.primary}")
        if int(full_state.shape[0]) != int(partition.total):
            raise ValueError("Exact-SP full input row count does not match partition")
        cut = partition.size(0)
        primary = full_state[:cut].contiguous()
        secondary = self.exchange.move_tensor(
            full_state[cut:].contiguous(), self.secondary
        )
        self.stats.ingress_calls += 1
        return ExactSPBlockState(primary, secondary, partition)

    def gather_output(self, state: ExactSPBlockState):
        import torch

        self._validate_state(state, self.primary, self.secondary)
        secondary = self.exchange.move_tensor(state.secondary.contiguous(), self.primary)
        self.stats.egress_calls += 1
        return torch.cat((state.primary, secondary), dim=0)

    def execute_sharded(self, primary_packet, secondary_packet,
                        state: ExactSPBlockState, t_emb, mod_segments,
                        rope_freqs, transformer_options=None) -> ExactSPBlockState:
        import torch

        self._validate_packets(primary_packet, secondary_packet)
        self._validate_state(state, self.primary, self.secondary)
        if t_emb.device != self.primary:
            raise ValueError(f"Exact-SP t_emb must be on {self.primary}")
        opts = {} if transformer_options is None else transformer_options
        p_segments = state.partition.localize_segments(mod_segments, 0)
        s_segments = [
            _move_segment_payload(seg, self.secondary)
            for seg in state.partition.localize_segments(mod_segments, 1)
        ]

        with self._device_context(self.primary):
            mods = primary_packet.adaln_proj(t_emb)
        mod_packet = torch.stack(tuple(mods), dim=0).contiguous()
        secondary_mods = self.exchange.move_tensor(mod_packet, self.secondary)
        p_mods = tuple(mods)
        s_mods = tuple(secondary_mods.unbind(0))

        with self._device_context(self.primary):
            primary_hidden = primary_packet.norm1(state.primary)
            _mod_scale_shift_local(primary_hidden, p_mods[0], p_mods[1], p_segments)
        with self._device_context(self.secondary):
            secondary_hidden = secondary_packet.norm1(state.secondary)
            _mod_scale_shift_local(secondary_hidden, s_mods[0], s_mods[1], s_segments)
        self._trace_pair("attention_input", primary_hidden, secondary_hidden)

        full_primary, full_secondary = self.exchange.gather_hidden(
            primary_hidden, secondary_hidden, state.partition, seq_axis=0
        )
        secondary_rope = self._rope_for_secondary(rope_freqs)
        with self._device_context(self.primary):
            primary_heads = self._attention_heads(
                primary_packet.attn, full_primary, rope_freqs, opts
            )
        with self._device_context(self.secondary):
            secondary_heads = self._attention_heads(
                secondary_packet.attn, full_secondary, secondary_rope, opts
            )

        primary_heads, secondary_heads = self.exchange.redistribute_heads(
            primary_heads, secondary_heads, state.partition,
            seq_axis=-2, head_axis=1,
        )
        with self._device_context(self.primary):
            primary_attn = self._outproj(primary_packet.attn, primary_heads)
        with self._device_context(self.secondary):
            secondary_attn = self._outproj(secondary_packet.attn, secondary_heads)
        self._trace_pair("attention_output", primary_attn, secondary_attn)
        with self._device_context(self.primary):
            _mod_gate_local(state.primary, p_mods[2], primary_attn, p_segments)
        with self._device_context(self.secondary):
            _mod_gate_local(state.secondary, s_mods[2], secondary_attn, s_segments)
        self._trace_pair("post_attention", state.primary, state.secondary)
        with self._device_context(self.primary):
            primary_hidden = primary_packet.norm2(state.primary)
            _mod_scale_shift_local(primary_hidden, p_mods[3], p_mods[4], p_segments)
        with self._device_context(self.secondary):
            secondary_hidden = secondary_packet.norm2(state.secondary)
            _mod_scale_shift_local(secondary_hidden, s_mods[3], s_mods[4], s_segments)
        self._trace_pair("mlp_input", primary_hidden, secondary_hidden)
        with self._device_context(self.primary):
            primary_mlp = primary_packet.mlp(primary_hidden)
        with self._device_context(self.secondary):
            secondary_mlp = secondary_packet.mlp(secondary_hidden)
        self._trace_pair("mlp_output", primary_mlp, secondary_mlp)
        with self._device_context(self.primary):
            _mod_gate_local(state.primary, p_mods[5], primary_mlp, p_segments)
        with self._device_context(self.secondary):
            _mod_gate_local(state.secondary, s_mods[5], secondary_mlp, s_segments)
        self._trace_pair("block_output", state.primary, state.secondary)

        self.stats.block_calls += 1
        return state

    def execute_full(self, primary_packet, secondary_packet, full_state, t_emb,
                     mod_segments, rope_freqs, *, primary_fraction=0.5,
                     transformer_options=None):
        partition = build_exact_sp_sequence_partition(
            int(full_state.shape[0]), float(primary_fraction)
        )
        state = self.split_input(full_state, partition)
        state = self.execute_sharded(
            primary_packet, secondary_packet, state, t_emb, mod_segments,
            rope_freqs, transformer_options=transformer_options,
        )
        return self.gather_output(state)

    def execute_chain(self, packet_pairs, full_state, t_emb, mod_segments,
                      rope_freqs, *, primary_fraction=0.5,
                      transformer_options=None):
        """Run an ordered block packet iterable with one ingress and one egress.

        Packet residency is deliberately owned by the caller.  The preview loader
        can therefore stream one pair at a time through DynamicVRAM without this
        mathematical runtime retaining 100 device-resident block copies.
        """
        partition = build_exact_sp_sequence_partition(
            int(full_state.shape[0]), float(primary_fraction)
        )
        state = self.split_input(full_state, partition)
        count = 0
        for primary_packet, secondary_packet in packet_pairs:
            state = self.execute_sharded(
                primary_packet, secondary_packet, state, t_emb, mod_segments,
                rope_freqs, transformer_options=transformer_options,
            )
            count += 1
        if count < 1:
            raise ValueError("Exact-SP block chain requires at least one packet pair")
        return self.gather_output(state)

    def clear_step(self):
        self._secondary_rope = None
        self._secondary_rope_signature = None

    def reset_stats(self):
        from .exact_sp_exchange import ExactSPExchangeStats

        self.stats = ExactSPBlockStats()
        self.exchange.stats = ExactSPExchangeStats()

    def summary(self):
        result = self.stats.as_dict()
        result["exchange"] = self.exchange.stats.as_dict()
        return result
