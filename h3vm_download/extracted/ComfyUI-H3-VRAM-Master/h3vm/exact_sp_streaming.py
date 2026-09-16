from __future__ import annotations

"""DynamicVRAM lifecycle bridge for the Exact-SP Preview block runtime."""

import logging
import time

from .exact_sp_block import ExactSPBlockState, build_exact_sp_sequence_partition
from .streaming_exact import StreamingExactRuntime


LOG = logging.getLogger("H3VM")


class ExactSPStreamingRuntime(StreamingExactRuntime):
    """Keep H3 token state sharded while streaming paired block packets."""

    def __init__(self, primary_device, secondary_device, primary_packets,
                 secondary_packets, *, block_runtime, space,
                 expected_steps=4, primary_fraction=0.5,
                 primary_runtime_reserve_gb=5.0,
                 secondary_runtime_reserve_gb=5.0,
                 primary_hot_cache_gb=2.0,
                 secondary_hot_cache_gb=2.0,
                 trim_interval=1, one_ahead_prefetch=False,
                 telemetry=True, hard_cleanup_after_sample=False):
        indices = sorted(int(i) for i in primary_packets)
        if indices != sorted(int(i) for i in secondary_packets):
            raise ValueError("Exact-SP primary/secondary packet indices differ")
        if indices != list(range(len(indices))):
            raise ValueError("Exact-SP packets must cover a contiguous zero-based block chain")
        super().__init__(
            primary_device,
            secondary_device,
            dict(primary_packets),
            {i: 0 for i in indices},
            space=space,
            expected_steps=expected_steps,
            stripe_size=1,
            primary_runtime_reserve_gb=primary_runtime_reserve_gb,
            secondary_runtime_reserve_gb=secondary_runtime_reserve_gb,
            primary_hot_cache_gb=primary_hot_cache_gb,
            secondary_hot_cache_gb=secondary_hot_cache_gb,
            one_ahead_prefetch=one_ahead_prefetch,
            trim_on_stripe_boundary=False,
            telemetry=telemetry,
            hard_cleanup_after_sample=hard_cleanup_after_sample,
            single_root=False,
        )
        self.packet_blocks = {
            0: dict(primary_packets),
            1: dict(secondary_packets),
        }
        self.owner_blocks = {0: list(indices), 1: list(indices)}
        self.block_runtime = block_runtime
        self.primary_fraction = float(primary_fraction)
        self.trim_interval = max(1, int(trim_interval))
        self.total_blocks = len(indices)
        self._exact_step_started = None

    def _make_prefetch_queue(self, owner: int, transformer_options):
        if not self.one_ahead_prefetch:
            return None
        try:
            import comfy.model_prefetch as mp

            opts = dict(transformer_options or {})
            opts["prefetch_dynamic_vbars"] = True
            modules = [self.packet_blocks[int(owner)][i] for i in self.owner_blocks[int(owner)]]
            return mp.make_prefetch_queue(modules, self.devices[int(owner)], opts)
        except Exception as exc:
            LOG.warning("H3VM Exact-SP prefetch disabled rank=%d: %r", int(owner), exc)
            return None

    def _begin_exact_step(self, transformer_options):
        self._step += 1
        self._exact_step_started = time.perf_counter()
        self.block_runtime.clear_step()
        self.block_runtime.reset_stats()
        self.prefetch_queues = {
            owner: self._make_prefetch_queue(owner, transformer_options)
            for owner in (0, 1)
        }
        for owner in (0, 1):
            if self.prefetch_queues[owner] is not None:
                first = self.owner_blocks[owner][0]
                self._prefetch_pop(owner, self.packet_blocks[owner][first])

    def execute(self, index, state, t_emb, mod_segments, rope_freqs,
                transformer_options=None):
        import torch

        index = int(index)
        opts = {} if transformer_options is None else transformer_options
        if index == 0:
            if isinstance(state, ExactSPBlockState):
                raise RuntimeError("Exact-SP received a stale sharded state at block 0")
            self._begin_exact_step(opts)
            partition = build_exact_sp_sequence_partition(
                int(state.shape[0]), self.primary_fraction
            )
            state = self.block_runtime.split_input(state, partition)
        elif not isinstance(state, ExactSPBlockState):
            raise RuntimeError(f"Exact-SP block {index} expected sharded state")

        primary_packet = self.packet_blocks[0][index]
        secondary_packet = self.packet_blocks[1][index]
        self._prefetch_pop(0, primary_packet)
        self._prefetch_pop(1, secondary_packet)
        state = self.block_runtime.execute_sharded(
            primary_packet,
            secondary_packet,
            state,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=opts,
        )

        trim_due = ((index + 1) % self.trim_interval == 0) or index + 1 == self.total_blocks
        if trim_due:
            torch.cuda.synchronize(self.primary_device)
            torch.cuda.synchronize(self.secondary_device)
            self._trim_owner(0)
            self._trim_owner(1)

        if index + 1 != self.total_blocks:
            return state

        output = self.block_runtime.gather_output(state)
        self._finish_prefetch(0)
        self._finish_prefetch(1)
        self.prefetch_queues = {0: None, 1: None}
        wall_ms = (time.perf_counter() - self._exact_step_started) * 1000.0
        if self.telemetry:
            LOG.info(
                "H3VM EXACT-SP STEP #%d | blocks=%d compute=%.0f/%.0f | wall=%.1fms | %s",
                self._step,
                self.total_blocks,
                self.primary_fraction * 100.0,
                (1.0 - self.primary_fraction) * 100.0,
                wall_ms,
                self.block_runtime.summary(),
            )
        # OUTER_SAMPLE lifecycle owns the real run boundary; expected_steps
        # is a planning/telemetry hint and must not reset state mid-sample.
        return output

    def close(self):
        super().close()
        self.packet_blocks[0].clear()
        self.packet_blocks[1].clear()
        self.block_runtime.clear_step()


class ExactSPBlockProxyFactory:
    @staticmethod
    def make(runtime, index):
        import torch
        import weakref

        class H3ExactSPBlockProxy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._runtime_ref = weakref.ref(runtime)
                self.index = int(index)

            def forward(self, x, t_emb, mod_segments, rope_freqs,
                        transformer_options=None, attention=None):
                if attention is not None:
                    raise RuntimeError("Exact-SP Preview does not support block attention replacement")
                rt = self._runtime_ref()
                if rt is None:
                    raise RuntimeError("Exact-SP Preview runtime was released")
                from .compiler_guard import pause_comfy_allocation_graph
                with pause_comfy_allocation_graph():
                    return rt.execute(
                        self.index,
                        x,
                        t_emb,
                        mod_segments,
                        rope_freqs,
                        transformer_options=transformer_options,
                    )

        return H3ExactSPBlockProxy()
