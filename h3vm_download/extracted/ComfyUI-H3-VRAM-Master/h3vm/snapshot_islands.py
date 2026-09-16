from __future__ import annotations

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor

LOG = logging.getLogger("H3VM")


def _tensor_nbytes(value):
    import torch
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_nbytes(x) for x in value)
    if isinstance(value, dict):
        return sum(_tensor_nbytes(x) for x in value.values())
    return 0


def _used_gib(device):
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / (1024 ** 3)


def _chebyshev_basis(x, degree):
    """Return T_0..T_degree at x using the stable three-term recurrence."""
    degree = max(0, int(degree))
    x = float(max(-1.0, min(1.0, x)))
    vals = [1.0]
    if degree == 0:
        return vals
    vals.append(x)
    for _ in range(2, degree + 1):
        vals.append(2.0 * x * vals[-1] - vals[-2])
    return vals


def _normalize_coordinate_set(history_coords, target_coord):
    """Affine-normalize arbitrary monotonic scheduler coordinates to [-1, 1].

    Including the target in the normalization keeps the tiny Chebyshev system
    well-conditioned while preserving non-uniform timestep spacing.  Affine
    transforms do not change the relative scheduler geometry we care about.
    """
    vals = [float(x) for x in history_coords] + [float(target_coord)]
    if not vals:
        return [], 0.0
    lo, hi = min(vals), max(vals)
    span = float(hi - lo)
    if abs(span) < 1e-12:
        return [0.0 for _ in history_coords], 0.0
    center = 0.5 * (lo + hi)
    half = 0.5 * span
    hist_x = [max(-1.0, min(1.0, (float(x) - center) / half)) for x in history_coords]
    target_x = max(-1.0, min(1.0, (float(target_coord) - center) / half))
    return hist_x, target_x


def _ridge_forecast_weights(history_coords, target_coord, degree=2, ridge=0.05):
    """Small CPU ridge solve for Chebyshev trajectory weights.

    ``history_coords`` can be actual H3 diffusion timesteps rather than ordinal
    step numbers.  The feature dimension never enters this solve: only a tiny
    (degree+1)x(degree+1) CPU system is solved, then scalar weights are applied
    to the host boundary snapshots.
    """
    import torch

    coords = [float(x) for x in history_coords]
    if not coords:
        return []
    degree = max(0, min(int(degree), len(coords) - 1))
    hist_x, target_x = _normalize_coordinate_set(coords, float(target_coord))
    rows = [_chebyshev_basis(x, degree) for x in hist_x]
    target = _chebyshev_basis(target_x, degree)
    X = torch.tensor(rows, dtype=torch.float64, device="cpu")
    phi = torch.tensor(target, dtype=torch.float64, device="cpu")
    reg = max(0.0, float(ridge))
    A = X.T @ X
    if reg > 0.0:
        A = A + reg * torch.eye(A.shape[0], dtype=A.dtype)
    try:
        coeff_map = torch.linalg.solve(A, X.T)
    except Exception:
        coeff_map = torch.linalg.pinv(A) @ X.T
    weights = (phi @ coeff_map).tolist()
    total = float(sum(weights))
    if abs(total) > 1e-9:
        weights = [float(w) / total for w in weights]
    return [float(w) for w in weights]


def _scalar_coordinate(value):
    """Extract one cheap finite scheduler coordinate from a timestep payload."""
    import math
    try:
        import torch
        if torch.is_tensor(value):
            if value.numel() <= 0:
                return None
            flat = value.detach().reshape(-1).float()
            # H3 may carry per-batch / modality entries. The mean is stable and
            # monotonic for the stock schedule while avoiding any feature-sized work.
            out = float(flat.mean().item())
        else:
            out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _sample_view(tensor, max_samples=65536):
    """Cheap deterministic 1-D float sample for predictor diagnostics/control."""
    import torch
    if tensor is None or not torch.is_tensor(tensor):
        return None
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= 0:
        return None
    stride = max(1, int(flat.numel() // max(1, int(max_samples))))
    sample = flat[::stride]
    if sample.numel() > int(max_samples):
        sample = sample[:int(max_samples)]
    return sample.float()


def _relative_rmse(pred, truth):
    import torch
    p = _sample_view(pred)
    t = _sample_view(truth)
    if p is None or t is None or p.numel() != t.numel():
        return None
    diff = p - t
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    denom = torch.sqrt(torch.mean(t * t)).item()
    return float(rmse / max(1e-8, denom))


def _quality_truth_metrics(pred, truth, current=None):
    """LAB7 sampled hidden-feature geometry diagnostics.

    These are diagnostics, not a claim that any scalar metric predicts final
    visual quality.  The goal is to expose whether a predictor preserves the
    direction/statistics of the true boundary even when relative L2 disagrees.
    """
    import torch
    p = _sample_view(pred, max_samples=65536)
    t = _sample_view(truth, max_samples=65536)
    if p is None or t is None or p.numel() != t.numel():
        return None

    eps = 1e-8
    diff = p - t
    p_rms = torch.sqrt(torch.mean(p * p)).item()
    t_rms = torch.sqrt(torch.mean(t * t)).item()
    rel_l2 = torch.sqrt(torch.mean(diff * diff)).item() / max(eps, t_rms)

    dot = torch.sum(p * t).item()
    pn = torch.sqrt(torch.sum(p * p)).item()
    tn = torch.sqrt(torch.sum(t * t)).item()
    cosine = dot / max(eps, pn * tn)

    p_mean = torch.mean(p).item()
    t_mean = torch.mean(t).item()
    p_std = torch.std(p, unbiased=False).item()
    t_std = torch.std(t, unbiased=False).item()
    mean_shift = abs(p_mean - t_mean) / max(eps, t_std)
    std_ratio = p_std / max(eps, t_std)
    norm_ratio = p_rms / max(eps, t_rms)

    delta_cosine = None
    if current is not None:
        c = _sample_view(current, max_samples=65536)
        if c is not None and c.numel() == p.numel():
            dp = p - c
            dt = t - c
            dpn = torch.sqrt(torch.sum(dp * dp)).item()
            dtn = torch.sqrt(torch.sum(dt * dt)).item()
            if dpn > eps and dtn > eps:
                delta_cosine = torch.sum(dp * dt).item() / (dpn * dtn)

    return {
        "rel_l2": float(rel_l2),
        "cos": float(max(-1.0, min(1.0, cosine))),
        "norm_ratio": float(norm_ratio),
        "mean_shift": float(mean_shift),
        "std_ratio": float(std_ratio),
        "delta_cos": None if delta_cosine is None else float(max(-1.0, min(1.0, delta_cosine))),
    }


def _weighted_sample_error(history, weights, truth):
    """Evaluate a scalar-weight forecast without materializing another full tensor."""
    import torch
    if not history or not weights or len(history) != len(weights):
        return None
    t = _sample_view(truth)
    if t is None:
        return None
    pred = None
    for (_, snap), w in zip(history, weights):
        s = _sample_view(snap)
        if s is None or s.numel() != t.numel():
            return None
        pred = s.mul(float(w)) if pred is None else pred.add(s, alpha=float(w))
    diff = pred - t
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    denom = torch.sqrt(torch.mean(t * t)).item()
    return float(rmse / max(1e-8, denom))


def _coordinate_ratio(history, target_coordinate, clip=3.0):
    """Return target step span / previous true step span for the last two anchors."""
    if history is None or len(history) < 2:
        return None
    x0 = float(history[-2][0])
    x1 = float(history[-1][0])
    den = x1 - x0
    if abs(den) < 1e-12:
        return None
    ratio = (float(target_coordinate) - x1) / den
    limit = max(1.0, float(clip))
    return float(max(-limit, min(limit, ratio)))


def _two_point_weights(history, target_coordinate, gain, ratio_clip=3.0):
    """Scheduler-aware secant weights: x_n + gain*r*(x_n-x_{n-1})."""
    usable = list(history)[-2:]
    ratio = _coordinate_ratio(usable, target_coordinate, clip=ratio_clip)
    if len(usable) < 2 or ratio is None:
        return usable, None, ratio
    advance = float(gain) * float(ratio)
    return usable, [-advance, 1.0 + advance], float(ratio)


def _fit_transition_gain(history, target_coordinate, truth, gain_min=0.0, gain_max=1.25):
    """Fit a scalar velocity gain from the newly realized true boundary."""
    import torch
    usable = list(history)[-2:]
    ratio = _coordinate_ratio(usable, target_coordinate)
    if len(usable) < 2 or ratio is None:
        return None, ratio, None
    s0 = _sample_view(usable[-2][1], max_samples=65536)
    s1 = _sample_view(usable[-1][1], max_samples=65536)
    t = _sample_view(truth, max_samples=65536)
    if s0 is None or s1 is None or t is None or s0.numel() != s1.numel() or s1.numel() != t.numel():
        return None, ratio, None
    basis = (s1 - s0).mul(float(ratio))
    desired = t - s1
    denom = torch.sum(basis * basis).item()
    if denom <= 1e-20:
        return 0.0, ratio, None
    raw = torch.sum(desired * basis).item() / denom
    clipped = max(float(gain_min), min(float(gain_max), float(raw)))
    residual = desired - basis * float(clipped)
    fit_rmse = torch.sqrt(torch.mean(residual * residual)).item()
    truth_rms = torch.sqrt(torch.mean(t * t)).item()
    fit_err = float(fit_rmse / max(1e-8, truth_rms))
    return float(clipped), float(ratio), fit_err


def _forecast_scalar_spectral(history, target_coordinate, degree=2, ridge=0.05,
                              lo=0.0, hi=1.25):
    """Spectrum-style Chebyshev/ridge forecast for a tiny scalar control signal."""
    hist = list(history)
    if len(hist) < 2:
        return None, None
    degree = max(1, min(int(degree), len(hist) - 1))
    usable = hist[-max(degree + 1, min(6, len(hist))):]
    if len(usable) < degree + 1:
        return None, None
    coords = [float(x) for x, _ in usable]
    weights = _ridge_forecast_weights(coords, float(target_coordinate), degree=degree, ridge=float(ridge))
    if not weights or len(weights) != len(usable):
        return None, None
    value = sum(float(w) * float(v) for w, (_, v) in zip(weights, usable))
    value = max(float(lo), min(float(hi), float(value)))
    return float(value), [float(w) for w in weights]


class BoundedPinnedMailbox:
    """Small explicit pinned host cache for activation/snapshot traffic only.

    This intentionally ignores ComfyUI's global pinned-memory policy. It never
    pins model weights and never grows beyond ``cap_mb``. The safe launch profile
    can therefore keep ``--disable-pinned-memory`` while H3VM owns a small DMA
    runway for its own host mailbox.
    """

    def __init__(self, cap_mb=1024, stats_recorder=None):
        self.cap_bytes = max(0, int(cap_mb)) * 1024 * 1024
        self._buffers = {}
        self._bytes = 0
        self._warned = set()
        self._stats_recorder = stats_recorder
        self._lock = threading.RLock()

    @staticmethod
    def _nbytes(t):
        return int(t.numel() * t.element_size())

    def _record(self, nbytes, dt):
        fn = self._stats_recorder
        if fn is not None:
            try:
                fn("bounded_pinned", int(nbytes), float(dt))
            except Exception:
                pass

    def _alloc(self, key, template):
        import torch
        with self._lock:
            spec = (tuple(template.shape), template.dtype)
            entry = self._buffers.get(key)
            if entry is not None and entry["spec"] == spec:
                return entry["tensor"]
            if entry is not None:
                self._bytes -= entry["bytes"]
                self._buffers.pop(key, None)

            need = self._nbytes(template)
            if self._bytes + need > self.cap_bytes:
                if key not in self._warned:
                    LOG.warning(
                        "H3VM Dev9.3.1 pinned mailbox fallback | key=%s need=%.1fMiB used/cap=%.1f/%.1fMiB",
                        key, need/(1024**2), self._bytes/(1024**2), self.cap_bytes/(1024**2),
                    )
                    self._warned.add(key)
                return None
            try:
                out = torch.empty(tuple(template.shape), dtype=template.dtype, device="cpu", pin_memory=True)
            except Exception as e:
                if key not in self._warned:
                    LOG.warning("H3VM Dev9.3.1 pinned mailbox allocation failed key=%s: %r", key, e)
                    self._warned.add(key)
                return None
            self._buffers[key] = {"tensor": out, "spec": spec, "bytes": need}
            self._bytes += need
            return out

    def to_host(self, src, key):
        import torch
        if src.device.type == "cpu":
            return src
        buf = self._alloc(key, src)
        t0 = time.perf_counter()
        if buf is None:
            out = src.to("cpu")
            torch.cuda.synchronize(src.device)
            self._record(self._nbytes(src), time.perf_counter() - t0)
            return out
        buf.copy_(src, non_blocking=True)
        torch.cuda.synchronize(src.device)
        self._record(self._nbytes(src), time.perf_counter() - t0)
        return buf

    def to_device(self, src, dst_device):
        import torch
        dst = torch.device(dst_device)
        if src.device == dst:
            return src
        t0 = time.perf_counter()
        if src.device.type == "cpu" and bool(src.is_pinned()):
            out = torch.empty_like(src, device=dst)
            out.copy_(src, non_blocking=True)
        else:
            out = src.to(dst, non_blocking=False)
        torch.cuda.synchronize(dst)
        self._record(self._nbytes(src), time.perf_counter() - t0)
        return out

    def predict(self, current, previous, beta, key="predict"):
        """First-order predictor with a pinned output when budget allows."""
        import torch
        if current.device.type != "cpu" or previous.device.type != "cpu":
            raise RuntimeError("Pinned mailbox predictor expects CPU snapshots")
        out = self._alloc(key, current)
        t0 = time.perf_counter()
        if out is None:
            out = torch.add(current, current, alpha=float(beta))
            out.add_(previous, alpha=-float(beta))
        else:
            torch.mul(current, 1.0 + float(beta), out=out)
            out.add_(previous, alpha=-float(beta))
        return out, (time.perf_counter() - t0) * 1000.0

    def weighted_sum(self, tensors, weights, key="spectral_predict"):
        """Scalar weighted sum into a bounded pinned output when possible."""
        import torch
        if not tensors or len(tensors) != len(weights):
            raise ValueError("weighted_sum expects matching non-empty tensors/weights")
        ref = tensors[0]
        if any(t.device.type != "cpu" for t in tensors):
            raise RuntimeError("Pinned mailbox weighted_sum expects CPU snapshots")
        if any(tuple(t.shape) != tuple(ref.shape) or t.dtype != ref.dtype for t in tensors):
            raise RuntimeError("Pinned mailbox weighted_sum expects shape/dtype parity")
        out = self._alloc(key, ref)
        t0 = time.perf_counter()
        if out is None:
            out = torch.mul(ref, float(weights[0]))
        else:
            torch.mul(ref, float(weights[0]), out=out)
        for t, w in zip(tensors[1:], weights[1:]):
            out.add_(t, alpha=float(w))
        return out, (time.perf_counter() - t0) * 1000.0

    @property
    def allocated_mib(self):
        return self._bytes / (1024 ** 2)

    def clear(self):
        with self._lock:
            self._buffers.clear()
            self._bytes = 0
            self._warned.clear()


def should_use_exact_step(step: int, expected_steps: int, refresh_interval: int, exact_last_step: bool, exact_steps=None) -> bool:
    """Return True when the tail must consume the current prefix snapshot.

    Step 1 is always exact to seed the mailbox. Optional periodic refreshes limit
    stale-feature drift. The last expected call can also be exact so the final
    denoise prediction is not based on a one-call-old boundary activation.
    """
    step = int(step)
    expected_steps = max(0, int(expected_steps))
    refresh_interval = max(0, int(refresh_interval))
    if exact_steps is not None and step in set(int(x) for x in exact_steps):
        return True
    if step <= 1:
        return True
    if exact_last_step and expected_steps > 0 and step == expected_steps:
        return True
    if refresh_interval > 0 and (step - 1) % refresh_interval == 0:
        return True
    return False


class SnapshotIslandRuntime:
    """Approximate two-island H3 pipeline with a host-owned stale snapshot mailbox.

    There is intentionally no GPU0->GPU1 or GPU1->GPU0 tensor handoff in this
    runtime. Each accelerator talks only to ordinary pageable system RAM:

      primary fixed/front-end -> host -> secondary prefix island
      secondary prefix output -> host snapshot mailbox
      previous host snapshot -> primary tail island

    After the warm-up call, the primary tail consumes boundary h from the
    previous denoise/model call while the secondary computes the current prefix.
    This removes the exact inter-island barrier at the cost of one-call-stale
    boundary features. It is an experiment, not an exact H3 execution mode.
    """

    def __init__(self, primary_device, secondary_device, prefix_blocks, tail_blocks, split_count, *,
                 space, expected_steps=20, refresh_interval=0, exact_last_step=True,
                 prefix_prefetch=True, tail_prefetch=True, telemetry=True,
                 predictor_mode="stale", predictor_beta=0.75,
                 spectral_degree=2, spectral_history=4, spectral_ridge=0.005,
                 spectral_mix=0.65, spectral_max_delta_ratio=1.6, spectral_adapt=True,
                 spectral_coordinate="timestep", spectral_confidence="conservative", spectral_debug="full",
                 launch_tail_before_stage=False, host_feeder="pageable",
                 pinned_mailbox_mb=1024, exact_steps=None, quality_profile="FAST | E-S-P-E"):
        self.primary_device = primary_device
        self.secondary_device = secondary_device
        self.prefix_blocks = dict(prefix_blocks)
        self.tail_blocks = dict(tail_blocks)
        self.split_count = int(split_count)
        self.first_prefix = 0
        self.last_prefix = self.split_count - 1
        self.first_tail = self.split_count
        self.last_tail = max(self.tail_blocks) if self.tail_blocks else self.split_count - 1
        self.space = space
        self.expected_steps = max(0, int(expected_steps))
        self.refresh_interval = max(0, int(refresh_interval))
        self.exact_last_step = bool(exact_last_step)
        self.prefix_prefetch = bool(prefix_prefetch)
        self.tail_prefetch = bool(tail_prefetch)
        self.telemetry = bool(telemetry)
        self.predictor_mode = str(predictor_mode).strip().lower()
        if self.predictor_mode == "hybrid_spectral":
            self.predictor_mode = "hybrid"
        aliases = {
            "stale_raw": "stale_raw",
            "linear_raw": "linear_raw",
            "spectral_raw": "spectral_raw",
            "phase_raw": "phase_raw",
            "safe_blend": "safe_blend",
            "adaptive_secant": "adaptive",
            "phase_adaptive": "phase",
            "brake_phase": "brake",
            "spectral_gain": "sgain",
            "auto": "auto_brake",
            "auto_brake": "auto_brake",
            "auto_phase": "auto_phase",
            "auto_legacy": "auto_legacy",
        }
        self.predictor_mode = aliases.get(self.predictor_mode, self.predictor_mode)
        if self.predictor_mode not in ("stale_raw", "linear_raw", "spectral_raw", "phase_raw", "safe_blend", "stale", "linear", "adaptive", "phase", "brake", "sgain", "spectral", "hybrid", "auto_brake", "auto_phase", "auto_legacy"):
            raise ValueError(f"Unsupported snapshot predictor mode: {self.predictor_mode}")
        self.predictor_beta = float(predictor_beta)
        self.spectral_degree = max(1, min(4, int(spectral_degree)))
        self.spectral_history = max(self.spectral_degree + 1, min(6, int(spectral_history)))
        self.spectral_ridge = max(0.0, float(spectral_ridge))
        self.spectral_mix = max(0.0, min(1.0, float(spectral_mix)))
        self.spectral_max_delta_ratio = max(1.0, float(spectral_max_delta_ratio))
        self.spectral_adapt = bool(spectral_adapt)
        self.spectral_coordinate = str(spectral_coordinate).strip().lower()
        if self.spectral_coordinate not in ("timestep", "step_index", "auto"):
            self.spectral_coordinate = "timestep"
        self.spectral_confidence = str(spectral_confidence).strip().lower()
        if self.spectral_confidence not in ("off", "conservative", "adaptive"):
            self.spectral_confidence = "conservative"
        self.spectral_debug = str(spectral_debug).strip().lower()
        if self.spectral_debug not in ("off", "summary", "full"):
            self.spectral_debug = "full"
        self._weight_cache = {}
        self._captured_timestep = None
        self._current_coordinate = None
        self._coordinate_source = None
        self._coordinate_warned = False
        self._error_ema = {"stale": None, "linear": None, "adaptive": None, "phase": None, "brake": None, "sgain": None, "spectral": None}
        self._error_count = {"stale": 0, "linear": 0, "adaptive": 0, "phase": 0, "brake": 0, "sgain": 0, "spectral": 0}
        self._diag_sum = {"used": 0.0, "stale": 0.0, "linear": 0.0, "adaptive": 0.0, "phase": 0.0, "brake": 0.0, "sgain": 0.0, "spectral": 0.0}
        self._diag_count = {"used": 0, "stale": 0, "linear": 0, "adaptive": 0, "phase": 0, "brake": 0, "sgain": 0, "spectral": 0}
        self._winner_counts = {"stale": 0, "linear": 0, "adaptive": 0, "phase": 0, "brake": 0, "sgain": 0, "spectral": 0}
        self._spectral_gate_open = self.spectral_confidence == "off"
        self._adaptive_spectral_mix = self.spectral_mix if self._spectral_gate_open else 0.0
        self._gain_ema = max(0.0, min(1.25, float(self.predictor_beta)))
        self._gain_history = []
        self._gain_fit_last = None
        self._phase_gain = max(0.0, min(1.25, float(self.predictor_beta)))
        self._phase_state = "warmup"
        self._phase_low_streak = 0
        self._phase_high_streak = 0
        self._phase_regret_streak = 0
        self._phase_switches = 0
        self._auto_choice = "linear"
        self._last_control_signature = None
        self._quality_sum = {"rel_l2": 0.0, "cos": 0.0, "norm_ratio": 0.0, "mean_shift": 0.0, "std_ratio": 0.0, "delta_cos": 0.0}
        self._quality_count = {k: 0 for k in self._quality_sum}
        self.launch_tail_before_stage = bool(launch_tail_before_stage)
        self.host_feeder = str(host_feeder)
        if self.host_feeder not in ("pageable", "bounded_pinned"):
            raise ValueError(f"Unsupported host feeder: {self.host_feeder}")
        self.pinned_mailbox_mb = max(0, int(pinned_mailbox_mb))
        self.exact_steps = frozenset(int(x) for x in (exact_steps or ()))
        self.quality_profile = str(quality_profile)
        self._pinned = (
            BoundedPinnedMailbox(cap_mb=self.pinned_mailbox_mb, stats_recorder=getattr(self.space.transport, "_record", None))
            if self.host_feeder == "bounded_pinned" else None
        )

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="H3VM-PrimaryIsland")
        self._step = 0
        self._snapshot_cpu = None
        self._snapshot_prev_cpu = None
        self._snapshot_history = []  # [(scheduler_coord, true host boundary)], oldest -> newest
        self._pending_snapshot_cpu = None
        self._future = None
        self._tail_output = None
        self._prefix_t_emb = None
        self._prefix_segments = None
        self._prefix_rope = None
        self._prefix_queue = None
        self._prefix_started = None
        self._prefix_compute_started = None
        self._step_started = None
        self._step_stats0 = None
        self._exact_this_step = True
        self._snapshot_write_ms = 0.0
        self._stage_prefix_ms = 0.0
        self._predictor_used = False
        self._peak_primary = 0.0
        self._peak_secondary = 0.0
        self._prefix_logged = False
        self._tail_logged = False

    @staticmethod
    def _stats_snapshot(space):
        s = space.transport.stats
        by = s.get("by_mode", {})
        direct = by.get("direct_d2d", {"moves": 0, "bytes": 0, "seconds": 0.0})
        return (
            int(s.get("moves", 0)), int(s.get("bytes", 0)), float(s.get("seconds", 0.0)),
            int(direct.get("moves", 0)), int(direct.get("bytes", 0)),
        )

    def _sample_memory(self):
        if not self.telemetry:
            return
        try:
            self._peak_primary = max(self._peak_primary, _used_gib(self.primary_device))
            self._peak_secondary = max(self._peak_secondary, _used_gib(self.secondary_device))
        except Exception:
            pass

    def _make_prefetch(self, blocks, device, transformer_options, enabled, which):
        if not enabled:
            return None
        try:
            import comfy.model_prefetch
            opts = dict(transformer_options or {})
            opts["prefetch_dynamic_vbars"] = True
            queue = [blocks[i] for i in sorted(blocks)]
            out = comfy.model_prefetch.make_prefetch_queue(queue, device, opts)
            flag = "_prefix_logged" if which == "prefix" else "_tail_logged"
            if not getattr(self, flag):
                LOG.info(
                    "H3VM Dev9 %s island prefetch | active=%s blocks=%d device=%s",
                    which, out is not None, len(queue), device,
                )
                setattr(self, flag, True)
            return out
        except Exception as e:
            LOG.warning("H3VM Dev9 %s island prefetch disabled: %r", which, e)
            return None

    def _finish_prefetch(self, queue, device, which):
        """Safely retire the last prefetched block without draining past the sentinel.

        ComfyUI's queue is shaped as [None] + blocks + [None]. One pop per block
        plus one final flush is valid. Dev9.3.1 experimentally added an extra early
        pop to 'prime' the queue, which shifted the queue by one and made the
        final flush consume the last sentinel, after which model_prefetch tried
        to read queue[0] from an empty list. Do not pre-consume the queue.
        """
        if queue is None:
            return
        try:
            import comfy.model_prefetch
            # Normal unprimed path reaches the flush with two entries left.
            # If a future ComfyUI change or partial failure leaves only the final
            # sentinel, do not pop it because prefetch_queue_pop indexes queue[0]
            # after the pop.
            if len(queue) >= 2:
                comfy.model_prefetch.prefetch_queue_pop(queue, device, None)
            elif len(queue) == 1:
                LOG.debug("H3VM Dev9.3.1 %s prefetch already drained to sentinel; skip final pop", which)
        except Exception as e:
            LOG.warning("H3VM Dev9.3.1 %s prefetch finalization failed: %r", which, e)

    def _host_only_tree(self, value, device, *, feeder_key=None):
        """Acquire a device replica strictly through host memory.

        Large activations can use the bounded pinned mailbox. Tiny metadata stays
        pageable so the explicit pinned budget is spent only on the hot path.
        """
        import torch
        if torch.is_tensor(value):
            if value.device == device:
                return value
            if self._pinned is not None and feeder_key is not None:
                host = self._pinned.to_host(value, feeder_key)
                return self._pinned.to_device(host, device)
            host = self.space.transport.move_tensor(value, "cpu", mode="neutral_pageable")
            return self.space.transport.move_tensor(host, device, mode="neutral_pageable")
        if isinstance(value, tuple):
            return tuple(self._host_only_tree(x, device) for x in value)
        if isinstance(value, list):
            return [self._host_only_tree(x, device) for x in value]
        if isinstance(value, dict):
            return {k: self._host_only_tree(v, device) for k, v in value.items()}
        return value

    def observe_timestep(self, timestep):
        """Called by a DIFFUSION_MODEL wrapper before block execution."""
        value = _scalar_coordinate(timestep)
        if value is not None:
            self._captured_timestep = float(value)

    def _coordinate_for_step(self, step):
        requested = self.spectral_coordinate
        captured = self._captured_timestep
        self._captured_timestep = None
        if requested in ("timestep", "auto") and captured is not None:
            source, coord = "timestep", float(captured)
        else:
            source, coord = "step_index", float(step)
            if requested == "timestep":
                # First-call scheduler metadata can be temporarily unavailable on
                # some wrapper/order combinations. Use the ordinal coordinate for
                # this exact/warmup call, but do not commit it as predictor history.
                # The first real timestep then becomes the initial coordinate source
                # instead of looking like a mid-run source switch.
                self._current_coordinate = None
                return float(coord), source
        # Do not fit one polynomial across mixed coordinate systems. If the
        # capture source changes, keep the stale mailbox but restart spectral history.
        if self._coordinate_source is not None and source != self._coordinate_source and self._snapshot_history:
            LOG.info("H3VM predictor coordinate source changed %s -> %s; restarting coordinate history", self._coordinate_source, source)
            self._snapshot_history = []
            self._gain_history = []
            self._gain_ema = max(0.0, min(1.25, float(self.predictor_beta)))
            self._phase_gain = max(0.0, min(1.25, float(self.predictor_beta)))
            self._phase_state = "warmup"
            self._phase_low_streak = 0
            self._phase_high_streak = 0
            self._phase_regret_streak = 0
            self._phase_switches = 0
            self._weight_cache.clear()
        self._coordinate_source = source
        self._current_coordinate = float(coord)
        return float(coord), source

    def _spectral_weights(self, history, target_coordinate):
        usable = list(history)[-self.spectral_history:]
        if len(usable) < self.spectral_degree + 1:
            return usable, None
        coords = tuple(round(float(coord), 10) for coord, _ in usable)
        target = round(float(target_coordinate), 10)
        key = (coords, target, int(self.spectral_degree), round(self.spectral_ridge, 8))
        weights = self._weight_cache.get(key)
        if weights is None:
            weights = _ridge_forecast_weights(
                coords, target, degree=self.spectral_degree, ridge=self.spectral_ridge,
            )
            self._weight_cache[key] = weights
        return usable, weights

    def _ema_update(self, name, value, alpha=0.35):
        if value is None:
            return
        value = float(value)
        old = self._error_ema.get(name)
        self._error_ema[name] = value if old is None else (1.0 - alpha) * float(old) + alpha * value
        self._error_count[name] = int(self._error_count.get(name, 0)) + 1

    def _update_control(self, stale_err=None, linear_err=None, adaptive_err=None,
                        phase_err=None, brake_err=None, sgain_err=None, spectral_err=None):
        """Update legacy LAB3 arbitration and LAB7 retained shadow statistics.

        AUTO_LEGACY preserves the LAB3 realized-error hysteresis controller.
        AUTO_PHASE is handled separately by ``_update_phase_control`` and uses
        the fitted local gain as a phase-transition signal.
        """
        for name, value in (
            ("stale", stale_err), ("linear", linear_err), ("adaptive", adaptive_err),
            ("phase", phase_err), ("brake", brake_err), ("sgain", sgain_err), ("spectral", spectral_err),
        ):
            self._ema_update(name, value)

        # Legacy LAB3 AUTO controller is retained as an explicit A/B option.
        candidates = {}
        for name in ("stale", "linear", "adaptive", "sgain"):
            value = self._error_ema.get(name)
            count = int(self._error_count.get(name, 0))
            minimum = 1 if name in ("stale", "linear") else 2
            if value is not None and count >= minimum:
                candidates[name] = float(value)

        if candidates:
            best = min(candidates, key=candidates.get)
            current = self._auto_choice if self._auto_choice in candidates else None
            if current is None:
                self._auto_choice = best
            else:
                margin = 0.985 if best == "stale" else 0.97
                if best != current and candidates[best] <= candidates[current] * margin:
                    self._auto_choice = best

        # Direct full-boundary spectral stays gated and shadow-only for AUTO paths.
        stale = self._error_ema.get("stale")
        linear = self._error_ema.get("linear")
        spectral = self._error_ema.get("spectral")
        base_err = None
        if stale is not None and linear is not None:
            base_err = min(float(stale), float(linear))
        elif stale is not None:
            base_err = float(stale)
        elif linear is not None:
            base_err = float(linear)

        gate = self._spectral_gate_open
        target_mix = 0.0
        evidence = int(self._error_count.get("spectral", 0))
        if self.spectral_confidence == "off":
            gate = True
            target_mix = self.spectral_mix
        elif spectral is not None and base_err is not None:
            ratio = float(spectral) / max(1e-9, float(base_err))
            if self.spectral_confidence == "conservative":
                if gate:
                    gate = ratio <= 1.01
                else:
                    gate = evidence >= 2 and ratio <= 0.97
                target_mix = self.spectral_mix if gate else 0.0
            else:
                gate = ratio < 0.995
                advantage = max(0.0, min(1.0, (1.0 - ratio) / 0.10))
                target_mix = self.spectral_mix * advantage if gate else 0.0
        else:
            gate = self.spectral_confidence == "off"
            target_mix = self.spectral_mix if gate else 0.0

        self._spectral_gate_open = bool(gate)
        if self.spectral_adapt:
            self._adaptive_spectral_mix = 0.70 * float(self._adaptive_spectral_mix) + 0.30 * float(target_mix)
        else:
            self._adaptive_spectral_mix = float(target_mix)
        if not self._spectral_gate_open:
            self._adaptive_spectral_mix = min(self._adaptive_spectral_mix, 0.05)

    def _update_phase_control(self, gain_fit=None, stale_err=None, phase_err=None):
        """LAB6 phase-aware scalar controller.

        The LAB3 run showed a repeatable pattern: useful positive secant gain in
        the early/middle diffusion phase, then a collapse toward zero.  Rather
        than waiting for error EMAs to lag behind that transition, LAB4 treats
        the newly fitted gain as an observable phase signal.  Downward gain
        changes are followed quickly; upward changes are admitted more slowly.
        Two consecutive low-gain or regret observations enter STALE phase.
        Recovery requires two strong-gain observations *and* a shadow phase
        prediction that beats stale, preventing oscillation near the boundary.
        """
        if gain_fit is not None:
            fit = max(0.0, min(1.25, float(gain_fit)))
            old = float(self._phase_gain)
            if fit < old:
                # Fast brake: LAB3's symmetric EMA was the main source of late-tail lag.
                self._phase_gain = 0.30 * old + 0.70 * fit
            else:
                # Slow throttle reopening keeps a single noisy rebound from overshooting.
                self._phase_gain = 0.75 * old + 0.25 * fit
            self._phase_gain = max(0.0, min(1.25, float(self._phase_gain)))

            if self._phase_state == "warmup":
                self._phase_state = "adaptive"

            if fit <= 0.15:
                self._phase_low_streak += 1
            else:
                self._phase_low_streak = 0
            if fit >= 0.25:
                self._phase_high_streak += 1
            else:
                self._phase_high_streak = 0

        if stale_err is not None and phase_err is not None:
            if float(phase_err) > float(stale_err) * 1.02:
                self._phase_regret_streak += 1
            elif float(phase_err) <= float(stale_err) * 1.005:
                self._phase_regret_streak = max(0, self._phase_regret_streak - 1)

        if self._phase_state != "stale":
            if self._phase_low_streak >= 2 or self._phase_regret_streak >= 2:
                self._phase_state = "stale"
                self._phase_switches += 1
                # AUTO_PHASE hard-brakes into stale. Explicit PHASE_ADAPTIVE
                # keeps its fast-decaying gain for an apples-to-apples A/B.
                if self.predictor_mode not in ("phase", "phase_raw"):
                    self._phase_gain = min(float(self._phase_gain), 0.08)
        else:
            can_reopen = (
                self._phase_high_streak >= 2 and stale_err is not None and phase_err is not None
                and float(phase_err) <= float(stale_err) * 0.995
            )
            if can_reopen:
                self._phase_state = "adaptive"
                self._phase_low_streak = 0
                self._phase_regret_streak = 0
                self._phase_switches += 1

    def _effective_mode(self, requested_mode, availability=None):
        requested = str(requested_mode).lower()
        availability = dict(availability or {})
        if requested == "auto":
            requested = "auto_brake"
        # LAB6 QUALITY TRUTH modes bypass hidden fallback semantics so the label
        # describes what is actually fed into the tail island.
        if requested == "stale_raw":
            return "stale", "quality-stale-raw"
        if requested == "linear_raw":
            return "linear", "quality-linear-raw"
        if requested == "phase_raw":
            return ("phase", "quality-phase-raw") if availability.get("phase") else ("stale", "quality-phase-warmup-stale")
        if requested == "safe_blend":
            return ("safe_blend", "quality-safe-blend") if availability.get("safe_blend") else ("stale", "quality-safe-blend-warmup-stale")
        if requested == "spectral_raw":
            return ("spectral_raw", "quality-spectral-raw") if availability.get("spectral") else ("linear", "quality-spectral-warmup-linear")
        if requested == "stale":
            return "stale", "requested-stale"
        if requested == "linear":
            return "linear", "requested-linear"
        if requested == "adaptive":
            return ("adaptive", "adaptive-active") if availability.get("adaptive") else ("linear", "adaptive-warmup")
        if requested == "phase":
            # Explicit A/B: keep using the phase gain even after the controller
            # would have hard-switched AUTO_PHASE to stale.
            return ("phase", "phase-adaptive-explicit") if availability.get("phase") else ("linear", "phase-warmup")
        if requested == "brake":
            return ("brake", "brake-phase-explicit") if availability.get("brake") else ("phase", "brake-warmup")
        if requested == "sgain":
            if availability.get("sgain"):
                return "sgain", "spectral-gain-active"
            if availability.get("adaptive"):
                return "adaptive", "spectral-gain-warmup"
            return "linear", "spectral-gain-warmup"
        if requested == "spectral":
            if not availability.get("spectral"):
                return "linear", "spectral-warmup"
            if self.spectral_confidence == "off" or self._spectral_gate_open:
                return "spectral", "spectral-gate-open"
            return "linear", "spectral-gate-closed"
        if requested == "hybrid":
            if not availability.get("spectral"):
                return "linear", "hybrid-warmup"
            if self.spectral_confidence != "off" and not self._spectral_gate_open:
                return "linear", "hybrid-gate-closed"
            if float(self._adaptive_spectral_mix) <= 0.01:
                return "linear", "hybrid-mix-zero"
            return "hybrid", "hybrid-active"

        if requested == "auto_brake":
            if self._phase_state == "stale":
                return "stale", "auto-brake-stale"
            if availability.get("brake"):
                return "brake", "auto-brake-active"
            if availability.get("phase"):
                return "phase", "auto-brake-warmup-phase"
            if availability.get("adaptive"):
                return "adaptive", "auto-brake-warmup-adaptive"
            return "linear", "auto-brake-warmup-linear"

        if requested == "auto_phase":
            if self._phase_state == "stale":
                return "stale", "auto-phase-stale"
            if availability.get("phase"):
                return "phase", "auto-phase-adaptive"
            if availability.get("adaptive"):
                return "adaptive", "auto-phase-warmup-adaptive"
            return "linear", "auto-phase-warmup-linear"

        # AUTO_LEGACY: exact LAB3 realized-error hysteresis for direct A/B.
        choice = str(self._auto_choice or "linear")
        if choice == "stale":
            return "stale", "auto-legacy-stale"
        if choice == "sgain" and availability.get("sgain"):
            return "sgain", "auto-legacy-spectral-gain"
        if choice == "adaptive" and availability.get("adaptive"):
            return "adaptive", "auto-legacy-adaptive"
        return "linear", "auto-legacy-linear"

    def _accumulate_diag(self, used=None, stale=None, linear=None, adaptive=None,
                         phase=None, brake=None, sgain=None, spectral=None):
        vals = {
            "used": used, "stale": stale, "linear": linear,
            "adaptive": adaptive, "phase": phase, "brake": brake, "sgain": sgain, "spectral": spectral,
        }
        for name, value in vals.items():
            if value is not None:
                self._diag_sum[name] += float(value)
                self._diag_count[name] += 1
        candidates = {
            k: v for k, v in (
                ("stale", stale), ("linear", linear), ("adaptive", adaptive),
                ("phase", phase), ("brake", brake), ("sgain", sgain), ("spectral", spectral),
            ) if v is not None
        }
        if candidates:
            winner = min(candidates, key=candidates.get)
            self._winner_counts[winner] += 1

    def _log_predictor_summary(self):
        if not self.telemetry or self.spectral_debug == "off":
            return

        def avg(name):
            n = self._diag_count[name]
            return self._diag_sum[name] / n if n else None

        LOG.info(
            "H3VM PREDICTOR-V7 LAB7 SUMMARY | requested=%s coord=%s confidence=%s | "
            "avg used/stale/linear/adaptive/phase/brake/sgain/spectral=%s/%s/%s/%s/%s/%s/%s/%s | "
            "wins stale/linear/adaptive/phase/brake/sgain/spectral=%d/%d/%d/%d/%d/%d/%d | "
            "legacy_auto=%s phase=%s phase_gain=%.3f low/high/regret=%d/%d/%d switches=%d direct_spectral_gate=%s",
            self.predictor_mode, self.spectral_coordinate, self.spectral_confidence,
            "%.6f" % avg("used") if avg("used") is not None else "n/a",
            "%.6f" % avg("stale") if avg("stale") is not None else "n/a",
            "%.6f" % avg("linear") if avg("linear") is not None else "n/a",
            "%.6f" % avg("adaptive") if avg("adaptive") is not None else "n/a",
            "%.6f" % avg("phase") if avg("phase") is not None else "n/a",
            "%.6f" % avg("brake") if avg("brake") is not None else "n/a",
            "%.6f" % avg("sgain") if avg("sgain") is not None else "n/a",
            "%.6f" % avg("spectral") if avg("spectral") is not None else "n/a",
            self._winner_counts["stale"], self._winner_counts["linear"],
            self._winner_counts["adaptive"], self._winner_counts["phase"],
            self._winner_counts["brake"], self._winner_counts["sgain"], self._winner_counts["spectral"],
            self._auto_choice, self._phase_state, float(self._phase_gain),
            int(self._phase_low_streak), int(self._phase_high_streak), int(self._phase_regret_streak),
            int(self._phase_switches),
            "OPEN" if self._spectral_gate_open else "CLOSED",
        )
        def qavg(name):
            n = int(self._quality_count.get(name, 0))
            return self._quality_sum.get(name, 0.0) / n if n else None
        LOG.info(
            "H3VM PRODUCTION-QUALITY LAB7 SUMMARY | mode=%s | rel_l2=%s cos=%s norm_ratio=%s mean_shift=%s std_ratio=%s delta_cos=%s",
            self.predictor_mode,
            "%.6f" % qavg("rel_l2") if qavg("rel_l2") is not None else "n/a",
            "%.6f" % qavg("cos") if qavg("cos") is not None else "n/a",
            "%.6f" % qavg("norm_ratio") if qavg("norm_ratio") is not None else "n/a",
            "%.6f" % qavg("mean_shift") if qavg("mean_shift") is not None else "n/a",
            "%.6f" % qavg("std_ratio") if qavg("std_ratio") is not None else "n/a",
            "%.6f" % qavg("delta_cos") if qavg("delta_cos") is not None else "n/a",
        )

    def _materialize_weighted_prediction(self, history, weights, key="spectral_predict"):
        import torch
        tensors = [snap for _, snap in history]
        if self._pinned is not None:
            return self._pinned.weighted_sum(tensors, weights, key=key)
        t0 = time.perf_counter()
        out = torch.mul(tensors[0], float(weights[0]))
        for tensor, weight in zip(tensors[1:], weights[1:]):
            out.add_(tensor, alpha=float(weight))
        return out, (time.perf_counter() - t0) * 1000.0

    def _apply_delta_guard(self, predicted, current, previous):
        """Bound spectral extrapolation against the recent true-boundary velocity."""
        import torch
        p = _sample_view(predicted, max_samples=32768)
        c = _sample_view(current, max_samples=32768)
        q = _sample_view(previous, max_samples=32768)
        if p is None or c is None or q is None or p.numel() != c.numel() or c.numel() != q.numel():
            return predicted, 1.0, None
        spec = p - c
        lin = c - q
        spec_rms = torch.sqrt(torch.mean(spec * spec)).item()
        lin_rms = torch.sqrt(torch.mean(lin * lin)).item()
        limit = self.spectral_max_delta_ratio * max(1e-8, float(self.predictor_beta) * lin_rms)
        scale = 1.0 if spec_rms <= limit else max(0.0, min(1.0, limit / max(1e-8, spec_rms)))
        if scale < 0.999:
            predicted.sub_(current).mul_(float(scale)).add_(current)
        ratio = float(spec_rms / max(1e-8, float(self.predictor_beta) * lin_rms))
        return predicted, float(scale), ratio

    def _run_tail(self, snapshot_cpu, t_emb, mod_segments, rope_freqs, transformer_options, step,
                  predictor_history=None, predictor_mode="stale", predictor_beta=0.75,
                  spectral_mix=None, target_coordinate=None, gain_history=None,
                  adaptive_gain=None, phase_gain=None):
        import torch
        import comfy.model_management
        import comfy.model_prefetch

        if snapshot_cpu is None:
            raise RuntimeError("H3VM Dev9 tail island received no host snapshot")
        worker_started = time.perf_counter()
        predictor_ms = 0.0
        predictor_used = False
        predictor_kind = "stale"
        predictor_weights = None
        predictor_history_used = None
        predictor_guard_scale = 1.0
        predictor_delta_ratio = None
        control_reason = "stale"
        tail_snapshot = snapshot_cpu
        history = list(predictor_history or ())
        gain_history = list(gain_history or ())
        previous = history[-2][1] if len(history) >= 2 else None
        target_coordinate = float(step if target_coordinate is None else target_coordinate)
        adaptive_gain = float(self._gain_ema if adaptive_gain is None else adaptive_gain)
        adaptive_gain = max(0.0, min(1.25, adaptive_gain))
        phase_gain = float(self._phase_gain if phase_gain is None else phase_gain)
        phase_gain = max(0.0, min(1.25, phase_gain))

        # Candidate 1: LAB1/LAB2 direct full-boundary spectral forecast.
        spectral_usable, spectral_weights = self._spectral_weights(history, target_coordinate)
        spectral_available = bool(spectral_weights is not None and previous is not None)

        # Candidate 2: scheduler-aware local secant with an online scalar gain.
        adaptive_usable, adaptive_weights, coord_ratio = _two_point_weights(
            history, target_coordinate, adaptive_gain,
        )
        adaptive_available = bool(adaptive_weights is not None and previous is not None)

        # Candidate 2b: LAB6 phase-aware local secant. Same two real anchors,
        # but with an asymmetric fast-brake gain controlled by diffusion phase.
        phase_usable, phase_weights, phase_coord_ratio = _two_point_weights(
            history, target_coordinate, phase_gain,
        )
        phase_available = bool(phase_weights is not None and previous is not None)

        # LAB6 SAFE_BLEND: preserve the stale manifold anchor and inject only a
        # small scheduler-aware residual.  This is intentionally conservative:
        # it uses 20% of the current phase gain and never replaces the base tensor.
        safe_gain = max(0.0, min(1.25, float(phase_gain) * 0.20))
        safe_usable, safe_weights, safe_coord_ratio = _two_point_weights(
            history, target_coordinate, safe_gain,
        )
        safe_available = bool(safe_weights is not None and previous is not None)

        # Candidate 3: Spectrum on the CPU control plane only. Forecast the tiny
        # scalar gain, not the feature tensor. Clamp it near the online EMA so a
        # polynomial wobble cannot suddenly throw the boundary across feature space.
        sgain_value, sgain_control_weights = _forecast_scalar_spectral(
            gain_history, target_coordinate,
            degree=min(2, self.spectral_degree),
            ridge=max(0.02, float(self.spectral_ridge)),
            lo=0.0, hi=1.25,
        )
        if sgain_value is not None:
            sgain_value = max(
                0.0,
                min(1.25, max(adaptive_gain - 0.35, min(adaptive_gain + 0.35, float(sgain_value)))),
            )
        sgain_usable, sgain_weights, _ = _two_point_weights(
            history, target_coordinate,
            adaptive_gain if sgain_value is None else sgain_value,
        )
        sgain_available = bool(
            sgain_value is not None and sgain_weights is not None and previous is not None
        )

        # Candidate 2c / LAB5: Spectrum is allowed to touch only the brake pedal.
        # It may reduce the causal phase gain in the late expanding-schedule region,
        # but it can never increase gain.  This uses the one place LAB4 showed
        # scalar Spectrum was useful: anticipating the gain collapse by one step.
        brake_gain = float(phase_gain)
        brake_active = False
        brake_advice = None
        if sgain_value is not None:
            brake_advice = float(sgain_value)
            late_ratio = phase_coord_ratio is not None and float(phase_coord_ratio) >= 1.20
            collapse_advice = float(sgain_value) <= 0.15
            if late_ratio and collapse_advice and float(sgain_value) < brake_gain:
                brake_gain = min(brake_gain, max(0.0, float(sgain_value) + 0.05))
                brake_active = brake_gain < float(phase_gain) - 1e-9
        brake_usable, brake_weights, brake_coord_ratio = _two_point_weights(
            history, target_coordinate, brake_gain,
        )
        brake_available = bool(brake_weights is not None and previous is not None)

        availability = {
            "adaptive": adaptive_available,
            "phase": phase_available,
            "safe_blend": safe_available,
            "brake": brake_available,
            "sgain": sgain_available,
            "spectral": spectral_available,
        }
        effective_mode, control_reason = self._effective_mode(predictor_mode, availability)

        if previous is not None and tuple(previous.shape) == tuple(snapshot_cpu.shape):
            if effective_mode == "linear":
                if self._pinned is not None:
                    tail_snapshot, predictor_ms = self._pinned.predict(
                        snapshot_cpu, previous, float(predictor_beta), key="predict_linear"
                    )
                else:
                    pred0 = time.perf_counter()
                    tail_snapshot = torch.add(snapshot_cpu, snapshot_cpu, alpha=float(predictor_beta))
                    tail_snapshot.add_(previous, alpha=-float(predictor_beta))
                    predictor_ms = (time.perf_counter() - pred0) * 1000.0
                predictor_used = True
                predictor_history_used = history[-2:]
                predictor_weights = [-float(predictor_beta), 1.0 + float(predictor_beta)]
                if predictor_mode == "linear_raw":
                    predictor_kind = "linear-raw"
                elif "warmup" in control_reason:
                    predictor_kind = "linear-warmup"
                elif "gate-closed" in control_reason or "mix-zero" in control_reason:
                    predictor_kind = "linear-gated"
                elif predictor_mode in ("auto_brake", "auto_phase", "auto_legacy", "auto"):
                    predictor_kind = "auto-linear"
                else:
                    predictor_kind = "linear"

            elif effective_mode in ("adaptive", "phase", "safe_blend", "brake", "sgain"):
                if effective_mode == "adaptive":
                    use_hist, use_weights, pkey = adaptive_usable, adaptive_weights, "predict_adaptive"
                elif effective_mode == "phase":
                    use_hist, use_weights, pkey = phase_usable, phase_weights, "predict_phase"
                elif effective_mode == "safe_blend":
                    use_hist, use_weights, pkey = safe_usable, safe_weights, "predict_safe_blend"
                elif effective_mode == "brake":
                    use_hist, use_weights, pkey = brake_usable, brake_weights, "predict_brake"
                else:
                    use_hist, use_weights, pkey = sgain_usable, sgain_weights, "predict_sgain"
                if use_weights is not None:
                    tail_snapshot, predictor_ms = self._materialize_weighted_prediction(
                        use_hist, use_weights, key=pkey,
                    )
                    predictor_used = True
                    predictor_history_used = use_hist
                    predictor_weights = list(use_weights)
                    if effective_mode == "adaptive":
                        predictor_kind = "auto-adaptive" if predictor_mode in ("auto_brake", "auto_phase", "auto_legacy", "auto") else "adaptive-secant"
                    elif effective_mode == "phase":
                        if predictor_mode == "phase_raw":
                            predictor_kind = "phase-raw"
                        else:
                            predictor_kind = "auto-phase-adaptive" if predictor_mode in ("auto_phase", "auto_brake", "auto") else "phase-adaptive"
                    elif effective_mode == "safe_blend":
                        predictor_kind = "safe-blend"
                    elif effective_mode == "brake":
                        predictor_kind = "auto-brake" if predictor_mode in ("auto_brake", "auto") else "brake-phase"
                    else:
                        predictor_kind = "auto-spectral-gain" if predictor_mode in ("auto_phase", "auto_brake", "auto_legacy", "auto") else "spectral-gain"

            elif effective_mode == "spectral_raw" and spectral_weights is not None:
                # QUALITY TRUTH: no confidence gate, no delta guard, no hidden
                # linear fallback once enough anchors exist.  This is the actual
                # full-boundary spectral A/B requested by the UI label.
                tail_snapshot, predictor_ms = self._materialize_weighted_prediction(
                    spectral_usable, spectral_weights, key="predict_spectral_raw"
                )
                predictor_history_used = spectral_usable
                predictor_weights = list(spectral_weights)
                predictor_kind = "spectral-raw"
                predictor_used = True

            elif effective_mode in ("spectral", "hybrid") and spectral_weights is not None:
                tail_snapshot, predictor_ms = self._materialize_weighted_prediction(
                    spectral_usable, spectral_weights, key="predict_spectral_gated"
                )
                tail_snapshot, predictor_guard_scale, predictor_delta_ratio = self._apply_delta_guard(
                    tail_snapshot, snapshot_cpu, previous
                )
                predictor_history_used = spectral_usable
                predictor_weights = list(spectral_weights)
                if effective_mode == "hybrid":
                    mix = self._adaptive_spectral_mix if spectral_mix is None else float(spectral_mix)
                    mix = max(0.0, min(1.0, mix))
                    tail_snapshot.mul_(mix)
                    tail_snapshot.add_(snapshot_cpu, alpha=(1.0 - mix) * (1.0 + float(predictor_beta)))
                    tail_snapshot.add_(previous, alpha=-(1.0 - mix) * float(predictor_beta))
                    predictor_kind = "hybrid-gated"
                else:
                    predictor_kind = "spectral-gated"
                predictor_used = True

            elif effective_mode == "stale":
                if predictor_mode == "stale_raw":
                    predictor_kind = "stale-raw"
                elif predictor_mode in ("auto_brake", "auto_phase", "auto_legacy", "auto"):
                    predictor_kind = "auto-stale"
                else:
                    predictor_kind = "stale"

        queue = self._make_prefetch(
            self.tail_blocks, self.primary_device, transformer_options,
            self.tail_prefetch, "tail",
        )
        with comfy.model_management.cuda_device_context(self.primary_device):
            read0 = time.perf_counter()
            if self._pinned is not None:
                h = self._pinned.to_device(tail_snapshot, self.primary_device)
            else:
                h = self.space.transport.move_tensor(
                    tail_snapshot, self.primary_device, mode="neutral_pageable"
                )
            snapshot_read_ms = (time.perf_counter() - read0) * 1000.0
            compute0 = time.perf_counter()
            for i in range(self.first_tail, self.last_tail + 1):
                block = self.tail_blocks[i]
                if queue is not None:
                    comfy.model_prefetch.prefetch_queue_pop(queue, self.primary_device, block)
                h = block(
                    h, t_emb, mod_segments, rope_freqs,
                    transformer_options=transformer_options or {},
                )
            self._finish_prefetch(queue, self.primary_device, "tail")
            torch.cuda.synchronize(self.primary_device)
            tail_compute_ms = (time.perf_counter() - compute0) * 1000.0
        return {
            "h": h,
            "snapshot_read_ms": snapshot_read_ms,
            "predictor_ms": predictor_ms,
            "predictor_used": predictor_used,
            "predictor_kind": predictor_kind,
            "predictor_requested": str(predictor_mode),
            "predictor_effective": effective_mode,
            "control_reason": control_reason,
            "predictor_weights": predictor_weights,
            "predictor_history": predictor_history_used,
            "predictor_guard_scale": predictor_guard_scale,
            "predictor_delta_ratio": predictor_delta_ratio,
            "predicted_snapshot_cpu": tail_snapshot if predictor_used else None,
            "target_coordinate": float(target_coordinate),
            "coordinate_source": str(self._coordinate_source or "step_index"),
            "adaptive_gain_used": float(adaptive_gain),
            "phase_gain_used": float(phase_gain),
            "coord_ratio": coord_ratio,
            "phase_coord_ratio": phase_coord_ratio,
            "adaptive_history": adaptive_usable,
            "adaptive_weights": list(adaptive_weights) if adaptive_weights is not None else None,
            "phase_history": phase_usable,
            "phase_weights": list(phase_weights) if phase_weights is not None else None,
            "safe_gain_used": float(safe_gain),
            "safe_coord_ratio": safe_coord_ratio,
            "safe_history": safe_usable,
            "safe_weights": list(safe_weights) if safe_weights is not None else None,
            "brake_gain_used": float(brake_gain),
            "brake_active": bool(brake_active),
            "brake_advice": brake_advice,
            "brake_coord_ratio": brake_coord_ratio,
            "brake_history": brake_usable,
            "brake_weights": list(brake_weights) if brake_weights is not None else None,
            "sgain_value": sgain_value,
            "sgain_control_weights": sgain_control_weights,
            "sgain_history": sgain_usable,
            "sgain_weights": list(sgain_weights) if sgain_weights is not None and sgain_available else None,
            "spectral_history_shadow": spectral_usable,
            "spectral_weights_shadow": list(spectral_weights) if spectral_weights is not None else None,
            "tail_compute_ms": tail_compute_ms,
            "tail_wall_ms": (time.perf_counter() - worker_started) * 1000.0,
            "step": int(step),
        }

    def _reset_call_state(self):
        self._pending_snapshot_cpu = None
        self._future = None
        self._tail_output = None
        self._prefix_t_emb = None
        self._prefix_segments = None
        self._prefix_rope = None
        self._prefix_queue = None
        self._prefix_started = None
        self._prefix_compute_started = None
        self._step_started = None
        self._step_stats0 = None
        self._snapshot_write_ms = 0.0
        self._stage_prefix_ms = 0.0
        self._predictor_used = False

    def _reset_sampling_run(self):
        self._snapshot_cpu = None
        self._snapshot_prev_cpu = None
        self._snapshot_history = []
        self._captured_timestep = None
        self._current_coordinate = None
        self._coordinate_source = None
        self._error_ema = {"stale": None, "linear": None, "adaptive": None, "phase": None, "brake": None, "sgain": None, "spectral": None}
        self._error_count = {"stale": 0, "linear": 0, "adaptive": 0, "phase": 0, "brake": 0, "sgain": 0, "spectral": 0}
        self._diag_sum = {"used": 0.0, "stale": 0.0, "linear": 0.0, "adaptive": 0.0, "phase": 0.0, "brake": 0.0, "sgain": 0.0, "spectral": 0.0}
        self._diag_count = {"used": 0, "stale": 0, "linear": 0, "adaptive": 0, "phase": 0, "brake": 0, "sgain": 0, "spectral": 0}
        self._winner_counts = {"stale": 0, "linear": 0, "adaptive": 0, "phase": 0, "brake": 0, "sgain": 0, "spectral": 0}
        self._quality_sum = {"rel_l2": 0.0, "cos": 0.0, "norm_ratio": 0.0, "mean_shift": 0.0, "std_ratio": 0.0, "delta_cos": 0.0}
        self._quality_count = {k: 0 for k in self._quality_sum}
        self._coordinate_warned = False
        self._spectral_gate_open = self.spectral_confidence == "off"
        self._adaptive_spectral_mix = self.spectral_mix if self._spectral_gate_open else 0.0
        self._gain_ema = max(0.0, min(1.25, float(self.predictor_beta)))
        self._gain_history = []
        self._gain_fit_last = None
        self._phase_gain = max(0.0, min(1.25, float(self.predictor_beta)))
        self._phase_state = "warmup"
        self._phase_low_streak = 0
        self._phase_high_streak = 0
        self._phase_regret_streak = 0
        self._phase_switches = 0
        self._auto_choice = "linear"
        self._last_control_signature = None
        self._step = 0
        self._reset_call_state()

    def sampling_begin(self):
        """Start one ComfyUI OUTER_SAMPLE run with a clean Mode4 mailbox."""
        self._reset_sampling_run()

    def sampling_end(self):
        """End one OUTER_SAMPLE run; never let hint-sized state cross prompts."""
        try:
            if self._step > 0:
                try:
                    self._log_predictor_summary()
                except Exception as exc:
                    # Summary/telemetry is observational only and must never turn
                    # an otherwise completed sampler run into a failure.
                    LOG.warning("H3VM predictor summary unavailable: %r", exc)
        finally:
            self._reset_sampling_run()

    def _begin_prefix(self, h, t_emb, mod_segments, rope_freqs, transformer_options):
        import torch
        self._step += 1
        step = self._step
        self._step_started = time.perf_counter()
        self._step_stats0 = self._stats_snapshot(self.space)
        self._peak_primary = self._peak_secondary = 0.0
        self._exact_this_step = should_use_exact_step(
            step, self.expected_steps, self.refresh_interval, self.exact_last_step, self.exact_steps
        )
        target_coordinate, coordinate_source = self._coordinate_for_step(step)

        # Shape changes imply a new logical sampling run even if expected_steps was
        # misconfigured. Never feed a stale mailbox across incompatible shapes.
        if self._snapshot_cpu is not None and tuple(self._snapshot_cpu.shape) != tuple(h.shape):
            LOG.warning(
                "H3VM Dev9 snapshot shape changed %s -> %s; resetting stale mailbox",
                tuple(self._snapshot_cpu.shape), tuple(h.shape),
            )
            self._snapshot_cpu = None
            self._snapshot_prev_cpu = None
            self._snapshot_history = []
            self._gain_history = []
            self._gain_ema = max(0.0, min(1.25, float(self.predictor_beta)))
            self._gain_fit_last = None
            self._phase_gain = max(0.0, min(1.25, float(self.predictor_beta)))
            self._phase_state = "warmup"
            self._phase_low_streak = 0
            self._phase_high_streak = 0
            self._phase_regret_streak = 0
            self._phase_switches = 0
            self._weight_cache.clear()
            self._spectral_gate_open = self.spectral_confidence == "off"
            self._adaptive_spectral_mix = self.spectral_mix if self._spectral_gate_open else 0.0
            self._auto_choice = "linear"
            self._step = 1
            step = 1
            target_coordinate, coordinate_source = self._coordinate_for_step(step)
            self._exact_this_step = True

        can_pipeline = (not self._exact_this_step and self._snapshot_cpu is not None)

        # Full-throttle mode launches the primary tail *before* staging the
        # current prefix to the secondary island. The tail only consumes the
        # previous host snapshot, so it has no dependency on this staging work.
        # This overlaps GPU0 tail startup with GPU0->RAM->GPU1 input staging.
        if can_pipeline and self.launch_tail_before_stage:
            self._future = self._executor.submit(
                self._run_tail,
                self._snapshot_cpu,
                t_emb,
                mod_segments,
                rope_freqs,
                dict(transformer_options or {}),
                step,
                list(self._snapshot_history),
                self.predictor_mode,
                self.predictor_beta,
                self._adaptive_spectral_mix,
                target_coordinate,
                list(self._gain_history),
                float(self._gain_ema),
                float(self._phase_gain),
            )
        else:
            self._future = None

        # Keep ComfyUI's DynamicVRAM prefetch queue unmodified until the first
        # real block call. Dev9.3's extra early queue pop was unsafe because it
        # consumed one queue entry too many and could poison CUDA before flush.
        self._prefix_queue = self._make_prefetch(
            self.prefix_blocks, self.secondary_device, transformer_options,
            self.prefix_prefetch, "prefix",
        )

        # Current prefix input and its tiny read-only metadata enter the
        # secondary island through host memory. There is still no GPU-to-GPU
        # tensor handoff.
        stage0 = time.perf_counter()
        h2 = self._host_only_tree(h, self.secondary_device, feeder_key="stage_h")
        self._prefix_t_emb = self._host_only_tree(t_emb, self.secondary_device)
        self._prefix_segments = self._host_only_tree(mod_segments, self.secondary_device)
        self._prefix_rope = self._host_only_tree(rope_freqs, self.secondary_device)
        self._stage_prefix_ms = (time.perf_counter() - stage0) * 1000.0

        if can_pipeline and self._future is None:
            self._future = self._executor.submit(
                self._run_tail,
                self._snapshot_cpu,
                t_emb,
                mod_segments,
                rope_freqs,
                dict(transformer_options or {}),
                step,
                list(self._snapshot_history),
                self.predictor_mode,
                self.predictor_beta,
                self._adaptive_spectral_mix,
                target_coordinate,
                list(self._gain_history),
                float(self._gain_ema),
                float(self._phase_gain),
            )
        self._prefix_started = time.perf_counter()
        self._prefix_compute_started = self._prefix_started
        self._sample_memory()
        return h2

    def prefix_execute(self, index, h, t_emb, mod_segments, rope_freqs, transformer_options=None):
        import torch
        import comfy.model_management
        import comfy.model_prefetch

        if index == self.first_prefix:
            h = self._begin_prefix(h, t_emb, mod_segments, rope_freqs, transformer_options)
        elif self._prefix_t_emb is None:
            h = self._begin_prefix(h, t_emb, mod_segments, rope_freqs, transformer_options)
        elif getattr(h, "device", None) != self.secondary_device:
            h = self._host_only_tree(h, self.secondary_device)

        block = self.prefix_blocks[index]
        with comfy.model_management.cuda_device_context(self.secondary_device):
            if self._prefix_queue is not None:
                comfy.model_prefetch.prefetch_queue_pop(self._prefix_queue, self.secondary_device, block)
            h = block(
                h, self._prefix_t_emb, self._prefix_segments, self._prefix_rope,
                transformer_options=transformer_options or {},
            )

        if index == self.last_prefix:
            self._finish_prefetch(self._prefix_queue, self.secondary_device, "prefix")
            torch.cuda.synchronize(self.secondary_device)
            self._prefix_compute_ms = (time.perf_counter() - self._prefix_compute_started) * 1000.0
            snap0 = time.perf_counter()
            if self._pinned is not None:
                ring_slots = max(3, self.spectral_history + 1)
                snap_key = f"snapshot_{self._step % ring_slots}"
                self._pending_snapshot_cpu = self._pinned.to_host(h, snap_key)
            else:
                self._pending_snapshot_cpu = self.space.transport.move_tensor(
                    h, "cpu", mode="neutral_pageable"
                ).contiguous()
            self._snapshot_write_ms = (time.perf_counter() - snap0) * 1000.0
            self._sample_memory()
        return h

    def _consume_tail(self, t_emb, mod_segments, rope_freqs, transformer_options):
        import torch
        step = self._step
        wait0 = time.perf_counter()
        if self._future is not None:
            result = self._future.result()
        else:
            # Exact warm-up/refresh/flush: consume the current prefix snapshot.
            result = self._run_tail(
                self._pending_snapshot_cpu,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options or {},
                step,
                [],
                "stale",
                self.predictor_beta,
                self._adaptive_spectral_mix,
                self._current_coordinate if self._current_coordinate is not None else float(step),
                list(self._gain_history),
                float(self._gain_ema),
                float(self._phase_gain),
            )
        wait_ms = (time.perf_counter() - wait0) * 1000.0
        self._tail_output = result["h"]
        self._sample_memory()

        s1 = self._stats_snapshot(self.space)
        s0 = self._step_stats0 or (0, 0, 0.0, 0, 0)
        moves = s1[0] - s0[0]
        moved = s1[1] - s0[1]
        transport_ms = max(0.0, s1[2] - s0[2]) * 1000.0
        direct_moves = s1[3] - s0[3]
        direct_bytes = s1[4] - s0[4]
        step_wall_ms = (time.perf_counter() - self._step_started) * 1000.0 if self._step_started else 0.0
        theoretical_serial = float(getattr(self, "_prefix_compute_ms", 0.0)) + float(result["tail_compute_ms"])
        overlap_factor = theoretical_serial / max(1e-9, step_wall_ms - self._stage_prefix_ms)

        predictor_used = bool(result.get("predictor_used", False))
        predictor_kind = str(result.get("predictor_kind", "stale"))
        diag_used = diag_stale = diag_linear = diag_adaptive = diag_phase = diag_brake = diag_sgain = diag_spectral = None
        quality_used = None
        gain_fit = gain_fit_ratio = gain_fit_err = None
        if (not self._exact_this_step) and self._pending_snapshot_cpu is not None:
            try:
                truth = self._pending_snapshot_cpu
                if self._snapshot_cpu is not None:
                    diag_stale = _relative_rmse(self._snapshot_cpu, truth)

                if len(self._snapshot_history) >= 2:
                    hist2 = list(self._snapshot_history)[-2:]
                    diag_linear = _weighted_sample_error(
                        hist2,
                        [-float(self.predictor_beta), 1.0 + float(self.predictor_beta)],
                        truth,
                    )

                ah = result.get("adaptive_history")
                aw = result.get("adaptive_weights")
                if ah and aw:
                    diag_adaptive = _weighted_sample_error(ah, aw, truth)

                ph = result.get("phase_history")
                pw = result.get("phase_weights")
                if ph and pw:
                    diag_phase = _weighted_sample_error(ph, pw, truth)

                bh = result.get("brake_history")
                bw = result.get("brake_weights")
                if bh and bw:
                    diag_brake = _weighted_sample_error(bh, bw, truth)

                gh = result.get("sgain_history")
                gw = result.get("sgain_weights")
                if gh and gw:
                    diag_sgain = _weighted_sample_error(gh, gw, truth)

                sh = result.get("spectral_history_shadow")
                sw = result.get("spectral_weights_shadow")
                if sh and sw:
                    diag_spectral = _weighted_sample_error(sh, sw, truth)

                used_snapshot = result.get("predicted_snapshot_cpu") if predictor_used else self._snapshot_cpu
                if predictor_used:
                    diag_used = _relative_rmse(used_snapshot, truth)
                else:
                    diag_used = diag_stale
                quality_used = _quality_truth_metrics(used_snapshot, truth, current=self._snapshot_cpu)
                if quality_used:
                    for qname, qvalue in quality_used.items():
                        if qvalue is not None:
                            self._quality_sum[qname] += float(qvalue)
                            self._quality_count[qname] += 1

                # Fit the scalar gain only after current GPU1 truth arrives. This
                # value is never allowed to retroactively affect the current tail;
                # it becomes evidence for the next step.
                gain_fit, gain_fit_ratio, gain_fit_err = _fit_transition_gain(
                    self._snapshot_history,
                    float(result.get("target_coordinate", self._current_coordinate or step)),
                    truth,
                )
                if gain_fit is not None:
                    self._gain_fit_last = float(gain_fit)
                    self._gain_ema = 0.65 * float(self._gain_ema) + 0.35 * float(gain_fit)
                    self._gain_ema = max(0.0, min(1.25, float(self._gain_ema)))
                    self._gain_history.append((
                        float(result.get("target_coordinate", self._current_coordinate or step)),
                        float(gain_fit),
                    ))
                    if len(self._gain_history) > 6:
                        self._gain_history = self._gain_history[-6:]

                self._accumulate_diag(
                    diag_used, diag_stale, diag_linear, diag_adaptive, diag_phase, diag_brake, diag_sgain, diag_spectral,
                )
                self._update_control(
                    diag_stale, diag_linear, diag_adaptive, diag_phase, diag_brake, diag_sgain, diag_spectral,
                )
                self._update_phase_control(gain_fit, diag_stale, diag_brake if self.predictor_mode == "auto_brake" else diag_phase)
            except Exception as exc:
                LOG.debug("H3VM Predictor-V6 diagnostics/control skipped: %r", exc)
        mode_name = "exact" if self._exact_this_step else (f"{predictor_kind}-pipeline" if predictor_used else "stale-pipeline")
        if self.telemetry and (step <= 3 or step % 10 == 0 or (self.expected_steps and step == self.expected_steps)):
            LOG.info(
                "H3VM Dev10.1 QUALITY step #%d | profile=%s mode=%s stale_age=%d | prefix=%.1fms tail=%.1fms "
                "bubble=%.1fms tail_wait=%.1fms overlap=%.2fx | host stage=%.1fms predictor=%.1fms beta=%.2f "
                "snapshot write/read=%.1f/%.1fms | host moves=%d payload=%.1fMiB transport=%.1fms | "
                "feeder=%s pinned=%.1fMiB | D2D moves=%d bytes=%.1fMiB | peak %s=%.2fGiB %s=%.2fGiB",
                step,
                self.quality_profile,
                mode_name,
                0 if self._exact_this_step else 1,
                float(getattr(self, "_prefix_compute_ms", 0.0)),
                float(result["tail_compute_ms"]),
                max(0.0, float(result["tail_compute_ms"]) - float(getattr(self, "_prefix_compute_ms", 0.0))),
                wait_ms,
                overlap_factor,
                self._stage_prefix_ms,
                float(result.get("predictor_ms", 0.0)),
                self.predictor_beta,
                self._snapshot_write_ms,
                float(result["snapshot_read_ms"]),
                moves, moved/(1024**2), transport_ms,
                self.host_feeder,
                float(self._pinned.allocated_mib if self._pinned is not None else 0.0),
                direct_moves, direct_bytes/(1024**2),
                self.primary_device, self._peak_primary,
                self.secondary_device, self._peak_secondary,
            )
        if self.telemetry and self.spectral_debug == "full" and (not self._exact_this_step) and quality_used:
            LOG.info(
                "H3VM PRODUCTION-QUALITY LAB7 step #%d | actual=%s | rel_l2=%.6f cos=%.6f norm_ratio=%.6f mean_shift=%.6f std_ratio=%.6f delta_cos=%s",
                step, predictor_kind,
                float(quality_used["rel_l2"]), float(quality_used["cos"]),
                float(quality_used["norm_ratio"]), float(quality_used["mean_shift"]),
                float(quality_used["std_ratio"]),
                "%.6f" % float(quality_used["delta_cos"]) if quality_used.get("delta_cos") is not None else "n/a",
            )
        should_log_predictor = (
            self.telemetry and self.spectral_debug == "full" and (not self._exact_this_step)
        )
        if should_log_predictor:
            LOG.info(
                "H3VM PREDICTOR-V7 LAB7 step #%d | requested=%s effective=%s kind=%s coord=%s:%.7g reason=%s | "
                "err used/stale/linear/adaptive/phase/brake/sgain/spectral=%s/%s/%s/%s/%s/%s/%s/%s | "
                "ema stale/linear/adaptive/phase/brake/sgain/spectral=%s/%s/%s/%s/%s/%s/%s | "
                "gain legacy=%.3f phase=%.3f brake=%.3f active=%s advice=%s fit=%s ema=%.3f sgain=%s ratio=%s fit_err=%s | "
                "auto_legacy=%s phase_state=%s low/high/regret=%d/%d/%d switches=%d direct_gate=%s spec_guard=%.3f spec_delta=%s",
                step,
                result.get("predictor_requested", self.predictor_mode),
                result.get("predictor_effective", "stale"),
                predictor_kind,
                result.get("coordinate_source", self._coordinate_source or "step_index"),
                float(result.get("target_coordinate", step)),
                result.get("control_reason", "n/a"),
                "%.6f" % diag_used if diag_used is not None else "n/a",
                "%.6f" % diag_stale if diag_stale is not None else "n/a",
                "%.6f" % diag_linear if diag_linear is not None else "n/a",
                "%.6f" % diag_adaptive if diag_adaptive is not None else "n/a",
                "%.6f" % diag_phase if diag_phase is not None else "n/a",
                "%.6f" % diag_brake if diag_brake is not None else "n/a",
                "%.6f" % diag_sgain if diag_sgain is not None else "n/a",
                "%.6f" % diag_spectral if diag_spectral is not None else "n/a",
                "%.6f" % self._error_ema["stale"] if self._error_ema["stale"] is not None else "n/a",
                "%.6f" % self._error_ema["linear"] if self._error_ema["linear"] is not None else "n/a",
                "%.6f" % self._error_ema["adaptive"] if self._error_ema["adaptive"] is not None else "n/a",
                "%.6f" % self._error_ema["phase"] if self._error_ema["phase"] is not None else "n/a",
                "%.6f" % self._error_ema["brake"] if self._error_ema["brake"] is not None else "n/a",
                "%.6f" % self._error_ema["sgain"] if self._error_ema["sgain"] is not None else "n/a",
                "%.6f" % self._error_ema["spectral"] if self._error_ema["spectral"] is not None else "n/a",
                float(result.get("adaptive_gain_used", self._gain_ema)),
                float(result.get("phase_gain_used", self._phase_gain)),
                float(result.get("brake_gain_used", result.get("phase_gain_used", self._phase_gain))),
                "YES" if result.get("brake_active", False) else "NO",
                "%.3f" % float(result.get("brake_advice")) if result.get("brake_advice") is not None else "n/a",
                "%.3f" % float(gain_fit) if gain_fit is not None else "n/a",
                float(self._gain_ema),
                "%.3f" % float(result.get("sgain_value")) if result.get("sgain_value") is not None else "n/a",
                "%.3f" % float(result.get("coord_ratio")) if result.get("coord_ratio") is not None else "n/a",
                "%.6f" % float(gain_fit_err) if gain_fit_err is not None else "n/a",
                self._auto_choice,
                self._phase_state, int(self._phase_low_streak), int(self._phase_high_streak),
                int(self._phase_regret_streak), int(self._phase_switches),
                "OPEN" if self._spectral_gate_open else "CLOSED",
                float(result.get("predictor_guard_scale", 1.0)),
                "%.3f" % float(result.get("predictor_delta_ratio")) if result.get("predictor_delta_ratio") is not None else "n/a",
            )
        return self._tail_output

    def tail_execute(self, index, h, t_emb, mod_segments, rope_freqs, transformer_options=None):
        if index == self.first_tail:
            h = self._consume_tail(t_emb, mod_segments, rope_freqs, transformer_options)
        else:
            h = self._tail_output if self._tail_output is not None else h

        if index == self.last_tail:
            # Publish the current secondary prefix snapshot only after the current
            # tail has consumed the prior version. Next call sees exactly age=1.
            self._snapshot_prev_cpu = self._snapshot_cpu
            self._snapshot_cpu = self._pending_snapshot_cpu
            if self._snapshot_cpu is not None and self._current_coordinate is not None:
                self._snapshot_history.append((float(self._current_coordinate), self._snapshot_cpu))
                keep = max(2, self.spectral_history)
                if len(self._snapshot_history) > keep:
                    self._snapshot_history = self._snapshot_history[-keep:]
            # expected_steps is a scheduling/quality hint only. OUTER_SAMPLE owns
            # the real run boundary so a wrong hint can never reset state mid-run
            # or leak stale predictor state into the next prompt.
            self._reset_call_state()
        return h

    def close(self):
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._snapshot_cpu = None
        self._snapshot_prev_cpu = None
        self._snapshot_history = []
        self._pending_snapshot_cpu = None
        self._future = None
        self._tail_output = None
        self._prefix_queue = None
        if self._pinned is not None:
            self._pinned.clear()

    def __del__(self):
        self.close()


class PrefixProxyFactory:
    @staticmethod
    def make(runtime, index):
        import torch
        import weakref

        class H3SnapshotPrefixProxy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._runtime_ref = weakref.ref(runtime)
                self.index = int(index)

            def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
                runtime_obj = self._runtime_ref()
                if runtime_obj is None:
                    raise RuntimeError("H3VM snapshot runtime has been released")
                from .compiler_guard import pause_comfy_allocation_graph
                with pause_comfy_allocation_graph():
                    return runtime_obj.prefix_execute(
                        self.index, x, t_emb, mod_segments, rope_freqs,
                        transformer_options=transformer_options,
                    )

        return H3SnapshotPrefixProxy()


class TailProxyFactory:
    @staticmethod
    def make(runtime, index):
        import torch
        import weakref

        class H3SnapshotTailProxy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._runtime_ref = weakref.ref(runtime)
                self.index = int(index)

            def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
                runtime_obj = self._runtime_ref()
                if runtime_obj is None:
                    raise RuntimeError("H3VM snapshot runtime has been released")
                from .compiler_guard import pause_comfy_allocation_graph
                with pause_comfy_allocation_graph():
                    return runtime_obj.tail_execute(
                        self.index, x, t_emb, mod_segments, rope_freqs,
                        transformer_options=transformer_options,
                    )

        return H3SnapshotTailProxy()


def make_island_root(
    source_base_model,
    source_dm,
    block_map,
    total_blocks,
    label,
    workspace_reserve_bytes=0,
):
    import torch

    class EmptySlot(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError(f"H3VM Dev9 empty {label} island slot executed")

    class IslandDiffusionView(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([
                block_map[i] if i in block_map else EmptySlot()
                for i in range(total_blocks)
            ])
            for name in ("hidden_size", "sigma_shift_video", "sigma_shift_audio", "use_adaln_curves", "dtype"):
                if hasattr(source_dm, name):
                    setattr(self, name, getattr(source_dm, name))

    class H3IslandRoot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = IslandDiffusionView()
            self.island_label = str(label)
            for name in ("manual_cast_dtype", "model_dtype", "dtype"):
                if hasattr(source_base_model, name):
                    setattr(self, name, getattr(source_base_model, name))

        def memory_required(self, input_shape=None):
            del input_shape
            return int(workspace_reserve_bytes)

    return H3IslandRoot()
