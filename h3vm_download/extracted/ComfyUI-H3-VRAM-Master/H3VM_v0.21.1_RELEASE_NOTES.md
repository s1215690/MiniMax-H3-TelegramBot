# H3VM v0.21.1 Release Notes

## Focused public release

The integrated Master Loader remains removed. The public surface is two infrastructure nodes:

1. `H3VM Core` for MODEL execution / VRAM / multi-GPU scheduling and unified acceleration controls.
2. `H3VM Video VAE` for optional MiniMax H3 Video VAE decoding.

Changes:

- Moved the **Dual-GPU VAE** switch into the H3VM Core unified panel.
- Added Core output `H3VM_VAE_POLICY`; the Video VAE node consumes this policy and no longer duplicates the switch.
- VAE OFF uses native single-GPU `vae.decode()`; single-GPU Core mode also forces native VAE decode.
- **Mode4 run-boundary hardening:** real sampling begin/end is now owned by the ComfyUI `OUTER_SAMPLE` lifecycle. `expected_steps_hint` is scheduling guidance only and can no longer end/reset a run mid-sample or leak stale mailbox/predictor state across prompts.
- **Predictor startup cleanup:** an initially unavailable real timestep no longer commits a temporary step-index coordinate source, avoiding the `step_index -> timestep` history-reset warning pair on normal startup.
- **Dual VAE identity guard:** secondary Video VAE weights are lightly fingerprinted against the connected primary VAE when comparable; obvious mismatches fail clearly instead of being silently combined.
- Restored the bundled reference workflow to the standard 20-step MiniMax H3 configuration and aligned the H3VM steps hint to 20; also removed a stale serialized VAE switch field.
- Restored Chinese/English labels for inputs and outputs with a label-only frontend extension.
- Master Loader and ownership of prompt/duration/resolution/seed/sampler remain removed.
- Runtime patches remain lazy; standard ComfyUI CLIP/VAE loaders are not globally replaced.
- Static/API compatibility baseline documented for ComfyUI v0.35.0.
