# Changelog

## v0.21.1

- Focused public surface: H3VM Core + H3VM Video VAE.
- Removed the integrated Master Loader.
- Runtime activation is lazy; plugin discovery does not globally patch standard CLIP/VAE loaders.
- Unified the Dual-GPU VAE switch into the H3VM Core panel.
- Video VAE now consumes Core `vae_policy` instead of owning a duplicate switch.
- Dual-GPU VAE can be disabled independently for asymmetric GPU pairs.
- Chinese/English label-only UI restored without widget reordering or serialization overrides.
- Mode4 sampling boundaries now follow `OUTER_SAMPLE`; steps hint is no longer a correctness/reset boundary.
- Predictor startup avoids committing a temporary step-index coordinate source while real timestep is not yet available.
- Dual Video VAE adds best-effort weight identity verification and clear mismatch failure.
- Bundled reference workflow uses the standard 20-step MiniMax H3 schedule with a matching steps hint of 20 and removes stale serialized VAE control state.
