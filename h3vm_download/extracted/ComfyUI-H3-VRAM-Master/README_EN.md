# ComfyUI-H3-VRAM-Master v0.21.1

The public release focuses on two infrastructure nodes: **H3VM Core** and **H3VM Video VAE**.

## Integration

Model path:
`Original Model Loader -> H3VM Core -> original workflow`

Video VAE path:
`LATENT + original Video VAE + Core mode / vae_policy -> H3VM Video VAE -> IMAGE`

H3VM does not take over Prompt, Seed, resolution, duration, Sampler, Sigmas, actual sampling steps, CLIP/text encoder, LoRA, or conditioning logic.

## Unified control panel

The **Dual-GPU VAE** switch lives in the H3VM Core panel.

- On: the Video VAE node follows Core policy and uses dual-GPU temporal-chunk decoding.
- Off: the Video VAE node falls back to native single-GPU `vae.decode()`.
- If Multi-GPU is disabled in Core, Video VAE is forced to single GPU.
- On highly asymmetric GPU pairs, turn Dual-GPU VAE off while keeping model multi-GPU enabled if desired.

The Video VAE node no longer exposes a duplicate switch; it consumes Core's `vae_policy`.

### The steps hint is not a sampling boundary

Core's steps hint is used only for H3VM prefetch / final-step quality scheduling. It **does not change the Sampler's real step count and no longer decides when a sampling run ends**. Actual run begin/end is owned by ComfyUI's `OUTER_SAMPLE` lifecycle, so a mismatched hint cannot carry an old mailbox or predictor history into the next prompt.

Keeping the hint close to your normal step count is still recommended for better end-of-run scheduling. The bundled reference workflow uses the standard `BasicScheduler = 20`, so its hint is preset to `20`.

### Video VAE weight identity guard

Dual-GPU Video VAE requires both GPUs to decode with the same MiniMax H3 Video VAE weights. The `vae` input comes from the upstream VAE Loader, while `vae_name` selects the file loaded for the secondary GPU. They must represent the same weights.

v0.21.1 performs a lightweight weight signature check when the model state is comparable and rejects an obvious mismatch instead of silently joining frames decoded by different VAEs.

## UI language

Both public nodes provide Chinese / English labels. The frontend changes labels only and never reorders widgets or overrides serialization.

## Isolation

Installing the plugin does not activate the H3VM runtime. Runtime patches are lazy and standard ComfyUI CLIP/VAE loaders remain untouched until an H3VM execution path runs.

## ComfyUI compatibility baseline

This package was statically/API-checked against **ComfyUI v0.35.0** interfaces. H3VM relies on recent multi-GPU clone and ModelPatcher wrapper capabilities; update ComfyUI if an older build does not provide them.

Hardware behavior still depends on the GPU pair, driver, model and workflow. If the platform itself shows compiler/VRAM regressions, disabling compiler is a useful diagnostic A/B rather than a permanent H3VM requirement.

## Installation

Copy `ComfyUI-H3-VRAM-Master` to `ComfyUI/custom_nodes/`, then restart ComfyUI.

Bundled reference workflow: `01_H3VM_Core_Reference_Workflow.json`.
