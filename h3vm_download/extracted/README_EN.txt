H3 VRAM Master v0.21.1 | Installation Guide

PACKAGE CONTENTS
1. ComfyUI-H3-VRAM-Master/
   Copy this folder to ComfyUI/custom_nodes/.

2. 01_H3VM_Core_Reference_Workflow.json
   Current reference workflow with H3VM Core and H3VM Video VAE already connected.

PUBLIC NODES
- H3VM Core | Model Engine
- H3VM Video VAE | Dual-GPU Decode

H3VM does not own Prompt, Seed, resolution, duration, Sampler, Sigmas, or actual sampling steps.

MODEL PATH
Original Model Loader -> H3VM Core -> original workflow MODEL inputs

VAE PATH
LATENT + original Video VAE -> H3VM Video VAE -> IMAGE
Connect Core outputs Mode and VAE Policy to H3VM Video VAE.

STEPS HINT
The Core steps hint is for H3VM prefetch/final-step quality scheduling only. It is not a run boundary and does not change real sampling steps. Actual begin/end is detected from the Sampler lifecycle, so a mismatched hint cannot carry old mailbox/predictor history into the next prompt. The bundled reference workflow uses the standard 20-step MiniMax H3 configuration and presets the hint to 20.

DUAL-GPU VAE
The switch lives in H3VM Core. Disable it for highly asymmetric GPU pairs when dual-VAE decoding is not beneficial. Single-GPU Core mode always forces native single-GPU VAE decode.

The secondary VAE file selected in H3VM Video VAE must use the same MiniMax H3 Video VAE weights as the upstream VAE Loader. v0.21.1 performs a lightweight identity check when possible and rejects an obvious mismatch.

COMPATIBILITY BASELINE
This package was statically/API-checked against current ComfyUI v0.35.0 interfaces. Update older ComfyUI builds if required multi-GPU clone/wrapper APIs are missing.

Installing the plugin alone does not activate H3VM runtime patches.
