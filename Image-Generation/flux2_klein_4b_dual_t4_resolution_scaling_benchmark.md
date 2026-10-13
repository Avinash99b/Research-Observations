# FLUX.2-klein-4B Dual-T4 Native Resolution & Scaling Benchmark

## 1. Executive Summary

This observation records measured generation and resolution-scaling results for the distilled FLUX.2-klein-4B text-to-image pipeline on two NVIDIA Tesla T4 GPUs (15,360 MiB reported memory per GPU). The successful placement was FP16 with the text encoder on GPU 0 and transformer + VAE on GPU 1. Putting the entire pipeline on one T4 previously failed from memory pressure; balanced component placement makes the model run without CPU offloading.

The highest successful native square resolution tested here was **1536 × 1536 in 31.418 seconds at four steps**. The corresponding measured peak allocated VRAM was 7.736 GiB on GPU 0 and 12.748 GiB on GPU 1. This is a tested point, not a proven absolute model limit.

## 2. Test Environment

- Date: 2026-10-13 UTC.
- GPUs: 2 × NVIDIA Tesla T4, each reporting 15,360 MiB total memory (14.562 GiB in PyTorch).
- CPU: Intel Xeon CPU @ 2.00 GHz, 2 cores / 4 threads reported; memory 31 GiB; no swap.
- Python 3.13.15; PyTorch 2.11.0+cu128; CUDA runtime reported by PyTorch 12.8; Diffusers 0.40.0; Transformers 5.16.1.
- Model checkpoint: black-forest-labs/FLUX.2-klein-4B distilled.
- Precision / placement: FP16, balanced component placement: text_encoder → cuda:0; transformer → cuda:1; VAE → cuda:1.
- Inference settings for the distilled checkpoint: 4 steps, guidance scale 1.0. A fixed seed was used for resolution-scaling comparisons.
- Timings below exclude downloading and pipeline/model load time. They are individual inference wall times, not multi-run medians unless explicitly stated.

## 3. Native Resolution Scaling

| Square resolution | Pixels | Steps | Inference time | Peak VRAM GPU 0 | Peak VRAM GPU 1 | Status |
|---:|---:|---:|---:|---:|---:|---|
| 512 × 512 | 0.262 MP | 4 | 4.370 s | 7.736 GiB | 7.990 GiB | Pass |
| 768 × 768 | 0.590 MP | 4 | 5.546 s | 7.736 GiB | 8.733 GiB | Pass |
| 1024 × 1024 | 1.049 MP | 4 | 10.559 s | 7.736 GiB | 9.773 GiB | Pass |
| 1216 × 1216 | 1.479 MP | 4 | 15.454 s | 7.736 GiB | 10.751 GiB | Pass |
| 1280 × 1280 | 1.638 MP | 4 | 16.660 s | 7.736 GiB | 11.114 GiB | Pass |
| 1408 × 1408 | 1.982 MP | 4 | 22.391 s | 7.736 GiB | 11.894 GiB | Pass |
| 1536 × 1536 | 2.359 MP | 4 | 31.418 s | 7.736 GiB | 12.748 GiB | Pass; highest tested so far |

The 1536 run used approximately 87.5% of GPU 1's PyTorch-reported total memory at peak allocation, leaving limited headroom for other GPU-1 workloads. Peak allocated memory is not the same as total reserved/device memory.

## 4. Higher-Quality Base Checkpoint Comparison

The non-distilled FLUX.2-klein-base-4B checkpoint was also tested using the reference-style 50-step, guidance-scale 4.0 setup:

| Square resolution | Steps | Inference time | Peak VRAM GPU 0 | Peak VRAM GPU 1 |
|---:|---:|---:|---:|---:|
| 512 × 512 | 50 | 67.135 s | 7.736 GiB | 7.997 GiB |
| 768 × 768 | 50 | 136.902 s | 7.736 GiB | 8.740 GiB |

Visual inspection found the base checkpoint produced coherent fox anatomy, lighting, fur and forest structure, but the cost was substantial. The 50-step base variant is the slower quality-oriented option; the four-step distilled checkpoint is the practical low-latency option. This was a visual spot comparison, not a blinded or statistically scored quality study.

## 5. Native Generation + 4× Upscaling

A 1408 × 1408 distilled output was upscaled on GPU 1 with realesr-general-x4v3:

| Input | Output | Upscaler inference (median of 3) | Peak upscaler VRAM | PNG save time | Output size |
|---:|---:|---:|---:|---:|---:|
| 1408 × 1408 | 5632 × 5632 | 0.665 s | 1.314 GiB | 3.498 s | 32.02 MB |

This provides a useful separation between native detail and output canvas size. The 5632-square image is generated from the 1408-square latent-decoded image; its 4× dimensions must not be interpreted as equivalent to native 5632-square diffusion detail. On this low-core-count host, PNG encoding took materially longer than the upscaler's GPU inference.

## 6. Key Observations

1. **Two-GPU placement is the enabling step.** The text encoder on GPU 0 and transformer + VAE on GPU 1 successfully loaded and ran in FP16. The single-GPU attempt did not.
2. **1536² is the current verified native ceiling, not the absolute limit.** It completed in 31.418 s at 4 steps. The GPU-1 peak of 12.748 GiB leaves less headroom than the 1408² run; higher resolutions still require an explicit safe probe.
3. **Resolution increases become expensive quickly.** Generation rose from 10.559 s at 1024² to 31.418 s at 1536², roughly a 2.98× time increase for a 2.25× pixel count.
4. **Base and distilled variants serve different purposes.** The base checkpoint took 67.135 s at only 512² and 136.902 s at 768² at 50 steps. It should be reserved for tasks where the visual gain is worth the latency.
5. **Upscaling is cheap in GPU time, not necessarily end-to-end time.** A 1408² image was upscaled to 5632² in 0.665 s, but CPU-side PNG encoding took 3.498 s.
6. **This is not yet an exhaustive quality benchmark.** The prompt, fixed seed and direct visual inspection are useful for workflow comparison but do not replace a diverse prompt suite, repeats, or a blind evaluation.

## 7. Current Recommendation

- **Fast high-resolution generation:** FLUX.2-klein-4B distilled, FP16 with balanced two-T4 placement, 4 steps, guidance scale 1.0; start at 1024² for speed and use 1408² when more native detail is worth the additional time.
- **More native detail for a final image:** use the base checkpoint with a higher step count only when a 1–2 minute generation budget is acceptable.
- **Very large deliverables:** native generation followed by a separate 4× upscaler is viable; measure save/encoding time separately, and do not equate upscaled dimensions to native diffusion detail.
- Do not run other GPU-1 jobs concurrently with the 1536² model until additional headroom probes are complete.

## 8. Reproducibility & Artifacts

The measured JSON records and PNG outputs are stored in the active benchmark workspace under /kaggle/working/bench_out/, including:

- flux2klein_balanced.json
- flux2klein_balanced_1024.json
- flux2klein_highres_scaling.json
- flux2klein_base_balanced.json
- flux2klein_1536_benchmark.json
- flux2klein_1408_upscale.json

The report uses measured values from these run outputs. Timings are specific to this software build, GPU host, model revision and memory placement; re-run before treating them as portable hardware constants.
