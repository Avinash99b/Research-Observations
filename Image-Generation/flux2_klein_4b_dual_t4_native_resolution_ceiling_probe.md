# FLUX.2-klein-4B Dual-T4 Native Resolution Ceiling Probe

## 1. Executive Summary

A higher-resolution probe extended the previous 1536² benchmark. The distilled FP16 checkpoint succeeded at **1728 × 1728 in 41.978 seconds** with four steps and balanced placement. The next tested size, **1792 × 1792, failed with CUDA out-of-memory on GPU 1** when a 1.53-GiB allocation was requested and only 1.00 GiB was free.

Therefore 1728² is the current highest successfully tested native square output on this setup. It is not an absolute theoretical limit: allocator tuning, more careful placement, VAE decode strategy, a different precision / quantization route or a changed software stack could shift the limit.

## 2. Environment & Method

- 2 × NVIDIA Tesla T4, 15,360 MiB reported per GPU; Python reports 14.562 GiB per GPU.
- Python 3.13.15; PyTorch 2.11.0+cu128; Diffusers 0.40.0; Transformers 5.16.1.
- Model: black-forest-labs/FLUX.2-klein-4B distilled; FP16.
- Device map: text_encoder → GPU 0; transformer and VAE → GPU 1.
- Sampling: 4 steps, guidance scale 1.0, fixed seed, square images.
- Per-resolution measurements were made sequentially after model loading. Loading is excluded from inference time. Failed attempts were caught and written to the JSON record rather than discarded.

## 3. Measured Results

| Resolution | Pixels | Inference time | Peak allocated VRAM GPU 0 | Peak allocated VRAM GPU 1 | Outcome |
|---:|---:|---:|---:|---:|---|
| 1536 × 1536 | 2.359 MP | 31.418 s | 7.736 GiB | 12.748 GiB | Pass; previous run |
| 1600 × 1600 | 2.560 MP | 29.177 s | 7.736 GiB | 13.204 GiB | Pass |
| 1664 × 1664 | 2.769 MP | 34.398 s | 7.736 GiB | 12.685 GiB | Pass |
| 1728 × 1728 | 2.986 MP | 41.978 s | 7.736 GiB | 13.100 GiB | Pass; highest verified so far |
| 1792 × 1792 | 3.211 MP | — | — | — | OOM on GPU 1 |

The inference time for 1600² was lower than the earlier separate 1536² run, and the observed PyTorch peak allocated bytes were not strictly monotonic across every resolution. Treat these results as point measurements, not a smooth latency curve. Repeats under controlled, idle conditions are required before fitting a scaling model.

## 4. OOM Details

At 1792² the pipeline raised:

- Requested allocation: 1.53 GiB.
- Free memory on GPU 1 at failure: 1.00 GiB.
- Current process usage reported by PyTorch: 13.56 GiB.
- PyTorch allocated: 10.08 GiB.
- PyTorch reserved but not allocated: 3.34 GiB.

The warning mentioned PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True as a possible way to reduce fragmentation. It has **not yet been tested** on this workload. Because the failed requested block was larger than reported free memory, allocator tuning alone is not guaranteed to make 1792² fit. The model must be reloaded into a clean process before repeating the failed test.

## 5. Key Observations

1. A native 1728-square image is feasible on this dual-T4 placement at four steps, using approximately 13.1 GiB of peak allocated VRAM on GPU 1.
2. 1792-square currently fails and should not be used as a production default with this placement.
3. The practical safety target should be lower than the maximum successful size. A 1600² or 1664² target gives more headroom for variability, other allocations and additional GPU workloads.
4. Native diffusion and upscaling are different targets. A 1408² output can be upscaled to 5632² with realesr-general-x4v3 in 0.665 seconds of inference; the higher raster dimensions do not create the same native generative detail as direct 5632² diffusion.
5. The OOM log points to two next custom-code experiments: allocator configuration and reducing VAE decode peak memory / avoiding extra activation copies. Each needs its own repeatable before/after test.

## 6. Next Tests

- Repeat 1536², 1600², 1664² and 1728² in randomized order (at least 3 repeats per size) to obtain medians and dispersion.
- Test PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in a fresh process.
- Try tiled VAE decode / VAE slicing if compatible with Flux2KleinPipeline; measure whether only decode fits more easily and compare seam/quality artifacts.
- Probe 1792² again after memory optimization. Do not infer success from an image-sized array alone; verify the saved image dimensions and inspect content validity.

## 7. Reproducibility & Artifacts

- JSON record: /kaggle/working/bench_out/flux2klein_highres_extended_scaling.json
- Script: /kaggle/working/test_flux2_highres_extended.py
- Successful output files: flux2klein_distilled_extended_1600_4step.png, flux2klein_distilled_extended_1664_4step.png, flux2klein_distilled_extended_1728_4step.png in /kaggle/working/bench_out/.

All figures are measurements on the active software stack / hardware. They are not universal limits for all FLUX.2-klein installations.
