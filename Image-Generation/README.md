# Image Generation Research — Dual Tesla T4

This directory contains reproducible observations from local/cloud inference experiments on the dual-T4 environment. Reports separate measured inference time, model load, GPU memory, file encoding, output validity, and qualitative inspection. Failed configurations are kept as failures rather than promoted because a call returned an image object.

## Hardware and Baseline Environment

- 2 × NVIDIA Tesla T4, 15,360 MiB reported each, compute capability 7.5.
- Intel Xeon CPU @ 2.00 GHz, 2 cores / 4 threads, 31 GiB host RAM.
- Python 3.13.15; PyTorch 2.11.0+cu128; Diffusers 0.40.0; Transformers 5.16.1.
- Measurements are environment-specific. Warm inference timings exclude model loading unless a report explicitly says otherwise.

## Recommended Workflows So Far

### Throughput: four-step SDXL-Lightning + parallel x4 upscale

Use SDXL-Lightning with FP16, four steps, guidance scale 0, and the Euler scheduler with trailing timestep spacing on GPU 0. Run realesr-general-x4v3 on GPU 1 with a bounded queue so it upscales image N while GPU 0 generates image N+1. In the measured queue, each 1024² generation took ~4.3–4.5 s; x4 inference took ~0.37–0.40 s; PNG save took ~1.8–2.1 s. Three 4096² outputs completed in 16.58 s total after model loading and warm-up.

### Higher native resolution: FLUX.2-klein-4B distilled

Use FP16 with a balanced component device map: text encoder on GPU 0, transformer and VAE on GPU 1, four steps and guidance scale 1.0. Valid outputs were measured from 512² through 1728². The highest tested successful native square size was 1728² at 41.98 s; 1792² OOMed on GPU 1. A 1408² output can be upscaled to 5632² by realesr-general-x4v3 in 0.665 s of warm inference, but that larger canvas does not equal 5632² native diffusion detail.

### Illustration: Pony V6 XL

For the tested Pony V6 XL checkpoint, a community Pony SDXL VAE, Euler Ancestral, 25 steps, CFG 7, score tags and **no clip_skip** produced valid coherent illustration outputs. The 1024² run took 35.52 s; RealESRGAN_x4plus_anime_6B upscaled it to 4096² in ~1.60 s of GPU inference.

### Upscaler selection

- realesr-general-x4v3: fastest, very low VRAM, throughput-first.
- RealESRGAN_x4plus_anime_6B: good specialized choice for anime/cel-shaded illustration.
- RealESRGAN_x4plus: slower, standard general-image restoration model.

### Optimizations measured so far

For SDXL-Lightning on this T4, default contiguous memory format with PyTorch SDPA enabled and cuDNN benchmark off was fastest at a 4.223 s median for 1024². channels_last and cuDNN autotuning both regressed. torch.compile is under investigation; its cache configuration must use writable paths because /tmp and /root/.triton are read-only in the current runtime.

## Reports

### Model and resolution
- [FLUX.2-klein-4B dual-T4 resolution scaling](flux2_klein_4b_dual_t4_resolution_scaling_benchmark.md)
- [FLUX.2-klein-4B native-resolution ceiling probe](flux2_klein_4b_dual_t4_native_resolution_ceiling_probe.md)
- [Pony V6 XL illustration conditioning](pony_v6_xl_dual_t4_illustration_conditioning_benchmark.md)
- [Z-Image-Turbo FP16 validity failure](z_image_turbo_dual_t4_fp16_output_validity_failure.md)
- [Z-Image-Turbo BF16 validity and performance](z_image_turbo_dual_t4_bf16_validity_and_performance.md)

### Pipeline and optimization
- [SDXL-Lightning parallel upscale throughput](sdxl_lightning_dual_t4_parallel_upscale_throughput_benchmark.md)
- [Real-ESRGAN speed and VRAM comparison](realesrgan_dual_t4_x4_upscaler_speed_vram_comparison.md)
- [SDXL-Lightning kernel optimization ablation](sdxl_lightning_dual_t4_kernel_optimization_ablation.md)

## Research Principles

1. Record model revision, pipeline version, precision, scheduler, step count, seed, resolution, and device map.
2. Separate cold-load time, first invocation/compile time, warm inference, post-processing, and save/encoding latency.
3. Repeat performance tests and report medians/ranges rather than a single timing.
4. Validate output pixels and inspect the image; a successful API call can still produce black or washed-out output.
5. Report allocated VRAM separately from total device usage where possible.
6. Preserve failed trials and configuration findings; do not overwrite history silently.
