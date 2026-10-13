# SDXL-Lightning Dual-T4 torch.compile Optimization Result

## 1. Executive Summary

The SDXL-Lightning four-step UNet was tested with `torch.compile(unet, mode='reduce-overhead')` on GPU 0 at 1024 × 1024, FP16, four steps and guidance scale 0. The first attempt failed because the runtime's /tmp and /root/.triton directories are read-only. Redirecting Triton, TorchInductor and temporary-file caches to /kaggle/working fixed the build path and yielded a successful valid image.

However, **compilation was slower than eager inference**:

- Eager default contiguous + SDPA: 4.2227 s median (three warm runs).
- Compiled UNet: 4.9490 s median after compilation (three warm runs), ~17.2% slower.
- First compiled call including graph compilation: 145.495 s.
- Peak allocated GPU 0 VRAM: 9.736 GiB.

For this specific T4 + four-step workload, do not enable torch.compile by default. It costs ~2.4 minutes of first-call startup and does not improve warm latency.

## 2. Environment / Workload

- Hardware: 2 × NVIDIA Tesla T4 (15,360 MiB each); this inference run used GPU 0.
- CPU: Intel Xeon CPU @ 2.00 GHz, 2 cores / 4 threads reported; 31 GiB system RAM.
- Python 3.13.15; PyTorch 2.11.0+cu128; Diffusers 0.40.0; Transformers 5.16.1.
- Model: stabilityai/stable-diffusion-xl-base-1.0 with ByteDance/SDXL-Lightning four-step UNet weights.
- Precision: FP16, 1024 × 1024, four steps, guidance scale 0, Euler scheduler with trailing timestep spacing.
- Same fox-in-forest prompt and seed 4242 were used for comparisons.
- Eager baseline came from a separate controlled ablation with the same scheduler, seed, prompt and size.

## 3. Results

| Mode | First invocation | Warm runs | Warm median | Peak allocated VRAM | Output validity |
|---|---:|---|---:|---:|---|
| Eager default contiguous + SDPA | Warmup excluded; model loaded separately | 4.1277 / 4.2227 / 4.3037 s | **4.2227 s** | 9.740 GiB | Valid |
| torch.compile, reduce-overhead | 145.495 s including compile | 4.7634 / 4.9490 / 5.0846 s | 4.9490 s | 9.737 GiB | Valid |

The compiled warm median is about 17.2% slower than eager. Since it does not provide a warm-inference saving, there is no request count at which the measured compile delay would break even for this configuration. The output pixel statistics were nearly identical between eager and compiled paths (RGB means ≈ [101.38, 86.55, 67.09], standard deviation ≈ 45.88), and both passed basic image-validity checks.

## 4. Cache / Toolchain Findings

The initial compilation attempt failed in Triton/Inductor because the runtime tried to write to /root/.triton and /tmp, both read-only in this shell environment.

A simple GCC probe confirmed the underlying issue: GCC aborted when it attempted to create temporary files in /tmp. Compiling the same small C helper with TMPDIR redirected to /kaggle/working succeeded.

The successful rerun set these writable paths before importing PyTorch:

- TMPDIR, TEMP and TMP → /kaggle/working/tmp
- TRITON_CACHE_DIR → /kaggle/working/triton_cache
- TORCHINDUCTOR_CACHE_DIR → /kaggle/working/torchinductor_cache
- TORCHINDUCTOR_COMPILE_THREADS=2 to avoid overwhelming the two-core host

This is a reusable environmental fix for other Triton/Inductor experiments. It does not make compilation profitable for SDXL-Lightning on this workload; it only makes the experiment runnable.

## 5. Interpretation & Recommendation

1. **Reject torch.compile for the current four-step T4 production path.** It adds significant cold-start time and slows the warm median by ~17.2%.
2. The graph compiled successfully after fixing writable cache/temp paths, so the poor result is a measured performance outcome rather than a compile failure.
3. The base UNet + default SDPA eager configuration remains preferable for interactive or repeated image generation.
4. A future compile test might be justified for a different larger model, a higher step count, a larger batch or newer GPU, but each requires its own baseline and cold/warm accounting.
5. Do not infer universal behavior from this result; compile performance depends on model shape, backend, architecture, compiler version and workload repetition.

## 6. Source References

- PyTorch torch.compile: https://pytorch.org/docs/stable/generated/torch.compile.html
- TorchInductor configuration / troubleshooting: https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html
- SDXL-Lightning model card: https://huggingface.co/ByteDance/SDXL-Lightning

## 7. Reproducibility & Artifacts

- Final JSON: /kaggle/working/bench_out/sdxl_lightning_torch_compile_tmpdir_benchmark.json
- Successful script: /kaggle/working/benchmark_sdxl_lightning_torch_compile_tmpdir.py
- Initial cache-permission failure JSON: /kaggle/working/bench_out/sdxl_lightning_torch_compile_benchmark.json
- Eager baseline JSON: /kaggle/working/bench_out/sdxl_lightning_kernel_ablation.json
- Valid compiled output: sdxl_lightning_torch_compile_tmpdir_warm_1024.png in /kaggle/working/bench_out/

This report distinguishes compilation failure caused by read-only filesystem paths from a successful compile that simply performed worse at runtime.
