# SDXL-Lightning Dual-T4 Kernel Optimization Ablation

## 1. Executive Summary

A controlled warm-inference comparison was run on the tested dual-T4 host for SDXL-Lightning's four-step UNet checkpoint. The baseline used FP16, the default contiguous memory format, PyTorch SDPA, cuDNN autotuning off, four scheduler steps, guidance scale 0, and 1024 × 1024 output.

The baseline median was **4.223 seconds** across three repeated inference calls. The tested layout and cuDNN autotuning changes were slower:

- channels_last: 5.862 s (+38.8% vs baseline)
- cuDNN benchmark/autotuning enabled: 4.884 s (+15.7%)
- channels_last + cuDNN benchmark: 5.434 s (+28.7%)

All generated images passed basic pixel-validity checks and appeared statistically equivalent at the coarse pixel-statistic level. For this model and hardware, **keep the default contiguous + SDPA path** unless later experiments demonstrate a benefit.

## 2. Environment and Setup

- Hardware: 2 × NVIDIA Tesla T4, 15,360 MiB reported per GPU; benchmark inference placed on GPU 0.
- CPU: Intel Xeon CPU @ 2.00 GHz, 2 cores / 4 threads reported; 31 GiB system RAM.
- Software: Python 3.13.15, PyTorch 2.11.0+cu128, Diffusers 0.40.0, Transformers 5.16.1.
- Base pipeline: stabilityai/stable-diffusion-xl-base-1.0 with ByteDance/SDXL-Lightning four-step UNet weights.
- Precision: FP16.
- Resolution: 1024 × 1024.
- Sampling: 4 steps, guidance scale 0, Euler scheduler with trailing timestep spacing.
- Prompt: “A cinematic photograph of a red fox in a misty enchanted forest at sunrise, detailed fur, moss and foliage, natural light, realistic wildlife photography.”
- Each timed sample used the same seed (4242). A shape-matched warmup was performed before each variant, then three measured repeats were taken. We reset peak-memory statistics before each measured sample and report the largest observed allocation.
- Model loading was excluded from inference timings; generator/UNet/VAE load plus checkpoint assembly took 65.013 seconds.

## 3. Results

| Variant | Three warm runs | Median | Min / max | Peak allocated VRAM | Relative to baseline |
|---|---|---:|---:|---:|---:|
| Default contiguous + SDPA, cuDNN benchmark off | 4.1277 / 4.2227 / 4.3037 s | **4.2227 s** | 4.1277 / 4.3037 s | 9.740 GiB | Baseline |
| channels_last on UNet + VAE | 5.6730 / 5.8616 / 6.0968 s | 5.8616 s | 5.6730 / 6.0968 s | 11.746 GiB | +38.8% slower |
| cuDNN benchmark on | 5.0164 / 4.8843 / 4.6346 s | 4.8843 s | 4.6346 / 5.0164 s | 9.743 GiB | +15.7% slower |
| channels_last + cuDNN benchmark | 5.4887 / 5.4245 / 5.4339 s | 5.4339 s | 5.4245 / 5.4887 s | 11.745 GiB | +28.7% slower |

Each variant produced a 1024 × 1024 PNG around 1.44 MB. Coarse RGB mean/std/range checks were nearly identical across variants; all were flagged valid by the benchmark's basic checks. This is not a full perceptual-quality evaluation.

## 4. Interpretation

1. **Default PyTorch SDPA + contiguous memory is the winner so far.** It had the lowest median latency and lowest measured peak allocation among these four variants.
2. **channels_last regressed on the T4 workload.** It raised median inference by about 39% and peak allocated memory by about 2.0 GiB. It is not a universal optimization; the benefit depends on model operators and GPU architecture.
3. **cuDNN benchmark/autotuning also regressed.** For this mostly attention-heavy, four-step diffusion path, tuning convolution algorithms did not offset its overhead or kernel choices.
4. **The combined change did not rescue channels_last.** It remained around 29% slower than baseline.
5. **Keep SDPA enabled.** PyTorch's SDPA backends are available in the installed stack. This test did not benchmark alternative attention processors such as xFormers or FlashAttention, which are not separately installed in this environment.
6. **Measured timings are warm and steady-state.** Compilation or dynamic algorithm selection can have significant cold-start cost. Any future optimization must report model load, first invocation/compile, warm median and total number of requests required to amortize setup.

## 5. Current Optimization Recommendation

For SDXL-Lightning 4-step inference on these T4 GPUs, start with:
- FP16,
- default contiguous memory format,
- PyTorch SDPA enabled,
- cuDNN benchmark off,
- four-step Lightning UNet, guidance 0, trailing Euler timestep spacing.

Do not set channels_last or cuDNN benchmark by default on this host based on these results. The next higher-effort test is torch.compile on the UNet with a fixed-shape repeated-generation workload. It should only be kept if steady-state gains materially offset compile time, compile-memory overhead and any change in output quality. Compilation will be evaluated separately rather than mixed into this kernel-format ablation.

## 6. Source References

- SDXL-Lightning model card and recommended sampling settings: https://huggingface.co/ByteDance/SDXL-Lightning
- Diffusers memory and speed optimizations: https://huggingface.co/docs/diffusers/optimization/memory
- PyTorch SDPA documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch channels-last tutorial: https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html

These documents describe techniques and expected trade-offs, not guaranteed speedups for this T4 workload.

## 7. Reproducibility & Artifacts

- JSON record: /kaggle/working/bench_out/sdxl_lightning_kernel_ablation.json
- Script: /kaggle/working/benchmark_sdxl_lightning_kernels.py
- Outputs: sdxl_lightning_kernel_baseline_sdpa_contiguous_1024.png, sdxl_lightning_kernel_channels_last_1024.png, sdxl_lightning_kernel_cudnn_benchmark_1024.png, sdxl_lightning_kernel_channels_last_cudnn_benchmark_1024.png in /kaggle/working/bench_out/.

All values are measurements on this specific software stack and GPU host.
