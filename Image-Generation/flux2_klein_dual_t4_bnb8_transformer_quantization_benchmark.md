# FLUX.2-klein-4B Dual-T4 bitsandbytes 8-bit Transformer Quantization

## 1. Executive Summary

The FLUX.2-klein-4B distilled transformer was loaded through bitsandbytes 8-bit quantization, with the text encoder and VAE left in FP16. The quantized transformer reported a memory footprint of **3.609 GiB**. On the tested two-T4 setup, the quantized pipeline generated valid pixel-varying outputs at 1024², 1536², 1728² and 1792² using four steps and guidance scale 1.0.

Most importantly, **1792 × 1792 completed in 43.040 seconds**. The same resolution had previously failed with an FP16 transformer and the same general component placement, because GPU 1 ran out of memory. The quantized test therefore extends the verified native square resolution beyond the current FP16 limit of 1728².

A planned 1920² probe was not completed or recorded: the benchmark workspace disappeared and the process was no longer running when the next status check occurred. Do not infer either success or failure at 1920² from this report.

## 2. Environment and Configuration

- Model: black-forest-labs/FLUX.2-klein-4B distilled.
- Hardware: 2 × NVIDIA Tesla T4, 15,360 MiB reported memory each; compute capability 7.5.
- Software: Python 3.13.15, PyTorch 2.11.0+cu128, Diffusers 0.40.0, Transformers 5.16.1, bitsandbytes installed.
- Transformer quantization: bitsandbytes 8-bit.
- Remaining modules: FP16.
- Component placement: text encoder → GPU 0; transformer → GPU 1; VAE → GPU 1.
- Sampling: 4 steps, guidance scale 1.0, seed 4242 for every size.
- Inference time excludes model loading and PNG save.
- The quantized run reported transformer memory footprint at 3.609 GiB and measured pipeline-load time at 35.484 seconds after transformer-load time of 32.139 seconds.

## 3. Measured Quantized Results

| Resolution | Inference time | Peak allocated VRAM GPU 0 | Peak allocated VRAM GPU 1 | Pixel-validity check |
|---:|---:|---:|---:|---|
| 1024 × 1024 | 11.136 s | 7.736 GiB | 6.174 GiB | Pass |
| 1536 × 1536 | 25.101 s | 7.736 GiB | 9.148 GiB | Pass |
| 1728 × 1728 | 37.002 s | 7.736 GiB | 10.569 GiB | Pass |
| 1792 × 1792 | 43.040 s | Not preserved in captured result | Not preserved in captured result | Pass; inference returned and image statistics varied |

The validity check rejects a uniform or near-uniform output, but it does not replace a careful visual-quality check or a perceptual metric. The captured results establish that the 1792² call finished and produced non-uniform pixels; they do not establish that quantization caused no quality loss.

## 4. Comparison with the FP16 Baseline

The previous FP16 resolution benchmark recorded:

| Resolution | FP16 baseline | bitsandbytes 8-bit | Interpretation |
|---:|---:|---:|---|
| 1024 × 1024 | 10.559 s; GPU 1 peak allocation 9.773 GiB | 11.136 s; GPU 1 peak allocation 6.174 GiB | 8-bit was ~5.5% slower, with a lower GPU 1 allocation |
| 1536 × 1536 | 31.418 s; GPU 1 peak allocation 12.748 GiB | 25.101 s; GPU 1 peak allocation 9.148 GiB | 8-bit was ~20.1% faster in these separate runs and used less allocated VRAM |
| 1728 × 1728 | 41.978 s; GPU 1 peak allocation 13.100 GiB | 37.002 s; GPU 1 peak allocation 10.569 GiB | 8-bit was ~11.9% faster in these separate runs and used less allocated VRAM |
| 1792 × 1792 | OOM | 43.040 s; pass | Quantization extended the verified size boundary |

These are separate sequential experiments, not an interleaved randomized A/B test. Different runtime state, host conditions and prompt/configuration details can affect timings, so treat speed differences as promising observations rather than a universal quantization speedup. The memory advantage is the more robust motivation for this experiment.

## 5. Interpretation

1. **Quantization changes the practical resolution boundary.** The 8-bit transformer fit a 1792² image on the current placement after the FP16 configuration failed there.
2. **At 1024², the quantized path is not faster.** It took 11.136 s compared with 10.559 s in the previous FP16 run. Quantization is most interesting when memory pressure is the limiting factor.
3. **At larger tested sizes, the quantized run was also faster in this measurement.** The 1536² and 1728² points are encouraging, but a repeatable randomized A/B benchmark is still needed.
4. **Do not treat 1792² as a safe production default yet.** A successful run leaves less protection against unrelated allocations and run-to-run peaks than a lower resolution. Use 1728² or below when reliability matters until the higher size is repeated.
5. **Quantized output quality still needs a controlled comparison.** The run performed simple image-statistics checks, but no blinded preference test, LPIPS/SSIM comparison, or matched FP16-versus-8-bit image review was completed in the captured session.
6. **1920² remains untested in the record.** The workspace disappeared before a confirmed result was retained. It must be rerun in a fresh, persistent workspace before claiming a larger ceiling.

## 6. Recommended Next Experiments

- Repeat 1536², 1728² and 1792² with both FP16 and 8-bit in randomized order, at least three runs each, with the same prompt, seed and scheduler, recording allocated and total device memory.
- Inspect matched FP16 and 8-bit images at 1024² and 1728² and add a perceptual metric where practical.
- Re-run 1920² only after ensuring the run output JSON is atomically written after every result to a workspace that survives session resets.
- If 1920² fails, capture the exact allocation request, free memory and peak allocator stats rather than describing the result only as OOM.

## 7. Reproducibility & Artifact Availability

The measurements were captured from the active benchmark process before its workspace disappeared. The following result file existed during the run but was not available at the subsequent status check:

- /kaggle/working/bench_out/flux2klein_bnb8_benchmark.json

The reproduction script used for this probe was:

- /kaggle/working/test_flux2_klein_bnb8.py

The individual successful PNGs recorded during execution were:

- flux2klein_bnb8_1024_4step.png
- flux2klein_bnb8_1536_4step.png
- flux2klein_bnb8_1728_4step.png
- flux2klein_bnb8_1792_4step.png

Those were temporary workspace artifacts and were not verified as persistent after the workspace disappeared. The table above therefore preserves only values explicitly captured in the session. Keep this caveat with the observation instead of implying that the result JSON or image files are still accessible.
