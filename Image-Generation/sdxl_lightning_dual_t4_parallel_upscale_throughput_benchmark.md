# SDXL-Lightning Dual-T4 Generate → Upscale Pipeline Benchmark

## 1. Executive Summary

This observation measures a real queued, two-GPU pipeline rather than adding independent timing estimates. GPU 0 generated 1024 × 1024 images with an SDXL-Lightning 4-step UNet; GPU 1 upscaled the previous output with Real-ESRGAN general x4v3 while the next generation was in progress. The final PNGs were 4096 × 4096.

The three-image queue completed successfully in **16.580 seconds of measured pipeline wall time**, after model loading and warm-up. The first output was complete 7.969 seconds after the pipeline timer started; later outputs completed at 12.246 and 16.563 seconds. Their completion intervals were approximately 4.28 and 4.32 seconds, showing why throughput should be reported separately from first-image latency.

## 2. Test Environment and Configuration

- Hardware: 2 × NVIDIA Tesla T4, 15,360 MiB per GPU; Intel Xeon CPU @ 2.00 GHz, 2 cores / 4 threads reported; 31 GiB system memory, no swap.
- Software: Python 3.13.15, PyTorch 2.11.0+cu128, Diffusers 0.40.0, Transformers 5.16.1.
- Generator: Stable Diffusion XL base with ByteDance SDXL-Lightning 4-step UNet checkpoint loaded on GPU 0 in FP16.
- Sampling: 1024 × 1024, four steps, guidance scale 0, Euler scheduler with trailing timestep spacing.
- Upscaler: Real-ESRGAN general x4v3 on GPU 1 in FP16.
- Pipeline: bounded queue of one image between generator and upscaler; generation and upscale/save run on separate worker threads. PNG was saved with compression level 1 to reduce CPU encoding latency.
- Generator and upscaler model loads plus one warm-up were excluded from pipeline wall time.

## 3. Measured Per-Image Results

| Image prompt label | Native generation | Upscaler GPU inference | PNG save / encode | Completion since pipeline start | Output dimensions | PNG size |
|---|---:|---:|---:|---:|---:|---:|
| fox / wildlife | 4.258 s | 0.401 s | 2.056 s | 7.969 s | 4096 × 4096 | 16.76 MB |
| botanist / portrait | 4.540 s | 0.366 s | 1.872 s | 12.246 s | 4096 × 4096 | 16.24 MB |
| coast / landscape | 4.522 s | 0.375 s | 1.795 s | 16.563 s | 4096 × 4096 | 17.06 MB |

The upscaler's measured peak allocated VRAM was 0.510 GiB, 0.604 GiB and 0.604 GiB for the three outputs. These figures cover the upscaler's measured allocation, not a complete GPU-1 device-memory accounting.

## 4. Throughput vs. Latency

- First complete 4K image: 7.969 seconds after starting the warm pipeline timer.
- Subsequent completion intervals: 4.277 seconds and 4.317 seconds.
- Overall three-image wall time: 16.580 seconds, or about 5.53 seconds per completed output over this short batch.
- Generation itself took about 4.26–4.54 seconds per image. General x4v3 inference took about 0.37–0.40 seconds. PNG writing took 1.80–2.06 seconds.

The queue successfully overlaps independent GPU work: GPU 0 can generate while GPU 1 upscales the preceding image. A bounded queue avoids unlimited output accumulation. The first-image latency is not the same as the steady-state interval between completed images.

## 5. Key Observations

1. **A second GPU is especially useful when it owns a separate pipeline stage.** It does not make one diffusion pass intrinsically twice as fast, but it lets the system generate image N while upscaling image N−1.
2. **The fast general upscaler is not the bottleneck.** At ~0.4 seconds of inference, it is much quicker than four-step generation (~4.3–4.5 seconds).
3. **PNG encoding is a visible bottleneck on this CPU.** The save operation took approximately 1.8–2.1 seconds per 4K file, several times the upscaler inference. End-to-end optimization must include file I/O.
4. **Model loading is cold-start overhead.** This run took 64.031 seconds to load the generator and 0.376 seconds to load the upscaler, plus warm-up. For interactive or service usage, keep both models resident if VRAM and process lifetime permit.
5. **The model is four-step distilled.** Four-step speed should not be assumed to represent conventional 25–50-step SDXL quality. Use visual review and a prompt suite before choosing for production.
6. **These are three prompt cases, not broad quality statistics.** The fox, botanist and coastal scenes were throughput probes; the run was not a statistically powered visual-quality comparison.

## 6. Production Recommendation

For maximum output throughput on the tested dual-T4 host, keep the SDXL-Lightning generator resident on GPU 0 and Real-ESRGAN general x4v3 resident on GPU 1. Use a one-item bounded queue, save asynchronously, and expose separate metrics for first-image latency, generation time, upscaler inference, encoding and completion intervals.

If image quality is prioritized over throughput, compare a higher-step checkpoint or another base model while preserving the same prompts, seeds, dimensions, PNG settings and queue instrumentation. Do not report only the sum of isolated GPU timings as if it were a measured overlapped pipeline.

## 7. Reproducibility & Artifacts

The successful run record is /kaggle/working/bench_out/dual_t4_fast_queue_benchmark.json. Output PNGs are fast_pipeline_0_fox_4096.png, fast_pipeline_1_botanist_4096.png and fast_pipeline_2_coast_4096.png in the same directory. The script used for this run is /kaggle/working/dual_t4_fast_queue.py.

This report preserves the separation between measured values, model-loading overhead and qualitative interpretation.
