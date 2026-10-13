# Z-Image-Turbo Dual-T4 BF16 vs FP16 Validity & Performance

## 1. Executive Summary

Z-Image-Turbo was not usable in the first FP16 test: the pipeline completed its call, but 512² and 768² outputs were uniform black. In the model-card-recommended BF16 configuration with balanced two-GPU placement, the same fox prompt produced valid, coherent wildlife images.

The valid BF16 run took **64.802 seconds at 512 × 512** and **143.568 seconds at 768 × 768** at nine scheduler steps (the model card describes this as eight DiT forwards), with guidance scale 0. The measured latency is too slow to compete with the four-step FLUX.2-klein pipeline on this particular T4 setup, despite valid image quality. We have not run the 1024² test because the smaller-resolution latency already makes it a lower-priority candidate.

## 2. Environment & Configuration

- Model: Tongyi-MAI/Z-Image-Turbo, official model card https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- License advertised by the model card: Apache-2.0.
- Hardware: two NVIDIA Tesla T4 GPUs, each 15,360 MiB reported memory, compute capability 7.5.
- Software: Python 3.13.15, PyTorch 2.11.0+cu128, Diffusers 0.40.0, Transformers 5.16.1.
- Device placement for valid BF16 run: transformer → GPU 0; text encoder and VAE → GPU 1.
- Sampling: BF16, 9 scheduler steps (8 DiT forwards according to the card), guidance scale 0, fixed seed 4242, same detailed red-fox/forest prompt at both resolutions.
- Inference times exclude the long model-loading stage and PNG writing. Load time measured at 132.875 s.

## 3. Results

| Precision | Resolution | Inference time | Peak allocated VRAM GPU 0 | Peak allocated VRAM GPU 1 | Image validity |
|---|---:|---:|---:|---:|---|
| FP16 | 512 × 512 | 7.642 s | 11.652 GiB | 8.255 GiB | Fail: uniform black |
| FP16 | 768 × 768 | 14.539 s | 11.794 GiB | 8.998 GiB | Fail: uniform black |
| BF16 | 512 × 512 | 64.802 s | 11.934 GiB | 9.128 GiB | Pass: coherent wildlife image |
| BF16 | 768 × 768 | 143.568 s | 13.177 GiB | 10.965 GiB | Pass: coherent wildlife image |

The FP16 timing is intentionally shown but **must not be treated as an actual generation performance result** because the outputs were entirely black (RGB mean = [0,0,0], standard deviation 0). The 512 and 768 BF16 images both passed basic pixel-range/statistics validation and visual review. Output PNGs were 0.38 MB at 512² and 0.81 MB at 768².

## 4. Interpretation

1. **Model-card precision matters.** The model card's recommended BF16 path produced valid images, while the T4-compatible FP16 experiment produced black outputs without throwing an inference exception. This test does not prove FP16 alone caused the failure, because precision, kernel selection, and component placement can interact.
2. **BF16 works but is very slow on this host.** T4 is compute capability 7.5 and does not have native BF16 Tensor Core throughput like newer architectures. PyTorch's BF16 support query returned true on this host and simple tensor operations passed, but that capability flag alone is not a promise of fast native BF16 model inference.
3. **The latency increases sharply with image size.** The 768² run took 2.22× as long as 512² and peaked at 13.177 GiB allocated on GPU 0.
4. **The result is valid but not competitive for low-latency generation.** FLUX.2-klein-4B distilled in FP16 on the same dual-T4 system generated 512² in 4.370 s, 768² in 5.546 s and 1024² in 10.559 s with four steps, and produced coherent imagery. Z-Image's quality may still justify future exploration, but this validated path is markedly slower.
5. **No 1024² BF16 test was run.** We stopped at 768² because its latency and peak memory already make the model low priority on this hardware. The tested values do not establish an absolute resolution ceiling.

## 5. Recommended Status

Keep Z-Image-Turbo as a **valid but currently low-priority T4 option**. Use BF16 only when valid output is required with the present software stack; do not use the FP16 path without fixing and validating its numerical failure. The next potentially worthwhile experiment is lower-level kernel/precision validation, but the current speed results do not justify broad resolution sweeps or Torch compilation without first finding an unexpectedly large speedup.

## 6. Reproducibility & Artifacts

- BF16 result record: /kaggle/working/bench_out/zimage_balanced_bf16_benchmark.json
- BF16 script: /kaggle/working/test_zimage_balanced_bf16.py
- FP16 failed record: /kaggle/working/bench_out/zimage_balanced_fp16_benchmark.json
- FP16 script: /kaggle/working/test_zimage_balanced_fp16.py
- Valid images: zimage_balanced_bf16_512_9step.png and zimage_balanced_bf16_768_9step.png.
- Failed black images: zimage_balanced_fp16_512_9step.png and zimage_balanced_fp16_768_9step.png.

The numbers are direct observations on this two-T4 software stack, not general claims about performance on newer GPUs. The official model card recommends BF16 for supported GPUs and reports eight DiT forwards for the nine-step setting: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
