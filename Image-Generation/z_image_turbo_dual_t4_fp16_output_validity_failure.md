# Z-Image-Turbo Dual-T4 FP16 Validity Failure

## 1. Observation

Z-Image-Turbo was loaded in FP16 with a balanced device map across two Tesla T4 GPUs. Both 512² and 768² calls completed without a thrown inference exception, but the saved images were entirely black. The benchmark correctly rejected these as invalid outputs; their API-level completion times are not valid generation-performance results.

This observation is kept because a model returning a PIL image is not sufficient to claim a successful generation.

## 2. Environment & Configuration

- Checkpoint: Tongyi-MAI/Z-Image-Turbo, local Diffusers pipeline.
- Hardware: 2 × NVIDIA Tesla T4, 15,360 MiB reported memory each; compute capability 7.5.
- Software: Python 3.13.15, PyTorch 2.11.0+cu128, Diffusers 0.40.0, Transformers 5.16.1.
- Precision: FP16.
- Balanced map used: transformer → cuda:0; text_encoder → cuda:1; VAE → cuda:1.
- Sampling: 9 scheduler steps (the model card describes this as 8 DiT forwards), guidance scale 0, fixed seed.
- Prompt: a photorealistic red fox in a misty forest; same prompt and seed were used at both sizes.

## 3. Results

| Resolution | Time reported by pipeline | Peak allocated VRAM GPU 0 | Peak allocated VRAM GPU 1 | Image validity |
|---:|---:|---:|---:|---|
| 512 × 512 | 7.642 s | 11.652 GiB | 8.255 GiB | Fail: every pixel black |
| 768 × 768 | 14.539 s | 11.794 GiB | 8.998 GiB | Fail: every pixel black |

Pixel checks returned mean RGB [0, 0, 0], standard deviation 0, minimum 0 and maximum 0 for both outputs. The PNG files were extremely small because they encode uniform black images.

**Do not compare these timings as usable model performance.** They measure an invalid output path.

## 4. Interpretation & Next Step

The model card's reference path recommends bfloat16 and uses nine scheduler steps with guidance scale zero. This FP16 run was an experiment to fit a T4-compatible pipeline, not the model card's preferred precision. The black output is consistent with a numerical/precision issue, but this single test does not prove FP16 alone is the cause.

A separate BF16 test is being run with the same model, balanced placement, prompt, seed and scheduler settings. It must pass pixel-validity and human visual checks before being ranked. PyTorch reports its BF16-support query as true on this host, and simple BF16 tensor operations pass, but the T4 is compute capability 7.5; actual model quality and speed must be measured, not inferred from the API flag.

## 5. Source & Reproducibility

- Official model card: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- Local JSON record: /kaggle/working/bench_out/zimage_balanced_fp16_benchmark.json
- Reproduction script: /kaggle/working/test_zimage_balanced_fp16.py
- Invalid outputs: zimage_balanced_fp16_512_9step.png and zimage_balanced_fp16_768_9step.png in /kaggle/working/bench_out/

The official card currently lists an Apache-2.0 license and recommends bfloat16 on supported hardware. Keep this FP16 failure in the report history rather than silently replacing it with the later BF16 result.
