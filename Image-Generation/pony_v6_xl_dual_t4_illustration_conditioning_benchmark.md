# Pony V6 XL Dual-T4 Illustration Benchmark & Conditioning Findings

## 1. Executive Summary

Pony V6 XL was tested as an open-weight illustration checkpoint on a single Tesla T4 of the dual-T4 host, with a Pony-community SDXL VAE and a Diffusers-native pipeline. Several early runs produced badly washed-out / near-blank outputs even though the pipeline returned valid PNGs. Those runs are retained as failed configurations rather than being counted as successful model output.

The successful configuration was: **Pony V6 XL + the community Pony SDXL VAE + score-tag prompt + Euler Ancestral + 25 steps + CFG 7 + no clip_skip**. It produced coherent anthropomorphic fox illustration outputs at 768² and 1024². The 1024² run took 35.524 seconds and used 9.734 GiB peak allocated VRAM. A targeted anime upscaler then produced a 4096² output from the 1024² image in 1.604 seconds of warm GPU inference.

These are character-illustration test prompts only; this benchmark did not test explicit content or assert a content-safety policy classification for the checkpoint.

## 2. Test Environment

- Date: 2026-10-13 UTC.
- Hardware: NVIDIA Tesla T4 on GPU 0, 15,360 MiB reported total; Intel Xeon CPU @ 2.00 GHz, 2 cores / 4 threads reported; 31 GiB system memory.
- Python 3.13.15; PyTorch 2.11.0+cu128; Diffusers 0.40.0; Transformers 5.16.1.
- Illustration checkpoint: Bakanayatsu/Pony-Diffusion-V6-XL-for-Anime, using the Diffusers-native SDXL repository layout.
- VAE: sdxl_vae.safetensors from LyliaEngine/Pony_Diffusion_V6_XL.
- Scheduler: Euler Ancestral.
- Inference: FP16, 25 steps, CFG 7; no clip_skip argument passed in successful runs.
- GPU timing excludes model load and PNG encoding unless specified.

## 3. Conditioning / VAE Troubleshooting

| Configuration | Prompt style | Resolution | Result | Interpretation |
|---|---|---:|---|---|
| Original single-file Pony checkpoint through generic single-file SDXL load | score tags + clip_skip 2 | 1024² | Near-blank / washed-out | Rejected |
| Diffusers Pony repo, default VAE, clip_skip 2 | score tags | 768² and 1024² | Near-blank / washed-out | Rejected |
| Diffusers Pony repo + community Pony VAE, clip_skip 2 | score tags | 768² and 1024² | Near-blank / washed-out | Rejected |
| Diffusers Pony repo + community Pony VAE, no clip_skip | plain natural-language prompt, CFG 5 | 768² | Coherent illustrated fox character | Pass |
| Diffusers Pony repo + community Pony VAE, score tags, no clip_skip | CFG 7 | 768² and 1024² | Coherent stylized character illustration | Pass |

The key practical discovery was to remove clip_skip for this exact model/configuration while using the Pony-specific VAE. The default or incompatible setting combination caused a false negative: a PNG existing on disk was not sufficient evidence the model was usable. The comparison was exploratory, so it does not prove that clip_skip is universally wrong for every Pony checkpoint or pipeline revision.

## 4. Successful Native Generation Results

| Prompt conditioning | Resolution | Steps | CFG | Inference time | Peak allocated VRAM | Output PNG size |
|---|---:|---:|---:|---:|---:|---:|
| Plain prompt, no clip_skip, custom VAE | 768 × 768 | 25 | 5 | 17.375 s | 8.422 GiB | 1.01 MB |
| Pony score tags, no clip_skip, custom VAE | 768 × 768 | 25 | 7 | 18.050 s | 8.422 GiB | 1.13 MB |
| Pony score tags, no clip_skip, custom VAE | 1024 × 1024 | 25 | 7 | 35.524 s | 9.734 GiB | 1.83 MB |

The successful 1024-square result showed a coherent anthropomorphic fox ranger, forest composition, layered costume and stylized shading. This is qualitative visual inspection only, not a benchmark score or a claim that this is the best possible Pony prompt.

## 5. Illustration Upscaler Comparison on the Same 1024² Input

Each upscaler generated a 4096 × 4096 image from pony_scoretags_noclip_1024_25step.png. Times below are warm medians from three inference runs.

| Model | x4 inference time | Peak allocated VRAM | PNG encode/save time | Output PNG size |
|---|---:|---:|---:|---:|
| RealESRGAN_x4plus_anime_6B | 1.604 s | 8.202 GiB | 1.644 s | 15.83 MB |
| realesr-general-x4v3 | 0.306 s | 0.696 GiB | 1.865 s | 17.96 MB |
| RealESRGAN_x4plus | 5.107 s | 8.225 GiB | 2.021 s | 20.12 MB |

For the illustration input, anime_6B was a useful speed/VRAM middle ground: it was about 3.2× faster than standard x4plus and explicitly targets anime-like content. General x4v3 was much faster and used less memory; it remains the throughput-first choice. File size should not be used as a proxy for perceptual quality.

## 6. Key Observations

1. **Validate pixels, not just successful return codes.** Early runs completed without exceptions but produced unusable images. A benchmark harness should include image-statistic checks and a human-visible contact sheet before recording a model/configuration as a quality success.
2. **Correct model conditioning is part of the model result.** The custom VAE and no-clip-skip setup changed the result from a near-blank output to a recognizable character illustration in this tested stack.
3. **1024² is feasible on one T4 but slow at 25 steps.** The 1024 run peaked at 9.734 GiB and took 35.524 seconds, making this a visual-quality / style-specific option rather than the fastest route.
4. **Specialized upscaling is worth comparing by image class.** Anime_6B produced the target dimensions faster than x4plus while preserving the composition in contact-sheet inspection. The general x4v3 candidate is fastest when minimal latency matters most.
5. **High-resolution PNG encoding costs time.** Save time was 1.6–2.0 seconds, a meaningful fraction of total sequential generation-plus-upscale time on the host.
6. **Prompt test coverage remains narrow.** This experiment used anthropomorphic fox illustrations. Additional scene types, human anatomy, hands, text rendering, line-art preservation and a blind preference study are required before making broad quality claims.

## 7. Current Recommendation

For illustration-specific local inference, start with the Diffusers-native Pony checkpoint, a compatible Pony VAE, 25 steps, CFG 7, Euler Ancestral, score tags and no clip_skip. Start at 768² for iteration; use 1024² when the extra native detail justifies roughly double the inference latency. For x4 output, compare anime_6B against general x4v3 on the actual art style.

Do not apply this recipe blindly to different Pony merges, Diffusers versions or VAE files. Confirm output quality after loading, record checkpoint revisions and keep the failed configurations in the benchmark history.

## 8. Reproducibility & Artifacts

- Native generation record: /kaggle/working/bench_out/pony_tags_noclip_benchmark.json
- Plain-prompt diagnostic: /kaggle/working/bench_out/pony_conditioning_test.json
- VAE diagnostic: /kaggle/working/bench_out/pony_customvae_benchmark.json
- Upscaler comparison: /kaggle/working/bench_out/pony_upscaler_comparison.json
- Main successful illustration: /kaggle/working/bench_out/pony_scoretags_noclip_1024_25step.png

The above paths refer to the active research workspace. Exact timing and conditioning behavior can change with pipeline, scheduler, VAE and model revisions.
