# Real-ESRGAN Dual-T4 Upscaler Comparison: Speed, VRAM & Image Type

## 1. Executive Summary

Three x4 upscalers were measured on GPU 1 of the dual-T4 host: Real-ESRGAN x4plus, realesr-general-x4v3, and RealESRGAN_x4plus_anime_6B. All produced the expected four-times output dimensions in the tested cases.

The fastest general-purpose candidate was **realesr-general-x4v3**: its warm median for 1024 × 1024 → 4096 × 4096 was 0.359 seconds in one set of three repeats and 0.306 seconds on the Pony illustration in a separate comparison. **RealESRGAN_x4plus_anime_6B** took 1.604 seconds on the same 1024-square Pony input, while standard **RealESRGAN_x4plus** took 5.107 seconds.

These are GPU inference timings. Saving a compressed 4K PNG on the two-core host usually added around 1.6–2.1 seconds and must be measured independently.

## 2. Test Environment & Method

- Hardware: GPU 1, NVIDIA Tesla T4, 15,360 MiB reported device memory.
- Precision: FP16, inference mode.
- Inputs were RGB images; output was clamped to [0, 1] and written to PNG.
- The standard scaling test warmed each model and timed three inferences per input. The reported median is the median of those three warm runs; model loading and PNG encoding are excluded from that column.
- Separate Pony illustration comparison used the same 1024 × 1024 source image for all three models, also with three warm inference repeats.
- VRAM values are peak allocated PyTorch memory during the measured upscale call, not total process/device occupancy.

## 3. Standard Resolution-Scaling Results

| Model | Input | Output | Warm median inference | Min / max of 3 runs | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|
| RealESRGAN_x4plus | 512 × 512 | 2048 × 2048 | 1.333 s | 1.326 / 1.335 s | 2.080 GiB |
| RealESRGAN_x4plus | 768 × 768 | 3072 × 3072 | 3.021 s | 2.996 / 3.040 s | 4.640 GiB |
| RealESRGAN_x4plus | 1024 × 1024 | 4096 × 4096 | 5.423 s | 5.369 / 5.560 s | 8.225 GiB |
| RealESRGAN_x4plus | 1216 × 832 | 4864 × 3328 | 5.489 s | 5.424 / 5.592 s | 7.937 GiB |
| realesr-general-x4v3 | 512 × 512 | 2048 × 2048 | 0.090 s | 0.090 / 0.091 s | 0.176 GiB |
| realesr-general-x4v3 | 768 × 768 | 3072 × 3072 | 0.202 s | 0.192 / 0.209 s | 0.392 GiB |
| realesr-general-x4v3 | 1024 × 1024 | 4096 × 4096 | 0.359 s | 0.358 / 0.362 s | 0.696 GiB |
| realesr-general-x4v3 | 1216 × 832 | 4864 × 3328 | 0.349 s | 0.347 / 0.351 s | 0.673 GiB |

The large speed gap is real for this software / hardware setup: at 1024-square input, general x4v3 used around one-twelfth the peak allocated VRAM and took around one-fifteenth the inference time of x4plus. This does not establish equal perceptual quality; the models have different restoration behavior and must be selected against the source image and intended use.

## 4. Illustration-Specific Head-to-Head

Source: Pony V6 XL score-tag image, 1024 × 1024. All outputs were 4096 × 4096.

| Upscaler | Warm median inference | Three runs | Peak allocated VRAM | PNG save time | Output PNG |
|---|---:|---|---:|---:|---:|
| RealESRGAN_x4plus_anime_6B | 1.604 s | 1.603 / 1.604 / 1.619 s | 8.202 GiB | 1.644 s | 15.83 MB |
| realesr-general-x4v3 | 0.306 s | 0.306 / 0.305 / 0.306 s | 0.696 GiB | 1.865 s | 17.96 MB |
| RealESRGAN_x4plus | 5.107 s | 5.074 / 5.107 / 5.129 s | 8.225 GiB | 2.021 s | 20.12 MB |

The anime-focused model is a strong middle ground for the tested cel-shaded character illustration: it was about 3.2× faster than standard x4plus and used slightly less allocated VRAM, while targeting illustration / anime restoration. The general x4v3 model remained dramatically faster and more memory efficient. The PNG file size is not a reliable image-quality metric; visual review of downsampled contact sheets showed similar overall composition, but no objective perceptual metric or blind preference study has yet been run.

## 5. Practical Interpretation

1. **Throughput-first:** use realesr-general-x4v3. Its inference cost is sub-second even for ~1.5 MP input, and its sub-0.7-GiB peak allocation leaves most of GPU 1 available.
2. **Anime / cel-shaded art:** evaluate RealESRGAN_x4plus_anime_6B first. The measured 1.604-second warm inference is still quite fast compared with standard x4plus.
3. **General restoration:** retain RealESRGAN_x4plus when its restoration characteristics are preferred, but account for several seconds of inference and up to about 8.2 GiB allocated VRAM at 1024-square input.
4. **Optimize the whole pipeline:** use PNG compression level 1 or consider JPEG/WebP where permitted, offload file encoding if practical, and record save time separately from GPU inference.
5. **Avoid output-size fallacy:** upscaling creates a larger raster and can improve apparent sharpness, but a 4096-square upscaled raster does not contain the same native diffusion detail as a 4096-square generation.

## 6. Source References

- Real-ESRGAN official implementation and model descriptions: https://github.com/xinntao/Real-ESRGAN
- Real-ESRGAN official inference/model documentation: https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md
- The general x4v3 model is intended as a compact general image restoration model; x4plus is the standard general-image restoration model; anime_6B targets anime-style content. Confirm the exact weight file and revision when reproducing.

## 7. Reproducibility & Artifacts

Result JSON files:
- /kaggle/working/bench_out/upscaler_repeated_benchmark.json
- /kaggle/working/bench_out/pony_upscaler_comparison.json
- /kaggle/working/bench_out/flux2klein_1408_upscale.json

The corresponding PNGs remain in /kaggle/working/bench_out/. Measurements are tied to this T4 host and installed stack, not universal latency guarantees.
