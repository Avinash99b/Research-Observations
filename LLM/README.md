# LLM Research — Dual Tesla T4

Observations, reproducible benchmarks, and architectural analyses for Large Language Models running on dual NVIDIA Tesla T4 GPUs (30.72 GB total VRAM).

Reports distinguish measured inference latency, prompt ingestion throughput (prefill / PP), token generation speed (decode / TG), KV cache compression, peak VRAM allocations, and task output validity.

---

## Hardware Environment Baseline

- **GPUs:** 2 × NVIDIA Tesla T4 (TU104, 15,360 MiB reported each, ~14,911 MiB CUDA visible capacity, compute capability 7.5).
- **CPU / Host RAM:** Intel Xeon @ 2.00–2.20 GHz (4 vCPUs), ~31–32 GiB Host RAM.
- **Interconnect:** PCIe Gen3 x16 (PHB topology, 5.66 GB/s host-staged effective P2P bandwidth, no NVLink).
- **Environment:** CUDA 12.8 / 13.0, Driver 580.x.

---

## Reports Index

### Full Models & Deep Context Scaling
- [TheDrummer Artemis-31B v1.2 Dual-T4 Full Performance & 262k Context Benchmark](artemis_31b_v1_2_dual_t4_performance_and_context_benchmark.md) — Comprehensive multi-configuration evaluation achieving full trained context (262,144 tokens) usability on dual T4s with tensor parallelism and CPU embedding offload; includes sweeps of layer vs tensor split, microbatch, Flash Attention, and draft speculation.
- [Artemis-31B Dual-T4 Prefill & Decode TPS Factor Attribution Report](artemis_31b_v1_2_dual_t4_tps_factors_prefill_and_decode_attribution.md) — Rigorous factor-by-factor empirical attribution of prompt processing (prefill) and token generation (decode) throughput across all 45 measured test variations.
- [Qwen 3.6 35B A3B Context Length vs TTFT and TPS Benchmark](qwen_3.6_35b_a3b_dual_t4_context_length_vs_ttft_and_tps_benchmark.md) — Extreme context scaling up to 262,144 tokens, 250k token prompt ingestion, and KV cache quantization comparison (`FP16`, `Q8_0`, `TurboQuant Q4_0`).
- [Qwen3.8-27B Dual-T4 vLLM vs llama.cpp Benchmark](qwen3_8_27b_dual_t4_vllm_vs_llama_cpp_benchmark.md) — Head-to-head comparison between vLLM and llama.cpp runtimes on dual T4 hardware.

### Speculative Decoding & Latency Optimization
- [Qwen2.5-Coder-3B Dual-T4 Suffix Decoding Benchmark](qwen2_5_coder_3b_dual_t4_suffix_decoding_benchmark.md) — Evaluation of training-free speculative suffix tree decoding over prompts and output histories, yielding up to 7.09x speedups on repetitive JSON structures.
- [Qwen3.8-27B Dual-T4 vLLM vs llama.cpp MTP Speculative Decoding Analysis](qwen3_8_27b_dual_t4_vllm_vs_llama_cpp_mtp_speculative_decoding_analysis.md) — Multi-token prediction and speculative draft decoding analysis across runtimes.

### Embeddings & Vector Throughput
- [Qwen3-Embedding-4B Dual-T4 FP16 Vector Throughput Benchmark](qwen3_embedding_4b_dual_t4_f16_vector_throughput_benchmark.md) — Dense vector embedding generation throughput and latency on dual T4.
- [Qwen3-Embedding-4B Dual-T4 Quantization Comparison](qwen3_embedding_4b_dual_t4_quantization_comparison.md) — Quantization precision effects on vector generation throughput and memory.
- [Nomic Embed Text v1.5 Dual-T4 Quantization Comparison](nomic_embed_text_v1_5_dual_t4_quantization_comparison.md) — Embedding latency, throughput, and VRAM scaling across quantization levels.
- [Qwen3 vs Nomic Embed Text Head-to-Head Comparison](qwen3_vs_nomic_embed_text_dual_t4_head_to_head_comparison.md) — Comparative analysis of embedding models on dual T4 GPUs.

---

## Companion Artifacts
- [Artemis-31B Measurements CSV](artemis_31b_v1_2_dual_t4_measurements.csv) — Complete raw CSV of all 45 measured benchmark configurations, memory samples, and timings.
- [Artemis-31B Recommended Commands](artemis_31b_v1_2_dual_t4_recommended_commands.sh) — Shell script providing verified launch commands for Maximum Context (`262k`), Balanced (`98k`), and Speed (`Draft`) profiles.
- [Artemis-31B Raw Evidence Archive](artemis_31b_v1_2_dual_t4_evidence.zip) — Compressed archive containing all server logs, raw timing JSONs, and hardware probe dumps.
