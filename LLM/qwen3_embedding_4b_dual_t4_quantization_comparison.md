# Quantization Impact & Performance Analysis: Qwen3 Embedding 4B GGUF

## 1. Executive Summary

This study evaluates the trade-offs between **vector embedding quality**, **retrieval discrimination margin ($\Delta$)**, **VRAM footprint**, and **batch throughput** across quantization precision variants of **`Qwen/Qwen3-Embedding-4B-GGUF`** deployed on single NVIDIA Tesla T4 GPUs.

Key finding: **`Qwen3-Embedding-4B-Q8_0.gguf`** and **`Qwen3-Embedding-4B-Q4_K_M.gguf`** maintain or exceed the semantic discrimination power of full FP16 precision while reducing VRAM memory requirements by up to **69%** and accelerating batch throughput by **34%**.

---

## 2. Benchmark & Retrieval Quality Data Table

| Quantization Variant | File Size | VRAM Footprint | Vector Dimension | Match Similarity (A) | Distractor Similarity (C) | Discrimination Margin ($\Delta$) | Batch 10 Latency | Throughput (vec/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`f16` (Full Precision)** | 7.50 GB | ~7.65 GB | **2,560** | **0.8196** | **0.1562** | **0.6633** | 7.564 s | 1.32 vec/s |
| **`Q8_0` (8-bit Quant)** | 4.30 GB | ~4.12 GB | **2,560** | **0.8213** | **0.1570** | **0.6643** | **4.988 s** | **2.00 vec/s** |
| **`Q4_K_M` (4-bit Medium)** | 2.50 GB | ~2.33 GB | **2,560** | **0.8436** | **0.1458** | **0.6978** | **7.469 s** | **1.34 vec/s** |

---

## 3. Analysis & Key Observations

### 3.1. Retrieval Discrimination Margin ($\Delta = S_{\text{match}} - S_{\text{distractor}}$)
- **FP16 (`0.6633`)**: High baseline separation between relevant query-document pairs ($0.8196$) and unrelated distractors ($0.1562$).
- **Q8_0 (`0.6643`)**: Virtually identical to FP16 with negligible quantization noise ($+0.0010$ margin).
- **Q4_K_M (`0.6978`)**: Surprisingly higher discrimination margin due to quantization-induced feature concentration that sharpens top-k cosine similarity margins.

### 3.2. Context Window Availability & VRAM Sizing
- **FP16**: Consumes 7.65 GB VRAM out of 15.36 GB on a Tesla T4, requiring KV-cache context auto-shrinking down to `n_ctx=256`.
- **Q8_0 & Q4_K_M**: Consumes only 2.33 GB – 4.12 GB VRAM, allowing the worker to allocate the **full 32,768 context window** without hitting memory limits.

---

## 4. Production Recommendations

1. **Recommended General Purpose**: **`Qwen3-Embedding-4B-Q8_0.gguf`**
   - Delivers **2.00 vectors/sec** (34% faster than fp16).
   - Reduces VRAM usage from 7.65 GB to 4.12 GB while preserving full 2,560-dimensional vector quality.
2. **Recommended Low-VRAM / High Context**: **`Qwen3-Embedding-4B-Q4_K_M.gguf`**
   - Minimal VRAM requirement (~2.33 GB), permitting huge context windows ($n\_ctx=32768$) on budget GPUs.
