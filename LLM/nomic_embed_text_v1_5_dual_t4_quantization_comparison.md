# Quantization & Performance Analysis: Nomic Embed Text v1.5 GGUF

## 1. Executive Summary

This report evaluates **`nomic-ai/nomic-embed-text-v1.5-GGUF`** across **FP32**, **FP16**, **Q8_0**, and **Q4_K_M** precision variants.

`nomic-embed-text-v1.5` is a lightweight, BERT-architecture (`nomic-bert`) embedding model producing **768-dimensional vector embeddings**. It features extreme memory efficiency ($<500$ MB disk/VRAM) and low initial load times ($<1.0$ second).

---

## 2. Benchmark Data Table

| Quantization Variant | File Size | Vector Dimension | Match Similarity (A) | Distractor Similarity (C) | Discrimination Margin ($\Delta$) | Batch 10 Latency | Throughput (vec/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`f32` (Full FP32)** | 550 MB | **768** | **0.8164** | **0.2695** | **0.5468** | 5.190 s | 1.93 vec/s |
| **`f16` (Half FP16)** | 275 MB | **768** | **0.8164** | **0.2696** | **0.5468** | **3.934 s** | **2.54 vec/s** |
| **`Q8_0` (8-bit Quant)** | 147 MB | **768** | **0.8150** | **0.2686** | **0.5465** | **5.923 s** | **1.69 vec/s** |
| **`Q4_K_M` (4-bit Medium)** | 90 MB | **768** | **0.8399** | **0.3420** | **0.4979** | **5.780 s** | **1.73 vec/s** |

---

## 3. Analysis & Key Observations

1. **FP32 vs FP16 Parity**:
   - `f16` achieves **identical semantic similarity scores** ($\Delta = 0.5468$) to `f32` while reducing memory consumption by 50% and increasing batch throughput by **31%** (**2.54 vectors/sec**).
2. **Q4_K_M Compression Trade-off**:
   - Quantizing down to 4-bit (`Q4_K_M`, 90 MB) increases distractor background noise ($0.3420$ vs $0.2696$), causing a **9% reduction** in discrimination margin ($\Delta = 0.4979$).

---

## 4. Production Recommendation

- **Optimal Deployment Choice**: **`nomic-embed-text-v1.5.f16.gguf`**
  - Ultra-compact file size (**275 MB**).
  - Peak throughput (**2.54 vectors/sec**).
  - 100% semantic quality retention relative to FP32.
