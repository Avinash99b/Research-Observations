# Benchmark & Architectural Performance: Qwen3 Embedding 4B F16

## 1. Executive Summary

This report documents the deployment, architectural compatibility, bug fixes, and performance benchmark metrics for **`Qwen/Qwen3-Embedding-4B-GGUF`** (variant `Qwen3-Embedding-4B-f16.gguf`) running on NVIDIA Tesla T4 GPU workers.

The model generates **2,560-dimensional dense float vectors** with complete OpenAI `/v1/embeddings` API specification compliance.

---

## 2. Model & Infrastructure Configuration

| Parameter | Value |
| :--- | :--- |
| **Model Repository** | `Qwen/Qwen3-Embedding-4B-GGUF` |
| **Model File** | `Qwen3-Embedding-4B-f16.gguf` |
| **Precision / Quantization** | FP16 (Full Precision Weights) |
| **Disk & Weight Size** | ~7.50 GB |
| **Vector Embedding Dimension** | **2,560** |
| **Hardware** | Single Tesla T4 GPU (`CUDA_VISIBLE_DEVICES=0`, 15.36 GB VRAM) |
| **VRAM Consumption** | ~7.65 GB / 15.36 GB |
| **Worker Host ID** | `nb-9aaad8e656` |
| **Endpoint Gateway** | `POST /v1/embeddings` |

---

## 3. Technical Findings & Architectural Compatibility

### 3.1. MoE Generation Models vs. Dedicated Embedding Models
- **Qwen3.6-35B-A3B (MoE)**: Attempting to instantiate generation models with `embedding=True` triggers native C++ assertion failures in `llama.cpp` (`llama_memory_recurrent` graph allocation). Mixture-of-Experts architectures with recurrent attention heads are currently unsupported for `create_embedding()` in `llama-cpp-python`.
- **Qwen3-Embedding-4B**: Dedicated embedding models load cleanly with `ENABLE_EMBEDDINGS=1` and initialize an embedding compute graph (`n_embd=2560`, `n_layer=36`).

### 3.2. Multi-GPU Tensor Split vs. Single-GPU Execution
- **Multi-GPU Tensor-Split Issue**: Running embedding inference across multiple GPUs (`split_mode=3` / tensor split across 2 Tesla T4 GPUs) triggers an assertion inside `llama_decode` (`GGML_ASSERT(offset == 0)`) when evaluating sequence batches.
- **Resolution**: Embedding worker nodes must be pinned to a single GPU (`CUDA_VISIBLE_DEVICES=0`). Because `Qwen3-Embedding-4B-f16` requires ~7.65 GB VRAM, it comfortably fits inside a single Tesla T4 GPU (15.36 GB VRAM).

### 3.3. Worker Subprocess Multi-String Array Handling
- **Issue**: Standard `llm.create_embedding(input=list_of_strings)` in `llama-cpp-python` sets multi-sequence offsets in `llama_batch`, triggering `GGML_ASSERT(offset == 0)` in `llama.cpp`.
- **Fix in `inference_runner_proc.py`**: Updated `run_embeddings_job()` to evaluate input string arrays sequentially per request and aggregate prompt token counts and output vector arrays before returning `done` to the proxy.

---

## 4. Benchmark Performance Metrics (`POST /v1/embeddings`)

All requests were benchmarked against `https://kaggle-proxy.avinash9.in/v1/embeddings` targeting worker `nb-9aaad8e656`.

| Batch Size | Prompt Tokens | Latency (sec) | Embedding Throughput | Prompt Token Speed |
| :--- | :--- | :--- | :--- | :--- |
| **1 vector** | 26 tokens | **2.825 s** | **0.35 vectors/sec** | **9.2 tok/s** |
| **5 vectors** | 130 tokens | **4.863 s** | **1.03 vectors/sec** | **26.7 tok/s** |
| **10 vectors** | 260 tokens | **6.599 s** | **1.52 vectors/sec** | **39.4 tok/s** |
| **20 vectors** | 530 tokens | **9.290 s** | **2.15 vectors/sec** | **57.0 tok/s** |

---

## 5. End-to-End OpenAI Verification

```bash
curl https://kaggle-proxy.avinash9.in/v1/embeddings \
  -H "Authorization: Bearer <api_key>" \
  -H "X-Worker-Id: nb-9aaad8e656" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-f16.gguf",
    "input": ["Text sample for vector embedding generation."]
  }'
```

**Response Payload**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [-0.03459, 1.85448, 3.22387, ...]
    }
  ],
  "model": "Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-f16.gguf",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```
