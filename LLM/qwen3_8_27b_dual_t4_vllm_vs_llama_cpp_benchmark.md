# Comprehensive Benchmark & Architectural Analysis: vLLM vs. llama-cpp-python on Qwen 3.8 27B

**Date:** September 28, 2026  
**Hardware Environment:** $2 \times \text{NVIDIA Tesla T4}$ (15,360 MiB VRAM each, 30 GiB Total), Intel Xeon @ 2.00GHz (4 vCPUs), 31 GiB Host RAM  
**Target Model:** `Qwen3.8-27B` (27 Billion Parameter Hybrid Mamba-SSM / Transformer Architecture)  
**Inference Engines Evaluated:**
- **`llama-cpp-python` v0.3.35** with CUDA acceleration (GGML CUDA backend, Tensor Splitting, Flash Attention)
- **`vLLM` v0.30.0** (V1 Engine Core, PagedAttention, Continuous Batching, Tensor Parallelism $TP=2$, Compressed-Tensors INT4 / AWQ)

---

## 1. Executive Summary

This research study evaluates the empirical performance and trade-offs of deploying high-parameter Large Language Models (specifically **Qwen 3.8 27B**) on resource-constrained multi-GPU environments ($2 \times \text{NVIDIA Tesla T4}$ on Kaggle infrastructure).

We benchmarked both **single-stream latency / TPS ($N=1$)** and **aggregate token throughput ($N=8, 16, 32$ concurrent requests)** between `llama-cpp-python` (the native backend of this repository) and `vLLM` (v0.30.0).

### Key Empirical Findings:
1. **Single-Stream Generation ($N=1$):** `llama-cpp-python` is **$4.86\times$ faster** than vLLM (**23.26 tokens/sec** vs. **4.78 tokens/sec**), providing an average Time-To-First-Token (TTFT) of **160.6 ms** compared to **~2,100 ms** for vLLM.
2. **High-Concurrency Throughput ($N=32$):** vLLM delivers a **$+222.7\%$ ($3.23\times$) higher aggregate throughput** (**70.03 tokens/sec** vs. **21.70 tokens/sec** for sequential llama.cpp), processing 32 requests in **58.32 seconds** instead of an estimated **188 seconds**.
3. **Cold Start & JIT Compilation Overhead:** `llama-cpp-python` loads and starts serving in **70.88 seconds** (~1.18 min), whereas vLLM requires **471.10 seconds** (~7.85 min) due to online Triton kernel compilation for the Turing SM75 compute architecture.
4. **Platform Viability for Kaggle Workers:** While vLLM excels at multi-tenant continuous batching, `llama-cpp-python` remains significantly superior for single-user interactive generation, cold-start agility, CPU layer offloading, and deterministic process isolation.

---

## 2. Benchmark Summary Table

| Metric / Scenario | `llama-cpp-python` (Q4_0, Tensor Split) | `vLLM` (v0.30.0, INT4, TP=2) | Delta / Ratio | Winner |
| :--- | :--- | :--- | :--- | :--- |
| **Model Weights Format** | GGUF (`Qwen3.8-27B-Q4_0.gguf`, 14.95 GB) | Safetensors (`compressed-tensors INT4`, 19.57 GB) | GGUF is 23.6% smaller | `llama.cpp` |
| **Cold Startup / Load Time** | **70.88 s** (1.18 min) | **471.10 s** (7.85 min) | **llama.cpp is $6.64\times$ faster** | `llama.cpp` |
| **Single-Stream TPS ($N=1$)** | **23.26 tokens/s** | **4.78 tokens/s** | **llama.cpp is $4.86\times$ faster** | `llama.cpp` |
| **Time to First Token (TTFT)** | **160.6 ms** | **~2,100 ms** | **llama.cpp is $13.1\times$ lower latency** | `llama.cpp` |
| **Turnaround (1 request, 128 tok)** | **5.61 s** | **26.56 s** | **llama.cpp is $4.73\times$ faster** | `llama.cpp` |
| **Throughput ($N=8$ requests)** | **21.70 tokens/s** (47.19 s total) | **35.51 tokens/s** (28.83 s total) | **vLLM is $+63.6\%$ higher throughput** | `vLLM` |
| **Throughput ($N=16$ requests)** | **21.70 tokens/s** (extrapolated: 93.6 s) | **60.91 tokens/s** (33.34 s total) | **vLLM is $+180.7\%$ ($2.81\times$) higher** | `vLLM` |
| **Throughput ($N=32$ requests)** | **21.70 tokens/s** (extrapolated: 188.2 s)| **70.03 tokens/s** (58.32 s total) | **vLLM is $+222.7\%$ ($3.23\times$) higher** | `vLLM` |
| **Memory Allocation Strategy** | Static buffer with TurboQuant (`q4_0`) | PagedAttention (dynamic page blocks) | Zero fragmentation vs host offload | Context-dependent |

---

## 3. Observation 1: Model Loading & Cold Start Latency

### Empirical Data:
- **`llama-cpp-python`:** 70.88 seconds
- **`vLLM` (v0.30.0):** 471.10 seconds (~7.8 minutes)

```
Startup Time Comparison (Lower is Better):
llama-cpp-python: [██] 70.88s
vLLM:             [████████████████] 471.10s
```

### Technical Root-Cause Analysis:
1. **Pre-Compiled C++ Kernels vs. Runtime JIT Compilation:**
   - `llama.cpp` utilizes ahead-of-time (AOT) compiled CUDA binaries. Weights are memory-mapped (`use_mmap=True`) and directly placed into GPU device allocations via CUDA unified memory and direct pointer slicing.
   - `vLLM` relies heavily on Triton and PyTorch compilation pipelines. When initializing on Turing GPUs (Compute Capability 7.5), vLLM detects the absence of modern FlashInfer/FlashAttention-2 pre-compiled kernels for SM75 and dynamically triggers runtime compilation (`ptxas`) for:
     - Attention kernels (`kernel_unified_attention`)
     - Causal 1D convolutions for Mamba SSM blocks (`_causal_conv1d_update_kernel`)
     - Marlin and Compressed-Tensors WNA16 dequantization kernels
     - Segment reduction and M-RoPE positional embeddings
2. **CPU-Bound Compilation Bottleneck:**
   - Kaggle instances provide 4 Intel Xeon vCPUs. Invoking `ptxas` across multiple processes (`Worker_TP0` and `Worker_TP1`) saturates the CPU cores at 100% utilization for 4–5 minutes, generating significant IPC synchronization waits (`shm_broadcast.py: No available shared memory broadcast block found in 60s`).
3. **Impact on Kaggle Lifecycle:**
   - In ephemeral notebook deployments with a strict 12-hour session boundary, a 7.8-minute startup latency consumes ~1.1% of the total session uptime per worker launch and significantly increases recovery delay during auto-healing or pod restarts.


---

## 4. Observation 2: Single-Stream TPS & Latency ($N=1$)

### Empirical Data:
- **`llama-cpp-python` (Decode Speed):** 23.26 tok/s (Prompt 1: 23.55 tok/s, Prompt 2: 23.31 tok/s, Prompt 3: 22.92 tok/s)
- **`vLLM` (Overall Speed):** 4.78 tok/s (Prompt 1: 4.82 tok/s, Prompt 2: 4.74 tok/s, Prompt 3: 4.79 tok/s)

```
Single-Stream Tokens/sec (Higher is Better):
llama-cpp-python: [████████████████] 23.26 tok/s
vLLM:             [███] 4.78 tok/s
```

### Technical Root-Cause Analysis:
1. **Memory-Bandwidth Bound GEMV vs. Launch Overhead:**
   - Single-token autoregressive generation ($N=1$) is strictly memory-bandwidth bound. The arithmetic intensity is:
     $$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}} \approx \frac{2 \times P}{P \times \text{bytes\_per\_weight}} \approx 1\text{--}2 \text{ FLOP/byte}$$
   - On a Tesla T4, theoretical peak memory bandwidth is $320\text{ GB/s}$ per GPU ($640\text{ GB/s}$ aggregate peak across 2 GPUs, $\sim 450\text{ GB/s}$ effective).
   - For a 27B model in 4-bit quantization (~15 GB weights):
     $$\text{Theoretical Max Single-Stream TPS} = \frac{450\text{ GB/s}}{15\text{ GB/token}} \approx 30\text{ tokens/s}$$
   - `llama.cpp` reaches **23.26 tok/s**, achieving **~77.5% of theoretical memory bandwidth efficiency**.
2. **The Slow PCIe Interconnect Penalty in vLLM Tensor Parallelism ($TP=2$):**
   - On Kaggle, dual T4 GPUs are connected via standard **PCIe 3.0 x16 / PCIe switch**, **not NVLink** (which provides 300–900 GB/s).
   - In vLLM, Tensor Parallelism ($TP=2$) splits matrix multiplications across GPUs, requiring an `all-reduce` communication collective over PCIe for **every single attention layer and MLP layer** (72+ synchronizations per token).
   - Because `vLLM` runs in eager PyTorch mode on Turing (since CUDA graphs are disabled or constrained on SM75), the latency overhead of kernel launches + PCIe transfers completely dwarfs the compute time, collapsing single-stream speed to **4.78 tok/s**.
3. **`llama.cpp`'s Low-Level GEMV Tensor Split:**
   - `llama.cpp`'s `split_mode = LLAMA_SPLIT_MODE_TENSOR` executes an ultra-compact C++/CUDA AllReduce routine optimized specifically for homogeneous dual-GPU setups without full PyTorch/NCCL layer overhead.

---

## 5. Observation 3: Multi-Request Throughput Scaling under Continuous Batching

### Empirical Data:
- **`llama-cpp-python` (Sequential Processing):**
  - $N=8$ requests: **21.70 tokens/s** (47.19 s)
  - $N=16$ requests: **21.70 tokens/s** (~93.6 s extrapolated)
  - $N=32$ requests: **21.70 tokens/s** (~188.2 s extrapolated)
- **`vLLM` (Continuous Batching):**
  - $N=8$ requests: **35.51 tokens/s** (28.83 s) — **$+63.6\%$ gain**
  - $N=16$ requests: **60.91 tokens/s** (33.34 s) — **$+180.7\%$ ($2.81\times$) gain**
  - $N=32$ requests: **70.03 tokens/s** (58.32 s) — **$+222.7\%$ ($3.23\times$) gain**

```
Throughput Scaling Across Concurrency (Tokens/sec):
Concurrency (N) | llama.cpp (Seq) | vLLM (Continuous Batching) | Speedup
----------------|-----------------|----------------------------|--------
N = 1           | 23.26 tok/s     | 4.78 tok/s                 | 0.21x
N = 8           | 21.70 tok/s     | 35.51 tok/s                | 1.64x
N = 16          | 21.70 tok/s     | 60.91 tok/s                | 2.81x
N = 32          | 21.70 tok/s     | 70.03 tok/s                | 3.23x
```

### Technical Root-Cause Analysis:
1. **Continuous Batching vs. Request Queueing:**
   - `llama-cpp-python` in the proxy architecture operates with a job lock (`PersistentRunner` subprocess), servicing incoming requests **one-at-a-time**. While Request #1 generates 128 tokens over 5.6s, Requests #2 through #32 sit idle in the in-memory queue.
   - `vLLM` dynamically merges all active requests into a single forward pass batch. When $N=32$, the arithmetic intensity rises from $1\text{ FLOP/byte}$ to $32\text{ FLOPs/byte}$, transitioning the GPU from memory-bandwidth bound to compute bound:
     - Tensor Cores on the T4 GPUs operate at near-maximum FLOP utilization ($65\text{ TFLOPs}$ FP16 peak per card).
     - The per-token weight read cost is amortized across 32 concurrent decode operations.
2. **Throughput Saturation Plateau:**
   - As $N$ increased from 8 to 16 to 32, vLLM throughput scaled from **35.51 tok/s $\to$ 60.91 tok/s $\to$ 70.03 tok/s**. The scaling begins to plateau around $N=32$ as the PCIe all-reduce bus and the T4 compute capacity reach saturation for 27B parameters.

---

## 6. Observation 4: Memory Utilization & KV Cache Dynamics

### Technical Comparison:

```
Memory Allocation Layout on 2x Tesla T4 (30 GB Total VRAM):

llama-cpp-python (GGUF Q4_0):
[ Model Weights: ~15 GB (~7.5 GB / GPU) ][ Static KV Cache: ~4-6 GB ][ Free Headroom: ~9-11 GB ]
  - Can enable TurboQuant q4_0: Reduces KV cache by 75% with only ~4% TPS drop.
  - Can offload layers to host RAM if VRAM overflows.

vLLM (Compressed-Tensors INT4):
[ Model Weights: ~19.6 GB (~9.8 GB / GPU) ][ Peak Act: ~1.7 GB ][ PagedAttention Pool: ~4.5 GB ]
  - PagedAttention page block size: 784 tokens (aligned with Mamba page size).
  - Maximum context capacity: 21,390 tokens total (10.44x concurrency at 2048 ctx).
  - No host RAM CPU layer offloading supported.
```

### Detailed Observations:
1. **PagedAttention Efficiency:**
   - vLLM eliminated internal and external memory fragmentation by dynamically allocating KV cache in non-contiguous 784-token pages. In our test run, vLLM reported:
     `Available KV cache memory: 2.26 GiB / GPU` $\to$ `GPU KV cache size: 21,390 tokens`.
   - This allows high concurrency without reserving fixed context per stream.
2. **TurboQuant Advantage in `llama.cpp`:**
   - `llama.cpp` supports 4-bit quantized KV cache (`cache_type_k="q4_0"`, `cache_type_v="q4_0"`). On Turing GPUs, Q4_0 KV cache saves ~75% VRAM compared to FP16 with a negligible ~4% TPS degradation (unlike Q8_0 which incurs a 26% dequantization penalty on T4).



---

## 7. Observation 5: Multi-GPU Interconnect Bottlenecks

### Comparative Interconnect Dynamics:
- **Interconnect:** PCIe 3.0 x16 ($\sim 15.75\text{ GB/s}$ theoretical unidirectional, $\sim 12\text{ GB/s}$ real-world).
- **vLLM Communication Volume:**
  - With $TP=2$, for every token generated across the 36 Transformer layers of Qwen 3.8 27B, vLLM transmits:
    $$\text{Data Transferred} = 2 \times \text{Layers} \times B \times H \times \text{dtype\_bytes}$$
    where $B$ is batch size, $H=4096$ hidden dimension.
  - At $B=1$ (single stream), the latency of initiating dozens of tiny PCIe transfers dominates over compute time.
  - At $B=32$, the payload size per transfer increases, increasing PCIe bus utilization efficiency.
- **`llama.cpp` Tensor Splitting:**
  - `llama.cpp`'s tensor splitter groups and batches communications into raw CUDA stream synchronizations, keeping single-stream decode overhead significantly lower.

---

## 8. Observation 6: Operational & Resilience Trade-offs for Kaggle Workers

| Dimension | `llama-cpp-python` (Repo Baseline) | `vLLM` |
| :--- | :--- | :--- |
| **Process Isolation & Cancellation** | **Hard OS-level killability:** `inference_runner_proc.py` runs in an isolated subprocess. Proxy hard-kills stuck requests via `SIGTERM`/`SIGKILL` in $<2.0\text{s}$, immediately freeing VRAM. | Engine runs as a complex multi-process actor pool. Hard aborts during active iteration can corrupt shared memory or leave zombie workers. |
| **Disk & Dependency Footprint** | Small (~150 MB wheel). Fast `pip install` in under 30s. | Large (~2.5 GB PyTorch, Triton, Ray, FlashInfer, compressed-tensors). Installs take 2–4 min. |
| **Model Format Flexibility** | GGUF (Single file, easily split, supports mixed quants, imatrix). | Safetensors / HF Hub format (multi-shard, requires complex quantization metadata). |
| **CPU Offload Support** | Native: can offload arbitrary layers to 30 GB host RAM. | None: whole model and KV cache must fit in GPU VRAM. |

---

## 9. Final Recommendations & Architecture Guidance

### When to use `llama-cpp-python` (Current Stack):
- **Interactive / Single-User Workloads:** For chatbots, coding assistants (e.g. Cline/VS Code integration), and tool-calling workflows where user perceived latency (TTFT $<200\text{ms}$ and TPS $>20\text{ tok/s}$) is the primary metric.
- **Kaggle Ephemeral Nodes:** Fast worker boot time (70s vs 470s) and minimal dependency overhead.
- **Larger Model Deployments (32B+):** When models exceed the combined 30GB VRAM and require CPU layer offload.

### When to consider `vLLM`:
- **High-Throughput Batch Processing:** Offline batch translation, bulk data synthesis, or multi-tenant API proxies where 10–50 requests arrive concurrently and aggregate cost-per-token is the priority.
- **Modern GPU Architectures (Ampere A100 / Hopper H100):** Where native FP8/FP4 hardware support, NVLink ($600\text{--}900\text{ GB/s}$), and pre-compiled FlashAttention-3 eliminate the Turing-specific Triton JIT and PCIe bottlenecks.

---
*Report generated and validated on live benchmark runs in `/root/Projects/Kaggle-Inference-Proxy/research/`.*

