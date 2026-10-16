# Artemis: measured Kaggle dual-T4 performance and context report

## Recommendations

Fastest shallow repetitive-prompt result: **13.63 [MEASURED] generation tokens/s** with layer-split draft decoding. This is workload dependent. Corrected chat controls and depth results are below.

Full trained context validated through **262,144 [MEASURED] tokens**, without truncation [MEASURED]. Median generation near the limit: **3.35 [MEASURED] tokens/s**. CPU placement is required as shown below.

### Maximum observed shallow speed

[MEASURED] (configuration) Numeric arguments below are tested settings. Run only one profile at a time. The speed profile benefits most from repetitive text; see the corrected chat comparison.

```bash
/kaggle/working/artemis-bench/llama-server-sm75 -m /tmp/artemis-tools/model.gguf -ngl all --fit off -sm layer -ts 1,1 -mg 0 -fa on -ctk f16 -ctv f16 -c 4096 -b 512 -ub 128 -t 4 -np 1 --host 127.0.0.1 --port 18080 --no-webui --no-warmup --model-draft /tmp/artemis-tools/draft.gguf --spec-type draft-simple --spec-draft-ngl all --spec-draft-n-max 3 --ctx-checkpoints 0 --cache-ram 0 -lv 4
```

### Maximum context

[MEASURED] (configuration) Numeric arguments below are tested settings. Run only one profile at a time. The speed profile benefits most from repetitive text; see the corrected chat comparison.

```bash
/kaggle/working/artemis-bench/llama-server-sm75 -m /tmp/artemis-tools/model.gguf -ngl all --fit off -sm tensor -ts 1,1 -mg 0 -fa on -ctk q4_0 -ctv q4_0 -c 262144 -b 2048 -ub 128 -t 4 -np 1 --host 127.0.0.1 --port 18080 --no-webui --no-warmup --override-tensor token_embd.weight=CPU --ctx-checkpoints 0 --cache-ram 0 --no-context-shift
```

### Balanced

[MEASURED] (configuration) Numeric arguments below are tested settings. Run only one profile at a time. The speed profile benefits most from repetitive text; see the corrected chat comparison.

```bash
/kaggle/working/artemis-bench/llama-server-sm75 -m /tmp/artemis-tools/model.gguf -ngl all --fit off -sm tensor -ts 1,1 -mg 0 -fa on -ctk q8_0 -ctv q8_0 -c 98304 -b 2048 -ub 128 -t 4 -np 1 --host 127.0.0.1 --port 18080 --no-webui --no-warmup --ctx-checkpoints 0 --cache-ram 0 --no-context-shift -lv 4
```

## Verification and scope

This is a bounded search for the best observed configuration, not a proof of a universal optimum. Literal model names, versions, commits, quant names and flags are identifiers. Numeric arguments in commands inherit the classification immediately above the command. Original machine logs are retained without inserting tags; the evidence README classifies those as [MEASURED].

| Environment item | Result |
|---|---|
| Hardware | Tesla T4, count 2 [MEASURED] |
| Raw VRAM per GPU | 15,360 MiB [MEASURED] |
| Driver reservation per GPU | 449 MiB [MEASURED] |
| CUDA-visible capacity per GPU | 14,911 MiB [MEASURED] |
| Initial CUDA context per GPU | about 105 MiB [MEASURED] |
| Driver / reported CUDA compatibility | `580.178.04` / `13.0` [MEASURED] (identifiers) |
| Installed CUDA compiler toolkit | `12.8.93` [MEASURED] (identifier) |
| Available CPU threads | 4 [MEASURED] |
| System RAM | 32,869,452 KiB [MEASURED] |
| Working filesystem capacity | 20,957,446,144 bytes [MEASURED] |
| Working filesystem free before model download | 20,930,285,568 bytes [MEASURED] |
| Temporary filesystem free before download | 1,145,490,419,712 bytes [MEASURED] |
| Interconnect | PHB, no NVLink path reported [MEASURED] |
| PCIe link, each GPU | generation 3, width 16 [MEASURED] |
| CUDA peer access | unsupported in topology query; direct-access test false [MEASURED] |

The GGUF would leave only **22,632,000 bytes [CALCULATED]** on the working disk before further logs. Large downloads were placed in the checked temporary filesystem. The working directory holds the evidence and convenience symlinks. GPU clocks were not locked. Temperature reached **79 degrees C [MEASURED]**, with substantial clock variation [MEASURED], so small differences in the chronological sweep are not conclusive.

The executable runs, has no missing `ldd` dependencies, contains `sm_75` cubins, and completed GPU inference [MEASURED]. It reports `0.6.0-dev`, build `0`, commit `unknown` [MEASURED] (identifiers). Its exact source and claim to be the latest build cannot be verified. SHA-256: `3d73017477cb64c9b8a9a32ae097d0a11b4ab48cab05932d5520d56edfbe8bc7` [MEASURED] (identifier).

| Capability | Runtime verification |
|---|---|
| Flash attention | enabled; on/off comparison completed [MEASURED] |
| f16, q8_0 and q4_0 symmetric KV | accepted and generated output [MEASURED] |
| Layer split | works [MEASURED] |
| Row split | flag exists, but fails: `device CUDA0 does not support split buffers` [MEASURED] |
| Tensor split | works in tested non-draft cases; experimental [MEASURED] |
| Draft model | layer-split draft decoding works [MEASURED] |
| Tensor split plus tested draft | aborts in meta-backend split-state assertion [MEASURED] |
| Mapping controls | this build uses `--load-mode`; `none` was tested [MEASURED] |

The target file has **20,907,653,568 bytes [MEASURED]** and its SHA-256 matches the repository LFS hash: `55da3ba2a0c7326110aa11234bf8a127a7f24d5fb58fc653da792cff011de920` [MEASURED] (identifier). Its tensor directory has **833 tensors [MEASURED]**, **30,697,345,596 parameters [MEASURED]**, and **20,891,818,224 tensor bytes [MEASURED]**. Q4_K_L is a mixed tensor quantization, not uniform four-bit weights.

| GGUF metadata | Value |
|---|---|
| Architecture | `gemma4` [MEASURED] |
| Layers / embedding width | 60 / 5,376 [MEASURED] |
| Query heads | 32 [MEASURED] |
| Local attention | 50 layers; KV heads 16; K/V head dimension 256; window 1,024 tokens [MEASURED] |
| Global attention | 10 layers; KV heads 4; K/V head dimension 512 [MEASURED] |
| Attention pattern | five local layers followed by one global layer, repeated [MEASURED] |
| Trained context | 262,144 tokens [MEASURED] |
| Shared KV layers | 0 [MEASURED] |
| Feed-forward width | 21,504 [MEASURED] |
| BOS | ID 2, required by metadata [MEASURED] |

Head dimensions must come from metadata. Dividing embedding width by query-head count gives the wrong cache sizes.

## Memory calculations and their limits

The ordinary layer-split load allocates **9,908.34 MiB and 10,015.73 MiB of GPU model buffers [MEASURED]**, plus **1,102.50 MiB CPU-mapped embedding [MEASURED]**. Tensor split allocates **10,515.87 MiB per GPU [MEASURED]**. Its model-layout overhead needs a separate budget. CPU-mapped bytes are not VRAM.

Let C be single-slot context, u physical microbatch, and sK/sV stored bytes per scalar. The following formulas are [CALCULATED] from the measured layout. Storage formats use **2 bytes/scalar for f16 [MEASURED] (format)**, **34/32 for q8_0 [MEASURED] (format)**, and **18/32 for q4_0 [MEASURED] (format)**, including quantization scales.

```text
S(C,u) = round_up(min(C, 1024 + u), 256)
KV(C,u) = 10*C*4*512*(sK+sV) + 50*S(C,u)*16*256*(sK+sV)
VRAM_i = weights_i + KV_i(C,u) + scratch_i(C,u,b,FA,split,KV_type) + runtime_i
```

[CALCULATED] (formulas) These apply without `--swa-full`, to a single slot. Padding and compact-window allocation were checked against logs. Each GPU must fit independently.

| Symmetric KV | Global bytes per context token | Local bytes per window cell | KV at trained limit with microbatch 128 [MEASURED] (setting) |
|---|---:|---:|---:|
| f16 | 81,920 [CALCULATED] | 819,200 [CALCULATED] | 20.98 GiB [CALCULATED] |
| q8_0 | 43,520 [CALCULATED] | 435,200 [CALCULATED] | 11.14 GiB [CALCULATED] |
| q4_0 | 23,040 [CALCULATED] | 230,400 [CALCULATED] | 5.90 GiB [CALCULATED] |

Microbatches **128, 256 and 512 [MEASURED] (settings)** allocate **1,280, 1,280 and 1,536 local cells [MEASURED]** respectively. Full-size SWA would make even q4 KV alone **61.875 GiB at the trained limit [CALCULATED]**, before weights. It cannot fit entirely in these GPUs.

Before benchmarking, a preliminary aggregate budget assumed **512 MiB runtime/scratch headroom [ESTIMATED]**, after measured driver reservation. It predicted ceilings of **107,264 f16 tokens [CALCULATED]**, **213,248 q8 tokens [CALCULATED]**, and **414,208 q4 tokens [CALCULATED]**, capped by the trained limit. These were predictions, not usable-context results. Large-context quantized scratch invalidated that headroom assumption.

### Scratch as a function of microbatch and context

| Layer/f16, context 4,096 [MEASURED] (setting) | GPU A compute MiB | GPU B compute MiB | Host compute MiB |
|---|---:|---:|---:|
| microbatch 128 [MEASURED] | 80.15 [MEASURED] | 56.52 [MEASURED] | 31.53 [MEASURED] |
| microbatch 256 [MEASURED] | 160.30 [MEASURED] | 113.04 [MEASURED] | 63.05 [MEASURED] |
| microbatch 512 [MEASURED] | 321.59 [MEASURED] | 227.07 [MEASURED] | 127.11 [MEASURED] |

At that context, total GPU scratch is approximately **1.068 MiB per physical-batch token [CALCULATED] (fit)**. At physical microbatch **128 [MEASURED]**, increasing context from **4,096 to 98,304 [MEASURED] (settings)** adds **92 MiB per GPU [MEASURED]** in the pipeline graph, matching **8*u*(C-4096) bytes [CALCULATED] (fit)** per GPU.

Quantized KV has an additional context-dependent workspace. At physical microbatch **128 [MEASURED] (setting)**, first-GPU pipeline requests are **97.77 MiB at context 4,096 [MEASURED]**, **1,789.77 MiB at context 196,608 [MEASURED] (attempted allocation)**, and **2,365.77 MiB at context 262,144 [MEASURED] (attempted allocation)**. Small-context q4 and q8 compute reservations are identical [MEASURED]. These points fit:

```text
pipeline scratch_A(C,u=128) = (8192 + 8*u)*C + 61.77 MiB
serial fallback scratch_A(C,u=128) = (8192 + 2*u)*C + 28.10 MiB
```

[CALCULATED] (empirical fits) The larger serial-fallback requests were **1,612.10 MiB and 2,140.10 MiB [MEASURED]**. The large context slope is consistent with full-cache conversion/work buffers [ESTIMATED] (interpretation); no profiler trace or verified source commit proves the exact internal cause. Extending these fits to other microbatches or split modes is [ESTIMATED].

At fixed physical microbatch **128 [MEASURED]**, changing logical batch **512 to 2,048 [MEASURED] (settings)** left shallow tensor-split compute reservation at **43.32 MiB per Meta instance [MEASURED]**. Logical and physical batches are different. Scratch is peak live graph allocation; graph fallback, attention mode, context, cache type and split matter. It cannot be modeled accurately as a fixed percentage of weights.

### Revised per-device pipeline budget

Using the measured layer weights, cache distribution and fitted scratch slopes, with **128 MiB per-device runtime headroom [ESTIMATED]**, gives the following [CALCULATED] budgets at physical microbatch **128 [MEASURED] (setting)**. These are build/graph-specific estimates, not hard allocator boundaries. Serial fallback can fit a larger context, and deeper graph allocations can consume the margin.

| KV | Revised pipeline context budget |
|---|---:|
| f16 | 105,728 tokens [CALCULATED] |
| q8_0 | 151,296 tokens [CALCULATED] |
| q4_0 | 232,192 tokens [CALCULATED] |

### Bandwidth bounds

Nominal T4 bandwidth is **320 GB/s per GPU [ESTIMATED] (input supplied by the task)**, not measured sustained kernel bandwidth. These are optimistic [CALCULATED] single-request, non-speculative, unbatched decode bounds:

```text
R_KV(C) = global_bytes_per_token*C + local_bytes_per_cell*min(C, window)
layer_step_time >= sum_i((W_i + R_KV_i(C))/BW_i)
tensor_step_time >= max_i((W_i + R_KV_i(C))/BW_i) + communication_time
row_step_time >= max_i(W_i/BW_i) + R_KV(C)/BW_main + communication_time
communication_time >= bytes_moved / link_bandwidth + synchronization_latency
```

[CALCULATED] (formulas) Ignoring KV and overhead, sequential layer splitting is bounded by **15.32 tokens/s [CALCULATED]** and ideal parallel weight reads by **30.63 tokens/s [CALCULATED]**. Minimum KV traffic assumes ideal GQA reuse; repeated reads, conversions, replicated weights, compute and synchronization reduce actual speed. Row split is unavailable in this executable [MEASURED]. Row KV would reside on the main device rather than enjoy the same sharded capacity as tensor mode.

The ideal minimum-read ceilings at selected depths are [CALCULATED], using the same nominal-bandwidth assumption. They are not achievable-throughput predictions:

| KV / depth | Minimum bytes read per target token | Layer ceiling | Ideal tensor ceiling |
|---|---:|---:|---:|
| f16 / 0 tokens [CALCULATED] (scenario) | 20,891,818,224 bytes [CALCULATED] | 15.32 tok/s [CALCULATED] | 30.63 tok/s [CALCULATED] |
| f16 / 32,768 tokens [CALCULATED] (scenario) | 24,415,033,584 bytes [CALCULATED] | 13.11 tok/s [CALCULATED] | 26.21 tok/s [CALCULATED] |
| q4_0 / 262,144 tokens [CALCULATED] (scenario) | 27,167,545,584.0 bytes [CALCULATED] | 11.78 tok/s [CALCULATED] | 23.56 tok/s [CALCULATED] |

PCIe link payload has a theoretical per-direction limit of **15.75 GB/s [CALCULATED]**. Median pinned-memory host-to-device bandwidth was **10.99 / 10.59 GB/s [MEASURED]**, device-to-host **12.12 / 11.02 GB/s [MEASURED]**, and serial host-staged GPU-to-GPU effective bandwidth **5.66 GB/s [MEASURED]**. This is a transfer probe, not NCCL collective profiling. There is no measured NVLink or peer-access path.

Speculation and concurrent-request batches amortize weight reads across tokens, so their aggregate throughput is not constrained by the single unbatched target-step bound in the same way.


## Supplied-server throughput measurements

Unless stated otherwise, cells are medians of repeated requests. Greedy sampling and EOS suppression keep generation length fixed; all tokens, including reasoning, are counted. Server decode timing excludes the first prompt-produced token. Cold-prompt reuse is disabled. VRAM is the maximum sample after completed requests, not a continuously measured peak.

### Short sweep
| Config | KV K/V | Context | Prompt depth | Batch / microbatch | Sampled VRAM MiB per GPU | pp tok/s | tg tok/s | Repeats | Notes |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| layer_f16 | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 10,809 [MEASURED] / 10,851 [MEASURED] | 231.97 [MEASURED] | 10.06 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| layer_q8 | q8_0/q8_0 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 10,511 [MEASURED] / 10,571 [MEASURED] | 227.20 [MEASURED] | 9.51 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| layer_q4 | q4_0/q4_0 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 10,341 [MEASURED] / 10,411 [MEASURED] | 223.88 [MEASURED] | 9.04 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| row_f16 | f16/f16 | 4,096 [MEASURED] | - | 512 [MEASURED] / 128 [MEASURED] | - | failed [MEASURED] | failed [MEASURED] | - | RuntimeError('server exited 1') |
| tensor_f16 | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 11,459 [MEASURED] / 11,459 [MEASURED] | 263.27 [MEASURED] | 11.06 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| layer_f16_no_fa | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 10,989 [MEASURED] / 11,031 [MEASURED] | 79.31 [MEASURED] | 8.80 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=off [MEASURED] (settings) |
| layer_f16_ub256 | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 256 [MEASURED] | 10,885 [MEASURED] / 10,903 [MEASURED] | 202.19 [MEASURED] | 9.11 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| layer_f16_ub512 | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 512 [MEASURED] | 11,155 [MEASURED] / 11,119 [MEASURED] | 213.68 [MEASURED] | 9.20 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| layer_f16_t2 | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 10,809 [MEASURED] / 10,851 [MEASURED] | 216.68 [MEASURED] | 9.19 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=2, FA=on [MEASURED] (settings) |
| layer_f16_split_4_5 | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 9,877 [MEASURED] / 11,783 [MEASURED] | 216.32 [MEASURED] | 9.07 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| row_f16_mg1 | f16/f16 | 4,096 [MEASURED] | - | 512 [MEASURED] / 128 [MEASURED] | - | failed [MEASURED] | failed [MEASURED] | - | RuntimeError('server exited 1') |
| layer_f16_nommap | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 10,811 [MEASURED] / 10,853 [MEASURED] | 215.22 [MEASURED] | 9.17 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| tensor_f16_ub256 | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 256 [MEASURED] | 11,483 [MEASURED] / 11,483 [MEASURED] | 306.17 [MEASURED] | 11.18 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| tensor_f16_ub512 | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 512 [MEASURED] | 11,699 [MEASURED] / 11,699 [MEASURED] | 333.67 [MEASURED] | 10.99 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| tensor_q8 | q8_0/q8_0 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 11,173 [MEASURED] / 11,173 [MEASURED] | 303.66 [MEASURED] | 11.63 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| tensor_q4 | q4_0/q4_0 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 11,009 [MEASURED] / 11,009 [MEASURED] | 310.53 [MEASURED] | 10.98 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| tensor_b2048 | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 11,459 [MEASURED] / 11,459 [MEASURED] | 270.94 [MEASURED] | 11.53 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| tensor_nocache | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 11,459 [MEASURED] / 11,459 [MEASURED] | 382.35 [MEASURED] | 11.06 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| tensor_p2p | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 11,459 [MEASURED] / 11,459 [MEASURED] | 261.19 [MEASURED] | 10.17 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| draft_layer | f16/f16 | 4,096 [MEASURED] | 512 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 11,411 [MEASURED] / 12,075 [MEASURED] | 283.69 [MEASURED] | 13.63 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |

### Corrected chat controls
| Config | KV K/V | Context | Prompt depth | Batch / microbatch | Sampled VRAM MiB per GPU | pp tok/s | tg tok/s | Repeats | Notes |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| chat_layer_f16 | f16/f16 | 4,096 [MEASURED] | 27 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 10,797 [MEASURED] / 10,839 [MEASURED] | 113.86 [MEASURED] | 9.55 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| chat_tensor_f16 | f16/f16 | 4,096 [MEASURED] | 27 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 11,457 [MEASURED] / 11,457 [MEASURED] | 163.06 [MEASURED] | 10.72 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| chat_tensor_q8 | q8_0/q8_0 | 4,096 [MEASURED] | 27 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 11,171 [MEASURED] / 11,171 [MEASURED] | 162.42 [MEASURED] | 12.15 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| chat_tensor_q4 | q4_0/q4_0 | 4,096 [MEASURED] | 27 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 11,007 [MEASURED] / 11,007 [MEASURED] | 149.16 [MEASURED] | 10.36 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| chat_layer_q4 | q4_0/q4_0 | 4,096 [MEASURED] | 27 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 10,329 [MEASURED] / 10,399 [MEASURED] | 113.48 [MEASURED] | 9.64 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| chat_draft_layer | f16/f16 | 4,096 [MEASURED] | 27 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 11,395 [MEASURED] / 12,059 [MEASURED] | 105.14 [MEASURED] | 10.51 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| chat_draft_tensor | f16/f16 | 4,096 [MEASURED] | - | 512 [MEASURED] / 128 [MEASURED] | - | failed [MEASURED] | failed [MEASURED] | - | RuntimeError('server exited -6') |

These use the model template and required BOS. Readable opening reasoning does not establish completion quality or long-context accuracy. The preliminary natural controls are excluded.

### Depth tests
| Config | KV K/V | Context | Prompt depth | Batch / microbatch | Sampled VRAM MiB per GPU | pp tok/s | tg tok/s | Repeats | Notes |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| depth_tensor_f16 | f16/f16 | 65,536 [MEASURED] | 1 [MEASURED] | 2,048 [MEASURED] / 512 [MEASURED] | 14,119 [MEASURED] / 14,119 [MEASURED] | 16.73 [MEASURED] | 11.53 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| depth_tensor_f16 | f16/f16 | 65,536 [MEASURED] | 4,096 [MEASURED] | 2,048 [MEASURED] / 512 [MEASURED] | 14,163 [MEASURED] / 14,163 [MEASURED] | 401.93 [MEASURED] | 11.28 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| depth_tensor_f16 | f16/f16 | 65,536 [MEASURED] | 16,384 [MEASURED] | 2,048 [MEASURED] / 512 [MEASURED] | 14,163 [MEASURED] / 14,163 [MEASURED] | 376.31 [MEASURED] | 10.98 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| depth_tensor_f16 | f16/f16 | 65,536 [MEASURED] | 32,768 [MEASURED] | 2,048 [MEASURED] / 512 [MEASURED] | 14,163 [MEASURED] / 14,163 [MEASURED] | 341.45 [MEASURED] | 10.77 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| depth_tensor_q8 | q8_0/q8_0 | 98,304 [MEASURED] | 4,096 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 13,891 [MEASURED] / 13,891 [MEASURED] | 357.86 [MEASURED] | 10.89 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| depth_tensor_q8 | q8_0/q8_0 | 98,304 [MEASURED] | 16,384 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 13,891 [MEASURED] / 13,891 [MEASURED] | 335.00 [MEASURED] | 10.21 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| depth_tensor_q8 | q8_0/q8_0 | 98,304 [MEASURED] | 32,768 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 13,891 [MEASURED] / 13,891 [MEASURED] | 302.83 [MEASURED] | 9.87 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| depth_tensor_q4 | q4_0/q4_0 | 196,608 [MEASURED] | 4,096 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 14,679 [MEASURED] / 14,679 [MEASURED] | 359.24 [MEASURED] | 10.83 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| depth_tensor_q4 | q4_0/q4_0 | 196,608 [MEASURED] | 16,384 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 14,679 [MEASURED] / 14,679 [MEASURED] | 334.70 [MEASURED] | 9.87 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| depth_tensor_q4 | q4_0/q4_0 | 196,608 [MEASURED] | 32,768 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 14,679 [MEASURED] / 14,679 [MEASURED] | 301.20 [MEASURED] | 9.38 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| depth_layer_q4 | q4_0/q4_0 | 229,376 [MEASURED] | 4,096 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 14,795 [MEASURED] / 14,865 [MEASURED] | 278.57 [MEASURED] | 8.60 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| depth_layer_q4 | q4_0/q4_0 | 229,376 [MEASURED] | 32,768 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 14,795 [MEASURED] / 14,865 [MEASURED] | 223.56 [MEASURED] | 6.61 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| balanced_tensor_q8_131072 | q8_0/q8_0 | 131,072 [MEASURED] | 4,096 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 14,835 [MEASURED] / 14,835 [MEASURED] | 361.73 [MEASURED] | 10.31 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| expA_tensor_cpuemb_ub128 | q4_0/q4_0 | 262,144 [MEASURED] | 4,096 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 14,825 [MEASURED] / 14,825 [MEASURED] | 387.70 [MEASURED] | 8.18 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| expA_tensor_depth_16384 | q4_0/q4_0 | 262,144 [MEASURED] | 16,384 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 14,825 [MEASURED] / 14,825 [MEASURED] | 332.17 [MEASURED] | 7.52 [MEASURED] | 2 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| expB_layer_balanced_ub256 | q4_0/q4_0 | 262,144 [MEASURED] | 4,096 [MEASURED] | 512 [MEASURED] / 256 [MEASURED] | 14,827 [MEASURED] / 14,761 [MEASURED] | 231.22 [MEASURED] | 5.12 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| full_cpu_output_ub64 | q4_0/q4_0 | 262,144 [MEASURED] | 4,096 [MEASURED] | 512 [MEASURED] / 64 [MEASURED] | 14,877 [MEASURED] / 14,447 [MEASURED] | 180.27 [MEASURED] | 5.04 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| full_cpu_output_ub128 | q4_0/q4_0 | 262,144 [MEASURED] | - | 512 [MEASURED] / 128 [MEASURED] | - | failed [MEASURED] | failed [MEASURED] | - | ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))) |
| full_cpu_output_balanced_ub128 | q4_0/q4_0 | 262,144 [MEASURED] | 4,096 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 14,865 [MEASURED] / 14,555 [MEASURED] | 209.22 [MEASURED] | 4.97 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| full_cpu_output_ffn_gpu_balanced | q4_0/q4_0 | 262,144 [MEASURED] | 4,096 [MEASURED] | 512 [MEASURED] / 128 [MEASURED] | 14,741 [MEASURED] / 14,679 [MEASURED] | 213.89 [MEASURED] | 5.01 [MEASURED] | 3 [MEASURED] | layer, ngl=all, threads=4, FA=on [MEASURED] (settings) |
| full_depth_validated | q4_0/q4_0 | 262,144 [MEASURED] | 261,952 [MEASURED] | 2,048 [MEASURED] / 128 [MEASURED] | 14,825 [MEASURED] / 14,825 [MEASURED] | 3.05 [MEASURED] | 3.35 [MEASURED] | 3 [MEASURED] | tensor, ngl=all, threads=4, FA=on [MEASURED] (settings) |

Context shifting is disabled. A BOS-only prompt is not a bulk prompt benchmark. Configured context and processed prompt depth are separate columns. The deep profiles have different microbatches to fit their memory budgets, so cross-profile differences are not a pure one-variable KV comparison.

### Full-depth validation

| Start depth | End depth | Cached tokens | New prompt tokens | pp tok/s | tg tok/s | Truncated |
|---:|---:|---:|---:|---:|---:|---|

| 261,952 [MEASURED] | 262,016 [MEASURED] | 0 [MEASURED] | 261,952 [MEASURED] | 123.63 [MEASURED] | 3.28 [MEASURED] | False [MEASURED] |

| 262,016 [MEASURED] | 262,080 [MEASURED] | 262,015 [MEASURED] | 1 [MEASURED] | 2.87 [MEASURED] | 3.35 [MEASURED] | False [MEASURED] |

| 262,080 [MEASURED] | 262,144 [MEASURED] | 262,079 [MEASURED] | 1 [MEASURED] | 3.05 [MEASURED] | 3.42 [MEASURED] | False [MEASURED] |

Only the first request cold-prefills the long prefix. Later requests append real generated token IDs and reuse the growing history. The generation median covers consecutive near-limit continuations; cold full-prefix PP is measured once, not as a repeated median. The cold request limit is **3,000 seconds [MEASURED] (experiment setting)**.

### Parallel slots
| Slots | Median per-request tg tok/s | Median aggregate end-to-end tok/s |
|---:|---:|---:|

| 2 [MEASURED] | 8.98 [MEASURED] | 16.07 [MEASURED] |

| 4 [MEASURED] | 6.67 [MEASURED] | 21.82 [MEASURED] |

Aggregate end-to-end includes prompt processing and request overhead. Each request generates **128 tokens [MEASURED] (setting)**, and each slot has **4,096 tokens of capacity [MEASURED] (setting)**. Aggregate end-to-end throughput is not isolated decode throughput.

## Separate official llama-bench cross-check

Official build `11696`, commit `ceefd16da`, version `0.7.0-dev` [MEASURED] (identifiers). Archive digest matches the release asset [MEASURED]: `8efa76e0d3c667ea3593e91a05068187c486113779f32a43fc1553496129c872`. A source build was stopped to avoid contention after its CUDA-driver CMake issue was repaired. This independent build is not the supplied executable.

| Split | Existing depth | Test | Median tok/s |
|---|---:|---|---:|

| layer | 0 [MEASURED] | pp 512 [MEASURED] | 296.86 [MEASURED] |

| layer | 0 [MEASURED] | tg 64 [MEASURED] | 9.02 [MEASURED] |

| layer | 4,096 [MEASURED] | pp 512 [MEASURED] | 276.74 [MEASURED] |

| layer | 4,096 [MEASURED] | tg 64 [MEASURED] | 9.51 [MEASURED] |

| tensor | 0 [MEASURED] | pp 512 [MEASURED] | 394.10 [MEASURED] |

| tensor | 0 [MEASURED] | tg 64 [MEASURED] | 10.84 [MEASURED] |

| tensor | 4,096 [MEASURED] | pp 512 [MEASURED] | 344.12 [MEASURED] |

| tensor | 4,096 [MEASURED] | tg 64 [MEASURED] | 10.54 [MEASURED] |

Bench pp is an incremental prompt block after the stated depth. Server depth pp is a whole cold prompt. Bench has no HTTP/chat/sampler path. Per-repeat samples and the exact command are retained.

## Allocation and failure observations
| Configuration | Result |
|---|---|

| alloc_layer_f16_106496 | loaded [MEASURED] |

| alloc_layer_f16_98304 | loaded [MEASURED] |

| alloc_layer_q4_229376 | loaded [MEASURED] |

| alloc_layer_q4_262144 | failed [MEASURED]: `RuntimeError('server exited 1')` |

| alloc_layer_q4_full_ngl59 | failed [MEASURED]: `RuntimeError('server exited 1')` |

| alloc_layer_q4_full_ngl60 | failed [MEASURED]: `RuntimeError('server exited 1')` |

| alloc_layer_q8_196608 | failed [MEASURED]: `RuntimeError('server exited 1')` |

| alloc_tensor_q4_229376 | failed [MEASURED]: `RuntimeError('server exited -6')` |

| alloc_tensor_q4_262144 | failed [MEASURED]: `RuntimeError('server exited -6')` |

| alloc_tensor_q8_98304 | loaded [MEASURED] |

| oom_layer_f16_114688 | failed [MEASURED]: `RuntimeError('server exited 1')` |

| oom_layer_q8_262144 | failed [MEASURED]: `RuntimeError('server exited 1')` |

## Feasibility, tradeoffs and limitations

At the full trained limit, f16 weights plus KV exceed raw combined VRAM [CALCULATED]. Symmetric q8 weights plus compact KV also exceed it before runtime overhead [CALCULATED]. Those all-GPU configurations are impossible under this layout. Q4 weights plus KV fit, but the measured scratch requests make the original all-GPU configuration fail [MEASURED]. Split ratios cannot cure an aggregate deficit. CPU tensor placement or different build-specific scratch behavior changes the calculation.

The f16 allocation at **106,496 tokens [MEASURED]** succeeds only after pipeline allocation fails and the server falls back to serial scheduling [MEASURED]. Loaded-only probes establish allocation feasibility, not full-depth generation. The measurement tables distinguish configured capacity, processed prompt depth and maximum-context validation.

| Choice | Speed | Capacity / memory | Quality / practical consequence |
|---|---|---|---|
| f16 KV | strong baseline | largest persistent KV | reference KV precision; fixed GGUF weight quantization unchanged |
| q8 KV | fastest non-draft corrected short-chat control here | smaller KV, but scratch still grows | less aggressive KV quantization; accuracy loss not quantified |
| q4 KV | check depth measurements | smallest tested persistent KV | more aggressive lossy KV; long-range quality untested |
| Layer split | lower shallow rates than leading tensor cases | lower model-buffer footprint | sequential GPU work; supports tested draft model |
| Tensor split | higher shallow throughput | extra model-layout overhead | experimental; draft combination asserted |
| Larger physical microbatch | improves prefill when memory permits | more scratch, sometimes more SWA padding | no intended precision change; floating-point ordering can change greedy output |
| Disable SWA checkpoints and RAM prompt cache | avoids expensive host copies | less host-RAM pressure | gives up rollback/prefix-edit reuse; pure append can still reuse a slot |
| Draft model | strong gain on repetitive input, smaller chat gain | extra model and runtime memory | acceptance is workload dependent |
| More slots | more aggregate service throughput | more total KV or less context per slot | slower individual requests |
| CPU tensor placement | enables otherwise unavailable capacity | frees VRAM, uses CPU RAM | slows affected operations without changing weight quantization |

Known failures and untested work:

- Row mode failed with either tested main-GPU choice [MEASURED]. Tensor mode plus the tested draft aborted in a meta-backend assertion [MEASURED]. OOM and fallback logs remain in the evidence.
- `GGML_CUDA_FORCE_MMQ` is a compile-time option in inspected source. No effective runtime `GGML_CUDA_PEER_ACCESS` support was found. `GGML_CUDA_P2P` was actually tested and showed no reliable gain. Relevant baseline environment variables were unset [MEASURED].
- Equivalent no-mmap loading was tested. Memory locking was not swept: host swap was absent [MEASURED], and it cannot directly accelerate already resident GPU weights.
- CPU offload was explored only after the full trained-context all-GPU allocation failed. Exhaustive split ratios, mixed K/V precisions, all layer counts and draft sizes were not swept.
- Clocks and temperature varied [MEASURED]. The chronological sweep is not randomized; small differences do not establish causal gains. No confidence intervals or universal optimality claim are made.
- Quality checks only verify prompt formatting and readable opening reasoning. There was no perplexity, retrieval-accuracy, task-accuracy or long-chat quality benchmark.
- Preliminary `natural_*` controls omitted required BOS and are marked excluded. Corrected `chat_*` controls supersede them. Main speed prompts already included BOS.
- The maximum-depth procedure cold-prefills once, then measures real sequential continuations near the limit. It does not provide repeated cold full-context prefill measurements. This exception avoids spending the session repeating the most expensive prefill; it is not concealed as a cached-prompt median.
- Beyond-trained-context RoPE extrapolation was not tested. A memory fit does not prove useful distant-token recall.
- A matching `llama-bench` for the supplied unknown-commit executable was unavailable. The separate official benchmark build is identified below; its timings are not merged into server results.

## Sources and reproduction

Primary evidence is live shell/GPU output, GGUF metadata and tensor directory, server logs, per-request timing JSON and benchmark samples in the accompanying bundle.

- [Target GGUF repository](https://huggingface.co/bartowski/TheDrummer_Artemis-31B-v1.2-GGUF)
- [Supplied executable](https://files.avinash9.in/movies/llama-server-sm75)
- [Draft GGUF repository](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF)
- [Official benchmark release](https://github.com/ggml-org/llama.cpp/releases/tag/b11696)
- [Inspected source snapshot](https://github.com/ggml-org/llama.cpp/tree/f3363593f457169cdfbdfe22ea399b1cc139ba18); this is not asserted to be the supplied executable's source.

The model and binary paths are session-local. Re-download after a Kaggle reset. The bundle includes exact commands, scripts, measurement CSV, metadata and raw evidence; it excludes GGUF weights, executables and source/build trees. Each experiment has a timeout and cleanup. Failed and excluded controls remain visible.
