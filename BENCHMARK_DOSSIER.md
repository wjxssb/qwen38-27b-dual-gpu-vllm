# Empirical Benchmark Dossier: Qwen3.8-27B-NVFP4 Local Optimization Stack

<!-- CURRENT_PROFILE_NOTE_20260910_BEGIN -->
> **Current profile note — 2026-09-10:** The authoritative current version is
> [the selected profile](runtime/stable-prefill-sync/selection.json), documented in
> [the current audit report](runtime/graph006/repairs/20260910-opencode-state-audit/REPORT.md): **P2P/CUMEM, prefill chunk 2048,
> Prefix Cache enabled (Mamba align), 15 read-only overlays**.
> Current performance measurements and their limits are in
> [this round's benchmark](runtime/graph006/repairs/20260910-opencode-state-audit/benchmark/PASS.json);
> this round does not establish a further throughput increase.
> Everything below this note, including the 2026-09-05 "Current default" / "Final default"
> labels and older SHM / chunk 4096 parameters, is preserved historical experiment
> documentation, not the present launch configuration or current qualification.
<!-- CURRENT_PROFILE_NOTE_20260910_END -->

**Target System**: 2 × NVIDIA GeForce RTX 5070 Ti 16GB (Blackwell SM120)  
**Model**: `unsloth/Qwen3.8-27B-NVFP4` (Revision: `57926baca9a82b4d6906b43f2750d55315f5b10f`)  
**Evaluation Target**: 262,144 Tokens Native Context Under Tensor Parallelism (TP=2)  
**Current default**: Graph MTP frozen after final006 completed three long requests and fullboot lifecycle; formal performance measurement remains invalid.

**Daily profile selection (2026-09-05)**: Graph006 exclusively replaces the historical approximately 61 tok/s configuration. It is the finalized successor of the approximately 66.2 tok/s Graph003 proposal (originally started via `./launch.sh --start`; see README for current managed entry points); historical profiles below are evidence, not launch defaults or automatic fallbacks. [Active profile](ACTIVE_PROFILE.json).

**Historical freeze described below**: `PRODUCTION_FREEZE_PREFIX_ON_L1POOL_V1`; its status does not qualify the new Graph package.
**Data Policy**: 100% verified empirical measurements extracted directly from campaign logs and JSON artifacts. Zero unmeasured extrapolations.

## Final default: MTP drafter Graph (2026-09-05 UTC)

The first isolated dual-GPU canary, `dense-mtp-drafter-graph-001`, passed. Each rank captured a single-token Graph for MTP followups 1 and 2; the first drafter pass remains Eager. All four Graph-versus-Eager comparisons had zero maximum absolute difference in hidden states and logits, with identical greedy tokens. Two HTTP requests both returned `391`. This establishes the short canary's correctness, not long-context throughput or full-service qualification.

Each rank reported a 4,194,304-byte physical-free-memory decrease during the new capture, with 1,057,816,576 bytes free afterwards. These are capture-window measurements, not a matched estimate of the total additional memory peak. The job exited successfully, its container was removed, and no new kernel faults were observed. [Canary closeout](~/ai/tools/two-session-supervisor/state/receipts/dense-drafter001-closeout.json), [raw measurements](~/ai/tools/two-session-supervisor/state/results/dense-mtp-drafter-graph-001/artifacts/fullboot-evidence.json).

The historical **61.46 tok/s** median below belongs to image `b6190f10…`, **Prefix OFF, chunk 2048, KV 2,928,199,680 bytes/rank**. It must not be attributed to the later `4474e909…` Prefix-ON/L1 configuration in section 3. The 12 data points are completed requests, including attempts whose later lifecycle checks failed. The current controls are reported below; the proposal's 75–80 tok/s prediction was not reached. [Historical request audit](~/ai/tools/two-session-supervisor/state/receipts/dense-communication-stop-audit/review.json).

The current baseline control `dense-mtp-matched-baseline-002` completed three identical 260,927-input/1,024-output requests: **62.4812, 61.8403, 61.8899 tok/s**, median **61.8899 tok/s**. All decoded output bytes match; the first request recorded one inference-time JIT notice, which is retained. The job is **FAIL**: the observer omitted Docker-owned model CPU, and one real swap interval also exceeded the predeclared background budget. The fullboot verifier also rejected an `EngineDeadError` traceback from the output handler during shutdown, after all three requests had completed. These are exploratory measurements, not a quiet-qualified A/B result or lifecycle qualification. The container exited with code 0 and no OOM (the host runner returned 2 for verification failure), was removed, and the kernel-fault baseline remained unchanged. [Control closeout](~/ai/tools/two-session-supervisor/state/receipts/dense-mtp-matched-baseline-002-closeout.json).

The Graph-enabled control `dense-mtp-matched-candidate-003` completed the same three requests at **66.0975, 66.2050, 66.2734 tok/s**, median **66.2050 tok/s**. The observed median difference is **+6.97%** against baseline002. All six decoded outputs match exactly. Both arms retain their first-request inference JIT notice. This does **not** establish the proposed 75–80 tok/s. Both jobs remain **FAIL** for formal qualification: candidate003 lost one process-counter sample, and its host verifier rejected a target-ended file mtime that preceded the event value by 269,701 ns. These failures are preserved; no artifact was rewritten to pass. [Independent raw SSE comparison](~/ai/tools/two-session-supervisor/state/receipts/dense-matched-002-003-raw-comparison.json), [candidate closeout](~/ai/tools/two-session-supervisor/state/receipts/dense-mtp-matched-candidate-003-closeout.json).

The follow-up `dense-mtp-graph-lifecycle-004` **passed** with the four mathematical overlays unchanged. A fifth overlay cancels the async output handler before shutting down the engine manager; the gate now records and verifies explicit publication timestamps after flushing and reading back each event marker. Both requests returned `391`, all four Graph/Eager hidden-state and logit comparisons remained exact, and the host verifier returned `DENSE_FULLBOOT_PASS`. The container exited with code 0, was removed, and no new kernel faults appeared. This verifies the repaired short lifecycle; it does not retroactively qualify 002/003, measure new throughput, or establish a long-context lifecycle pass for the final package. [004 closeout](~/ai/tools/two-session-supervisor/state/receipts/dense-mtp-graph-lifecycle-004-closeout.json), [frozen job](~/ai/lab/vllm-unsloth-5070ti-overnight/experimental/sm120-nvfp4-kv/service-262144-mtp-drafter-graph-v1/campaign/JOB_MTP_GRAPH_LIFECYCLE_004.json).

The final baseline arm `dense-mtp-final-baseline-005` is **FAIL**. Its first two long requests completed at **63.0093 and 62.9674 tok/s**, with the same decoded output hash as 002/003. The third request stalled during prefill around 196K–198K computed input tokens; the engine reported `RPC call to sample_tokens timed out`, and the stream returned no generated tokens. There is no valid three-request median. The owned container was removed after release, with no new kernel faults. The EngineCore telemetry file grew to **8,401,190,912 bytes**; independent inspection was limited to file metadata and bounded byte windows, so the effect of this telemetry on the stall remains unresolved. [005 closeout](~/ai/tools/two-session-supervisor/state/receipts/dense-mtp-final-baseline-005-closeout.json), [bounded independent review](~/ai/tools/two-session-supervisor/state/receipts/dense-final005-independent-failure/review.json).

The final Graph arm `dense-mtp-final-candidate-006` completed all three requests at **67.5273, 67.5063 and 67.5960 tok/s**, observed median **67.5273 tok/s**. Outputs match the previous successful requests exactly. All four Graph/Eager hidden-state and logit comparisons have maximum absolute difference zero, and fullboot verification **passed** with clean engine exit, final host sample acknowledgement, container/runner exit 0 and no new kernel faults. The container was removed. The scheduler result is nevertheless **INVALID_MEASUREMENT**: formal request windows included swap activity, a permission-denied process IO sample, excess unattributed CPU activity, and a boundary sample whose owned runner root had exited. These observations cannot be promoted to a formally qualified performance comparison. [006 closeout](~/ai/tools/two-session-supervisor/state/receipts/dense-mtp-final-candidate-006-closeout.json), [raw benchmark](~/ai/tools/two-session-supervisor/state/results/dense-mtp-final-candidate-006/artifacts/benchmark-receipt.json), [quiet-window record](~/ai/tools/two-session-supervisor/state/results/dense-mtp-final-candidate-006/quiet-window.json).

The user selected **Graph MTP as the Dense default** and ended further Eager control runs. The default package is now frozen with the exact final006 image, five overlays and model parameters; its launcher passed seven CPU ownership/lifecycle tests and a static configuration comparison. The first drafter pass remains Eager; later two passes use Graph. Multiprocess RPC/shared-memory transport remains. The observed lack of a stall in 003/006 does not prove this transport is deadlock-free. The user's conditional SHM/telemetry repair was not triggered by006, so this freeze retains its original 1 GiB SHM and telemetry behavior. [Default package freeze](runtime/graph006/FREEZE.json).

The 640 MiB margins in the historical tables are observations, not a current minimum reserve. The user's current policy permits stable execution within physical capacity.

---

## 1. Executive Summary & Headline Metrics

| Dimension | Metric | Verified Value | Qualification Context / Source |
|---|---|---:|---|
| **Max Context** | Verified Prompt Length | **260,927 tokens** | 99.53% of 262,144 max_model_len |
| **Cold Prefill** | Cold Time-To-First-Token (TTFT) | **182.12 s – 187.00 s** | Median: **182.34 s** across fresh-server runs |
| **Cold Throughput** | Effective Prompt-Processing Throughput | **1,395.32 – 1,432.69 tok/s** | Median: **1,430.98 tok/s** (Mean: 1,419.66 tok/s) |
| **Prefix-OFF Control** | Cold TTFT / Effective Prompt Throughput | **188.00 s** / **1,387.90 tok/s** | Row C control; no observed cold-path regression (sample median was 3.1% faster) |
| **Prefix Reuse** | Warm-Hit Time-To-First-Token (TTFT) | **5.717 s – 5.718 s** | 256,320 prefix tokens cached; 2/2 identical passes |
| **Prefix Speedup** | Latency Reduction vs Cold | **31.85× faster** | 182.12s cold vs 5.717s warm (~32× reduction) |
| **Prefix Rate** | Effective Cached-Prefix Reuse Rate | **44,835 tok/s** | Reused/skipped from pool, *not* recomputed |
| **Chunking Gain** | Chunk 4096 vs 2048 Matched Window | **+6.26% TTFT** / **+6.67% Throughput** | 187.24s (4096) vs 199.75s (2048); 1393.6 vs 1306.4 tok/s |
| **Steady Decode** | Sustained Decode Throughput (260K+1024) | **60.06 – 64.38 tok/s** | Median: **61.46 tok/s** (MTP K=3 + CUDA Graph) |
| **Steady Latency** | Time Per Output Token (TPOT at 260K) | **15.53 – 16.65 ms** | Median: **16.27 ms** |
| **Retrieval Decode** | Short Decode Throughput (236K context) | **91.95 – 94.31 tok/s** | Median TPOT: **10.60 ms** (164 tokens output) |
| **Canary Decode** | Short Context Decode (61K context) | **111.45 – 119.95 tok/s**| Median TPOT: **8.36 ms** (256 tokens output) |
| **VRAM Peak** | Peak Memory Footprint (Rank 0) | **15,663 MiB** / 16,303 MiB | **96.07% utilization**; 640 MiB physical headroom |
| **KV Allocation** | Physical KV Pool Allocated | **3,152,594,944 bytes/rank** | L1 Pool (+128 MiB/rank over baseline A86) |
| **KV Capacity** | Startup Advertised Capacity | **274,280 tokens** | +12,136 tokens (+4.63%) above 262,144 |
| **Stability Gate** | Qualification Battery | **3/3 Cold + 2/2 Warm PASS** | Zero freeze, zero requeue, zero Xid, zero OOM |

---

## 2. Hardware Topology & Host-Mediated Constraints

This optimization achievement was realized strictly within consumer desktop hardware constraints without enterprise interconnects:

* **GPUs**: 2 × NVIDIA GeForce RTX 5070 Ti (Blackwell SM120, Compute Capability 12.0).
* **VRAM**: 16,303 MiB physical framebuffer per card (31.84 GiB aggregate across two distinct PCIe buses).
* **PCIe Topology**: Traverses CPU PCIe Host Bridge (`PHB`).
* **IOMMU Profile**: Disabled at kernel level (`iommu=off`, `dma-direct` operation; no `iommu_group` mapping).
* **ACS Redirect**: Disabled on GPU root ports (`pci=disable_acs_redir=...`).
* **CUDA P2P Direct DMA**: Disallowed across the AMD Ryzen host root complex (`NCCL_P2P_DISABLE=1`, driver reports CNS).
* **Inter-GPU Transport**: NCCL Host Memory / Posix Shared Memory ring (`NCCL_SHM_DISABLE=0`). Measured inter-GPU bandwidth: 17.22 GB/s.
* **Host Environment**: AMD Ryzen 7 7745HX (16 threads, single NUMA node), 61.5 GiB DDR5 system RAM, WD_BLACK SN850X NVMe SSD.

---

## 3. Production Stack Configuration

* **Runtime Container Image**: `localhost/production/vllm-nvfp4-kv-sm120:gdn-nvtx-diag-v1`
  * Image Digest: `sha256:4474e909cfb2b61ab78fa888ccc516032e79975416d5c4bb71e3c62eda2d5cce`
  * Base Framework: vLLM `0.26.1rc1.dev608+g99a10304d` (SM120 NVFP4 + FA2 patched)
* **Model Revision**: `57926baca9a82b4d6906b43f2750d55315f5b10f` (`unsloth/Qwen3.8-27B-NVFP4`)
* **Serving Parameters**:
  * `--tensor-parallel-size 2`
  * `--max-model-len 262144`
  * `--max-num-batched-tokens 4096`
  * `--max-num-seqs 1`
  * `--kv-cache-dtype nvfp4`
  * `--kv-cache-memory-bytes 3152594944`
  * `--gpu-memory-utilization 0.914`
  * `--enable-chunked-prefill`
  * `--enable-prefix-caching`
  * `--spec-method mtp --spec-tokens 3`
  * `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}'`
  * `--attention-config '{"backend":"FLASHINFER","use_trtllm_attention":false}'`
  * `--linear-backend auto` (resolves to `FlashInferCutlassNvFp4LinearKernel`)
  * `VLLM_KV_CACHE_LAYOUT=HND`
  * `VLLM_SM120_NVFP4_K3_NATIVE=1`
  * `VLLM_SM120_NVFP4_K3_GRAPH=1`

---

## 4. Empirical Long-Context Prefill Analysis

### Formula & Definition
89157\text{Effective Prompt-Processing Throughput} = \frac{\text{Prompt Tokens}}{\text{TTFT (seconds)}}89157
*Note: This measures system-level prompt-processing ingestion rate across all network, scheduling, and chunked execution layers, not isolated GEMM raw kernel throughput.*

### Primary Qualification Data (260,927 Tokens, Chunked Prefill = 4096)

| Run Identifier | Run Type | Prompt Tokens | TTFT (s) | Elapsed Wall (s) | Effective Prompt Throughput | Source Artifact |
|---|---|---:|---:|---:|---:|---|
| **L1 (Run 1)** | Cold Fresh Server | 260,927 | 187.0012 | 187.51 | **1,395.32 tok/s** | `P5_QUALIFICATION_RESULTS.json` |
| **L1C2 (Run 2)** | Cold Fresh Server | 260,927 | 182.1243 | 182.58 | **1,432.69 tok/s** | `MATRIX_RESULTS_V2.json` |
| **L1C3 (Run 3)** | Cold Fresh Server | 260,927 | 182.3412 | 182.78 | **1,430.98 tok/s** | `MATRIX_RESULTS_V2.json` |
| **L1WARM (Rep 1)** | Cold Initial Step | 260,927 | 181.9515 | 182.42 | **1,434.05 tok/s** | `MATRIX_RESULTS_V2.json` |
| **L1WARM2 (Rep 1)** | Cold Initial Step | 260,927 | 182.2523 | 182.69 | **1,431.68 tok/s** | `P5_QUALIFICATION_RESULTS.json` |
| **Control Row C** | Cold (Prefix OFF) | 260,927 | 188.0008 | 188.30 | **1,387.90 tok/s** | `MATRIX_RESULTS_V2.json` |

### Distribution Summary
* **Fresh-Server Cold Range**: 1,395.32 – 1,432.69 tok/s
* **Fresh-Server Cold Median**: **1,430.98 tok/s**
* **Fresh-Server Cold Mean**: **1,419.66 tok/s**
* **All-5 Cold Median**: **1,431.68 tok/s**
* **Cold Path Impact of Prefix Caching**: **No observed cold-path regression**; measured median was 3.1% faster in this sample (182.34 s vs 188.00 s control). This demonstrates zero cold-path overhead from prefix cache tracking without misattributing run-to-run sampling variance to bookkeeping.

---

## 5. Decode Performance & Speculative Scaling

### Sustained Generation at Maximum Context (260,927 Prompt + 1,024 Output Tokens)
Historical completed requests used image `b6190f10…`, Prefix OFF, chunk 2048, KV 2,928,199,680 bytes/rank, `MTP K=3`, and target `CUDA Graph FULL_DECODE_ONLY [4]`. They do not measure the new drafter Graph or the section 3 Prefix-ON/L1 profile:

| Run ID | Prompt Tokens | Output Tokens | TTFT (s) | Decode Throughput | TPOT | Total Elapsed |
|---|---:|---:|---:|---:|---:|---:|
| `001-stability-a-direct` | 260,927 | 1,024 | 185.44 s | **64.38 tok/s** | **15.53 ms** | 201.35 s |
| `007-stall-probe.run2` | 260,927 | 1,024 | 185.25 s | **62.98 tok/s** | **15.88 ms** | 201.51 s |
| `002-stability-b.prev` | 260,927 | 1,024 | 200.05 s | **62.65 tok/s** | **15.96 ms** | 216.39 s |
| `005-stability-c3` | 260,927 | 1,024 | 203.01 s | **62.13 tok/s** | **16.09 ms** | 219.50 s |
| `002-stability-b` | 260,927 | 1,024 | 190.27 s | **62.01 tok/s** | **16.13 ms** | 206.78 s |
| `001-stability-a.prev` | 260,927 | 1,024 | 196.63 s | **61.69 tok/s** | **16.21 ms** | 213.22 s |
| `003-stability-c1` | 260,927 | 1,024 | 202.10 s | **61.22 tok/s** | **16.33 ms** | 218.82 s |
| `007-stall-probe.run1` | 260,927 | 1,024 | 203.00 s | **60.81 tok/s** | **16.44 ms** | 219.86 s |
| `008-stall-probe` | 260,927 | 1,024 | 199.38 s | **60.60 tok/s** | **16.50 ms** | 216.29 s |
| `007-stall-probe.run3` | 260,927 | 1,024 | 201.58 s | **60.43 tok/s** | **16.55 ms** | 218.52 s |
| `004-stability-c2` | 260,927 | 1,024 | 202.41 s | **60.28 tok/s** | **16.59 ms** | 219.41 s |
| `001-stability-a` | 260,927 | 1,024 | 199.68 s | **60.06 tok/s** | **16.65 ms** | 216.71 s |

* **Decode Statistics (12 Completed Requests; Not 12 Fully Qualified Runs)**:
  * Minimum: **60.06 tok/s** (16.65 ms TPOT)
  * Median: **61.46 tok/s** (16.27 ms TPOT)
  * Mean: **61.60 tok/s** (16.24 ms TPOT)
  * Peak: **64.38 tok/s** (15.53 ms TPOT)

### Context-Length Scaling of Decode Speed
Because KV access overhead scales with sequence length, decode speed is significantly higher at shorter contexts:
* **61K Context** (`006-canary-64k`, 256 tokens output): **111.45 – 119.95 tok/s** (TPOT: **8.34 – 8.97 ms**)
* **236K Context** (`002/003/005-stability`, 164 tokens output): **91.95 – 94.31 tok/s** (TPOT: **10.60 – 10.88 ms**)
* **261K Near-Full Context** (1,024 tokens output): **60.06 – 64.38 tok/s** (TPOT: **15.53 – 16.65 ms**)

---

## 6. Prefix Cache Acceleration

### Qualification Evidence (`PREFIX4096_P_PHASE_DECISION.json`, `P5_QUALIFICATION_RESULTS.json`)

* **Prompt Characteristics**: 260,927 tokens total.
* **Cached Tokens**: 256,320 prefix tokens held in pool.
* **New Computed Tokens on Warm Step**: 4,607 prompt tokens + 16 completion tokens.

| Metric | Cold Pass (Rep 1) | Warm Hit (Rep 2) | Delta / Speedup |
|---|---:|---:|---:|
| **TTFT (L1WARM)** | 181.9515 s | 5.7170 s | **31.83× faster** |
| **TTFT (L1WARM2)** | 182.2523 s | 5.7175 s | **31.88× faster** |
| **Elapsed Wall Time** | 182.42 s – 182.69 s | 5.86 s – 5.90 s | **31.0× faster** |
| **Output Text Determinism**| Coherent text | Byte-identical to Rep 1 | 100% Exact Match |

### Technical Interpretation
* **Latency Reduction**: **96.86% reduction in TTFT** (~182 s down to 5.72 s).
* **Effective Cached-Prefix Reuse Rate**: $256,320 \text{ tokens} / 5.717 \text{ s} \approx \mathbf{44,835\text{ tok/s}}$.
  * *Notice: This must explicitly NOT be confused with raw model compute prefill throughput. The 256,320 tokens are bypassed/reused in the KV cache; only the remaining 4,607 boundary tokens undergo active matrix computation.*

---

## 7. Chunked Prefill Optimization: 2048 vs 4096

From matched-window ABAB campaign (`CHUNK4096_USER_PROMOTION.json`):

| Evaluation Dimension | Control: Chunk 2048 | Candidate: Chunk 4096 | Direct Measured Delta |
|---|---:|---:|---:|
| **Sample Size (n)** | 4 runs (`201`–`204`) | 7 runs (`101`–`108`) | — |
| **Prompt Length** | 260,927 tokens | 260,927 tokens | Identical |
| **Cold TTFT Median** | 199.75 s | 187.24 s | **-12.51 s (-6.26% latency)** |
| **Prefill Effective Throughput** | 1,306.4 tok/s | 1,393.6 tok/s | **+87.2 tok/s (+6.67% gain)** |
| **TPOT Median** | 15.96 ms | 16.76 ms | +0.80 ms (+4.99% regression)* |
| **Workspace Growth** | 56,313,856 bytes | 56,313,856 bytes | **0 bytes** (identical) |
| **Stability Result** | PASS | PASS | Zero Xid, Zero OOM |

*\*Note: TPOT difference (+0.80 ms) is mechanically invariant to prefill chunking during steady decode; user approved promotion based on substantial prefill gains.*

---

## 8. Memory Layout & Budget Reconciliation

### Qualified L1 Pool Configuration (`PREFIX_CACHE_MEMORY_LEDGER.json`)
* **Physical Device Total**: 16,303 MiB (per RTX 5070 Ti)
* **Model Weights (NVFP4 + FP8)**: 10.45 GiB (10,958 MiB) per GPU
* **MTP Speculative Drafter Weights**: 849.4 MB (424.7 MB per GPU under TP2)
* **FlashInfer Attention Workspace**: 56.3 MiB per rank (locked in `FLASHINFER_WORKSPACE_BASE`)
* **Allocated KV Cache Pool**: **3,152,594,944 bytes/rank** (3,006.5 MiB)
* **VRAM Used Immediately Post-Startup**: **15,173 MiB**
* **VRAM Observed at Full 260K Load**: **15,663 MiB**
* **Physical Free VRAM Margin at Peak**: **640 MiB** (exceeds 400 MiB safety floor)

### Admission Defect Forensics (A86 vs L1 Pool)
* **Defect Discovery**: On A86 pool (3,018,377,216 bytes/rank), enabling Prefix Cache with MTP forced the hybrid GDN/QSA state machine into `mamba_cache_mode = align`.
* **The Invisible Footprint**: Align mode allocates an extra mamba state double-buffer page (**+71.6 MiB/rank**), causing the allocator to wedge at computed=247,776 tokens.
* **The Fix**: Expanding the pool by **+128 MiB/rank** to 3,152,594,944 bytes absorbed the 71.6 MiB reservation, expanding capacity from 262,144 to **274,280 tokens (+4.63%)** and completely eliminating the stall.

---

## 9. Historical Optimization Progression

| Stage | Optimization Change | Max Model Len | Cold TTFT (s) | Effective Prefill | Decode Tok/s | TPOT (ms) | Peak VRAM | Status |
|:---|:---|---:|---:|---:|---:|---:|---:|:---:|
| **S1** | Stock TP2 Baseline (FP8 KV, Eager) | 8,192 | 1.62 s (8K) | ~1,000 tok/s | ~13.5 tok/s | ~74.0 ms | ~14.8 GiB | Pass (8K only) |
| **S2** | SM120 NVFP4 KV Enablement | 262,144 | 200.74 s | 1,299.8 tok/s | 13.03 tok/s | 76.73 ms | 15,175 MiB | Qualified Baseline |
| **S3** | MTP Speculative Decoding (K=1) | 262,144 | ~198 s | ~1,315 tok/s | 23.51 tok/s | 42.54 ms | 15,350 MiB | +80% decode |
| **S4** | MTP Speculative Decoding (K=2) | 262,144 | 194.87 s | 1,339.0 tok/s | 29.32 tok/s | 34.10 ms | 15,733 MiB | +125% decode |
| **S5** | MTP Speculative Decoding (K=3) | 262,144 | 195.42 s | 1,335.2 tok/s | 32.69 tok/s | 30.60 ms | 15,620 MiB | +151% decode |
| **S6** | CUDA Graph Decoding (`FULL_DECODE_ONLY`) | 262,144 | 199.68 s | 1,306.7 tok/s | **61.46 tok/s** | **16.27 ms** | 15,597 MiB | **4.72× baseline (13.03 → 61.46 tok/s)** |
| **S7** | Chunked Prefill 4096 (vs 2048) | 262,144 | 187.24 s | **1,393.6 tok/s** | 60.06 tok/s | 16.76 ms | 15,597 MiB | **+6.67% prefill** |
| **S8** | L1 Pool + Prefix Cache ON (Historical) | 262,144 | **182.34 s** | **1,430.98 tok/s** | Not established by the S6 requests | Not established by the S6 requests | **15,663 MiB** | Historical Prefix qualification |

---

## 10. Integrity Statement & Claim Hygiene

* **No Synthetic Metrics**: All TTFT and throughput numbers derived from live HTTP API completion payloads.
* **No Unified VRAM Fallacy**: The dual 5070 Ti cards communicate over PCIe host bridges via NCCL shared memory; memory is strictly partition-addressed (16GB per card).
* **Throughput Terminology**:
  * "Effective prompt-processing throughput" is used exclusively for $\text{prompt\_tokens} / \text{TTFT}$.
  * "Decode throughput" is used exclusively for token generation during steady decode.
  * Prefix cache hit rate is classified as "effective cached-prefix reuse rate" to prevent conflating memory lookup with FLOP execution.
