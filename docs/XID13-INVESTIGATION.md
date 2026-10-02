# Xid13 investigation

Conclusion: **NOT_PROVEN**. Profiling correlation is measured; the defective component or exact failing kernel is not established. No production mathematics, NCCL, driver or CUDA library was changed for this investigation. Diagnostic artifacts are separately sealed and excluded from the recovery production payload.

## Historical incidents

HISTORICAL/MEASURED: incident1 Xid at2026-09-23T00:18:48.733842UTC (monotonic77092679205000ns), GPU1/PCI03:00, during the second profiling session near16K. First-failure receipt00:18:48.949513UTC. Incident2 Xid at2026-09-23T03:24:36.787765UTC (monotonic88240733128000ns), GPU0/PCI01:00, fresh-process35K; first-failure03:24:36.988097UTC. Both reported EXTRA_INLINE_DATA, ESR0x404600=0x83f00001, Class0000cec0. Exact integer-clock conversion,106/142-event sequences and source links are in [historical timeline](evidence/xid/XID13-HISTORICAL-TIMELINE.json).

Environment: dualRTX5070Ti SM120/PHB, custom P2P-enabled610.57.04, pinned imageb6190f10…, PyTorch2.13.0+cu129, CUPTI12.9.79, vLLM0.26.1rc1.dev608+g99a10304d.cu129, TP2/MTP K3/FULL_DECODE_ONLY capture4. Historical profiler used CPU+CUDA Torch activities, no stacks/shapes/memory/flops, frontend ignored, max128iterations. See [package evidence](evidence/xid/image-package-versions.json), individual instance/source manifests and historical profiler configuration.

The async failure surfaced at sampler_index_copy_wait. That host synchronization can report earlier kernel errors; it does not establish index copy as origin. Historical logs do not contain the exact last successful CUDA API, first anomalous CUDA return or failing kernel. Profiling RPC timestamps bound enablement but are not an internal CUPTI-call trace. Fresh process did not eliminate the historical problem. Successful1K/16K CUPTI and12event-only requests constrain hypotheses without proving universal safety.

## Bounded minimal matrix

MEASURED: isolated fresh containers run persistent256×256 GEMM, NCCL all-reduce, graph replay (three per step), index copy and explicit wait for64steps on two ranks. Features are separate modes, each enabled once after capture. This is not full Qwen/MTP, a representative Qwen layer, high-VRAM or long-context execution. Torch can map libcupti even in modeA; no profiling-enable API is invoked there. B-D distinguish subscriber initialization, activities and callbacks; no CUDA operation occurs inside our callbacks. Activity modes require actual records and zero reported drops; Torch modeH requires actual trace kernels.

| Mode | Feature | Result | Xid |
|---|---|---|---|
| A | No profiler enable (Torch can map CUPTI) | PASS_NO_REPRODUCTION | False |
| B | Subscribe only, no collection | PASS_NO_REPRODUCTION | False |
| C | Memcpy activity | PASS_NO_REPRODUCTION | False |
| D | Driver/runtime callbacks | PASS_NO_REPRODUCTION | False |
| E | Concurrent kernel activity | PASS_NO_REPRODUCTION | False |
| F | CUDA runtime API activity | PASS_NO_REPRODUCTION | False |
| G | CUDA Graph trace activity | FAIL_OR_UNSUPPORTED | False |
| H | Torch CPU+CUDA profiler | NOT_RUN_AFTER_BOUNDED_STOP | NOT_RUN |

Preserved harness failures: the first matrix's offline hostname rendezvous timed out before rank GPU work; no Xid and full restore. Static loopback rendezvous was then verified with two CPU-only ranks. R2 completed both ranks' GPU workload but hung in communicator teardown while a CUDA graph still retained it; timeout/cleanup means whole-case FAIL, not PASS. R3 destroys the graph object before communicator teardown and records final success only after cleanup. This graph-object reset is not a GPU hardware reset. No harness failure is mislabeled as an Xid reproduction.

The policy stops on the first new Xid, timeout, feature failure or execution error. Each diagnostic container has exact ID/image/label ownership and the shared GPU lease. Before fallback admission, a fresh CUDA/P2P proof and full60s hardware observation are mandatory. Failed proof remains fail-closed; no driver upgrade or physical reset was attempted.

## Separate Graph trace host crash — MEASURED, not an Xid reproduction

A–F completed cleanly on both ranks. C recorded65memcpy events/rank; E770concurrent-kernel events/rank; F1566runtime-API events/rank; D actual callbacks were observed. ModeG returned success from enable but both ranks then suffered CPU SIGSEGV. Kernel logs locate both instruction pointers at ELF offset0x13c170 in libcupti.so.12 (SHA2562fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a; BuildID96115811519276a52012e4879e8ce8acbe93d5c2). There was no new Xid. The retained rank0 Apport core shows rdx=0 and a read from0x18(%rdx), with the top three stack frames inside CUPTI and subsequent driver frames including cuGraphLaunch. This is direct evidence of a CPU null-pointer read in Graph trace mode; it does not explain historical GPU Xid13 or establish the underlying defect. Exact CUDA API return at the failure is unavailable because the process crashed. The core's command line binds it to this G test.

Only offline core analysis followed; G was not retried. GDB lacked some other image libraries and debug symbols, so higher frames and nearest-exported-function names are not treated as exact attribution. Original Apport report remains untouched; selected metadata/core are retained locally. See [analysis and core hash](evidence/xid/graph-trace-host-crash/analysis.json) and [offline GDB output](evidence/xid/graph-trace-host-crash/gdb-offline.txt). The raw1.67GB core remains externally indexed, not copied into the small documentation pack.

NVRM refcntRequestReference_IMPL messages also occurred, including before historical incident1 and during minimal modes. Their meaning and relation to either failure remain NOT_PROVEN. They are preserved separately and are not silently counted as Xid. Fresh post-diagnostic CUDA/P2P probes, full observation and fallback completion passed before recovery production qualification resumed.

H and the full-model35K probe were deliberately NOT_RUN after the native failure. No graph-disabled/MTP-disabled sweep or Compute Sanitizer GPU rerun followed. This bounded stop is a disclosed limit, not a successful full A–H matrix.

## Full-model original-profiler probe

One35K session is permitted only after all eight minimal modes complete safely. It uses an isolated release with accepted production mathematics plus the original profiler options, after one unprofiled35K request; no historical event/host-stage instrumentation is mixed in. This differs from historicalv35 observers, so a negative result is not an exact historical configuration disproof. Actual mapped worker libraries, CUPTI SHA, profiling control dispatch/return timestamps, output/logits and traces are captured when available.

```json
{
  "state": "NOT_RUN",
  "reason": "Minimal matrix did not complete all features safely"
}
```

No further active reproduction is justified after a bounded safety stop. A successful reduced sequence cannot exclude Qwen-specific kernels, memory pressure, TP/MTP scheduling, graph lifetime, CUPTI/driver interaction or intermittent hardware behavior. Compute Sanitizer was not run on258K or production; heavy observers would have their own scheduling effect. No unsupported graph-off/MTP-off result is invented.

## Official evidence and operating guidance

The [official research record](evidence/xid/OFFICIAL-RESEARCH.md) separates NVIDIA documentation/release notes, upstream reports, user forum cases and local observations. It covers all requested search families. Newer CUPTI graph/session fixes and reports on other GPUs are potential future qualification leads, not this machine's confirmed root cause or a mandate to upgrade610.57.04.

Production avoids CUPTI profiling. For attribution, the previously qualified bounded CUDA-event method has12successful requests in its tested scope; it is not a proof that arbitrary event instrumentation is harmless. Avoid repeated profiling sessions or higher-load original CUPTI configuration on canonical production. Any driver/CUDA upgrade requires a separate system qualification preserving P2P/ACS, model correctness, Graph/MTP, long context, fault recovery and fallback. No evidence here justifies altering the production copy kernel or dropping correctness synchronization.
