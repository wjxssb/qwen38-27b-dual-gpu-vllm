"""Refresh durable next-stage documents from completed receipts, no GPU work."""
from common import *
import shutil
P=PROD/'docs/qwen38-optimization'

def copy(src,rel):
    target=P/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)

def main():
    r=read(Q/'recovery/RECOVERY-OPTIMIZATION-RESULTS.json')
    for name in ['RECOVERY-OPTIMIZATION-RESULTS.json','RECOVERY-OPTIMIZATION-RESULTS.md','RECOVERY-NATIVE-FAULT-TIMELINES.json','RECOVERY-RELOAD-BREAKDOWN.json','RECOVERY-RELOAD-BREAKDOWN.md','RECOVERY-CACHE-INVENTORY.json','CANDIDATE-DECISIONS.md']:
        copy(Q/'recovery'/name,'evidence/recovery/'+name)
    for rel in ['renderer-campaign-r2/receipt.json','inventory-cache-campaign/receipt.json','renderer-handshake-overlap-assembly/receipt.json','renderer-handshake-overlap-assembly/source.patch','inventory-cache-assembly/receipt.json','inventory-cache-assembly/source.patch','renderer-handshake-cpu-tests.txt','inventory-cache-cpu-tests.txt','combined-cpu-regression.txt','combined-cpu-regression.json','renderer-v43-source-rejection.json','observer-v1-limitations.json','spawn-pipe-attribution.json']:
        copy(Q/'recovery/evidence'/rel,'evidence/recovery/'+rel)
    stats=(Q/'recovery/RECOVERY-OPTIMIZATION-RESULTS.md').read_text()
    text='''# Recovery optimization

MEASURED/HISTORICAL: accepted recovery v1 completed in203.288s, including109.010s supervisor→ready. This receipt is preserved separately; it is not the contemporaneous A/B baseline. New optimization measurements and final production fault are in [results](evidence/recovery/RECOVERY-OPTIMIZATION-RESULTS.json).

## Attribution and decision

MEASURED: the startup observer found about16s API renderer warmup,11–13s repeated model SHA, blocked fresh-spawn pipe writes about7.9s each, nested EngineCore initialization about55s, and graph capture about1.8s. Safetensors iteration was about2.9s. The123.259s instrumented reload and old109.010s are different launches: no quantified observer-overhead claim. Nested spans, ranks and peer waits cannot be summed into an exclusive total. Detailed stage coverage, CPU/RSS/I/O/NVML/PCIe samples, cache records and missing hooks are in [breakdown](evidence/recovery/RECOVERY-RELOAD-BREAKDOWN.json) and [cache inventory](evidence/recovery/RECOVERY-CACHE-INVENTORY.json).

DERIVED/SOURCE: worker spawn payloads119743–120764bytes exceeded a65536-byte pipe; pickle serialization was about1ms but pipe writes waited for child imports. This explains much of the apparent distributed-init peer wait; PyNccl communicator construction itself was about0.44s. No global spawn/pipe patch or fork of CUDA was adopted.

The first accepted candidate moves the existing multimodal CPU renderer warmup to after every EngineCore HELLO metadata reply and before waiting for READY. The API thread runs it synchronously while the independent fresh EngineCore initializes. Admission still waits for both. Cache clearing, thread restoration, chat warmup, repeated explicit warmup and failure propagation remain. The rejectedv43 callback preceded HELLO replies and would serialize work; it was never GPU activated. Current DP1/internal-engine and MM configuration were qualified. This is not proof that every future multimodal connector/IPC configuration is CPU-only.

The second candidate replaces repeated model SHA in the critical path with a sealed fast identity check. Offline sealing reads and verifies every original inventory file through an O_NOFOLLOW file descriptor and checks identity before/after hashing and again by pathname. Reload compares device/inode/size/mtime_ns/ctime_ns and the inventory, model revision, image, profile/environment, driver/kernel/P2P/GPU inputs. Any mismatch invokes the unchanged original full verifier; invalid full SHA remains fatal. The launcher verifies both helper and receipt as sealed release files before use. There is no automatic reseal. This assumes trusted immutable local model/release storage, not an adversary able to rewrite the trusted release or mutate files after validation; it does not eliminate all TOCTOU from the underlying loader.

MEASURED cache limit: all1095seeded Triton files retained identical hashes;8new files belonged to one rejection_greedy_sample_kernel specialization and appeared after model readiness. Their artifact-write timestamps are not a compiler-duration measurement. No zero-JIT or universal cache-hit claim is made. Additional cache blessing was not mixed into the two-variable recovery experiment.

Both optimizations occur after the hardware gate. No preparer was launched during hardware observation. All replacements use fresh containers, processes, CUDA contexts and communicators. Ownership, first-failure, pidfd signaling,15s cleanup grace,8s forensics, CUDA/P2P probe and the full60s hardware observation remain. Graph strategy C retains FULL_DECODE_ONLY capture4; eager-first/lazy/background capture was rejected because measured capture cost was small and unqualified concurrent paths add risk. Weight/H2D parallelism, persistent loaders, precreated containers and worker-only restart were not justified by measured benefit.

## Startup ordering

```mermaid
flowchart LR
    H[Fresh CUDA/P2P proof + full 60s observation] --> V[Sealed identity or full SHA fallback]
    V --> S[Fresh API and EngineCore processes]
    S --> C[All HELLO metadata replies sent]
    C --> M[API process: CPU MM warmup and cache cleanup]
    C --> E[EngineCore: fresh workers, CUDA/NCCL, model and KV]
    E --> G[Existing Graph capture and GPU warmup]
    M --> J[Both paths complete; ordinary chat warmup]
    G --> J
    J --> A[Gateway admission and real completion]
```

The engine and API are separate fresh processes. No new background thread or GPU context survives the failed instance. The diagram is ordering, not a sum of measured durations.

## Measurement and qualification

'''+stats.split('\n',2)[2]+'''

CPU regression covers handshake ordering and error paths, repeated multimodal warmup/cache cleanup, stale/changed files and inputs, same-size/mtime tampering, replacement/link/missing-file cases and unchanged full-SHA fallback behavior. Real1K/16K/66K/258K completions compare exact output hashes and selected raw logits; Vision/Tools and state are checked separately. Nine repeated faults use real66K in-flight requests and exact pidfd worker ownership. Each requires fresh hardware proof, complete admission and another60s replacement observation. Per-trial native weight/graph/ready events and inventory fast-path records are in [native timelines](evidence/recovery/RECOVERY-NATIVE-FAULT-TIMELINES.json).

The immutable production copy, rollback to the accepted fallback, final fault, live source/installed/process identity and independent review have their own receipts. CURRENT-PRODUCTION.json and the final report determine promotion status; a candidate PASS alone does not mean production promotion. Historical v1 and the new repeated v2 measurements are both retained in PERFORMANCE-BASELINES.json.

## Qwen4 relevance

Transfer the measured-stage method, sealed-identity/full-SHA fallback principle, safe handshake ordering, fresh-process ownership and rollback gates. Re-derive renderer APIs, connector behavior, config structure, inventory coverage, graph shapes and CPU/GPU boundaries from Qwen4 source. Do not copy these four vLLM files or the model stat receipt into a new architecture without qualification. Rebuild identity receipts and caches for changed model/runtime/driver inputs.
'''
    (P/'RECOVERY-OPTIMIZATION.md').write_text(text)
    failures=P/'FAILURES-AND-REJECTED-IDEAS.md'
    marker='## Next-stage startup and diagnostic lessons'
    body=failures.read_text().split(marker)[0].rstrip()
    body+='\n\n'+marker+'\n\n'+'''HISTORICAL/MEASURED: the first startup observer used a sitecustomize path that was shadowed by the image's earlier system sitecustomize. Its missing in-container hooks were not accepted as measurement coverage. A uniquely named .pth import fixed attachment, and per-process markers verified it. For Qwen4, prove that instrumentation actually loaded before interpreting absence of spans.

SOURCE/REJECTED: renderer v43 put CPU warmup before HELLO metadata replies, which blocked the engine's initialization rather than overlapping it. It never reached GPU activation. v45 waits until all replies are sent. The lesson is to trace protocol readiness, not infer overlap from a lexical context-manager boundary.

MEASURED/METHOD: the first Vision oracle rejected a correct geometry response because of a complete JSON code fence and capitalization. The raw failed receipt remains. Later checks explicitly normalize only those formatting differences and compare exact shape/color/position/schema content; this is a revised semantic oracle, not a retroactive PASS for the earlier campaign.

MEASURED/REJECTED: initial isolated matrix rendezvous failed before rank GPU work; R2 completed GPU work but failed communicator teardown with the graph retained. Whole-case results remain bounded failures, not successful Xid reproductions. R3 uses loopback rendezvous, destroys the graph object before communicator teardown and requires a final post-cleanup success marker. No hardware GPU reset is involved.

MEASURED/DECISION: graph capture around1.8s and safetensors iteration around2.9s did not justify eager-first capture or parallel weight I/O changes. Persistent loaders, precreated containers and spawn implementation changes add lifetime/ownership complexity; none was adopted. These choices are evidence-specific, not a prohibition on separately qualified Qwen4 work.
'''
    failures.write_text(body)

    perf=read(P/'PERFORMANCE-BASELINES.json');perf['recovery']['v2']={'classification':'MEASURED','state':'QUALIFIED_SEE_CURRENT_PRODUCTION_FOR_PROMOTION','groups':r['groups'],'comparisons':r['comparisons'],'evidence':'evidence/recovery/RECOVERY-OPTIMIZATION-RESULTS.json'}
    if 'production_copy_fault' in r:perf['recovery']['v2']['production_copy_fault']=r['production_copy_fault']
    save(P/'PERFORMANCE-BASELINES.json',perf)
    minimal_path=Q/'xid/evidence/minimal-matrix-r3/receipt.json';minimal=read(minimal_path)
    assert minimal['state'] in ['COMPLETED_RESTORED','BOUNDED_STOP_RESTORED']
    for name in ['minimal-matrix','minimal-matrix-r2','minimal-matrix-r3']:
        copy(Q/'xid/evidence'/name/'receipt.json','evidence/xid/'+name+'/receipt.json')
        copy(Q/'xid/evidence'/name/'manifest.json','evidence/xid/'+name+'/manifest.json')
    full_path=Q/'xid/evidence/full-original-35k/receipt.json'
    full=read(full_path) if full_path.exists() else None
    if full:copy(full_path,'evidence/xid/full-original-35k/receipt.json')
    crash=Q/'xid/evidence/graph-trace-host-crash'
    if (crash/'analysis.json').exists():
        for name in ['analysis.json','metadata.json','ProcMaps.txt','gdb-offline.txt','gdb-command.json','kernel-window.jsonl']:
            copy(crash/name,'evidence/xid/graph-trace-host-crash/'+name)
        copy(Q/'xid/evidence/refcnt-kernel-message-history.json','evidence/xid/refcnt-kernel-message-history.json')
    copy(Q/'xid/OFFICIAL-RESEARCH.md','evidence/xid/OFFICIAL-RESEARCH.md')
    rows=['| Mode | Feature | Result | Xid |','|---|---|---|---|']
    features={'A':'No profiler enable (Torch can map CUPTI)','B':'Subscribe only, no collection','C':'Memcpy activity','D':'Driver/runtime callbacks','E':'Concurrent kernel activity','F':'CUDA runtime API activity','G':'CUDA Graph trace activity','H':'Torch CPU+CUDA profiler'}
    indexed={x['mode']:x for x in minimal['rows']}
    for mode,feature in features.items():
        row=indexed.get(mode);rows.append(f"| {mode} | {feature} | {row['state'] if row else 'NOT_RUN_AFTER_BOUNDED_STOP'} | {row['new_xid'] if row else 'NOT_RUN'} |")
    probe_text=json.dumps({k:full.get(k) for k in ['state','result','new_xid','error']} if full else {'state':'NOT_RUN','reason':'Minimal matrix did not complete all features safely'},indent=2)
    xid='''# Xid13 investigation

Conclusion: **NOT_PROVEN**. Profiling correlation is measured; the defective component or exact failing kernel is not established. No production mathematics, NCCL, driver or CUDA library was changed for this investigation. Diagnostic artifacts are separately sealed and excluded from the recovery production payload.

## Historical incidents

HISTORICAL/MEASURED: incident1 Xid at2026-09-23T00:18:48.733842UTC (monotonic77092679205000ns), GPU1/PCI03:00, during the second profiling session near16K. First-failure receipt00:18:48.949513UTC. Incident2 Xid at2026-09-23T03:24:36.787765UTC (monotonic88240733128000ns), GPU0/PCI01:00, fresh-process35K; first-failure03:24:36.988097UTC. Both reported EXTRA_INLINE_DATA, ESR0x404600=0x83f00001, Class0000cec0. Exact integer-clock conversion,106/142-event sequences and source links are in [historical timeline](evidence/xid/XID13-HISTORICAL-TIMELINE.json).

Environment: dualRTX5070Ti SM120/PHB, custom P2P-enabled610.57.04, pinned imageb6190f10…, PyTorch2.13.0+cu129, CUPTI12.9.79, vLLM0.26.1rc1.dev608+g99a10304d.cu129, TP2/MTP K3/FULL_DECODE_ONLY capture4. Historical profiler used CPU+CUDA Torch activities, no stacks/shapes/memory/flops, frontend ignored, max128iterations. See [package evidence](evidence/xid/image-package-versions.json), individual instance/source manifests and historical profiler configuration.

The async failure surfaced at sampler_index_copy_wait. That host synchronization can report earlier kernel errors; it does not establish index copy as origin. Historical logs do not contain the exact last successful CUDA API, first anomalous CUDA return or failing kernel. Profiling RPC timestamps bound enablement but are not an internal CUPTI-call trace. Fresh process did not eliminate the historical problem. Successful1K/16K CUPTI and12event-only requests constrain hypotheses without proving universal safety.

## Bounded minimal matrix

MEASURED: isolated fresh containers run persistent256×256 GEMM, NCCL all-reduce, graph replay (three per step), index copy and explicit wait for64steps on two ranks. Features are separate modes, each enabled once after capture. This is not full Qwen/MTP, a representative Qwen layer, high-VRAM or long-context execution. Torch can map libcupti even in modeA; no profiling-enable API is invoked there. B-D distinguish subscriber initialization, activities and callbacks; no CUDA operation occurs inside our callbacks. Activity modes require actual records and zero reported drops; Torch modeH requires actual trace kernels.

'''+ '\n'.join(rows)+'''

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
'''+probe_text+'''
```

No further active reproduction is justified after a bounded safety stop. A successful reduced sequence cannot exclude Qwen-specific kernels, memory pressure, TP/MTP scheduling, graph lifetime, CUPTI/driver interaction or intermittent hardware behavior. Compute Sanitizer was not run on258K or production; heavy observers would have their own scheduling effect. No unsupported graph-off/MTP-off result is invented.

## Official evidence and operating guidance

The [official research record](evidence/xid/OFFICIAL-RESEARCH.md) separates NVIDIA documentation/release notes, upstream reports, user forum cases and local observations. It covers all requested search families. Newer CUPTI graph/session fixes and reports on other GPUs are potential future qualification leads, not this machine's confirmed root cause or a mandate to upgrade610.57.04.

Production avoids CUPTI profiling. For attribution, the previously qualified bounded CUDA-event method has12successful requests in its tested scope; it is not a proof that arbitrary event instrumentation is harmless. Avoid repeated profiling sessions or higher-load original CUPTI configuration on canonical production. Any driver/CUDA upgrade requires a separate system qualification preserving P2P/ACS, model correctness, Graph/MTP, long context, fault recovery and fallback. No evidence here justifies altering the production copy kernel or dropping correctness synchronization.
'''
    (P/'XID13-INVESTIGATION.md').write_text(xid)
    print(P)

if __name__=='__main__':main()
