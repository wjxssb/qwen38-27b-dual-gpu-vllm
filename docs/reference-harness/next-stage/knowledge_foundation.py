"""Durable, evidence-indexed history. Next-stage conclusions appended separately."""
from common import *
import shutil,datetime
P=PROD/'docs/qwen38-optimization';P.mkdir(parents=True,exist_ok=True)
OLD=Path('/home/frank/dense-opt-repair-20260922/qualification');F=NV/'production-dense-grouped-20260922';sources=[]
def copy(src,rel,role,kind='evidence'):
 src=Path(src);dst=P/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 sources.append({'path':str(src),'sha256':sha(src),'role':role,'created_at':datetime.datetime.fromtimestamp(src.stat().st_mtime,datetime.timezone.utc).isoformat(),'created_at_semantics':'source filesystem mtime; not asserted authoring time','type':kind,'durable_copy':str(dst),'durable_sha256':sha(dst)})
for rel in ['FINAL-REPORT.md','METHODOLOGY.md','evidence/final-performance-table.json','evidence/final-inflight-peer-loss/receipt.json','evidence/finalization/receipt.json','evidence/final-control-identity.json','evidence/independent-review.json','evidence/independent-review-closure.json','evidence/synchronization-inventory.json','evidence/installed-synchronization-inventory.json','evidence/decode-event-attribution.json','evidence/decode-cupti-attribution-1k16k.json','evidence/decode-decision.json','evidence/patches/index.json','evidence/sync-grouped-comparison.json','evidence/synchronization-measured-all-candidates.json']:
 copy(OLD/rel,'evidence/qualification/'+rel,rel)
for row in read(OLD/'evidence/patches/index.json')['rows']:copy(row['patch'],'evidence/qualification/evidence/patches/'+Path(row['patch']).name,'qualified/rejected source patch','candidate')
for rel in ['MANIFEST.json','PROFILES.json','environment.json','overlay/vllm/model_executor/models/qwen3_next.py','overlay/vllm/stability_telemetry.py','overlay/vllm/v1/executor/multiproc_executor.py','overlay/vllm/v1/worker/gpu_model_runner.py','overlay/vllm/v1/engine/async_llm.py','launcher.py']:
 copy(F/rel,'sources/fallback/'+rel,'accepted fallback '+rel,'production')
for rel in ['recovery.py','rpc_progress.py','wedge_forensics.py','nvidia-launch.py']:copy(PROD/'scripts'/rel,'sources/control/'+rel,'qualified control source','production')
for src,rel in [('/home/frank/p2p-stable/STABLE_BASELINE.md','p2p-STABLE_BASELINE.md'),(NV/'mtp-k3-closeout-20260914/CLOSEOUT.md','mtp-CLOSEOUT.md'),(NV/'prefill-tp2-closeout-20260914/MEASURED_AB.md','chunk-MEASURED_AB.md'),(NV/'prefill-tp2-closeout-20260914/CLOSEOUT.md','chunk-CLOSEOUT.md'),('/home/frank/qwen38-27b-dual-gpu-vllm/runtime/graph006/source-spec006.json','initial-graph006.json'),('/home/frank/qwen38-27b-dual-gpu-vllm/runtime/graph006/repairs/20260910-opencode-state-audit/REPORT.md','graph006-repair-REPORT.md'),(NV/'closeout-migration-20260914/unique-production-migration/MIGRATION_CLOSEOUT.md','migration-CLOSEOUT.md')]:copy(src,'evidence/history/'+rel,'historical '+rel)
copy(Q/'evidence/hardware-foundation-current.json','evidence/hardware-foundation-current.json','hardware read-only observations')
copy(Q/'evidence/frozen-baseline.json','evidence/frozen-baseline.json','accepted fallback freeze')
# Keep qualification methodology/harness and small fixture manifest durable.
for name in ['bench.py','compare.py','fault_test.py','fault_workload.py','loaded_identity.py','safe_restore.py','lifecycle.py','common.py','cache_release.py','log_observation.py','stop_timeline.py','promote.py']:
 copy(OLD/'scripts'/name,'reference-harness/'+name,'historical harness reference; absolute paths require new workspace rebinding')
copy(OLD/'fixtures/inventory.json','reference-harness/fixture-inventory.json','fixture source hashes')
perf=read(OLD/'evidence/final-performance-table.json');fault=read(OLD/'evidence/final-inflight-peer-loss/receipt.json');owner=read(OLD/'evidence/final-control-identity.json')['owner']
base={'classification':'MEASURED','production_manifest':sha(F/'MANIFEST.json'),'hardware':{'gpus':['RTX 5070 Ti','RTX 5070 Ti'],'gpu_uuids':owner['expected_gpu_uuids'],'topology':'PHB','evidence':'evidence/hardware-foundation-current.json'},'driver':{'version':'610.57.04','evidence':'evidence/hardware-foundation-current.json'},'runtime':{'image':owner['image'],'vllm':'0.26.1rc1.dev608+g99a10304d.cu129','evidence':str(Q/'xid/evidence/image-package-versions.json')},'model':{'revision':owner['model_revision'],'class':'Qwen3_5ForConditionalGeneration','evidence':'evidence/qualification/evidence/finalization/receipt.json'},'tp':2,'mtp_k':3,'chunk':4096,'max_context':262144,'configuration_evidence':'sources/fallback/PROFILES.json','benchmarks':{},'recovery':{'v1':{'latencies_seconds':fault['latencies'],'reload_seconds':fault['latencies']['model_ready']-fault['latencies']['new_supervisor_started'],'reload_classification':'DERIVED','evidence':'evidence/qualification/evidence/final-inflight-peer-loss/receipt.json'},'v2':{'state':'IN_PROGRESS_NOT_PROMOTED'}},'memory':{'sampling_interval_s':.2,'scope':'NVML sampled peaks, not allocator high-water','evidence':'reference-harness/bench.py'},'known_limits':['Fixed deterministic retained-record workload; selected raw logits only, not full vocabulary equivalence.','Historical baseline and final differ in process allocation history; use matched fresh-process comparisons for causal prefill attribution.','MTP batched delivery intervals are not individual token GPU latency.','n=3 formal repeats within one process per length do not establish population-wide speedup or independent deployment variance.','Near-limit test uses 258048 prompt tokens, not an assertion that every 262144-token multimodal request fits.']}
for n,values in perf['rows']['final'].items():base['benchmarks'][n]={'final':values,'baseline':perf['rows']['baseline'][n],'evidence':'evidence/qualification/evidence/final-performance-table.json','json_pointer':'/rows/final/'+n}
save(P/'PERFORMANCE-BASELINES.json',base)
save(P/'CURRENT-PRODUCTION.json',{'state':'ACCEPTED_CANONICAL_FALLBACK_FROZEN_NEXT_STAGE_IN_PROGRESS','runtime':str(F),'manifest_sha256':sha(F/'MANIFEST.json'),'binding':str(PROD/'runtime-bindings/production-dense-grouped-20260922-control-4767c3b893fe'),'configuration':{'tp':2,'mtp_k':3,'chunk':4096,'actual_mamba_alignment':2848,'max_context':262144,'graph':'FULL_DECODE_ONLY','capture':4,'kv_bytes_per_gpu':3039750144,'kv_dtype':'nvfp4','layout':'HND','max_num_seqs':1,'async_scheduling':False,'custom_all_reduce':False,'nccl_proto':'Simple'},'configuration_evidence':'sources/fallback/PROFILES.json','model':base['model'],'hardware':base['hardware'],'live_identity':'MUST_REFRESH: experiments can temporarily select an isolated candidate. Do not infer active process identity from this ledger.','source_control':'sources/control/','fallback_remains_immutable':True})
def doc(name,s): (P/name).write_text(s.strip()+'\n')
doc('START-HERE.md','''If you are preparing a new Dense Qwen runtime, especially Qwen4,
read this document and QWEN4-MIGRATION-PLAYBOOK.md before changing
the current production runtime.

This is the durable Qwen3.8 knowledge pack. Read CURRENT-PRODUCTION.json for the canonical release and final identity receipt; verify live state again before actions. This pack records facts, derived values, inference and unresolved questions separately. Next-stage recovery/Xid work is still in progress until its final report and independent review close.

The accepted fallback is `/home/frank/nvidia-dense-runtime/production-dense-grouped-20260922`, manifest `87971548b51369f2b028f18bef4275196d70d7079aec3ef830d725257dafb0a5`. It is immutable. Dual RTX 5070 Ti, PHB topology, driver 610.57.04, current-boot ACS/P2P proof and NCCL P2P/CUMEM underpin its qualification. Do not treat a driver version string as proof that a stock driver has these custom P2P capabilities.

MEASURED/HISTORICAL: accepted optimizations are chunk4096, telemetry single serialization, grouped per-layer prefill fault checks, and ownership-bound TP failure recovery. K3 and capture4 are qualified settings for this model/runtime, not Qwen4 defaults. See QWEN38-OPTIMIZATION-HISTORY.md and FAILURES-AND-REJECTED-IDEAS.md before experimenting.

Evidence is indexed in EVIDENCE-INDEX.json and sealed with SHA256 in SOURCE-MANIFEST.json. Small core records and exact source snapshots are copied into this directory; original raw benchmark events and large traces remain indexed at their original paths. Never describe a copied summary as a new GPU test.

Read RECOVERY-ARCHITECTURE.md before any restart, stop or fault test. Ownership, replacement guard, pidfd, full fresh CUDA/P2P probe and 60-second hardware observation remain required. Health=200 alone is insufficient; require a real Gateway completion.

To reproduce benchmarks, first read `evidence/qualification/METHODOLOGY.md`, then `reference-harness/bench.py` and the fixture inventory. The copied harness is a reference with historical absolute paths: rebind it in a new isolated workspace, seal it, verify idle and source identity, match cache state, then run fresh-process single-variable A/B. Do not execute the historical lifecycle files blindly. Use 1 warmup + 3 formal requests per length for performance; check cold-prefix hits, exact token hashes and selected raw logits. Include 258K and Vision/Tools if the changed path can affect them.

Rollback: use the current task's verified ownership-aware transactional lifecycle and exact before-image bindings. Wait for idle or recovery completion, check the current instance/PID/start ticks/container again, restore the accepted binding, run fresh hardware gates and a real Gateway completion. Never use broad process kills or substitute a different model. Unknown ownership/hardware state remains fail-closed.

For Qwen4, start with QWEN4-AGENT-BOOTSTRAP.md. Port invariants and methods; re-derive model-specific patches.
''')
doc('QWEN38-OPTIMIZATION-HISTORY.md','''# Qwen3.8 optimization history

## Hardware foundation — HISTORICAL, with current observations

The platform is dual RTX 5070 Ti on PHB topology. The historical stable foundation used the Aikitoria 610.57.04-p2p-v3 open-kernel branch, commit `461d638b7a91f73702001e343ec91b275a43ec35`, on kernel 7.0.0-30. Secure Boot was disabled and lockdown was none. Current read-only observations confirm driver, topology, Secure Boot and boot receipt in `evidence/hardware-foundation-current.json`.

Targeted ACS handling identifies AMD 1022:14db root ports 00:01.1 and 00:01.3, changes ACS control at 0x2a6 to clear request/completion redirection, and verifies readback. System ACS → system P2P proof → user P2P gate → actual NCCL channel validation form separate gates. The current-boot root receipt is `/run/p2p-stable/verified`. Do not generalize this register recipe to a different motherboard. The historical Sep10 report contains a later-GRUB-reboot-pending note; it is not the current boot's status.

MEASURED/HISTORICAL: qualified transport is P2P/CUMEM both directions, without silent SHM/NET substitution. Chunk closeout measured 128 MiB peer copies about 28.63 GB/s and NCCL 64 MiB bus bandwidth about 23.3 GB/s; these old microbenchmarks are not current model throughput. Stock-driver rollback assets existed, but their automatic application was not qualified. No driver upgrade or rollback is implied by this pack.

## Initial runtime and safety repairs — HISTORICAL

The retained graph006 specification pins image `sha256:b6190f100526423e5855c5c0f98773744c91a69a7e112993d66bed038c9b76c6`, Unsloth revision `57926b...`, TP2/NVFP4, max262144, chunk2048, MTP K3, FULL_DECODE_ONLY capture4, custom all-reduce off, KV 2928199680 bytes/card, CPU4, language-only and no-prefix-cache. That earliest spec used NCCL_P2P_DISABLE=1. Later P2P/CPU8/prefix repairs must not be retroactively attributed to the earliest spec. The pinned image's inspected package version is vLLM 0.26.1rc1.dev608+g99a10304d.cu129; source-spec alone does not prove a running image.

Graph006 repairs preserved pinned-source lifetime across asynchronous DMA, side-stream ownership, graph input strides/device identity, fatal reporting on non-output ranks, and API cancellation cleanup. Keeping defensive checks is justified by these invariants, not their age. The historical 14-request/28,672-decode-token repair exercise was functional evidence, not a controlled speedup benchmark. Sources: `evidence/history/initial-graph006.json`, `graph006-repair-REPORT.md`.

The NVIDIA checkpoint migration pinned revision `dbb8f445b3145f8a4c18ddc769f032d57d32867c`. Actual class is Qwen3_5ForConditionalGeneration despite the public Qwen3.8 alias. Native MTP tensors were retained, shared embedding/head semantics inspected, and the NVFP4 shared head was not incorrectly required to be BF16. Vision first pass remains outside the custom draft graph; follow-up draft steps use their own embedding buffers. Vision, Tools, prefix state and MTP required actual qualification. See `evidence/history/migration-CLOSEOUT.md` and `mtp-CLOSEOUT.md`.

## Prefill tuning — HISTORICAL and MEASURED

Chunk2048→4096 was a controlled cold-prefix A/B. The Sep14 67K result was 3064.2 prefill tok/s, TTFT22.083s, decode111.53 tok/s and +14.01% prefill versus2048. Actual aligned chunks dropped48→24. Profile gaps suggested a mechanism but did not define the measured speedup. Chunk8192 failed warmup on a 160MiB allocation in native FP32 normalization; cleanup passed and no Xid occurred. Larger chunks were not run. These are old configuration-specific outcomes (`chunk-MEASURED_AB.md`, `chunk-CLOSEOUT.md`).

The Sep22 campaign inventoried synchronization sites and separated Class A correctness boundaries from redundant prefill fault checks. Input-check removal and postnorm removal each passed numerical A/B, but initial gains did not cross the campaign's 3%×2-length screening rule. That rule was an experimental filter, not a user requirement or a claim of unsafety. Event replacement preserved boundaries but gave no material gain.

Grouped checking keeps one device-wide fault boundary after each MLP and retains Class A invariants. Across the 258K test, measured per-card fault-check events were23660(all),17745(input removed),11830(postnorm also removed),23660(event),5915(grouped). These are specific model telemetry boundaries, not every CUDA synchronization. All four strategies ran16requests each across35K/66K/128K/258K with exact output/selected-logit equality and no OOM/Xid.

Matched fresh-process long-request history is essential: grouped versus the matched telemetry reference improved prefill6.318%,4.747%,3.626%,2.392% at35K,66K,128K,258K. Corresponding decode deltas were−0.210%,−0.665%,−0.268%,+0.183%; no new decode math patch was justified. Source: `evidence/qualification/evidence/sync-grouped-comparison.json` and original final report.

## Decode attribution — MEASURED; causes beyond measurements UNPROVEN

CUDA-event observations showed index-copy stream time about0.141–0.153ms while its host wait was20.9–23.3ms. Target forward was21.96–24.45ms, MTP draft5.77–6.26ms and host gap around0.74–0.79ms. A host wait includes earlier queued work; its stack frame is not proof of a slow copy kernel or a fault origin.

K3 accepted193/195 draft tokens at1K–66K,190/195 at128K and192/195 at258K in the attribution workload. Successful1K/16K CUPTI traces assigned about9.7–10.4% of the observed interval to collective execution, including peer wait, with three graph launches per step. This does not isolate a bandwidth bottleneck. K2/K4 and alternate NCCL protocols were therefore not promoted. Twelve event-only requests passed; two CUPTI-associated Xid13 incidents remain NOT_PROVEN in cause.

## Telemetry — MEASURED/HISTORICAL

Serializing each event once instead of twice preserved fields/order/lifecycle evidence. CPU writer microbenchmarks gained about10–14%; a smaller serializer microbenchmark improved27.4%. Neither number is model throughput. Initial cold-cache observations were confounded by retuning/OOM fallback and were rejected. The first proper observer A3/B3 comparison failed at66K (−3.8023% decode) and actually rolled back.

Reverse-order B5/A5 replication included all eight samples per side; decode changes were−0.181% at1K and−0.117% at66K, burst+2.507%. This supported accepting CPU work reduction with no material observed model regression, while explicitly not proving population-wide non-regression by confidence interval. Rejected receipts remained intact.

## Recovery evolution — HISTORICAL → accepted production

The old same-RPC90s monitor missed asymmetric TP failures unless both ranks presented a matching pending record. Fatal first-failure evidence now takes priority and binds boot/instance/generation/RPC to exact owned processes. PID plus start ticks, container identity and pidfd prevent PID-reuse signals; replacement protection prevents an old incident from killing a new instance. Forensics are bounded to8s. Stop grace changed120→15s after evidence, while full hardware probe and60s observation stayed intact.

A first CUPTI incident exposed fatal priority being masked by the old120s path and a PCIe current-link-speed downshift32→2.5 being mistaken for hardware identity drift. Idle negotiated speed is not identity. Those controls were repaired and the later incident completed recovery/rollback. No root-cause claim follows from recovery success.

MEASURED v1 final fault: failure→detection9.846s; stop9.912s; old supervisor gone27.142s; observation complete91.585s; new supervisor93.969s; ready202.978s; first real Gateway completion203.288s; replacement survived another60.706s. DERIVED reload109.010s. Fifteen-second grace is not full recovery. Next-stage recovery measurements are tracked separately in RECOVERY-OPTIMIZATION.md.

## Production qualification — MEASURED/HISTORICAL

Independent review recomputed176 real benchmark requests, including the final32requests, with token hashes and selected raw logits. Formal performance is in PERFORMANCE-BASELINES.json. Final sampled per-card VRAM was about15.545/15.543GiB versus15.609/15.609;200ms NVML samples are not allocator high-water. The final original review was PASS_WITH_DISCLOSED_LIMITATIONS with zero findings. Canonical grouped manifest is `87971548b51369f2b028f18bef4275196d70d7079aec3ef830d725257dafb0a5`.

Every number above is scoped to its cited stage; old67K tuning results and later258K campaign results are not one continuous apples-to-apples benchmark. Original review, closure, patches, source identities and receipts are preserved under `evidence/qualification/`.
''')
doc('TRANSFERABLE-LESSONS.md','''# Transferable lessons

## Methods that transfer

MEASURED/HISTORICAL: immutable releases/manifests, candidate→qualification→promotion, exact rollback ledgers, actual worker-namespace source verification, request isolation, matched fresh-process A/B, equal compile/autotune cache state, independent request cache salt, output hashes plus selected raw logits, real in-flight pidfd fault injection, first-failure ownership, replacement-instance guards, fresh CUDA/P2P probes, independent review, preserved rejected results, and CUDA-event decode attribution were useful here.

Transfer these invariants and evidence methods. A method's usefulness does not automatically qualify its implementation against a new scheduler or failure model. Check every ownership field and admission boundary again. Source code, installed mounts, and actual loaded process paths are separate evidence. Retain exact model affinity through recovery.

## Parameters that require fresh qualification

Qwen3.8-specific: K3, chunk4096, actual2848 Mamba alignment, capture4, grouped synchronization placement, model loading assumptions, KV3039750144 bytes/card, exact graph shapes, NVFP4 packing/implementation, long-context memory headroom, NCCL Simple, CPU8 and custom-all-reduce-off. These are reference settings, not Qwen4 answers.

Grouped checks must be re-derived from Qwen4 source and stream dependencies. MTP must be judged on draft cost, verification cost, acceptance and effective throughput. Read-only processor warmup, DMA source lifetime, graph stride, cancellation and fault priority are semantic constraints even when their implementation changes.

## Measurement discipline

Distinguish MEASURED, DERIVED, INFERRED, UNPROVEN and HISTORICAL. Do not turn a CUDA wait location into a kernel root cause; a microbenchmark into model speedup; HTTP liveness into healthy inference; a timeout into recovery completion; or within-process samples into independent deployments. Report observer effects, cache mismatch, rejected comparisons and selected-logit limits explicitly. Stop bounded fault diagnostics when hardware safety gates fail.
''')
doc('RECOVERY-ARCHITECTURE.md','''# Recovery architecture

MEASURED/HISTORICAL: Gateway18080 owns admission and model affinity; relay18082 forwards to native18096. The user services are local-model-switch, qwen27b, qwen27b-relay and local-gpu-recovery. The monitor detects owned TP fatal evidence, or the narrowly defined same-generation/RPC no-progress stall. `/health` alone cannot diagnose a stuck worker.

```text
owned worker failure / bounded no-progress evidence
  → capture first-failure and exact owner
  → stop admission and drain/release under Gateway ownership
  → bounded forensic capture (8s)
  → exact owned stop, 15s grace, pidfd escalation if necessary
  → prove old process group and supervisor gone
  → fresh CUDA/P2P probe, recovery-action and hardware checks
  → full 60s hardware observation
  → reconcile exact old instance
  → launch immutable selected release through preflight and GPU lease
  → engine ready + real Gateway completion
  → replacement observation, prevent stale incident from stopping it
```

No GPU reset or physical disconnect is used in qualification. A new model must not silently replace the failed requested model. The monitor's10s polling phase, forensic and cleanup timings, full observation, reload and completion all contribute to outage. V1 full recovery203.288s contains109.010s reload;15s grace is only one component.

Exact authorization binds boot ID, instance ID, service MainPID, supervisor start ticks, container ID/image/labels/manifest and worker membership. Before signaling, revalidate the same captured owner; do not reread a replacement and treat it as the original authorized target. Use pidfd so PID reuse cannot redirect a signal. Cleanup reconciliation is evidence for one old instance, never blanket authority over a new one.

Failure containment remains fail-closed: unknown ownership, failed fresh CUDA/P2P probe, reset requirement, missing trusted current-boot P2P receipt or sustained hardware abnormality prevents admission. PCIe current speed can downshift when idle and is not by itself an identity change. Rollback waits for an ongoing recovery and healthy replacement; it does not interrupt recovery probes.

Source snapshots: `sources/control/`; immutable fallback launcher and TP executor: `sources/fallback/`. The exact live binding and source hash chain must be refreshed from CURRENT-PRODUCTION.json and final identity evidence. Reference fault harness is `reference-harness/fault_test.py`; use a new sealed workspace before execution.
''')
save(P/'SOURCE-MANIFEST.foundation.json',{'state':'FOUNDATION_SOURCES_VERIFIED_NEXT_STAGE_PENDING','files':sources})
save(P/'EVIDENCE-INDEX.foundation.json',{'state':'FOUNDATION_READY_NEXT_STAGE_PENDING','roles':[{'role':s['role'],'path':s['durable_copy'],'original_path':s['path'],'sha256':s['sha256']} for s in sources]})
print({'path':str(P),'sources':len(sources),'benchmarks':list(base['benchmarks'])})
