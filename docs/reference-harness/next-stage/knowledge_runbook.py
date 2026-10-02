from common import *
P=PROD/'docs/qwen38-optimization'
def doc(name,s):(P/name).write_text(s.strip()+'\n')
doc('QWEN4-MIGRATION-PLAYBOOK.md','''Do not port Qwen3.8 patches line-for-line into Qwen4.
Port invariants and methodology first, then re-derive model-specific patches from Qwen4 source.

# Qwen4 migration playbook

This is a future execution runbook, not evidence that Qwen4 has been tested. All model-specific outcomes below are UNPROVEN until measured. Verify current authorization, live ownership and idle state before service changes; preserve the accepted Qwen3.8 fallback.

## Phase 1 — Freeze Qwen3.8

Read CURRENT-PRODUCTION.json and final review; hash the manifest and source files, capture live service PID/start ticks/exe/argv, container labels/image/mounts, and both worker namespaces. Preserve the complete release, bindings, cache provenance, model revision and rollback ledger. Never overwrite it with Qwen4 files. A freeze is complete only with an executable, independently checked rollback path and source identity receipt.

## Phase 2 — Inspect Qwen4 architecture

Inspect the actual configured/loaded model class, not its public alias. Record attention, MLP, any Mamba/state-space component, KV and recurrent-state layout, MTP/speculation, multimodal paths, quantization, graph compatibility, stream ownership and synchronization sites. Trace shared embedding/head precision and graph input lifetime/stride. Build a patch applicability matrix from Qwen4 source; mark unsupported assumptions and uncertainty explicitly.

## Phase 3 — Establish a clean baseline

Start without Qwen3.8 performance patches. Measure correctness and fixed-workload1K/16K/35K/66K/128K/near-limit requests where the new model's proven capacity permits. Record exact input/output IDs, selected raw logits, TTFT, server prefill time, client decode formula, VRAM samples/allocator limits, MTP acceptance and cost, NCCL/P2P evidence and full fault recovery. Keep small/eager numerical oracles separate from production performance. Include Vision, Tools, cancellation and prefix/recurrent-state tests when supported. An unsupported length is NOT_TESTED, not PASS.

## Phase 4 — Port infrastructure and invariants

Port release management, transactional rollback, telemetry field/order preservation, recovery ownership and first-failure receipts, benchmark isolation and source/evidence sealing. Adapt exact PID/start-ticks/container binding to Qwen4's actual worker topology. Preserve requested-model affinity. Do not import K3, chunk4096, grouped positions, NVFP4 layout or KV memory budgets as defaults.

## Phase 5 — Re-audit synchronization

Inventory Class A correctness dependencies, fault checks, debug barriers and capture boundaries in Qwen4. Establish which stream owns each allocation and pending operation. Never infer that qwen3_next.py's grouped check location transfers. Any removal must retain detection/containment and pass numerical, long-context, graph/MTP/TP2 and real worker-loss tests.

## Phase 6 — Re-evaluate prefill chunks

Begin with a safe memory-qualified baseline. Qwen3.8's4096 is only a reference; its2848 Mamba alignment may not exist. Change one variable, maintain the same context/output/cache conditions, and compare independent fresh processes. Record actual executed chunk shapes and peak memory. Preserve OOM results, clean the exact owner, and verify hardware before reuse. Do not sweep larger chunks after a safety stop without reviewing the cause.

## Phase 7 — Re-evaluate MTP/speculation

Measure draft cost, verifier cost, acceptance, live memory, graph replay and end-to-end decode across lengths. High Qwen3.8 K3 acceptance is no guarantee for Qwen4. Compare K values only after attribution supports the experiment. Preserve multimodal first-pass and state-consumption invariants; no early sampler copy that bypasses post-forward validation.

## Phase 8 — Matched A/B

Use fresh processes, the same request order and allocation history, independent cache salt with zero prefix hits, identical compile/autotune state, pinned model/runtime/hardware and exactly one variable. Take warmups then formal samples, retain all samples and failed comparisons, compare exact output hash and selected logits, and disclose their scope. Reverse order or replicate deployments when effects approach variance. CUDA-event attribution and CUPTI profiling data never substitute for unprofiled throughput measurements.

## Phase 9 — Fault recovery

Use real66K-or-longer in-flight requests only within proven model capacity. Terminate one exact owned worker using pidfd. Record first failure, detection, stop issue, group/supervisor exit, fresh probe, full hardware observation, new supervisor, model loading/graph readiness, model ready, first real Gateway completion and replacement+60s. Repeat multiple fresh replacements and test rollback. Compare reload against Qwen3.8 v1≈109s and the separately preserved v2 result. Keep hardware gate semantics and stale-incident replacement protection intact.

## Phase 10 — Independent review, then promotion

Request independent review after all candidate tests and documentation are complete. Reviewer checks methodology, attribution, output/logits, actual source identity, ownership/hardware safety, real faults, replacement survival, rollback, source hashes and rejected results. Resolve valid findings. Build a new immutable production release and transactional promotion ledger; verify final live service/container/worker identities and a real completion. Keep Qwen3.8 as the accepted fallback until the separately authorized retirement gate.

Do not upgrade the custom driver as an incidental migration step. Driver/CUDA/NCCL changes require their own system/P2P/graph/fault qualification and cache boundary. CUPTI-associated Xid13 remains an unresolved diagnostic risk; use the latest XID13-INVESTIGATION.md and bounded stop policy.
''')
doc('QWEN4-AGENT-BOOTSTRAP.md','''# Qwen4 agent bootstrap

Read these files in order from `/home/frank/local-inference-production/docs/qwen38-optimization/`:

1. START-HERE.md and CURRENT-PRODUCTION.json.
2. QWEN4-MIGRATION-PLAYBOOK.md.
3. TRANSFERABLE-LESSONS.md and FAILURES-AND-REJECTED-IDEAS.md.
4. RECOVERY-ARCHITECTURE.md and RECOVERY-OPTIMIZATION.md.
5. PERFORMANCE-BASELINES.json and QWEN38-OPTIMIZATION-HISTORY.md.
6. XID13-INVESTIGATION.md, EVIDENCE-INDEX.json and SOURCE-MANIFEST.json; verify referenced hashes and final independent review.

Inherit release/rollback methods, evidence discipline, first-failure ownership, replacement protection and matched A/B. Re-test model class, quantization, chunks, K, graphs, KV/state, NCCL, memory and recovery. Never copy Qwen3.8 sync placements or numerical patches line-for-line into Qwen4. Freeze the current release as fallback and refresh live identity before changes. Historical PASS is scoped evidence, not Qwen4 qualification.
''')
failures=[
('8192 chunk warmup OOM','Larger chunks reduce launch overhead.','HISTORICAL: native FP32 normalization requested160MiB and warmup OOMed; cleanup/no-Xid checks passed.','Rejected; larger16384/32768/65536 not run.','Budget real temporary allocations, not only KV/weights.','Re-measure Qwen4 peak intermediates and alignment.','evidence/history/chunk-CLOSEOUT.md'),
('Event replacement gave no measurable gain','Current-stream event waits might beat device-wide checks.','MEASURED: real four-length A/B correct but no useful throughput gain.','Keep simpler qualified device-wide boundaries.','A cheaper primitive can still wait for the same critical work.','Re-attribute streams first.','evidence/qualification/FINAL-REPORT.md'),
('Early sampler index copy','Start copy before target forward to hide latency.','SOURCE/INFERRED: bypasses required post-forward validation and buffer readiness/lifetime.','Rejected without GPU deployment.','A wait location is not permission to move its dependent operation.','Re-derive state and DMA invariants.','evidence/qualification/evidence/decode-decision.json'),
('0.25s SHM retry','Shorter retry catches a dead peer sooner.','SOURCE: interruption can split message framing and change transport semantics.','Rejected.','Detect failure using lifecycle/ownership instead of breaking framing.','Audit Qwen4 IPC protocol.','evidence/qualification/FINAL-REPORT.md'),
('K2/K4 and NCCL protocol switches','Different speculation/protocol might accelerate decode.','MEASURED attribution: high K3 acceptance; collective≈10% includes peer wait, not isolated transport bottleneck.','Not justified; no performance claim for untested settings.','Choose one variable from attribution.','New model needs new K/protocol evidence.','evidence/qualification/evidence/decode-event-attribution.json'),
('First cold-cache telemetry comparison invalid','CPU serializer savings improve model throughput.','HISTORICAL: cache/autotune mismatch and OOM fallback contaminated observations.','Rejected and repeated with sealed compatible caches.','Control compile/cache history explicitly.','Never reuse incompatible driver/model caches.','evidence/qualification/FINAL-REPORT.md'),
('First proper telemetry observer A/B failed','Single serialization is harmless under GPU load.','MEASURED:66K decode−3.8023%; actual rollback; later reverse-order replication preserved the failed receipt.','Promoted only after further qualified evidence with disclosed limits.','Never rewrite a failed comparison as PASS.','Require meaningful GPU tests even for hot-path CPU work.','evidence/qualification/FINAL-REPORT.md'),
('CUPTI Xid13 incidents','Profiling is observational and fresh process avoids failure.','HISTORICAL: two Xid13 incidents; second was fresh-process35K. Successful1K/16K and12event-only requests do not prove universal profiler safety.','Separate bounded incident investigation; no production CUPTI.','Observer effects can include GPU failure; async sampler error is not root cause.','Requalify instrumentation independently.','XID13-INVESTIGATION.md'),
('Idle PCIe link downshift mistaken for identity drift','Current link speed must stay at the active-transfer rate.','HISTORICAL:32→2.5 downshift failed an overstrict recovery check.','Identity logic repaired; real P2P probe retained.','Distinguish dynamic power state from hardware identity.','Do not reproduce speed-only false failures.','evidence/qualification/FINAL-REPORT.md'),
('15s stop grace reported as recovery','Short cleanup means service recovered.','MEASURED v1 full completion203.288s; reload109.010s.','Report end-to-end timeline separately.','Detection/cleanup/admission are different endpoints.','Measure Qwen4 first real Gateway completion.','evidence/qualification/evidence/final-inflight-peer-loss/receipt.json'),
('Microbenchmark treated as throughput','27.4% serialization gain means comparable decode gain.','MEASURED: microbench CPU saving; model effects much smaller.','No extrapolated model speedup.','Measure the user workload directly.','Separate CPU, stream, model and request timings.','evidence/qualification/FINAL-REPORT.md'),
('Same-process repeats treated as independent deployments','Several requests establish restart/release variance.','METHOD: shared caches, allocation history and warmup correlate samples.','Preserve sample hierarchy; use fresh deployments when needed.','Do not inflate statistical independence.','Match process/request/cache order.','evidence/qualification/METHODOLOGY.md'),
]
lines=['# Failures and rejected ideas','','These records preserve rejected evidence. “Rejected” does not automatically mean unsafe; some choices were merely unsupported or weaker than the accepted alternative.','']
for title,h,e,d,l,q,ref in failures:lines += ['## '+title,'',f'Hypothesis: {h}',f'Evidence: {e} Source: [{ref}]({ref}).',f'Decision: {d}',f'Lesson: {l}',f'Qwen4 relevance: {q}','']
doc('FAILURES-AND-REJECTED-IDEAS.md','\n'.join(lines))
print(P)
