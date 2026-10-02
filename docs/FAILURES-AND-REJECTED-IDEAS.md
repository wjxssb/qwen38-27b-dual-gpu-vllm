# Failures and rejected ideas

These records preserve rejected evidence. “Rejected” does not automatically mean unsafe; some choices were merely unsupported or weaker than the accepted alternative.

## 8192 chunk warmup OOM

Hypothesis: Larger chunks reduce launch overhead.
Evidence: HISTORICAL: native FP32 normalization requested160MiB and warmup OOMed; cleanup/no-Xid checks passed. Source: [evidence/history/chunk-CLOSEOUT.md](evidence/history/chunk-CLOSEOUT.md).
Decision: Rejected; larger16384/32768/65536 not run.
Lesson: Budget real temporary allocations, not only KV/weights.
Qwen4 relevance: Re-measure Qwen4 peak intermediates and alignment.

## Event replacement gave no measurable gain

Hypothesis: Current-stream event waits might beat device-wide checks.
Evidence: MEASURED: real four-length A/B correct but no useful throughput gain. Source: [evidence/qualification/FINAL-REPORT.md](evidence/qualification/FINAL-REPORT.md).
Decision: Keep simpler qualified device-wide boundaries.
Lesson: A cheaper primitive can still wait for the same critical work.
Qwen4 relevance: Re-attribute streams first.

## Early sampler index copy

Hypothesis: Start copy before target forward to hide latency.
Evidence: SOURCE/INFERRED: bypasses required post-forward validation and buffer readiness/lifetime. Source: [evidence/qualification/evidence/decode-decision.json](evidence/qualification/evidence/decode-decision.json).
Decision: Rejected without GPU deployment.
Lesson: A wait location is not permission to move its dependent operation.
Qwen4 relevance: Re-derive state and DMA invariants.

## 0.25s SHM retry

Hypothesis: Shorter retry catches a dead peer sooner.
Evidence: SOURCE: interruption can split message framing and change transport semantics. Source: [evidence/qualification/FINAL-REPORT.md](evidence/qualification/FINAL-REPORT.md).
Decision: Rejected.
Lesson: Detect failure using lifecycle/ownership instead of breaking framing.
Qwen4 relevance: Audit Qwen4 IPC protocol.

## K2/K4 and NCCL protocol switches

Hypothesis: Different speculation/protocol might accelerate decode.
Evidence: MEASURED attribution: high K3 acceptance; collective≈10% includes peer wait, not isolated transport bottleneck. Source: [evidence/qualification/evidence/decode-event-attribution.json](evidence/qualification/evidence/decode-event-attribution.json).
Decision: Not justified; no performance claim for untested settings.
Lesson: Choose one variable from attribution.
Qwen4 relevance: New model needs new K/protocol evidence.

## First cold-cache telemetry comparison invalid

Hypothesis: CPU serializer savings improve model throughput.
Evidence: HISTORICAL: cache/autotune mismatch and OOM fallback contaminated observations. Source: [evidence/qualification/FINAL-REPORT.md](evidence/qualification/FINAL-REPORT.md).
Decision: Rejected and repeated with sealed compatible caches.
Lesson: Control compile/cache history explicitly.
Qwen4 relevance: Never reuse incompatible driver/model caches.

## First proper telemetry observer A/B failed

Hypothesis: Single serialization is harmless under GPU load.
Evidence: MEASURED:66K decode−3.8023%; actual rollback; later reverse-order replication preserved the failed receipt. Source: [evidence/qualification/FINAL-REPORT.md](evidence/qualification/FINAL-REPORT.md).
Decision: Promoted only after further qualified evidence with disclosed limits.
Lesson: Never rewrite a failed comparison as PASS.
Qwen4 relevance: Require meaningful GPU tests even for hot-path CPU work.

## CUPTI Xid13 incidents

Hypothesis: Profiling is observational and fresh process avoids failure.
Evidence: HISTORICAL: two Xid13 incidents; second was fresh-process35K. Successful1K/16K and12event-only requests do not prove universal profiler safety. Source: [XID13-INVESTIGATION.md](XID13-INVESTIGATION.md).
Decision: Separate bounded incident investigation; no production CUPTI.
Lesson: Observer effects can include GPU failure; async sampler error is not root cause.
Qwen4 relevance: Requalify instrumentation independently.

## Idle PCIe link downshift mistaken for identity drift

Hypothesis: Current link speed must stay at the active-transfer rate.
Evidence: HISTORICAL:32→2.5 downshift failed an overstrict recovery check. Source: [evidence/qualification/FINAL-REPORT.md](evidence/qualification/FINAL-REPORT.md).
Decision: Identity logic repaired; real P2P probe retained.
Lesson: Distinguish dynamic power state from hardware identity.
Qwen4 relevance: Do not reproduce speed-only false failures.

## 15s stop grace reported as recovery

Hypothesis: Short cleanup means service recovered.
Evidence: MEASURED v1 full completion203.288s; reload109.010s. Source: [evidence/qualification/evidence/final-inflight-peer-loss/receipt.json](evidence/qualification/evidence/final-inflight-peer-loss/receipt.json).
Decision: Report end-to-end timeline separately.
Lesson: Detection/cleanup/admission are different endpoints.
Qwen4 relevance: Measure Qwen4 first real Gateway completion.

## Microbenchmark treated as throughput

Hypothesis: 27.4% serialization gain means comparable decode gain.
Evidence: MEASURED: microbench CPU saving; model effects much smaller. Source: [evidence/qualification/FINAL-REPORT.md](evidence/qualification/FINAL-REPORT.md).
Decision: No extrapolated model speedup.
Lesson: Measure the user workload directly.
Qwen4 relevance: Separate CPU, stream, model and request timings.

## Same-process repeats treated as independent deployments

Hypothesis: Several requests establish restart/release variance.
Evidence: METHOD: shared caches, allocation history and warmup correlate samples. Source: [evidence/qualification/METHODOLOGY.md](evidence/qualification/METHODOLOGY.md).
Decision: Preserve sample hierarchy; use fresh deployments when needed.
Lesson: Do not inflate statistical independence.
Qwen4 relevance: Match process/request/cache order.

## DFlash2 V2 lifecycle admission

Hypothesis: a different speculative implementation can replace native MTP.
Evidence: HISTORICAL real target+draft load reached a missing V2 TP request-state snapshot contract; fatal TPGroupPoisoned, not proof of OOM or damaged weights. See [migration closeout](evidence/history/migration-CLOSEOUT.md).
Decision: archived, not adopted; existing production actually restored.
Lesson: never remove lifecycle/state guards or synthesize empty state just to make a faster runner start.
Qwen4 relevance: re-derive scheduler, slot, recurrent-state, staged/UVA write and stream/copy ownership protocols before porting infrastructure.

## Historical45K numerical drift

Hypothesis: a local captured layer mismatch explains final-logit variation.
Evidence: HISTORICAL local payload captures agreed; some final shared logits differed by up to1.25 with stable top1. Root cause stayed NOT_PROVEN_BUT_NON_BLOCKING. See [drift closeout](evidence/history/drift-CLOSEOUT.md).
Decision: preserve captures and uncertainty; no speculative mathematical workaround.
Lesson: selected payload equality does not prove complete execution equivalence.
Qwen4 relevance: calibrate baseline numerical variability and keep selected-logit versus full-vocabulary claims separate.

## Next-stage startup and diagnostic lessons

HISTORICAL/MEASURED: the first startup observer used a sitecustomize path that was shadowed by the image's earlier system sitecustomize. Its missing in-container hooks were not accepted as measurement coverage. A uniquely named .pth import fixed attachment, and per-process markers verified it. For Qwen4, prove that instrumentation actually loaded before interpreting absence of spans.

SOURCE/REJECTED: renderer v43 put CPU warmup before HELLO metadata replies, which blocked the engine's initialization rather than overlapping it. It never reached GPU activation. v45 waits until all replies are sent. The lesson is to trace protocol readiness, not infer overlap from a lexical context-manager boundary.

MEASURED/METHOD: the first Vision oracle rejected a correct geometry response because of a complete JSON code fence and capitalization. The raw failed receipt remains. Later checks explicitly normalize only those formatting differences and compare exact shape/color/position/schema content; this is a revised semantic oracle, not a retroactive PASS for the earlier campaign.

MEASURED/REJECTED: initial isolated matrix rendezvous failed before rank GPU work; R2 completed GPU work but failed communicator teardown with the graph retained. Whole-case results remain bounded failures, not successful Xid reproductions. R3 uses loopback rendezvous, destroys the graph object before communicator teardown and requires a final post-cleanup success marker. No hardware GPU reset is involved.

MEASURED/DECISION: graph capture around1.8s and safetensors iteration around2.9s did not justify eager-first capture or parallel weight I/O changes. Persistent loaders, precreated containers and spawn implementation changes add lifetime/ownership complexity; none was adopted. These choices are evidence-specific, not a prohibition on separately qualified Qwen4 work.
