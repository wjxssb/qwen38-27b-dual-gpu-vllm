"""Render factual final/provisional report and durable current-state pointer."""
from common import *
import lifecycle
P=PROD/'docs/qwen38-optimization'

def main():
    r=read(Q/'recovery/RECOVERY-OPTIMIZATION-RESULTS.json')
    stage=read(Q/'evidence/production-copy-qualification/receipt.json')
    assert stage['state']=='QUALIFIED_ACTIVE_PENDING_INDEPENDENT_REVIEW'
    release=Path(stage['production']);digest=sha(release/'MANIFEST.json')
    committed=(Q/'evidence/promotion-commit.json').exists() and read(Q/'evidence/promotion-commit.json')['state']=='PROMOTED'
    review=read(Q/'evidence/independent-review.json') if (Q/'evidence/independent-review.json').exists() else None
    minimal=read(Q/'xid/evidence/minimal-matrix-r3/receipt.json')
    full=read(Q/'xid/evidence/full-original-35k/receipt.json') if (Q/'xid/evidence/full-original-35k/receipt.json').exists() else None
    current=read(P/'CURRENT-PRODUCTION.json')
    current.update(state='PROMOTED' if committed else 'QUALIFIED_ACTIVE_PENDING_INDEPENDENT_REVIEW',runtime=str(release) if committed else str(lifecycle.accepted()),manifest_sha256=digest if committed else sha(lifecycle.accepted()/'MANIFEST.json'),active_qualified_release=str(release),active_qualified_manifest_sha256=digest,fallback={'runtime':str(NV/'production-dense-grouped-20260922'),'manifest_sha256':sha(NV/'production-dense-grouped-20260922/MANIFEST.json')},binding=str(lifecycle.bindings(release)),live_identity={'path':'evidence/next-stage/final-control-identity.json','sha256':sha(Q/'evidence/final-control-identity.json'),'scope':'Actual live process snapshot; refresh identity before future actions'},source_control='sources/control/',new_optimizations=['CPU renderer warmup after engine HELLO replies overlaps independent engine startup','sealed model file stat identity with unchanged full SHA fallback'],diagnostics_in_production=False,review={'verdict':review['verdict'] if review else 'PENDING','path':'evidence/next-stage/independent-review.json'},recovery_evidence='evidence/recovery/RECOVERY-OPTIMIZATION-RESULTS.json',performance_evidence='PERFORMANCE-BASELINES.json',performance_scope='Previous accepted throughput baseline preserved; this round measures reload recovery and correctness, not new model throughput speedup.')
    save(P/'CURRENT-PRODUCTION.json',current)
    start=P/'START-HERE.md';s=start.read_text()
    original='Next-stage recovery/Xid work is still in progress until its final report and independent review close.'
    pending='Next-stage recovery and bounded Xid testing are complete; the active new release remains provisional until independent review and canonical commit.'
    final='Next-stage recovery and bounded Xid investigation are closed; see CURRENT-PRODUCTION.json and the final independent review for the promoted release, retained fallback and disclosed limits.'
    for old in [original,pending,final]:
        if old in s:s=s.replace(old,final if committed else pending)
    start.write_text(s)
    a=r['groups']['A_fallback'];c=r['groups']['C_renderer_inventory'];f=r['production_copy_fault']['metrics']
    total_gain=r['comparisons']['A_fallback__C_renderer_inventory']['failure_to_completion']
    reload_gain=r['comparisons']['A_fallback__C_renderer_inventory']['supervisor_to_ready']
    matrix='; '.join(x['mode']+':'+x['state']+('/Xid' if x['new_xid'] else '') for x in minimal['rows'])
    new_xids=sum(bool(x['new_xid']) for x in minimal['rows'])+int(bool(full and full.get('new_xid')))
    title='QWEN3.8 NEXT-STAGE FINAL REPORT'
    report=f'''# {title}

STATUS: {'COMPLETE — PROMOTED' if committed else 'TECHNICAL WORK COMPLETE — PENDING FINAL INDEPENDENT REVIEW AND CANONICAL COMMIT'}

## 1. RECOVERY OPTIMIZATION

HISTORICAL accepted v1: peer loss→first successful Gateway completion203.288s; new supervisor→ready109.010s. Its immutable release and original receipts remain unchanged.

MEASURED current A3 baseline: completion median{a['failure_to_completion']['median_seconds']:.3f}s; reload median{a['supervisor_to_ready']['median_seconds']:.3f}s.

MEASURED candidate C3: completion median{c['failure_to_completion']['median_seconds']:.3f}s, range{c['failure_to_completion']['range_seconds']}; reload median{c['supervisor_to_ready']['median_seconds']:.3f}s, range{c['supervisor_to_ready']['range_seconds']}.

DERIVED from contemporaneous A3/C3 medians: full recovery reduced{total_gain['reduction_seconds']:.3f}s ({total_gain['reduction_percent']:.2f}%); reload reduced{reload_gain['reduction_seconds']:.3f}s ({reload_gain['reduction_percent']:.2f}%). A→B isolates renderer overlap; B→C isolates model identity caching. Block order A3/B3/C3 is disclosed; n=3 descriptive measurements do not prove population-wide variance or non-regression.

MEASURED final production-copy worker fault: completion{f['failure_to_completion']['median_seconds']:.3f}s, reload{f['supervisor_to_ready']['median_seconds']:.3f}s; replacement observation{r['production_copy_fault']['replacement_observation_seconds']:.3f}s. [Complete timing table]({Q}/recovery/RECOVERY-OPTIMIZATION-RESULTS.md), [all fault/native startup timelines]({Q}/recovery/RECOVERY-NATIVE-FAULT-TIMELINES.json).

Actual measured contributors: CPU renderer warmup≈16s, repeated full model SHA≈11–13s, serialized child-import pipe waits≈7.9s each; graph capture≈1.8s and safetensors iteration≈2.9s. Nested stages overlap and are not additive. Some CUDA context, quantization/device-placement and graph-validation subphases were not independently observable without heavier tracing; [startup coverage and limitations]({Q}/recovery/RECOVERY-RELOAD-BREAKDOWN.json) remain explicit.

Changes: preserve original CPU renderer warmup but perform it after all HELLO metadata replies while the fresh EngineCore initializes; preseal full model SHA then check file and environment identities on reload, invoking the unchanged full verifier on any mismatch. No preparation bypasses the hardware gate. Model mathematics, Graph/MTP/chunk/KV/TP/NCCL, exact ownership, pidfd, replacement protection, CUDA/P2P probe and full60s hardware observation remain. Trusted immutable local file assumptions and residual loader TOCTOU are disclosed. No cache is forcibly reused across incompatible driver/model/architecture changes.

Qualification: CPU handshake/warmup/integrity fallback tests; exact output hash and selected raw-logit checks at1K/16K/66K/258K; Vision/Tools/state checks;9candidate/baseline real66K worker faults plus final production-copy fault; fresh probes and replacement observations; actual fallback restore/completion followed by final qualified-copy reactivation. This is not a new prefill/decode throughput A/B. [Production copy receipt]({Q}/evidence/production-copy-qualification/receipt.json).

Promoted: {'YES' if committed else 'NO — FINAL REVIEW GATE PENDING'}. Current accepted fallback is not overwritten. Xid observers are not in the new payload.

## 2. XID13 INVESTIGATION

Historical incidents:2. Historical Xid timestamps2026-09-23T00:18:48.733842UTC (GPU1) and03:24:36.787765UTC (GPU0), EXTRA_INLINE_DATA. Fresh35K failed historically as well as a second16K session. The sampler wait is an asynchronous surfacing point, not an established fault origin.

New diagnostic configurations with Xid:{new_xids}. Minimal matrix: {matrix}. Final matrix state:{minimal['state']}. Full fresh35K original-profiler probe:{json.dumps({k:full.get(k) for k in ['state','result','new_xid','error']} if full else {'state':'NOT_RUN_AFTER_MINIMAL_STOP'},ensure_ascii=False)}.

Separate observed failure: modeG produced CPU SIGSEGV on both ranks inside libcupti.so.12 at offset0x13c170; retained core shows a null-pointer read during cuGraphLaunch. No new Xid was logged. H/full35K were NOT_RUN after this safety stop; only offline core analysis followed. See the durable investigation for source hashes, stack limitations and retained core.

Root cause confidence: **NOT_PROVEN**. Negative minimal results do not cover full Qwen/MTP, high VRAM, all kernels or historical observer code. Earlier rendezvous and communicator-teardown harness failures are preserved separately; they are not Xid reproductions. Active experiments are bounded and stop on new Xid/error; restore includes fresh hardware proof and observation. No driver upgrade or production mathematical workaround is justified. [Detailed investigation]({P}/XID13-INVESTIGATION.md), [official evidence tiers]({Q}/xid/OFFICIAL-RESEARCH.md).

Production impact: CUPTI remains excluded; previously qualified bounded CUDA-event attribution has12successful requests in its tested scope, not a universal safety guarantee. Any new driver/CUPTI test needs independent system qualification.

## 3. QWEN3.8 KNOWLEDGE PACK

Path: `{P}`. Includes START-HERE, CURRENT-PRODUCTION, optimization history, performance baselines, recovery architecture/optimization, Xid investigation, failures/rejected ideas, transferable lessons, Qwen4 migration playbook/bootstrap, source manifest and evidence index. Core sources and receipts are durably copied; large raw traces/data remain explicitly indexed at their original paths. The copied historical FINAL-SEAL covers original qualification evidence. Reference harnesses retain historical paths and require a newly sealed/rebound workspace before execution.

Hash manifest: [SOURCE-MANIFEST.json]({P}/SOURCE-MANIFEST.json); separate SOURCE-MANIFEST.sha256 seals both indexes. Self-referential index files are explicitly excluded from their own file list. File created_at fields use filesystem mtime, not asserted original authoring time.

## 4. QWEN4 MIGRATION ENTRY POINT

Read: [{P}/QWEN4-AGENT-BOOTSTRAP.md]({P}/QWEN4-AGENT-BOOTSTRAP.md), then continue Qwen4 migration. Transfer invariants, evidence, ownership, A/B and promotion methods. Re-derive model-specific patches and re-test K/chunk/capture/quantization/NCCL/memory; Qwen3.8 parameters are references, not Qwen4 defaults.

## 5. CURRENT PRODUCTION

Accepted canonical: `{release if committed else lifecycle.accepted()}`.
Qualified active release: `{release}`.
Qualified manifest: `{digest}`.
Immutable fallback manifest: `87971548b51369f2b028f18bef4275196d70d7079aec3ef830d725257dafb0a5`.
Runtime: pinned imageb6190f10…, vLLM0.26.1rc1.dev608+g99a10304d.cu129, custom610.57.04 P2P-enabled driver.
Model: NVIDIA revisiondbb8f445b3145f8a4c18ddc769f032d57d32867c; actualQwen3_5ForConditionalGeneration.
TP2; MTP K3; chunk4096/actual Mamba2848; context262144; FULL_DECODE_ONLY capture4; KV3039750144bytes/card; NVFP4/HND; NCCL Simple and current P2P/CUMEM/ACS.
Actual host override: images2147483647, video0, MM processor cache4GiB; profile's image2 is not the effective live value. This is not an arbitrary image-count memory qualification.
Source, installed binding/mount and actual live process executable/PID/start-ticks/argv are separate in [final identity]({Q}/evidence/final-control-identity.json); worker classes/effective configuration are recorded in [worker inspection]({Q}/evidence/production-copy-qualification/final-live/worker-runtime-inspection.json).

## 6. REMAINING UNPROVEN ITEMS

- Exact CUPTI/Xid root cause and failing kernel; driver/CUPTI/runtime/hardware alternatives remain.
- Full-vocabulary numerical equivalence; tests compare deterministic tokens and selected raw logits.
- Population-wide recovery/performance confidence, arbitrary multimodal loads and future driver/model cache compatibility.
- Some fine CUDA startup subphases and byte-exact H2D attribution; instrumentation effects are not a causal overhead A/B.
- Zero compilation/cache misses:8new Triton artifacts from one rejection sampler specialization appeared after readiness; unchanged seeded files alone do not prove every lookup hit.
- CPU imports during hardware observation, persistent loaders, spawn pipe changes, eager/lazy graph startup and Qwen4-specific parameters were not adopted or qualified.

## 7. INDEPENDENT REVIEW

Verdict: {review['verdict'] if review else 'PENDING — only after all technical workflows and documentation are complete'}.
Findings: {json.dumps(review.get('findings',[]) if review else 'PENDING',ensure_ascii=False)}.
Review and closure: `{Q}/evidence/independent-review.json`, `independent-review-closure.json`.
'''
    (Q/'FINAL-REPORT.md').write_text(report)
    print({'state':current['state'],'report':str(Q/'FINAL-REPORT.md')})

if __name__=='__main__':main()
