# Recovery architecture

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
