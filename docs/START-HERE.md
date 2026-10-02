If you are preparing a new Dense Qwen runtime, especially Qwen4,
read this document and QWEN4-MIGRATION-PLAYBOOK.md before changing
the current production runtime.

This is the durable Qwen3.8 knowledge pack. Read CURRENT-PRODUCTION.json for the canonical release and final identity receipt; verify live state again before actions. This pack records facts, derived values, inference and unresolved questions separately. Next-stage recovery and bounded Xid investigation are closed; see CURRENT-PRODUCTION.json and the final independent review for the promoted release, retained fallback and disclosed limits.

MEASURED/REVIEWED: current canonical is `/home/frank/nvidia-dense-runtime/production-dense-recovery-20260923`, manifest `053466c0ea4101f3c53bc22d4550e3474e3b3a431f99b62bda995737e17350ef`. Its additional qualified optimizations overlap CPU renderer warmup with fresh engine initialization and use sealed model identities with full-SHA fallback. Reload median is 85.561s and full recovery median is 179.818s across three candidate faults; final production-copy fault completed in 181.952s. See RECOVERY-OPTIMIZATION.md and the final report for matched baselines and limits.

The accepted fallback is `/home/frank/nvidia-dense-runtime/production-dense-grouped-20260922`, manifest `87971548b51369f2b028f18bef4275196d70d7079aec3ef830d725257dafb0a5`. It is immutable. Dual RTX 5070 Ti, PHB topology, driver 610.57.04, current-boot ACS/P2P proof and NCCL P2P/CUMEM underpin its qualification. Do not treat a driver version string as proof that a stock driver has these custom P2P capabilities.

MEASURED/HISTORICAL: accepted optimizations are chunk4096, telemetry single serialization, grouped per-layer prefill fault checks, and ownership-bound TP failure recovery. K3 and capture4 are qualified settings for this model/runtime, not Qwen4 defaults. See QWEN38-OPTIMIZATION-HISTORY.md and FAILURES-AND-REJECTED-IDEAS.md before experimenting.

Evidence is indexed in EVIDENCE-INDEX.json and sealed with SHA256 in SOURCE-MANIFEST.json. Small core records and exact source snapshots are copied into this directory; original raw benchmark events and large traces remain indexed at their original paths. Never describe a copied summary as a new GPU test.

Read RECOVERY-ARCHITECTURE.md before any restart, stop or fault test. Ownership, replacement guard, pidfd, full fresh CUDA/P2P probe and 60-second hardware observation remain required. Health=200 alone is insufficient; require a real Gateway completion.

To reproduce benchmarks, first read `evidence/qualification/METHODOLOGY.md`, then `reference-harness/bench.py` and the fixture inventory. The copied harness is a reference with historical absolute paths: rebind it in a new isolated workspace, seal it, verify idle and source identity, match cache state, then run fresh-process single-variable A/B. Do not execute the historical lifecycle files blindly. Use 1 warmup + 3 formal requests per length for performance; check cold-prefix hits, exact token hashes and selected raw logits. Include 258K and Vision/Tools if the changed path can affect them.

Rollback: use the current task's verified ownership-aware transactional lifecycle and exact before-image bindings. Wait for idle or recovery completion, check the current instance/PID/start ticks/container again, restore the accepted binding, run fresh hardware gates and a real Gateway completion. Never use broad process kills or substitute a different model. Unknown ownership/hardware state remains fail-closed.

For Qwen4, start with QWEN4-AGENT-BOOTSTRAP.md. Port invariants and methods; re-derive model-specific patches.

MEASURED configuration nuance: sealed PROFILES.json says image limit2, but the accepted host launcher overrides it to2147483647 (video0). Actual worker inspection confirms the override and4GiB MM processor cache. Preserve actual behavior and inspect host bindings as well as profile files; do not accidentally restore a limit while copying the profile. This is not an arbitrary-image memory qualification.
