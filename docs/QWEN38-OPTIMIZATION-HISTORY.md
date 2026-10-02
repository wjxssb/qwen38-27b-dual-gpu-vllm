# Qwen3.8 optimization history

## Hardware foundation — HISTORICAL, with current observations

The platform is dual RTX 5070 Ti on PHB topology. The historical stable foundation used the Aikitoria 610.57.04-p2p-v3 open-kernel branch, commit `461d638b7a91f73702001e343ec91b275a43ec35`, on kernel 7.0.0-30. Secure Boot was disabled and lockdown was none. Current read-only observations confirm driver, topology, Secure Boot and boot receipt in `evidence/hardware-foundation-current.json`.

Targeted ACS handling identifies AMD 1022:14db root ports 00:01.1 and 00:01.3, changes ACS control at 0x2a6 to clear request/completion redirection, and verifies readback. System ACS → system P2P proof → user P2P gate → actual NCCL channel validation form separate gates. The current-boot root receipt is `/run/p2p-stable/verified`. Do not generalize this register recipe to a different motherboard. The historical Sep10 report contains a later-GRUB-reboot-pending note; it is not the current boot's status.

MEASURED/HISTORICAL: qualified transport is P2P/CUMEM both directions, without silent SHM/NET substitution. Chunk closeout measured 128 MiB peer copies about 28.63 GB/s and NCCL 64 MiB bus bandwidth about 23.3 GB/s; these old microbenchmarks are not current model throughput. Stock-driver rollback assets existed, but their automatic application was not qualified. No driver upgrade or rollback is implied by this pack.

## Initial runtime and safety repairs — HISTORICAL

The retained graph006 specification pins image `sha256:b6190f100526423e5855c5c0f98773744c91a69a7e112993d66bed038c9b76c6`, Unsloth revision `57926b...`, TP2/NVFP4, max262144, chunk2048, MTP K3, FULL_DECODE_ONLY capture4, custom all-reduce off, KV 2928199680 bytes/card, CPU4, language-only and no-prefix-cache. That earliest spec used NCCL_P2P_DISABLE=1. Later P2P/CPU8/prefix repairs must not be retroactively attributed to the earliest spec. The pinned image's inspected package version is vLLM 0.26.1rc1.dev608+g99a10304d.cu129; source-spec alone does not prove a running image.

Graph006 repairs preserved pinned-source lifetime across asynchronous DMA, side-stream ownership, graph input strides/device identity, fatal reporting on non-output ranks, and API cancellation cleanup. Keeping defensive checks is justified by these invariants, not their age. The historical 14-request/28,672-decode-token repair exercise was functional evidence, not a controlled speedup benchmark. Sources: `evidence/history/initial-graph006.json`, `graph006-repair-REPORT.md`.

The NVIDIA checkpoint migration pinned revision `dbb8f445b3145f8a4c18ddc769f032d57d32867c`. Actual class is Qwen3_5ForConditionalGeneration despite the public Qwen3.8 alias. Native MTP tensors were retained, shared embedding/head semantics inspected, and the NVFP4 shared head was not incorrectly required to be BF16. Vision first pass remains outside the custom draft graph; follow-up draft steps use their own embedding buffers. Vision, Tools, prefix state and MTP required actual qualification. See `evidence/history/migration-CLOSEOUT.md` and `mtp-CLOSEOUT.md`.

## Intermediate NVIDIA no-spec migration and rejected V2 route — HISTORICAL

V26 first unified production on NVIDIA NVFP4 with TP2, Graph capture1, prefix caching and **no speculative decoding**. Native MTP K3 was enabled in a later qualified stage; do not conflate the historical Unsloth K3 configuration with this intermediate NVIDIA no-spec baseline. Its67K+256 result (about43.92 decode tok/s) is a different configuration and must not be merged with the later K3 performance table. The earlier30-minute/278-request V23 soak is historical evidence, not a newly repeated soak for grouped or recovery releases.

The migration moved active mounts and GPU/Gateway leases to WD storage, removed hidden `.resolve()` traversal into the failing Samsung path, and retained the old campaign string solely as an identity label. All public aliases resolved to the same NVIDIA backend; an `unsloth/...` wire name did not mean Unsloth weights or fallback. The migration audited131 inherited optimization items and retired only an exact verified file manifest. Original incidents and archived source stayed preserved. The durable archive index is `/home/frank/nvidia-runtime-history/FINAL_ARCHIVE_INDEX.json`; retired Unsloth/Flash entries are reference material, not current fallback targets.

DFlash2 loaded target+draft in a real V2 candidate but lacked the inherited TP request-state snapshot/lifecycle protocol. Startup correctly failed with TPGroupPoisoned; disabling guards or inventing empty state was rejected. Actual selector tensor qualification, request A/B and performance were not reached. The accepted no-spec baseline was restored, then native MTP was qualified separately. Source: `evidence/history/migration-CLOSEOUT.md`.

A separate historical45K numerical-drift investigation retained NOT_PROVEN_BUT_NON_BLOCKING: captured local payloads agreed, while selected final shared-logit differences reached1.25 with stable top1. It did not justify changing production math. This is distinct from the later Sep22 single-variable comparisons that produced exact output and selected-logit matches. Source: `evidence/history/drift-CLOSEOUT.md`.

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
