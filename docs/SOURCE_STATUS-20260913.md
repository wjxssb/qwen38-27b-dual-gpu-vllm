# Current mode: CLOSEOUT / MIGRATION

Latest user scope: `closeout-migration-20260914/REQUEST.txt`. Known 45K FP8 repeat variation no longer blocks migration or DFlash2. Executing three quick gates, then production integration, then independent DFlash2 A/B. Historical sealed evidence below remains unchanged.

## 2026-09-14T02:59:36.994894+00:00 — 同 boot RESTART_APP 恢复完成，V18 跑完并正常释放 GPU

无需整机重启：两卡 None、NVML/PCI/P2P 与 40 项 CUDA/双向 P2P 完整性检查通过。V18r2 已实际完成 7 个审计 RPC 和 16/16 数值请求；全程无新增内核故障。正常 exit0/OOM=false，之后 61.77 秒连续空闲，cleanup_verified=true，两卡 15 MiB、0%、None。

1K 同方法重复精确；45K 非 warmup 重复的最大共享 logit 差异为 prefill1.5/decode1.0。已观测到的部分首处分歧位于本地 FP8 量化/GEMM、发生在 TP 归约之前；量化 activation 与保守参考仍待验证。一个 prefill 对在 chunk21 才分歧，超出细粒度 GDN 窗口。未把这些浮点差异直接判作 checkpoint 损坏。

所需 64 层与 8 阶段 GDN 覆盖完整，native reported_drops=0；原监视器的 4 条偏移回退告警保留，相关轮转读取缺陷已通过 6 项 CPU 测试修复。NVIDIA 完整基线仍未通过，DFlash2 未启用。

[本轮完整记录](execution-20260913T1802/v18r2-startup-fatal-diagnostic/CLOSEOUT.md) · [恢复收据](execution-20260913T1802/v18r2-restart-app-recovery/RESULT.json)

## 2026-09-14T02:12:18.527101+00:00 — Xid 8 同次启动恢复通过，V18r2 已运行

按用户指定的 NVIDIA RESTART_APP 路径完成恢复，无主机重启或 GPU reset。两卡 recovery action=None，NVML/PCI/root P2P 身份正常；独立 CUDA 进程完成 8 项本卡 alloc/kernel-write/read、16 项双向 P2P copy、16 项直接跨卡 kernel-write，全量 1 MiB 校验均无差异。退出后连续超过 60 秒空闲，无新增内核故障。旧应用早已退出；只终止其仍锁住租约的精确 CPU 监督器 PID 250991，获取双租约后运行探针。

旧 V16r2 Xid 8、exit1、cleanup_verified=false 均保留。独立不可变恢复收据允许同一 boot 新实例；旧的“仅 Xid 8 必须新启动”要求由本次用户指令取代。Xid79 / Xid154 Node Reboot / GPU 不可访问 / 恢复探针失败才进入整机重启请求路径。

V18r2 manifest `f2c0a304c15197336710885b138c78352cfff63a670cbf2b47da825340f3e7be`，GPU 挂载及四种配置与 V18 完全一致，只增加宿主恢复验证。12 项恢复逻辑测试、78 文件重建、原镜像全挂载 CPU 检查通过。实例 `7ac72e4b943b4164b7872b1429b2cb1a` 已实际启动，正在原生模型启动与调优阶段；尚未宣称模型基线通过。

[恢复收据](execution-20260913T1802/v18r2-restart-app-recovery/RESULT.json) · [V18r2 计划](execution-20260913T1802/v18r2-startup-fatal-diagnostic/PLAN.json) · [NVIDIA Xid 官方分类](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)

NVIDIA_PRODUCTION_BASELINE=NO；DFlash2=BLOCKED_BY_NVIDIA_BASELINE。

## 2026-09-14T01:55:00.473988+00:00 — V18 修复已完成 CPU 验证，等待新启动环境

两处卷积修复已在 V15 的两张 GPU 上通过独立回归。新增四处启动/失败/退出流程修复通过 14 项 CPU 测试，观测器 5 项、分析器 4 项通过；V18 完整挂载与四种配置检查通过，76 个文件重建一致。新增流程修复尚未执行 GPU 验证。

V15 的 16 个请求完成，1K 重复精确，45K prefill/decode 最大共享 logit 偏差仍为 1.375/0.75。V16r2 固定 FP8 算法重复精确，随后启动调优出现 GPU 1 Xid 8；故障候选退出 1/OOM=false，当前无计算进程，两卡空闲。既有硬件隔离门禁仍保持，cleanup_verified=false，没有清除旧故障记录、GPU 复位、主机重启或生产切换。NVIDIA 完整基线未通过，DFlash2 未启用。

[修复及证据](REPAIR_REVIEW_V18.md) · [下一候选计划](execution-20260913T1802/v18-startup-fatal-ready/PLAN.json)

## 2026-09-14 01:17 UTC — V15 completed; fixed-tactic FP8 experiment dispatched

V15 completed16/16. Same-method1Kprefill/decode raw logits exact;45056prefill/decode shared-logit max errors1.375/0.75. Full64layer/TP coverage complete; corresponding rank outputs exact, nonfinite0, drops0. First observed nonwarmup divergences:prefill chunk5layer40 GDN outputprojection;decode chunk6layer50 outputprojection. Earlier observed gatednorm outputs equal. This isolates the region, not yet localGEMM vsTPsum. Both convolution repairs retain independent8cases perrank GPUproof. V15 normal Dockerexit0/OOMfalse,60.627seconds continuousidle,cleanuptrue; full diagnostic sealed658d7c367a6eb0b57aa6a64be979aba78506af68bded4c47a29017a690f0b434.

V16 private FP8 startup experiment is dispatched after fresh full-weight/P2P/ownership admission. Manifestacd4bcd7d044a3a7334270dae12a0f75ebaebbd63c7f516d9d594857cb8abb0a,77files reproduce byteexactly; nativeCPU allmounts/fourprofiles PASS11.962s and4 reference/fixture/metric/hook tests PASS. Sixprivate cases/rank compare8staticquant repeats,32fixedcuDNN and16fixedcuBLAS GEMMs plus independentCPUFP64 sampledrows. Modelweights read-only, noTPcollective orapplicationPOST. Modeldispatch unchanged. [Plan](execution-20260913T1802/v16-fp8-private-diagnostic/PLAN.json). NVIDIA baselineNOT_PROVEN;DFlash2blocked.

## 2026-09-14 00:44 UTC — two causal-conv defects repaired on both GPUs; NVIDIA numerics running

V15 native private-tensor regression passed8/8 cases on eachrank. Original BF16 products differ fromCPUFP32reference; repairedoutputs areexact inall8cases. Ordinarypacked-varlendecode has16/514wronghistoryelements inoriginalcode; repairedstates areexact. Fixedtwo productpromotions andguardedonlyspeculativehistorylengthadjustment. Nullslot/input/weightcanaries pass. These fixesdo notattribute maxseq1modeldrift inadvance. V14startupfailure remainssealed; noapplicationPOSTwasissued and61.772scontinuousidlecleanupverified.

OriginalV13 realactivation CPUreplay8cases confirms theoldproductarithmetic inliveNVIDIAmodel: oldreference differs13–18of10,485,760elements(within1BF16ULP), whereasFP32productreferencediffers3.25–3.28millionelements. Stateinputtails areexact. Thisis evidenceforrealpathprecision impact, not45Knondeterminism causality.

V15 prelude7RPC+20secondsPASS;1544quantscalars/1964coordinatesPASS; nativeTP2P2P32roundsx2directionsPASS. Thefrozen16request1K/45K eager-no-cache sequenceisrunning onisolated18098 withall48GDNlayers observed. Coldfirstrequestcompilerworkisowned. [NativeGPUrepairreview](execution-20260913T1802/v15-causal-state-diagnostic/private-kernel-regression/REVIEW.json), [actualactivationreplay](execution-20260913T1802/v13-causal-conv-cpu-replay/RESULT.json), [V15plan](execution-20260913T1802/v15-causal-state-diagnostic/PLAN.json). NVIDIAproductionbaselineNOT_PROVEN;DFlash2blocked.

## 2026-09-14 — V13 complete; V14 causal-conv precision repair dispatched

V13 completed16/16:1K repeats exact;45056 prefill/decode shared-logit variation1.5625/1.0. Full64layer/rank coverage complete, no nonfinite/drop; first nonwarmup differences atlayer16(chunk2),25/57(chunk8). Layer18 alone is not a root cause. Exactownedstop exited0/OOMfalse, no failure_reason,61.966seconds continuousidle andcleanuptrue.

V14 seals two FP32 product promotions in causal-conv plus8private-tensor native GPU regression cases perrank. Four CPU precision tests and11 observer tests pass; allmountedsource/fourprofiles nativeCPUvalidationPASS,71files reproduce byteexactly. Wider observation coversall48GDNlayers7stagesfirst12prefillchunks, no binarydumps. GPUregression andmodelnumerics pending. No productionadoption orDFlash2. [V14plan](execution-20260913T1802/v14-causal-precision-diagnostic/PLAN.json), [V13review](execution-20260913T1802/v13-gdn-focus-diagnostic/ROOT_NUMERICAL_REVIEW.json).

## 2026-09-14 00:01 UTC: pending-request shutdown fixed live; focusing the numerical divergence

V12 completed all16 frozen observations. All64layers/bothranks/everyprefill chunk are covered, no nonfinite or telemetry drops, and each request matches acrossTP ranks. Identical-input raw-logit variation remains:45Kprefill max2.375,45Kteacherdecode max0.5625;1Kbothmethods0. First nonwarmup prefill differences include chunk1/layer18, while other pairs first differ at layers46 and39; a warmup/decode pair differs at33. Layer18 is a focused investigation target, not an attributed sole root cause.

Root then submitted one bounded nonstream request and stopped the exact instance while generation17 was active. The client returned the expected shutdown HTTP500 in0.0858s without any client signal; Docker exited0/OOMfalse in0.7846s. Supervisor verified61.5379s continuousidle. Its historical first postexitNVMLutil100 reason/outcomeFAILED remains preserved. A separate66-test CPU repair allows at most10s utilization settling only after allGPU memory restores, computePIDs disappear and hardware fault checks pass, before starting the unchanged full60s window.

V13 is sealed38b4e51246a3eadf52c096cdf90f320c37aa676258a1a1da6543370b501da171 and passed actual26-mount/fourprofile pinned-image CPU validation. It retains64layer hashes and adds layer18 norm/projection/conv/GDN initial/final/cache state hashes plus bounded binary snapshots for generations9/11 chunks0/1. Observer source passed8nativeCPUtests including exact reconstruction and unchanged computationAST. No compute kernel change has yet been made or proven. Fresh GPU admission is in progress. NVIDIA productionbaselineNO; DFlash2 remainsblocked.

## 2026-09-13 23:31 UTC: owned shutdown reconciled; numerical layer diagnostic preparing

V10 no-split completed 40 observations and still differed within identical-input methods (maximum 2.125 shared raw-logit error). Root stopped it; the next 67,584-token request was interrupted by that stop. Engine workers exited, but a local consumer-before-producer shutdown patch stranded the pending HTTP collector. Closing the exact owned client allowed Docker to exit0/OOMfalse within0.70s. Supervisor then verified62.05s continuous idle and cleanup=true. Original RUNNING-with-ended client receipt is preserved and explicitly reconciled as root interruption, not a model crash.

Separate repairs now wake pending collectors on the API event loop, reject submissions resuming across shutdown, preserve positive drain, bound remaining HTTP tasks and record a stop exceeding120s without GPU reset or force-kill. Native pinned-image CPU tests reproduce the old bug and show connected-client HTTP shutdown exits after repair. Candidate v12 adds whole hidden/residual byte hashes after existing prefill decoder synchronization to find the first divergent layer, with v9 synchronous compute settings. This observer changes timing and cannot qualify performance. First unlaunched v12 assembly was correctly rejected for stale overlay integrity; r2 regenerates the integrity surface and is being validated. No NVIDIA production PASS or DFlash2 activation.

## 2026-09-13 23:05 UTC — v10 no-split full-context diagnostic running

Fresh candidate-v10-no-split instance f270df0aabcb41188b64b7ad87ad86af is running eager/no-prefix/TP2/NVFP4/no-spec. Only the existing SM120/121 NVFP4 FA2 split-KV flag differs from v8; source/profile/environment/ownership checks and full19-file checkpoint hashing passed. The seven-RPC prelude and 20second observation completed. Offline reconciliation again passed 1544 quant scalar comparisons and 1964 sample coordinates. Native startup logs show split-KV disabled and both NCCL P2P/CUMEM directions. The original64-request context calibration is now running; no numeric acceptance is implied.

The separate decoder-boundary observation source passed8 CPU tests, including the actual bounded telemetry writer persisting a pending boundary while a CPU synchronization fixture is blocked. It adds no GPU calls/synchronizations, remains unassembled/unactivated, and has not changed v10. Historical v8 stall/variation remain unresolved; DFlash2 stays blocked.

[Current plan](execution-20260913T1802/v10-no-split-diagnostic/PLAN.json) · [Current numeric state](execution-20260913T1802/v10-no-split-diagnostic/numeric/state.json) · [Observer source](decoder-boundary-observation-v11/README.md).

## 2026-09-13 22:56 UTC — numerical variation reproduced with async disabled

The frozen fresh scheduling A/B completed 19 requests per instance (16 numeric + 3 unforced controls), all with exact input/identity/lifecycle receipts. A (async on): same-method 1K/45K repeats exact. B (async off, live flags verified on both ranks): 1K exact, 45K prefill maximum shared-logit difference 1.125 and decode 1.5. Both groups returned to clean backend generation 19. Thus disabling async is not a sufficient fix; the old v8 235K stall remains separately unresolved. B has an owned stop request pending its 60second exit observation.

[Paired result](execution-20260913T1802/paired-async-sync-diagnostic/ROOT_COMPARISON.json). The next sealed diagnostic is candidate-v10-no-split, changing only the existing NVFP4 FA2 split-KV flag over v8, with original64-context sequence frozen. Source route-policy checks and pinned-image CPU validation passed; no GPU execution yet. [Plan](execution-20260913T1802/v10-no-split-diagnostic/PLAN.json).

## 2026-09-13 22:35 UTC — 235K stall preserved; fresh scheduling A/B underway

v8 calibration stopped at 60/64 completed: 235520-token prefill repeat 1 (index 60, generation 87) stopped inside both workers’ model forward RPC 15853; native pending sample RPC timed out after 300 seconds. HTTP 500 and automatic candidate exit were preserved, without retry. Docker exit 0/OOM false does not mean success: the supervisor outcome is FAILED/exit 2, with initial post-exit GPU utilization failure preserved and subsequent 60.113 seconds idle cleanup verified. Current captured kernel has no NVIDIA Xid; unrelated AMD display warnings remain un-attributed. Numeric same-path variation is unresolved; full baseline and DFlash2 remain blocked.

[Incident review](execution-20260913T1802/v8-stall-review/REVIEW.json) and [partial numeric report](execution-20260913T1802/v8-numeric-calibration/repeat-noise.json). A fresh, source-verified asynchronous instance has been dispatched for the frozen 1K/45K matched scheduling diagnostic; the synchronous counterpart changes only `--no-async-scheduling` and remains unlaunched. This limited comparison does not clear the 235K stall or reproduce the entire previous request history.

2026-09-13 22:01 UTC: v8 live on isolated18098. Native schema fix verified: malformed JSON/SSE bothHTTP400 before generation; P2P32x2x1MiB plus actual native P2P/CUMEM bothdirections PASS; full parameter/quant scalar/sample reconciliation PASS. Original21tool/state observations completed:16tool strictPASS,2no-tool exact-textFAIL from generated leadingnewlines; original cancelNOT_EXERCISED after4096reasoning tokens. Hostcancellation trigger repaired in cancellation-repair-v2(22CPUtests), actualgeneration22 cancelled; bothrank request/Mambaownership cleared, exact B prompt70/output63/raw63rows preserved,3unforcedcontrols exact initialdecode.64request numericcalibration1K..235K is now running. No production switch or DFlash2.

2026-09-13 21:40 UTC: v8 launched as isolated eager-no-cache instance f46190c7d5a54c0e8fedc2174aabd92c after source integrity, 119 controller tests, full native24mount CPU schema tests, deterministic assembly and fresh GPU/P2P/full-weight preflight. Startup/qualification in progress. Plan: seven audits, initial numeric pair, all18tools +cancel recovery, controls, then broader numeric calibration. NVIDIA fullbaseline NOT_PROVEN; DFlash2 blocked. Historical v6 schema failures and v4 Xid79 preserved.

## 2026-09-13 21:06 UTC: original40 complete; nearby schema API repair in progress

V6 completed all40 original application requests. The former generation40 nested-tool stream completed87tokens with exactschema; generationcomputedtokens passed409 and reached471, all177engine RPCs finalized. Three subsequent unforced1023-token probes reconstructed the identical1024-token prefix and exactly matched initialv6 decode rawtop20. Same-method v6 repeats are exact; prefill/decode delta0.9375 and cross-v2/v6 delta1.0 remain unexplained. Relevant saved M1/M1024 autotune tactics are unchanged, so changed other buckets do not attribute this difference.

Of the remaining15tool/state cases,8validcases passed; malformed-schemaJSONreturned500, then malformed-schemaSSEreturned200/error500 and the strict client stopped. Actual source: xgrammar rejected the bad type, guidance fallback raised KeyError(triggers), and lazy generation wrapped an internal error. Exact107-file capture preserved. Root requestedownedstop; containerexited0/OOMfalse; supervisorcompleted60sidlerelease. Its firstpostexitNVMLutil100 observation reason and exitstatus2 remain recorded despite finalcleanuptrue. NoXidobservedthisrun.

Independent requirement mapping corrects the earlier whole-Vision attribution: usersection18 finite matrix PASS(13image+11textstateprobes), original18mechanicalFAIL and3compilerfinite-validationFAIL retained separately. Broaderstate/graph/context/soak stillunproven.

V7logging-onlycandidate is sealed but unlaunched; exact-imageCPUproof confirmed NCCL_DEBUG_FILE=/dev/stdout restoresnativeNCCLlogs underPython-I/spawn. V8 will add peer-reviewed CPU schema validation/fallback repair before resumingtool/cancel andnumericalsweeps. NVIDIA productionbaselineNO; DFlash2blocked.

## 2026-09-13 20:34 UTC: related repairs reviewed; continuing NVIDIA baseline

V6 sealed `42ce156ad6ae6c6d321c4fb71b23d622a740851ad2450e9aeed2c667080ceb70`. Supervisor/launcher 119 CPU tests, inspection 55 CPU + 7 peer tests, 324 assembly checks pass. The v5 seal was never launched; v6 also preserves Python `-I` across the actual API handoff. GPU overlay and all model profiles remain identical to v4. Responses background storage explicitly disabled for the single API process.

Old v4 fault evidence and `cleanup_verified=false` remain unchanged. An independently checked immutable current-boot reconciliation binds the 64-second idle interval, current root P2P proof, old exited container and full kernel journal. Fresh preflight and current-boot fault scan pass. Combined pinned-image CPU verification precedes actual launch. Original 40 application requests remain frozen; seven ordered audit RPCs and a separate 20-second observation occur first. Neither original stall nor Xid79 has been attributed or cleared. Production baseline remains NO; DFlash2 remains blocked.

# NVIDIA baseline: repairing related runtime and audit process defects

Update after the user's explicit instruction to repair nearby bugs and continue: v4 completed its seven ordered inspection RPCs at 19:27:46 UTC, then GPU1 emitted Xid 79 at 19:27:51; both GPUs subsequently requested OS reboot recovery. None of the 40 planned application replay requests had been dispatched. The supervisor stopped the owned container, which exited at 19:27:53, but could not establish GPU release on that boot. Historical evidence is in `execution-20260913T1802/incident-v4-archive-corrected-2010/`. The later current boot is `3d20bce6-ba56-4fdc-b472-0793d9a23682`; this task did not reboot the machine. Its separate 60-second idle observation passed and does not rewrite the failed prior-boot cleanup.

Repairs are being developed separately in `supervisor-repair-v5/` and `inspection-repair-v5/`: preserve the first kernel fault, collect container logs even when NVML fails, coalesce repeated quarantine errors, retain fatal-event latches, reconcile prior-boot ownership explicitly, and exclude concurrent application requests during inspection. The historical snapshot helper's rejected dashed journal boot ID and missing command-failure status have been corrected and verified against the archived journal. These are identified audit/lifecycle defects, not a proven fix for Xid 79. The original generation-40 stall remains a separate unresolved event. The next candidate must pass source/CPU review and fresh admission before further GPU work.

NVIDIA_PRODUCTION_BASELINE=NO. UNSLOTH_ROLLBACK_AVAILABLE=YES. DFlash2=BLOCKED_BY_NVIDIA_BASELINE. The active stage remains the NVIDIA TP2 baseline with speculative decoding disabled. Original Dense, Flash and Gateway units were inactive at this execution window and have not been switched or restarted. The previous-boot Flash incident and its stop marker remain preserved.

The official checkpoint's 19 files are verified. Actual v2 execution on boot `51965c80-f35a-460d-8171-2bd566916607` passed the requested quant mapping audit: complete metadata for 1,893 parameters per rank, all 1,544 quant scalars, and 1,964 exact sampled coordinates across 254 tensor/rank entries. The existing TP communicator passed 32 rounds in each direction with 1-MiB deterministic/random byte checksums; current-worker NCCL logs identify both P2P/CUMEM routes. These finite observations do not establish stability or model mathematical correctness. Full TP mapping still lacks direct observation of an identified logits gather. See `runtime-gate-review/REVIEW.md`.

The eager/no-prefix v2 instance completed 29 of 45 text/Vision/tool cases. Its next nested-object streaming request stopped making generation progress after both workers entered execute RPC 18579, overlapping sample RPC 18578. No compiler process explained the delay. First incident evidence is preserved in `execution-20260913T1802/incident-nested-stream-1857/`; `incident-analysis/FINDINGS.md` separates observed boundaries from unproven root causes. TOOL_CALLING=FAIL for operational noncompletion. No new Xid/CUDA/NCCL error was observed in this execution's retained evidence.

Only the owned v2 candidate was stopped. It exited, and the supervisor subsequently verified a full 60-second GPU release window. An initial post-exit non-idle utilization sample was retained as a supervisor failure reason; the later clean observation does not erase it. A separately planned fresh instance of the same sealed v2 configuration completed the nested-object JSON/stream pair, with 87 tokens each and exact valid arguments; stream duration was 7.85 seconds. It then stopped and passed its own 60-second release observation. This diagnostic does not clear the original failure. See `execution-20260913T1802/repro-nested-fresh-result/RESULT.json`. A separate v2-derived diagnostic with CPU execution-boundary logs is being prepared, preserving the original scheduling behavior.

The 1K prefill and teacher-forced decode paths repeat exactly within each method, but shared raw top-20 logits differ by up to 0.875 across methods. This has not been accepted as normal noise; longer-context and causal numerical checks remain pending. Vision responses recognize the images' central facts, while original strict formatting and compiler-fix finite-validation assertions remain unsatisfied. Eleven interleaved fixed-text probes match the first probe's 70 token IDs and raw-logit rows exactly; this is bounded evidence, not completion of the cross-request/state gate. See `vision-runtime-review/REVIEW.md`.

Graph/prefix/long-context qualification, complete tools/cancellation, long soak, real coding A/B, OpenCode/Gateway E2E and production cutover remain unproven. CPU-tested diagnostic, soak and isolated coding harnesses are staged. `ACCEPTANCE_LEDGER.json` is the detailed evidence ledger; no baseline acceptance record exists.
