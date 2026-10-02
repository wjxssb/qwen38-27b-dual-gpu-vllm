# 2026-09-23 — 生产记录更新：production-dense-recovery-20260923

> 本节记录宿主机 dense 栈的当前状态，不是新的容器发布；`v1.0.0` 镜像契约（`release.json`）保持不变。

- 当前生产 profile：`production-dense-recovery-20260923`，manifest
  `053466c0ea4101f3c53bc22d4550e3474e3b3a431f99b62bda995737e17350ef`，
  2026-09-23T00:21:21-07:00 晋升（`validation/production-dense-recovery-20260923/`）。
- 模型改为 `nvidia/Qwen3.8-27B-NVFP4`，revision `dbb8f445b3145f8a4c18ddc769f032d57d32867c`
  （`Qwen3_5ForConditionalGeneration`）；served 名与旧 API 别名 `unsloth/Qwen3.8-27B-NVFP4` 保留，
  统一 API 模型 ID 为 `qwen38-27b-dense`。
- 调度：prefill chunk 2048 → **4096**；Prefix Cache 开启（Mamba align，实际 block 2848）；
  native MTP K3 + FULL_DECODE_ONLY Graph（capture 4）；NCCL Simple、双向 P2P 门禁每实例校验。
- API 拓扑：Gateway 18080 为唯一入口；18081/18094 兼容转发至 18080；18082 relay → 18096 后端。
- 回退（不可变）：`production-dense-grouped-20260922`，manifest
  `87971548b51369f2b028f18bef4275196d70d7079aec3ef830d725257dafb0a5`。
- 本轮新增：CPU renderer warmup 与引擎 HELLO 重叠；密封模型文件 stat 身份 + 全 SHA 回退校验。
- 评审结论：`PASS_WITH_DISCLOSED_LIMITATIONS`；性能范围声明：本轮测量的是 reload 恢复与正确性，
  不是新的模型吞吐提升；已接受的吞吐基线保留于 `docs/PERFORMANCE-BASELINES.json`。
- 长期稳定性与首次非法内核唯一根因仍未证明；诊断代码不在生产路径内。
- 仓库新增公开文本资产：`profiles/production-dense-recovery-20260923/`（含 overlay 修复源码与
  control-binding 副本）、`legacy/production-dense-grouped-20260922/`、`docs/`（优化史、失败与
  否决想法、XID13 调查、恢复架构、参考 harness）、`validation/production-dense-recovery-20260923/`。
  `runs/`、`state/`、`cache-blessed/`、遥测与私有 relay 代码仍仅限宿主机。

---

# Release v1.0.0 — sm120-nvfp4-k3

> **Historical release note for tag `v1.0.0`.**  
> These commands and instructions apply to the `v1.0.0` release tag, not current `main`.  
> Current `main` is a local reference record; direct-launch scripts `launch.sh`/`stop.sh` have been retired (see [README.md](README.md) for current positioning and managed entry points).

公开定制运行时镜像 `ghcr.io/wjxssb/qwen38-27b-vllm:sm120-nvfp4-k3`（启动器按 digest 钉死拉取），
让任何一台双卡 RTX 5070 Ti（SM120）机器通过 `git clone --branch v1.0.0 --depth 1 https://github.com/wjxssb/qwen38-27b-dual-gpu-vllm.git && ./launch.sh` 直接复现
Qwen3.8-27B-NVFP4 的 262K 满血上下文 + MTP K=3 CUDA Graph 推理服务。

## 本镜像包含什么

1. **SM120 NVFP4 KV 移植（layer-1，6 文件 + 原生内核）**
   - SM120 HND KV 布局契约（NHD 会拒绝启动）
   - uint8 打包 `[data | scale]` KV + 显式 stride 视图读取
   - FA2 prefill/decode 路由（SM12x 无 trtllm-gen FP4 FMHA，全部走 FA2 paged reader）
   - SM12x 线性 V-scale writer（SM100 TRTLLM 才使用 swizzle）+ 启动失败即报错的 writer 防护
   - `_C_stable_libtorch.abi3.so` 以 `TORCH_CUDA_ARCH_LIST=12.0` 从源码编译，构建时校验含 `.sm_120a.cubin`
2. **Graph006 运行时 overlay（layer-2，12 文件，逐字节钉死）**
   - MTP K=3 native 投机解码（target/drafter 注意力后端分区、双工作区探针与账本）
   - FULL_DECODE_ONLY CUDA Graph（b=1 / q_len=4）合约解析与启动自检
   - TP 请求生命周期 fail-closed、SHM 广播护栏、引擎/执行器稳定性加固
3. **哈希钉死的构建契约与血缘（Hash-Pinned Build Contract & Provenance）**：`build/inputs.json` 对全部 26 个输入文件做 SHA-256 钉死；镜像构建时校验输入清单、
   在镜像内复核 13 个安装产物的最终哈希；镜像内 `/opt/qwen38-runtime/release.json` 为运行时契约
   （SHA-256 记录于 `release.json` 的 `image_contract_sha256`，启动器逐字节比对后才启动）。
   注：精确源码级重构（rebuild）除公开上游 commit 外，仍需叠加 6 个 SM120 关键修改文件（详见 `runtime-build/` 说明；维护者按需提供 patch payload）。

## 验证结果（真实门禁，证据在 validation/）

- `mtp3-graph`（默认模式）：严格 HTTP 质量门禁 **QUALITY_GATE_PASS**
  （中文语义金丝雀 + 8K/64K/128K/196K/262K，每窗口 2 次，严格 JSON/无重复输出/无空响应），
  0 OOM、0 Xid、0 死锁，容器干净退出并复核 GPU 无残留进程。
- `eager-baseline`：同门禁 **QUALITY_GATE_PASS**。
- 门禁使用与发布完全一致的 argv / 环境变量 / 资源上限，动态 GPU UUID，固定模型 revision。
- 吞吐：≈67.5 tok/s 为维护者在 Graph006（MTP K=3 图模式）配置下的双 5070 Ti 实测值；
  门禁本身只校验正确性，不重测吞吐。

## 修复与已知问题

- 修复：冷 autotune 缓存下 196K/262K 首次长预填的调优扫描会超过 vLLM 默认 300s RPC 超时导致
  TP fail-closed。发布 profile 内置 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900`（必需项）。
- 修复：`VLLM_SM120_MTP_GRAPH_EVIDENCE_DIR` 为 MTP 图代码硬依赖路径，早期草稿误按“遥测”删除导致
  启动后崩溃；已按验证过的 Graph006 配置恢复。
- 已知：eager 模式 262K 曾出现一次未复现的 Xid 13（图模式门禁全程未出现）；持续关注。
- 已知：Prefix Caching 在本 profile 显式关闭。

## 升级与回滚

- 镜像不可变，按 digest 拉取；回滚 = 改回上一个 digest。
- `eager-baseline` 为任何图模式异常下的兜底模式（在 `v1.0.0` tag 下为 `./launch.sh --mode eager-baseline`；当前 `main` 请参阅 README 定位说明）。
