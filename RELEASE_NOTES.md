# Release v1.0.0 — sm120-nvfp4-k3

公开定制运行时镜像 `ghcr.io/wjxssb/qwen38-27b-vllm:sm120-nvfp4-k3`（启动器按 digest 钉死拉取），
让任何一台双卡 RTX 5070 Ti（SM120）机器 `git clone && ./launch.sh` 直接复现
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
3. **可复现构建**：`build/inputs.json` 对全部 26 个输入文件做 SHA-256 钉死；镜像构建时校验输入清单、
   在镜像内复核 13 个安装产物的最终哈希；镜像内 `/opt/qwen38-runtime/release.json` 为运行时契约
   （SHA-256 记录于 `release.json` 的 `image_contract_sha256`，启动器逐字节比对后才启动）。

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
- `eager-baseline` 为任何图模式异常下的兜底模式（`./launch.sh --mode eager-baseline`）。
