# Qwen3.8-27B Dual-GPU vLLM Production Stack

> 在双卡 16GB 消费级 Blackwell 显卡（2 × RTX 5070 Ti，SM120）上满血运行 **262,144 (256K) 原生上下文**的
> Qwen3.8-27B-NVFP4 生产级推理栈：**NVFP4 4-bit KV Cache + MTP K=3 投机解码 + FULL_DECODE_ONLY CUDA Graph**。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-Qwen3.8--27B--NVFP4-orange.svg)](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
[![Context](https://img.shields.io/badge/Native%20Context-262K%20Tokens-green.svg)](#)
[![GPU](https://img.shields.io/badge/Hardware-2%C3%97RTX%205070%20Ti%20SM120-red.svg)](#)

---

## 一键启动

```bash
git clone https://github.com/wjxssb/qwen38-27b-dual-gpu-vllm.git
cd qwen38-27b-dual-gpu-vllm
./launch.sh
```

脚本自动完成：拉取**摘要钉死**的公开运行镜像（GHCR）→ 校验镜像内运行时契约（SHA-256）→
按固定 revision 下载模型权重（仅首次，约 11GB）→ 绑定 GPU 锁 → 以验证过的完整参数启动 OpenAI 兼容服务
（`127.0.0.1:8000`）。`Ctrl-C` 或 `./launch.sh --stop` 干净退出并释放 GPU 锁。

常用参数：

```bash
./launch.sh --plan                        # 只打印将执行的容器配置
./launch.sh --mode eager-baseline         # 无 MTP / 无 CUDA Graph 的兜底模式
./launch.sh --download-only               # 只下载模型
./launch.sh --gpus GPU-xxxx,GPU-yyyy      # 显式指定两卡 UUID
```

## 手动一行 docker run（等价于验证过的 mtp3-graph 模式）

<details>
<summary>展开完整命令</summary>

```bash
docker run --gpus '"device=GPU-XXXX,GPU-YYYY"' --ipc=private --shm-size=1g \
  --cpus=4 --memory=35g --ulimit memlock=67108864 --pids-limit=1024 \
  -p 127.0.0.1:8000:8000 \
  -v ~/.qwen38-runtime/models:/models -v ~/.qwen38-runtime/cache:/cache \
  --env-file <(curl -s https://raw.githubusercontent.com/wjxssb/qwen38-27b-dual-gpu-vllm/main/mtp3-graph.env) \
  ghcr.io/wjxssb/qwen38-27b-vllm:sm120-nvfp4-k3 \
  python3 -m vllm.entrypoints.openai.api_server \
  --model /models/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/57926baca9a82b4d6906b43f2750d55315f5b10f \
  --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 2 --distributed-executor-backend mp \
  --max-model-len 262144 --max-num-batched-tokens 2048 --max-num-seqs 1 \
  --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.914 \
  --linear-backend auto --disable-custom-all-reduce \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --language-model-only \
  --attention-config '{"backend":"FLASHINFER","use_trtllm_attention":false}' \
  --block-size 16 --enable-chunked-prefill --no-enable-prefix-caching \
  --spec-method mtp --spec-tokens 3 --no-enforce-eager \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}' \
  --cudagraph-metrics --kv-cache-memory-bytes 2928199680
```

推荐直接用 `./launch.sh`：它还负责 GPU 锁互斥、镜像契约校验与模型完整性检查。
</details>

或使用 Compose（先把 `config.env.example` 复制为 `config.env`）：

```bash
docker compose up -d
```

## 核心卖点

| 维度 | 说明 |
| :--- | :--- |
| **适配硬件** | 双卡 RTX 5070 Ti / 消费级 Blackwell（SM120，16GB × 2，TP=2，PCIe P2P 关闭走系统内存中继） |
| **NVFP4 4-bit KV Cache** | SM120 专属移植：HND KV 布局 + uint8 打包 `[data\|scale]` 契约 + FA2 prefill/decode 路由 + SM12x 线性 V-scale writer（`_C_stable_libtorch` 原生内核按 SM120a 编译） |
| **MTP K=3 + CUDA Graph** | Qwen3.8 MTP 投机解码（K=3），FULL_DECODE_ONLY 图（b=1 / q_len=4），启动时图合约自检；target/drafter 双图工作区探针 |
| **真实吞吐** | **≈67.5 tok/s**（维护者在 Graph006 MTP K=3 图模式下的双 5070 Ti 实测；历史对照数据见 [BENCHMARK_DOSSIER.md](BENCHMARK_DOSSIER.md)） |
| **262,144 满血上下文** | 8K / 64K / 128K / 196K / 262K 长度召回全部通过严格 HTTP 质量门禁（含中文语义金丝雀），显存 0.914 利用率不爆卡、0 OOM |
| **工程稳定性** | TP 生命周期 fail-closed 看门狗、SHM 广播护栏、GPU 锁互斥、镜像内契约校验、只读根文件系统、无任何 `latest` 漂移 |

## 公开验证（可复核）

`validation/` 内为分阶段 GPU 门禁的原始 JSON 证据（严格 HTTP 质量门禁：
中文语义金丝雀 → 8K → 64K → 128K → 196K → 262K，每窗口 2 次，零重复输出、严格 JSON 答案校验）：

| 模式 | 结果 | 证据 |
| :--- | :--- | :--- |
| `mtp3-graph`（默认） | `QUALITY_GATE_PASS`（6/6 用例 PASS） | [validation/stage-b-mtp3graph-20260905T234604Z/](validation/stage-b-mtp3graph-20260905T234604Z/quality-report.json) |
| `eager-baseline`（兜底） | `QUALITY_GATE_PASS`（6/6 用例 PASS） | [validation/stage-a-eager-20260905T214445Z/](validation/stage-a-eager-20260905T214445Z/quality-report.json) |

固定版本链：

| 组件 | 锁定值 |
| :--- | :--- |
| 运行镜像 | `ghcr.io/wjxssb/qwen38-27b-vllm:sm120-nvfp4-k3`（启动器按 digest 拉取，见 `release.json`） |
| 镜像基础 | `vllm/vllm-openai@sha256:96a70bbb…e73`（vLLM `99a10304`，torch 2.13.0+cu129） |
| 定制运行时 | vLLM `99a10304` + SM120 NVFP4 rebase（6 文件 + 原生内核）；Graph006 运行时 overlay（12 文件）逐字节钉死于 `build/inputs.json`（SHA-256 全清单） |
| FlashInfer | `9dc1b249`（0.6.16.post3），关键源文件与安装版逐字节核对 |
| 模型 | `unsloth/Qwen3.8-27B-NVFP4@57926bac`（快照完整性校验后才会启动 GPU） |
| 驱动 | 实测 `595.84` / CUDA 12.9 |

## 环境要求

- Linux x86_64 + Docker + NVIDIA Container Toolkit
- 2 × SM120（compute capability 12.0）16GB GPU；驱动 ≥ 595.84（实测版本）
- 可用主机内存 ≥ 28GB；磁盘：镜像 ~36GB + 模型 ~11GB

## 已知问题（如实披露）

- eager 模式 262K 窗口在维护者验证期间出现过一次未复现的 Xid 13（图模式全量门禁未出现）；如复请附 `dmesg`。
- 冷 FlashInfer autotune 缓存下，196K/262K 首次长预填的调优扫描会超过默认 300s RPC 超时；发布 profile 已内置
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900` 解决（这是必需项，不是可选项）。
- 前缀缓存（Prefix Caching）在此 profile 中显式关闭（`--no-enable-prefix-caching`）。

## License

Apache-2.0（与上游 vLLM / FlashInfer 一致）。
