# Qwen3.8-27B Dual-GPU vLLM Production Stack

> 🚀 **在双卡 16GB 消费级显卡（2 × RTX 5070 Ti / 4090）上满血运行 262,144（262K）原生超长上下文的 Qwen3.8-27B-NVFP4 生产级推理方案**。  
> 包含 NVFP4 KV 显存压缩、MTP 多 Token 投机预测、CUDA Graph 静态图、分块预填（Chunked Prefill 4096）与前缀缓存（Prefix Caching）。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-Qwen3.8--27B--NVFP4-orange.svg)](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
[![Revision](https://img.shields.io/badge/Revision-57926ba-brightgreen.svg)](#pinned-versions)
[![Context](https://img.shields.io/badge/Native%20Context-262K%20Tokens-green.svg)](#)
[![vLLM](https://img.shields.io/badge/Engine-vLLM%20%2B%20FlashInfer-purple.svg)](https://github.com/vllm-project/vllm)
[![Hardware](https://img.shields.io/badge/Hardware-2%C3%97%20RTX%205070%20Ti%20(16GB)-red.svg)](#)

---

## 📌 核心性能一览 (Verified Benchmark)

所有数据均为双卡 RTX 5070 Ti 16GB（TP=2，PCIe 走系统内存中继）上的 **100% 真实实测数据**（无任何合成外推，详见 [BENCHMARK_DOSSIER.md](BENCHMARK_DOSSIER.md)）：

| 性能维度 | 实测指标 | 对比与说明 |
| :--- | :--- | :--- |
| **最大原生上下文** | **260,927 Tokens**（满跑 262,144 上下文） | 物理显存安全占用率 96.07%，余留 640MB 安全边界 |
| **极限满血生成速度** | **60.06 ~ 64.38 tok/s**（中位数 **61.46 tok/s**） | 较未经优化的 13.03 tok/s **提速 4.72 倍** |
| **日常中短生成速度** | **92 ~ 120 tok/s**（单字仅需 8.3 ~ 10.6 ms） | 61K 文本跑 119 tok/s，236K 文本跑 94 tok/s |
| **首字延迟 (前缀命中)** | **5.72 秒**（25.6 万 Token 命中缓存） | 较冷启动的 182 秒 **提速 31.85 倍**，前缀复用率达 44,835 tok/s |
| **冷启动全量预填** | **182.34 秒**（吞吐 1,431 tok/s） | Chunked Prefill 4096 优化，TTFT 减少 12.5 秒 |
| **工业级稳定性** | **3/3 Cold + 2/2 Warm 100% 通过** | 0 Xid 掉卡、0 OOM 崩溃、0 进程死锁 |

---

## 🔒 固定版本与环境校验清单 (Pinned Versions & Reproducibility)

为了保证 100% 可复现与生产一致性，本项目**锁定了以下关键组件与版本**：

| 组件 / 软件 | 生产验证锁定的具体版本 / Hash | 作用与技术规格 |
| :--- | :--- | :--- |
| **vLLM 框架版本** | `0.26.1rc1.dev608+g99a10304d` (commit `99a10304d`) | 原生支持 SM120 Blackwell 与 NVFP4 KV 缓存管理 |
| **目标模型仓库** | `unsloth/Qwen3.8-27B-NVFP4` | 4-bit 量化密集模型，全量权重仅 ~11GB |
| **模型 Git Revision** | `57926baca9a82b4d6906b43f2750d55315f5b10f` | **精确锁定模型权重版本**，确保与评测完全一致 |
| **生产镜像 Digest** | `sha256:4474e909cfb2b61ab78fa888ccc516032e79975416d5c4bb71e3c62eda2d5cce` | 经过数万次稳定性压力测试的固化生产镜像 |
| **基座镜像 Digest** | `sha256:96a70bbb56acb6c4d22b3153b090ca322da927361c48a3129a1a258f4c702e73` | `docker.io/vllm/vllm-openai` 官方上游基础快照 |
| **Attention / GEMM 算子** | FlashInfer `0.6.16+cu129` | SM120 NVFP4 Cutlass Linear + FlashAttention-2 核心后端 |
| **深度学习运行时** | PyTorch `2.7.0.dev20250217+cu128` / Python `3.12` | 容器内核心推理运行时环境 |
| **NVIDIA 驱动与 CUDA** | Driver `595.84` / CUDA `12.8` (兼容 12.9) | 官方 Blackwell 架构架构推荐基准驱动 |

---

## ⚡ 极速开始 (Quick Start)

> **注意**：本仓库遵循极致轻量化原则（代码与配置仅约 60KB），**绝不包含数十 GB 的大模型权重文件**。
> 执行启动脚本时，引擎会自动检测本地缓存；若本地未缓存，会自动从 Hugging Face 官方高速拉取对应 commit 的模型并补齐！

### 1. 克隆本仓库
```bash
git clone https://github.com/YOUR_USERNAME/qwen38-27b-dual-gpu-vllm.git
cd qwen38-27b-dual-gpu-vllm
```

### 2. 一键启动服务（自动补齐模型并启动）
```bash
./launch.sh
```
* 自动检测本地显卡数量与拓扑；
* 自动检测模型缓存；若缺失，自动连接 Hugging Face 并精准拉取固化版本 `57926ba`；
* 在后台拉起服务，映射到本地 `127.0.0.1:18094`。

> **提示**：如果你想在启动前先看到下载进度条把 11GB 权重拉下来，也可以先运行：
> ```bash
> ./download_model.sh
> ```

### 3. 校验健康状态与测速
```bash
./test_api.sh
```
或直接通过 curl 探测：
```bash
curl http://127.0.0.1:18094/v1/models
```

### 4. 停止与清理
```bash
./stop.sh
```

---

## 🛠️ 硬件与系统需求 (Prerequisites)

- **显卡配置**：2 × NVIDIA 显卡（每张卡显存 ≥ 16GB，支持 Blackwell SM120、Ada Lovelace 或 Hopper / Ampere 架构）。
- **系统环境**：Linux（推荐 Ubuntu 22.04 / 24.04 LTS），内核 6.x / 7.x。
- **软件依赖**：
  - NVIDIA 驱动（≥ 550 / 595 系列）
  - Docker + NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- **特别说明（消费级主板友好）**：
  - 本配置专为**无 NVLink、无企业级 PCIe P2P 交换芯片的消费级主板**优化；
  - 默认启用 `NCCL_P2P_DISABLE=1` 与 `NCCL_SHM_DISABLE=0`（走主机 17.2 GB/s DDR5 内存共享环）；
  - 强制开启 `--disable-custom-all-reduce`，彻底杜绝消费级芯片组上的总线锁死问题。

---

## 🧠 五大核心调优突破 (Why It's So Fast)

1. **SM120 原生 NVFP4 KV Cache 压缩**：
   - 262K 上下文的全部 KV Cache 在 4-bit 量化下被极度压缩，单卡仅占约 **2.40 GB**，突破了 16GB 显存容纳 27B 模型的物理不可能。
2. **多 Token 投机预测 (MTP K=3)**：
   - 采用多头草稿机制，步进耗时仅 ~42.7 ms，每步通过验证接受 ~2.586 个 Token，使真实单 Token 耗时降至 ~16.2 ms。
3. **CUDA Graph 全图捕获 (`FULL_DECODE_ONLY`)**：
   - 将 130 次跨卡 All-Reduce 与 GEMM 计算全部静态录制入 CUDA 图，彻底抹除 Python/CPU 调度延迟，使解码速度从 32 tok/s 翻倍跃升至 **61.46 tok/s**。
4. **L1 Pool 显存对齐拓扑 (+128MB/rank)**：
   - 解决了 Mamba GDN 与 QSA 混合状态机因双缓冲对齐在 24.7 万 Token 处的隐形死锁问题，将可用容量完全释放至 **274,280 Tokens**。
5. **分块预填与前缀缓存 (Chunk 4096 + Prefix Caching)**：
   - 多轮对话中，已输入的数十万字前缀直接复用跳过计算，首字延迟从 182 秒直接降至 **5.72 秒**。

---

## 🔌 客户端与 IDE 无缝集成

服务完全兼容 OpenAI API 规范，Base URL 统一配置为：`http://127.0.0.1:18094/v1`，模型名为 `unsloth/Qwen3.8-27B-NVFP4`。

### 1. OpenCode 配置 (`opencode.jsonc`)
```jsonc
"models": {
  "qwen-local": {
    "providerID": "local-vllm",
    "modelID": "unsloth/Qwen3.8-27B-NVFP4",
    "endpoint": "http://127.0.0.1:18094/v1"
  }
}
```

### 2. Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18094/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="unsloth/Qwen3.8-27B-NVFP4",
    messages=[{"role": "user", "content": "你好，请写一个高性能快排算法。"}],
    temperature=0.6,
)
print(response.choices[0].message.content)
```

---

## 📄 开源许可 (License)
本项目基于 [Apache 2.0 License](LICENSE) 协议开源。
