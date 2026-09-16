# 运行时镜像构建源（runtime-build/）

`ghcr.io/wjxssb/qwen38-27b-vllm:sm120-nvfp4-k3` 的完整构建定义。仓库只保存**构建定义与钉死清单**，
上游源码 checkout 由提交哈希重建（保持本仓库轻量）：

```bash
git clone https://github.com/vllm-project/vllm.git      && git -C vllm      checkout 99a10304dce8945119bd0b1a072297803c52a749
git clone https://github.com/flashinfer-ai/flashinfer.git && git -C flashinfer checkout 9dc1b2495b40314dec8a8cde8cd7faf5c5206702
```

目录布局（与构建上下文一致）：

```
runtime-build/
├── build/Dockerfile          # 两阶段构建：native-builder（SM120a 原生内核）+ runtime（overlay 装载）
├── build/gen_inputs.py       # 生成 build/inputs.json（26 个输入的字节级 SHA-256 钉死清单）
├── build/inputs.json         # 已生成的输入清单（manifest sha256=32dd7daa…，构建参数必须匹配）
├── build/verify_inputs.py    # 构建内输入校验（含 FlashInfer 关键源文件与安装版逐字节比对）
├── build/verify_overlay.py   # 镜像内 13 个安装产物的最终哈希复核
├── build/release.json        # 烘焙进镜像的运行时契约（runtime_abi=qwen38-sm120-v1）
├── overlay-graph006/         # Graph006 验证过的 12 个运行时文件 + PROVENANCE.json（血缘与来源）
├── stages/                   # 分阶段 GPU 验证：run_stage 规格、内核故障基线、质量门禁封装
└── image-contract.example.json
```

构建命令：

```bash
# 1) 组装构建上下文（Dockerfile 以该目录为 context 根，相对路径 checkouts/、tests/、build/、overlay-graph006/）
mkdir -p context/checkouts
git -C context/checkouts clone https://github.com/vllm-project/vllm.git
git -C context/checkouts/vllm checkout 99a10304dce8945119bd0b1a072297803c52a749
git -C context/checkouts clone https://github.com/flashinfer-ai/flashinfer.git
git -C context/checkouts/flashinfer checkout 9dc1b2495b40314dec8a8cde8cd7faf5c5206702
cp -r runtime-build/{build,overlay-graph006,tests} context/

# 2) 构建（inputs.json 与 Dockerfile 已在本目录）
cd context
docker build --progress=plain -f build/Dockerfile \
  -t ghcr.io/wjxssb/qwen38-27b-vllm:sm120-nvfp4-k3 \
  --build-arg INPUT_MANIFEST_SHA256=32dd7daa90005ac0de13c4fd25bf2c460a52b59955c13b274c26bbee69e0d5c5 .
```

> 注意：`checkouts` 内由上游提交重建的 rebase 差集（6 个文件：`CMakeLists.txt`、
> `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`、`vllm/v1/attention/backends/flashinfer.py`、
> `vllm/utils/torch_utils.py` 与两个测试文件）的期望哈希全部记录在 `build/inputs.json`；
> 上游 checkout 本身不带这些修改，需要按 `git log`/PR 或维护者发布的源码包应用后，哈希一致才能通过
> `verify_inputs.py`。完整可构建源码包由维护者按需发布（issue 索取）。

约束与校验链：

- `overlay-graph006/` 的 12 个 Python 文件与 GPU 验证过的 Graph006 血缘（candidate-006 运行时 overlay +
  stability-core 镜像层）逐字节一致，任何漂移都会在 `verify_inputs.py` / `verify_overlay.py` 处构建失败。
- 原生内核源（`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`、`CMakeLists.txt` 等 rebase）由
  `checkouts/vllm` 工作区提供，其哈希同样钉死在 `inputs.json`；checkout 若与哈希不符即拒绝构建。
- 需要本机 Docker BuildKit 与网络（构建中会按提交哈希拉取 CUTLASS `da5e086d`）。
