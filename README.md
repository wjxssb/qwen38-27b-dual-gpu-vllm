# Qwen3.8-27B 双卡本地服务

> **仓库定位**：本仓库是本地 dense 栈的**参考记录**，不是独立部署包。
> 当前生产入口以本机 systemd 单元与 `~/nvidia-dense-runtime/inference-control` 为准
> （桌面 `~/桌面/qwen 27/` 内有说明，副本见 `docs/LOCAL-ENTRY.zh.txt`）。
> `runtime/` 下的大体积状态（runs、core dump、遥测、审计原件）仅存在于本机，
> 不随仓库分发；文中 `runtime/...` 与 `~/...` 链接指向本机档案，在 GitHub 上会 404。
> 直接启动脚本 `launch.sh`/`stop.sh` 已退役且不在仓库中；`test_api.sh` 及
> `runtime/...` 相关命令仅限持有 runtime/ 状态的宿主机运行。

## 当前记录（2026-09-23 晋升）

当前生产 profile：**`production-dense-recovery-20260923`**
（文本资产见 [`profiles/production-dense-recovery-20260923/`](profiles/production-dense-recovery-20260923/)，
晋升与验收记录见 [`validation/production-dense-recovery-20260923/`](validation/production-dense-recovery-20260923/)，
机读摘要见 [ACTIVE_PROFILE.json](ACTIVE_PROFILE.json)）。

- 模型：`nvidia/Qwen3.8-27B-NVFP4`，revision `dbb8f445b3145f8a4c18ddc769f032d57d32867c`
  （统一 API 模型 ID `qwen38-27b-dense`；`unsloth/Qwen3.8-27B-NVFP4` 为 served 别名）
- 固定 **P2P/CUMEM、prefill chunk 4096、Prefix Cache 开启（Mamba align，实际 block 2848）、
  native MTP K3、FULL_DECODE_ONLY Graph（capture 4）、NCCL Simple、TP2、单请求、262144 上下文**；
  每卡 NVFP4 KV 3,039,750,144 字节
- API：Gateway `http://127.0.0.1:18080/v1`；18081、18094 兼容转发至 18080；
  18082 为 HTTP/SSE relay → 18096；18096 为唯一推理后端
- 回退（不可变）：`production-dense-grouped-20260922`
  （文本副本在 [`legacy/production-dense-grouped-20260922/`](legacy/production-dense-grouped-20260922/)）
- 本轮新增：CPU renderer warmup 与引擎 HELLO 重叠；密封模型文件 stat 身份 + 全 SHA 回退校验
- 评审结论：`PASS_WITH_DISCLOSED_LIMITATIONS`；长期稳定性与首次非法内核唯一根因仍未证明
- 宿主机私有件不公开：`runs/`、`state/`、`cache-blessed/`、`relay.mjs`/`relay-supervisor.mjs`、权重

## 启动、停止与日志（当前）

```bash
# 仅限宿主机；公开克隆不包含 systemd 单元与 runtime/ 状态
~/nvidia-dense-runtime/inference-control start|stop|restart|status|logs
journalctl --user -u qwen27b.service -f
```

桌面 `~/桌面/qwen 27/` 重复启动保留现有实例。停止先停止监督程序，再清理所属模型。
`docker-compose.yml` 继续禁用直接启动，仅作为固定参考模板。
systemd active 只说明恢复管理器存活；实际模型状态需结合 active.json、恢复状态和真实推理结果。

有工作时引擎与两卡必须持续完成执行 RPC；90 秒无推进会完整停止所属实例，`/health` 200 不能豁免。
空闲不因无 token 被误杀；长 prefill 依据实际执行推进判断。已授权的有界恢复策略保留：
完整清理并通过身份、GPU、主机内存和内核检查后至少等 30 秒，30 分钟最多两次，
达到限制后冷却再试；systemd `Restart=no`。主动停止不会自动拉起；归属不明、
Host OOM、存储错误、掉卡或资源未释放时停止。

启动依赖系统 ACS/P2P 门禁及当前 boot 的 GPU UUID/PCI/驱动验证；每个实例检查实际 NCCL 双向 P2P 通道。
`/health`、进程存在或 GPU 利用率不能代替真实推理与结束、清理证据。

## 掉线与任务状态

当前转发入口返回明确的 503/502 或流式错误；OpenCode/LoopRail 连接故障不会再被当作代码失败反复迭代。
已退出的任务进程显示输出不完整。详见宿主机档案
`runtime/graph006/repairs/20260910-disconnect-audit/REPORT.md`（不随仓库分发）。
原 GPU 等待首因仍未唯一归因。

转发代码更新后执行 `systemctl --user reload qwen27b-relay.service`；日志出现 `relay_worker_ready`
表示新 worker 已就绪，已有 SSE 继续由旧 worker 完成。

---

## 历史记录：graph006 时代（2026-09-10，chunk 2048）

> 以下内容记录 2026-09-10 主机内存修复后的 P2P/CUMEM 日常版状态，
> 已被 2026-09-23 晋升取代；完整机读记录见
> [legacy/ACTIVE_PROFILE.graph006-20260910.json](legacy/ACTIVE_PROFILE.graph006-20260910.json)。

日常版当时固定为 **P2P/CUMEM、prefill chunk 2048、Prefix Cache 开启（Mamba align）**。
开机、桌面和根入口统一读取 `runtime/stable-prefill-sync/selection.json`（本机档案）。

当时已修复主机内存误拒：启动门槛 22 GiB，容器上限 45 GiB 不视为预留；该次恢复内存峰值约 7.75 GiB。
修复与实测回执见本机档案 `runtime/graph006/repairs/20260910-host-memory-admission/REPORT.md`。

当时 API：`http://127.0.0.1:18094/v1`，本地转发至 18096；模型名 `unsloth/Qwen3.8-27B-NVFP4`。
固定 TP2、单请求、262144 上下文、MTP K3、FULL_DECODE_ONLY Graph、NCCL Simple；
每卡 NVFP4 KV 3,039,750,144 字节，CPU 配额 8 核。align 最终 block size 为 2848，
调度器 chunk 上限当时为 2048；109 个物理块包含 null block，不承诺额外空闲块。

此前修复 draft/count 副流存储生命周期、draft CPU 清零与旧 D2H 竞争、backup pinned
H2D 复用、MTP graph 输入身份检查，以及 worker 故障传播和 API 断连/取消清理。
真实 GPU 小型 DMA 旧/新对照证明两类数据污染已受保护；完整验收及适用范围见
`runtime/graph006/repairs/20260910-opencode-state-audit/REPORT.md`（本机档案）。
OpenCode 样式请求验证属于合成客户端验收，真实用户工具未被代为执行。

此前 accepted-count 行重排、proposer 元数据别名、Mamba 请求记录清理、FlashInfer
staging、首请求/长 prefill watchdog 等修复继续保留。逐值索引校验、关键边界同步、
真实执行进度和错误证据保留。历史故障有 SHM 时期也有 P2P 时期，不能仅按通信方式归因；
首次非法内核/原挂死唯一根因和长期生产稳定性尚未证明。

前版性能实测来自 `runtime/graph006/repairs/20260910-opencode-state-audit/benchmark/PASS.json`
（本机档案），这些是上版 GPU 运行时的历史测量；该轮仅修复主机内存门槛，
不将旧 run 验收冒充新实例验收。吞吐与利用率是有界测量，受到负载和 MTP 接受率影响；
该轮没有证明吞吐进一步提高。

| 输入 token | TTFT 秒 | decode token/s | prefill 双卡利用率 | decode 双卡利用率 |
| --- | --- | --- | --- | --- |
| 8192 | 2.789 | 82.63 | 59.67%/48.50% | 76.27%/78.55% |
| 32768 | 9.623 | 88.42 | 91.39%/88.83% | 83.45%/83.73% |

精确冻结 SHA：`a889909905ba544aa867251b3ba38220a841706d1877fb765afcaca594d8c4eb`，十五份只读覆盖逐文件校验。
当时 runtime：`~/qwen38-27b-dual-gpu-vllm/runtime/graph006/repairs/20260910-host-memory-admission/candidate`
（本机本地路径；runtime/ 不随仓库分发）。
历史 P2P 与缓存审计、prefill 审计、索引审计与全部失败原件保留于本机档案。

当时的启动命令（已取代）：

```bash
systemctl --user start qwen27b-relay.service qwen27b.service qwen27b-monitor.service
~/nvidia-dense-runtime/inference-control stop
./test_api.sh
python3 -I runtime/stable-prefill-sync/verify.py
```

驱动 610.57.04、内核 7.0.0-30-generic，该轮未重启整机；新 GRUB 自然冷启动当时仍待验证
（见本机档案 `~/audit/qwen-p2p-boot-20260910/REPORT.md`）。
