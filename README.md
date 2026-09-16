# Qwen3.8-27B 双卡本地服务

日常版固定为 **P2P/CUMEM、prefill chunk 2048、Prefix Cache 开启（Mamba align）**。
开机、桌面和根入口统一读取 [selection.json](runtime/stable-prefill-sync/selection.json)。

当前已修复主机内存误拒：启动门槛22 GiB，容器上限45 GiB不视为预留；本次恢复内存峰值约7.75 GiB。
[修复与实测回执](runtime/graph006/repairs/20260910-host-memory-admission/REPORT.md)。

API：`http://127.0.0.1:18094/v1`，本地转发至18096；模型名 `unsloth/Qwen3.8-27B-NVFP4`。
固定 TP2、单请求、262144上下文、MTP K3、FULL_DECODE_ONLY Graph、NCCL Simple；
每卡 NVFP4 KV 3,039,750,144字节，CPU配额8核。align最终block size为2848，
调度器chunk上限仍为2048；109个物理块包含null block，不承诺额外空闲块。

此前修复 draft/count 副流存储生命周期、draft CPU清零与旧D2H竞争、backup pinned
H2D复用、MTP graph输入身份检查，以及worker故障传播和API断连/取消清理。
真实GPU小型DMA旧/新对照证明两类数据污染已受保护；完整验收及适用范围见
[前版报告](runtime/graph006/repairs/20260910-opencode-state-audit/REPORT.md)。OpenCode样式请求验证属于合成客户端验收，真实用户工具未被代为执行。

此前 accepted-count行重排、proposer元数据别名、Mamba请求记录清理、FlashInfer
staging、首请求/长prefill watchdog等修复继续保留。逐值索引校验、关键边界同步、
真实执行进度和错误证据保留。历史故障有SHM时期也有P2P时期，不能仅按通信方式归因；
首次非法内核/原挂死唯一根因和长期生产稳定性尚未证明。

前版性能实测来自 [benchmark/PASS.json](runtime/graph006/repairs/20260910-opencode-state-audit/benchmark/PASS.json)，
这些是上版GPU运行时的历史测量；本轮仅修复主机内存门槛，不将旧run验收冒充新实例验收。吞吐与利用率是有界测量，受到负载和MTP接受率影响；
本轮没有证明吞吐进一步提高。

| 输入 token | TTFT 秒 | decode token/s | prefill 双卡利用率 | decode 双卡利用率 |
| --- | --- | --- | --- | --- |
| 8192 | 2.789 | 82.63 | 59.67%/48.50% | 76.27%/78.55% |
| 32768 | 9.623 | 88.42 | 91.39%/88.83% | 83.45%/83.73% |

精确冻结 SHA：`a889909905ba544aa867251b3ba38220a841706d1877fb765afcaca594d8c4eb`，十五份只读覆盖逐文件校验。
当前runtime：`~/qwen38-27b-dual-gpu-vllm/runtime/graph006/repairs/20260910-host-memory-admission/candidate`（本机本地路径；runtime/ 不随仓库分发）。profile内的candidate文字保留冻结时状态，
当前恢复与内存测量见 [内存门槛修复](runtime/graph006/repairs/20260910-host-memory-admission/REPORT.md)。
历史 [上轮P2P与缓存审计](runtime/graph006/repairs/20260910-p2p-hang-audit/REPORT.md)、
[prefill审计](runtime/graph006/repairs/20260910-prefill-audit/REPORT.md)、
[索引审计](runtime/graph006/repairs/20260910-index-audit/REPORT.md)与全部失败原件保留。

## 启动、停止与日志

```bash
systemctl --user start qwen27b-relay.service qwen27b.service qwen27b-monitor.service
./stop.sh
./test_api.sh
python3 -I runtime/stable-prefill-sync/verify.py
journalctl --user -u qwen27b.service -f
```

桌面 `~/桌面/qwen 27/` 重复启动保留现有实例。停止先停止监督程序，再清理所属模型。
`docker-compose.yml`继续禁用直接启动；查看固定命令可运行 `./launch.sh --print-command`。
当前模型与监督输出进入 systemd journal。前版实例输出保留于 [candidate-launch.log](runtime/graph006/repairs/20260910-opencode-state-audit/candidate-launch.log)。
systemd active只说明恢复管理器存活；实际模型状态需结合active.json、恢复状态和真实推理结果。

有工作时引擎与两卡必须持续完成执行RPC；90秒无推进会完整停止所属实例，`/health`200不能豁免。
空闲不因无token被误杀；长prefill依据实际执行推进判断。已授权的有界恢复策略保留：
完整清理并通过身份、GPU、主机内存和内核检查后至少等30秒，30分钟最多两次，
达到限制后冷却再试；systemd `Restart=no`。主动停止不会自动拉起；归属不明、
Host OOM、存储错误、掉卡或资源未释放时停止。

启动依赖系统ACS/P2P门禁及当前boot的GPU UUID/PCI/驱动验证；每个实例检查实际NCCL双向P2P通道。
驱动610.57.04、内核7.0.0-30-generic，本轮未重启整机；新GRUB自然冷启动仍待验证，见
[开机配置审计](~/audit/qwen-p2p-boot-20260910/REPORT.md)（本机本地档案）。
`/health`、进程存在或GPU利用率不能代替真实推理与结束、清理证据。

## 掉线与任务状态

当前转发入口返回明确的503/502或流式错误；OpenCode/LoopRail连接故障不会再被当作代码失败反复迭代。已退出的任务进程显示输出不完整。详见 [链路审计](runtime/graph006/repairs/20260910-disconnect-audit/REPORT.md)。原GPU等待首因仍未唯一归因。

转发代码更新后执行`systemctl --user reload qwen27b-relay.service`；日志出现`relay_worker_ready`表示新worker已就绪，已有SSE继续由旧worker完成。正在运行的LoopRail会话保留已加载的旧模块，新启动会话使用本地修复。

任务交接审计确认：OpenCode 2 CLI超时并不会停止后台会话。已安装的LoopRail适配器先创建可追踪的独立会话，超时/异常时准确中断所属会话，并停止整轮自动重试，避免旧任务和重试任务同时执行。
