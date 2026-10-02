# profiles/ — 当前生产 profile 的公开文本记录

`production-dense-recovery-20260923/` 是当前晋升生产（2026-09-23）的运行时 profile 的
**文本资产副本**：launcher/supervisor、overlay 修复源码、service 单元、MANIFEST/PROFILES/
environment、CPU 测试与修复台账。它是对宿主机档案
`~/nvidia-dense-runtime/production-dense-recovery-20260923/` 的记录性复制，
不是可直接运行的部署包（路径、GPU 与缓存绑定仅在该宿主机成立）。

按仓库既定策略，以下内容**不随仓库分发**，只存在于宿主机：

- `runs/`、`state/`、`cache-blessed/`（多 GB 运行状态、遥测、blessed 缓存）
- `relay.mjs` / `relay-supervisor.mjs`（私有 HTTP/SSE 转发代码，其哈希见
  `production-dense-recovery-20260923/control-binding/binding.json`）
- 模型权重（21 GiB，`nvidia/Qwen3.8-27B-NVFP4`，用根目录 `download_model.sh` 获取）

`control-binding/` 是宿主机 systemd 单元实际加载的 drop-in 与启动/停止/门禁脚本副本
（`qwen27b.service.conf` 等），用于记录真实启动命令链。

回退 profile 的文本副本在 `../legacy/production-dense-grouped-20260922/`。

## CPU 测试说明

profile 内的 `test_*_cpu.py` 是按 MANIFEST 哈希钉死的**忠实副本**，不得修改。它们依赖
宿主机目录布局（launcher 会读取 profile 父目录的 `NVIDIA_MODEL_INVENTORY.json` 等状态），
在公开克隆中直接运行会有环境性失败；其中
`test_launcher_cpu.Boundaries.test_incomplete_inventory_rejected_before_hashing`
在宿主机原件上同样报错（2026-10-01 复核，既有状态，如实记录，未作修补）。
公开克隆的守卫版 CPU 测试在仓库根 `tests/`。
