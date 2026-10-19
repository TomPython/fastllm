# DSV4-Flash EP / KV Handoff

## 给接手 Agent 的消息

你接手的是一个尚未开始编码、但已完成代码审查、容量推导和架构拆分的 DSV4-Flash 推理项目。不要把任务简化成“打开现有 EP 环境变量”或“打开 CPU KV 开关”：这两个开关目前都不足以完成目标。

先阅读本文件和下面三份文档，再接触代码：

1. `docs/superpowers/specs/2026-07-10-dsv4-flash-ep-kv-design.md`
2. `docs/superpowers/plans/2026-07-10-dsv4-flash-expert-parallel.md`
3. `docs/superpowers/plans/2026-07-10-dsv4-flash-indexer-kv-offload.md`

分支：`agent/dsv4-flash-ep-handoff`

审查基线：`d176be5ab71f65c1c47aa0cadff1a759b9f47ace`

拉取后用下面的命令确认你看到的是交接提交，而不是只看到基线：

```bash
git fetch origin
git switch agent/dsv4-flash-ep-handoff
git log -3 --oneline --decorate
git status --short
```

## 用户目标

在一台无 NVLink 的服务器上开发 DeepSeek-V4-Flash 专家并行，并尽量提高可用上下文：

- 8 × RTX 2080 Ti 22GB；
- 2 × Xeon E5-2680 v4；
- 64GB DDR4 ECC 2133；
- 用户可提供 SSH 进行实验；
- 用户不接受只做小型重构或表面优化，希望形成有实质性能价值的后端能力。

## 当前硬结论

1. 开发和验证专家并行肯定可行。
2. 官方权重目录约 160GB，总显存 176GB，完整模型短上下文大概率可装，但必须均衡全部大权重并流式加载。
3. 64GB 主存不能保存完整模型权重副本，也不适合 CPU offload 路由专家。
4. 无 NVLink 不是 output-reduce EP 的致命问题：每 MoE 层、每 token 的 BF16/FP16 输出归约只有约 8KiB。
5. 单 token 只激活 6 个路由专家；随机均匀放到 8 卡时平均只命中约 4.41 张卡。因此主指标应是 8 并发 decode 总吞吐，不应承诺单请求 8 倍。
6. RTX 2080 Ti 没有新架构原生 NVFP4/FP8 Tensor Core。fastllm 的小 batch NVFP4 路径会软件解码 E2M1 并 FP32 累加，功能可用但可能成为后续性能瓶颈。
7. DSV4 的 cache 是 128-token window + ratio=4/128 compressed KV，不是普通全量 K/V。
8. 官方 43 个主层内有 21 个 ratio=4、20 个 ratio=128、2 个 ratio=0 层。
9. attention compressed KV 增长约 5,536 bytes/token/sequence；官方 indexer cache 另约 1,344 bytes/token/sequence，总计约 6,880 bytes/token/sequence。
10. 1M 单请求的缓存约 6.9GB，容量可行；但模型权重已接近显存极限，所以推荐 CPU cold ratio=4 attention KV + GPU indexer/window/ratio128 cache。
11. 推荐混合模式每 token 只回传 top-512：21 个 ratio=4 层合计约 10.5MiB/generated token，而不是搬运完整约 5.38GB cold KV。

## 代码中的关键发现

### 隐藏的专家并行路径

文件：`src/devices/multicuda/multicudadevice.cpp`

搜索：

```bash
rg -n "FASTLLM_MULTICUDA_MOE_EXPERT_PARALLEL|MultiCudaDoMergeMOEOp|FastllmNcclAllReduce" src/devices/multicuda
```

当前路径：

- 用 `expert % devices.size()` 分配专家；
- 复制 input/index/score 到各卡；
- batch=1 路由筛选包含 D2H；
- 本地输出后 NCCL AllReduce；
- 没有正式的 all-to-all/send/recv abstraction。

第一版应扩展这条 output-reduce 路径，不应从零做 token dispatch。

### DSV4 共享专家已拆开

文件：`src/models/deepseekv4.cpp`

prefill/decode 都可以先运行共享专家，然后把 MergeMOE 中共享专家权重位置设为 null。实验时使用 `--cuda_se true`，并确认共享专家没有被重复计算。

### 当前 FP4 路径可在 SM75 运行

文件：`src/devices/cuda/linear/fastllm-linear-fp8.cu`

搜索：

```bash
rg -n "FastllmGemvNVFP4Kernel1MultiRow|FastllmCudaNVFP4UseGemv" src/devices/cuda/linear/fastllm-linear-fp8.cu
```

decode 小 batch 使用自定义 GEMV 解码 FP4，不依赖原生 NVFP4 指令。

### DSV4 自定义 cache 绕过普通 CPU KV 语义

文件：`include/models/deepseekv4.h`、`src/models/deepseekv4.cpp`

`DeepSeekV4DecodeLayerCache` 保存 `windowKV`、compressor raw state 和 `compressedKV`。当前 `SparseAttentionDecodeCachedReference` 遇到 CPU `compressedKV` 会创建 CUDA 临时副本并复制整段缓存。

因此不能把通用 `--cpu_kvcache` 当作完成的 DSV4 offload。

### 官方 indexer 未接入 forward

fastllm 会加载：

```text
layers.*.attn.indexer.wq_b.weight
layers.*.attn.indexer.weights_proj.weight
layers.*.attn.indexer.compressor.wkv.weight
layers.*.attn.indexer.compressor.wgate.weight
```

但当前 forward 没有执行官方 `Indexer.forward` 等价逻辑。它直接让 sparse attention 遍历所有压缩行。实现长上下文和 top-512 cold gather 前，必须先实现 index cache、score 和稳定 top-k。

官方参考：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py>

## 第一工作会话应该做什么

不要立即修改 MergeMOE 或 DSV4 forward。第一会话只完成服务器事实基线：

1. 从用户取得 SSH host/user/key 的安全访问方式，不把口令或私钥写入仓库；
2. 在服务器克隆本分支；
3. 执行专家并行计划 Task 1 的 `tools/dsv4_ep_probe.sh`；
4. 保存 `nvidia-smi topo -m`、P2P 和 NCCL 8KiB 延迟；
5. 确认 8 张卡实际可用显存、PCIe link width 和 NUMA 归属；
6. 用 `CUDA_ARCH=75` 构建当前基线；
7. 仅用小尺寸 MergeMOE 测试现有环境变量路径是否可运行；
8. 把真实结果更新回总设计文档。

## 实施顺序

推荐顺序：

1. EP Task 1-4：先得到无 D2H 的正确 output-reduce EP；
2. KV Task 1-4：容量模型、CPU reference、CUDA indexer、index cache；
3. KV Task 5-7：cold store、top-512 gather、混合模式；
4. EP Task 5-6：观测、完整 DSV4 共享专家和权重放置；
5. KV Task 8：4K parity -> 128K -> 256K -> 1M；
6. 若 profiling 表明 NVFP4 expert GEMV 主导，再单独设计 SM75 kernel 计划。

不要同时在一个提交里实现 EP、indexer 和 offload。每个任务按计划中的中文 commit 边界提交。

## 远程实验必须收集的证据

- 精确 commit SHA；
- 完整启动命令和环境变量；
- CUDA/driver/NCCL 版本；
- `nvidia-smi topo -m`；
- P2P 可用矩阵；
- 8KiB/16KiB/1MiB NCCL AllReduce 延迟；
- 每卡权重字节和峰值显存；
- 进程 RSS、NUMA 分配；
- batch=1/8/16 的 TTFT、总 tok/s、p50/p95 inter-token latency；
- 每层 local expert、AllReduce、index、CPU gather、H2D、attention 时间；
- 固定 prompt 的 token/logit parity；
- compute-sanitizer 结果。

任何“提升明显”的结论都必须附这些证据中的相关子集。

## 已知风险

- 仓库当前本地 checkout 最初是 grafted/shallow 状态；需要 `git fetch` 后确认历史和目标分支可 rebase。
- 模型权重可放下不等于运行时可放下；GPU0/根卡可能因共享专家、稠密层或 workspace OOM。
- P2P 可能因主板、PLX、ACS/IOMMU 配置不可用。
- 64GB RAM 在下载/加载 160GB safetensors 时必须依赖 mmap/流式加载和受控加载线程。
- 当前 DSV4 文件仍带有 reference fallback；结果正确不证明走了目标 CUDA fast path。
- 大块 `cudaHostRegister` 会伤害系统；不要 pin 全量 cold cache。
- 1M 的 indexer 本身要对约 250K index blocks 打分，可能成为主要计算瓶颈；必须单独测 score 与 top-k 时间。

## 完成定义

交接后的实现只有同时满足下列条件才可称为阶段完成：

- 合成回归与 sanitizer 通过；
- 真实 8 卡服务器路径执行；
- EP on/off 或 GPU/hybrid 的数值 parity 达标；
- 256K 端到端生成通过；
- 性能报告含完整命令、commit 和硬件拓扑；
- 未把用户凭据、模型 token、SSH key 或服务器地址秘密提交到仓库。
