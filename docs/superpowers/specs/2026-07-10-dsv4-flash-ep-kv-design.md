# DSV4-Flash 专家并行与长上下文缓存设计

> 状态：设计与可行性基线，尚未实现。后续 Agent 应先在目标服务器复现实验基线，再按两份实施计划推进。

## 1. 目标

在以下单机硬件上为 fastllm 建立可验证的 DeepSeek-V4-Flash 推理路径：

- 8 × RTX 2080 Ti 22GB，合计 176GB 显存；
- 无 NVLink；
- 2 × Xeon E5-2680 v4；
- 64GB DDR4 ECC 2133；
- 预期是双 NUMA 节点、PCIe 3.0，实际 GPU/CPU/PCIe 拓扑必须上机确认。

设计同时解决两个相互关联但可独立实施的问题：

1. 让 256 个路由专家按 GPU 所有权执行，并以低通信量归约结果；
2. 在模型权重几乎占满显存时，让 DSV4-Flash 的 256K 至 1M 上下文缓存仍可工作。

第一阶段验收目标固定为：

- 先完成 256K 单请求的数值正确性与性能闭环；
- 再扩展到官方上限 1,048,576 token；
- 专家并行以 8 并发 decode 总吞吐为主指标，单请求延迟为约束指标；
- 不承诺单请求 8 倍加速，不尝试超出模型官方上下文上限。

## 2. 基线与来源

### 2.1 代码基线

- 仓库：`TomPython/fastllm`
- 审查提交：`d176be5ab71f65c1c47aa0cadff1a759b9f47ace`
- 主实现文件：
  - `src/models/deepseekv4.cpp`
  - `include/models/deepseekv4.h`
  - `src/devices/cuda/models/deepseekv4-kernels.cu`
  - `src/devices/multicuda/multicudadevice.cpp`
  - `src/devices/multicuda/fastllm-multicuda.cu`
  - `src/devices/cuda/linear/fastllm-linear-fp8.cu`

### 2.2 官方模型事实

官方配置与参考实现：

- 配置：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json>
- 参考推理：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py>
- 权重目录：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main>

关键参数：

| 参数 | 值 |
|---|---:|
| 总参数 | 284B |
| 每 token 激活参数 | 13B |
| 权重仓库体积 | 约 160GB |
| 主层数 | 43 |
| hidden size | 4096 |
| attention heads | 64 |
| KV heads | 1 |
| attention head dim | 512 |
| sliding window | 128 |
| 路由专家 | 256 |
| 每 token 路由专家 | 6 |
| 共享专家 | 1 |
| expert intermediate size | 2048 |
| expert dtype | FP4 E2M1 |
| 官方最大上下文 | 1,048,576 |

43 个主层对应的 `compress_ratios` 为：

- 2 个 ratio=0 的纯滑窗层；
- 21 个 ratio=4 的 CSA/indexer 层；
- 20 个 ratio=128 的 HCA 层。

配置数组最后一个 ratio=0 对应额外 MTP 层，不计入 43 个主层。

## 3. 已确认的 fastllm 现状

### 3.1 已有但未完成的专家并行原型

`MultiCudaMergeMOE::Run` 已包含环境变量控制的实验路径：

```text
FASTLLM_MULTICUDA_MOE_EXPERT_PARALLEL=1
```

当前行为：

1. 仅在多 GPU、没有内嵌共享专家时启用；
2. 用 `expert % device_count` 确定专家所有者；
3. 把输入、专家索引和分数复制到所有 GPU；
4. 每张 GPU 只计算本地拥有的已命中专家；
5. 对 `[batch, hidden]` 的局部输出执行 NCCL AllReduce。

这是一条适合无 NVLink 机器的 output-reduce EP 路径，但仍有以下缺陷：

- 专家分配只按模数轮转，不知道 PCIe/NUMA 拓扑；
- batch=1 路由过滤包含每 GPU、每层的 D2H 同步；
- batch>1 的局部权重/空权重行为缺少独立正确性回归；
- 运行时没有输出每卡权重、路由负载和 collective 延迟；
- 共享专家和稠密层可能导致 GPU0 显存不均衡；
- 当前只有 AllReduce 封装，没有适合正式 token-dispatch EP 的 send/recv 或 all-to-all 接口。

### 3.2 DSV4 共享专家已可从路由专家中拆出

`deepseekv4.cpp` 在 `--cuda_se true` 时先单独运行共享专家，再把 MergeMOE 权重数组中的共享专家位置设为 null，之后调用路由专家 MergeMOE。这使现有 EP 原型具备被激活的基本条件。

必须在真实模型上验证：

- prefill 和 decode 两条路径都不会重复计算共享专家；
- EP 启用与禁用时 logits 在容差内一致；
- 共享专家所在 GPU 不成为显存和计算热点。

### 3.3 Turing 可以执行 FP4，但不是原生 NVFP4 Tensor Core

2080 Ti 是 SM75。fastllm 当前 NVFP4 decode 小 batch 路径会在 CUDA kernel 中解码 E2M1，使用 FP32 累加；它不要求新架构原生 FP4 指令，因此功能上可运行。

影响是：

- 专家并行能够降低每卡工作量；
- 但单卡专家 GEMV 的性能上限仍受当前 FP4 解码与 FP32 FMA 实现限制；
- 若 EP 后仍由专家 kernel 主导延迟，后续应开发 SM75 专用 FP4 GEMV/grouped GEMM，而不是继续堆调度开关。

### 3.4 DSV4 使用模型专属 decode cache

DSV4 不依赖普通全量 K/V cache。`DeepSeekV4DecodeLayerCache` 维护：

- `windowKV`：最近 128 token；
- compressor 的少量 raw tail；
- `compressedKV`：按 ratio=4 或 ratio=128 产生的压缩行；
- 未来还需要官方 indexer 的压缩索引 cache。

通用 `SetKVCacheInCPU`/`--cpu_kvcache` 不能直接形成高效 DSV4 offload。当前 decode 遇到 CPU 上的 `compressedKV` 时，会先把整段缓存复制到当前 GPU，再执行 attention。

### 3.5 当前 DSV4 forward 没有接上官方 indexer

fastllm 会识别并加载 indexer 相关权重名称，但主 forward 中没有执行官方 `Indexer.forward` 的等价逻辑。当前 sparse decode 直接对全部压缩行计算 attention。

这会产生两个问题：

1. ratio=4 层在长上下文下的计算量和 HBM 读取量随所有压缩块增长；
2. CPU offload 时无法只取 top-512 行，只能搬运完整 `compressedKV`。

因此 indexer 不是额外优化，而是 256K/1M 正确且高效实现的前置条件。

## 4. 容量与通信计算

### 4.1 权重容量

官方权重目录约 160GB，总显存为 176GB。按账面平均：

```text
160GB / 8 = 20GB 权重/卡
22GB - 20GB = 约 2GB 名义余量/卡
```

这 2GB 还要容纳：

- CUDA context；
- NCCL communicator 和缓冲；
- attention/MoE 临时张量；
- KV/indexer cache；
- 内存碎片和加载峰值。

结论：完整模型可能全 GPU 常驻，但必须同时分片路由专家、稠密层、共享专家和其他大权重，并采用流式加载。64GB CPU RAM 不足以保存完整权重副本。

### 4.2 压缩 attention KV 斜率

BF16 压缩 KV 的每 token 增量：

```text
21 * (512 dims * 2 bytes / 4)
+ 20 * (512 dims * 2 bytes / 128)
= 5,536 bytes/token/sequence
```

官方 ratio=4 indexer cache 的 BF16 每 token 增量：

```text
21 * (128 dims * 2 bytes / 4)
= 1,344 bytes/token/sequence
```

总增长量约为 6,880 bytes/token/sequence：

| 上下文 | 估算缓存 |
|---:|---:|
| 128K | 约 0.88GB |
| 256K | 约 1.76GB |
| 1M | 约 6.88GB |
| 1,048,576 | 约 7.21GB |

`windowKV` 和 compressor raw tail 是几十 MB 级固定/有界开销，不改变结论。

### 4.3 冷热分层后的驻留量

推荐把 ratio=4 的完整 attention `compressedKV` 放在 CPU：

```text
CPU cold KV = 21 * 512 * 2 / 4
            = 5,376 bytes/token/sequence
```

1M 单请求约占 5.38GB CPU 内存。GPU 保留：

- ratio=4 indexer cache：约 1.34GB/1M；
- ratio=128 compressed KV：约 0.16GB/1M；
- window、raw tail 和 staging：几十 MB。

### 4.4 top-512 gather 流量

每个 ratio=4 层仅取 top-512 压缩行：

```text
512 rows * 512 dims * 2 bytes = 512KiB/layer
21 layers = 10.5MiB/generated token/sequence
```

这个流量远小于每 token 搬运完整约 5.38GB 的方案。瓶颈将由带宽转为 21 次逐层 gather/copy 的启动延迟，因此需要：

- NUMA 本地冷存储；
- 排序/coalesce 后的 CPU gather；
- 双缓冲 pinned staging；
- H2D async copy；
- 可观测的 gather、copy 和 attention 分段计时。

### 4.5 EP output-reduce 流量

每个 MoE 层、每个 token 的归约张量：

```text
4096 hidden * 2-byte BF16/FP16 = 8KiB
```

无 NVLink 下，小消息延迟比有效带宽更重要，但该流量仍明显低于 token all-to-all 的复杂度和风险。因此第一版选择 output-reduce，不选择真正 all-to-all token dispatch。

## 5. 选定架构

### 5.1 专家并行：8-way replicated-input/output-reduce

数据流：

```text
router on current/root GPU
  -> expertIndex/expertScore 常驻 GPU
  -> 将 activation + routing metadata 复制到 8 GPU
  -> 各 GPU 在设备端筛选本地专家任务
  -> local grouped expert GEMV/GEMM
  -> NCCL AllReduce([batch, 4096])
  -> shared expert result add
```

所有权策略分两步：

1. 基线使用确定性 round-robin，便于与现有实现比对；
2. 取得真实路由直方图后，使用按层的 greedy balance，在保持每卡权重字节均衡的同时降低专家负载偏斜。

拓扑策略：

- 先按 `nvidia-smi topo -m` 把 GPU 分成两个 NUMA socket group；
- 每个 worker 线程绑到对应 CPU node；
- 权重和 CPU staging 使用 first-touch/NUMA bind；
- 第一版仍使用一个 8 卡 NCCL communicator；
- 只有 profiling 证明跨 socket AllReduce 是主瓶颈后，才引入 4+4 分层归约。

### 5.2 长上下文：GPU indexer + CPU cold KV

数据流：

```text
current hidden / qr on GPU
  -> 构建并追加 indexer compressed cache
  -> GPU index scoring over index cache
  -> top-512 block ids
  -> D2H copy 512 ids（很小）
  -> CPU 从 NUMA-local ratio=4 cold KV gather 512 rows
  -> pinned double buffer H2D
  -> attention(windowKV + gathered rows)
```

ratio=128 层不使用 indexer，完整压缩 cache 继续留在 GPU，因为 1M 时所有 ratio=128 层合计只有约 160MB/sequence。

### 5.3 权重与 cache 的设备放置

每层必须有一个明确的 compute owner 或 TP group。对应层的：

- attention 权重；
- window cache；
- ratio=128 cache；
- indexer cache；
- gather staging；

应位于同一 GPU 或同一 TP group，避免每 token 跨 GPU 搬 cache。

路由专家所有权可以与 attention 层 owner 独立，因为 MoE 输入本来会复制到所有 EP GPU。

## 6. 明确不采用的方案

### 6.1 全量 CPU KV + 每层整段 H2D

容量可行，性能不可接受。1M 时每生成一个 token 会搬运约 5.38GB ratio=4 cold KV。

### 6.2 所有 attention 在 CPU 计算

双 E5-2680 v4 的内存带宽和计算吞吐不足以承担 43 层长上下文 attention，同时还要为 GPU expert pipeline 提供数据。

### 6.3 第一版 true all-to-all EP

当前 fastllm 缺少 NCCL send/recv/all-to-all 抽象；batch=1 top-6 也不会填满 8 卡。增加协议复杂度不能解决首要容量和同步问题。

### 6.4 路由专家 CPU/NUMA offload

64GB RAM 无法同时容纳完整模型的主要专家池，而且 E5 v4 上逐 token CPU expert 会成为绝对瓶颈。

## 7. 性能边界

DSV4 每 token 只选择 6 个路由专家，因此单 token 最多激活 6 张 GPU。若专家均匀随机映射到 8 卡，6 个专家期望命中的不同 GPU 数约为：

```text
8 * (1 - (7/8)^6) ~= 4.41
```

所以：

- 单请求不可能实现 8 倍 EP 加速；
- 8/16 并发才可能让所有 GPU 持续有本地专家任务；
- 第一阶段应分别报告 batch=1 延迟和 batch=8/16 总吞吐；
- 任何只报告单个 MoE kernel 而不报告端到端 decode 的结果都不算完成。

## 8. 阶段与验收门槛

### 阶段 A：服务器事实基线

必须保存：

- `nvidia-smi -L`
- `nvidia-smi topo -m`
- `lscpu -e=CPU,NODE,SOCKET,CORE`
- `numactl --hardware`
- P2P bandwidth/latency；
- NCCL AllReduce 8KiB 至 1MiB 延迟；
- 每卡空闲显存和 CUDA context 后空闲显存；
- SSD 顺序读取和 46 个 safetensors 的加载时间。

### 阶段 B：EP 正确性

- CPU/单 CUDA reference 与 8-way EP 输出对齐；
- 覆盖 batch=1、8、16，top-k=6；
- 覆盖专家 owner 碰撞、某卡零任务、所有任务落单卡；
- EP on/off 的固定 prompt logits：max abs error <= 5e-2，cosine >= 0.999；
- CUDA memcheck/compute-sanitizer 不报告越界。

### 阶段 C：Indexer 正确性

- 小尺寸 CPU reference 与 CUDA index score/top-k 完全一致或在稳定 tie-break 规则下等价；
- 官方参考实现固定输入的 top-k 集合一致；
- index cache 增长行数等于 `floor(total_tokens / 4)`；
- prefix restore 后继续 decode 与无 restore 结果一致。

### 阶段 D：混合 KV 正确性

- GPU 全驻留与 CPU cold KV 的 gathered rows 字节一致；
- 4K、128K、256K 上 logits 与全 GPU 路径满足同一容差；
- CPU 内存随 token 线性增长，斜率接近 5,376 bytes/token/sequence；
- GPU 内存不包含 ratio=4 全量 attention compressed KV；
- 256K 持续 decode 100 token 无增长性泄漏。

### 阶段 E：完整模型和 1M

- 权重加载峰值不超过 64GB 主存；
- 所有 8 卡均成功常驻模型权重；
- 256K 单请求先完成端到端生成；
- 之后验证 1,048,576 context 的 prefill/cache 构建和至少 16 个 decode token；
- 报告 batch=1、8、16 的 prefill tok/s、decode tok/s、TTFT、每卡显存、CPU 内存、PCIe H2D 和 NCCL 时间。

## 9. 风险和失败处理

### 9.1 权重放得下但 GPU0 OOM

原因通常是共享专家、稠密层、临时 workspace 或 cache 集中在 root GPU。处理顺序：

1. 输出每个 tensor 的 owner 和字节；
2. 分散稠密层/attention owner；
3. 限制 workspace 和 prefill chunk；
4. CPU offload ratio=4 cold KV；
5. 最后才考虑进一步量化非专家权重。

### 9.2 P2P 被 IOMMU/主板拓扑禁用

不得假定 `cudaDeviceCanAccessPeer` 为真。P2P 不可用时保留 NCCL host-staging 基线，并记录实际 collective 延迟。若 8KiB AllReduce 成为主要瓶颈，再评估 4+4 分层归约。

### 9.3 Indexer top-k tie 导致测试不稳定

测试输入必须避免完全相同的 score；实现明确使用 `(score desc, index asc)` 作为稳定 tie-break。

### 9.4 大块 pinned memory 伤害系统

不 pin 5GB 冷 cache。只 pin 双缓冲 staging，每层每 request 默认 512KiB，按最大并发预分配并设置全局上限。

### 9.5 当前 DSV4 实现仍有参考路径

部分 DSV4 逻辑仍包含 CPU reference fallback。每个阶段都要用 profiler 确认真正走到 CUDA fast path，不以“结果正确”替代“目标路径已执行”。

## 10. 决策记录

| 决策 | 选择 | 原因 |
|---|---|---|
| 第一版 EP 通信 | output AllReduce | 小消息，现有 NCCL 支持，适合无 NVLink |
| token all-to-all | 暂不做 | batch=1 收益不足，协议和实现风险高 |
| ratio=4 attention KV | CPU cold store | 释放约 5.38GB/1M GPU 容量 |
| indexer cache | GPU | 每 token 要扫描并打分，不能反复从 CPU 搬全量索引 |
| ratio=128 KV | GPU | 1M 全层合计约 160MB/sequence |
| window KV | GPU | 体积小、每 token 必用 |
| 上下文里程碑 | 256K -> 1M | 先建立可调试闭环，再挑战极限 |
| 主性能指标 | batch=8 decode throughput | top-6 单 token 无法填满 8 卡 |
| 第一版 expert placement | 确定性均匀分片 | 易复现和验证，再基于路由直方图优化 |

## 11. 相关实施计划

- `docs/superpowers/plans/2026-07-10-dsv4-flash-expert-parallel.md`
- `docs/superpowers/plans/2026-07-10-dsv4-flash-indexer-kv-offload.md`
- 根目录交接：`AGENT_HANDOFF_DSV4_FLASH.md`
