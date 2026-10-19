# DeepSeek V4.1 Flash 1017 部署调研

调研日期：2026-10-19。目标是让另一台设备的 Agent C 在活动103部署用户已下载的官方模型。结论：使用最新上游的 V4.1 适配和自动规划，从可控窗口建立服务基线，再验证 DSpark 和快速 prefill；不需要先转 GGUF 或重新量化。

## 已核对版本

上游 `ztxz16/fastllm` 核对提交为 `f691f13f675a98bef36ce0eafe094e8ba927f9bf`，提交时间为北京时间 2026-10-19 09:48:56，消息为“整理部署文档与验证入口”。相比上次 10 月 1 日调研基线 `2faae1d993f2df589a246dce8d2e3e5fda4f9e82`，Git 历史新增 391 个提交。

最近直接相关的提交：

| 提交 | 内容 |
| --- | --- |
| `f691f13f` | 整理部署文档与验证入口，包含面向 Agent 的部署速查 |
| `c7066a34` | 新增自动启动与可选资源限额 |
| `17de4705` | 启用 DeepSeek 自动推测解码与 1M 上下文 |
| `22774117` | 修复 DSpark Graph 校验绕过专家缓存的问题 |
| `677da73d` | DeepSeek 混推分块 prefill 与跨块流水线 |
| `b64a5de5` | 统一混推 prefill 专家缓存与动态分流 |
| `2a15a3e8` | 修复 packed FP4 内存估算 |

来源：[固定版本源码](https://github.com/ztxz16/fastllm/tree/f691f13f675a98bef36ce0eafe094e8ba927f9bf)、[版本对比](https://github.com/ztxz16/fastllm/compare/2faae1d993f2df589a246dce8d2e3e5fda4f9e82...f691f13f675a98bef36ce0eafe094e8ba927f9bf)。

查询时 PyPI 稳定包仍是 2026-09 发布的 `ftllm 0.1.8.2`，nightly 页面仍显示 2026-03 的 `0.0.0.3`；上游版本日志把 V4.1 等能力列为未发布。包名称叫 nightly 不代表内容最新，源码 wheel 也可能沿用旧版本号。应固定源码 SHA，检查实际导入路径和原生库哈希，而不是只执行 pip 升级。[稳定包](https://pypi.org/project/ftllm/)、[nightly](https://pypi.org/project/ftllm-nightly/)、[上游版本日志](https://github.com/ztxz16/fastllm/blob/f691f13f675a98bef36ce0eafe094e8ba927f9bf/docs/version.md)。这些是调研时点状态，Agent C 开始时可重新核对。

## 作者沟通摘要

以下来自用户提供的聊天截图，不是独立硬件或性能测试：

- 用户询问是否直接下载官方 1017 原版，作者确认原版即可。
- 作者表示自己已经试过，并说明 auto 配置仍是实验性功能，后续会调整。
- 作者确认自己使用 DDR5，CPU 属于 AMD 9005 系列；具体型号、socket 数和内存布局不完整。
- 用户提到 7532、不支持 AVX-512、DDR4-2666×16，以及另一台双 8260 服务器；容量、socket 数和实际通道需现场核实。
- 用户说速度差约三倍，作者认为不应仅有如此大的差距。这是待诊断现象，不能证明某个硬件瓶颈。
- 作者说已编写给 Agent 的部署教程，并确认在 FastLLM 最新更新中。

截图小表格无法可靠辨认完整条件与数值，本交接不引用其速度。原图位于用户设备的临时文件中，本仓库不依赖该原图。不得把“×16”直接解释为 16GB 或 16 个有效内存通道，也不得继续沿用早先双 E5、64GB 的配置。

## 官方模型与格式

目标是 `deepseek-ai/DeepSeek-V4.1-Flash-1017`，架构为 `deepseek_v41`。官方说明这是 V4.1 Flash 更新版，包含编码、工具调用、思考和图像协议。模型没有普通 Jinja chat template，提供专用编码实现。部署应使用 FastLLM 的 V4.1 适配，不套用其他模型模板。[官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)。

模型已下载是用户确认的状态；部署 Agent 仍需检查本地路径、config、tokenizer、索引、权重分片、Engram 和 `mtp.*`，但不应重新下载或无必要地扫描全部权重正文。上游 Agent 教程明确将官方 1017 列为推荐模型，直接使用本地目录。[部署速查](https://github.com/ztxz16/fastllm/blob/f691f13f675a98bef36ce0eafe094e8ba927f9bf/docs/model-deployment-agent.md)。

## 自动规划的行为和限制

在已核对版本，Linux NVIDIA CUDA 的 V4.1 支持 `--auto`。它按可见 GPU、内存、模型元数据和拓扑选择主干、专家放置、缓存及 chunk，不会反复加载模型进行实测选优。

- V4.1 默认规划不超过原生窗口的 1,048,576 token，单请求，DSpark 7、默认置信度阈值 0.5、fast_prefill。
- 容量不足时调整放置、缓存和 chunk，仍不可行则报错，不自动缩短指定窗口。
- 显式 `--max_context_length`、`--dspark 0`、`--no_fast_prefill` 等覆盖默认值。
- `--auto_gpu_memory_limit` 是每卡规划上限；`--auto_host_memory_limit` 是主机规划上限，均不是运行时强制内存限额。
- 自动 TP 搜索主要限双卡相近规格，不代表八卡布局已经实测最优。
- `--ngram_device disk` 只控制 Engram；`--moe_device disk` 才控制磁盘专家。
- `--moe_cpu_cache` 不是整个进程 RSS 上限；CUDA 专家缓存也不是 KV 预算。
- Qwen4 的自动精确 KV streaming 不能移植理解成 V4.1 已有同样功能。

来源：[自动规划说明](https://github.com/ztxz16/fastllm/blob/f691f13f675a98bef36ce0eafe094e8ba927f9bf/docs/hybrid-auto.md)。

## 正确性与资源边界

`fast_prefill` 是近似计算：部分后段层仅重放窗口尾部，输出差异不局限于浮点舍入。关闭它可建立非近似 prefill 对照，但普通数学模式仍不等于官方参考逐位对齐。参考模式用于专项诊断，不能不加说明地拿来衡量正常部署速度。[V4.1 指南](https://github.com/ztxz16/fastllm/blob/f691f13f675a98bef36ce0eafe094e8ba927f9bf/docs/deepseek_v41.md)、[精度说明](https://github.com/ztxz16/fastllm/blob/f691f13f675a98bef36ce0eafe094e8ba927f9bf/docs/deepseek_v41_precision.md)。

DSpark 使用内置 `mtp.*`。草稿验证、回滚、缓存、算子路径都影响实际收益；随机采样相同 seed 不保证开启与关闭后逐 token 相同。当前文档对图文和批量 decode 与 DSpark 的组合仍有限制，先验收单请求文本。

历史 V4.1 指南描述约 200GB Engram 和数百 GB 专家权重，不能将此前小内存机器视为可直接全常驻。1017 实际文件、加载转换、DSpark、KV 和工作区必须重新预算。磁盘专家可缓解容量，但会付出 I/O 成本。[V4.1 指南](https://github.com/ztxz16/fastllm/blob/f691f13f675a98bef36ce0eafe094e8ba927f9bf/docs/deepseek_v41.md)。

仓库 V4.1 benchmark 当前包含 2026-10-02 的 Q2_K/Q4_K 测试，不是官方 1017 原版在用户服务器的性能承诺。不能混用其速度、权重格式和内存数字作为本次验收线。[Benchmark](https://github.com/ztxz16/fastllm/blob/f691f13f675a98bef36ce0eafe094e8ba927f9bf/docs/benchmarks/deepseek_v41.md)。

## 建议执行顺序

1. 核实目标服务器与已下载模型，不沿用旧设备假设。
2. 独立环境构建固定上游版本，验证实际库，保留原服务。
3. 先采用 32K 单请求、`--no_fast_prefill --dspark 0` 建立可控基线；这是本次建议覆盖，不是作者默认配置。
4. 验证 ready、模型列表、文本、流式、多轮、取消及无副作用工具调用。
5. 依次打开 DSpark 和 fast_prefill，固定其余条件比较质量与 TTFT、decode、内存、磁盘读取。
6. 再逐级测试长上下文、设备布局和线程，不将规划通过视作长上下文已验收。
7. 保留日志和复现脚本，汇总给用户决定是否反馈作者。

完整执行规则见[部署提示词](agent-c-prompt.md)。构建应遵循选定版本的 [wheel 构建说明](https://github.com/ztxz16/fastllm/blob/f691f13f675a98bef36ce0eafe094e8ba927f9bf/docs/wheel_build.md)，避免新 Python 包加载旧原生库。
