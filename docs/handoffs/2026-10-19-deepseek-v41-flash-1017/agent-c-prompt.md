# Agent C 部署 DeepSeek V4.1 Flash 1017 提示词

以下为可直接交给另一台设备上的 Agent C 的完整任务。所有路径在 Agent C 或目标服务器上自行确定；不依赖原 ThinkPad、原聊天记录或原截图。相关调研见同目录的 [research.md](research.md)，仓库与跨设备说明见 [README.md](README.md)。

## 目标和背景

你是活动103中的大语言模型部署 Agent C。请在本窗口已有的服务器连接与部署环境中，实际部署 DeepSeek-V4.1-Flash-1017，完成可复现的功能与性能验收，不要只提供建议。

目标模型是 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 官方原版。用户已下载完毕；先定位并校验现有目录，不重复下载、不转换格式、不重新量化、不修改原始权重。

用户与 FastLLM 作者沟通得到以下信息：作者确认官方 1017 原版即可，已亲自测试；auto 仍是实验性功能，会继续调整。作者使用 AMD 9005 系列和 DDR5，但完整测试配置未知。用户提到了 7532、DDR4-2666×16、备用双路 8260。这些只是背景，不是已探测硬件，不能据此推断容量、通道或承诺速度。

旧聊天中的双 E5-2680v4、64GB RAM、8×2080 Ti 可能过时。以活动103已有记录和实际机器探测为准。若有多台服务器，确认当前目标，不擅自切换。活动103是任务标识，不是可推导的 IP 或端口。

此次是部署官方模型，不是开发新专家并行内核。

## 跨设备输入和版本资料

本交接仓库为 `https://github.com/TomPython/fastllm`，文档分支为 `agent/dsv4-flash-ep-handoff`。该分支的旧引擎代码不是本次部署基线，不能直接编译它当最新 FastLLM。另行获取上游 `https://github.com/ztxz16/fastllm`。

2026-10-19 核对的上游 SHA：`f691f13f675a98bef36ce0eafe094e8ba927f9bf`。

先阅读选定上游提交中的：

1. `docs/model-deployment-agent.md`
2. `docs/deepseek_v41.md`
3. `docs/hybrid-auto.md`
4. `docs/wheel_build.md`
5. `docs/deepseek_v41_precision.md`
6. `docs/benchmarks/deepseek_v41.md`

固定版本入口：<https://github.com/ztxz16/fastllm/tree/f691f13f675a98bef36ce0eafe094e8ba927f9bf>。
官方模型：<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017>。

可 fetch 检查更新。若采用更新提交，记录完整 SHA 和相关差异；测试期间固定版本，不自动追新。不能依赖原设备本地的 Git 引用、临时文件或已安装环境。

本次查询时 PyPI ftllm 仍为 0.1.8.2，nightly 仍是旧包。源码构建可能仍显示相同版本号。不要只看 pip 版本或 CLI help 就断言原生后端已更新。文档不同段落可能讲手动默认值或旧 Launcher；冲突时核对选定版本源码、CLI 与日志。

只使用 Agent C 自己可访问的 SSH 配置和凭据。没有凭据或模型路径时，先查活动103现有记录；仍缺失才向用户询问，不要求 ThinkPad 在线、不尝试取走其凭据。依赖下载需要目标环境网络或离线包，但不需要原设备。

## 执行边界

使用独立环境、构建目录和空闲端口，保留旧服务与回滚路径。密钥和密码不得进入日志、提交或报告。

不要擅自停止其他业务、杀死未知进程、修改驱动/BIOS/全局 sysctl、删除模型和旧环境、开放无认证外网服务、运行破坏性磁盘测试或全局 drop_caches。不要通过减少激活专家数或改权重追速度。这些操作如有必要，先解释影响并请求批准。

本任务隔离环境中的依赖安装、编译、启动服务和只读诊断可以继续执行；授权范围仍受目标环境权限约束。

## 环境和模型检查

记录以下实测结果：

- 系统、内核、glibc、Python、编译器、CUDA 工具链和驱动。
- CPU 型号、socket、物理核、SMT、AVX2/AVX-512 flags。
- NUMA 节点和各节点内存、进程 affinity/cpuset。
- RAM 总量、MemAvailable、swap、cgroup 限额。
- GPU 型号、数量、显存、compute capability、PCIe/NUMA 拓扑和实际 peer access/NVLink。
- 模型磁盘、文件系统、空间、是否本地 NVMe。
- 端口、已有服务和资源占用。

可用 `lscpu`、`numactl --hardware`、`free`、`nvidia-smi`、`nvidia-smi topo -m`、`lsblk`、`df`、`ss`。不要把 16 根 DIMM 当成 16 个有效通道或把 DDR 频率当实测带宽。不要将另一台 AVX-512 机器的 native 构建直接搬到不支持的机器。

模型检查包括本地路径、来源、`config.json`、tokenizer、safetensors index、`deepseek_v41` 架构、1017 身份、Engram 与 `mtp.*`。检查索引分片存在、大小合理、没有未下载完成文件或 LFS 占位。不要为检查把全部权重读入 RAM，也不无必要地全量重算哈希。

官方没有普通 Jinja 模板，使用 FastLLM V4.1 编码适配，不套用 V3/V4 或其他模型模板。Engram 元数据缺失时按当前后端机制自动生成，保存日志。

## 安装和构建核验

先检查源码状态和未提交修改，不覆盖用户工作。在独立 checkout、构建目录和 Python 环境构建上游选定提交。阅读 `make_whl.sh --help` 与构建文档后决定方式；按资源限制编译并发，不照抄大内存机器的线程数，不为编译擅自升级系统驱动。

记录 Git SHA、构建选项、CUDA 架构、wheel 路径/哈希、Python 可执行文件、ftllm 入口、模块导入路径、实际加载的原生库路径/哈希及 NUMA/CUDA 依赖状态。排除新 Python 包加载旧 `.so`，以及 PATH/PYTHONPATH/LD_LIBRARY_PATH 污染。不要混装 ftllm 和 ftllm-nightly。

## 容量规划与首轮启动

核算权重、Engram、专家、DSpark、KV、工作区、加载峰值与系统余量。旧 V4.1 的数百 GB 只是容量警示，1017 必须按实际元数据和规划结果核算。

明确参数语义：

- `--ngram_device disk` 只控制 Engram，`--moe_device disk` 才控制专家。
- `--moe_cpu_cache` 不是整个进程 RAM 上限。
- `--moe_cuda_cache` 通常是参与 GPU 每卡预算，不是 KV 预算；核对最终计划。
- `--auto_host_memory_limit` 和 `--auto_gpu_memory_limit` 是规划限制，不是运行时 cgroup。
- `--max_context_length` 是输入加输出上限；`--tokens` 是共享 KV 池；API `max_tokens` 是输出预算。
- 自动 TP 搜索主要限双卡相近规格，不会穷举八卡最优布局；不要同时指定 TP 和 cudapp。
- Qwen4 的自动 KV streaming 不能直接当作 V4.1 能力。

上游推荐以 `ftllm server "$MODEL_DIR" --auto` 启动。V4.1 auto 会默认规划最多 1M、DSpark 7、近似 fast_prefill，预算不足不会自动缩短窗口。

本任务先用以下保守覆盖建立文本基线；这不是上游默认值。`MODEL_DIR` 和 `PORT` 必须在目标服务器上确定，端口不得占用；命令执行前核对当前 help。

```bash
ftllm server "$MODEL_DIR" --auto \
  --model_name deepseek-v4.1-flash-1017 \
  --host 127.0.0.1 --port "$PORT" \
  --max_context_length 32768 --max_batch 1 \
  --no_fast_prefill --dspark 0
```

按实际资源添加必要规划预算。若 32K 不可行，解释后使用更小测试窗口，不反复 OOM。保留 `[auto]`、设备、线程、专家/Engram 后端、缓存、chunk、KV dtype、Graph、上下文与资源日志。估算不等于运行峰值，继续监测 RSS、显存、swap 和物理磁盘读取。依据加载进度判断，不因 HTTP 暂时无响应而反复重启。

## 功能验收

满足以下条件后才能说部署成功：

1. ready 或等价就绪事件。
2. `/v1/models` 返回正确 ID。
3. 短文本 HTTP 200、输出合理、无 NaN/设备错误/异常循环。
4. 流式正常结束，记录 finish_reason。
5. 两轮对话正确；取消或断开后可以处理后续请求。
6. 无副作用工具调用及工具结果回传正确。
7. 无非法显存访问、通信超时、OOM、反复恢复前缀等错误。

思考模型可能先返回 reasoning_content，不因 content 暂空判失败。为冒烟测试设置合理输出预算，不照搬超长推理评测值。先做单请求文本；图像、多并发独立验收，当前 DSpark 组合有边界。

## 逐项优化和测速

依次比较：A 关闭 fast_prefill 与 DSpark；B 保持非近似 prefill、开启 DSpark 7；C 开启 fast_prefill 并保留 DSpark 7。必要时再测 DSpark 5、阈值、线程、缓存和布局，每次只改一项。

auto 重新规划可能连带改变设备、缓存和 chunk。要归因单项优化，必须固定其他实际配置；否则标注整套配置比较。随机采样不要求 DSpark 开关或相同 seed 逐 token 相同。

固定测试集、采样参数、GPU 可见性和长度：短输入约 256 输出；约 4K 输入约 256 输出；稳定后逐级测试 32K、128K，最后决定更长窗口。每组预热后至少三次，报告中位数和实际 token 数。前缀冷请求与命中分开，做计算、代码及多位置检索检查。

保存 TTFT、输入/输出 token 数、decode token/s、总耗时、RSS/显存峰值、物理磁盘读取、接受情况、CPU/GPU 利用率、回退及 Graph 是否实际生效。

fast_prefill 是近似计算，不是无损开关；普通非近似 prefill 也不等于官方参考逐位对齐。参考精度模式仅用于必要诊断，不偷偷拿它作为正常速度基线。规划到 1M 不代表已实测 1M。

## 性能异常诊断

不要先认定 DDR4 比 DDR5 慢三倍。按证据依次排查：

- 旧库、错误环境、参考数学或标量慢路径、CPU ISA 不匹配。
- FT_THREADS、FASTLLM_NUMA_THREADS、FT_NUMAS、FASTLLM_TP、Graph/profiling 旧变量覆盖。
- affinity、NUMA 绑核、内存分配、跨 socket 通信。
- 意外磁盘专家/Engram、swap、磁盘饱和、缓存命中与实际放置。
- 主干和共享专家是否确实在 GPU，DSpark 验证是否用缓存。
- DSpark 收益是否抵消验证和 I/O，GPU 降频或 PCIe 链路问题。
- prefill、思考输出、decode、冷启动是否混在同一速度指标。
- 构建是否包含目标 GPU 和 NUMA 支持。

先保存环境变量，再仅在本任务进程清理或覆盖；不要全局调系统掩盖问题。作者截图缺少完整条件，不是严格性能基线。

## 交付和继续工作

在 Agent C 的活动103项目目录保存硬件/软件/模型清单、版本与构建步骤、基础及推荐配置、启动脚本、日志、调用示例、性能质量结果、停止恢复与回滚方法。明确未测能力及近似计算。

准备供用户反馈作者的摘要：硬件、SHA、命令、模型、输入输出长度、性能、资源峰值、问题和最小复现。不要自动向作者发消息，不自动推送仓库，不保存凭据。

本次交接本身尚未远程部署。现在先完成只读检查与版本核验，给出实际硬件和容量判断，再继续隔离部署及验收；需要额外权限或缺少关键输入时再请求用户补充。
