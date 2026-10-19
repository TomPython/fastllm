# DeepSeek V4.1 Flash 1017 跨设备部署交接

交接对象：另一台设备上的 Agent C，在“活动103”中部署已下载的官方 DeepSeek-V4.1-Flash-1017。调研日期：2026-10-19，用户时区 Pacific/Auckland。

## 阅读顺序

1. [调研结果](research.md)：版本依据、作者沟通摘要、默认行为和风险。
2. [完整部署提示词](agent-c-prompt.md)：可直接交给 Agent C 执行。

## 仓库和引擎版本

- 交接仓库：<https://github.com/TomPython/fastllm>
- 交接分支：`agent/dsv4-flash-ep-handoff`
- 本交接目录：`docs/handoffs/2026-10-19-deepseek-v41-flash-1017/`
- 引擎上游：<https://github.com/ztxz16/fastllm>
- 本次核对的引擎提交：`f691f13f675a98bef36ce0eafe094e8ba927f9bf`

**交接分支的引擎代码不是上述最新上游版本。** 该分支承载历史方案与本次交接文档；Agent C 必须另行获取上游指定提交构建，不能直接编译交接分支并声称已更新。也不能依赖文档作者机器上已有的 Git research-upstream 引用；普通 clone 不会复制那些本地引用。

在 Agent C 自己选择的工作目录中，可新建独立文档副本：

```bash
git clone --single-branch --branch agent/dsv4-flash-ep-handoff \
  https://github.com/TomPython/fastllm.git fastllm-handoff
```

若同名目录存在，先检查，不覆盖。已有 checkout 应先检查工作区，再 fetch 对应分支；不要强制 reset 或自动丢弃改动。

## 跨设备独立性

交接包是 UTF-8 纯文本，内部链接均为仓库相对链接，外部依据均为公开 HTTPS 链接。不需要原 ThinkPad 在线，也不依赖其磁盘、操作系统、Python 环境、临时文件、SSH agent、凭据、浏览器会话或聊天窗口。

原沟通截图没有作为附件入库。与部署有关的信息已经转写到调研结果和提示词中，Agent C 无需原图。无法可靠辨认的小表格数值未作为性能依据。

以下依赖仍然必要，但属于 **Agent C 与目标服务器环境**，不是 ThinkPad 依赖：

- Agent C 自己可用的服务器访问权限；缺少凭据时向用户索取安全的连接方式，不从 ThinkPad 提取。
- 已下载模型在目标服务器上的实际路径和完整文件；“已下载”是用户提供的状态，本次未远程验证。
- 目标机器的 Linux、GPU、内存和磁盘资源，以及安装或构建所需依赖。
- 获取上游源码和依赖的网络，或者由用户在目标环境提供的离线副本。

`MODEL_DIR`、`PORT`、源码目录、输出目录均由 Agent C 在目标环境确定，不指向当前设备。不得把“活动103”推断成 IP 地址、端口号或服务器目录。

## 当前状态

已完成上游代码与文档调研、部署提示词和跨设备交接。尚未连接目标服务器、安装模型运行时、启动模型或测量性能。本次交接不声称已经部署成功。

历史 2026-07-10 的 DSV4 专家并行和 KV 卸载方案仅作背景；本次以 1017 官方模型部署为目标，不默认继续旧内核开发，也不沿用旧硬件容量假设。
