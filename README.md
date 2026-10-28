# PyAgent · 工作区智能助手

Python 3.10+ 实现的 ReAct Agent，支持**图形界面（PyQt5）与控制台双模式**，
可**挂载工作区**执行文件任务、**知识库检索**、**联网采集数据**（多引擎 + 筛选模型
自动过滤）。核心逻辑与 UI 完全解耦，两种模式共用同一套 `core/`。

界面仿 Reasonix Studio：左侧栏（Logo / 新建会话 / 搜索框 / 会话树 / 用户区）+
主聊天区（右对齐用户气泡、带头像的助手消息、可折叠「执行过程」工具卡）+
大圆角输入卡（圆形发送按钮），内置**暗色 / 亮色两套主题**（顶栏 ☀/🌙 一键切换，
持久化保存）。

- **文字可复制**：消息区所有文字（用户气泡、助手正文、工具调用、观察结果、
  系统提示）均可鼠标拖选 / `Ctrl+C` / 右键「复制」
- **界面缩放**：**Ctrl + 鼠标滚轮**在任意位置全局缩放（70%–250%），
  `Ctrl+0` 复位到 100%；字号/图标/侧栏/控件同步变化，缩放值持久化保存

## 安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate ；Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 使用

```bash
python main.py                 # 默认图形界面
python main.py -c              # 控制台 REPL
python main.py --task "整理工作区里的 .py 文件清单"   # 单次任务（跑完退出）
python main.py -w D:\docs\kb   # 启动时直接挂载工作区
```

控制台命令：`/mount <路径>`、`/unmount`、`/reset`、`/tools`、`/skills`、
`/key <API_KEY>`、`/models`、`/model <名称>`、`/web [on|off]`、`/exit`。

## 便携式工程目录

所有状态保存在**工程目录内**，拷走整个文件夹 = 带走全部配置/会话/日志：

```
pyagent/
├── config.json      # 全部配置（首次启动自动从 ~/.pyagent/config.json 迁移）
├── sessions/        # 会话持久化（每会话一个 json：消息 + 工作区路径，最多 50 个）
├── logs/pyagent.log # 运行时日志（1MB × 3 轮转，API Key 自动脱敏）
└── icons/app.png    # 程序图标（可复现：python scripts/make_icon.py）
```

启动自动恢复**最近会话**（含其挂载的工作区）；无会话时恢复**上次的工作区**。
左侧「历史会话」双击切换，点会话行右侧 **✕ 删除**（确认后该会话的对话与历史
记录一并清除），右键可置顶 / 删除。每轮对话结束 / 关闭窗口自动归档。

> ⚠️ **安全提示**：`config.json` 内含明文 API Key，`logs/` 与 `sessions/` 含对话
> 内容——三者均已列入 `.gitignore`，**不要提交、不要把整个目录发给他人**。
> 需要分享时请先删除这三项，或改用 `AGENT_API_KEY` 环境变量提供密钥。

## 模型角色

「设置…」中三组角色可分别配置**模型名称 / Base URL / API Key**（留空 = 使用主配置）：

| 角色 | 用途 |
|---|---|
| **思考模型** (main) | ReAct 主循环推理（默认 = 对话模型） |
| **知识库检索** (kb) | `kb_search` 工具：工作区文件相关性判定与摘录 |
| **联网搜索** (web) | `web_collect` 工具：多引擎采集与结果筛选 |

模型名均为**可编辑下拉框**：点「获取模型」后主模型与三角色下拉同步填充候选项。
典型搭配：本地 qwen2.5:7b 当思考模型 + 云端轻量模型（glm-4-flash）当 kb/web 角色。

## 工具调用协议

`auto`（默认）优先用 **OpenAI 原生 function calling**，端点返回 HTTP 400
不支持 tools 时自动回退到文本 `ACTION/ANSWER` 协议；也可在设置里强制
`native` / `text`。原生协议对 gpt-oss、deepseek-r1 这类思考模型明显更稳
（文本协议会卡在 analysis 通道、final 通道为空）。

## 技能（Skill）

技能 = 含 `SKILL.md` 的目录，是可复用的操作规范包。发现位置：
`<工程>/skills/`、`<工程>/.agents/skills/`、`<工程>/.claude/skills/`、
`<工作区>/.pyagent/skills/`、`~/.pyagent/skills/`、`~/.agents/skills/`
（同名后者覆盖前者）。

渐进式披露：常驻上下文只有「名字 + 一句话描述」；按任务关键词预筛后才暴露成
工具（`skill_<slug>`）；模型判断需要时才加载完整规范。任务与某技能强匹配时
规范会被自动注入。

内置两个示例技能：`code-review`（代码审查规范）、`report-writer`（报告写作规范）。

## 高危操作确认

以下操作在执行前会弹确认框（`confirm_dangerous`，默认开启）：

- 递归删除目录（`delete_file` 且 `recursive=true`）——每次确认
- 执行 shell 命令（`run_command`，还需 `allow_shell=true`）——每次确认
- 执行本地 Python 脚本（`run_python`，无沙箱，与 shell 同级风险）
  ——**同一任务内只确认一次**：修脚本是「写→跑→改→再跑」的循环，
  每次都弹窗太吵；一次允许后本任务内后续执行直接放行，任务结束授权失效；
  点了拒绝则该任务内继续拒绝（不会反复骚扰）

`--task` 单次模式没有交互通道，**默认拒绝**这些操作；受信自动化可在配置中
关闭 `confirm_dangerous`。

## 工程结构

```
pyagent/
├── main.py               # 入口：默认 GUI，-c 控制台，--task 单次任务
├── cli.py                # 控制台 REPL（/mount /unmount /key /skills …）
├── requirements.txt
├── pyproject.toml        # 打包元数据 + ruff / pytest 配置
├── core/                 # ★ 核心（无 UI 依赖，双模式共用）
│   ├── log.py            # 运行时日志：轮转 + Key 脱敏
│   ├── config.py         # 配置：环境变量 > 工程目录 config.json > 默认值
│   ├── llm_client.py     # OpenAI 兼容 SSE 客户端（原生 tools / reasoning 捕获 / 重连）
│   ├── web_search.py     # 联网采集：多引擎链 + 风控切换 + 筛选 + 补爬
│   ├── workspace.py      # 工作区沙箱：路径越界防护
│   ├── sessions.py       # 会话持久化（最多 50 个，置顶免清理）
│   ├── skills.py         # 技能系统（发现/预筛/披露/git 安装，带 mtime 缓存）
│   ├── tools.py          # 工具注册表（文件 / shell / web / kb / python / skill）
│   └── agent.py          # ReAct 循环：原生 + 文本双协议 + 事件流
├── ui/
│   ├── main_window.py    # PyQt5 主窗口（侧栏 / 消息流 / 输入卡 / 会话树 / 工作台）
│   ├── theme.py          # 双主题配色表 + QSS 生成（与窗口逻辑解耦）
│   ├── workers.py        # 后台线程：Agent 执行 / 模型列表拉取
│   └── md_render.py      # Markdown 富文本渲染（自写转换器，无第三方依赖）
├── skills/               # 内置示例技能（code-review / report-writer）
├── scripts/make_icon.py  # 用 QPainter 生成 icons/app.png
└── tests/                # 测试与评测
    ├── run_all.py            # 一键跑全部套件（桩 / 离屏 Qt，不需要真实模型）
    ├── real_eval.py          # 真实端点评测（会调用你的 LLM，耗时数分钟）
    ├── test_robustness.py    # 空回复终止 / 高危确认 / 参数校验
    ├── test_reasoning_fallback.py  # reasoning_content 兜底（文本协议）
    ├── test_protocol_nudge.py      # 协议追问闭环（文本协议）
    ├── test_native_protocol.py     # 原生 function calling 协议
    ├── test_skills.py              # 技能系统：发现/披露/工具/安全/集成
    ├── test_web_collect.py         # 联网采集：心跳 / 可中断 / 预算
    ├── test_session_delete.py      # 会话删除（离屏 Qt）
    └── test_ui_features.py         # 界面功能（离屏 Qt）
```

## 测试

```bash
python tests/run_all.py        # 全量离线套件（当前 203 项断言）
python tests/real_eval.py      # 真实端点评测（需配置 API Key）
```

`run_all.py` 用子进程退出码做主判据，子进程统一 UTF-8 编解码，
在中文 Windows 控制台下也能直接运行。

## 主要行为说明

- **执行结果闭环校验**：工具报错 / 脚本非零退出 / Traceback 判定为失败并自动
  注入修复指令（连续失败会提示换思路）；成功则提醒对照任务目标自检
- **落盘追踪**：执行失败且目标文件本任务从未被 `write_file` 写入时，明确指出
  「你展示的修改没有落盘」
- **空回复兜底**：端点把全部输出放进 `reasoning_content`（llama.cpp 系网关常见）
  时，把思考过程作为回复输出；连续 3 次空回复带原因终止
- **联网采集**：进度实时流入对话流（爬到哪页 / 筛过几条 / 切换引擎），
  有时间预算（`web_timeout`）与「停止」中断；必应跳转链接会自动解码成真实 URL
- **会话/工作区**：挂载/卸载即时写入 `config.last_workspace`；切会话会切到该
  会话记录的工作区

## 已知限制

- 联网采集基于搜索引擎 HTML 页面解析，**属脆弱实现**：引擎改版或触发风控即失效，
  且涉及搜索引擎使用条款；生产环境建议换成正式搜索 API
- `run_python` 与 `run_command` 都没有沙箱，仅靠工作区路径限制 + 用户确认
- 界面文字为中文硬编码，未做多语言
