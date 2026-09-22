# 流萤商城 AI 智能客服

<p align="center">
  <b>中文</b> · <a href="README.en.md"><u>English</u></a>
</p>

> 流萤商城（Firefly Mall）的智能客服子系统。基于 **LangGraph + LangChain + FastAPI** 构建的多智能体（Multi-Agent）客服系统：一个主
> Agent 负责"调度 + 发言"，下辖售前、售中、售后三位专家 Agent；内置自研的 **衡忆多维认知架构（BalancedMultiDimensionMemory）**
> 记忆框架，让客服在跨会话、跨天数的对话中真正"记住"用户。

![系统架构图](graph.png)

## 项目整体内容与效果

本项目是流萤商城的 AI 客服后端，实现了从"用户提问"到"问题解决"的完整闭环：

| 能力           | 说明                                                                                                        |
|--------------|-----------------------------------------------------------------------------------------------------------|
| 🤖 多智能体协作    | 主 Agent 听懂需求后自动路由：售前（商品咨询 / 以图搜商品 / 智能推荐）、售中（下单 / 购物车 / 订单查询 / 发票）、售后（退换货 / 退款 / 进度查询），多领域问题可**并行指派**多个专家 |
| 🧠 自研记忆框架    | **BalancedMultiDimensionMemory**：三层记忆架构 + 多维加权评分 + 参数化艾宾浩斯遗忘曲线 + LLM 动态调参，实现真正的跨会话长期记忆（详见下文重点模块）          |
| 💬 自然对话体验    | SSE 流式输出（打字机效果 + 工具调用提示 + 思考过程回滚），语气亲切，符合商城客服人设"小刘"                                                       |
| 🛠 工具调用      | 主 Agent 与专家共注册 20+ 工具：FAQ 向量检索（Milvus）、商品处理、订单/购物车/发票、退换货、邮件发送等                                           |
| 👨‍💼 人工客服协作 | 支持转人工、人工接管期间 AI 静默、人工回复后 AI 转达并恢复正常服务，全流程按会话线程隔离                                                          |
| 🔒 安全与可靠性    | JWT 双密钥轮换认证、文件路径穿越防护、RabbitMQ 错误恢复队列、邮件告警、节点重试与超时、LangGraph 节点缓存                                          |

**效果示例**（以一段真实使用场景演示）：

1. 用户今天问"我家是南方比较潮湿，推荐防潮的收纳盒" → 售前专家根据偏好推荐商品，记忆框架记录下"用户在意防潮"这一偏好片段；
2. 用户明天问"上次买的收纳盒我想退了" → 无需重复说明背景，主 Agent 自动检索到昨天的记忆片段与用户画像，售后专家直接基于上下文处理退换货；
3. 用户性格急躁、说话简短 → 画像层记住沟通风格，后续回复自动适配。

## 技术栈

- **语言 / 框架**：Python 3.10+、FastAPI、Uvicorn
- **Agent 编排**：LangGraph 1.x（手写 StateGraph）、LangChain Agents（子 Agent 用 `create_agent`）、`Send` 并行路由
- **模型**：DeepSeek（`deepseek-v4-flash`，OpenAI 兼容协议接入）、DashScope Embedding（`text-embedding-v4`，支持视觉模型）
- **存储**：Redis（JWT 密钥 / 记忆游标）、RabbitMQ（错误恢复队列）、SQLite + sqlite-vec（记忆片段向量库）、Milvus（FAQ
  向量检索）、MySQL / PostgreSQL（商城业务数据，接口预留）
- **流式**：SSE（`text/event-stream`）
- **认证**：PyJWT 双密钥轮换

## 目录结构

```
├── main.py                  # FastAPI 入口：JWT 校验中间件、CORS、路由注册、lifespan（启动 RabbitMQ 消费者）
├── run.py                   # 启动脚本：配置 HF 镜像/缓存目录后拉起 uvicorn
├── config.yaml.template     # 全局配置模板：复制为 config.yaml 后使用（模型、数据库、记忆框架调参，见"配置说明"）
├── .env.template            # 环境变量模板（API Key 等；真实 .env 不入库）
├── Dockerfile               # 镜像构建：多阶段构建（builder 装依赖 → runtime 瘦身）
├── docker-compose.yaml      # 全栈编排：应用 + 7 个依赖服务，一键起完整环境
├── config.docker.yaml       # 容器内配置：用服务名寻址，挂载为容器内 config.yaml
├── .dockerignore            # 构建上下文排除项（密钥、运行时数据、前端产物）
├── k8s/                     # Kubernetes 清单（kustomize 组织，见"容器化部署"章节）
├── SPO/                     # 结构化对象层（State/模型/响应）
│   ├── state.py             #   LangGraph 图状态、输入输出 Schema、路由分类
│   ├── memory.py            #   记忆数据模型：UserProfile 画像、MemoryFragments 片段、SummaryMemoryAi
│   ├── user.py              #   用户业务画像（售前专家用：喜好/憎恶/品牌/风格）
│   └── route_results.py     #   统一 API 响应格式与状态码
├── agent/                   # Agent 层
│   ├── main_agent.py        #   ★ 主 Agent：手写 StateGraph、路由分发、中间件洋葱链组装
│   ├── front_desk_salesperson.py  # 售前专家（商品推荐 / 以图搜商品）
│   ├── during_sale_service.py     # 售中专家（下单 / 订单 / 发票）
│   ├── after_sales_service.py     # 售后专家（退换货 / 退款）
│   └── prompt/              #   三位专家的系统提示词
├── Tools/                   # 工具与中间件层
│   ├── registry.py          #   工具注册中心（@register_tool 装饰器，按 Agent 名分组）
│   ├── main_agent_tool.py   #   主 Agent 工具：FAQ 检索 / FAQ 元数据查询（Milvus）
│   ├── product_process.py   #   售前工具：商品搜索 / 推荐 / 以图搜商品
│   ├── during_sale_tool.py  #   售中工具：下单 / 购物车 / 订单状态 / 发票
│   ├── after_sales_tool.py  #   售后工具：退换货申请 / 退款 / 进度查询
│   ├── db.py                #   商城数据库访问
│   ├── email.py             #   邮件发送（含异常告警）
│   ├── jwt_key_manage.py    #   JWT 双密钥轮换管理
│   ├── log_settings.py      #   日志配置（按模块分文件、滚动）
│   └── middleware/          #   ★ 中间件层
│       ├── compose.py       #     手写"洋葱链"组合器（脱离 create_agent 复现官方接线）
│       ├── file_notice.py   #     文件消息通知中间件
│       ├── tool_notice.py   #     工具调用通知中间件（流式"调用了工具X"提示）
│       └── memory/          #     ★★★ 自研记忆框架（详见下文）
│           ├── time_memory.py          #     BalancedMultiDimensionMemory 核心中间件
│           ├── memory_rag.py           #     分片器 + 记忆 RAG 存储 + RabbitMQ 恢复消费者
│           ├── customize_sqlite_vec.py #     自定义 sqlite-vec 向量库（支持元数据过滤）
│           ├── token_calculate.py      #     Token 计算工厂（tiktoken → HF → 兜底估算）
│           └── prompt.py               #     提示词分层组装（core → profile → fragments）
├── routes/                  # FastAPI 路由
│   ├── ai_chat.py           #   对话（SSE 流式）/ 人工客服协作 / 会话历史
│   ├── file.py              #   文件上传 / 下载
│   └── manager.py           #   管理后台：FAQ 增删改查
├── load_config/             # 配置加载（yaml 解析 + .env 合并）
├── model/                   # 注入检测模型（ONNX 从 Kaggle 导出后手动放置）
│   └── injection_detector.py   # 注入检测器：ONNX 推理 + 分级 + 恶意值（Redis 原子）
├── tests/                   # 测试脚本与测试数据
│   ├── graph_test.py        #   图结构测试脚本
│   ├── injection_two_hop_test.py   # 二跳注入验证
│   ├── bench_cache_rate.py  #   K-V 缓存命中率基准
│   ├── bench_memory_recall.py      # 记忆召回基准
│   ├── build_injection_dataset.py  # 注入数据集构建
│   ├── check_injection_dataset.py  # 注入数据集质量检查
│   ├── train_injection_classifier.ipynb  # Kaggle 训练 notebook（生成器 make_train_notebook.py）
│   ├── verify_onnx.py       #   ONNX 本地验证（复现阈值指标）
│   └── data/                #   测试数据与结果（eval_gold.json 等）
└── graph.png                # LangGraph 系统架构图
```

> 前端代码（Vue3）独立维护于 `static/`，暂时未包含在本仓库中（已 gitignore）。

> **模型文件说明**：`model/roberta_inj.onnx`（提示词注入检测模型，约 400MB）不随仓库分发，
> 请从 [Kaggle notebook](https://www.kaggle.com/code/mrli55/preventive-prompt-word-injection-model)
> 的 **Output** 标签页下载 `roberta_inj.onnx`，放入 `model/` 目录后运行
> `python tests/verify_onnx.py` 验证推理与阈值指标。

## 重点模块讲解

### ★★★ 记忆框架：BalancedMultiDimensionMemory（衡忆多维认知架构）

这是本项目的核心自研模块（`Tools/middleware/memory/`），作为一个 **LangChain Agent 中间件** 挂在主 Agent
的模型调用前后，实现"记忆的分层存储 — 检索 — 巩固 — 遗忘"。设计参照了认知心理学中的 **艾宾浩斯遗忘曲线** 与 **自我参照效应
**。

#### 1. 三层记忆架构

| 层级           | 载体                            | 内容                                                                 |
|--------------|-------------------------------|--------------------------------------------------------------------|
| ① 工作记忆（短期）   | 带时间戳的 messages 列表             | 当前会话的原始对话，逐轮积累                                                     |
| ② 记忆暂存库（待巩固） | SQLite + sqlite-vec 向量库       | 从工作记忆卸载的**原始对话切片**，带元数据（主题、类型、时间、巩固次数），是长期记忆的原材料                   |
| ③ 长期记忆库（已巩固） | LangGraph Store（按 user_id 隔离） | LLM 归纳出的**结构化用户画像**（UserProfile：姓名/偏好/沟通风格/目标/价值观等 13+ 维度），高持久、高重要 |

#### 2. 写入流程：成熟度触发 + LLM 主题分片

1. 每条消息入队时由中间件打上真实时间戳（`awrap_tool_call` 在工具执行完成当刻打，避免下一轮补打产生时间偏差）；
2. 系统周期性计算最早消息的**记忆成熟度**：

   ```
   M(Δt) = 1 - exp( -(Δt / τ_m)^c )
   ```

   当 `M ≥ slice` 阈值（客服场景默认 0.7）时触发切片——时间越久越"成熟"的消息优先进入分片流程；
3. 调用轻量 LLM 按**主题**切分对话（`MemoryFragmentsAiSpliter`），为每个片段生成主题 + 类型标签，类型固定六种：
   `identity`（身份）/ `preference`（偏好）/ `decision`（决策）/ `fact`（事实）/ `episode`（经历）/ `chat`（闲聊），
   并附上 `scope`（左闭右开索引区间）便于回溯原始消息；
4. 片段落库到记忆暂存库（sqlite-vec，按 `user_id` 过滤隔离），同时维护 Redis 游标记录每个用户的片段序号。

#### 3. 检索流程：多维加权评分 + LLM 动态调参

当上下文 token 数 / 消息数超过触发阈值（支持 `fraction` / `tokens` / `messages` 三种模式）时，中间件重写系统提示词，注入检索到的记忆片段。片段得分公式：

```
S(m) = α·R(m,q) + β·T(m) + γ·F(m) + δ
```

| 项            | 公式                                                               | 含义                                                                                                                             |
|--------------|------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| R(m,q) 语义相关性 | `max(0, cos(m,q) - θ_min)`                                       | 片段与当前主题的余弦相似度（硬阈值过滤）                                                                                                           |
| T(m) 时间衰减    | `exp( -(Δt / τ)^c )`                                             | **参数化艾宾浩斯遗忘曲线**：τ 越大遗忘越慢，c 控制曲线形状（c<1 先快后慢，贴合艾宾浩斯早期形状）                                                                         |
| F(m) 固有重要性   | `clamp(w0 + w1·Σ(v_i·I_type(i)) + w2·(1-exp(-refresh·k)), 0, 1)` | 类型先天重要性 × 巩固次数的后天强化；`I_type` 依据自我参照效应设定：identity=0.95 > decision=0.85 > preference=0.80 > fact=0.60 > episode=0.40 > chat=0.15 |
| α/β/γ/δ 动态权重 | LLM 按当前对话主题实时调整，约束 `α+β+γ=1`                                     | 客服场景 α 大（重语义），闲聊场景 β 大（重时效）——权重随场景自适应                                                                                          |

检索后取 Top-K 片段注入提示词，**成功注入并辅助生成回复后触发"再巩固"**：

- 该片段 `strengthen_num + 1`（落库采用先删除再添加，sqlite-vec 不支持原地更新）；
- 刷新时间戳，变相重置其遗忘曲线起点。

#### 4. 整理流程：增量归纳为长期记忆

延续成熟度公式，当**最早片段成熟 AND 积累量达到阈值**时触发记忆整理：

- 只对归纳游标（`last_summarized_id`，存于画像中）**之后的新片段**做增量总结，绝不重复总结；
- LLM 产出结构化增量画像后，通过 `merge_user_profile` **智能合并**
  进旧画像：列表字段去重合并、字典字段递归合并、标量字段仅在非空且非默认值时覆盖——避免增量总结把未提及的字段重置为默认值。

#### 5. 防上下文分裂与提示词分层

- 维护 `new_msg_idx` 游标，只把**未处理的增量消息**交给模型，避免整段历史反复重写系统提示词导致"上下文分裂"；
- 提示词分三层拼接（`prompt.py`）：**静态核心（角色设定）→ 半静态画像（每轮刷新）→ 动态记忆片段（每轮变化）**
  ，遵循缓存前缀规则，让最稳定的内容排最前，最大化模型服务端缓存命中率。

#### 6. 可靠性设计

- **失败兜底**：分片 / 总结的 LLM 调用失败时，消息不丢弃，投递到 RabbitMQ 错误队列（`unclassified_fragments` / `uncut_text`
  ），消费者恢复后自动续接 Redis 游标重试；入队超 1 天未处理成功则邮件告警；
- **场景化参数**：内置三套职业调参（`_VOCATION_PARAM_MAP`）：`customer service`（短时记忆、快遗忘）、`collaborative creation`
  （长时记忆、慢遗忘）、`accompany`（陪伴型折中），可通过 `config.yaml` 一键切换，也可 `customize` 全自定义；
- **Token 计算工厂**：按模型自动选择 tiktoken → HuggingFace 分词器 → 字数/3.3 兜底三级降级，避免触发阈值失准。

### ★ 主 Agent：手写 StateGraph + 中间件洋葱链

- `agent/main_agent.py` 手工构建 LangGraph
  图：`msg_handle → chat_node → forward_node → (user / artificial / tools / 三位专家) → END`；
- **手写"洋葱链"**（`Tools/middleware/compose.py`）：脱离 `create_agent` 手动复现官方中间件接线——before/after
  钩子并入图状态、`awrap_model_call` 组合成洋葱链包住真实模型调用、`awrap_tool_call` 传给 ToolNode，并实现了 Command
  顺序累积、最外层胜出的语义；
- **路由三路径解析**：模型响应优先解析工具调用（`tool_calls`）→ 其次解析 JSON 内容 →
  最后把自然语言直接当作最终回答，一次调用零额外成本完成"调度 or 发言"决策；结构异常时兜底转人工，绝不把异常抛给用户；
- **并行指派**：多领域问题通过 `Send` 并行派发给多个专家，每个节点带独立超时（售前 20 分钟 / 售中 10 分钟 / 售后 15 分钟）；
- **人工客服协作**：`manual_intervention` 标记按线程隔离，人工接管期间 AI 静默（不调 LLM
  直接返回固定话术），人工回复经 `human_reply` 入口注入后 AI 转达并恢复正常服务。

### ★ 子 Agent 专家系统

三位专家均用 `create_agent` 构建，通过 `@register_tool('agent_name')` 注册机制按需装配工具：

- **售前专家**：商品搜索 / 智能推荐 / 以图搜商品（读取用户业务画像 `user_person`：喜好、憎恶点、品牌与风格偏好，推荐时自动避雷）；
- **售中专家**：辅助下单 / 购物车 / 订单状态查询 / 发票申请与下载；
- **售后专家**：退换货规则咨询 / 退换货申请 / 售后进度查询 / 退款问题。

### ★ SSE 流式输出（routes/ai_chat.py）

对话接口 `POST /ai/chat` 通过 SSE 推送五类事件，前端据此实现打字机效果：
`token`（逐字增量）→ `tool`（调用了工具X）→ `rollback`（上一段是内部思考，前端回滚清除）→ `turn`
（本段为最终回答，前端固定）→ `done`（结束，含 out_msg / manual_intervention）。

### ★ 其他亮点

- **JWT 双密钥轮换**：新旧密钥并存签发/校验，通过 `replace_jwt` 响应头通知前端换新 token，平滑完成密钥轮换；
- **文件安全**：上传文件校验路径穿越（abspath 必须位于 upload_dir 内），仅返回引用，专家工具按需读取；
- **节点级容错**：每个 Agent 节点配置 RetryPolicy（指数退避重试 4 次）与缓存策略（固定话术节点缓存 1 个月）。

## 快速开始

### 1. 环境依赖

| 依赖                  | 用途                               | 必装 |
|---------------------|----------------------------------|----|
| Python 3.10+        | 运行环境                             | ✅  |
| Redis               | JWT 密钥、记忆片段游标                    | ✅  |
| RabbitMQ            | 记忆处理失败重试队列                       | ✅  |
| Milvus              | FAQ 向量检索                         | ✅  |
| Elasticsearch       | 商品检索（售前工具，**导入期即连接，不可缺**）        | ✅  |
| SQLite + sqlite-vec | 记忆片段向量库（sqlite-vec 需单独安装扩展）      | ✅  |
| PostgreSQL          | LangGraph Checkpoint / Store 持久化 | ✅  |
| MySQL               | 商城业务数据（当前以测试桩数据运行）               | 可选 |

> Elasticsearch 是**硬依赖**：`Tools/product_process.py` 在模块导入期就会 `connect_to_elasticsearch`，
> 而该模块经 `agent/front_desk_salesperson.py` 被 `main.py` 导入，所以 ES 不可达时应用会在启动阶段直接失败。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置（使用说明）

本项目共两份配置模板，均需**复制一份再改**，原件不入库：

```bash
# ① 环境变量：真实 API Key 都在这里
cp .env.template .env

# ② 全局配置：模型 / 数据库 / 记忆框架参数
cp config.yaml.template config.yaml
```

#### .env.template 使用说明

| 变量                                                              | 必填 | 说明                                                  |
|-----------------------------------------------------------------|----|-----------------------------------------------------|
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`     | ✅  | 主 Agent 对话模型（默认 `deepseek-v4-flash`，OpenAI 兼容协议）    |
| `DASHSCOPE_API_KEY`                                             | ✅  | Embedding（`text-embedding-v4`，记忆片段向量化与 FAQ 检索）与视觉模型 |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`           | 可选 | 若对话模型改走 OpenAI 兼容的其它服务商                             |
| `TAVILY_API_KEY`                                                | 可选 | 联网搜索（预留）                                            |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | 可选 | LangSmith 链路追踪                                      |
| `EMAIL_AUTH_CODE`                                               | 可选 | QQ 邮箱授权码，用于异常告警邮件（SMTP 服务器与收件人在 config 中配置）         |

#### config.yaml.template 使用说明

复制为 `config.yaml` 后，按需修改；模板内已保留本地开发默认值，最小改动即可跑通。配置文件本身带完整注释，这里只说明**必改项**
与**记忆框架相关参数**：

**必改项**

- `AMQP.rabbitmq`：RabbitMQ 的账号密码（模板已脱敏）
- `databases.rag.milvus.conn_args.uri`：Milvus 服务地址（模板已改为 localhost 占位）
- `email.sender / receiver`：告警邮件的发件人与收件人

**★ 记忆框架相关参数（`model` 段，全部有注释）**

| 参数                                              | 位置      | 作用与建议                                                                                                                              |
|-------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------|
| `model.vocation.name`                           | 场景预设    | 记忆框架的**应用场景开关**，三选一：`customer service`（客服：短时记忆、快遗忘）、`collaborative creation`（协同创作：长时记忆、慢遗忘）、`accompany`（陪伴型折中），或 `customize` 完全自定义 |
| `model.vocation.kwargs`                         | 自定义调参   | 仅 `customize` 模式必填，结构见模板内注释：`M(Δt)`（成熟度曲线 τ_m/c）、`T(m)`（遗忘曲线 τ/c）、`slice`（切片触发阈值）、`long-term`（归纳触发阈值）                              |
| `model.summary.name`                            | 分片/总结模型 | 负责"对话主题切分"与"用户画像归纳"的 LLM，可与对话模型不同                                                                                                  |
| `model.summary.kwargs.profile.max_input_tokens` | 基准容量    | 该模型的最大输入 token 数，`fraction` 模式的触发基准                                                                                                |
| `model.summary.pattern`                         | 触发模式    | `fraction`（按占比）/ `tokens`（按 token 数）/ `messages`（按消息条数）                                                                            |
| `model.summary.trigger_threshold`               | 触发阈值    | fraction 建议 ≥ 0.7；tokens 建议 ≥ 10000；messages 建议 ≥ 50                                                                               |

**调参建议**：三套场景预设的具体数值硬编码在 `Tools/middleware/memory/time_memory.py` 的 `_VOCATION_PARAM_MAP`
中（源码内同样有注释）。日常使用直接切 `vocation.name` 即可；只有当你想针对自己的业务精调"多久切一次片、记忆衰减多快"
时，才需要用 `customize` 模式 + `kwargs` 手写参数。具体公式与含义见上文"记忆框架"章节。

### 4. 启动

```bash
python run.py
```

启动成功后访问 `http://127.0.0.1:8000`。API 文档见 `http://127.0.0.1:8000/docs`（需携带有效 token）。

### 5. 常用接口

| 方法   | 路径                     | 说明                    |
|------|------------------------|-----------------------|
| GET  | `/ai/health`           | 健康检查（**免鉴权**，供容器探针使用） |
| POST | `/ai/chat`             | 用户对话（SSE 流式）          |
| POST | `/ai/human/end`        | 结束人工客服（AI 转达结语）       |
| GET  | `/ai/history`          | 会话历史（人工客服接手前查看上下文）    |
| POST | `/files/upload`        | 文件上传                  |
| GET  | `/manager/get/all/faq` | 获取 FAQ 列表             |

> 除 `/ai/health` 外，所有接口都经过 `main.py` 的 JWT 中间件校验，需在请求头带 `Authorization: Bearer <token>`。

### 6. 容器化部署

三种用法按场景选：**直接跑镜像**（6.2，已有 Redis / Milvus 等外部依赖时最轻量）、
**Docker Compose**（6.3，单机一键起全栈，适合本地验证与演示）、**Kubernetes**
（6.4，`k8s/` 目录，kustomize 组织，适合集群部署）。

#### 6.1 构建镜像

```bash
# 在仓库根目录构建（依赖 ./Dockerfile 与 ./requirements.txt）
docker build -t intelligent-customer-service:latest .

# 查看构建结果
docker images intelligent-customer-service
```

镜像特点与注意事项：

- **多阶段构建**：`builder` 阶段装编译依赖与 Python 包，`runtime` 阶段只复制 `/root/.local`，
  镜像里不含编译工具链。
- **首次构建较慢**：要装 `torch`、`transformers` 等大依赖；后续构建命中层缓存会快很多。
- **构建上下文已排除密钥与数据**：`.dockerignore` 排除了 `.env`、`config.yaml`、`data/`、`database/`、
  `static/`（169MB 前端产物）等，所以镜像里**没有**配置文件和 API Key，全靠运行时注入。
- **构建前请确认两个模型文件就位**（见上文"模型文件说明"）：`model/roberta_inj.onnx`（约 400MB，
  注入检测）与 `model/hfl/`（tokenizer 缓存）。这两个都不入库，缺了镜像能构建成功但运行时会报
  `FileNotFoundError`。
- 镜像内已固化 `HF_ENDPOINT=https://hf-mirror.com` 与 `HF_HOME=/app/huggingface_cache`
  （容器入口是 uvicorn 而非 `run.py`，`config.yaml` 里的 huggingface 段不会自动生效）。
  需要海外直连时用 `-e HF_ENDPOINT=https://huggingface.co` 覆盖。

推送到镜像仓库（Kubernetes 多节点部署时需要）：

```bash
docker tag intelligent-customer-service:latest <registry>/intelligent-customer-service:v1.0.0
docker push <registry>/intelligent-customer-service:v1.0.0
# 然后把 k8s/app.yaml 的 image 改成该地址
```

#### 6.2 直接运行容器（docker run）

适合已经有现成的 Redis / PostgreSQL / MySQL / RabbitMQ / Milvus / Elasticsearch，只想跑应用的场景。

```bash
# ① 准备环境变量：真实 API Key 写入 .env（不入库）
cp .env.template .env

# ② 运行（PowerShell 里路径变量用 ${PWD}）
docker run -d --name intelligent-app \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/config.docker.yaml:/app/config.yaml:ro" \
  -v app-uploads:/app/data \
  -v app-database:/app/database \
  intelligent-customer-service:latest

# ③ 健康检查（免鉴权）
curl http://localhost:8000/ai/health
```

容器内需要挂载 / 会写入的路径：

| 容器内路径                    | 内容                            | 是否必须                              |
|--------------------------|-------------------------------|-----------------------------------|
| `/app/config.yaml`       | 应用配置（挂载 `config.docker.yaml`） | **必须**，不挂载启动即 `FileNotFoundError` |
| `/app/data`              | 上传 / 下载文件                     | 建议持久化                             |
| `/app/database`          | SQLite 记忆库（`test1.db`）        | 建议持久化                             |
| `/app/cache`             | LangGraph SQLite 缓存           | 可选，丢了只是缓存失效                       |
| `/app/log`               | 日志                            | 可选                                |
| `/app/huggingface_cache` | HF 模型缓存（镜像 `HF_HOME` 指向这里）    | 可选，能加速冷启动                         |

> **网络注意**：`config.docker.yaml` 里各依赖的 host 写的是 compose 服务名
> （`redis` / `postgres` / `mysql` / `rabbitmq` / `milvus` / `elasticsearch`），单独 `docker run`
> 时容器解析不了这些名字。两种做法：
> ① 加入 compose 创建的网络——先用 `docker network ls` 查到实际网络名（compose 默认带项目名前缀），
> 再加 `--network <该网络名>`；
> ② 复制一份配置，把 host 改成宿主机地址（Docker Desktop 用 `host.docker.internal`），
> 并确保各依赖的端口已映射到宿主机（6379 / 5432 / 3306 / 5672 / 19530 / 9200）。

#### 6.3 Docker Compose

```bash
# ① 准备环境变量：真实 API Key 写入 .env（不入库）
cp .env.template .env

# ② 一键起全栈：应用 + redis / postgres / rabbitmq / mysql / milvus(+etcd+minio) / elasticsearch
docker compose up -d

# ③ 查看状态与日志
docker compose ps
docker compose logs -f app

# ④ 健康检查（免鉴权）
curl http://localhost:8000/ai/health
```

常用运维命令：

| 场景             | 命令                             |
|----------------|--------------------------------|
| 构建 / 重建应用镜像    | `docker compose build app`     |
| 只重启应用          | `docker compose restart app`   |
| 跟随日志           | `docker compose logs -f app`   |
| 进入容器           | `docker compose exec app bash` |
| 停止并删除容器（保留数据卷） | `docker compose down`          |
| 连数据卷一起清掉       | `docker compose down -v`       |

- 应用配置走 `config.docker.yaml`（已把各依赖的 host 改为 compose 服务名），由 compose 挂载为容器内
  `config.yaml`，无需改本地 `config.yaml`。
- 依赖服务均配置了 healthcheck，应用通过 `depends_on: service_healthy` 等待其就绪后再启动。
- 首启动需构建镜像（要装 `torch`、`transformers` 等，耗时较长），后续启动走层缓存。
- 建议给 Docker 分配 **≥ 8GB** 内存：Milvus + Elasticsearch + MySQL + PostgreSQL 同时常驻。

#### 6.4 Kubernetes

```bash
# ① 先把 secret.yaml 里的占位符替换为真实值（API Key / 数据库密码）
#    base64 编码命令：echo -n "your-value" | base64

# ② 一键部署（kustomize）
kubectl apply -k k8s/

# ③ 查看状态
kubectl get all -n intelligent-cs
kubectl logs -f deploy/intelligent-cs-app -n intelligent-cs

# ④ 集群外访问（port-forward，Ingress 未就绪时用）
kubectl port-forward -n intelligent-cs svc/intelligent-cs-service 8000:8000
curl http://localhost:8000/ai/health
```

| 文件                   | 内容                                                                                   |
|----------------------|--------------------------------------------------------------------------------------|
| `k8s/namespace.yaml` | 命名空间 `intelligent-cs`                                                                |
| `k8s/pvc.yaml`       | 7 个 PVC（postgres / mysql / redis / rabbitmq / app / milvus / es）                     |
| `k8s/secret.yaml`    | API Key 与各依赖的密码（**部署前必须替换占位符**）                                                      |
| `k8s/configmap.yaml` | 应用 `config.yaml`（用集群内 DNS 寻址各服务）                                                     |
| `k8s/*.yaml`         | redis / postgres / rabbitmq / mysql / milvus / elasticsearch 各自 Deployment + Service |
| `k8s/app.yaml`       | 应用 Deployment + Service（含 initContainer 建目录与三类探针）                                    |
| `k8s/ingress.yaml`   | 外部入口（需集群已装 Ingress Controller，含 SSE 透传配置）                                            |

**先让集群拿到镜像**，三选一：

```bash
# 方式 A：推到镜像仓库，节点自行拉取（生产环境做法）
#        把 k8s/app.yaml 的 image 改成 <registry>/intelligent-customer-service:v1.0.0；
#        私有仓库还需补 imagePullSecrets
docker push <registry>/intelligent-customer-service:v1.0.0

# 方式 B：kind（本地开发最常用）
kind load docker-image intelligent-customer-service:latest

# 方式 C：minikube
minikube image load intelligent-customer-service:latest
```

> `k8s/app.yaml` 用的是 `intelligent-customer-service:latest` + `imagePullPolicy: IfNotPresent`，
> 这正是为了配合方式 B/C 的本地镜像加载（若改成 `Always`，节点会去 registry 拉取而
> `ImagePullBackOff`）。反之，走方式 A 的仓库部署建议改用版本化标签，避免节点沿用旧的 `latest` 缓存。

常用运维命令：

| 场景           | 命令                                                                     |
|--------------|------------------------------------------------------------------------|
| 全部资源状态       | `kubectl get all -n intelligent-cs`                                    |
| 应用日志         | `kubectl logs -f deploy/intelligent-cs-app -n intelligent-cs`          |
| 进入容器         | `kubectl exec -it -n intelligent-cs deploy/intelligent-cs-app -- bash` |
| 滚动重启应用       | `kubectl rollout restart deploy/intelligent-cs-app -n intelligent-cs`  |
| 看 Pod 起不来的原因 | `kubectl describe pod -n intelligent-cs -l component=api`              |
| 整体删除（保留 PVC） | `kubectl delete -k k8s/`                                               |

**几个已固化的约束与坑，改动前请先读这条：**

- **应用固定单副本**（`replicas: 1`，HPA 已注释）。`app-data-pvc` 是 `ReadWriteOnce`，跨节点第二副本挂不上盘；
  且每个副本都会独立跑 `KeyManage` 轮换 JWT 密钥，并发轮换会互相覆盖导致 token 失效；会话缓存还是 Pod 内本地
  SQLite，多副本不共享。要扩容需先把这三项改成共享/无状态方案，详见 `k8s/app.yaml` 内注释。
- **密码要写两处**：`configmap.yaml`（应用连库用）与 `secret.yaml`（基础设施 Pod 启动用）。因为 `config.py` 只读
  YAML，其 `${path:default}` 占位符是配置树内部引用、不支持读环境变量，没法单靠 Secret 注入。两处不一致时
  表现为「Pod 都健康但应用连不上库」。
- **app 的 `database` 子目录由 initContainer 创建**：空 PVC 上没有该目录，主容器的 `subPath` 挂载会失败并卡在
  `CreateContainerConfigError`。
- **Elasticsearch 是硬依赖**，清单里已包含；它还带了调 `vm.max_map_count` 的 privileged initContainer，否则启动即报错。
- **HuggingFace 缓存是 emptyDir**：Pod 重建会重新下载 tokenizer。对启动时间敏感的话换成 PVC 或预置进镜像。
- **镜像标签与拉取策略需配套**：当前是 `latest` + `IfNotPresent`，配合 kind / minikube 的本地镜像加载；
  改走镜像仓库时建议换成版本化标签（如 `:v1.0.0`）。

## 与流萤商城的关系

本项目为 **流萤商城（Firefly Mall）的子系统**，独立于商城主服务部署，通过 HTTP 接口与商城的用户体系（JWT）、商品 / 订单 /
发票业务（数据库）协作。前端（Vue3）暂时不会放在本仓库内。

## License

[MIT](LICENSE)
