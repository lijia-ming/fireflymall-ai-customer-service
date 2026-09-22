# Firefly Mall AI Customer Service

<p align="center">
  <a href="README.md"><u>中文</u></a> · <b>English</b>
</p>

> The AI customer-service subsystem of **Firefly Mall**, built with **LangGraph + LangChain + FastAPI**. It is a
> multi-agent system in which a single main agent handles "dispatch + speaking" while three expert agents take care of
> pre-sales, in-sales, and after-sales. It ships with a self-developed memory framework — *
*BalancedMultiDimensionMemory (
Balanced Multi-Dimension Memory)** — so the assistant genuinely *remembers* users across sessions and across days.

![System Architecture](graph.png)

## Overview: What It Does and How It Behaves

This is the AI customer-service backend of Firefly Mall, covering the full loop from "user question" to "issue
resolved":

| Capability                         | Description                                                                                                                                                                                                                                                                                                                                     |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🤖 Multi-agent collaboration       | The main agent understands the request and routes automatically: pre-sales (product consultation / image-based product search / smart recommendations), in-sales (checkout / cart / order tracking / invoices), after-sales (returns / refunds / progress tracking). Multi-domain requests can be dispatched to several experts **in parallel** |
| 🧠 Self-developed memory framework | **BalancedMultiDimensionMemory**: three-layer memory architecture + multi-dimensional weighted scoring + parameterized Ebbinghaus forgetting curve + LLM-adaptive weight tuning, enabling true cross-session long-term memory (see Key Modules below)                                                                                           |
| 💬 Natural conversational UX       | SSE streaming (typewriter effect + tool-call hints + thinking-text rollback), friendly tone matching the mall's assistant persona "Xiao Liu"                                                                                                                                                                                                    |
| 🛠 Tool calling                    | 20+ tools registered across the main agent and the experts: FAQ vector search (Milvus), product processing, order / cart / invoice, returns, email sending, etc.                                                                                                                                                                                |
| 👨‍💼 Human-agent collaboration    | Seamless hand-off to human agents: AI stays silent during takeover, relays the human's reply to the user, and resumes normal service — all isolated per conversation thread                                                                                                                                                                     |
| 🔒 Security & reliability          | Dual-key JWT rotation, upload path-traversal protection, RabbitMQ error-recovery queues, email alerting, node retry & timeout policies, LangGraph node caching                                                                                                                                                                                  |

**Example flow** (a realistic usage scenario):

1. Today the user asks "I live in the south, it's humid — recommend a moisture-proof storage box" → the pre-sales expert
   recommends products based on preferences, and the memory framework records the "cares about moisture-proofing"
   preference fragment;
2. Tomorrow the user says "I want to return the box I bought yesterday" → no need to re-explain the background: the main
   agent retrieves yesterday's memory fragments and the user profile automatically, and the after-sales expert handles
   the return based on that context;
3. The user is impatient and writes tersely → the profile layer remembers their communication style and the replies
   adapt accordingly.

## Tech Stack

- **Language / framework**: Python 3.10+, FastAPI, Uvicorn
- **Agent orchestration**: LangGraph 1.x (hand-written StateGraph), LangChain Agents (`create_agent` for
  sub-agents), `Send` for parallel routing
- **Models**: DeepSeek (`deepseek-v4-flash`, OpenAI-compatible protocol), DashScope Embedding (`text-embedding-v4`;
  vision models supported)
- **Storage**: Redis (JWT keys / memory cursors), RabbitMQ (error-recovery queues), SQLite + sqlite-vec (memory-fragment
  vector store), Milvus (FAQ vector search), MySQL / PostgreSQL (mall business data; interfaces reserved)
- **Streaming**: SSE (`text/event-stream`)
- **Auth**: PyJWT with dual-key rotation

## Directory Structure

```
├── main.py                  # FastAPI entry: JWT middleware, CORS, router registration, lifespan (starts RabbitMQ consumers)
├── run.py                   # Startup script: configures HF mirror/cache dirs, then boots uvicorn
├── config.yaml.template     # Global config template: copy to config.yaml before use (models, databases, memory-framework tuning; see Configuration)
├── .env.template            # Environment-variable template (API keys etc.; the real .env is never committed)
├── Dockerfile               # Image build: multi-stage (builder installs deps -> slim runtime)
├── docker-compose.yaml      # Full-stack orchestration: app + 7 dependencies in one command
├── config.docker.yaml       # In-container config: services addressed by name, mounted as config.yaml
├── .dockerignore            # Build-context exclusions (secrets, runtime data, frontend assets)
├── k8s/                     # Kubernetes manifests (kustomize; see Containerized Deployment)
├── SPO/                     # Structured objects layer (State / models / responses)
│   ├── state.py             #   LangGraph state, input/output schemas, route classification
│   ├── memory.py            #   Memory data models: UserProfile, MemoryFragments, SummaryMemoryAi
│   ├── user.py              #   User business persona (used by the pre-sales expert: likes/dislikes/brands/styles)
│   └── route_results.py     #   Unified API response format and status codes
├── agent/                   # Agent layer
│   ├── main_agent.py        #   ★ Main agent: hand-written StateGraph, routing, middleware onion-chain assembly
│   ├── front_desk_salesperson.py  # Pre-sales expert (product recommendation / image search)
│   ├── during_sale_service.py     # In-sales expert (checkout / orders / invoices)
│   ├── after_sales_service.py     # After-sales expert (returns / refunds)
│   └── prompt/              #   System prompts for the three experts
├── Tools/                   # Tools and middleware layer
│   ├── registry.py          #   Tool registry (@register_tool decorator, grouped by agent name)
│   ├── main_agent_tool.py   #   Main-agent tools: FAQ search / FAQ metadata (Milvus)
│   ├── product_process.py   #   Pre-sales tools: product search / recommendation / image search
│   ├── during_sale_tool.py  #   In-sales tools: checkout / cart / order status / invoices
│   ├── after_sales_tool.py  #   After-sales tools: return requests / refunds / progress
│   ├── db.py                #   Mall database access
│   ├── email.py             #   Email sending (incl. error alerts)
│   ├── jwt_key_manage.py    #   JWT dual-key rotation
│   ├── log_settings.py      #   Logging config (per-module files, rotation)
│   └── middleware/          #   ★ Middleware layer
│       ├── compose.py       #     Hand-written "onion chain" composer (replicates official create_agent wiring)
│       ├── file_notice.py   #     File-message notice middleware
│       ├── tool_notice.py   #     Tool-call notice middleware (streams "AI called tool X" hints)
│       └── memory/          #     ★★★ Self-developed memory framework (see below)
│           ├── time_memory.py          #     BalancedMultiDimensionMemory core middleware
│           ├── memory_rag.py           #     Spliter + memory RAG store + RabbitMQ recovery consumers
│           ├── customize_sqlite_vec.py #     Custom sqlite-vec vector store (metadata filtering)
│           ├── token_calculate.py      #     Token-count factory (tiktoken → HF → fallback estimation)
│           └── prompt.py               #     Layered prompt assembly (core → profile → fragments)
├── routes/                  # FastAPI routes
│   ├── ai_chat.py           #   Chat (SSE streaming) / human-agent collaboration / conversation history
│   ├── file.py              #   File upload / download
│   └── manager.py           #   Admin backend: FAQ CRUD
├── load_config/             # Config loading (yaml parsing + .env merge)
├── model/                   # Injection-detection model (ONNX placed manually after Kaggle export)
│   └── injection_detector.py   # Detector: ONNX inference + tiers + malice score (atomic Redis)
├── tests/                   # Test scripts and test data
│   ├── graph_test.py        #   Graph-structure test script
│   ├── injection_two_hop_test.py   # Two-hop injection verification
│   ├── bench_cache_rate.py  #   K-V cache hit-rate benchmark
│   ├── bench_memory_recall.py      # Memory recall benchmark
│   ├── build_injection_dataset.py  # Injection dataset builder
│   ├── check_injection_dataset.py  # Injection dataset quality checks
│   ├── train_injection_classifier.ipynb  # Kaggle training notebook (generated by make_train_notebook.py)
│   ├── verify_onnx.py       #   Local ONNX verification (reproduce threshold metrics)
│   └── data/                #   Test data and results (eval_gold.json etc.)
└── graph.png                # LangGraph system architecture diagram
```

> The frontend (Vue 3) is maintained as a separate project under `static/` and is **not** currently included in this
> warehouse (
> gitignored).

> **Model file**: `model/roberta_inj.onnx` (prompt-injection detection model, ~400MB) is not
> distributed with this repository. Download it from the **Output** tab of the
> [Kaggle notebook](https://www.kaggle.com/code/mrli55/preventive-prompt-word-injection-model),
> place it under `model/`, then run `python tests/verify_onnx.py` to verify inference and thresholds.

## Key Modules

### ★★★ The Memory Framework: BalancedMultiDimensionMemory

This is the core self-developed module of the project (`Tools/middleware/memory/`). It plugs into the main agent as a *
*LangChain agent middleware**, hooking into model calls to implement "layered memory storage — retrieval —
consolidation — forgetting". Its design draws on the **Ebbinghaus forgetting curve** and the **self-reference effect**
from cognitive psychology.

#### 1. Three-Layer Memory Architecture

| Layer                                       | Carrier                                  | Content                                                                                                                                                                       |
|---------------------------------------------|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ① Working memory (short-term)               | Timestamped `messages` list              | Raw conversation of the current session, accumulated turn by turn                                                                                                             |
| ② Memory staging store (to-be-consolidated) | SQLite + sqlite-vec vector store         | **Raw conversation slices** offloaded from working memory, with metadata (theme, type, time, consolidation count) — the raw material for long-term memory                     |
| ③ Long-term memory (consolidated)           | LangGraph Store (isolated per `user_id`) | **Structured user profile** (UserProfile: name / preferences / communication style / goals / values — 13+ dimensions) distilled by the LLM; high persistence, high importance |

#### 2. Write Path: Maturity Trigger + LLM Theme Splitting

1. Every incoming message is stamped with a real timestamp by the middleware (`awrap_tool_call` stamps at the moment a
   tool finishes, avoiding skew from back-filling on the next round);
2. The system periodically computes the **memory maturity** of the earliest message:

   ```
   M(Δt) = 1 - exp( -(Δt / τ_m)^c )
   ```

   When `M ≥ slice` threshold (0.7 by default for customer service), splitting is triggered — older, "more mature"
   messages are prioritized for the split path;
3. A lightweight LLM splits the conversation **by theme** (`MemoryFragmentsAiSpliter`), producing a theme + type labels
   per fragment. Types are fixed to six:
   `identity` / `preference` / `decision` / `fact` / `episode` / `chat`,
   together with a `scope` (half-open index interval) for tracing back to the original messages;
4. Fragments are persisted to the staging store (sqlite-vec, filtered by `user_id`), while a Redis cursor keeps the
   per-user fragment sequence number.

#### 3. Retrieval Path: Multi-Dimensional Weighted Scoring + LLM-Adaptive Weights

When the context exceeds the trigger threshold (three modes: `fraction` / `tokens` / `messages`), the middleware
rewrites the system prompt and injects retrieved fragments. The fragment score:

```
S(m) = α·R(m,q) + β·T(m) + γ·F(m) + δ
```

| Term                      | Formula                                                                                    | Meaning                                                                                                                                                                                              |
|---------------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| R(m,q) Semantic relevance | `max(0, cos(m,q) - θ_min)`                                                                 | Cosine similarity to the current theme (hard threshold filtering)                                                                                                                                    |
| T(m) Time decay           | `exp( -(Δt / τ)^c )`                                                                       | **Parameterized Ebbinghaus forgetting curve**: larger τ decays slower; c shapes the curve (c<1 decays fast-then-slow, matching the early Ebbinghaus shape)                                           |
| F(m) Inherent importance  | `clamp(w0 + w1·Σ(v_i·I_type(i)) + w2·(1-exp(-refresh·k)), 0, 1)`                           | Type-based innate importance × consolidation-based reinforcement; `I_type` follows the self-reference effect: identity=0.95 > decision=0.85 > preference=0.80 > fact=0.60 > episode=0.40 > chat=0.15 |
| α/β/γ/δ Adaptive weights  | LLM adjusts them in real time per the current conversation theme, constrained by `α+β+γ=1` | Customer service favors α (semantics); small talk favors β (recency) — weights adapt to the scenario                                                                                                 |

Top-K fragments are injected into the prompt; once they **successfully help produce a reply, "reconsolidation" fires**:

- The fragment's `strengthen_num` is incremented (persisted as delete-then-insert, since sqlite-vec has no in-place
  update);
- Its timestamp is refreshed, effectively resetting the forgetting-curve origin.

#### 4. Consolidation Path: Incremental Induction into Long-Term Memory

Continuing with the maturity formula, when **the earliest fragment is mature AND the accumulated amount crosses the
threshold**, memory consolidation triggers:

- Only fragments **after the induction cursor** (`last_summarized_id`, stored in the profile) are summarized
  incrementally — never re-summarized;
- The LLM produces an incremental profile, merged into the old one via `merge_user_profile` **intelligently**: list
  fields merge de-duplicated, dict fields merge recursively, scalar fields only override when non-empty and
  non-default — so an incremental summary never resets unmentioned fields to defaults.

#### 5. Context-Fragmentation Defense & Layered Prompts

- A `new_msg_idx` cursor keeps track of which messages are **unprocessed**, so only the deltas go to the model — the
  system prompt is never rewritten wholesale from the full history, preventing "context fragmentation";
- The prompt is assembled in three layers (`prompt.py`): **static core (persona) → semi-static profile (refreshed per
  turn) → dynamic fragments (change per turn)**, following cache-prefix rules: the most stable content comes first,
  maximizing model-side cache hits.

#### 6. Reliability

- **Failure fallback**: when the split/summary LLM call fails, the message is not dropped — it is published to RabbitMQ
  error queues (`unclassified_fragments` / `uncut_text`); recovery consumers resume from the Redis cursor and retry
  automatically. If a message stays unresolved for over 24h, an alert email is sent;
- **Scenario presets**: three vocation parameter sets (`_VOCATION_PARAM_MAP`): `customer service` (short memory, fast
  forgetting), `collaborative creation` (long memory, slow forgetting), `accompany` (companion-style compromise) —
  switchable in `config.yaml`; `customize` mode allows full custom parameters;
- **Token-count factory**: auto-selects tiktoken → HuggingFace tokenizer → char-count/3.3 fallback based on the model,
  keeping trigger thresholds accurate.

### ★ Main Agent: Hand-Written StateGraph + Middleware Onion Chain

- `agent/main_agent.py` builds the LangGraph
  manually: `msg_handle → chat_node → forward_node → (user / artificial / tools / three experts) → END`;
- **Hand-written "onion chain"** (`Tools/middleware/compose.py`): replicates the official `create_agent` middleware
  wiring without `create_agent` — before/after hooks merge into graph state, `awrap_model_call` composes into an onion
  chain around the real model call, `awrap_tool_call` is passed to the ToolNode, with Command accumulation order and "
  outermost-wins" semantics preserved;
- **Three-path route parsing**: model output is parsed first as tool calls, then as JSON content, and plain natural
  language is treated directly as the final answer — one call, zero extra cost, completing "dispatch or speak".
  Structural failures fall back to human hand-off, never surfacing exceptions to users;
- **Parallel dispatch**: multi-domain requests fan out to multiple experts via `Send`, each node with its own timeout (
  pre-sales 20 min / in-sales 10 min / after-sales 15 min);
- **Human-agent collaboration**: `manual_intervention` flag is isolated per thread; during takeover the AI stays
  silent (returns fixed copy without calling the LLM); the human's reply is injected via the `human_reply` entry point,
  relayed by the AI, and normal service resumes.

### ★ Sub-Agent Expert System

The three experts are built with `create_agent` and assemble tools via the `@register_tool('agent_name')` registry:

- **Pre-sales expert**: product search / smart recommendation / image-based product search (reads the user business
  persona `user_person`: likes, pain points, brand & style preferences — recommendations automatically avoid dislikes);
- **In-sales expert**: assisted checkout / cart / order status / invoice application & download;
- **After-sales expert**: return-policy consultation / return requests / after-sales progress / refunds.

### ★ SSE Streaming (routes/ai_chat.py)

`POST /ai/chat` pushes five event types over SSE, enabling the frontend typewriter effect:
`token` (incremental text) → `tool` ("AI called tool X") → `rollback` (previous segment was internal thinking; frontend
rolls it back) → `turn` (this segment is the final answer; frontend pins it) → `done` (end; carries out_msg /
manual_intervention).

### ★ Other Highlights

- **JWT dual-key rotation**: new and old keys co-exist for signing/verification; a `replace_jwt` response header tells
  the frontend to refresh its token, enabling smooth key rotation;
- **File safety**: uploads are validated against path traversal (abspath must stay inside the upload dir); only
  references are returned, and expert tools read files on demand;
- **Node-level fault tolerance**: every agent node has a RetryPolicy (exponential backoff, 4 attempts) and caching
  policies (fixed-copy nodes cached for a month).

## Getting Started

### 1. Environment Dependencies

| Dependency          | Purpose                                                                     | Required |
|---------------------|-----------------------------------------------------------------------------|----------|
| Python 3.10+        | Runtime                                                                     | ✅        |
| Redis               | JWT keys, memory-fragment cursors                                           | ✅        |
| RabbitMQ            | Memory-processing failure retry queues                                      | ✅        |
| Milvus              | FAQ vector search                                                           | ✅        |
| Elasticsearch       | Product search (pre-sales tools; **connects at import time, not optional**) | ✅        |
| SQLite + sqlite-vec | Memory-fragment vector store (sqlite-vec extension required)                | ✅        |
| PostgreSQL          | LangGraph Checkpoint / Store persistence                                    | ✅        |
| MySQL               | Mall business data (currently runs on test stubs)                           | Optional |

> Elasticsearch is a **hard dependency**: `Tools/product_process.py` calls `connect_to_elasticsearch` at module
> import time, and that module is imported by `main.py` via `agent/front_desk_salesperson.py`. If ES is
> unreachable the application fails during startup.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configuration

Two config templates are provided; **copy each one and edit the copy** — the originals are never committed:

```bash
# ① Environment variables: real API keys live here
cp .env.template .env

# ② Global config: models / databases / memory-framework parameters
cp config.yaml.template config.yaml
```

#### .env.template Reference

| Variable                                                        | Required | Description                                                                                           |
|-----------------------------------------------------------------|----------|-------------------------------------------------------------------------------------------------------|
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`     | ✅        | Main-agent chat model (default `deepseek-v4-flash`, OpenAI-compatible)                                |
| `DASHSCOPE_API_KEY`                                             | ✅        | Embedding (`text-embedding-v4`, used for memory-fragment vectors and FAQ retrieval) and vision models |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`           | Optional | If the chat model is routed through another OpenAI-compatible provider                                |
| `TAVILY_API_KEY`                                                | Optional | Web search (reserved)                                                                                 |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | Optional | LangSmith tracing                                                                                     |
| `EMAIL_AUTH_CODE`                                               | Optional | QQ mailbox auth code for alert emails (SMTP server & recipients are configured in config)             |

#### config.yaml.template Reference

Copy it to `config.yaml` and adjust as needed; the template keeps local dev defaults so a minimal config runs out of the
box. The file itself carries full inline comments; here we highlight **must-change items** and **memory-framework
parameters**:

**Must change**

- `AMQP.rabbitmq`: RabbitMQ credentials (redacted in the template)
- `databases.rag.milvus.conn_args.uri`: Milvus address (template uses localhost placeholders)
- `email.sender / receiver`: sender and recipients for alert emails

**★ Memory-framework parameters (the `model` section, fully commented in the template)**

| Parameter                                       | Location            | Purpose & advice                                                                                                                                                                                               |
|-------------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model.vocation.name`                           | Scenario preset     | The **scenario switch** of the memory framework — one of `customer service` (short memory, fast forgetting), `collaborative creation` (long memory, slow forgetting), `accompany` (compromise), or `customize` |
| `model.vocation.kwargs`                         | Custom tuning       | Required only in `customize` mode; structure is commented in the template: `M(Δt)` (maturity curve τ_m/c), `T(m)` (forgetting curve τ/c), `slice` (split trigger), `long-term` (induction trigger)             |
| `model.summary.name`                            | Split/summary model | The LLM that splits conversations by theme and distills user profiles; can differ from the chat model                                                                                                          |
| `model.summary.kwargs.profile.max_input_tokens` | Capacity baseline   | Max input tokens of that model; the baseline for `fraction` mode                                                                                                                                               |
| `model.summary.pattern`                         | Trigger mode        | `fraction` (ratio) / `tokens` (token count) / `messages` (message count)                                                                                                                                       |
| `model.summary.trigger_threshold`               | Trigger threshold   | fraction ≥ 0.7 advised; tokens ≥ 10000 advised; messages ≥ 50 advised                                                                                                                                          |

**Tuning advice**: the concrete values of the three presets are hard-coded in `_VOCATION_PARAM_MAP`
in `Tools/middleware/memory/time_memory.py` (also commented in the source). For daily use, just switch `vocation.name`.
Only when you want to fine-tune "how often to split / how fast to forget" for your own business should you
use `customize` mode with hand-written `kwargs`. See the Memory Framework section above for the formulas.

### 4. Start

```bash
python run.py
```

After startup, visit `http://127.0.0.1:8000`. API docs at `http://127.0.0.1:8000/docs` (requires a valid token).

### 5. Common Endpoints

| Method | Path                   | Description                                                |
|--------|------------------------|------------------------------------------------------------|
| GET    | `/ai/health`           | Health check (**no auth**, used by container probes)       |
| POST   | `/ai/chat`             | User chat (SSE streaming)                                  |
| POST   | `/ai/human/end`        | End human-agent service (AI relays the closing words)      |
| GET    | `/ai/history`          | Conversation history (for the human agent before takeover) |
| POST   | `/files/upload`        | File upload                                                |
| GET    | `/manager/get/all/faq` | Get FAQ list                                               |

> Except for `/ai/health`, every endpoint goes through the JWT middleware in `main.py` and requires
> `Authorization: Bearer <token>`.

### 6. Containerized Deployment

Pick by scenario: **run the image directly** (6.2 — lightest when Redis / Milvus etc. already exist),
**Docker Compose** (6.3 — full stack on a single host, good for local verification and demos), or
**Kubernetes** (6.4 — the `k8s/` directory, organized with kustomize).

#### 6.1 Building the Image

```bash
# Build from the repository root (uses ./Dockerfile and ./requirements.txt)
docker build -t intelligent-customer-service:latest .

# Inspect the result
docker images intelligent-customer-service
```

What to know about the image:

- **Multi-stage build**: the `builder` stage installs build deps and Python packages; the `runtime` stage
  only copies `/root/.local`, so no toolchain ships in the final image.
- **The first build is slow** (installs `torch`, `transformers`, ...); later builds hit the layer cache.
- **The build context excludes secrets and data**: `.dockerignore` drops `.env`, `config.yaml`, `data/`,
  `database/`, and `static/` (169MB of frontend assets), so the image ships **no** config file and no API
  keys — everything is injected at runtime.
- **Two model files must be in place before building** (see "Model Files" above): `model/roberta_inj.onnx`
  (~400MB, injection detection) and `model/hfl/` (tokenizer cache). Neither is committed; without them the
  build succeeds but the app fails at runtime with `FileNotFoundError`.
- The image bakes in `HF_ENDPOINT=https://hf-mirror.com` and `HF_HOME=/app/huggingface_cache` (the
  container entrypoint is uvicorn, not `run.py`, so the `huggingface` section of `config.yaml` never
  applies). Override with `-e HF_ENDPOINT=https://huggingface.co` if you are outside China.

Push to a registry (needed for multi-node Kubernetes):

```bash
docker tag intelligent-customer-service:latest <registry>/intelligent-customer-service:v1.0.0
docker push <registry>/intelligent-customer-service:v1.0.0
# then point k8s/app.yaml's image at that address
```

#### 6.2 Running the Container Directly (docker run)

For when Redis / PostgreSQL / MySQL / RabbitMQ / Milvus / Elasticsearch already exist and you only want
to run the app.

```bash
# 1) Environment variables: put real API keys in .env (not committed)
cp .env.template .env

# 2) Run (in PowerShell use ${PWD} for the path)
docker run -d --name intelligent-app \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/config.docker.yaml:/app/config.yaml:ro" \
  -v app-uploads:/app/data \
  -v app-database:/app/database \
  intelligent-customer-service:latest

# 3) Health check (no auth)
curl http://localhost:8000/ai/health
```

Paths the container mounts / writes to:

| Container path           | Contents                                     | Required?                                                 |
|--------------------------|----------------------------------------------|-----------------------------------------------------------|
| `/app/config.yaml`       | App config (mount `config.docker.yaml`)      | **Required** — starts with `FileNotFoundError` without it |
| `/app/data`              | Uploaded / downloaded files                  | Persist recommended                                       |
| `/app/database`          | SQLite memory store (`test1.db`)             | Persist recommended                                       |
| `/app/cache`             | LangGraph SQLite cache                       | Optional — losing it only drops the cache                 |
| `/app/log`               | Logs                                         | Optional                                                  |
| `/app/huggingface_cache` | HF model cache (image points `HF_HOME` here) | Optional — speeds up cold starts                          |

> **Networking caveat**: `config.docker.yaml` uses compose service names as dependency hosts
> (`redis` / `postgres` / `mysql` / `rabbitmq` / `milvus` / `elasticsearch`), which a standalone
> `docker run` container cannot resolve. Two options:
> (1) join the network compose created — find the real name with `docker network ls` (compose prefixes it
> with the project name), then add `--network <that name>`;
> (2) copy the config and change the hosts to your host machine (use `host.docker.internal` on Docker
> Desktop), making sure the dependency ports are published (6379 / 5432 / 3306 / 5672 / 19530 / 9200).

#### 6.3 Docker Compose

```bash
# 1) Environment variables: put real API keys in .env (not committed)
cp .env.template .env

# 2) Bring up the full stack: app + redis / postgres / rabbitmq / mysql / milvus(+etcd+minio) / elasticsearch
docker compose up -d

# 3) Check status and logs
docker compose ps
docker compose logs -f app

# 4) Health check (no auth)
curl http://localhost:8000/ai/health
```

- The app is configured by `config.docker.yaml` (dependency hosts already point at the compose service
  names), mounted as the in-container `config.yaml` — your local `config.yaml` is left untouched.
- Every dependency has a healthcheck; the app waits for them via `depends_on: service_healthy`.
- The first run builds the image (installs `torch`, `transformers`, etc.) and takes a while; later runs hit
  the layer cache.
- Allocate **≥ 8GB** to Docker: Milvus + Elasticsearch + MySQL + PostgreSQL all run at once.

Common maintenance commands:

| Task                                      | Command                        |
|-------------------------------------------|--------------------------------|
| Build / rebuild the app image             | `docker compose build app`     |
| Restart only the app                      | `docker compose restart app`   |
| Follow logs                               | `docker compose logs -f app`   |
| Shell into the container                  | `docker compose exec app bash` |
| Stop and remove containers (keep volumes) | `docker compose down`          |
| Also remove the data volumes              | `docker compose down -v`       |

#### 6.4 Kubernetes

```bash
# 1) Replace the placeholders in secret.yaml with real values (API keys / DB passwords)
#    base64 helper: echo -n "your-value" | base64

# 2) Deploy everything (kustomize)
kubectl apply -k k8s/

# 3) Check status
kubectl get all -n intelligent-cs
kubectl logs -f deploy/intelligent-cs-app -n intelligent-cs

# 4) Reach it from outside the cluster (port-forward, when Ingress is not ready)
kubectl port-forward -n intelligent-cs svc/intelligent-cs-service 8000:8000
curl http://localhost:8000/ai/health
```

| File                 | Contents                                                                              |
|----------------------|---------------------------------------------------------------------------------------|
| `k8s/namespace.yaml` | Namespace `intelligent-cs`                                                            |
| `k8s/pvc.yaml`       | 7 PVCs (postgres / mysql / redis / rabbitmq / app / milvus / es)                      |
| `k8s/secret.yaml`    | API keys and dependency passwords (**replace placeholders before deploying**)         |
| `k8s/configmap.yaml` | The app's `config.yaml` (services addressed via in-cluster DNS)                       |
| `k8s/*.yaml`         | Deployment + Service for redis / postgres / rabbitmq / mysql / milvus / elasticsearch |
| `k8s/app.yaml`       | App Deployment + Service (initContainer for dirs, three probe types)                  |
| `k8s/ingress.yaml`   | External entry point (requires an Ingress Controller; includes SSE passthrough)       |

**First, get the image into the cluster** — pick one:

```bash
# Option A: push to a registry, nodes pull it themselves (production)
#           point k8s/app.yaml's image at <registry>/intelligent-customer-service:v1.0.0;
#           private registries also need imagePullSecrets
docker push <registry>/intelligent-customer-service:v1.0.0

# Option B: kind (most common for local dev)
kind load docker-image intelligent-customer-service:latest

# Option C: minikube
minikube image load intelligent-customer-service:latest
```

> `k8s/app.yaml` uses `intelligent-customer-service:latest` with `imagePullPolicy: IfNotPresent`
> precisely to support the local image loading in options B/C (switching it to `Always` makes nodes try
> the registry and fail with `ImagePullBackOff`). For registry-based deploys (option A), prefer versioned
> tags so nodes do not reuse a stale `latest` cache.

Common maintenance commands:

| Task                           | Command                                                                |
|--------------------------------|------------------------------------------------------------------------|
| All resource status            | `kubectl get all -n intelligent-cs`                                    |
| App logs                       | `kubectl logs -f deploy/intelligent-cs-app -n intelligent-cs`          |
| Shell into the container       | `kubectl exec -it -n intelligent-cs deploy/intelligent-cs-app -- bash` |
| Rolling restart                | `kubectl rollout restart deploy/intelligent-cs-app -n intelligent-cs`  |
| Why a Pod will not start       | `kubectl describe pod -n intelligent-cs -l component=api`              |
| Delete everything (keeps PVCs) | `kubectl delete -k k8s/`                                               |

**Constraints and pitfalls baked into these manifests — read before changing them:**

- **The app runs a single replica** (`replicas: 1`; the HPA is commented out). `app-data-pvc` is
  `ReadWriteOnce`, so a second replica on another node cannot mount it; each replica also runs its own
  `KeyManage` JWT-key rotation, and concurrent rotation overwrites each other and invalidates tokens; the
  conversation cache is per-Pod local SQLite and is not shared. Scaling out requires fixing all three
  first — see the comments in `k8s/app.yaml`.
- **Passwords must be written in two places**: `configmap.yaml` (for the app to connect) and `secret.yaml`
  (for the infrastructure Pods to start). `config.py` only reads YAML, and its `${path:default}`
  placeholders are config-tree-internal references that cannot read environment variables, so a Secret
  alone cannot inject them. A mismatch shows up as "all Pods healthy but the app cannot reach the DB".
- **The app's `database` subdirectory is created by an initContainer**: it does not exist on a fresh PVC,
  and the main container's `subPath` mount would otherwise fail with `CreateContainerConfigError`.
- **Elasticsearch is a hard dependency** and is included; it also ships a privileged initContainer that
  raises `vm.max_map_count`, without which ES fails to start.
- **The HuggingFace cache is an emptyDir**: a rebuilt Pod re-downloads the tokenizer. Switch it to a PVC or
  bake the model into the image if startup time matters.
- **Image tag and pull policy must match**: currently `latest` + `IfNotPresent`, which pairs with the local
  image loading in kind / minikube. Switch to a versioned tag (e.g. `:v1.0.0`) when deploying from a
  registry.

## Relationship with Firefly Mall

This project is a **subsystem of Firefly Mall**, deployed independently from the mall's main service. It collaborates
with the mall's user system (JWT) and business data (products / orders / invoices) over HTTP. The frontend (Vue 3) will
not be placed in this warehouse temporarily.

## License

[MIT](LICENSE)
