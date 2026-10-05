# RAG Agent — 智能知识库问答平台


面向「Agent 后端」方向校招的项目一：一个生产级 RAG（检索增强生成）问答后端。
用户上传文档，系统完成 **解析 → 分块 → 向量化 → 混合检索 → 大模型生成（带引用）** 全流程，
并覆盖传统后端的核心能力：认证鉴权、多租户隔离、异步任务、缓存、限流、可观测、容器化部署。

> 核心代码自研（分块器、混合检索、RRF 融合、Agent 式引用回答），不依赖 LangChain/LlamaIndex，
> 便于在面试中讲清每一步的原理。

---

## ✨ 功能一览

| 模块 | 能力 |
|---|---|
| 用户认证 | 注册 / 登录（JWT），bcrypt 密码哈希，OAuth2 兼容 Swagger Authorize |
| 多租户知识库 | 每个用户独立知识库，资源归属校验（越权访问返回 404） |
| 文档管理 | 上传（txt/md/pdf，≤10MB）→ 状态机（pending→processing→ready/failed）→ 删除 |
| 文档摄入 | PDF 解析（pypdf）、段落感知分块（chunk_size/overlap 可配）、批量向量化 |
| RAG 检索 | **向量检索 + 关键词检索 + RRF 融合初筛 + rerank 精排**（pgvector HNSW 索引；rerank 支持 RRF/LLM 两种模式） |
| 智能问答 | 基于引用的回答（[1][2] 标注），引用溯源，无答案时拒答不编造 |
| 流式输出 | SSE 流式问答（sources → delta* → done 事件流） |
| 热点缓存 | 相同问题 1 小时缓存（Redis），命中直接返回 |
| 限流 | 滑动窗口限流（Redis INCR + EXPIRE，每用户每分钟 N 次） |
| 异步任务 | Celery（Redis broker）处理文档摄入，失败自动退避重试（幂等：重复处理会先清旧块） |
| 健康检查 | /health 汇总 DB / Redis / 组件状态 |
| 部署 | docker-compose 一键启动（postgres+pgvector / redis / minio / api / worker） |

## 🏗️ 架构

```text
┌─────────────┐   HTTP/SSE   ┌──────────────────────────────┐
│  Web 前端     │ ──────────▶ │  FastAPI (api)                │
└─────────────┘              │  ├─ /auth      认证/JWT        │
                             │  ├─ /knowledge-bases  多租户 KB │
                             │  ├─ /documents    上传/状态    │
                             │  ├─ /conversations 问答/SSE   │
                             │  └─ /health       探活        │
                             └───────┬──────────────┬────────┘
                                     │              │
                          ┌──────────▼───┐   ┌─────▼─────┐
                          │ PostgreSQL   │   │  Redis    │
                          │ + pgvector   │   │ 缓存/限流  │
                          │ (chunks 向量) │   │ Celery broker│
                          └──────────────┘   └─────┬─────┘
                                                   │ 任务
                                          ┌────────▼────────┐
                                          │ Celery worker   │
                                          │ 解析→分块→向量化 │
                                          └─────────────────┘
                                   ┌─────────────────────────┐
                                   │ 对象存储（MinIO/S3 或本地）│
                                   └─────────────────────────┘
```

## 🧱 技术栈

- **Python 3.12+ / FastAPI / Pydantic v2 / SQLAlchemy 2.0（异步）**
- **PostgreSQL + pgvector**（HNSW 向量索引）
- **Redis**（缓存 / 限流 / Celery broker）
- **Celery**（异步任务，acks_late + 自动重试）
- **httpx**（OpenAI 兼容 LLM / Embedding 客户端）
- **MinIO / S3**（对象存储，本地调试可用 local 后端）
- **Docker + docker-compose**、pytest、ruff

## 🚀 快速开始

### 方式一：Docker Compose（推荐，一键全栈）

```bash
# 1. 准备环境变量
cp .env.example .env
# 编辑 .env：
#   - 填入真实 LLM_API_KEY / EMBEDDING_API_KEY（OpenAI 兼容均可，如 DeepSeek/通义）
#   - 修改 JWT_SECRET 为随机长字符串
#   - 确保 CELERY_TASK_ALWAYS_EAGER=false（生产走 Celery worker）

# 2. 一键启动（自动完成：建表 + pgvector 扩展 + HNSW 索引 + MinIO 桶初始化）
docker compose up -d --build

# 3. 查看状态与日志
docker compose ps              # 全部 healthy 即成功
docker compose logs -f api     # API 日志
docker compose logs -f worker  # 文档处理任务日志

# 4. 访问
open http://localhost:8000/docs

# 停止 / 清理（保留数据卷）
docker compose down
# 彻底清理（删数据卷，慎用）
docker compose down -v
```

> 说明：compose 里的 `init` 与 `minio-init` 是一次性初始化服务，
> 首次启动自动执行（幂等），完成后退出；api/worker 等待其成功后才启动。
> 数据库/对象存储数据保存在命名卷 `pgdata` / `miniodata`，`docker compose down` 不会丢失。

### 方式二：本机开发（无 Docker）

需要：Python 3.12+、PostgreSQL（启用 pgvector）、Redis。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# 本地无 API key 调试：LLM_MODE=fake、EMBEDDING_MODE=fake（确定性伪向量，可跑通全链路）
# 本地无 Redis 调试：CELERY_TASK_ALWAYS_EAGER=true（文档处理同步执行）

python -m app.db.init_db          # 建表 + 扩展 + 索引
uvicorn app.main:app --reload     # 启动 API
celery -A app.workers.celery_app worker -l info   # 启动 worker（eager 模式可跳过）
```

### 最小演示（fake 模式，无需任何 API key）

```bash
# 注册 → 登录 → 建知识库 → 上传文档 → 提问
curl -X POST localhost:8000/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"a@b.com","password":"password123","name":"demo"}'

TOKEN=$(curl -X POST localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=a@b.com&password=password123' | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

KB_ID=$(curl -X POST localhost:8000/api/v1/knowledge-bases \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"demo"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

printf '弹性打卡时间九点到十点。' > doc.txt
curl -X POST localhost:8000/api/v1/knowledge-bases/$KB_ID/documents \
  -H "Authorization: Bearer $TOKEN" -F file=@doc.txt

CONV_ID=$(curl -X POST localhost:8000/api/v1/conversations \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"knowledge_base_id\":$KB_ID}" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

curl -N -X POST localhost:8000/api/v1/conversations/$CONV_ID/ask-stream \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question":"几点上班？"}'   # SSE 流式
```

## 📚 API 一览（前缀 /api/v1）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /auth/register | 注册 |
| POST | /auth/login | 登录（OAuth2 form），返回 JWT |
| GET/POST | /knowledge-bases | 知识库列表 / 创建 |
| GET/PATCH/DELETE | /knowledge-bases/{id} | 详情 / 更新 / 删除 |
| GET/POST | /knowledge-bases/{id}/documents | 文档列表 / 上传 |
| GET/DELETE | /documents/{id} | 文档详情 / 删除 |
| GET | /documents/{id}/status | 处理状态 |
| POST | /conversations | 创建会话 |
| GET | /conversations | 会话列表 |
| GET | /conversations/{id}/messages | 消息历史 |
| POST | /conversations/{id}/ask | 非流式问答 |
| POST | /conversations/{id}/ask-stream | **SSE 流式问答** |
| GET | /health | 健康检查 |

交互式文档：`/docs`（Swagger，支持 Authorize 按钮直接登录）。

## 🔬 RAG 链路设计（面试重点）

### 摄入流水线（Celery 异步）

```text
上传 → 存对象存储 → 建 Document(pending) → Celery 任务
     → 下载 → 解析(pdf/txt/md) → 段落感知分块
     → 批量 Embedding → 写 Chunk(含向量) → Document(ready)
```

**分块策略**：先按空行切段落，尽量在段落边界合并（保留语义完整性）；
超长段落按 chunk_size 硬切并保留 overlap 字符衔接上下文。
（可扩展：按 Markdown 标题分块、语义分块。）

### 混合检索 + RRF 融合

```text
         ┌─ 向量检索（pgvector 余弦距离，HNSW 索引）─ 取 top_k×2
问题 ────┤
         └─ 关键词检索（ILIKE 命中查询词）────────── 取 top_k×2
                    │
                    ▼
        RRF 融合：score = Σ 1/(k + rank)，取最终 top_k
                    │
                    ▼
          LLM 生成（系统提示注入片段 + 引用编号）
```

为什么混合检索？纯向量检索对**专有名词 / 编号 / 精确短语**不敏感；
纯关键词检索没有语义能力。两者融合 + RRF（不依赖分数可比性，只依赖排序）
是工程上最稳健的组合。

**两阶段精排**（`RERANK_MODE` 可切换）：
1. **RRF 融合初筛**：向量 top_k×2 + 关键词 top_k×2 → 取候选 top_k×2（保证召回）；
2. **rerank 精排**：
   - `rrf`（默认）：直接按融合分数排序，零成本；
   - `llm`：让 LLM 对"问题 + 片段"逐条打 0-10 相关性分再排序（更准、耗 token）；
   - 接口预留 `RerankScorer` 回调，可无缝替换为 bge-reranker 等交叉编码器模型。

### 引用溯源与防幻觉

- 系统提示强制：**仅依据片段回答，无答案时明确拒答，不编造**；
- 回答中用 [1][2] 标注引用，响应携带 sources 元数据（文档名、片段、相似度）；
- 可扩展：置信度阈值（检索分数过低直接拒答）、rerank 模型、RAG 评测集（ragas）。

## 🧪 测试

```bash
# 需要本机 PostgreSQL（含 pgvector）+ Redis；测试自动创建 rag_agent_test 库
PYTHONPATH=.deps python3 -m pytest -v
```

覆盖：分块器、密码/JWT、注册登录、知识库多租户隔离、文档摄入、缓存命中、SSE 流式。

## 📊 评测（evals）

检索质量不靠"感觉还行"，靠固定的语料 + 查询集出一张可对比的表：

- 语料 12 篇 / 查询 24 条（18 条可回答 + 6 条不可回答），跑在真实 PostgreSQL + pgvector 上；
- 指标：Recall@1/@3/@5、MRR@5、nDCG@5（文档级）、不可回答问题的零命中率、引用格式覆盖率、p50/p95 延迟；
- 通道消融：仅关键词通道 vs 混合检索（向量 + 关键词 + RRF + 精排）。

```bash
docker compose up -d postgres
TEST_DB_PORT=5432 TEST_DB_PASSWORD=rag PYTHONPATH=.deps python3 -m evals.run_evals --check --report evals/report.md
```

本机离线实测（fake embedding，向量无语义，只反映管道行为）：

| 通道 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 仅关键词通道 | 0.889 | 1.000 | 1.000 | 0.935 | 0.952 | 1.8 | 2.6 |
| 混合检索 | 0.778 | 0.778 | 0.778 | 0.778 | 0.778 | 8.4 | 49.3 |

评测还暴露了一个真实问题：**不可回答的问题里，混合通道几乎总会返回候选片段（零命中率 0.000）**，
拒答完全依赖提示词约束 —— 应该补一个相关性阈值或"检索为空即拒答"的短路分支。
配好 `EMBEDDING_API_KEY` 后用 `EMBEDDING_MODE=real` 复跑同一套评测，即可得到真实模型下的检索质量。
细节见 [`evals/README.md`](evals/README.md)。

## 📁 项目结构

```text
rag-agent/
├── app/
│   ├── main.py                 # FastAPI 入口（路由注册 / CORS / lifespan）
│   ├── config.py               # pydantic-settings 配置（.env）
│   ├── api/routes/             # 路由层：auth / knowledge_bases / documents / conversations / health
│   ├── core/                   # 公共能力：security(JWT/bcrypt) / deps(依赖) / redis / storage
│   ├── db/                     # session / base / models（user/kb/document/chunk/conversation）
│   ├── schemas/                # Pydantic 请求/响应模型
│   ├── services/               # 业务层：auth / kb / ingest / embedding / llm / rag / chat
│   └── workers/                # Celery 应用与任务（文档摄入）
├── tests/                      # pytest（单元 + API 集成）
├── evals/                      # 检索质量评测（golden 语料 + 查询集 + 报告）
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

## 🔭 可扩展方向（简历加分）

- [ ] 接入 rerank 模型（bge-reranker）替代/叠加 RRF
- [ ] RAG 质量评测集（召回率 / 忠实度 / 答案相关性）
- [ ] 文档增量更新与自动重新向量化（webhook / 定时任务）
- [ ] 多文档混合引用、追问改写（query rewriting）
- [ ] Prometheus + Grafana 指标面板（检索延迟、token 消耗、缓存命中率）
- [ ] 前端演示界面（聊天 + 引用卡片 + 知识库管理）

## 📄 License

MIT
