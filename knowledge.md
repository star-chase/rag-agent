# RAG Agent 技术知识手册 & 面试考点

> 配套项目：`rag-agent`（FastAPI + PostgreSQL/pgvector + Redis + Celery 的 RAG 问答后端）
>
> **怎么用这份文档**
> - **A 部分**：原理补课。每节末尾的「本项目落点」给出真实代码位置（文件:函数），把原理和实现对上号。
> - **B 部分**：面试问答与深挖，含答题骨架、追问预判、以及本仓库 **5 个真实缺陷**（可当"你项目有什么不足"的标准答案）。
> - **C 部分**：速查表（参数/复杂度/术语/数字锚点）与复习清单。
>
> 文中所有数值（800/100、top_k=5、k=60、1024 维、0.778 等）均来自本仓库实际代码与实测报告
> （`app/config.py`、`evals/report.md`），不是"一般情况下的经验值"——面试时能落到具体数字是加分项。
> [B3](#b3-深挖本项目-5-个真实缺陷与修复方案) 的 5 个缺陷**全部在真实代码上复现过**，附命令与实测输出。

---

## 目录

- [A 部分：技术知识体系](#a-部分技术知识体系)
  - [A1. RAG 为什么存在：参数化记忆 vs 检索增强](#a1-rag-为什么存在参数化记忆-vs-检索增强)
  - [A2. 文档解析（Parsing）](#a2-文档解析parsing)
  - [A3. 分块（Chunking）](#a3-分块chunking)
  - [A4. Embedding 与向量数学](#a4-embedding-与向量数学)
  - [A5. 向量数据库与 ANN 索引](#a5-向量数据库与-ann-索引)
  - [A6. 关键词检索与稀疏检索](#a6-关键词检索与稀疏检索)
  - [A7. 混合检索与 RRF 融合](#a7-混合检索与-rrf-融合)
  - [A8. 重排（Rerank）](#a8-重排rerank)
  - [A9. 生成、Prompt 与防幻觉](#a9-生成prompt-与防幻觉)
  - [A10. 评测（Evals）](#a10-评测evals)
  - [A11. 后端工程化考点](#a11-后端工程化考点)
  - [A12. MCP 协议考点](#a12-mcp-协议考点)
- [B 部分：面试题库与深挖](#b-部分面试题库与深挖)
  - [B1. 一分钟项目自述模板](#b1-一分钟项目自述模板)
  - [B2. 高频问答 20 题](#b2-高频问答-20-题)
  - [B3. 深挖：本项目 5 个真实缺陷与修复方案](#b3-深挖本项目-5-个真实缺陷与修复方案)
  - [B4. 手写题参考实现](#b4-手写题参考实现)
- [C 部分：速查与清单](#c-部分速查与清单)

---

# A 部分：技术知识体系

## A1. RAG 为什么存在：参数化记忆 vs 检索增强

大模型的知识有两类载体：

| 载体 | 说明 | 更新成本 | 可溯源 |
|---|---|---|---|
| **参数化记忆**（权重） | 预训练/微调时写进参数的知识 | 重训或微调，天级+ | 不可溯源 |
| **非参数化记忆**（外部库） | 推理时检索回来的文本 | 秒级（写库即生效） | 可标注来源 |

LLM 幻觉的工程根因可以拆成三条：**知识缺失**（训练时没见过）、**知识过期**（见过但是旧版本）、
**无约束生成**（模型倾向"编一个像样的答案"而不是说"我不知道"）。
RAG 用检索结果作为**条件证据**压制第三条，用外部库解决前两条——因此它不是"让模型更聪明"，
而是"把模型的知识边界从权重挪到可运维的数据系统里"。

**RAG vs 微调 vs 长上下文**（面试常考的选型题）：

| 方案 | 适合 | 不适合 | 代价 |
|---|---|---|---|
| RAG | 私有知识、频繁更新、要引用溯源 | 需要改变模型**行为风格/格式** | 需要检索系统运维 |
| 微调（SFT/LoRA） | 固定输出风格、领域黑话、指令遵循 | 装事实（易过拟合、难更新、仍会编） | 数据标注 + 训练 + 版本管理 |
| 长上下文 | 单篇长文档问答、一次性分析 | 海量库、高 QPS（token 成本 ×N） | 成本高、有 lost-in-the-middle |

> 工程结论：**更新知识用 RAG，改变行为用微调，两者不互斥**。

### 本项目落点

链路（`README.md` 的"摄入流水线"）：

```text
上传 → 对象存储 → Document(pending) → Celery 任务
     → 下载 → 解析(pdf/txt/md) → 段落感知分块
     → 批量 Embedding → 写 Chunk(含向量) → Document(ready)
问答：问题 → 向量检索 + 关键词检索 → RRF 融合 → rerank 精排 → Prompt 拼片段 → LLM → 带 [1][2] 引用回答
```

| 环节 | 代码位置 |
|---|---|
| 上传/状态查询 | `app/api/routes/documents.py` |
| 摄入流水线 | `app/services/ingest_service.py::process_document` |
| 分块 | `app/services/ingest_service.py::chunk_text` |
| 向量化 | `app/services/embedding_service.py` |
| 混合检索 + RRF | `app/services/rag_service.py::search_knowledge_base` / `_rrf_fuse` |
| 精排 | `app/services/rerank_service.py::rerank` |
| Prompt 构造 | `app/services/rag_service.py::build_messages` |
| 生成/流式 | `app/services/llm_service.py`、`app/services/chat_service.py` |

---

## A2. 文档解析（Parsing）

解析质量是 RAG 的**天花板**：后面的分块、检索、生成都只能做"信息不增不减"的搬运，
解析阶段丢掉的表格结构、页眉页脚噪声、阅读顺序错乱，后面无法补救。

PDF 要分三类对待：

| 类型 | 特征 | 方案 |
|---|---|---|
| 文本层 PDF | 能选中文字 | `pypdf` / `pdfplumber` / `PyMuPDF` 抽文本 |
| 扫描件/图片 PDF | 抽出来是空串 | OCR（PaddleOCR / Tesseract），需版面分析 |
| 复杂版面 | 双栏、表格、公式 | 版面模型（unstructured / MinerU / marker）+ 表格转 Markdown |

其他格式：DOCX（`python-docx`）、HTML（`trafilatura` 去导航噪声）、Markdown（直接保留标题层级）。

**常见坑**：双栏 PDF 被按行拼接成语义错乱的段落；表格抽成一行行数字丢失列关系；
页眉页脚/页码重复进入每个 chunk 污染向量；CJK 断行不补空格；`extract_text()` 返回 `None` 未判空。

### 本项目落点

`app/services/ingest_service.py::parse_document` / `_parse_pdf`：

- txt/md 直接 `decode("utf-8", errors="replace")`；
- pdf 用 `pypdf` 逐页 `extract_text()`，拼接时插入 `--- page N ---` 分隔，**保留页码线索**（可用于后续引用定位）；
- 解析结果为空会抛错 → 文档进入 `failed` 状态（避免"空文档静默入库"）。

已知局限（面试可主动说）：无 OCR、无版面分析、无表格结构化、页码标记会混进 chunk 正文。

---

## A3. 分块（Chunking）

### 为什么要分块

1. **模型输入上限**：embedding 模型有最大 token 数（常见 512/8192）；
2. **检索粒度**：块太大会稀释语义（一个块讲五件事，向量是"平均语义"，与任何具体问题都不像）；
   块太小会丢上下文（"他于次年离职"这种块无法独立理解）；
3. **上下文预算**：塞进 Prompt 的片段数 × 块长度 = token 成本与延迟。

### 策略谱系（从简到繁）

| 策略 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| 固定窗口 | 每 N 字符/ token 切一刀 | 简单、可控 | 切断句子/语义 |
| **段落感知** | 按空行切段，再在段落边界合并到接近 N | 保语义完整 | 依赖段落格式 |
| 递归字符 | 按 `\n\n → \n → 句号 → 空格` 逐级回退切 | 通用、平衡 | 无结构感知 |
| 标题层级 | 按 Markdown 标题切，带父标题前缀 | 结构清晰、可做父子块 | 需结构化文档 |
| 语义分块 | 相邻句向量相似度骤降处切 | 语义边界准 | 需额外计算、不稳定 |
| 父子块 / small-to-big | 小块检索、大块喂 LLM | 检索准 + 上下文全 | 存储与实现复杂 |
| 句窗检索 | 命中句 ± 窗口句一起给 LLM | 精准 | 实现复杂 |

**chunk_size 怎么定**（高频题）：没有普适最优，按"一个问题需要多大上下文才能被回答"来定。
经验区间 256–1024 字符；本项目取 **800 字符**——中文信息密度高，800 字符约 400–500 字，
大致等价于 2–3 个自然段，能覆盖一条制度条款而不混入太多主题。
定完必须用评测集验证（见 A10），而不是拍脑袋。

**overlap 的作用**：避免"答案正好被切在边界上"导致两边都不完整。
代价是重复向量化（本项目 100/800 = **12.5% 的额外 embedding 成本**，
块数越多、块越小，这个比例越高）。overlap 一般取 chunk_size 的 10%–20%。

### 本项目落点

`app/services/ingest_service.py::chunk_text`（参数 `CHUNK_SIZE=800`、`CHUNK_OVERLAP=100`）：

```python
text = re.sub(r"\n{3,}", "\n\n", text).strip()
paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
for para in paragraphs:
    if current and len(current) + len(para) + 2 <= chunk_size:  # 段落在边界内合并
        current = f"{current}\n\n{para}"; continue
    ...
    while len(para) > chunk_size:            # 超长段落硬切
        chunks.append(para[:chunk_size])
        para = para[chunk_size - overlap :]  # 保留 overlap 衔接上下文
```

三句话讲清楚：**先按空行切段 → 尽量按段落边界合并 → 超长段落才硬切并留 overlap**。
⚠️ 这段代码有两个边界问题（`chunk_overlap >= chunk_size` 时死循环/内容截断），
见 [B3 缺陷 4](#缺陷-4or-吞掉合法 0 值--chunk_overlap--chunk_size-不校验)。

---

## A4. Embedding 与向量数学

### 概念

Embedding 把文本映射到 d 维实向量，使**语义相近的文本在空间中距离近**。
它不是"哈希"，而是训练出来的连续表示（对比学习：正样本拉近、负样本推远）。
检索能力来自**相似度排序**，而不是向量本身可解释——单看某一维没有意义。

### 三种距离度量

| 度量 | 公式 | pgvector 操作符 | 何时用 |
|---|---|---|---|
| 余弦距离 | `1 - (a·b)/(‖a‖‖b‖)` | `<=>` | 文本检索默认（只关心方向） |
| 欧氏距离 | `‖a-b‖₂` | `<->` | 向量模长有含义时 |
| 负内积 | `-(a·b)` | `<#>` | 向量已归一化时（等价余弦，更快） |

**关键结论**：向量 L2 归一化后 `cosine(a,b) = a·b`，三种度量排序一致。

### 本项目落点

`app/services/embedding_service.py`：

- 真实模式：豆包 `doubao-embedding-vision-251215`，**1024 维**（`EMBEDDING_DIM`）。地址以 `/embeddings/multimodal` 结尾时，每段文本单独请求，`input` 是 `[{type: text, text}]`，`dimensions` 等于配置宽度。返回在 `data.embedding`。超时 connect 10s / read 60s，请求跟随系统代理；
- OpenAI 兼容地址仍按字符串数组分批，`EMBEDDING_BATCH_SIZE=16`，批间 `sleep(0.05)`。返回顺序不保证与输入一致，[`parse_embedding_batch`](app/services/embedding_service.py) 按 `index` 对齐，否则 chunk 与向量错位；
- fake 模式 `_fake_embedding`：`sha256(f"{text}:::{i}")` 取 4 字节 → `[0,1)` → 减 0.5 → L2 归一化。
  同一文本永远同一向量（可复现、可离线跑测试），但**向量之间无语义关系**——
  所以 fake 模式下的评测数字只反映"管道行为"，不能证明检索质量（见 A10）。

**换模型的代价**（面试常问）：embedding 模型决定向量空间，**换模型 = 全量重建**。
维度变化还要改表结构（例如 `vector(1536)` → `vector(1024)`）并重建 HNSW。旧向量宽度对不上，列改完只能清空再重嵌。
因此生产上要记录每个 chunk 的模型版本，支持灰度双写与回滚。

---

## A5. 向量数据库与 ANN 索引

### 精确检索 vs 近似检索（ANN）

- **精确（KNN）**：全表算距离排序，召回 100%，复杂度 O(N·d)；10 万块以内其实够用；
- **近似（ANN）**：用图/聚类结构跳着找，召回 95%+，延迟毫秒级；代价是**可能漏掉真正最近的邻居**。

工程上就是"**召回率 ↔ 延迟 ↔ 内存**"三角，没有免费午餐。

### HNSW（Hierarchical Navigable Small World）

思想 = **跳表 + 近邻图**：多层图，上层稀疏用于长距离跳跃，下层稠密用于精细收敛。
查询从顶层入口点开始贪心下降，每层找局部最近邻，逐层细化。
复杂度约 **O(log N)**，召回高、增量插入友好，代价是**内存占用大、构建慢**。

参数（pgvector 默认 `m=16`、`ef_construction=64`、`hnsw.ef_search=40`）：

| 参数 | 含义 | 调大 → |
|---|---|---|
| `m` | 每节点最大连接数 | 召回↑、内存↑、构建慢 |
| `ef_construction` | 建索引时的候选队列长度 | 索引质量↑、构建更慢 |
| `hnsw.ef_search` | **查询**时的候选队列长度（运行时 `SET`） | 召回↑、延迟↑ |

> `ef_search` 是唯一可以在**不重建索引**的情况下调优召回/延迟的点，是线上调参首选。

### IVFFlat

思想 = **倒排 + 聚类**：先用 k-means 把向量分到 `lists` 个桶，查询时只扫最近的 `probes` 个桶。
建索引快、内存省，但**需要先有数据再建**（聚类要数据分布），召回对 `probes` 敏感，
且对增量数据不如 HNSW。`probes = sqrt(lists)` 是常见起点。

**选型口诀**：数据要频繁增删 + 内存够 → HNSW；静态大库 + 想省内存/构建快 → IVFFlat。

### 必须记住的坑

**操作符必须与索引 opclass 匹配，否则索引静默失效（退化为顺序扫描）**：

| 查询写法 | 需要的索引 opclass |
|---|---|
| `ORDER BY embedding <=> $1` | `vector_cosine_ops` |
| `ORDER BY embedding <-> $1` | `vector_l2_ops` |
| `ORDER BY embedding <#> $1` | `vector_ip_ops` |

pgvector 的维度限制：`vector` 类型存储最多 **16000 维**，但 **HNSW/IVFFlat 索引最多 2000 维**
（`halfvec` 可到 4000）。本项目用豆包的 1024 维，列是 `vector(1024)`，索引是 `vector_cosine_ops`。同一模型也支持 2048，但超过 2000，HNSW 只能改 `halfvec`。

### pgvector vs 专用向量库

| 方案 | 优势 | 劣势 | 适用 |
|---|---|---|---|
| **pgvector** | 与业务数据同库同事务、SQL 过滤/join、少一个组件 | 超大规模（亿级）性能与水平扩展弱；标量过滤 + ANN 需自己调 | 中小规模、多租户业务库（本项目） |
| FAISS | 单机性能极强、索引类型最全 | 库不是服务：无并发写/权限/持久化，需自己包 | 离线批处理、研究 |
| Milvus / Qdrant / Weaviate | 分布式、原生过滤、多租户、量化 | 多一套运维与一致性成本 | 大规模生产 |

面试点：**"为什么不用专用向量库"** 的正确答法是讲**一致性与运维成本**——
知识库/文档/chunk 在一个事务里，删除知识库靠 `ON DELETE CASCADE` 就干净了；
引入外部向量库要处理"关系库删了、向量库没删"的双写不一致（需要 outbox/补偿任务）。

### 本项目落点

`app/db/init_db.py`：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_kb_id ON chunks (knowledge_base_id);
```

`app/services/rag_service.py::_vector_search`：

```python
select(Chunk).where(Chunk.knowledge_base_id == kb_id, Chunk.embedding.is_not(None))
            .order_by(Chunk.embedding.cosine_distance(query_vec).asc()).limit(limit)
```

**多租户 + ANN 的深层问题**（能答出来是分水岭）：
`kb_id` 过滤条件无法"下推"进 HNSW 图遍历——pgvector 会先取近似候选再过滤，
于是**大库 + 小租户**场景可能召回不足（扫了 40 个邻居，全被 `kb_id` 过滤掉，结果为空）。
缓解手段：按 `kb_id` 做分区表/partial index、每个租户独立 collection、
或 pgvector 0.8+ 的 **iterative scan**（`SET hnsw.iterative_scan = relaxed_order`）。
本项目的规模下不成为问题，但**必须知道这个边界**（面试官很喜欢问"租户隔离会不会影响召回"）。

---

## A6. 关键词检索与稀疏检索

### 为什么还需要关键词检索

稠密向量擅长"语义相近"，但在以下场景失手：

- **专有名词/型号/编号**：`A-1024`、`ISO 27001`、员工工号——训练语料里罕见，向量表示不可靠；
- **精确短语**：法规条文、错误码、函数名；
- **长尾 OOV**：分词后几乎不出现的新词。

原因是 embedding 是**有损压缩**：低频实体的区分度被平均掉了。

### BM25 与倒排索引

关键词检索的标准答案是 **BM25**（TF-IDF 家族），核心两项修正：

```text
score(q,d) = Σ_t IDF(t) · (f(t,d)·(k1+1)) / (f(t,d) + k1·(1 - b + b·|d|/avgdl))

IDF(t) = ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 )
k1 ≈ 1.2（词频饱和）   b ≈ 0.75（长度归一化）
```

- **词频饱和**（`k1`）：一个词出现 10 次不该比出现 1 次重要 10 倍；
- **长度归一化**（`b`）：长文档天然命中更多词，要惩罚。

底层是**倒排索引**（term → posting list），查询是集合求交/求并 + 打分。

### 中文分词的坑

中文没有空格，**必须先分词**才有正确的 term：jieba（应用层）、
PostgreSQL 的 `zhparser`/`pg_jieba`（`to_tsvector('chinese', ...)`）、
或 n-gram（`pg_trgm` 的 trigram，无需词典）。

### 本项目落点与局限

`app/services/rag_service.py::_keyword_search`。Postgres 镜像在 `docker/postgres/Dockerfile` 里基于 `pgvector/pgvector:pg16` 编译了 `pg_jieba`。问句和正文都用 `to_tsvector('jiebacfg', ...)`。问句切出的词用 OR 拼成 `to_tsquery`，再按 `ts_rank` 排序。GIN 索引是 `idx_chunks_content_jieba`。

还要自己说清的局限：

1. **没有共同词的改写仍然召回不到**。「几点上班」和「九点到十点」分词后没有相同词元，这一路是 0 条，要靠向量通道。
2. **`jiebacfg` 的 `to_tsquery` 默认把词用 AND 连起来**。问句一长就会过严，所以代码先取出词元再拼 OR，最多 16 个。
3. **这不是 BM25**。`ts_rank` 只按词频和文档长度做一个相关度，专有名词靠「词在不在」。

`evals/report.md` 是这条链路改完之前跑的，关键词 Recall@5 = 1.000 不能当成现在的成绩。

---

## A7. 混合检索与 RRF 融合

### 为什么混合

| 通道 | 强项 | 弱项 |
|---|---|---|
| 稠密（向量） | 语义改写、同义表达、模糊提问 | 专名/编号/精确短语、低频词 |
| 稀疏（关键词/BM25） | 字面精确命中、OOV、可解释 | 无同义能力、词形不匹配就废 |

两者错误**不相关**，因此融合能同时提高召回与鲁棒性——这是"混合检索"的全部理由。

### 为什么不能直接加权求和：分数不可比

`cosine ∈ [-1, 1]`，BM25 是 `[0, +∞)` 且**分布随查询浮动**（同一查询里最高分可能是 8 也可能是 30）。
直接 `0.5·cos + 0.5·bm25` 等于让 BM25 主导；做 min-max 归一化又会引入新问题：
单查询内的极值不稳定（候选集一变，归一化结果就变），且**没有跨查询可比性**。

### RRF（Reciprocal Rank Fusion）

原论文 Cormack et al. 2009 的公式：

```text
RRF(d) = Σ_{r ∈ 通道} 1 / (k + rank_r(d))        常用 k = 60
```

**核心洞察：只用排名，不用分数**——彻底绕开"分数不可比"，零训练、零调参、
对各通道分数尺度免疫，还能自然处理"两路都命中"（分数相加 → 加权）。

`k` 的作用：`k` 越大，`1/(k+rank)` 对各排名越接近（曲线越平），
即**削弱头部优势、让中段候选有机会**；`k` 越小越"赢者通吃"，越依赖单通道排名。
`k=60` 是原论文在 TREC 数据上的经验值，本项目沿用。

**手算示例**（`k=60`）：某块在向量通道排第 1、关键词通道排第 2 →
`1/61 + 1/62 ≈ 0.01639 + 0.01613 = 0.03252`；只被向量通道命中的块（排第 1）= `0.01639`。
——**双通道命中 = 双倍权重**，这正是融合想表达的"多路一致性即置信度"。

### 本项目落点

`app/services/rag_service.py::search_knowledge_base`：

```python
vector_hits  = await _vector_search(db, kb_id, query_vec, limit=top_k * 2)   # top_k=5 → 取 10
keyword_hits = await _keyword_search(db, kb_id, query,    limit=top_k * 2)   # 取 10
fused    = _rrf_fuse(vector_hits, keyword_hits, top_k * 2)                    # 融合出 10
reranked = await rerank_service.rerank(query, fused, top_k)                   # 精排取 3（rerank_top_k）
```

为什么两路都取 `top_k*2`：**给融合留候选余量**——若每路只取 5，某块的"双通道命中"优势
可能因为它在某一路排第 6 而被截断，融合就丢掉了信号。融合后再精排收敛到 3 条进 Prompt。

`_rrf_fuse` 曾经把两路拼进同一次 `enumerate`，关键词名次被向量列表的长度推后。这一处已经按「每路从 rank=1 计分」修好，说明见 [B3 缺陷 1](#缺陷-1rrf-两路-rank-串扰融合被系统性偏向向量通道已修复)。

---

## A8. 重排（Rerank）

### 两阶段检索的范式

```text
粗排（召回）   ：海量 → 几十~几百条，要求高召回、低延迟、可扩展   → 向量/BM25/混合
精排（rerank）：几十条 → 3~5 条，要求高精度，可以慢、可以贵      → cross-encoder / LLM
```

**为什么不能直接上精排**：cross-encoder 要把 `(query, doc)` 拼在一起过一遍模型，
**无法预计算文档向量**，复杂度 O(候选数) 次前向；100 万块 = 100 万次前向，不可接受。

### bi-encoder vs cross-encoder

| | bi-encoder（embedding） | cross-encoder（reranker） |
|---|---|---|
| 编码方式 | query、doc **分别**编码 | `[query, doc]` **一起**编码 |
| 交互 | 无（靠向量空间对齐） | 全交叉注意力（token 级交互） |
| 预计算 | ✅ 文档向量可离线算好 | ❌ 每个 doc 都要现算 |
| 精度 | 中 | 高（显著优于 bi-encoder） |
| 延迟 | 毫秒（ANN） | ∝ 候选数（几十~几百 ms） |

代表模型：`bge-reranker-base/large`、`Cohere Rerank`、`jina-reranker`。

### 本项目落点

`app/services/rerank_service.py`，`RERANK_MODE` 二选一：

- `rrf`（默认）：`_rrf_fuse` 的融合分数直接当精排分（`rerank_score = item.score`），零额外成本；
- `llm`：一次请求给全部候选打 0–10 分，只接受长度刚好的 JSON 数组；解析失败打日志并保留 RRF 顺序。

回调签名是 `BatchScorer = Callable[[str, list[str]], Awaitable[list[float] | None]]`。换成交叉编码器时实现这一个函数即可。

默认 `rerank_mode` 是 `rrf`。`llm` 只在打开该模式时发生。以前的串行打分和「0-10 分」被解析成 0 已经去掉，见 B3 缺陷 5。精排收益仍要用评测集证明，现在的报告是修改前的。

---

## A9. 生成、Prompt 与防幻觉

### 上下文构造

`app/services/rag_service.py::build_messages`：

```python
for i, item in enumerate(chunks, start=1):
    context_parts.append(f"[{i}] (来源: {filename})\n{content}")

system = (
    "你是企业知识库智能助手。请仅依据下面提供的知识片段回答问题；"
    "如果片段中没有答案，请明确回答“知识库中没有相关信息”，不要编造。"
    "回答时在引用处用 [1][2] 标注对应片段编号，最后列出“参考资料”清单。"
    "========== 知识片段 ==========\n" + "\n\n".join(context_parts)
)
messages = [{"role": "system", "content": system}, *history[-6:], {"role": "user", "content": question}]
```

四个关键设计，都是可直接讲的点：

1. **编号 + 来源**：`[1] (来源: xxx.md)` 让模型能引用、前端能渲染引用卡片（`format_sources` 产出
   `chunk_id/document_id/filename/snippet/score` 元数据，前端点引用可溯源）；
2. **必须拒答的显式指令**："片段中没有答案就回答不知道"——对抗"编一个像样的答案"的默认倾向；
3. **历史截断**：只带最近 6 条（`get_history` 侧再取 10 条），控制 tokens 与"历史噪声污染当前检索"；
4. **片段放在 system 里**：与"指令"同段，遵循度通常高于塞进 user 消息。

### 幻觉的四个来源与对应手段（答题框架）

| 来源 | 现象 | 手段 |
|---|---|---|
| 检索没召回 | 库里明明有，答"没有相关信息" | 混合检索、rerank、查询改写（HyDE/多查询）、**评测召回率** |
| 召回不相关 | 拿着无关片段硬答 | **相关性阈值**、rerank、拒答短路 |
| 上下文过长/位置差 | 中间片段被忽略（lost-in-the-middle） | 压缩上下文、少而精（本项目只喂 3 条）、把最相关放头尾 |
| 模型不遵循指令 | 有片段仍编造 | 强约束 prompt、few-shot、引用校验、拒答兜底 |

**输出侧的可验证性**（进阶）：解析回答里的 `[n]`，校验 `n` 是否在 sources 范围内；
若回答带引用但引用率低/引用不存在，可打回重答或降级为"未找到依据"。
这是把"防幻觉"从 prompt 玄学变成**可度量指标**的关键一步（`evals` 里已经统计"引用格式覆盖率"）。

### 上下文预算（会算账很加分）

`top_k=5 → rerank_top_k=3`，每块 ≤800 字符 ≈ 2400 字符片段 ≈ 1.5–2k tokens（中文约 1 字 ≈ 1 token），
加系统提示与历史 6 条，单次请求约 3–5k 输入 tokens。
**这个数字直接决定成本与 p95 延迟**：块长翻倍，Prompt tokens 与 LLM 首 token 延迟都跟着涨。

### 本项目落点与已知缺口

实现在 `app/services/chat_service.py::ask` / `ask_stream`，回答与 sources 一起落库（`Message`）。
缺口：**检索没有相关性阈值**——只要库非空，混合检索几乎总会返回候选，
所以"拒答"完全依赖提示词（`evals/report.md` 实测不可回答问题零命中率 **0.000**）。
修复见 [B3 缺陷 3](#缺陷-3无相关性阈值不可回答问题的零命中率为-000)。

---

## A10. 评测（Evals）

### 为什么必须做

没有评测，一切"优化"都是不可证伪的玄学；有评测，才能回答"改了分块参数是变好还是变坏"。
评测集也是**回归测试**：防止一次 Prompt 改动把召回从 0.9 掉到 0.6 而无人察觉。

### 黄金集（Golden Set）怎么建

1. 按业务域选语料（本项目 12 篇制度文档：`evals/golden/corpus/*.md`）；
2. 写查询，**每条标注 gold 文档**（评测是文档级：命中 gold 文档即算命中）；
3. 覆盖三类：**可回答问题**（18 条）、**不可回答问题**（6 条，测拒答）、
   边界问题（跨文档、口语化改写、含专名/编号）；
4. 版本化：语料或查询改了要重新跑基线，报告存档（`evals/report.md`）。

### 检索指标定义（要能写公式）

| 指标 | 定义 | 说明 |
|---|---|---|
| **Recall@k** | 前 k 个结果里包含的 gold 数 / gold 总数 | 衡量"有没有捞到"，RAG 最关心 |
| **Precision@k** | 前 k 里 gold 数 / k | 衡量"噪声多不多" |
| **Hit@k** | 前 k 是否至少命中一个 gold | 二值，最宽松 |
| **MRR@k** | `(1/Q)·Σ 1/rank_i`（首个 gold 的排名倒数） | 关心"第一个正确答案排多前" |
| **nDCG@k** | `DCG@k / IDCG@k`，`DCG = Σ (2^rel_i - 1)/log2(i+1)` | 带分级相关性、带位置折扣，最全面 |
| **零命中率**（不可答） | 不可答问题检索结果为空的比例 | **拒答的前置条件**；越低越容易被编造 |

> 记忆锚点：**Recall 管捞得全不全，MRR 管排得前不前，nDCG 管整体排序质量，零命中率管该拒答时有没有依据。**

### 生成侧指标

- **忠实度 Faithfulness**：回答的每个断言能否由检索上下文支撑（防幻觉核心指标）；
- **答案相关性 Answer Relevancy**：答非所问；
- **上下文精度/召回**（Context Precision/Recall）：检索质量的另一视角。

RAGAS 是常见框架。**LLM-as-judge 的偏差**要主动说：位置偏差、长度偏好、被流畅度骗，
必须**人工抽检校准**（本项目 24 条全集人工可核，反而更可靠）。

### 项目实测数据（`evals/report.md`）

| 通道 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|
| 仅关键词通道 | 0.889 | 1.000 | 1.000 | 0.935 | 0.952 | 1.7 | 2.1 |
| 混合检索（向量+关键词+RRF+精排） | 0.778 | 0.778 | 0.778 | 0.778 | 0.778 | 7.8 | 40.0 |

| 指标 | 数值 | 说明 |
|---|---|---|
| 关键词通道零命中率（不可回答） | 1.000 | 没有召回就没有可引用的内容 |
| 混合通道零命中率（不可回答） | **0.000** | 无相关性阈值，混合通道几乎总会返回候选 |
| 引用格式覆盖率（可回答） | 1.000 | 回答是否带 `[1]` 标注 |

**怎么解读这张表（面试关键，别被数字带偏）**：

- 这是 `EMBEDDING_MODE=fake` 的结果——伪向量与文本语义无关，**向量通道在当前数据上不携带信息**，
  混合检索被"噪声通道"拖低是**预期现象**，不能得出"混合检索没用"；
  要得到结论必须配真实 key 用 `EMBEDDING_MODE=real` 复跑；
- 但**延迟差异是真实且有意义的**：混合 + rerank 的 p95 从 2.1ms 涨到 40ms（多一次向量查询 + 融合 + 精排）；
- 零命中率 0.000 是**真实缺陷**（与 embedding 质量无关，因为向量通道永远返回最近的 5 个），
  这就是"**评测发现问题 → 定位 → 修复**"的完整叙事，面试时比"指标很好看"有价值得多。

---

## A11. 后端工程化考点

### 多租户隔离

`app/services/knowledge_base_service.py` / `chat_service.get_conversation` 的做法：
所有查询都带 `user_id`/`knowledge_base_id` 条件，
**越权访问返回 404 而不是 403**（`get_conversation` 查不到即 404）——
不泄露"该资源存在但不属于你"，防资源枚举。

面试常问"隔离级别怎么选"：本项目是**共享库 + 行级过滤**（成本低、够用）；
更严格的可选 schema 级隔离或库级隔离（成本高）。加索引 `(kb_id)`、`(user_id)` 是前提。

### 文档状态机与幂等

`document.status`：`pending → processing → ready | failed`（`app/db/models/document.py`）。

- 状态机让**前端可轮询、失败可解释**（`error_message` 截断 1000 字符）；
- **幂等**：重试前先 `DELETE FROM chunks WHERE document_id = ?` 再写新块
  （`ingest_service.process_document`），避免重复块导致召回重复、分数虚高；
- 失败时 `raise` 让 Celery 记录并可重试，同时状态落 `failed`（**双写**：
  任务系统与业务状态各记一份，便于人工重跑）。

### 异步任务（Celery）

为什么要异步：解析 + 向量化是秒级到分钟级的重活，同步做会**占满 worker 线程、拖垮 API 延迟**、
超时后前端重试造成重复处理。

要点：`acks_late`（任务完成后才 ack，worker 崩溃不丢任务）+ 自动重试退避；
`CELERY_TASK_ALWAYS_EAGER=true` 让本地无 Redis 也能跑通链路（**可测试性的设计**）。
**必须幂等**，因为 at-least-once 语义意味着任务可能被执行两次。

### 缓存

`chat_service._cache_key`：`qa:{kb_id}:{md5(question.strip().lower())}`，TTL 3600s。

- key 里带 `kb_id`：**不同知识库的同名问题不能命中彼此的缓存**（多租户下这是正确性 bug）；
- 归一化（`strip().lower()`）：提高命中率；
- 存 `{answer, sources}`：命中时 sources 也要能回放；
- **降级**：Redis 异常时 `return None` 继续走正常链路（**缓存不可用不能拖垮主流程**）；
- 文档变为 ready（`process_document`）或被删除（`delete_document`）时，`invalidate_kb_cache` 用 `SCAN` 删掉 `qa:{kb_id}:*`。拒答也会进缓存，所以入库和删除必须清掉，否则错误的「没有相关信息」会挂一小时。Redis 失败只打 warning，不让摄入失败。

### 限流

`chat_service.check_rate_limit`：键 `ratelimit:{user}`，Redis 有序集合存最近 60 秒的请求时间戳。一段 Lua 先 `ZREMRANGEBYSCORE` 丢掉窗口外的成员，未超限才 `ZADD`，避免并发下先读后写都通过。超限返回 429。

- 以前是自然分钟 `INCR`。12:00:50 的 30 次和 12:01:01 的 30 次可以都通过。滑动窗口看的是此刻往前的 60 秒。
- **Redis 故障时放行并打 warning**：可用性优先，降级必须能在日志里看见。
- 429 还没有 `Retry-After`。

### SSE 流式输出

`app/api/routes/conversations.py::ask_stream`：`media_type="text/event-stream"` +
`Cache-Control: no-cache` + `X-Accel-Buffering: no`（**禁用 Nginx 缓冲**，否则流被攒成一坨），
事件格式 `data: {json}\n\n`，事件类型在 JSON 里：`sources → delta* → done`（错误时为 `error`）。

- **SSE vs WebSocket vs 轮询**：SSE 是**单向（服务端→客户端）**、基于 HTTP、
  自动重连（`Last-Event-ID`）、无需额外协议；LLM 输出正是单向流，所以 SSE 是标准选择。
  WebSocket 双向但重（连接管理、心跳、网关兼容）；轮询延迟高、浪费请求；
- **断连处理**：`await request.is_disconnected()` 在**每产出一个片段时**检查，
  但**无法中断正在进行的那次 LLM 调用**——真正的取消需要把 `asyncio.CancelledError`/
  事件传进 httpx 流（`client.stream` 上下文退出）并停止计费；
- **第一个 `sources` 事件先于文本**：前端可以先渲染引用卡片再填字，体验更好；
- 缺 `: keep-alive` 心跳注释，长时间无输出可能被中间层判定空闲而断连；
- **DB session 生命周期**：流式响应期间依赖注入的 session 一直持有（FastAPI 0.106+ 在响应
  发送完才执行依赖的退出逻辑），长连接 + 连接池小 → 可能耗尽连接。高并发下应缩短持有时间
  （先取数据、流式期间只写一次）。

### 可观测与健康检查

`/health` 汇总 DB / Redis / LLM / Embedding 的探活（`check_llm`、`check_embedding` 各发一次极短请求）。
要点：健康检查要区分 **liveness（进程活着）与 readiness（依赖可用）**，
否则依赖抖动会导致编排系统无脑重启容器；对外暴露的 health 不应泄露内部细节。

### 可测试性（本项目最值钱的设计之一）

- `LLM_MODE=fake` / `EMBEDDING_MODE=fake`：无 key、无外网即可跑通全链路（本地、CI）；
- `CELERY_TASK_ALWAYS_EAGER=true`：无 Redis 也能测摄入；
- **MCP 协议层是纯函数**（`handle_message`），协议测试**不需要数据库**；
- `pytest` 覆盖：分块器、JWT/bcrypt、多租户隔离、摄入、缓存命中、SSE 流式。

---

## A12. MCP 协议考点

MCP（Model Context Protocol）把工具/资源以标准协议暴露给任意客户端（Claude Desktop、Cursor、Agent 平台）。
本项目实现了 **tools 子集 + stdio 传输**（`app/mcp/protocol.py` / `server.py` / `tools.py`），
自研约两百行、不依赖官方 SDK。

### 必须讲清的三件事

1. **stdio 的分帧是"换行分隔的 JSON-RPC 2.0"，不是 LSP 的 `Content-Length:` 头**——
   一行一个完整 JSON 消息。这是最容易踩的坑（照抄 LSP 会直接连不上）；
2. **stdout 只允许协议消息**，日志一律走 stderr（loguru 默认 stderr）。
   测试里用真子进程断言"stdout 每一行都是合法 JSON"，专门防日志污染协议；
3. **工具执行失败 ≠ 协议错误**：
   - 工具失败 → 正常 `result` + `{"content": [...], "isError": true}`，
     让**模型看到失败原因并自己决定下一步**（可恢复）；
   - 只有编解码失败 / 方法不存在 / 参数非法 → JSON-RPC `error`，错误码 `-32700 / -32601 / -32602`。

其他考点：

- **通知（notification）没有 `id`，按协议不能回响应**（`is_notification`）——回了会破坏客户端状态机；
- 生命周期：`initialize` → `notifications/initialized` → `tools/list` → `tools/call`；
- **写工具默认不挂载**（`MCP_ALLOW_WRITE=1` 才出现）——工具暴露面就是安全边界，
  能读就不给写，是"最小权限"原则在 Agent 工具设计里的体现；
- **身份来自环境变量**（`MCP_USER_EMAIL`）并强制按该用户过滤（`MCP_DEFAULT_KB_ID` 提供默认库）——
  局限：没有 OAuth，多用户场景需要真正的鉴权；
- 工具设计原则：描述要能被模型读懂（名称+参数 schema+何时使用）、
  返回值要**可被模型直接消费**（结构化文本而非 dump）、危险操作要显式开关。

---

# B 部分：面试题库与深挖

## B1. 一分钟项目自述模板

> **结构：定位 → 链路 → 亮点 → 数据 → 不足**（最后一句留钩子，引导面试官追问你最熟的部分）

"这是一个面向企业私有知识库的 RAG 问答后端，用 FastAPI + PostgreSQL/pgvector + Redis + Celery 实现。
用户上传文档后，系统异步完成解析、段落感知分块、批量向量化入库；
问答时走**混合检索**——向量检索和关键词检索各取 2×top_k，用 **RRF 融合**初筛，再精排取 3 条片段，
拼进带引用编号的 Prompt，让模型只依据片段作答、无依据时明确拒答，响应通过 **SSE 流式**返回并带引用元数据。

核心逻辑（分块器、混合检索、RRF、引用回答、MCP 协议层）都是自研的，没用 LangChain，
所以每一处都能讲清原理。工程上覆盖了 JWT 认证、多租户隔离、Celery 异步 + 幂等重试、
Redis 缓存与滑动窗口限流、docker-compose 一键部署，并带一套**离线评测集**——
12 篇语料 / 24 条查询，能算 Recall@k、MRR、nDCG，也做通道消融。

评测也暴露了真实问题：不可回答的问题上，混合检索的零命中率是 0.000，
说明缺少相关性阈值、拒答过度依赖提示词——这正是我下一步要补的。
另外我也把检索能力封装成了 MCP Server，任何 MCP 客户端都能直接当工具调用。"

---

## B2. 高频问答 20 题

> 每题给出**答题骨架**（按这个顺序说）+ 本项目落点。追问是面试官最可能接的第二刀。

**Q1. 完整讲一遍 RAG 链路。**
骨架：离线摄入（解析→分块→向量化→入库）与在线问答（检索→融合→精排→Prompt→生成→引用）两条线分开讲，
再说每一环的**失败模式**（哪个环节出问题会导致什么现象）。
落点：A1 映射表。

**Q2. 你项目为什么不用 LangChain？**
骨架：① 核心链路只有几处算法（分块/RRF/Prompt 拼装），抽象收益低于理解成本；
② 依赖树与版本升级不可控，出问题定位在框架内部；
③ 面试要讲原理，自研才讲得清；
④ 不是反对框架——原型期用框架是对的，生产上要评估**可控性 vs 开发速度**。
落点：`app/services/ingest_service.py` 开头注释"刻意不用 LangChain/LlamaIndex"。

**Q3. 分块大小怎么定？overlap 有什么用？**
骨架：按"一个问题需要多大上下文"定（本项目 800 字符 ≈ 2–3 段），
overlap 防边界截断（10%–20%），
代价是重复向量化（本项目 12.5% 额外成本），
最后强调**用评测集验证而不是拍脑袋**。追问：语义分块/父子块了解吗 → 说 A3 表格。

**Q4. 为什么向量检索用 HNSW 不用 IVFFlat？**
骨架：HNSW 召回高、增量插入友好、查询复杂度 O(log N)；IVFFlat 建得快省内存但要先有数据、
召回对 probes 敏感。本项目数据量中等、要支持文档随时增删 → HNSW。
追问 ef_search 怎么调 → 运行时 `SET hnsw.ef_search`，是唯一不用重建索引就能调召回/延迟的旋钮。

**Q5. 为什么混合检索？RRF 为什么比加权求和好？**
骨架：稠密/稀疏错误不相关（语义 vs 字面）→ 融合提升鲁棒性；
分数不可比（cosine 有界 vs BM25 无界且分布随查询浮动）→ 归一化不稳定 →
RRF 只用排名，`1/(k+rank)`，k=60，零训练、尺度免疫、双通道命中自然加权。落点：A7。

**Q6. 余弦距离和欧氏距离怎么选？**
骨架：文本 embedding 关心**方向**不关心模长 → 余弦；
归一化后 `cosine = 内积`，三者排序一致；
pgvector 里**操作符必须与索引 opclass 匹配**，否则索引静默失效（这是坑，一定要说）。

**Q7. 怎么防幻觉？**
骨架：四层——检索侧（阈值+rerank+查询改写）、上下文侧（少而精、去重、位置）、
生成侧（强约束 prompt + 拒答指令 + 引用编号）、输出侧（引用校验/覆盖率指标）。
再说本项目现状与缺口（无阈值，零命中率 0.000）。落点：A9 + B3 缺陷 3。

**Q8. 怎么评测 RAG？**
骨架：黄金集（可答 + **不可答**）→ 检索指标（Recall@k/MRR/nDCG 写得出公式）→
生成指标（忠实度/答案相关性）→ 端到端（p50/p95）→ 消融与线上 A/B。
强调"不可回答问题"是必需的，否则测不出幻觉。落点：A10 真实数字。

**Q9. 多租户怎么隔离？越权返回什么？**
骨架：共享库 + 行级过滤（所有查询带 `user_id`/`kb_id`），
越权返回 **404 而非 403**（防资源枚举）；
再讲深一层：**ANN + 标量过滤会伤召回**（大库小租户），缓解手段见 A5。

**Q10. 文档处理为什么异步？幂等怎么做？**
骨架：解析/向量化秒级到分钟级，同步会阻塞 API、超时重试造成重复；
Celery `acks_late` + 重试退避；**幂等 = 重试前先删旧块再写新块**；
状态机双写便于人工重跑。落点：`process_document`。

**Q11. 缓存 key 怎么设计？数据更新后缓存怎么办？**
骨架：`qa:{kb_id}:{md5(normalized_question)}`（**kb_id 必须在 key 里**，否则跨租户串味），
TTL 1h，存 answer+sources，Redis 挂了要降级；
**脏读问题**：文档变更后应通过版本号进 key 或删除知识库相关 key；本项目当前未做（诚实说）。

**Q12. 限流怎么实现？有什么缺陷？**
骨架：Redis `INCR` + 首次 `EXPIRE`（原子、分布式安全）；
缺陷：固定窗口边界双倍突发 → 滑动窗口/令牌桶；
Redis 故障时放行（可用性优先）但必须监控；429 应带 `Retry-After`。

**Q13. SSE 和 WebSocket、轮询的区别？**
骨架：SSE 单向、基于 HTTP、浏览器原生自动重连、无需协议升级；LLM 输出天然单向 → SSE 最优；
再说工程细节：`X-Accel-Buffering: no`、心跳、断连检测、
**DB session 在流式期间被长期持有**的资源问题。

**Q14. PostgreSQL 存向量 vs 用 Milvus/Qdrant？**
骨架：pgvector 胜在**同库同事务 + SQL 过滤 join + 少一个组件**，
删除知识库靠 `ON DELETE CASCADE` 就干净；
专用库胜在超大规模与原生多租户/量化；
引入外部库要处理双写不一致（outbox/补偿）。
本项目规模选 pgvector，是**权衡而非能力不足**。

**Q15. 上下文超长怎么办？**
骨架：减少喂入（rerank 取 3、去重、压缩摘要）、
分层（先摘要再细节）、滑动窗口历史、
以及"最相关放头尾"对抗 lost-in-the-middle；再说 token 预算怎么算（A9 结尾）。

**Q16. 你的 p95 延迟主要花在哪？怎么优化？**
骨架：拆解链路（embedding 一次远程调用 + 向量查询 + 关键词查询 + rerank + LLM 首 token）；
优化点：并发两路检索（`asyncio.gather`）、embedding 结果缓存、
LLM 流式输出降低**感知**延迟、rerank 批量/并发、
索引与连接池调参。诚实交代：本项目 evals 里混合通道 p95 40ms vs 仅关键词 2.1ms，
差异来自多一次向量计算与精排（fake 模式下 LLM 不参与）。

**Q17. 成本怎么优化？**
骨架：分块数 × 块长决定 embedding 成本 → 去重/增量更新避免重复向量化（content hash 缓存）；
Prompt tokens 决定生成成本 → rerank 后只喂 3 条、历史截断；
热点问答缓存直接省掉整条链路；小模型做 rerank/分类，大模型只做最终生成。

**Q18. 文档更新了怎么重新向量化？**
骨架：当前是"重传即覆盖"（先删旧块）；生产要**增量**——
按文件 hash/版本判断是否需要重算、变更块做 diff（只重算变化的块）、
旧版本保留用于回滚、缓存版本号同步 `INCR`（否则脏答案）。

**Q19. 你项目有什么不足？如果重做会怎么做？**
骨架：直接给 5 条（B3），每条都带**证据**与**修复方案**——
这题答得好坏几乎决定面试评价，因为它测的是"你有没有真的在维护它"。

**Q20. MCP 是什么？你实现时踩了什么坑？**
骨架：标准化工具协议；
坑：stdio 分帧是**换行**不是 LSP 的 `Content-Length`；
**stdout 只能有协议消息**（日志走 stderr）；
工具失败走 `isError: true` 而不是 JSON-RPC error（让模型能自我纠错）；
通知不能回响应。落点：A12。

---

## B3. 深挖：本项目 5 个真实缺陷与修复方案

> 这一节是这份文档的**最高价值部分**：全部基于真实代码，且都有验收方式。
> 面试时"我发现了 N 个问题"比"我实现了 N 个功能"更能证明工程成熟度。
>
> **五条和会话归属都已按下面的修法改完。** 下面保留「原来的代码」是为了讲发现过程。[`evals/report.md`](evals/report.md) 仍是修改前的数字。

### 缺陷 1：RRF 两路 rank 串扰，融合被系统性偏向向量通道（已修复）

**状态**：`_rrf_fuse` 已改为每一路从 `rank=1` 单独计分。单测是 `tests/test_unit.py::test_rrf_ranks_each_channel_independently`。面试里把它讲成「发现过、修过、用测试钉住」，不要再说当前代码还有这个 bug。离线评测报告仍是修复前跑的，数字不能直接当成修复后的结果。

**原来的位置**：`app/services/rag_service.py::_rrf_fuse`

```python
for rank, chunk in enumerate(vector_hits + keyword_hits):     # ⚠️ 两个列表拼接后共用连续 rank
    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
```

**问题**：标准 RRF 是**每个通道各自从 rank 1 开始**计算 `1/(k+rank)`。
这里把两个列表拼起来 `enumerate`，于是关键词通道的 rank 从 `len(vector_hits)` 开始：
`top_k=5` 时向量 rank 是 `0..9`（权重 `1/61..1/70`），关键词 rank 是 `10..19`（权重 `1/71..1/80`）——
**同样排第一，向量通道的权重是关键词通道的 1.164 倍**，且这个偏差**随 `top_k` 变化**
（`top_k` 一改，融合权重就变，结果不可复现）。

实测（两路各 10 条候选，`k=60`）：

```text
向量通道 rank0  score = 1/61 = 0.016393
关键词通道 rank0 score = 1/71 = 0.014085      # 本应为 1/61
比值 = 1.164
```

> 注意：它**不会**让融合完全失效（双通道命中的块仍然拿两项之和，仍被奖励），
> 而是"不该有的隐性先验"——这正是 code review 应该抓的东西。

**修复**（每路独立排名）：

```python
def _rrf_fuse(vector_hits, keyword_hits, top_k, k=60):
    scores: dict[int, float] = {}
    chunks: dict[int, Chunk] = {}
    for hits in (vector_hits, keyword_hits):              # 每路独立编号
        for rank, chunk in enumerate(hits, start=1):
            chunks[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(chunks.values(), key=lambda c: scores[c.id], reverse=True)
    return [RetrievedChunk(chunk=c, score=scores[c.id]) for c in ranked[:top_k]]
```

**验收**：加单测断言"某块在向量 rank 1 + 关键词 rank 1 的分数 == 2/(k+1)"；
在真实 embedding 下用 evals 复跑，观察 Recall@1/MRR 变化。

---

### 缺陷 2：中文未分词，关键词通道形同"整句短语匹配"（已修复）

**状态**：官方 `pgvector/pgvector:pg16` 没有 `zhparser` / `pg_jieba`。`docker/postgres/Dockerfile` 在这层镜像上编译了 `pg_jieba`。[`_keyword_search`](app/services/rag_service.py) 对 `content` 使用 `jiebacfg`，问句抽出的词元用 OR 拼查询，按 `ts_rank` 排序。应用层没有再装一套 jieba。没有共同词的改写仍然要靠向量通道。旧文档不用为了分词重新入库。

**原来的位置**：`app/services/rag_service.py::_extract_terms`

```python
terms = re.findall(r"[\w\u4e00-\u9fff]+", query.lower())
```

**问题**：Python 正则里 `\w` 是 Unicode 感知的，**连续汉字会被匹配成一整个 term**。
问"几点上班" → term = `["几点上班"]` → SQL 变成 `content ILIKE '%几点上班%'`，
而语料里写的是"弹性打卡时间**九点到十点**"→ **命中 0 条**。
也就是说这个通道只在"问句原文与文档原文有连续子串"时才工作，
对中文问法的改写（上班/打卡、年假/休假）零容忍。

实测（真实函数）：

```text
_extract_terms("几点上班？")   -> ['几点上班']        # 未分词，整串一个 term
_extract_terms("年假有多少天") -> ['年假有多少天']      # 任何改写都会导致 ILIKE 落空
```

**修复**（分层，成本从低到高）：

1. **立刻可做**：中文查询按 2-gram 切分再 `or_` 组合
   （`几点上班` → `几点`、`点上`、`上班`），配 `pg_trgm` GIN 索引让 `ILIKE '%x%'` 走索引；
2. **标准做法**：`tsvector` + `zhparser`/`pg_jieba` + `ts_rank_cd`（得到真正的 BM25 式打分与排序）；
3. **应用层**：jieba 分词后拼 `to_tsquery`。

**验收**：把"几点上班""年假有多少天"这类改写问句加进评测集，
断言关键词通道召回率提升（当前混合通道 Recall@5 = 0.778 有相当部分损失来自这里）。

---

### 缺陷 3：无相关性阈值，不可回答问题的零命中率为 0.000（已修复，阈值未标定）

**状态**：真实向量下，RRF 之后、rerank 之前丢掉余弦相似度低于 `retrieval_min_similarity`（默认 0.35）的候选。相似度 = 1 − 余弦距离。一个都不剩时 [`ask`](app/services/chat_service.py) / `ask_stream` 返回「知识库中没有相关信息」，不调用模型。`embedding_mode=fake` 或该值 ≤ 0 时关闭阈值：假向量没有语义，0.35 会把能答的题也拒掉。0.35 是起点，不是评测标定值。拒答会进缓存；文档 ready 或删除时 `invalidate_kb_cache` 清掉 `qa:{kb_id}:*`。

**原来的位置**：`rag_service.search_knowledge_base` + `build_messages` 的拒答指令

**证据**：`evals/report.md` 实测——关键词通道零命中率 **1.000**，
混合通道 **0.000**：只要知识库非空，向量通道总会返回"最近的 5 个"（哪怕毫无关系），
于是 Prompt 里永远有片段，**拒答完全靠模型愿不愿意听指令**。

**修复**（"检索为空即拒答"的短路分支）：

```python
MIN_SCORE = 0.35   # 由评测集标定：在可答/不可答之间取使 F1 最大的阈值

reranked = await rerank_service.rerank(query, fused, top_k)
hits = [item for item in reranked if item.rerank_score >= MIN_SCORE]
if not hits:
    return NoAnswer("知识库中没有找到相关信息")   # 连 LLM 都不调用，省一次费用
```

**注意**：RRF 分数（`~0.03` 量级）**不适合直接当阈值**——它只反映排名不反映相似度。
要么用向量余弦相似度做阈值（`1 - cosine_distance`），要么用 rerank 分数（`llm` 模式下是 0–10，很直观）。
**阈值必须由评测集标定**，不能拍脑袋——这正是"有评测集"的价值兑现点。

---

### 缺陷 4：`or` 吞掉合法 0 值 + `chunk_overlap >= chunk_size` 不校验（已修复）

**状态**：`chunk_text`、`search_knowledge_base`、`rerank` 只在参数为 `None` 时用默认值。`chunk_size <= 0` 或 overlap 不在 `[0, chunk_size)` 时抛 `ValueError`。单测 `tests/test_unit.py::test_chunk_overlap_bounds`。

**原来的位置**：`app/services/ingest_service.py::chunk_text`、`rag_service.search_knowledge_base`、`rerank_service.rerank`

**(a) `or` 当默认值会吞掉 0 / False**：

```python
chunk_size = chunk_size or settings.chunk_size   # 传 0 会被静默替换成 800
overlap    = overlap    or settings.chunk_overlap
top_k      = top_k      or settings.top_k
```

对 `0`、`0.0`、`""`、`False` 这类合法值，`or` 会跳转默认值——**调用方以为关掉了，其实没关**。
正确写法：`if chunk_size is None: chunk_size = settings.chunk_size`。

**(b) `overlap >= chunk_size` 时死循环或静默丢内容**：

```python
while len(para) > chunk_size:
    chunks.append(para[:chunk_size])
    para = para[chunk_size - overlap:]     # ⚠️ 未校验 overlap < chunk_size
```

- `overlap == chunk_size`：`para[chunk_size - chunk_size:]` = `para[0:]` = **原串不变** →
  `while` 条件恒真 → **无限循环 + 内存暴涨**（直到 OOM）；
- `overlap > chunk_size`：`chunk_size - overlap` 为负 → 变成"取末尾 N 个字符" →
  **段落中间大段内容被静默丢弃**（最危险的一类 bug：不报错、只是答案找不到了）。

实测（真实函数，`python3 -c "chunk_text(...)"`）：

```text
chunk_text('x'*500, chunk_size=100, overlap=100)   -> 挂住，5 秒超时被 kill（死循环确认）
chunk_text('A'*100 + 'X'*200 + 'B'*100, chunk_size=100, overlap=200)
    -> 2 个块，共 200 字符（原文 400）  # 中间 200 字符被静默丢弃
```

**修复**：

```python
def chunk_text(text, chunk_size=None, overlap=None):
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    overlap    = settings.chunk_overlap if overlap is None else overlap
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"chunk_overlap 必须满足 0 <= overlap < chunk_size，当前 {overlap} >= {chunk_size}")
```

**验收**：pytest 参数化测试 `overlap ∈ {0, 99, 100, 200}` + `chunk_size=100`，
断言 `overlap < chunk_size` 时正常、越界时抛 `ValueError`（而不是挂住）。

---

### 缺陷 5：限流的静默降级 + LLM rerank 的串行与静默 fallback（已修复）

**状态**：限流改为 Redis 有序集合的 60 秒滑动窗口，检查和写入在同一段 Lua 里。Redis 失败打 warning 后放行。`rerank_mode=llm` 时一次请求打完全部片段；`parse_score_list` 只接受等长 JSON 数组，「0-10 分，我给 8」不会被当成 0。解析失败保留 RRF 顺序并打日志。默认模式仍是 `rrf`。单测 `test_parse_score_list_rejects_prose_with_zero`。

**原来的位置**：`app/services/chat_service.py::check_rate_limit`、`app/services/rerank_service.py::_llm_score`

**(a) 限流在 Redis 故障时静默放行**：

```python
except Exception:   # noqa: BLE001
    pass            # Redis 挂了 = 完全不限流，且没有任何信号
```

可用性优先是合理取舍，但**必须可观测**：降级时打 `warning` + 计数指标 + 告警，
否则出现"限流失效导致被刷爆"时无从查起。此外固定窗口的边界双倍突发问题（A11）也应一并说明。

**(b) LLM rerank 逐条串行 + 异常返回 0.0**：

- 10 个候选 = 10 次 LLM 往返（串行）→ p95 直接被拉高；
  改为 **listwise 一次打分**（一次请求给 N 个片段，要求返回 JSON 数组）
  或 `asyncio.gather` + 并发上限（`Semaphore`）；
- 异常 `return 0.0`：全部 0 分时 `sort` 稳定 → **退化为 RRF 顺序**，功能不崩但**静默降级**，
  应打点告警并考虑回退到 `rrf` 模式；
- 打分用 `re.findall(r"\d+")` 抓数字：模型回"0-10 分"时可能先抓到 `0`，
  应要求结构化输出（JSON schema / function calling）再解析。

**验收**：`llm` 模式下跑 evals，对比 `rrf` 模式的 Recall@k 与 p95，量化 rerank 的收益/代价
（这正是 `RERANK_MODE` 开关存在的意义）。

---

### 附：创建会话不校验知识库主人（已修复）

这不在上面五条里。`POST /conversations` 以前只把请求里的 `knowledge_base_id` 写进行，不查 `owner_id`。上传文档那条路会 404，这条不会。别人的库 id 可以写进自己的会话，检索只按库 id 取块，缓存键里也没有用户 id。

**现在**：[`create_conversation`](app/services/chat_service.py)、[`ask`](app/services/chat_service.py)、[`ask_stream`](app/services/chat_service.py) 都先走 `_require_owned_kb`。不是当前用户的库，以及库已被删成空，都返回 404。提问时的检查在读缓存之前，避免旧会话读到别人的答案。

---

## B4. 手写题参考实现

> 白板/在线评测常考。写完主动补一句"边界条件"和"复杂度"，比写完就停更得分。

### RRF 融合

```python
def rrf_fuse(ranked_lists: list[list[int]], k: int = 60, top_n: int = 10) -> list[tuple[int, float]]:
    """ranked_lists: 每个通道的 doc_id 有序列表（从最相关到最不相关）"""
    scores: dict[int, float] = {}
    for hits in ranked_lists:                       # 每路独立排名（关键！）
        for rank, doc_id in enumerate(hits, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # 分数相同时按 doc_id 兜底，保证结果确定性（可复现）
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
```
复杂度 O(总候选数 log 候选数)。边界：空列表、重复 doc_id（同路内应去重，保留最优 rank）。

### 余弦相似度

```python
import math

def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("维度不一致")            # 换 embedding 模型后的典型故障
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0                                 # 零向量：除零保护
    return dot / (na * nb)
```

### 滑动窗口限流（Redis ZSET）

```python
async def allow(r, user_id: int, limit: int = 30, window: int = 60) -> bool:
    key = f"rl:{user_id}"
    now = time.time()
    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, now - window)    # 清窗口外
    pipe.zcard(key)                                # 计数
    pipe.zadd(key, {f"{now}": now})
    pipe.expire(key, window)
    _, count, *_ = await pipe.execute()
    return count < limit
```
比固定窗口平滑；代价是内存 O(limit)。生产常用**令牌桶**（允许突发且内存 O(1)）。

### 分块器（段落感知）

```python
def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    assert 0 <= overlap < size, "overlap 必须小于 size"
    paras = [p.strip() for p in re.split(r"\n\s*\n", re.sub(r"\n{3,}", "\n\n", text).strip()) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}"; continue
        if cur:
            out.append(cur); cur = ""
        while len(p) > size:                       # 超长段落硬切 + overlap
            out.append(p[:size]); p = p[size - overlap:]
        cur = p
    if cur:
        out.append(cur)
    return out
```

### nDCG@k

```python
def ndcg_at_k(ranked_rel: list[float], k: int) -> float:
    dcg = sum((2 ** rel - 1) / math.log2(i + 1) for i, rel in enumerate(ranked_rel[:k], start=1))
    ideal = sorted(ranked_rel, reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0
```

---

# C 部分：速查与清单

## C1. 参数速查（本项目实际值）

| 参数 | 值 | 位置 / 含义 |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 100 | 段落感知分块的合并上限与衔接字符数（overlap 占 12.5%） |
| `EMBEDDING_DIM` | 1024 | `doubao-embedding-vision-251215`；`vector` 的 HNSW 上限 2000 维 |
| `EMBEDDING_BATCH_SIZE` | 16 | 每批向量化条数（控制请求体与超时） |
| `TOP_K` | 5 | 每个通道召回数 = `top_k × 2` = 10 |
| `RERANK_TOP_K` | 3 | 精排后进 Prompt 的片段数 |
| `RERANK_MODE` | `rrf` | `rrf`（零成本）/ `llm`（0–10 打分） |
| RRF `k` | 60 | `1/(k+rank)`，原论文经验值 |
| `LLM_TEMPERATURE` | 0.2 | 生成；rerank 打分用 0 |
| HNSW | `m=16, ef_construction=64, ef_search=40` | pgvector 默认值，`vector_cosine_ops` |
| `RATE_LIMIT_PER_MINUTE` | 30 | Redis `INCR` + `EXPIRE 90s`，固定窗口，按用户 |
| 问答缓存 TTL | 3600 s | key = `qa:{kb_id}:{md5(question.strip().lower())}` |
| 历史截断 | 6 条（`get_history` 取 10） | 控制上下文 tokens |
| 上传上限 | 10 MB | txt / md / pdf |
| 文档状态 | pending → processing → ready / failed | `documents.status`，失败原因截断 1000 字符 |
| SSE 事件 | `sources → delta* → done`（异常 `error`） | `data: {json}\n\n`，`X-Accel-Buffering: no` |
| JWT | HS256，过期 1440 分钟 | bcrypt 存密码哈希 |
| MCP | 协议版本 `2025-06-18` | stdio + 换行分隔 JSON-RPC 2.0 |

## C2. 复杂度速查

| 操作 | 复杂度 | 备注 |
|---|---|---|
| 精确 KNN 检索 | O(N·d) | 全表扫描，10 万块内可接受 |
| HNSW 查询 | ~O(log N)，常数大 | 内存 ∝ N·M·d |
| IVFFlat 查询 | O(N/lists · probes · d) | 需先有数据再建索引 |
| BM25 查询 | O(Σ posting list 长度) | 倒排索引；ILIKE `%x%` 退化为 O(N·L) |
| RRF 融合 | O((C₁+C₂) + K log K) | 只有排序，无距离计算 |
| 余弦距离 | O(d) | 归一化后等于内积 |

## C3. 术语表

| 中文 | 英文 / 缩写 | 一句话 |
|---|---|---|
| 检索增强生成 | RAG | 用检索证据约束生成 |
| 分块 | Chunking | 把长文切成检索单元 |
| 向量化 | Embedding | 文本 → 稠密向量 |
| 稠密检索 | Dense Retrieval | 向量近邻检索 |
| 稀疏检索 | Sparse Retrieval | 词项匹配（BM25） |
| 近似最近邻 | ANN | 用索引换速度，牺牲部分召回 |
| 分层可导航小世界 | HNSW | 图索引，高召回低延迟 |
| 倒排文件+聚类 | IVFFlat | 聚类桶索引，省内存 |
| 倒数排名融合 | RRF | 只用排名的多路融合 |
| 精排 | Rerank | 对候选做交叉编码打分 |
| 双塔 / 交叉编码器 | bi-encoder / cross-encoder | 分开编码 / 拼接编码 |
| 忠实度 | Faithfulness | 回答能否被上下文支撑 |
| 倒数排名均值 | MRR | 首个正确答案排名的倒数均值 |
| 归一化折损累计增益 | nDCG | 带位置折扣的排序质量 |
| 幻觉 | Hallucination | 无依据的生成内容 |
| 服务端事件推送 | SSE | 单向 HTTP 流 |
| 模型上下文协议 | MCP | 把能力暴露成标准工具 |

## C4. 复习清单（面试前自测）

**能不看稿讲清：**
- [ ] 摄入与问答两条链路的每一步，以及每步的失败模式
- [ ] 为什么分块 + 800/100 的取舍 + overlap 的成本
- [ ] 余弦/欧氏/内积的关系，以及"操作符必须匹配 opclass"
- [ ] HNSW 原理与 `m` / `ef_construction` / `ef_search` 三个旋钮
- [ ] 为什么混合 + RRF 公式手算一遍 + `k=60` 的作用
- [ ] bi-encoder vs cross-encoder，为什么不能全量精排
- [ ] Recall@k / MRR / nDCG 的公式与区别
- [ ] 幻觉四来源与四层防护
- [ ] 多租户 404、状态机、幂等、缓存 key、限流缺陷、SSE 工程细节
- [ ] MCP 三个坑（换行分帧 / stdout 干净 / isError）

**能拿出证据：**
- [ ] 说出 evals 里两个通道的 Recall@5（0.778 / 1.000）与零命中率（0.000），
      **并解释为什么 fake embedding 下这些数字不能证明混合检索无用**
- [ ] 说出 5 个真实缺陷中的至少 3 个，每个带**修复代码思路 + 验收方式**

**主动准备好的"钩子"：**
- [ ] "评测暴露了零命中率 0.000" → 引导到相关性阈值与评测驱动开发
- [ ] "RRF 的 rank 实现和原论文不一致" → 引导到 code review 能力
- [ ] "中文没有分词，关键词通道很脆弱" → 引导到 BM25 / zhparser
