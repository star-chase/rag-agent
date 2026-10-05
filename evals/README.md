# RAG Agent 检索质量评测（evals）

用固定的语料与查询集，对**真实的检索链路**（分块 → 向量化 → 向量 + 关键词 → RRF 融合 → 精排）算指标，
把"我觉得检索效果还行"换成一张可以对比、可以进 CI 的表。

- 语料：`evals/golden/corpus/*.md`（12 篇企业制度文档）
- 查询：`evals/golden/queries.jsonl`（24 条：18 条可回答 + 6 条不可回答，标注期望文档）
- 运行环境：真实 PostgreSQL + pgvector（`docker compose up -d postgres`），复用 app 的分块器与向量化服务

## 指标定义

| 指标 | 含义 |
| --- | --- |
| Recall@1 / @3 / @5 | 期望文档是否出现在前 1 / 3 / 5 条检索结果里（文档级） |
| MRR@5 | 第一条命中结果的排名倒数均值，衡量"正确的答案排得够不够前" |
| nDCG@5 | 考虑排名位置的排序质量 |
| 关键词通道零命中率（不可回答） | 没有召回就没有可引用的内容，是拒答的前置条件 |
| 混合通道零命中率（不可回答） | 当前实现没有相关性阈值，混合通道几乎总会返回候选 |
| 引用格式覆盖率 | 回答里是否带 `[1]` 引用标注 |
| p50 / p95 延迟 | 单次检索耗时 |

## 怎么跑

```bash
# 1) 起一个带 pgvector 的数据库（宿主机 5432）
docker compose up -d postgres

# 2) 跑评测
TEST_DB_PORT=5432 TEST_DB_PASSWORD=rag PYTHONPATH=.deps python3 -m evals.run_evals --check --report evals/report.md
```

CI（`.github/workflows/ci.yml`）里 PostgreSQL 与 Redis 是 service container，直接跑：

```bash
python -m evals.run_evals --check --report evals/report.md
```

## 离线模式与真实模式

| 模式 | 环境变量 | 说明 |
| --- | --- | --- |
| 离线（CI 默认） | `EMBEDDING_MODE=fake` `LLM_MODE=fake` | 向量是确定性**伪向量**（无语义），所以离线模式用查询自带的 `terms` 当检索词，指标只反映**管道行为**，用于回归卡口 |
| 真实 | `EMBEDDING_MODE=real` + `EMBEDDING_API_KEY` | 用真实 Embedding 模型跑同一套语料与查询，指标才代表检索质量 |

离线模式用**关键词通道**做阈值卡口（`recall@5 ≥ 0.95`、`recall@1 ≥ 0.85`、`MRR@5 ≥ 0.9`、
不可回答问题零命中率 `≥ 0.9`）；真实模式卡口自动切到混合检索（`recall@5 ≥ 0.9`）。

## 最近一次离线实测（本机，pgvector + fake embedding）

| 通道 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 仅关键词通道 | 0.889 | 1.000 | 1.000 | 0.935 | 0.952 | 1.8 | 2.6 |
| 混合检索（向量 + 关键词 + RRF + 精排） | 0.778 | 0.778 | 0.778 | 0.778 | 0.778 | 8.4 | 49.3 |

其他指标：不可回答问题关键词零命中率 `1.000`、混合通道零命中率 `0.000`、引用格式覆盖率 `1.000`。

## 评测发现（这就是评测的价值）

- **混合通道对不可回答问题几乎总会返回候选片段**（零命中率 0.000）：拒答目前完全依赖系统提示词的约束。
  建议补一个相关性阈值，或在"检索为空 / 分数过低"时直接短路拒答。
- 离线模式下混合检索弱于关键词通道，是**伪向量无语义**导致的预期现象，不代表真实模型表现；
  真实模型下用 `EMBEDDING_MODE=real` 复跑即可得到可对比的数字。

## 怎么扩

1. 往 `evals/golden/corpus/` 加 `.md` 文档；
2. 往 `evals/golden/queries.jsonl` 加查询，字段：
   `{"id": "q25", "question": "自然语言问题", "terms": ["关键词1", "关键词2"], "relevant": ["文档文件名.md"], "answerable": true}`；
   不可回答的查询把 `relevant` 留空、`answerable` 设为 `false`；
3. 重跑评测，`evals/report.md` 会连逐条明细一起更新。
