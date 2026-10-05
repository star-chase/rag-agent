#!/usr/bin/env python3
"""RAG Agent 检索质量评测（evals）。

跑在真实的 PostgreSQL + pgvector 上，复用 app 的分块器、向量化、混合检索与 RRF/精排，
对着 `evals/golden/` 里固定的语料与查询集算指标：

- 检索质量：Recall@1 / @3 / @5、MRR@5、nDCG@5（文档级）
- 通道消融：关键词通道 vs 混合检索（向量 + 关键词 + RRF + 精排）
- 拒答可靠性：不可回答的问题里，有多少条检索结果是空的（拒答的前置条件）
- 延迟：每次检索的 p50 / p95

离线模式（EMBEDDING_MODE=fake、LLM_MODE=fake）用于 CI 回归：向量是确定性伪向量、没有语义，
所以离线模式用查询自带的 `terms` 当检索词，指标只反映**管道行为**；
要看真实语义质量，配好 EMBEDDING_API_KEY 后用 `EMBEDDING_MODE=real` 跑同一套评测。

用法：
    python -m evals.run_evals                          # 跑全部，打印摘要
    python -m evals.run_evals --report evals/report.md
    python -m evals.run_evals --check                  # 低于阈值时退出码 1（CI 用）
    python -m evals.run_evals --top-k 5 --json evals/result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 环境必须在 import app 之前设置（和 tests/conftest.py 同一套约束）
os.environ["DEBUG"] = "false"  # 评测报告要干净：关掉 SQL echo
os.environ.setdefault("TESTING", "1")
os.environ.setdefault("EMBEDDING_MODE", "fake")
os.environ.setdefault("LLM_MODE", "fake")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/0")
os.environ.setdefault("JWT_SECRET", "evals-secret-0123456789abcdef0123456789abcdef")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://{user}:{password}@{host}:{port}/rag_agent_test".format(
        user=os.environ.get("TEST_DB_USER", "rag"),
        password=os.environ.get("TEST_DB_PASSWORD", ""),
        host=os.environ.get("TEST_DB_HOST", "127.0.0.1"),
        port=os.environ.get("TEST_DB_PORT", "5433"),
    ),
)

from sqlalchemy import delete, text  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.models.chunk import Chunk  # noqa: E402
from app.db.models.document import DOC_STATUS_READY, Document  # noqa: E402
from app.db.models.knowledge_base import KnowledgeBase  # noqa: E402
from app.db.models.user import User  # noqa: E402
from app.db.session import AsyncSessionLocal, engine  # noqa: E402
from app.services import embedding_service, llm_service, rag_service  # noqa: E402
from app.services.ingest_service import chunk_text  # noqa: E402
from app.services.rag_service import RetrievedChunk, _keyword_search  # noqa: E402

GOLDEN = Path(__file__).with_name("golden")
CORPUS = GOLDEN / "corpus"
QUERIES = GOLDEN / "queries.jsonl"

# --check 的默认阈值（离线模式实测值略高，留出回归余量）
DEFAULT_THRESHOLDS = {
    "recall@5": 0.9,
    "recall@1": 0.7,
    "mrr@5": 0.75,
    "unanswerable_keyword_zero_hit": 0.9,
}


def load_corpus() -> list[tuple[str, str]]:
    return sorted(
        (path.name, path.read_text(encoding="utf-8"))
        for path in CORPUS.glob("*.md")
    )


def load_queries() -> list[dict]:
    items = []
    for line in QUERIES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant & set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for rank, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(ranked[:k], start=1)
        if doc in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def dedup(ids: list[int]) -> list[int]:
    seen: list[int] = []
    for item in ids:
        if item not in seen:
            seen.append(item)
    return seen


@dataclass
class QueryResult:
    query_id: str
    question: str
    effective_query: str
    answerable: bool
    relevant: list[str]
    ranked_docs: list[str]
    keyword_docs: list[str]
    hybrid_latency_ms: float
    keyword_latency_ms: float
    cited: bool | None = None
    refused: bool | None = None


@dataclass
class Metrics:
    name: str
    recall_1: float
    recall_3: float
    recall_5: float
    mrr_5: float
    ndcg_5: float
    p50_ms: float
    p95_ms: float
    detail: dict = field(default_factory=dict)


async def ensure_test_db() -> None:
    """测试库不存在时创建（和 tests/conftest.py 同一套逻辑）。"""
    import asyncpg

    url = os.environ["DATABASE_URL"]
    parts = url.split("://", 1)[1]
    credentials, host_part = parts.split("@", 1)
    user, password = credentials.split(":", 1)
    host, port_and_db = host_part.split(":", 1)
    port, _database = port_and_db.split("/", 1)

    conn = await asyncpg.connect(
        user=user, password=password or None, host=host, port=int(port), database="postgres"
    )
    exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = 'rag_agent_test'")
    if not exists:
        await conn.execute('CREATE DATABASE "rag_agent_test"')
    await conn.close()


async def seed_corpus(corpus: list[tuple[str, str]]) -> tuple[int, dict[str, int], User]:
    """把 golden 语料写进一个临时知识库，走真实的分块器与向量化服务。"""
    async with AsyncSessionLocal() as db:
        user = User(
            email=f"evals-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password="evals",  # noqa: S106 —— 评测用户，不参与登录
            name="evals",
        )
        db.add(user)
        await db.flush()
        kb = KnowledgeBase(name="evals-golden", description="检索评测语料", owner_id=user.id)
        db.add(kb)
        await db.flush()

        filename_to_doc: dict[str, int] = {}
        for filename, content in corpus:
            doc = Document(
                knowledge_base_id=kb.id,
                filename=filename,
                storage_key=f"evals/{filename}",
                file_type="md",
                file_size=len(content.encode("utf-8")),
                status=DOC_STATUS_READY,
                chunk_count=0,
            )
            db.add(doc)
            await db.flush()
            chunks = chunk_text(content)
            vectors = await embedding_service.embed_texts(chunks)
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                db.add(
                    Chunk(
                        document_id=doc.id,
                        knowledge_base_id=kb.id,
                        seq=index,
                        content=chunk,
                        embedding=vector,
                        meta={
                            "filename": filename,
                            "chunk_index": index,
                            "chunk_count": len(chunks),
                        },
                    )
                )
            doc.chunk_count = len(chunks)
            filename_to_doc[filename] = doc.id
        await db.commit()
        return kb.id, filename_to_doc, user


async def cleanup(kb_id: int, user: User) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Chunk).where(Chunk.knowledge_base_id == kb_id))
        await db.execute(delete(Document).where(Document.knowledge_base_id == kb_id))
        await db.execute(delete(KnowledgeBase).where(KnowledgeBase.id == kb_id))
        await db.execute(delete(User).where(User.id == user.id))
        await db.commit()


async def run_query(
    db,
    kb_id: int,
    doc_id_to_filename: dict[int, str],
    query: dict,
    top_k: int,
    real_mode: bool,
) -> QueryResult:
    effective = query["question"] if real_mode else " ".join(query["terms"])

    started = time.perf_counter()
    hybrid_hits: list[RetrievedChunk] = await rag_service.search_knowledge_base(
        db, kb_id, effective, top_k=top_k
    )
    hybrid_latency = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    keyword_hits = await _keyword_search(db, kb_id, effective, top_k * 2)
    keyword_latency = (time.perf_counter() - started) * 1000

    ranked_docs = [
        doc_id_to_filename[doc_id]
        for doc_id in dedup([hit.chunk.document_id for hit in hybrid_hits])
    ]
    keyword_docs = [
        doc_id_to_filename[doc_id] for doc_id in dedup([hit.document_id for hit in keyword_hits])
    ]

    cited: bool | None = None
    refused: bool | None = None
    if query["answerable"]:
        # 引用格式覆盖率：有检索结果时，回答是否带 [1] 标注（fake/真实模型都适用）
        messages = rag_service.build_messages(effective, hybrid_hits, None)
        answer = await llm_service.chat(messages)
        cited = "[1]" in answer if hybrid_hits else False
    else:
        messages = rag_service.build_messages(effective, hybrid_hits, None)
        answer = await llm_service.chat(messages)
        refused = "没有相关信息" in answer or not hybrid_hits

    return QueryResult(
        query_id=query["id"],
        question=query["question"],
        effective_query=effective,
        answerable=query["answerable"],
        relevant=query["relevant"],
        ranked_docs=ranked_docs,
        keyword_docs=keyword_docs,
        hybrid_latency_ms=hybrid_latency,
        keyword_latency_ms=keyword_latency,
        cited=cited,
        refused=refused,
    )


def summarize(name: str, results: list[QueryResult], top_k: int, channel: str) -> Metrics:
    answerable = [item for item in results if item.answerable]
    docs = [
        item.ranked_docs if channel == "hybrid" else item.keyword_docs
        for item in answerable
    ]
    latencies = [
        item.hybrid_latency_ms if channel == "hybrid" else item.keyword_latency_ms
        for item in answerable
    ]
    relevant_sets = [set(item.relevant) for item in answerable]
    return Metrics(
        name=name,
        recall_1=statistics.mean([recall_at(d, r, 1) for d, r in zip(docs, relevant_sets)]),
        recall_3=statistics.mean([recall_at(d, r, 3) for d, r in zip(docs, relevant_sets)]),
        recall_5=statistics.mean([recall_at(d, r, 5) for d, r in zip(docs, relevant_sets)]),
        mrr_5=statistics.mean([reciprocal_rank(d[:5], r) for d, r in zip(docs, relevant_sets)]),
        ndcg_5=statistics.mean([ndcg_at(d, r, 5) for d, r in zip(docs, relevant_sets)]),
        p50_ms=statistics.median(latencies) if latencies else 0.0,
        p95_ms=sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)] if latencies else 0.0,
    )


def render_markdown(
    results: list[QueryResult], top_k: int, real_mode: bool, settings, corpus_size: int
) -> str:
    hybrid = summarize("混合检索（向量 + 关键词 + RRF + 精排）", results, top_k, "hybrid")
    keyword = summarize("仅关键词通道", results, top_k, "keyword")
    answerable = [item for item in results if item.answerable]
    unanswerable = [item for item in results if not item.answerable]
    mode = "真实模型" if real_mode else "离线（fake embedding / fake llm）"
    keyword_zero = (
        sum(1 for item in unanswerable if not item.keyword_docs) / len(unanswerable)
        if unanswerable
        else 0.0
    )
    hybrid_zero = (
        sum(1 for item in unanswerable if not item.ranked_docs) / len(unanswerable)
        if unanswerable
        else 0.0
    )
    citation_rate = (
        sum(1 for item in answerable if item.cited) / len(answerable) if answerable else 0.0
    )

    lines = [
        "# RAG Agent 检索质量评测报告",
        "",
        f"- 运行时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- 模式：{mode}（embedding_mode={settings.embedding_mode}, llm_mode={settings.llm_mode}, "
        f"rerank_mode={settings.rerank_mode}）",
        f"- 语料：{corpus_size} 篇文档 / 查询：{len(results)} 条"
        f"（可回答 {len(answerable)}、不可回答 {len(unanswerable)}）",
        f"- top_k：{top_k}",
        "",
        "## 检索质量（文档级，{0} 条可回答问题）".format(len(answerable)),
        "",
        "| 通道 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 | p50 (ms) | p95 (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in (keyword, hybrid):
        lines.append(
            f"| {item.name} | {item.recall_1:.3f} | {item.recall_3:.3f} | {item.recall_5:.3f} | "
            f"{item.mrr_5:.3f} | {item.ndcg_5:.3f} | {item.p50_ms:.1f} | {item.p95_ms:.1f} |"
        )
    lines += [
        "",
        "## 拒答可靠性与引用",
        "",
        "| 指标 | 数值 | 说明 |",
        "| --- | --- | --- |",
        f"| 关键词通道零命中率（不可回答） | {keyword_zero:.3f} | 没有召回就没有可引用的内容，是拒答的前置条件 |",
        f"| 混合通道零命中率（不可回答） | {hybrid_zero:.3f} | 当前实现没有相关性阈值，混合通道几乎总会返回候选 |",
        f"| 引用格式覆盖率（可回答） | {citation_rate:.3f} | 回答里是否带 [1] 引用标注 |",
        "",
        "## 逐条明细",
        "",
        "| 查询 | 有效检索词 | 期望文档 | 混合 Top-1 | 命中@5 | 关键词命中 | 延迟 (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        expected = "、".join(item.relevant) if item.relevant else "（不可回答）"
        top1 = item.ranked_docs[0] if item.ranked_docs else "—"
        hit = "—" if not item.answerable else ("✅" if set(item.relevant) & set(item.ranked_docs[:5]) else "❌")
        keywords = "、".join(item.keyword_docs[:2]) if item.keyword_docs else "—"
        lines.append(
            f"| {item.question} | {item.effective_query} | {expected} | {top1} | {hit} | "
            f"{keywords} | {item.hybrid_latency_ms:.1f} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        f"- 混合检索 Recall@5 = {hybrid.recall_5:.3f}，MRR@5 = {hybrid.mrr_5:.3f}；"
        f"关键词通道 Recall@5 = {keyword.recall_5:.3f}。",
    ]
    if hybrid_zero < 0.5 and unanswerable:
        lines.append(
            "- 混合通道对不可回答问题几乎总会返回候选片段：拒答目前依赖提示词约束，"
            "建议补一个相关性阈值（或「检索为空/分数过低时直接拒答」的短路分支）。"
        )
    if not real_mode:
        lines.append(
            "- 离线模式用的是确定性伪向量（无语义），指标只反映管道行为；"
            "真实语义质量请配 EMBEDDING_API_KEY 后以 `EMBEDDING_MODE=real` 复跑本评测。"
        )
    lines.append("")
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    real_mode = settings.embedding_mode != "fake"
    corpus = load_corpus()
    queries = load_queries()

    await ensure_test_db()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    kb_id, filename_to_doc, user = await seed_corpus(corpus)
    doc_id_to_filename = {doc_id: name for name, doc_id in filename_to_doc.items()}
    try:
        async with AsyncSessionLocal() as db:
            results = [
                await run_query(db, kb_id, doc_id_to_filename, query, args.top_k, real_mode)
                for query in queries
            ]
    finally:
        if not args.keep:
            await cleanup(kb_id, user)
        await engine.dispose()

    report = render_markdown(results, args.top_k, real_mode, settings, len(corpus))
    print(report)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"报告已写入：{target}")
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([item.__dict__ for item in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.check:
        hybrid = summarize("hybrid", results, args.top_k, "hybrid")
        keyword = summarize("keyword", results, args.top_k, "keyword")
        unanswerable = [item for item in results if not item.answerable]
        keyword_zero = (
            sum(1 for item in unanswerable if not item.keyword_docs) / len(unanswerable)
            if unanswerable
            else 0.0
        )
        # 离线模式的向量是伪向量，混合指标不代表质量；回归卡口以关键词通道 + 拒答前置条件为准。
        if real_mode:
            measured = {
                "recall@5": hybrid.recall_5,
                "recall@1": hybrid.recall_1,
                "mrr@5": hybrid.mrr_5,
                "unanswerable_keyword_zero_hit": keyword_zero,
            }
            thresholds = DEFAULT_THRESHOLDS
        else:
            measured = {
                "recall@5": keyword.recall_5,
                "recall@1": keyword.recall_1,
                "mrr@5": keyword.mrr_5,
                "unanswerable_keyword_zero_hit": keyword_zero,
            }
            thresholds = {**DEFAULT_THRESHOLDS, "recall@1": 0.85, "mrr@5": 0.9, "recall@5": 0.95}
        failed = [
            f"{key}={measured[key]:.3f} < {thresholds[key]:.3f}"
            for key in thresholds
            if measured[key] < thresholds[key]
        ]
        if failed:
            print("阈值未达标：" + "；".join(failed))
            return 1
        print("阈值检查通过：" + "；".join(f"{k}={v:.3f}" for k, v in measured.items()))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG Agent 检索质量评测")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report", help="把 markdown 报告写到指定路径")
    parser.add_argument("--json", help="把逐条结果写成 JSON")
    parser.add_argument("--check", action="store_true", help="低于阈值时返回退出码 1（CI 用）")
    parser.add_argument("--keep", action="store_true", help="保留评测写入的临时知识库")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
