"""RAG 检索服务：混合检索（向量 + 关键词）+ 融合重排 + Prompt 构造。

检索链路：
1. 向量检索：pgvector 余弦距离，取 top_k * 2；
2. 关键词检索：内容 ILIKE 匹配查询词，取 top_k * 2；
3. RRF（Reciprocal Rank Fusion）融合两个列表，重排取最终 top_k。

这样既吃语义相似，又保留精确关键词命中（专有名词、编号等场景），
是 RAG 检索质量的经典工程优化。
"""
import re
from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Chunk
from app.services import embedding_service, rerank_service


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def _extract_terms(query: str) -> list[str]:
    """提取查询词：中英文、数字。停用词可在此扩展。"""
    terms = re.findall(r"[\w\u4e00-\u9fff]+", query.lower())
    stopwords = {"的", "了", "是", "在", "和", "与", "或", "吗", "呢", "吧", "the", "a", "an", "is", "are", "of", "for", "and", "to"}
    return [t for t in terms if len(t) >= 1 and t not in stopwords][:8]


async def _vector_search(db: AsyncSession, kb_id: int, query_vec: list[float], limit: int) -> list[Chunk]:
    stmt = (
        select(Chunk)
        .where(Chunk.knowledge_base_id == kb_id, Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_vec).asc())
        .limit(limit)
    )
    return list(await db.scalars(stmt))


async def _keyword_search(db: AsyncSession, kb_id: int, query: str, limit: int) -> list[Chunk]:
    terms = _extract_terms(query)
    if not terms:
        return []
    conditions = [Chunk.content.ilike(f"%{term}%") for term in terms]
    stmt = (
        select(Chunk)
        .where(Chunk.knowledge_base_id == kb_id, or_(*conditions))
        .order_by(Chunk.id.desc())
        .limit(limit)
    )
    return list(await db.scalars(stmt))


def _rrf_fuse(
    vector_hits: list[Chunk],
    keyword_hits: list[Chunk],
    top_k: int,
    k: int = 60,
) -> list[RetrievedChunk]:
    """RRF 融合：score = Σ 1/(k + rank)，对两个来源的排序取并集。"""
    scores: dict[int, float] = {}
    chunks: dict[int, Chunk] = {}
    for rank, chunk in enumerate(vector_hits + keyword_hits):
        chunk_id = chunk.id
        chunks[chunk_id] = chunk
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    ranked = sorted(chunks.values(), key=lambda c: scores[c.id], reverse=True)
    return [RetrievedChunk(chunk=c, score=scores[c.id]) for c in ranked[:top_k]]


async def search_knowledge_base(
    db: AsyncSession,
    kb_id: int,
    query: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """混合检索入口：向量 + 关键词 + RRF 融合初筛 -> rerank 精排。"""
    top_k = top_k or settings.top_k
    query_vec = await embedding_service.embed_query(query)

    vector_hits = await _vector_search(db, kb_id, query_vec, limit=top_k * 2)
    keyword_hits = await _keyword_search(db, kb_id, query, limit=top_k * 2)

    if not vector_hits and not keyword_hits:
        return []

    # 第一步：RRF 融合（候选扩大到 top_k*2，避免初筛漏掉相关片段）
    fused = _rrf_fuse(vector_hits, keyword_hits, top_k * 2)

    # 第二步：rerank 精排（rrf=融合分数；llm=LLM 相关性打分），取最终 top_k
    reranked = await rerank_service.rerank(query, fused, top_k)
    return [item.chunk for item in reranked]


def build_messages(question: str, chunks: list[RetrievedChunk], history: list[dict] | None = None) -> list[dict]:
    """构造 LLM 消息：系统提示注入检索片段，要求基于引用作答。"""
    context_parts = []
    for i, item in enumerate(chunks, start=1):
        c = item.chunk
        context_parts.append(f"[{i}] (来源: {c.meta.get('filename', '未知') if c.meta else '未知'})\n{c.content}")

    system = (
        "你是企业知识库智能助手。请仅依据下面提供的知识片段回答问题；"
        "如果片段中没有答案，请明确回答“知识库中没有相关信息”，不要编造。\n\n"
        "回答时在引用处用 [1][2] 标注对应片段编号，最后列出“参考资料”清单。\n\n"
        "========== 知识片段 ==========\n" + "\n\n".join(context_parts)
    )
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-6:])  # 只带最近 6 条历史，控制上下文
    messages.append({"role": "user", "content": question})
    return messages


def format_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    """把检索结果整理为给前端的引用元数据。"""
    return [
        {
            "chunk_id": item.chunk.id,
            "document_id": item.chunk.document_id,
            "filename": (item.chunk.meta or {}).get("filename", ""),
            "snippet": item.chunk.content[:200],
            "score": round(item.score, 4),
        }
        for item in chunks
    ]
