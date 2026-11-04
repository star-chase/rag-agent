"""RAG 检索服务：混合检索（向量 + 关键词）+ 融合重排 + Prompt 构造。

检索链路：
1. 向量检索：pgvector 余弦距离，取 top_k * 2；
2. 关键词检索：pg_jieba 的 jiebacfg 分词，ts_rank 排序，取 top_k * 2；
3. RRF（Reciprocal Rank Fusion）融合两个列表；
4. 真实向量上按余弦相似度丢弃过低的候选，再 rerank 取最终 top_k。

这样既吃语义相似，又保留精确关键词命中（专有名词、编号等场景），
是 RAG 检索质量的经典工程优化。
"""
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Chunk
from app.services import embedding_service, rerank_service

REFUSAL_TEXT = "知识库中没有相关信息"


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    similarity: float | None = None


async def _vector_search(db: AsyncSession, kb_id: int, query_vec: list[float], limit: int) -> list[Chunk]:
    stmt = (
        select(Chunk)
        .where(Chunk.knowledge_base_id == kb_id, Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_vec).asc())
        .limit(limit)
    )
    return list(await db.scalars(stmt))


# jiebacfg 把问句切成词，停用词不会进入 tsvector。拼成 OR，避免整句 AND 把召回打没。
# 词元已经是词典输出，再用 simple 包一层，避免查询解析器把这些词重新切碎。
_KEYWORD_SQL = text(
    """
    SELECT c.id
    FROM chunks AS c
    JOIN LATERAL (
        SELECT string_agg(quote_literal(lexeme), ' | ' ORDER BY ord) AS tsq
        FROM (
            SELECT lexeme, row_number() OVER () AS ord
            FROM unnest(to_tsvector('jiebacfg', :query)) AS t(lexeme, positions)
            LIMIT :term_limit
        ) terms
    ) q ON q.tsq IS NOT NULL
    WHERE c.knowledge_base_id = :kb_id
      AND to_tsvector('jiebacfg', c.content) @@ to_tsquery('simple', q.tsq)
    ORDER BY ts_rank(to_tsvector('jiebacfg', c.content), to_tsquery('simple', q.tsq)) DESC,
             c.id DESC
    LIMIT :limit
    """
)


async def _keyword_search(db: AsyncSession, kb_id: int, query: str, limit: int) -> list[Chunk]:
    rows = await db.execute(
        _KEYWORD_SQL,
        {"query": query, "kb_id": kb_id, "limit": limit, "term_limit": 16},
    )
    ids = [int(row[0]) for row in rows]
    if not ids:
        return []
    chunks = list(await db.scalars(select(Chunk).where(Chunk.id.in_(ids))))
    order = {chunk_id: index for index, chunk_id in enumerate(ids)}
    chunks.sort(key=lambda chunk: order[chunk.id])
    return chunks


async def _cosine_similarities(
    db: AsyncSession, chunk_ids: list[int], query_vec: list[float]
) -> dict[int, float]:
    """余弦相似度 = 1 - 余弦距离。只计算已经有向量的块。"""
    if not chunk_ids:
        return {}
    distance = Chunk.embedding.cosine_distance(query_vec)
    rows = await db.execute(
        select(Chunk.id, distance).where(Chunk.id.in_(chunk_ids), Chunk.embedding.is_not(None))
    )
    return {int(chunk_id): 1.0 - float(dist) for chunk_id, dist in rows.all()}


def _min_similarity() -> float:
    """假向量彼此没有语义，阈值会把能回答的题也拒掉，所以只在真实向量上启用。"""
    if settings.embedding_mode == "fake" or settings.retrieval_min_similarity <= 0:
        return 0.0
    return settings.retrieval_min_similarity


def _rrf_fuse(
    vector_hits: list[Chunk],
    keyword_hits: list[Chunk],
    top_k: int,
    k: int = 60,
) -> list[RetrievedChunk]:
    """RRF 融合：每一路单独从 rank=1 计分，score = Σ 1/(k + rank)。

    两路必须分开编号。拼成一个列表再 enumerate 时，关键词第 1 名会接在向量列表后面，
    同样的名次权重变小，而且偏差随向量路长度变化。
    """
    scores: dict[int, float] = {}
    chunks: dict[int, Chunk] = {}
    for hits in (vector_hits, keyword_hits):
        for rank, chunk in enumerate(hits, start=1):
            chunks[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(chunks.values(), key=lambda c: scores[c.id], reverse=True)
    return [RetrievedChunk(chunk=c, score=scores[c.id]) for c in ranked[:top_k]]


async def search_knowledge_base(
    db: AsyncSession,
    kb_id: int,
    query: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """混合检索入口：向量 + 关键词 + RRF 融合初筛 -> 相似度阈值 -> rerank 精排。"""
    if top_k is None:
        top_k = settings.top_k
    if top_k <= 0:
        return []
    query_vec = await embedding_service.embed_query(query)

    vector_hits = await _vector_search(db, kb_id, query_vec, limit=top_k * 2)
    keyword_hits = await _keyword_search(db, kb_id, query, limit=top_k * 2)

    if not vector_hits and not keyword_hits:
        return []

    # 第一步：RRF 融合（候选扩大到 top_k*2，避免初筛漏掉相关片段）
    fused = _rrf_fuse(vector_hits, keyword_hits, top_k * 2)
    similarities = await _cosine_similarities(db, [item.chunk.id for item in fused], query_vec)
    for item in fused:
        item.similarity = similarities.get(item.chunk.id)

    # 余弦相似度低于阈值的候选不进精排，也不进模型。RRF 分数只有 0.03 量级，不能当阈值。
    min_similarity = _min_similarity()
    if min_similarity > 0:
        fused = [item for item in fused if (item.similarity or 0.0) >= min_similarity]
        if not fused:
            return []

    # 第二步：rerank 精排（rrf=融合分数；llm=一次请求给全部分数），取最终 top_k
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
