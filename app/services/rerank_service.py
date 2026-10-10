"""重排（Rerank）服务：对候选片段做最终相关性排序。

两种模式（settings.rerank_mode）：
- rrf：RRF 融合分数直接作为重排依据（轻量、零成本、默认）；
- llm：让 LLM 对“问题 + 候选片段”逐条打相关性分（0-10），
  按分数降序取 top_k。适合追求精排质量的场景，代价是额外 token。

接口保持简单，后续可替换为 bge-reranker 等交叉编码器模型。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from app.config import settings

if TYPE_CHECKING:
    from app.services.rag_service import RetrievedChunk


# LLM 打分回调：输入 (query, chunk_content) -> 0-10 分数
RerankScorer = Callable[[str, str], Awaitable[float]]


@dataclass
class RerankedItem:
    chunk: RetrievedChunk
    rerank_score: float


async def _llm_score(query: str, content: str) -> float:
    """默认 LLM 打分器：用 llm_service.chat 让模型输出一个 0-10 的分数。"""
    from app.services import llm_service

    prompt = (
        "你是检索相关性评估器。请判断下面这段知识片段与问题的相关程度，"
        "只输出一个 0 到 10 之间的数字，不要任何其他文字。\n\n"
        f"问题：{query}\n\n片段：{content[:800]}"
    )
    try:
        text = await llm_service.chat(
            [{"role": "user", "content": prompt}], temperature=0
        )
        # 从回答中提取数字（容忍模型输出额外文字）
        import re

        nums = re.findall(r"\d+(?:\.\d+)?", text)
        score = float(nums[0]) if nums else 0.0
        return max(0.0, min(10.0, score))
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm rerank failed, fallback to 0: {}", exc)
        return 0.0


async def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int | None = None,
    scorer: RerankScorer | None = None,
) -> list[RerankedItem]:
    """对候选片段重排，返回 top_k 个带重排分数的结果。

    输入 candidates 应已按某种策略（如 RRF 融合）初筛，数量通常为 top_k*2，
    这里做精排，把最相关的片段顶到最前。
    """
    top_k = top_k or settings.rerank_top_k
    if not candidates:
        return []

    if settings.rerank_mode == "llm":
        scorer = scorer or _llm_score
        scored: list[RerankedItem] = []
        for item in candidates:
            score = await scorer(query, item.chunk.content)
            scored.append(RerankedItem(chunk=item, rerank_score=score))
        scored.sort(key=lambda x: x.rerank_score, reverse=True)
        return scored[:top_k]

    # rrf 模式：直接用融合分数（已按分数降序）
    return [RerankedItem(chunk=item, rerank_score=item.score) for item in candidates[:top_k]]
