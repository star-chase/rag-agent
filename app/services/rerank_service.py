"""重排（Rerank）服务：对候选片段做最终相关性排序。

两种模式（settings.rerank_mode）：
- rrf：RRF 融合分数直接作为重排依据（轻量、零成本、默认）；
- llm：一次请求让 LLM 给全部候选打 0-10 分，按分数降序取 top_k。
  解析失败时保留 RRF 顺序，并打日志，不把全部分数写成 0。

接口保持简单，后续可替换为 bge-reranker 等交叉编码器模型。
"""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from app.config import settings

if TYPE_CHECKING:
    from app.services.rag_service import RetrievedChunk


# 一次给全部片段打分。失败时返回 None，调用方退回 RRF 顺序。
BatchScorer = Callable[[str, list[str]], Awaitable[list[float] | None]]


@dataclass
class RerankedItem:
    chunk: RetrievedChunk
    rerank_score: float


def parse_score_list(text: str, expected: int) -> list[float] | None:
    """只接受长度刚好的 JSON 数组。句子里的「0-10」不会被当成 0 分。"""
    match = re.search(r"\[[\s\S]*\]", text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or len(data) != expected:
        return None
    scores: list[float] = []
    for item in data:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        scores.append(max(0.0, min(10.0, float(item))))
    return scores


async def _llm_score_batch(query: str, contents: list[str]) -> list[float] | None:
    """一次请求给所有候选打分。"""
    from app.services import llm_service

    numbered = "\n\n".join(f"[{i}] {text[:800]}" for i, text in enumerate(contents, start=1))
    prompt = (
        "你是检索相关性评估器。请判断每个知识片段与问题的相关程度。\n"
        f"只输出一个 JSON 数组，长度必须是 {len(contents)}，"
        "每个元素是 0 到 10 的数字，顺序与片段编号一致。不要输出其他文字。\n\n"
        f"问题：{query}\n\n片段：\n{numbered}"
    )
    try:
        text = await llm_service.chat([{"role": "user", "content": prompt}], temperature=0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm rerank failed, keep RRF order: {}", exc)
        return None
    scores = parse_score_list(text, len(contents))
    if scores is None:
        logger.warning("llm rerank returned unparsable scores, keep RRF order: {}", text[:200])
    return scores


async def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int | None = None,
    scorer: BatchScorer | None = None,
) -> list[RerankedItem]:
    """对候选片段重排，返回 top_k 个带重排分数的结果。

    输入 candidates 应已按某种策略（如 RRF 融合）初筛，数量通常为 top_k*2，
    这里做精排，把最相关的片段顶到最前。
    """
    if top_k is None:
        top_k = settings.rerank_top_k
    if not candidates or top_k <= 0:
        return []

    if settings.rerank_mode == "llm":
        scorer = scorer or _llm_score_batch
        try:
            scores = await scorer(query, [item.chunk.content for item in candidates])
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm rerank scorer failed, keep RRF order: {}", exc)
            scores = None
        if scores is None or len(scores) != len(candidates):
            if scores is not None:
                logger.warning("llm rerank score count mismatch, keep RRF order")
            return [RerankedItem(chunk=item, rerank_score=item.score) for item in candidates[:top_k]]
        scored = [
            RerankedItem(chunk=item, rerank_score=score)
            for item, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda x: x.rerank_score, reverse=True)
        return scored[:top_k]

    # rrf 模式：直接用融合分数（已按分数降序）
    return [RerankedItem(chunk=item, rerank_score=item.score) for item in candidates[:top_k]]
