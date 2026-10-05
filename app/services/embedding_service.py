"""Embedding 服务：OpenAI 兼容接口，支持批量与分批。"""
import asyncio

import httpx
from loguru import logger

from app.config import settings

_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.embedding_api_key}", "Content-Type": "application/json"}


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    payload = {"model": settings.embedding_model, "input": texts}
    async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
        resp = await client.post(f"{settings.embedding_base_url}/embeddings", json=payload, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
        ranked = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ranked]


def _fake_embedding(text: str) -> list[float]:
    """确定性伪向量：同一文本永远得到同一向量（供无 key 时调试/测试）。"""
    import hashlib

    dim = settings.embedding_dim
    vec = []
    for i in range(dim):
        h = hashlib.sha256(f"{text}:::{i}".encode()).digest()
        vec.append(int.from_bytes(h[:4], "big") / 2**32 - 0.5)
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """按 batch 分批向量化（控制单次请求大小与超时）。"""
    if settings.embedding_mode == "fake":
        return [_fake_embedding(t) for t in texts]
    results: list[list[float]] = []
    batch_size = settings.embedding_batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results.extend(await _embed_batch(batch))
        if i + batch_size < len(texts):
            await asyncio.sleep(0.05)  # 轻微限速，避免 429
    return results


async def embed_query(text: str) -> list[float]:
    if settings.embedding_mode == "fake":
        return _fake_embedding(text)
    return (await _embed_batch([text]))[0]


async def check_embedding() -> bool:
    """健康检查：向量化一句话。"""
    try:
        vec = await embed_query("ping")
        return isinstance(vec, list) and len(vec) > 0
    except Exception as e:  # noqa: BLE001
        logger.warning("embedding health check failed: {}", e)
        return False
