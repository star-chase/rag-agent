"""Embedding 服务。

OpenAI 兼容接口按批发送字符串。豆包多模态接口一次只接受一段文本，
请求体是 `{type, text}`，并用 `dimensions` 指定输出宽度。
"""
import asyncio

import httpx
from loguru import logger

from app.config import settings

_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.embedding_api_key}", "Content-Type": "application/json"}


def embedding_endpoint() -> str:
    base = settings.embedding_base_url.rstrip("/")
    if base.endswith("/embeddings/multimodal") or base.endswith("/embeddings"):
        return base
    return f"{base}/embeddings"


def uses_multimodal() -> bool:
    return embedding_endpoint().endswith("/embeddings/multimodal")


def _as_vector(raw: object) -> list[float]:
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    if not isinstance(raw, list):
        raise ValueError("embedding 字段不是数组")
    return [float(value) for value in raw]


def parse_embedding_batch(body: dict) -> list[list[float]]:
    """兼容 OpenAI 的 data 数组，以及豆包多模态的 data 对象。"""
    data = body["data"]
    if isinstance(data, dict):
        return [_as_vector(data["embedding"])]
    ranked = sorted(data, key=lambda item: item.get("index", 0))
    return [_as_vector(item["embedding"]) for item in ranked]


def _check_dim(vectors: list[list[float]]) -> list[list[float]]:
    expected = settings.embedding_dim
    for vector in vectors:
        if len(vector) != expected:
            raise ValueError(f"模型返回 {len(vector)} 维，配置 EMBEDDING_DIM={expected}")
    return vectors


async def _post_embeddings(payload: dict) -> list[list[float]]:
    # 直连方舟会超时，跟随 HTTPS_PROXY。
    async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=True) as client:
        resp = await client.post(embedding_endpoint(), json=payload, headers=_headers())
        resp.raise_for_status()
        return _check_dim(parse_embedding_batch(resp.json()))


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    if uses_multimodal():
        vectors: list[list[float]] = []
        for index, text in enumerate(texts):
            payload = {
                "model": settings.embedding_model,
                "input": [{"type": "text", "text": text}],
                "dimensions": settings.embedding_dim,
                "encoding_format": "float",
            }
            vectors.extend(await _post_embeddings(payload))
            if index + 1 < len(texts):
                await asyncio.sleep(0.05)
        return vectors
    return await _post_embeddings({"model": settings.embedding_model, "input": texts})


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
