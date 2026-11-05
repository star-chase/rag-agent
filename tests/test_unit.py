"""纯逻辑单元测试：分块器、密码、JWT、rerank、RRF。"""
import pytest

from app.core.security import create_access_token, decode_token, hash_password, verify_password
from app.services.ingest_service import chunk_text
from app.services.rag_service import RetrievedChunk, _rrf_fuse
from app.services.rerank_service import rerank


def test_chunk_text_short():
    chunks = chunk_text("你好世界。\n\n这是第二段。", chunk_size=1000, overlap=0)
    assert len(chunks) == 1
    assert "你好世界" in chunks[0]


def test_chunk_text_hard_split_with_overlap():
    text = "词" * 500
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) >= 3
    # 相邻块应共享 overlap 部分
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_text_paragraph_merge():
    paras = "\n\n".join(f"段落{i} " + "内容" * 30 for i in range(10))
    chunks = chunk_text(paras, chunk_size=300, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_password_hash_roundtrip():
    hashed = hash_password("secret123")
    assert hashed != "secret123"
    assert verify_password("secret123", hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_roundtrip():
    token = create_access_token("42", {"role": "user"})
    payload = decode_token(token)
    assert payload["sub"] == "42"
    assert payload["role"] == "user"


def test_rrf_ranks_each_channel_independently():
    """每一路从第 1 名计分。向量第 1 与关键词第 1 的权重相同，双冠军是 2/(k+1)。"""
    # _rrf_fuse 的入参是检索返回的 Chunk，不是已经包过的 RetrievedChunk。
    jia, yi, bing = (
        _fake_chunk(1, "甲").chunk,
        _fake_chunk(2, "乙").chunk,
        _fake_chunk(3, "丙").chunk,
    )
    k = 60
    fused = _rrf_fuse([jia, yi, bing], [bing, jia], top_k=3, k=k)
    scores = {item.chunk.id: item.score for item in fused}

    assert scores[1] == pytest.approx(1 / (k + 1) + 1 / (k + 2))
    assert scores[2] == pytest.approx(1 / (k + 2))
    assert scores[3] == pytest.approx(1 / (k + 3) + 1 / (k + 1))
    assert [item.chunk.id for item in fused] == [1, 3, 2]

    both_first = _rrf_fuse([jia], [yi], top_k=2, k=k)
    assert both_first[0].score == pytest.approx(1 / (k + 1))
    assert both_first[1].score == pytest.approx(1 / (k + 1))

    double = _rrf_fuse([jia], [jia], top_k=1, k=k)
    assert double[0].score == pytest.approx(2 / (k + 1))


def _fake_chunk(chunk_id: int, content: str, score: float = 1.0) -> RetrievedChunk:
    from types import SimpleNamespace

    return RetrievedChunk(chunk=SimpleNamespace(id=chunk_id, content=content), score=score)


async def test_rerank_rrf_mode():
    """rrf 模式：按融合分数取 top_k，保持原顺序。"""
    candidates = [
        _fake_chunk(1, "a", score=0.9),
        _fake_chunk(2, "b", score=0.5),
        _fake_chunk(3, "c", score=0.2),
    ]
    result = await rerank("question", candidates, top_k=2)
    assert [r.chunk.chunk.id for r in result] == [1, 2]
    assert result[0].rerank_score == 0.9


async def test_rerank_llm_mode_with_custom_scorer():
    """llm 模式：一次给全部分数，按分数降序重排。"""
    from app.config import settings

    candidates = [
        _fake_chunk(1, "只谈论天气的片段", score=0.3),
        _fake_chunk(2, "与问题高度相关的片段", score=0.4),
    ]

    async def scorer(query: str, contents: list[str]) -> list[float]:
        return [9.0 if "相关" in content else 1.0 for content in contents]

    old_mode = settings.rerank_mode
    settings.rerank_mode = "llm"
    try:
        result = await rerank("q", candidates, top_k=2, scorer=scorer)
    finally:
        settings.rerank_mode = old_mode
    assert [r.chunk.chunk.id for r in result] == [2, 1]
    assert result[0].rerank_score == 9.0


def test_parse_embedding_batch_accepts_doubao_object():
    from app.services.embedding_service import parse_embedding_batch

    body = {"data": {"embedding": [0.1, 0.2], "object": "embedding"}}
    assert parse_embedding_batch(body) == [[0.1, 0.2]]
    openai = {"data": [{"index": 1, "embedding": [3.0]}, {"index": 0, "embedding": [1.0, 2.0]}]}
    assert parse_embedding_batch(openai) == [[1.0, 2.0], [3.0]]


def test_parse_score_list_rejects_prose_with_zero():
    from app.services.rerank_service import parse_score_list

    assert parse_score_list("0-10 分，我给 8", 2) is None
    assert parse_score_list("[1, 8]", 2) == [1.0, 8.0]


def test_chunk_overlap_bounds():
    import pytest

    from app.services.ingest_service import chunk_text

    text = "词" * 250
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert chunks[1] == text[100:200]
    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=0, overlap=0)


