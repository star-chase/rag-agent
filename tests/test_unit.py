"""纯逻辑单元测试：分块器、密码、JWT、rerank。"""
from app.core.security import create_access_token, decode_token, hash_password, verify_password
from app.services.ingest_service import chunk_text
from app.services.rag_service import RetrievedChunk
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
    """llm 模式：用自定义打分器验证按分数降序重排。"""
    from app.config import settings

    candidates = [
        _fake_chunk(1, "只谈论天气的片段", score=0.3),
        _fake_chunk(2, "与问题高度相关的片段", score=0.4),
    ]
    # 打分器：片段含"相关"得分高
    async def scorer(query: str, content: str) -> float:
        return 9.0 if "相关" in content else 1.0

    old_mode = settings.rerank_mode
    settings.rerank_mode = "llm"
    try:
        result = await rerank("q", candidates, top_k=2, scorer=scorer)
    finally:
        settings.rerank_mode = old_mode
    assert [r.chunk.chunk.id for r in result] == [2, 1]
    assert result[0].rerank_score == 9.0
