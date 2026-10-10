"""MCP 工具测试：真连数据库（PG + fake embedding/LLM），验证检索、问答、身份与越权。

身份来自 `MCP_USER_EMAIL`，所以每个用例都要显式设置它 —— 顺便也验证了"没配身份就报错"这条。
"""

from __future__ import annotations

import pytest

from app.mcp.tools import (
    ToolError,
    ask_knowledge,
    build_tools,
    list_knowledge_bases,
    search_knowledge,
    upload_document,
)

DOC_TEXT = "第一条：员工每周工作五天，周末双休。\n\n第二条：年假五天起，每满一年增加一天。"


async def seed_kb(client, auth_headers) -> int:
    """用真实 API 建库 + 上传文档（走完整摄入链路：分块 → 向量化 → 落库）。"""
    resp = await client.post(
        "/api/v1/knowledge-bases", json={"name": "MCP 测试库"}, headers=auth_headers
    )
    kb_id = resp.json()["id"]
    resp = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        headers=auth_headers,
        files={"file": ("rules.txt", DOC_TEXT.encode(), "text/plain")},
    )
    assert resp.status_code == 201
    return kb_id


@pytest.fixture
def as_u1(monkeypatch):
    monkeypatch.setenv("MCP_USER_EMAIL", "u1@test.com")


async def test_search_knowledge_returns_sources(client, auth_headers, as_u1, monkeypatch):
    kb_id = await seed_kb(client, auth_headers)
    monkeypatch.setenv("MCP_DEFAULT_KB_ID", str(kb_id))

    text = await search_knowledge({"query": "年假"})

    assert "rules.txt" in text
    assert "年假五天起" in text
    assert "score=" in text


async def test_search_knowledge_with_explicit_kb_id(client, auth_headers, as_u1):
    kb_id = await seed_kb(client, auth_headers)

    text = await search_knowledge({"query": "周末", "kb_id": kb_id, "top_k": 3})

    assert "周末双休" in text


async def test_ask_knowledge_returns_answer_with_citations(client, auth_headers, as_u1, monkeypatch):
    kb_id = await seed_kb(client, auth_headers)
    monkeypatch.setenv("MCP_DEFAULT_KB_ID", str(kb_id))

    text = await ask_knowledge({"question": "员工每周工作几天？"})

    assert "引用来源" in text
    assert "rules.txt" in text


async def make_empty_kb(client, auth_headers) -> int:
    resp = await client.post("/api/v1/knowledge-bases", json={"name": "空库"}, headers=auth_headers)
    return resp.json()["id"]


async def test_ask_knowledge_refuses_on_empty_kb(client, auth_headers, as_u1, monkeypatch):
    """空知识库 → 检索为空 → 明确拒答，而不是编一个答案。"""
    monkeypatch.setenv("MCP_DEFAULT_KB_ID", str(await make_empty_kb(client, auth_headers)))

    text = await ask_knowledge({"question": "年假几天？"})

    assert "没有相关信息" in text


async def test_search_knowledge_reports_no_hits_on_empty_kb(client, auth_headers, as_u1, monkeypatch):
    monkeypatch.setenv("MCP_DEFAULT_KB_ID", str(await make_empty_kb(client, auth_headers)))

    text = await search_knowledge({"query": "年假"})

    assert "没有检索到" in text


async def test_irrelevant_query_still_returns_candidates(client, auth_headers, as_u1, monkeypatch):
    """记录真实行为：库里有内容时，混合检索没有相关性阈值，不相关的问题也会拿到候选片段。

    这是评测（evals/report.md）里已经记录的问题，改造方向是相关性阈值 / 空检索短路。
    """
    kb_id = await seed_kb(client, auth_headers)
    monkeypatch.setenv("MCP_DEFAULT_KB_ID", str(kb_id))

    text = await search_knowledge({"query": "火星殖民计划"})

    assert "命中" in text


async def test_list_knowledge_bases(client, auth_headers, as_u1):
    kb_id = await seed_kb(client, auth_headers)

    text = await list_knowledge_bases({})

    assert f"id={kb_id}" in text
    assert "MCP 测试库" in text


async def test_missing_identity_is_tool_error(client, auth_headers, monkeypatch):
    monkeypatch.delenv("MCP_USER_EMAIL", raising=False)

    with pytest.raises(ToolError) as excinfo:
        await search_knowledge({"query": "年假"})

    assert "MCP_USER_EMAIL" in str(excinfo.value)


async def test_unknown_user_is_tool_error(client, auth_headers, monkeypatch):
    monkeypatch.setenv("MCP_USER_EMAIL", "nobody@example.com")

    with pytest.raises(ToolError):
        await search_knowledge({"query": "年假"})


async def test_other_users_kb_is_rejected(client, auth_headers, as_u1):
    """越权知识库一律当作不存在：不泄漏"这个 id 存在"这个事实。"""
    kb_id = await seed_kb(client, auth_headers)

    with pytest.raises(ToolError) as excinfo:
        await search_knowledge({"query": "年假", "kb_id": kb_id + 999})

    assert "不存在" in str(excinfo.value)


async def test_invalid_arguments_are_rejected(as_u1):
    with pytest.raises(ToolError):
        await search_knowledge({"query": "  "})
    with pytest.raises(ToolError):
        await search_knowledge({"query": "年假", "top_k": 999})
    with pytest.raises(ToolError):
        await search_knowledge({"query": "年假", "kb_id": "not-an-int"})


def test_write_tool_is_hidden_by_default(monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_WRITE", raising=False)
    assert "upload_document" not in [tool.name for tool in build_tools()]

    monkeypatch.setenv("MCP_ALLOW_WRITE", "1")
    assert "upload_document" in [tool.name for tool in build_tools()]


async def test_upload_document_requires_flag(client, auth_headers, as_u1, monkeypatch):
    kb_id = await seed_kb(client, auth_headers)
    monkeypatch.delenv("MCP_ALLOW_WRITE", raising=False)

    with pytest.raises(ToolError):
        await upload_document({"filename": "new.md", "content": DOC_TEXT, "kb_id": kb_id})


async def test_upload_document_then_search_finds_it(client, auth_headers, as_u1, monkeypatch):
    kb_id = await seed_kb(client, auth_headers)
    monkeypatch.setenv("MCP_ALLOW_WRITE", "1")

    result = await upload_document(
        {
            "filename": "mcp-note.md",
            "content": "报销标准：一线城市住宿每晚不超过六百元。",
            "kb_id": kb_id,
        }
    )

    assert "document_id=" in result
    found = await search_knowledge({"query": "住宿", "kb_id": kb_id})
    assert "mcp-note.md" in found
