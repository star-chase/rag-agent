"""MCP 工具表：把 rag-agent 的检索/问答能力描述成模型可调用的工具。

约定：
- 每个 handler 是 `async def (args: dict) -> str`，返回给模型的文本；
- **业务失败抛 `ToolError`**（协议层映射成 `isError: true`），不抛 HTTPException、不返回半个结果；
- 身份来自环境变量 `MCP_USER_EMAIL`：MCP server 是本机进程，不走 HTTP 鉴权，也不能凭空造一个用户。

环境变量：
- `MCP_USER_EMAIL`（必填）  以哪个用户的身份检索，所有查询强制按该用户过滤；
- `MCP_DEFAULT_KB_ID`（可选）未显式传 kb_id 时用哪个知识库；
- `MCP_ALLOW_WRITE=1`（可选）才把 `upload_document` 挂进工具表（默认只读）。
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.db.models.chunk import Chunk
from app.db.models.document import DOC_STATUS_READY, Document
from app.db.models.knowledge_base import KnowledgeBase
from app.db.models.user import User
from app.db.session import AsyncSessionLocal
from app.services import embedding_service, knowledge_base_service, llm_service, rag_service
from app.services.ingest_service import chunk_text

# 单个工具结果给模型的字符上限：检索片段太长会把上下文顶爆，超过就截断并说明。
MAX_RESULT_CHARS = 4000


class ToolError(Exception):
    """业务失败（参数不对、知识库不存在、身份缺失……）→ `isError: true`。"""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def _user_email() -> str:
    email = os.environ.get("MCP_USER_EMAIL", "").strip()
    if not email:
        raise ToolError("MCP server 未配置 MCP_USER_EMAIL，无法确定检索身份")
    return email


def _default_kb_id() -> int | None:
    raw = os.environ.get("MCP_DEFAULT_KB_ID", "").strip()
    return int(raw) if raw.isdigit() else None


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…（结果已截断，原始长度 {len(text)} 字符）"


async def _load_user(db: Any) -> User:
    email = _user_email()
    user = await db.scalar(select(User).where(User.email == email))
    if user is None:
        raise ToolError(f"MCP_USER_EMAIL 指向的用户不存在：{email}")
    return user


async def _resolve_kb_id(db: Any, user: User, args: dict[str, Any]) -> int:
    """kb_id 显式传入优先，其次取服务端默认；**始终校验归属**。"""
    kb_id = args.get("kb_id")
    if kb_id is None:
        kb_id = _default_kb_id()
    if kb_id is None:
        raise ToolError("未指定 kb_id，且服务端没有配置 MCP_DEFAULT_KB_ID")
    if not isinstance(kb_id, int) or isinstance(kb_id, bool):
        raise ToolError("kb_id 必须是整数")
    kb = await db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id, KnowledgeBase.owner_id == user.id
        )
    )
    if kb is None:
        raise ToolError(f"知识库 {kb_id} 不存在或不属于当前身份")
    return kb.id


def _top_k(args: dict[str, Any]) -> int:
    raw = args.get("top_k", 5)
    if not isinstance(raw, int) or isinstance(raw, bool) or not (1 <= raw <= 20):
        raise ToolError("top_k 必须是 1~20 的整数")
    return raw


def _require_text(args: dict[str, Any], key: str, max_len: int = 2000) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{key} 不能为空")
    if len(value) > max_len:
        raise ToolError(f"{key} 太长（上限 {max_len} 字符）")
    return value.strip()


async def search_knowledge(args: dict[str, Any]) -> str:
    query = _require_text(args, "query")
    top_k = _top_k(args)
    async with AsyncSessionLocal() as db:
        user = await _load_user(db)
        kb_id = await _resolve_kb_id(db, user, args)
        chunks = await rag_service.search_knowledge_base(db, kb_id, query, top_k=top_k)
        sources = rag_service.format_sources(chunks)

    if not chunks:
        return f"知识库 {kb_id} 中没有检索到与「{query}」相关的片段。"

    lines = [f"知识库 {kb_id} 命中 {len(sources)} 个片段："]
    for index, source in enumerate(sources, start=1):
        lines.append(
            f"[{index}] {source['filename']}（score={source['score']}）\n{source['snippet']}"
        )
    return _truncate("\n".join(lines))


async def ask_knowledge(args: dict[str, Any]) -> str:
    question = _require_text(args, "question")
    top_k = _top_k(args)
    async with AsyncSessionLocal() as db:
        user = await _load_user(db)
        kb_id = await _resolve_kb_id(db, user, args)
        chunks = await rag_service.search_knowledge_base(db, kb_id, question, top_k=top_k)

    # 没检索到就明确拒答：这是产品语义，不是错误。
    if not chunks:
        return f"知识库中没有相关信息，无法回答「{question}」。"

    answer = await llm_service.chat(rag_service.build_messages(question, chunks, None))
    sources = rag_service.format_sources(chunks)
    citations = "\n".join(
        f"[{index}] {source['filename']}：{source['snippet'][:80]}"
        for index, source in enumerate(sources, start=1)
    )
    return _truncate(f"{answer}\n\n引用来源：\n{citations}")


async def list_knowledge_bases(args: dict[str, Any]) -> str:  # noqa: ARG001 —— 工具签名统一收 args
    async with AsyncSessionLocal() as db:
        user = await _load_user(db)
        rows = await knowledge_base_service.list_knowledge_bases(db, user.id)

    if not rows:
        return "当前身份下没有知识库。"
    lines = ["当前身份可见的知识库："]
    for row in rows:
        lines.append(
            f"- id={row['id']} name={row['name']} 文档 {row['document_count']} 篇"
            f"（就绪 {row['ready_count']}）"
        )
    return "\n".join(lines)


async def upload_document(args: dict[str, Any]) -> str:
    """写工具：默认不挂载；即使挂载了，这里仍然再判一次环境变量（fail closed）。"""
    if os.environ.get("MCP_ALLOW_WRITE") != "1":
        raise ToolError("写操作未开启：需要设置 MCP_ALLOW_WRITE=1")

    filename = _require_text(args, "filename", max_len=200)
    content = _require_text(args, "content", max_len=200_000)
    async with AsyncSessionLocal() as db:
        user = await _load_user(db)
        kb_id = await _resolve_kb_id(db, user, args)
        document = Document(
            knowledge_base_id=kb_id,
            filename=filename,
            storage_key=f"mcp/{kb_id}/{uuid.uuid4().hex}/{filename}",
            file_type="md" if filename.endswith(".md") else "txt",
            file_size=len(content.encode("utf-8")),
            status=DOC_STATUS_READY,
            chunk_count=0,
        )
        db.add(document)
        await db.flush()

        chunks = chunk_text(content)
        vectors = await embedding_service.embed_texts(chunks)
        for index, (piece, vector) in enumerate(zip(chunks, vectors, strict=True)):
            db.add(
                Chunk(
                    document_id=document.id,
                    knowledge_base_id=kb_id,
                    seq=index,
                    content=piece,
                    embedding=vector,
                    meta={
                        "filename": filename,
                        "chunk_index": index,
                        "chunk_count": len(chunks),
                        "source": "mcp",
                    },
                )
            )
        document.chunk_count = len(chunks)
        await db.commit()
        document_id, chunk_count = document.id, len(chunks)

    return f"已写入知识库 {kb_id}：document_id={document_id}，{chunk_count} 个分块。"


def build_tools(include_write: bool | None = None) -> list[Tool]:
    """组装工具表。写工具默认不出现——客户端看不到，也就不会尝试。"""
    tools = [
        Tool(
            name="search_knowledge",
            description="在知识库中做混合检索（向量 + 关键词 + RRF + 精排），返回片段、来源文件名与分数。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词或问题"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                    "kb_id": {"type": "integer", "description": "知识库 id，不传则用服务端默认知识库"},
                },
                "required": ["query"],
            },
            handler=search_knowledge,
        ),
        Tool(
            name="ask_knowledge",
            description="基于知识库回答问题，返回带 [1][2] 引用的答案；知识库没有依据时明确拒答。",
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要回答的问题"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                    "kb_id": {"type": "integer", "description": "知识库 id，不传则用服务端默认知识库"},
                },
                "required": ["question"],
            },
            handler=ask_knowledge,
        ),
        Tool(
            name="list_knowledge_bases",
            description="列出当前身份可见的知识库及其文档数量。",
            input_schema={"type": "object", "properties": {}},
            handler=list_knowledge_bases,
        ),
    ]

    if include_write is None:
        include_write = os.environ.get("MCP_ALLOW_WRITE") == "1"
    if include_write:
        tools.append(
            Tool(
                name="upload_document",
                description="把一段文本写进知识库（按段落分块并向量化），返回 document_id。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string"},
                        "kb_id": {"type": "integer"},
                    },
                    "required": ["filename", "content"],
                },
                handler=upload_document,
            )
        )
    return tools
