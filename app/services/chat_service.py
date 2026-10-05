"""对话业务逻辑：检索 -> 生成 -> 持久化，支持流式与非流式。"""
import hashlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis import get_redis
from app.db.models import Conversation, Message, User
from app.services import llm_service, rag_service
from app.services.rag_service import RetrievedChunk


def _cache_key(kb_id: int, question: str) -> str:
    digest = hashlib.md5(question.strip().lower().encode("utf-8")).hexdigest()
    return f"qa:{kb_id}:{digest}"


async def _get_cached_answer(key: str) -> dict | None:
    """读热点问答缓存；Redis 不可用时返回 None（降级）。"""
    try:
        raw = await get_redis().get(key)
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001
        return None


async def _set_cached_answer(key: str, answer: str, sources: list[dict], ttl: int = 3600) -> None:
    try:
        await get_redis().set(key, json.dumps({"answer": answer, "sources": sources}), ex=ttl)
    except Exception:  # noqa: BLE001
        pass


async def _save_message(
    db: AsyncSession,
    conversation_id: int,
    role: str,
    content: str,
    sources: list | None = None,
) -> Message:
    msg = Message(conversation_id=conversation_id, role=role, content=content, sources=sources)
    db.add(msg)
    await db.flush()
    return msg


# ---------- 会话管理 ----------

async def create_conversation(db: AsyncSession, user: User, kb_id: int, title: str = "新对话") -> Conversation:
    conv = Conversation(user_id=user.id, knowledge_base_id=kb_id, title=title)
    db.add(conv)
    await db.flush()
    return conv


async def list_conversations(db: AsyncSession, user_id: int) -> list[Conversation]:
    return list(
        await db.scalars(
            select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.id.desc()).limit(50)
        )
    )


async def get_conversation(db: AsyncSession, conv_id: int, user_id: int) -> Conversation:
    conv = await db.scalar(select(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user_id))
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    return conv


async def list_messages(db: AsyncSession, conv: Conversation) -> list[Message]:
    return list(await db.scalars(select(Message).where(Message.conversation_id == conv.id).order_by(Message.id)))


async def get_history(db: AsyncSession, conv: Conversation) -> list[dict]:
    """取最近对话历史，转为 LLM messages 格式。"""
    messages = await list_messages(db, conv)
    return [{"role": m.role, "content": m.content} for m in messages[-10:]]


# ---------- 问答 ----------

async def _retrieve(db: AsyncSession, kb_id: int, question: str) -> list[RetrievedChunk]:
    return await rag_service.search_knowledge_base(db, kb_id, question)


async def ask(db: AsyncSession, user: User, conv: Conversation, question: str) -> dict:
    """非流式问答（带缓存）。"""
    cache_key = _cache_key(conv.knowledge_base_id or 0, question)
    cached = await _get_cached_answer(cache_key)
    if cached:
        await _save_message(db, conv.id, "user", question)
        await _save_message(db, conv.id, "assistant", cached["answer"], cached.get("sources"))
        return {"answer": cached["answer"], "sources": cached.get("sources", []), "cached": True}

    chunks = await _retrieve(db, conv.knowledge_base_id or 0, question)
    history = await get_history(db, conv)
    messages = rag_service.build_messages(question, chunks, history)
    answer = await llm_service.chat(messages)
    sources = rag_service.format_sources(chunks)

    await _save_message(db, conv.id, "user", question)
    await _save_message(db, conv.id, "assistant", answer, sources)
    await _set_cached_answer(cache_key, answer, sources)
    return {"answer": answer, "sources": sources, "cached": False}


async def ask_stream(
    db: AsyncSession, user: User, conv: Conversation, question: str
) -> AsyncIterator[str]:
    """流式问答：产出 SSE 事件 JSON 字符串（sources -> delta* -> done）。"""
    cache_key = _cache_key(conv.knowledge_base_id or 0, question)
    cached = await _get_cached_answer(cache_key)

    if cached:
        # 命中缓存：保存消息并原样回放
        await _save_message(db, conv.id, "user", question)
        msg = await _save_message(db, conv.id, "assistant", cached["answer"], cached.get("sources"))
        yield json.dumps({"type": "sources", "sources": cached.get("sources", [])})
        yield json.dumps({"type": "delta", "text": cached["answer"]})
        yield json.dumps({"type": "done", "message_id": msg.id, "cached": True})
        return

    chunks = await _retrieve(db, conv.knowledge_base_id or 0, question)
    sources = rag_service.format_sources(chunks)
    history = await get_history(db, conv)
    messages = rag_service.build_messages(question, chunks, history)

    await _save_message(db, conv.id, "user", question)
    yield json.dumps({"type": "sources", "sources": sources})

    full_answer: list[str] = []
    try:
        async for piece in llm_service.chat_stream(messages):
            full_answer.append(piece)
            yield json.dumps({"type": "delta", "text": piece})
    except Exception as exc:  # noqa: BLE001
        logger.error("stream error: {}", exc)
        yield json.dumps({"type": "error", "detail": "生成回答时出错"})
        return

    answer = "".join(full_answer)
    msg = await _save_message(db, conv.id, "assistant", answer, sources)
    await _set_cached_answer(cache_key, answer, sources)
    yield json.dumps({"type": "done", "message_id": msg.id, "cached": False})


# ---------- 限流 ----------

async def check_rate_limit(user_id: int, limit: int | None = None) -> None:
    """滑动窗口限流：Redis INCR + EXPIRE；Redis 不可用时放行（降级）。"""
    limit = limit or 30
    try:
        r = get_redis()
        minute = datetime.now(UTC).strftime("%Y%m%d%H%M")
        key = f"ratelimit:{user_id}:{minute}"
        count = await r.incr(key)
        if count == 1:
            await r.expire(key, 90)
        if count > limit:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"请求过于频繁，每分钟最多 {limit} 次")
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        pass
