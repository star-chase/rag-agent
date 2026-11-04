"""对话业务逻辑：检索 -> 生成 -> 持久化，支持流式与非流式。"""
import hashlib
import json
import time
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis import get_redis
from app.db.models import Conversation, KnowledgeBase, Message, User
from app.services import llm_service, rag_service
from app.services.rag_service import REFUSAL_TEXT, RetrievedChunk

_WINDOW_SECONDS = 60

# 检查和写入放在同一段 Lua 里。先数再加会在并发下都通过。
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
  return 0
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, math.floor(window * 1000))
return 1
"""


def _cache_key(kb_id: int, question: str) -> str:
    digest = hashlib.md5(question.strip().lower().encode("utf-8")).hexdigest()
    return f"qa:{kb_id}:{digest}"


async def invalidate_kb_cache(kb_id: int) -> None:
    """文档入库或删除后清掉该库的问答缓存。失败只记日志，不让摄入失败。"""
    pattern = f"qa:{kb_id}:*"
    try:
        client = get_redis()
        batch: list[str] = []
        async for key in client.scan_iter(match=pattern, count=100):
            batch.append(key)
            if len(batch) >= 100:
                await client.delete(*batch)
                batch.clear()
        if batch:
            await client.delete(*batch)
    except Exception as exc:  # noqa: BLE001
        logger.warning("invalidate qa cache failed for kb {}: {}", kb_id, exc)


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


async def _require_owned_kb(db: AsyncSession, user: User, kb_id: int | None) -> int:
    """知识库必须属于当前用户。不存在和别人的库都返回 404，避免泄露存在性。"""
    if kb_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    owned = await db.scalar(
        select(KnowledgeBase.id).where(KnowledgeBase.id == kb_id, KnowledgeBase.owner_id == user.id)
    )
    if owned is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    return int(owned)


# ---------- 会话管理 ----------

async def create_conversation(db: AsyncSession, user: User, kb_id: int, title: str = "新对话") -> Conversation:
    await _require_owned_kb(db, user, kb_id)
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
    """非流式问答（带缓存）。检索为空时直接拒答，不调用模型。"""
    kb_id = await _require_owned_kb(db, user, conv.knowledge_base_id)
    cache_key = _cache_key(kb_id, question)
    cached = await _get_cached_answer(cache_key)
    if cached:
        await _save_message(db, conv.id, "user", question)
        await _save_message(db, conv.id, "assistant", cached["answer"], cached.get("sources"))
        return {"answer": cached["answer"], "sources": cached.get("sources", []), "cached": True}

    chunks = await _retrieve(db, kb_id, question)
    if not chunks:
        await _save_message(db, conv.id, "user", question)
        await _save_message(db, conv.id, "assistant", REFUSAL_TEXT, [])
        await _set_cached_answer(cache_key, REFUSAL_TEXT, [])
        return {"answer": REFUSAL_TEXT, "sources": [], "cached": False}

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
    kb_id = await _require_owned_kb(db, user, conv.knowledge_base_id)
    cache_key = _cache_key(kb_id, question)
    cached = await _get_cached_answer(cache_key)

    if cached:
        # 命中缓存：保存消息并原样回放
        await _save_message(db, conv.id, "user", question)
        msg = await _save_message(db, conv.id, "assistant", cached["answer"], cached.get("sources"))
        yield json.dumps({"type": "sources", "sources": cached.get("sources", [])})
        yield json.dumps({"type": "delta", "text": cached["answer"]})
        yield json.dumps({"type": "done", "message_id": msg.id, "cached": True})
        return

    chunks = await _retrieve(db, kb_id, question)
    if not chunks:
        await _save_message(db, conv.id, "user", question)
        msg = await _save_message(db, conv.id, "assistant", REFUSAL_TEXT, [])
        await _set_cached_answer(cache_key, REFUSAL_TEXT, [])
        yield json.dumps({"type": "sources", "sources": []})
        yield json.dumps({"type": "delta", "text": REFUSAL_TEXT})
        yield json.dumps({"type": "done", "message_id": msg.id, "cached": False})
        return

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
    """滑动窗口限流：Redis 有序集合只保留最近 60 秒。Redis 不可用时打日志并放行。"""
    if limit is None:
        limit = 30
    now = time.time()
    key = f"ratelimit:{user_id}"
    member = f"{now}:{uuid4().hex}"
    try:
        allowed = await get_redis().eval(
            _SLIDING_WINDOW_LUA,
            1,
            key,
            str(now),
            str(_WINDOW_SECONDS),
            str(limit),
            member,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("rate limit redis unavailable, allow request: {}", exc)
        return
    if int(allowed) == 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"请求过于频繁，每分钟最多 {limit} 次",
        )
