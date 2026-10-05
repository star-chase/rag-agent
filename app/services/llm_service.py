"""LLM 服务：OpenAI 兼容接口的普通与流式对话。

只需配 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，即可接 OpenAI、DeepSeek、通义、Ollama 等任意 OpenAI 兼容服务。
"""
from collections.abc import AsyncIterator

import httpx
from loguru import logger

from app.config import settings

_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}


def _fake_answer(messages: list[dict]) -> str:
    """模板回答：提取系统提示中的知识片段，模拟基于引用的回答。"""
    user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    chunks = [line for line in system.split("\n") if line.startswith("[") and "]" in line and "(来源:" in line]
    if chunks:
        body = f"根据知识库，关于“{user_text[:40]}”的回答如下：\n\n{chunks[0][:300]}"
        return body + "\n\n参考资料：[1]"
    return f"（fake 模式）未检索到相关片段。问题：{user_text[:80]}"


async def chat(messages: list[dict], temperature: float | None = None) -> str:
    """普通（非流式）对话，返回完整文本。"""
    if settings.llm_mode == "fake":
        return _fake_answer(messages)
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature if temperature is not None else settings.llm_temperature,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
        resp = await client.post(f"{settings.llm_base_url}/chat/completions", json=payload, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""


async def chat_stream(messages: list[dict], temperature: float | None = None) -> AsyncIterator[str]:
    """流式对话，逐个产出增量文本（SSE）。"""
    if settings.llm_mode == "fake":
        answer = _fake_answer(messages)
        for piece in answer:
            yield piece
        return
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature if temperature is not None else settings.llm_temperature,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
        async with client.stream(
            "POST", f"{settings.llm_base_url}/chat/completions", json=payload, headers=_headers()
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    import json

                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    yield piece


async def check_llm() -> bool:
    """健康检查：做一次极短的对话。"""
    try:
        text = await chat([{"role": "user", "content": "ping，只回复 pong"}], temperature=0)
        return "pong" in text.lower() or bool(text)
    except Exception as e:  # noqa: BLE001
        logger.warning("llm health check failed: {}", e)
        return False
