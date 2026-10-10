"""MCP server（stdio）：把 rag-agent 的检索能力暴露成标准工具。

    python -m app.mcp.server                # 跑 stdio 服务（给 MCP 客户端连）
    python -m app.mcp.server --list-tools   # 打印工具 schema（自检 / CI 用，不是协议会话）

环境变量见 `app/mcp/tools.py` 的模块说明。

为什么单开一个线程读 stdin：整个进程只跑**一个事件循环**。
SQLAlchemy 的 async 引擎连接池是绑在事件循环上的，如果每条消息都 `asyncio.run()` 一次，
第二次就会拿到属于已关闭循环的连接 —— 报错信息还特别难懂。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
from typing import Any

from loguru import logger

from app.mcp import protocol
from app.mcp.protocol import ProtocolError
from app.mcp.tools import Tool, ToolError, build_tools

SERVER_NAME = "rag-agent"
SERVER_VERSION = "0.1.0"
INSTRUCTIONS = (
    "rag-agent 检索工具：search_knowledge（混合检索，返回来源与分数）、"
    "ask_knowledge（带引用回答，无依据时拒答）、list_knowledge_bases。"
)


def tools_payload(tools: list[Tool]) -> list[dict[str, Any]]:
    return [tool.to_wire() for tool in tools]


def take_over_stdout() -> Any:
    """把 fd 1 让给协议：协议写 dup 出来的原始 fd，其它标准输出全部改道 stderr。

    为什么必须做到 fd 级（实测踩过两次）：
    1. `print()` / loguru / uvicorn 之类在 Python 层写 `sys.stdout`，换 `sys.stdout` 能拦住；
    2. 但 SQLAlchemy 的 `echo=True` 在 **engine 创建时**（import 期）就抓住了原始 stdout 对象，
       之后它一直往那条老通道写 —— 只换 `sys.stdout` 拦不住，协议里就混进了 SQL 日志。
    所以这里 dup 出协议专用流，再用 `os.dup2` 把 fd 1（真正的标准输出）指向 stderr。
    """
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr  # 让 print / 新建的 StreamHandler 也走 stderr
    return protocol_out


async def handle_message(message: dict[str, Any], tools: list[Tool]) -> dict[str, Any] | None:
    """处理一条消息，返回响应；通知类消息返回 None（协议规定不能回响应）。

    故意设计成不碰 stdin/stdout 的纯函数：协议测试可以直接喂消息进来断言结果。
    """
    request_id = message.get("id")
    method = message["method"]

    if protocol.is_notification(message):
        # notifications/initialized 等一律忽略：协议里它们不需要、也不允许有响应。
        return None

    if method == "initialize":
        params = message.get("params") or {}
        return protocol.success(
            request_id,
            {
                "protocolVersion": protocol.PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": INSTRUCTIONS
                + (
                    f"（客户端协议版本 {params.get('protocolVersion')}，"
                    f"本服务按 {protocol.PROTOCOL_VERSION} 应答）"
                    if params.get("protocolVersion")
                    else ""
                ),
            },
        )

    if method == "ping":
        return protocol.success(request_id, {})

    if method == "tools/list":
        return protocol.success(request_id, {"tools": tools_payload(tools)})

    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return protocol.failure(
                request_id, protocol.INVALID_PARAMS, "params 需要 {name: string, arguments: object}"
            )
        tool = next((item for item in tools if item.name == name), None)
        if tool is None:
            # 未知"工具"按业务失败返回：模型能从内容里读到可用工具列表，比协议错误更有用。
            available = "、".join(item.name for item in tools)
            return protocol.tool_result(
                request_id, f"未知工具：{name}。可用工具：{available}", is_error=True
            )
        try:
            text = await tool.handler(arguments)
        except ToolError as exc:
            logger.warning("mcp_tool_error", tool=name, reason=str(exc))
            return protocol.tool_result(request_id, str(exc), is_error=True)
        except Exception as exc:  # noqa: BLE001 —— 兜底：任何异常都不能把 server 打挂
            logger.exception("mcp_tool_crashed", tool=name)
            return protocol.tool_result(
                request_id, f"工具执行失败：{type(exc).__name__}: {exc}", is_error=True
            )
        return protocol.tool_result(request_id, text)

    return protocol.failure(request_id, protocol.METHOD_NOT_FOUND, f"不支持的方法：{method}")


async def serve(tools: list[Tool]) -> int:
    """读 stdin → 分发 → 写 stdout。所有响应串行写出，避免多路交错把协议写坏。"""
    protocol_out = take_over_stdout()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def reader() -> None:
        for line in sys.stdin:
            loop.call_soon_threadsafe(queue.put_nowait, line)
        loop.call_soon_threadsafe(queue.put_nowait, None)  # EOF：让主循环收尾

    threading.Thread(target=reader, name="mcp-stdin", daemon=True).start()

    while True:
        line = await queue.get()
        if line is None:
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            message = protocol.parse_message(line)
            response = await handle_message(message, tools)
        except ProtocolError as exc:
            # 编解码失败时拿不到 id，按 JSON-RPC 规定回 id=null
            response = protocol.failure(None, exc.code, exc.message, exc.data)
        except Exception as exc:  # noqa: BLE001
            response = protocol.failure(
                None, protocol.INTERNAL_ERROR, f"服务内部错误：{type(exc).__name__}: {exc}"
            )
        if response is not None:
            protocol_out.write(protocol.dumps(response) + "\n")
            protocol_out.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rag-agent MCP server（stdio）")
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="打印工具 schema（JSON）后退出：自检用，不进入协议会话",
    )
    args = parser.parse_args(argv)

    tools = build_tools()
    if args.list_tools:
        print(protocol.dumps({"tools": tools_payload(tools)}))
        return 0

    logger.info("mcp_server_starting", server=SERVER_NAME, version=SERVER_VERSION, tools=len(tools))
    return asyncio.run(serve(tools))


if __name__ == "__main__":
    raise SystemExit(main())
