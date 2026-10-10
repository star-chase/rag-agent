"""MCP 协议层测试：握手、错误码、通知、isError 映射、以及 stdout 纯净性。

这一批**不需要数据库**：`handle_message` 是纯函数，工具表用假实现注入。
真正把这层测住，业务工具再怎么改都不会把协议写坏。
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.mcp import protocol
from app.mcp.protocol import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    ProtocolError,
)
from app.mcp.server import handle_message, tools_payload
from app.mcp.tools import Tool, ToolError

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


def fake_tools() -> list[Tool]:
    async def echo(args):
        return f"echo:{args.get('text')}"

    async def boom(_args):
        raise ToolError("业务失败：知识库不存在")

    async def crash(_args):
        raise RuntimeError("意外异常")

    return [
        Tool(
            name="echo",
            description="回显",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=echo,
        ),
        Tool(name="boom", description="业务失败", input_schema={"type": "object"}, handler=boom),
        Tool(name="crash", description="意外异常", input_schema={"type": "object"}, handler=crash),
    ]


def test_initialize_handshake_reports_capabilities():
    response = run(
        handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "pytest"}},
            },
            fake_tools(),
        )
    )
    result = response["result"]
    assert result["protocolVersion"] == protocol.PROTOCOL_VERSION
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "rag-agent"
    assert "search_knowledge" in result["instructions"]


def test_notification_gets_no_response():
    response = run(
        handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}, fake_tools())
    )
    assert response is None


def test_tools_list_exposes_valid_schema():
    response = run(
        handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, fake_tools())
    )
    tools = response["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["echo", "boom", "crash"]
    for tool in tools:
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"


def test_tools_call_returns_text_content():
    response = run(
        handle_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"text": "hi"}},
            },
            fake_tools(),
        )
    )
    result = response["result"]
    assert result["isError"] is False
    assert result["content"] == [{"type": "text", "text": "echo:hi"}]


def test_tool_error_becomes_is_error_not_protocol_error():
    """工具失败必须是 result.isError，不是 JSON-RPC error —— 否则模型看不到原因。"""
    response = run(
        handle_message(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "boom"}},
            fake_tools(),
        )
    )
    assert "error" not in response
    assert response["result"]["isError"] is True
    assert "知识库不存在" in response["result"]["content"][0]["text"]


def test_unexpected_exception_still_returns_is_error():
    response = run(
        handle_message(
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "crash"}},
            fake_tools(),
        )
    )
    assert response["result"]["isError"] is True
    assert "RuntimeError" in response["result"]["content"][0]["text"]


def test_unknown_tool_lists_available_tools():
    response = run(
        handle_message(
            {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "nope"}},
            fake_tools(),
        )
    )
    assert response["result"]["isError"] is True
    assert "echo" in response["result"]["content"][0]["text"]


def test_unknown_method_is_method_not_found():
    response = run(handle_message({"jsonrpc": "2.0", "id": 7, "method": "resources/list"}, fake_tools()))
    assert response["error"]["code"] == METHOD_NOT_FOUND


def test_invalid_params_for_tools_call():
    response = run(
        handle_message(
            {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": 123}},
            fake_tools(),
        )
    )
    assert response["error"]["code"] == INVALID_PARAMS


def test_ping_returns_empty_result():
    response = run(handle_message({"jsonrpc": "2.0", "id": 9, "method": "ping"}, fake_tools()))
    assert response["result"] == {}


def test_parse_errors_are_classified():
    with pytest.raises(ProtocolError) as broken:
        protocol.parse_message("{not json")
    assert broken.value.code == PARSE_ERROR

    with pytest.raises(ProtocolError):
        protocol.parse_message(json.dumps({"id": 1, "method": "ping"}))  # 少了 jsonrpc

    with pytest.raises(ProtocolError):
        protocol.parse_message(json.dumps({"jsonrpc": "2.0", "id": 1}))


def test_tools_payload_shape():
    payload = tools_payload(fake_tools())
    assert set(payload[0]) == {"name", "description", "inputSchema"}


def test_stdio_subprocess_keeps_stdout_pure():
    """真起一个子进程跑协议：stdout 每一行都必须是合法 JSON-RPC（日志不许混进来）。"""
    env = {**os.environ, "MCP_USER_EMAIL": "nobody@example.com"}
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "app.mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    try:
        assert process.stdin and process.stdout
        process.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": protocol.PROTOCOL_VERSION},
                }
            )
            + "\n"
        )
        process.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n")
        process.stdin.flush()

        first = json.loads(process.stdout.readline())
        second = json.loads(process.stdout.readline())
        assert first["id"] == 1 and first["result"]["serverInfo"]["name"] == "rag-agent"
        assert second["id"] == 2
        names = [tool["name"] for tool in second["result"]["tools"]]
        assert "search_knowledge" in names and "upload_document" not in names

        process.stdin.close()
        assert process.wait(timeout=20) == 0
    finally:
        if process.poll() is None:
            process.kill()



def test_stdout_is_taken_over_for_protocol():
    """fd 级接管：print 不再进协议通道，协议流仍然干净。"""
    import os
    import sys as _sys

    from app.mcp.server import take_over_stdout

    saved_fd = os.dup(1)
    saved_stdout, saved_stderr = _sys.stdout, _sys.stderr
    read_fd, write_fd = os.pipe()
    try:
        os.dup2(write_fd, 1)
        os.close(write_fd)
        _sys.stdout = os.fdopen(os.dup(1), "w", buffering=1)

        protocol_out = take_over_stdout()
        print("这行是普通输出，应该走 stderr")  # noqa: T201
        protocol_out.write('{"jsonrpc":"2.0","id":1}\n')
        protocol_out.flush()

        os.set_blocking(read_fd, False)
        captured = os.read(read_fd, 4096).decode()
        assert captured.strip() == '{"jsonrpc":"2.0","id":1}'
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(read_fd)
        _sys.stdout, _sys.stderr = saved_stdout, saved_stderr
