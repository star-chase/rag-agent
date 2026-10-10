"""MCP（Model Context Protocol）服务端：把 rag-agent 的检索能力暴露成标准工具。

模块划分：
- protocol.py  JSON-RPC 2.0 编解码 + 错误码（stdio 传输是换行分隔的 JSON，不是 LSP 分帧）
- tools.py     工具表：name / description / inputSchema / handler
- server.py    stdio 服务：python -m app.mcp.server
"""
