"""API 集成测试：认证 -> 知识库 -> 文档摄入 -> 问答 全链路。"""


async def test_register_login(client):
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "a@test.com", "password": "password123", "name": "Alice"},
    )
    assert resp.status_code == 201
    assert resp.json()["email"] == "a@test.com"

    # 重复注册应 409
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "a@test.com", "password": "password123", "name": "Alice"},
    )
    assert resp.status_code == 409

    # 登录成功
    resp = await client.post(
        "/api/v1/auth/login", data={"username": "a@test.com", "password": "password123"}
    )
    assert resp.status_code == 200
    assert resp.json()["access_token"]

    # 错误密码 401
    resp = await client.post(
        "/api/v1/auth/login", data={"username": "a@test.com", "password": "wrong-pass"}
    )
    assert resp.status_code == 401


async def test_kb_crud_and_isolation(client, auth_headers):
    # 创建知识库
    resp = await client.post(
        "/api/v1/knowledge-bases",
        json={"name": "KB-1", "description": "测试"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    kb_id = resp.json()["id"]

    # 列表可见
    resp = await client.get("/api/v1/knowledge-bases", headers=auth_headers)
    assert resp.status_code == 200
    assert any(kb["id"] == kb_id for kb in resp.json())

    # 未登录访问应 401
    resp = await client.get("/api/v1/knowledge-bases")
    assert resp.status_code == 401

    # 另一个用户访问他人知识库应 404（多租户隔离）
    await client.post(
        "/api/v1/auth/register",
        json={"email": "b@test.com", "password": "password123", "name": "Bob"},
    )
    resp = await client.post(
        "/api/v1/auth/login", data={"username": "b@test.com", "password": "password123"}
    )
    bob_headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    resp = await client.get(f"/api/v1/knowledge-bases/{kb_id}", headers=bob_headers)
    assert resp.status_code == 404


async def test_document_ingest_and_ask(client, auth_headers):
    # 建知识库
    resp = await client.post(
        "/api/v1/knowledge-bases", json={"name": "制度库"}, headers=auth_headers
    )
    kb_id = resp.json()["id"]

    # 上传 txt 文档
    content = "第一条：员工每周工作五天。\n\n第二条：年假五天起。".encode()
    resp = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        headers=auth_headers,
        files={"file": ("rules.txt", content, "text/plain")},
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    # eager 模式同步处理，应已 ready 且 chunk_count > 0
    resp = await client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] > 0

    # 创建对话并提问
    resp = await client.post(
        "/api/v1/conversations",
        json={"knowledge_base_id": kb_id, "title": "咨询"},
        headers=auth_headers,
    )
    conv_id = resp.json()["id"]

    resp = await client.post(
        f"/api/v1/conversations/{conv_id}/ask",
        json={"question": "每周工作几天？"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["sources"], "应返回引用来源"

    # 第二次相同问题应命中缓存
    resp2 = await client.post(
        f"/api/v1/conversations/{conv_id}/ask",
        json={"question": "每周工作几天？"},
        headers=auth_headers,
    )
    assert resp2.json()["cached"] is True

    # 对话历史应包含 user/assistant 消息
    resp = await client.get(f"/api/v1/conversations/{conv_id}/messages", headers=auth_headers)
    assert resp.status_code == 200
    roles = [m["role"] for m in resp.json()]
    assert "user" in roles and "assistant" in roles


async def test_stream_ask(client, auth_headers):
    resp = await client.post(
        "/api/v1/knowledge-bases", json={"name": "流式库"}, headers=auth_headers
    )
    kb_id = resp.json()["id"]
    await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        headers=auth_headers,
        files={"file": ("doc.txt", "弹性打卡时间九点到十点。".encode(), "text/plain")},
    )
    resp = await client.post(
        "/api/v1/conversations",
        json={"knowledge_base_id": kb_id},
        headers=auth_headers,
    )
    conv_id = resp.json()["id"]

    async with client.stream(
        "POST",
        f"/api/v1/conversations/{conv_id}/ask-stream",
        json={"question": "弹性打卡时间是？"},
        headers=auth_headers,
    ) as resp:
        assert resp.status_code == 200
        text = ""
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                text += line
        assert '"type": "sources"' in text
        assert '"type": "done"' in text


async def test_document_chunks_view(client, auth_headers):
    """演示台「查看分块内容」：能读到解析后的分块，且受多租户隔离保护。"""
    resp = await client.post("/api/v1/knowledge-bases", json={"name": "分块库"}, headers=auth_headers)
    kb_id = resp.json()["id"]

    body = "第一段：年假五天起。\n\n第二段：加班可调休。\n\n第三段：报销七个工作日到账。"
    resp = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        headers=auth_headers,
        files={"file": ("chunk-demo.txt", body.encode(), "text/plain")},
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    resp = await client.get(f"/api/v1/documents/{doc_id}/chunks", headers=auth_headers)
    assert resp.status_code == 200
    chunks = resp.json()
    assert chunks, "应能读到分块内容"
    assert [c["seq"] for c in chunks] == sorted(c["seq"] for c in chunks)
    assert "年假" in "".join(c["content"] for c in chunks)

    # 未登录 401
    assert (await client.get(f"/api/v1/documents/{doc_id}/chunks")).status_code == 401

    # 他人文档 404（不泄露存在性）
    await client.post(
        "/api/v1/auth/register",
        json={"email": "c@test.com", "password": "password123", "name": "Carol"},
    )
    resp = await client.post(
        "/api/v1/auth/login", data={"username": "c@test.com", "password": "password123"}
    )
    carol_headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    assert (
        await client.get(f"/api/v1/documents/{doc_id}/chunks", headers=carol_headers)
    ).status_code == 404
