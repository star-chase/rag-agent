"""演示前端（静态托管）冒烟测试：根路径与 /demo 都能打开演示台。"""


async def test_root_serves_demo_page(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert '<div id="app">' in resp.text


async def test_demo_page_served(client):
    resp = await client.get("/demo/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert '<div id="app">' in resp.text


async def test_demo_assets_served(client):
    for path, needle in (("/demo/app.js", "ask-stream"), ("/demo/styles.css", "--accent")):
        resp = await client.get(path)
        assert resp.status_code == 200
        assert needle in resp.text


async def test_root_assets_served(client):
    for path, needle in (("/app.js", "ask-stream"), ("/styles.css", "--accent")):
        resp = await client.get(path)
        assert resp.status_code == 200
        assert needle in resp.text


async def test_demo_sample_questions(client):
    resp = await client.get("/demo/sample-questions.json")
    assert resp.status_code == 200
    questions = resp.json()
    assert isinstance(questions, list) and questions


async def test_api_routes_not_shadowed_by_static(client):
    """静态挂载不能抢走 API 路由。"""
    assert (await client.get("/api/v1/health")).status_code == 200
    assert (await client.get("/docs")).status_code == 200
