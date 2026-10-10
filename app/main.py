"""FastAPI 应用入口。"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import auth, conversations, documents, health, knowledge_bases
from app.config import settings
from app.core.redis import close_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：可在此初始化连接池 / 预热
    yield
    # 关闭：释放资源
    await close_redis()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册
app.include_router(health.router, prefix=settings.api_v1_prefix)
app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(knowledge_bases.router, prefix=settings.api_v1_prefix)
app.include_router(documents.router, prefix=settings.api_v1_prefix)
app.include_router(conversations.router, prefix=settings.api_v1_prefix)

# 演示前端：与 API 同源托管（/ 与 /demo 都指向同一份静态资源），前端用相对路径 /api/v1，无需 CORS
# 挂在所有 API 路由之后，因此 /docs、/api/v1/* 仍优先匹配，不会被静态资源抢走
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/demo", StaticFiles(directory=STATIC_DIR, html=True), name="demo")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")
