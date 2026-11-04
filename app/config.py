"""应用配置：基于 pydantic-settings，从环境变量 / .env 读取。"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 应用
    app_name: str = "rag-agent"
    env: str = "dev"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"

    # CORS（前端地址，逗号分隔）
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # 数据库 / Redis
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag_agent"
    redis_url: str = "redis://localhost:6379/0"

    # JWT
    jwt_secret: str = "please-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 1 天

    # LLM（OpenAI 兼容；mode=fake 时返回模板回答，便于无 key 调试）
    llm_mode: str = "api"  # api | fake
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = "sk-xxx"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2

    # Embedding。base 以 /embeddings/multimodal 结尾时走豆包多模态，dimensions=embedding_dim。
    # mode=fake 时用确定性伪向量，不请求接口。
    embedding_mode: str = "api"  # api | fake
    embedding_base_url: str = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
    embedding_api_key: str = "sk-xxx"
    embedding_model: str = "doubao-embedding-vision-251215"
    embedding_dim: int = 1024

    # 对象存储（local=本机目录调试；s3=MinIO/S3 生产）
    storage_backend: str = "local"
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "rag"
    s3_secret_key: str = "ragsecret"
    s3_bucket: str = "rag-documents"
    s3_secure: bool = False

    # Celery（always_eager=true 时同步执行，便于无 Redis 本地调试）
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_always_eager: bool = False

    # RAG 检索（rerank_mode: rrf=融合重排；llm=LLM 一次打分重排）
    top_k: int = 5
    rerank_top_k: int = 3
    rerank_mode: str = "rrf"
    # 真实 embedding 下，余弦相似度低于该值的候选在调用模型前丢弃。0 表示不启用。
    # 假向量没有语义，检索代码会忽略这个值。0.35 只是起点，要用评测集重标定。
    retrieval_min_similarity: float = 0.35
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_batch_size: int = 16

    # 限流
    rate_limit_per_minute: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
