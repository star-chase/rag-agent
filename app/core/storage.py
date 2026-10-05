"""对象存储抽象：local（本机调试）与 s3（MinIO/S3，生产）双后端。

通过 settings.storage_backend 切换：docker-compose 中为 s3，本机默认 local。
"""
import aioboto3
import aiofiles
from loguru import logger

from app.config import settings

ALLOWED_EXTENSIONS = {"pdf": "application/pdf", "md": "text/markdown", "txt": "text/plain"}


def guess_file_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型 .{ext}，仅支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    return ext


class StorageBackend:
    """统一上传/下载接口。"""

    async def upload(self, key: str, data: bytes, content_type: str | None = None) -> None: ...
    async def download(self, key: str) -> bytes: ...
    async def delete(self, key: str) -> None: ...


class LocalStorage(StorageBackend):
    """把文件存到本机 uploads/ 目录（无 S3 时的调试后端）。"""

    def _path(self, key: str) -> str:
        import os

        root = os.path.join(os.getcwd(), "uploads")
        safe = key.lstrip("/").replace("..", "_")
        full = os.path.abspath(os.path.join(root, safe))
        root_abs = os.path.abspath(root)
        if not full.startswith(root_abs + os.sep):
            raise ValueError("非法存储键")
        return full

    async def upload(self, key: str, data: bytes, content_type: str | None = None) -> None:
        import os

        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)

    async def download(self, key: str) -> bytes:
        async with aiofiles.open(self._path(key), "rb") as f:
            return await f.read()

    async def delete(self, key: str) -> None:
        import os

        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)


class S3Storage(StorageBackend):
    """MinIO / AWS S3 后端（异步）。"""

    def __init__(self) -> None:
        self.session = aioboto3.Session()

    async def upload(self, key: str, data: bytes, content_type: str | None = None) -> None:
        async with self.session.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name="us-east-1",
        ) as s3:
            await s3.put_object(Bucket=settings.s3_bucket, Key=key, Body=data, ContentType=content_type or "application/octet-stream")

    async def download(self, key: str) -> bytes:
        async with self.session.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name="us-east-1",
        ) as s3:
            resp = await s3.get_object(Bucket=settings.s3_bucket, Key=key)
            return await resp["Body"].read()

    async def delete(self, key: str) -> None:
        async with self.session.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name="us-east-1",
        ) as s3:
            await s3.delete_object(Bucket=settings.s3_bucket, Key=key)


_storage: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _storage
    if _storage is None:
        if settings.storage_backend == "s3":
            _storage = S3Storage()
            logger.info("storage backend: s3 ({})", settings.s3_endpoint)
        else:
            _storage = LocalStorage()
            logger.info("storage backend: local (uploads/)")
    return _storage
