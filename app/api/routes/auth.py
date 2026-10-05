"""认证路由：注册 / 登录。"""
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.auth import Token, UserCreate, UserOut
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201)
async def register(data: UserCreate, db: AsyncSession = Depends(get_db)) -> UserOut:
    return await auth_service.register(db, data)


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)) -> Token:
    """OAuth2 密码模式：body 用 form 字段 username/password（兼容 Swagger 的 Authorize 按钮）。"""
    user = await auth_service.authenticate(db, form.username, form.password)
    return await auth_service.issue_token(user)
