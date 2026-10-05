"""认证业务逻辑：注册 / 登录 / 令牌。"""
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User
from app.schemas.auth import Token, UserCreate, UserOut


async def register(db: AsyncSession, data: UserCreate) -> UserOut:
    exists = await db.scalar(select(User).where(User.email == data.email))
    if exists:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="邮箱已被注册")
    user = User(email=data.email, hashed_password=hash_password(data.password), name=data.name)
    db.add(user)
    await db.flush()
    return UserOut.model_validate(user)


async def authenticate(db: AsyncSession, email: str, password: str) -> User:
    user = await db.scalar(select(User).where(User.email == email))
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已禁用")
    return user


async def issue_token(user: User) -> Token:
    access_token = create_access_token(subject=str(user.id))
    return Token(access_token=access_token, user=UserOut.model_validate(user))
