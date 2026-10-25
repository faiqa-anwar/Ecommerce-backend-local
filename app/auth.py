import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, InvalidPasswordException, UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_async_session
from app.models import Store, User


async def get_user_db(session: AsyncSession = Depends(get_async_session)) -> AsyncGenerator:
    yield SQLAlchemyUserDatabase(session, User)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = settings.SECRET_KEY
    verification_token_secret = settings.SECRET_KEY

    async def validate_password(self, password: str, user: User) -> None:
        if len(password) < 8:
            raise InvalidPasswordException(reason="Password must be at least 8 characters long.")
        if user.email and password.lower() == user.email.lower():
            raise InvalidPasswordException(reason="Password cannot be the same as your email.")

    async def on_after_register(self, user: User, request: Request | None = None) -> None:
        """Every new admin account gets its own store automatically, so
        there's no separate 'create a store' step before you can use the
        API. In a real multi-admin-per-store setup you'd instead let a user
        join an existing store by invite -- this is the simple default."""
        session: AsyncSession = self.user_db.session  # type: ignore[attr-defined]
        store = Store(name=f"{user.email}'s store")
        session.add(store)
        await session.flush()
        user.store_id = store.id
        session.add(user)
        await session.commit()


async def get_user_manager(user_db=Depends(get_user_db)) -> AsyncGenerator:
    yield UserManager(user_db)


bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=settings.SECRET_KEY, lifetime_seconds=settings.ACCESS_TOKEN_LIFETIME_SECONDS)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

current_active_user = fastapi_users.current_user(active=True)
