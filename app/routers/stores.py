from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import current_active_user
from app.database import get_async_session
from app.models import Store, User
from app.routers.products import _require_store

router = APIRouter(prefix="/stores", tags=["stores"])


@router.get("/me")
async def get_my_store(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    store_id = _require_store(user)
    result = await session.execute(select(Store).where(Store.id == store_id))
    store = result.scalar_one()
    return {
        "id": store.id,
        "name": store.name,
        "public_api_key": store.public_api_key,
    }
