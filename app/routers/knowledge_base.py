from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import current_active_user
from app.database import get_async_session
from app.models import KnowledgeChunk, User
from app.rag import ingest_document, search_knowledge_base, sync_products_to_knowledge_base
from app.routers.products import _require_store
from app.schemas import DocumentIn, KnowledgeChunkOut, KnowledgeSearchIn, KnowledgeSearchResult

router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def add_document(
    payload: DocumentIn,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    store_id = _require_store(user)
    count = await ingest_document(session, store_id, payload.title, payload.content)
    return {"chunks_created": count}


@router.post("/sync-products")
async def sync_products(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    store_id = _require_store(user)
    count = await sync_products_to_knowledge_base(session, store_id)
    return {"products_indexed": count}


@router.get("", response_model=list[KnowledgeChunkOut])
async def list_chunks(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[KnowledgeChunk]:
    store_id = _require_store(user)
    result = await session.execute(select(KnowledgeChunk).where(KnowledgeChunk.store_id == store_id))
    return list(result.scalars().all())


@router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chunk(
    chunk_id: int,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    store_id = _require_store(user)
    result = await session.execute(
        select(KnowledgeChunk).where(KnowledgeChunk.id == chunk_id, KnowledgeChunk.store_id == store_id)
    )
    chunk = result.scalar_one_or_none()
    if chunk is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Knowledge chunk not found.")
    await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.id == chunk_id))
    await session.commit()


@router.post("/search", response_model=list[KnowledgeSearchResult])
async def search(
    payload: KnowledgeSearchIn,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[KnowledgeSearchResult]:
    """Debug/manual-testing endpoint: run retrieval directly without going
    through the LLM, so you can sanity-check what the chat endpoint would
    actually see as context."""
    store_id = _require_store(user)
    results = await search_knowledge_base(session, store_id, payload.query, payload.top_k)
    return [KnowledgeSearchResult(chunk=chunk, score=score) for chunk, score in results]
