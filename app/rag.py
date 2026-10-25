import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.embeddings import embed_batch, embed_text
from app.models import KnowledgeChunk, Product


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Naive fixed-size character chunker with overlap. Good enough for
    FAQs/policies; swap for a sentence/markdown-aware splitter if your
    documents get long and structured."""
    size = size or settings.CHUNK_SIZE_CHARS
    overlap = overlap or settings.CHUNK_OVERLAP_CHARS
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap  # step back so context isn't hard-cut
    return [c for c in chunks if c]


async def ingest_document(session: AsyncSession, store_id: int, title: str, content: str) -> int:
    """Chunk + embed a free-text document (policy, FAQ, manual, etc.) and
    store it. Returns the number of chunks created."""
    pieces = chunk_text(content)
    if not pieces:
        return 0
    vectors = embed_batch(pieces)
    for piece, vector in zip(pieces, vectors, strict=True):
        session.add(
            KnowledgeChunk(
                store_id=store_id,
                source_type="document",
                source_id=None,
                title=title,
                content=piece,
                embedding=vector,
            )
        )
    await session.commit()
    return len(pieces)


async def sync_products_to_knowledge_base(session: AsyncSession, store_id: int) -> int:
    """(Re)embed every product's name/description/category as a knowledge
    chunk, so the RAG layer can answer "do you sell X" / "what's the return
    policy on Y" style questions grounded in the actual catalog. Safe to
    call repeatedly -- it replaces the store's existing product-sourced
    chunks each time rather than duplicating them."""
    await session.execute(
        delete(KnowledgeChunk).where(
            KnowledgeChunk.store_id == store_id, KnowledgeChunk.source_type == "product"
        )
    )

    result = await session.execute(select(Product).where(Product.store_id == store_id))
    products = result.scalars().all()
    if not products:
        await session.commit()
        return 0

    texts = [
        f"{p.name} ({p.category}). {p.description} Price: ${p.price}. SKU: {p.sku}."
        for p in products
    ]
    vectors = embed_batch(texts)
    for product, text, vector in zip(products, texts, vectors, strict=True):
        session.add(
            KnowledgeChunk(
                store_id=store_id,
                source_type="product",
                source_id=str(product.id),
                title=product.name,
                content=text,
                embedding=vector,
            )
        )
    await session.commit()
    return len(products)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


async def search_knowledge_base(
    session: AsyncSession, store_id: int, query: str, top_k: int | None = None
) -> list[tuple[KnowledgeChunk, float]]:
    """Brute-force cosine similarity over this store's chunks. Fine up to a
    few thousand chunks; if a store's knowledge base gets much bigger than
    that, move the embedding column to pgvector and let Postgres do the
    nearest-neighbor search instead of pulling everything into Python."""
    top_k = top_k or settings.RAG_TOP_K
    result = await session.execute(select(KnowledgeChunk).where(KnowledgeChunk.store_id == store_id))
    chunks = result.scalars().all()
    if not chunks:
        return []

    query_vector = embed_text(query)
    scored = [(chunk, _cosine_similarity(query_vector, chunk.embedding)) for chunk in chunks]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
