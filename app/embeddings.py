"""
Free, local embeddings via fastembed (ONNX runtime -- no PyTorch, no API key).
Model: all-MiniLM-L6-v2 (384 dimensions). It's downloaded once on first run
(~90MB) and cached, so the first start needs internet; after that it's offline.

If the model can't load (e.g. first run with no internet), we fall back to a
simple deterministic hashing embedding so the app still starts. The fallback is
keyword-ish, NOT semantic -- check the startup log for a warning if search
quality looks poor.
"""
import hashlib
import logging

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_model = None
_USING_FALLBACK = False

try:
    from fastembed import TextEmbedding

    _model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
except Exception as exc:
    logger.warning("fastembed model unavailable (%s); using hashing fallback.", exc)
    _USING_FALLBACK = True


def is_using_fallback_embeddings() -> bool:
    return _USING_FALLBACK


def _hashing_embed(text: str, dim: int = settings.EMBEDDING_DIM) -> list[float]:
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def embed_text(text: str) -> list[float]:
    if not text or not text.strip():
        return [0.0] * settings.EMBEDDING_DIM
    if _model is not None:
        return next(iter(_model.embed([text]))).tolist()
    return _hashing_embed(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    if _model is not None:
        return [v.tolist() for v in _model.embed(texts)]
    return [_hashing_embed(t) for t in texts]
