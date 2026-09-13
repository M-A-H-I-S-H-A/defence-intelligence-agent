"""
Embedding generation.

Uses ChromaDB's built-in ONNX-based MiniLM embedding function
(onnxruntime — no PyTorch) instead of sentence-transformers, to keep
memory usage low enough for constrained hosting (e.g. Render's free
512MB tier). Same underlying MiniLM model family, much lighter runtime.
"""

from __future__ import annotations
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

_embedding_fn = None


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        logger.info("Loading ONNX MiniLM embedding function (no PyTorch)")
        _embedding_fn = DefaultEmbeddingFunction()
    return _embedding_fn


def embed_texts(texts: List[str], model_name: str = None, batch_size: int = 64) -> np.ndarray:
    """model_name is kept for compatibility with existing call sites but
    is ignored — this always uses ChromaDB's lightweight ONNX model."""
    fn = _get_embedding_fn()
    embeddings = fn(texts)
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return arr / norms