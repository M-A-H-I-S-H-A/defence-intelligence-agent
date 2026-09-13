"""
Embedding generation.

Uses sentence-transformers (all-MiniLM-L6-v2 by default: 384-dim, small,
fast on CPU — fine for a class/hackathon-scale project and for the demo
laptop your teammate will run the Streamlit dashboard on). Swap the model
name in config.py if you later want higher quality (e.g. bge-small-en,
or a Gemini embedding endpoint, since the rest of the project already
uses Gemini for the agents).
"""

from __future__ import annotations
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

_model = None


def get_model(model_name: str):
    global _model
    if _model is None:
        print(f"[CHECKPOINT] Starting to load embedding model: {model_name}", flush=True)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
        print(f"[CHECKPOINT] Finished loading embedding model", flush=True)
    return _model


def embed_texts(texts: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,  # so cosine similarity == dot product
    )
    return np.asarray(embeddings)
