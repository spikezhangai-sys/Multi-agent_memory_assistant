from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Any, Protocol

import certifi
import numpy as np
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class MockEmbedder:
    """Deterministic hash-based embedder for tests.

    Produces unit-norm vectors where each token contributes a fixed-direction
    bump. Semantically similar texts that share tokens land near each other;
    disjoint texts are ~orthogonal. Good enough to exercise the retrieval
    math without hitting any external API.
    """

    def __init__(self, *, dim: int = 64, model: str = "mock") -> None:
        self.dim = dim
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _simple_tokenize(text) or [text.lower()]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, idx] += sign
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


class OpenAIEmbedder:
    """OpenAI embeddings API client (text-embedding-3-small by default)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        base_url: str = "https://api.openai.com/v1",
        timeout_sec: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max(1, max_retries)

    @classmethod
    def from_env(cls, *, model: str | None = None) -> "OpenAIEmbedder":
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        resolved_model = model or os.getenv("DRIFTSCOPE_EMBEDDING_MODEL", "text-embedding-3-small")
        dim_env = os.getenv("DRIFTSCOPE_EMBEDDING_DIM")
        dim = int(dim_env) if dim_env else 1536
        return cls(
            api_key=api_key,
            model=resolved_model,
            dim=dim,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = {"model": self.model, "input": texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/embeddings"
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.timeout_sec,
            verify=certifi.where(),
        )
        response.raise_for_status()
        body = response.json()
        vectors = np.array(
            [item["embedding"] for item in body["data"]],
            dtype=np.float32,
        )
        if vectors.shape[1] != self.dim:
            self.dim = vectors.shape[1]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


class LocalSentenceTransformerEmbedder:
    """Local sentence-transformers embedder (e.g. BAAI/bge-small-en-v1.5).

    Model instances are cached per (model_name, device) so that multiple
    workers reuse a single loaded checkpoint. PyTorch inference releases
    the GIL during encode(), so concurrent threads actually progress.
    """

    _model_cache: dict[tuple[str, str | None], Any] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dim: int = 384,
        device: str | None = None,
    ) -> None:
        self.model = model_name
        self.dim = dim
        self.device = device
        key = (model_name, device)
        with LocalSentenceTransformerEmbedder._cache_lock:
            st = LocalSentenceTransformerEmbedder._model_cache.get(key)
            if st is None:
                from sentence_transformers import SentenceTransformer

                st = SentenceTransformer(model_name, device=device)
                LocalSentenceTransformerEmbedder._model_cache[key] = st
        self._st = st
        getter = getattr(self._st, "get_embedding_dimension", None) or self._st.get_sentence_embedding_dimension
        inferred_dim = int(getter() or dim)
        if inferred_dim != self.dim:
            self.dim = inferred_dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._st.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(
    *,
    backend: str | None = None,
    model: str | None = None,
) -> Embedder | None:
    load_dotenv()
    resolved = (backend or os.getenv("DRIFTSCOPE_EMBEDDING_BACKEND", "mock")).strip().lower()
    if resolved in {"none", "off", "disabled"}:
        return None
    if resolved == "openai":
        try:
            return OpenAIEmbedder.from_env(model=model)
        except Exception as exc:
            logger.warning("build_embedder: OpenAI backend unavailable (%s); falling back to mock", exc)
            return MockEmbedder()
    if resolved in {"local", "sentence-transformers", "st"}:
        try:
            model_name = model or os.getenv(
                "DRIFTSCOPE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
            )
            dim_env = os.getenv("DRIFTSCOPE_EMBEDDING_DIM")
            dim = int(dim_env) if dim_env else 384
            device = os.getenv("DRIFTSCOPE_EMBEDDING_DEVICE") or None
            return LocalSentenceTransformerEmbedder(
                model_name=model_name, dim=dim, device=device
            )
        except Exception as exc:
            logger.warning(
                "build_embedder: local backend unavailable (%s); falling back to mock", exc
            )
            return MockEmbedder()
    return MockEmbedder()


def _simple_tokenize(text: str) -> list[str]:
    buf: list[str] = []
    for token in (text or "").lower().split():
        cleaned = "".join(ch for ch in token if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        if cleaned:
            buf.append(cleaned)
    return buf
