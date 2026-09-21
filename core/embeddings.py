"""
Embeddings wrapper for MedSafe.
Supports:
1. Google Gemini Embeddings (via google-genai SDK)
2. Local deterministic dense embeddings (for offline mode, testing, and fallback)
"""

import os
import hashlib
import logging
from typing import List
import numpy as np
from langchain_core.embeddings import Embeddings
from google import genai

from config import Config

logger = logging.getLogger("medsafe.embeddings")


class LocalEmbeddings(Embeddings):
    """
    Offline/local embeddings provider using deterministic hashing.
    Generates normalized 768-dimensional dense vectors with zero API calls.
    Used for offline testing, local development, and automatic fallback.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def _embed(self, text: str) -> List[float]:
        v = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return v.tolist()

        for w in words:
            # Hash word into dimension index and pseudo-random sign
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h // self.dimension) % 2 == 0 else -1.0
            v[idx] += sign

        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class GeminiEmbeddings(Embeddings):
    """
    LangChain-compatible Embeddings provider using the Google GenAI SDK.
    Includes automated fallback to local embeddings if API key is invalid or offline.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model = model or Config.EMBEDDING_MODEL
        self._local_fallback = LocalEmbeddings()
        self._fallback_active = False

        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Using local offline embeddings.")
            self._client = None
            if Config.AUTO_FALLBACK_EMBEDDINGS:
                self._fallback_active = True
        else:
            try:
                self._client = genai.Client(api_key=self.api_key.strip("\"' "))
            except Exception as e:
                logger.error("Failed to initialize genai.Client: %s", e)
                self._client = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for document chunks."""
        if self._fallback_active or not self._client or Config.EMBEDDING_PROVIDER == "local":
            return self._local_fallback.embed_documents(texts)

        vectors: List[List[float]] = []
        for idx, text in enumerate(texts):
            content = text.strip() if text and text.strip() else "empty document"
            try:
                response = self._client.models.embed_content(
                    model=self.model,
                    contents=content,
                )
                vectors.append(response.embeddings[0].values)
            except Exception as e:
                err_str = str(e)
                logger.warning("Gemini embedding call failed on chunk %d: %s", idx, err_str)

                if Config.AUTO_FALLBACK_EMBEDDINGS:
                    logger.warning(
                        "Activating local offline embeddings fallback. "
                        "(To use Gemini embeddings, provide a valid GEMINI_API_KEY from Google AI Studio)."
                    )
                    self._fallback_active = True
                    return self._local_fallback.embed_documents(texts)

                raise RuntimeError(
                    f"Gemini embedding failed: {e}. Check GEMINI_API_KEY in .env."
                ) from e

        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding vector for a single query text."""
        if self._fallback_active or not self._client or Config.EMBEDDING_PROVIDER == "local":
            return self._local_fallback.embed_query(text)

        content = text.strip() if text and text.strip() else "empty query"
        try:
            response = self._client.models.embed_content(
                model=self.model,
                contents=content,
            )
            return response.embeddings[0].values
        except Exception as e:
            if Config.AUTO_FALLBACK_EMBEDDINGS:
                self._fallback_active = True
                return self._local_fallback.embed_query(text)
            raise RuntimeError(
                f"Gemini query embedding failed: {e}. Check GEMINI_API_KEY in .env."
            ) from e


def get_embeddings_model() -> Embeddings:
    """Factory to retrieve configured embeddings model."""
    if Config.EMBEDDING_PROVIDER == "local":
        return LocalEmbeddings()
    return GeminiEmbeddings()
