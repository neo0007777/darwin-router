"""Embeddings client supporting both local and remote providers.
- Local mode: sentence-transformers (all-MiniLM-L6-v2, 384-dim, fast on CPU/MPS)
- Original mode: SIE /v1/embeddings; falls back to DashScope if SIE_API_KEY is unset
"""

import httpx
import numpy as np

from router import config

BATCH = 10  # DashScope caps embedding input at 10 items; harmless for SIE


class LocalEmbedClient:
    """Local embedding client using sentence-transformers (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str | None = None):
        self.backend = "local"
        self.model_name = model_name or config.LOCAL_EMBED_MODEL
        from sentence_transformers import SentenceTransformer

        # Load sentence transformer model (cached locally)
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Normalizing embeddings ensures cosine similarity is simple dot product
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()


class RemoteEmbedClient:
    """Original hackathon remote client: SIE or DashScope."""

    def __init__(self):
        if config.SIE_API_KEY:
            self.backend = "sie"
            self.base_url = config.SIE_BASE_URL
            self.api_key = config.SIE_API_KEY
            self.model = config.SIE_EMBED_MODEL
        else:
            self.backend = "dashscope"
            self.base_url = config.DASHSCOPE_BASE_URL
            self.api_key = config.DASHSCOPE_API_KEY
            self.model = config.DASHSCOPE_EMBED_MODEL
        self.http = httpx.Client(timeout=30)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            out.extend(self._call(texts[i : i + BATCH]))
        return out

    def _call(self, batch: list[str], _retried: bool = False) -> list[list[float]]:
        try:
            r = self.http.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": batch},
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        except Exception:
            if _retried:
                raise
            return self._call(batch, _retried=True)


# Backwards compatibility alias
EmbedClient = RemoteEmbedClient

_client: LocalEmbedClient | RemoteEmbedClient | None = None


def get_client() -> LocalEmbedClient | RemoteEmbedClient:
    global _client
    if _client is None:
        if config.PROVIDER == "local":
            _client = LocalEmbedClient()
        else:
            _client = RemoteEmbedClient()
    return _client
