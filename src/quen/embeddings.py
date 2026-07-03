"""Embedding backends.

``QwenEmbedder`` is the production path (text-embedding-v4 on DashScope).
``HashingEmbedder`` is a deterministic, offline, dependency-free stand-in
used by tests and QUEN_OFFLINE mode — cosine on it behaves sensibly for
token-overlapping texts, which is all the tests rely on.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class QwenEmbedder:
    """text-embedding-v4 via quen.alibaba_client (the only network path)."""

    # DashScope hard limits: at most 10 texts per embeddings request
    # ("batch size ... should not be larger than 10"), 8192 tokens per text.
    MAX_BATCH = 10
    MAX_CHARS = 28_000  # ~7k tokens — defensive truncation under the cap

    def __init__(self, model: str | None = None, dim: int = 1024):
        from quen import alibaba_client

        self._embed = alibaba_client.embed
        self.model = model or alibaba_client.DEFAULT_EMBED_MODEL
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH):
            chunk = [t[: self.MAX_CHARS] for t in texts[start : start + self.MAX_BATCH]]
            # dimensions pinned explicitly — text-embedding-v4 supports 64-2048
            resp = self._embed(chunk, model=self.model, dimensions=self.dim)
            data = sorted(resp.data, key=lambda d: d.index)  # preserve order
            out.extend(d.embedding for d in data)
        return out


class HashingEmbedder:
    """Deterministic bag-of-tokens hashing embedder (offline/tests only)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9_]+", text.casefold())
        for tok in tokens:
            h = hashlib.md5(tok.encode()).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec
