"""Shared fixtures. Everything is offline and deterministic:
HashingEmbedder + ScriptedLLM + an injectable MutableClock.

Wave rule: parallel work must NOT edit this file — define extra fixtures
locally in your own test module.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quen.config import QuenConfig
from quen.embeddings import HashingEmbedder
from quen.llm import ScriptedLLM
from quen.models import MemoryItem, make_memory
from quen.store import MemoryStore


class MutableClock:
    """Injectable clock. Every time-dependent test advances it explicitly —
    FSRS reviews at elapsed_days=0 are a no-op by construction (R=1)."""

    def __init__(self, start: datetime | None = None):
        self.current = start or datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.current

    def advance(self, *, days: float = 0.0, hours: float = 0.0) -> datetime:
        self.current += timedelta(days=days, hours=hours)
        return self.current

    __call__ = now


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def cfg() -> QuenConfig:
    return QuenConfig()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def scripted() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def store(tmp_path, clock) -> MemoryStore:
    s = MemoryStore(str(tmp_path / "test.db"), clock=clock.now)
    yield s
    s.close()


@pytest.fixture
def mem_factory(embedder, clock):
    """Create a MemoryItem with a real embedding, at the clock's current time."""

    def _make(content: str, **kw) -> MemoryItem:
        kw.setdefault("now", clock.now())
        return make_memory(content, embedding=embedder.embed([content])[0], **kw)

    return _make
