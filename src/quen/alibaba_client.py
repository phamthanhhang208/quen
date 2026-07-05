"""alibaba_client.py — THE PROOF ARTIFACT.

Every model call and every embedding call in Quên goes through this module,
to Qwen models served by Alibaba Cloud Model Studio (DashScope), via the
OpenAI-compatible endpoint. No other provider is ever used.

Region note: keys are region-scoped and NOT interchangeable. This uses the
international (Singapore) endpoint — the one the Qwen Cloud hackathon
mandates. Mainland-China keys need
``https://dashscope.aliyuncs.com/compatible-mode/v1`` instead — override via
the ``DASHSCOPE_BASE_URL`` environment variable.

A lightweight per-model usage accumulator (``usage_summary()``) powers the
eval cost reports; it never records prompt contents, only token counts.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # keys live in .env (gitignored), never in code

DASHSCOPE_INTL_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

DEFAULT_CHAT_MODEL = os.environ.get("QUEN_CHAT_MODEL", "qwen3.5-plus")
DEFAULT_FAST_MODEL = os.environ.get("QUEN_FAST_MODEL", "qwen-flash")
DEFAULT_EMBED_MODEL = os.environ.get("QUEN_EMBED_MODEL", "text-embedding-v4")

_usage: dict[str, dict[str, int]] = {}
_usage_lock = threading.Lock()


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. Quên talks only to Qwen on Alibaba "
            "Cloud DashScope; set the key (see .env.example) or run with "
            "QUEN_OFFLINE=1 for the deterministic offline mode."
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DASHSCOPE_BASE_URL", DASHSCOPE_INTL_BASE_URL),
        max_retries=5,  # rate-limit resilience; eval runners are serial
    )


def _track(model: str, resp) -> None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    with _usage_lock:
        entry = _usage.setdefault(
            model,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "total_tokens": 0, "cached_tokens": 0},
        )
        entry["calls"] += 1
        entry["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        entry["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        entry["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        # DashScope implicit context cache: prefix hits are billed at the
        # cached rate and surface here — count them so cost reports can
        # separate fresh from cached input
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            entry["cached_tokens"] += getattr(details, "cached_tokens", 0) or 0


def usage_summary() -> dict[str, dict[str, int]]:
    """Per-model token counts accumulated this process (for cost reports)."""
    with _usage_lock:
        return {m: dict(v) for m, v in _usage.items()}


def reset_usage() -> None:
    with _usage_lock:
        _usage.clear()


def chat(messages: list[dict], model: str = DEFAULT_CHAT_MODEL, **kw):
    """Chat completion on Qwen via DashScope."""
    resp = _client().chat.completions.create(model=model, messages=messages, **kw)
    _track(model, resp)
    return resp


def embed(texts: list[str], model: str = DEFAULT_EMBED_MODEL,
          dimensions: int | None = None):
    """Text embeddings on Qwen via DashScope (text-embedding-v4 supports
    64-2048 dims; we pin explicitly for determinism)."""
    kw = {"dimensions": dimensions} if dimensions else {}
    resp = _client().embeddings.create(model=model, input=texts, **kw)
    _track(model, resp)
    return resp


def smoke_test() -> str:
    """One tiny round-trip to prove the DashScope wiring (P0 smoke test)."""
    resp = chat(
        [{"role": "user", "content": "Reply with exactly: quen-ok"}],
        model=DEFAULT_FAST_MODEL,
        max_tokens=8,
    )
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    print(smoke_test())
    emb = embed(["quen smoke"], dimensions=1024)
    print(f"embedding dim: {len(emb.data[0].embedding)}")
    print(f"usage: {usage_summary()}")
