"""alibaba_client.py — THE PROOF ARTIFACT.

Every model call and every embedding call in Quên goes through this module,
to Qwen models served by Alibaba Cloud Model Studio (DashScope), via the
OpenAI-compatible endpoint. No other provider is ever used.

Region note: keys are region-scoped and NOT interchangeable. This uses the
international (Singapore) endpoint. Mainland-China keys need
``https://dashscope.aliyuncs.com/compatible-mode/v1`` instead — override via
the ``DASHSCOPE_BASE_URL`` environment variable.
"""

from __future__ import annotations

import os
from functools import lru_cache

from openai import OpenAI

DASHSCOPE_INTL_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

DEFAULT_CHAT_MODEL = os.environ.get("QUEN_CHAT_MODEL", "qwen3.5-plus")
DEFAULT_FAST_MODEL = os.environ.get("QUEN_FAST_MODEL", "qwen-flash")
DEFAULT_EMBED_MODEL = os.environ.get("QUEN_EMBED_MODEL", "text-embedding-v4")


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
    )


def chat(messages: list[dict], model: str = DEFAULT_CHAT_MODEL, **kw):
    """Chat completion on Qwen via DashScope."""
    return _client().chat.completions.create(model=model, messages=messages, **kw)


def embed(texts: list[str], model: str = DEFAULT_EMBED_MODEL):
    """Text embeddings on Qwen via DashScope."""
    return _client().embeddings.create(model=model, input=texts)


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
