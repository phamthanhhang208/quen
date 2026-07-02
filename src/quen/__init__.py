"""Quên — trust-calibrated forgetting: an agent memory that knows what to
forget, says how sure it is, and verifies before it asserts.

All LLM and embedding calls run on Qwen via Alibaba Cloud DashScope
(see ``quen.alibaba_client``).
"""

__version__ = "0.1.0"

__all__ = ["QuenConfig", "QuenEngine", "MemoryItem", "__version__"]


def __getattr__(name):  # lazy re-exports; avoids import cycles at package load
    if name == "QuenConfig":
        from quen.config import QuenConfig

        return QuenConfig
    if name == "QuenEngine":
        from quen.engine import QuenEngine

        return QuenEngine
    if name == "MemoryItem":
        from quen.models import MemoryItem

        return MemoryItem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
