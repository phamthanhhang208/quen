"""Engine configuration. Every threshold the spec names lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class QuenConfig:
    # --- storage ---
    db_path: str = "data/quen.db"

    # --- write pipeline (§4.3) ---
    salience_threshold: float = 0.3      # skip what a strong base model knows
    dedup_cosine_threshold: float = 0.92  # cosine dedup, after triple-key dedup
    # confidence priors by source authority (PR/commit > casual mention)
    authority_confidence: dict[str, float] = field(
        default_factory=lambda: {
            "commit": 0.9,
            "pr": 0.9,
            "doc": 0.85,
            "user": 0.8,
            "chat": 0.6,
            "default": 0.7,
        }
    )

    # --- retrieval (§4.4): relevance-dominant ---
    w_relevance: float = 0.65
    w_retrievability: float = 0.20
    w_importance: float = 0.15
    default_token_budget: int = 1500
    candidate_pool: int = 50  # score at most this many nearest actives

    # --- retention / eviction (§4.5) ---
    eviction_r_threshold: float = 0.30   # theta
    eviction_ttl_days: float = 14.0      # unaccessed past TTL
    desired_retention: float = 0.9

    # --- dream pass (§4.6) ---
    reabstract_min_episodics: int = 3
    selftest_sample_size: int = 5
    nli_confidence_gate: float = 0.7
    nli_knn_k: int = 8
    compress_min_chars: int = 600

    # --- trust gate (§4.7) ---
    # Below this top-relevance the engine abstains instead of answering from
    # memories that are merely fresh but irrelevant. Tuned for the hashing
    # embedder (stopword-only overlap lands ~0.15-0.2); real embedding models
    # sit on a different cosine scale — retune when switching embedders.
    abstain_relevance_floor: float = 0.25
    trust_threshold: float = 0.55
    freshness_half_life_days: float = 30.0
    verify_max_per_ask: int = 3
    confidence_bump_on_confirm: float = 0.15
    confidence_cut_on_refute: float = 0.3  # multiplier

    # --- models (all Qwen on Alibaba DashScope) ---
    chat_model: str = "qwen3.5-plus"
    fast_model: str = "qwen-flash"
    embed_model: str = "text-embedding-v4"

    @staticmethod
    def from_env() -> "QuenConfig":
        cfg = QuenConfig()
        cfg.db_path = os.environ.get("QUEN_DB_PATH", cfg.db_path)
        cfg.chat_model = os.environ.get("QUEN_CHAT_MODEL", cfg.chat_model)
        cfg.fast_model = os.environ.get("QUEN_FAST_MODEL", cfg.fast_model)
        cfg.embed_model = os.environ.get("QUEN_EMBED_MODEL", cfg.embed_model)
        floor = os.environ.get("QUEN_ABSTAIN_FLOOR")
        if floor:
            cfg.abstain_relevance_floor = float(floor)
        return cfg
