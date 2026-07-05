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
    # Soft gate (default): below-threshold facts are stored WEAK (minimal
    # importance + lapse-level stability) instead of skipped — a wrongly
    # skipped fact is unrecoverable, while a wrongly kept one just decays.
    # Set False for the spec §4.3 hard gate.
    salience_soft_gate: bool = True
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
    # exact identifier-overlap bonus — dense embeddings smooth over rare
    # code tokens (hybrid-retrieval gap); weighted into the score
    w_lexical: float = 0.15
    default_token_budget: int = 1500
    candidate_pool: int = 50  # score at most this many nearest actives
    # each included memory costs its content PLUS the trust tag/hedge the
    # reader actually sees — the budget must account for delivered tokens
    per_memory_overhead_tokens: int = 12
    # short tag grammar [t=0.82 3d ✓] + terse hedges instead of the long
    # forms — memory SELECTION is identical (overhead constant above is
    # unchanged); only delivered prompt_tokens shrink. Flag OFF restores
    # the long forms byte-for-byte.
    compact_trust_tags: bool = True
    # Memories below this (relevance + lexical) never enter the context:
    # budget-padding with irrelevant memories pollutes the reader AND
    # touch-resets their eviction TTL on every ask (eviction starvation).
    include_relevance_floor: float = 0.10

    # --- retention / eviction (§4.5) ---
    # Under the FSRS-4.5 power curve, R < theta needs t > 43*S days at
    # theta=0.3 (160d at initial GOOD stability) — forgetting was
    # mathematically out of reach. theta=0.5 puts the horizon at ~12.8*S
    # (~47d for a never-reinforced fact), which matches the TTL scale.
    eviction_r_threshold: float = 0.50   # theta
    eviction_ttl_days: float = 14.0      # unaccessed past TTL
    desired_retention: float = 0.9

    # --- dream pass (§4.6) ---
    reabstract_min_episodics: int = 3
    selftest_sample_size: int = 5
    # A self-test pass is retrieval health, NOT human recall: the probe is
    # built from the memory and the memory stays in the pool, so passes are
    # near-certain and independent of R. Left as GOOD reviews they hit the
    # FSRS spacing term exactly where it explodes (x27.8 S at R=0.3) and
    # immortalize whatever gets tested. Passes grade HARD with this hard cap
    # on stability growth per self-test.
    selftest_pass_growth_cap: float = 2.0
    # After a failed self-test, don't re-test the same memory for this many
    # days (failing memories otherwise monopolize the whole sample forever).
    selftest_fail_backoff_days: float = 7.0
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
    # Stability-tempered freshness (anti recency-bias): a fact whose FSRS
    # stability has been earned through many successful reviews keeps its
    # trust longer — old-but-stable truths ("we use Postgres") should not be
    # hedged like day-old gossip. eff_half_life = base * (1 + gain*ln(1+S/S0)).
    trust_stability_tempering: bool = True
    trust_stability_gain: float = 1.0
    trust_stability_ref: float = 3.7145  # initial GOOD stability (FSRS w2)
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
