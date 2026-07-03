"""FAMA scorer — adopted from Memora (arXiv 2604.20006).

FAMA = presence-of-valid ∧ absence-of-invalidated: an answer only scores if
it contains every still-valid fact AND relies on no invalidated one. The two
components are always reported separately so the headline can't hide a
presence/absence trade-off.

Absence uses a negation window: "we no longer use useApi" is not *reliance*
on useApi — a mention preceded (≤ NEGATION_WINDOW tokens) by a negation or
supersession cue does not count against the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NEGATION_WINDOW = 8       # tokens looked at BEFORE a mention
LOOKAHEAD_WINDOW = 6      # tokens looked at AFTER a mention (passives:
                          # "the Redis cluster is decommissioned")

# NOTE: "old" is deliberately absent — answers carry trust tags like
# "[... 45d old]" and a weak cue there would blind the absence check.
NEGATION_CUES = frozenset(
    """
    not no never dont don nobody stop stopped drop dropped longer
    instead deprecated removed deleted replaced retired legacy former
    previously was were migrated away superseded obsolete gone ripped
    decommissioned
    """.split()
)


_RUN_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")


def _tokens(text: str) -> list[str]:
    # camel-aware alpha/digit runs so 'CloudflareR2', 'Cloudflare R2' and
    # 'Node16'/'Node 16' all tokenize identically
    return [t.casefold() for t in _RUN_RE.findall(text)]


def _fact_pattern(fact: str) -> str:
    """Word-boundary pattern tolerant of spacing/hyphens between the fact's
    camel-aware alphanumeric runs: 'Node20' also matches 'Node 20', and
    'CloudflareR2' matches 'Cloudflare R2' (real readers re-space compound
    tokens), while 'S3' still rejects 'S3000'."""
    runs = _tokens(fact)
    if not runs:
        return re.escape(fact.casefold())
    return r"(?<!\w)" + r"[\s\-_./]*".join(re.escape(r) for r in runs) + r"(?!\w)"


def word_present(fact: str, text: str) -> bool:
    """Word-boundary, case-insensitive containment of the fact string."""
    return re.search(_fact_pattern(fact), text.casefold()) is not None


def mentioned_positively(fact: str, text: str) -> bool:
    """True if the fact appears at least once WITHOUT a preceding negation
    cue inside the window — i.e. the answer relies on it."""
    fact_toks = _tokens(fact)
    if not fact_toks:
        return False
    toks = _tokens(text)
    n = len(fact_toks)
    for i in range(len(toks) - n + 1):
        if toks[i : i + n] == fact_toks:
            before = toks[max(0, i - NEGATION_WINDOW) : i]
            after = toks[i + n : i + n + LOOKAHEAD_WINDOW]
            if not any(t in NEGATION_CUES for t in before + after):
                return True
    return False


@dataclass
class FamaScore:
    presence: bool   # every valid fact present
    absence: bool    # no invalidated fact relied upon
    fama: bool       # presence ∧ absence
    abstain_ok: bool | None = None  # only for expects_abstain cases


def score_answer(
    answer: str,
    *,
    valid_facts: list[str],
    invalidated_facts: list[str],
    expects_abstain: bool = False,
    abstained: bool = False,
) -> FamaScore:
    absence = not any(mentioned_positively(f, answer) for f in invalidated_facts)
    if expects_abstain:
        # Judged from the DELIVERED TEXT only — a system's self-reported
        # abstention flag is not credited (that would score our own config
        # by a signal the baselines cannot emit). The honest move is an
        # answer that says "I don't know"; and even an abstention must not
        # rely on an invalidated fact.
        ok = _looks_like_abstention(answer)
        return FamaScore(presence=ok, absence=absence,
                         fama=ok and absence, abstain_ok=ok)
    presence = all(word_present(f, answer) for f in valid_facts)
    return FamaScore(presence=presence, absence=absence, fama=presence and absence)


_ABSTAIN_MARKERS = (
    "don't know", "do not know", "don't have", "no reliable memory",
    "not sure", "cannot answer", "can't answer", "unknown", "no memory",
)


def _looks_like_abstention(answer: str) -> bool:
    low = answer.casefold()
    return any(m in low for m in _ABSTAIN_MARKERS)
