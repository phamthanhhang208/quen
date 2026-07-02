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

NEGATION_WINDOW = 8

NEGATION_CUES = frozenset(
    """
    not no never n't dont don't stop stopped drop dropped longer instead
    deprecated removed deleted replaced retired legacy old former previously
    was were migrated away superseded obsolete
    """.split()
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.casefold())


def word_present(fact: str, text: str) -> bool:
    """Word-boundary, case-insensitive containment of the fact string."""
    return re.search(rf"(?<!\w){re.escape(fact.casefold())}(?!\w)", text.casefold()) is not None


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
            window = toks[max(0, i - NEGATION_WINDOW) : i]
            if not any(t in NEGATION_CUES for t in window):
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
    if expects_abstain:
        # the honest move is to say "I don't know" — any confident fabricated
        # fact fails; hedged non-answers pass.
        ok = abstained or _looks_like_abstention(answer)
        return FamaScore(presence=ok, absence=True, fama=ok, abstain_ok=ok)
    presence = all(word_present(f, answer) for f in valid_facts)
    absence = not any(mentioned_positively(f, answer) for f in invalidated_facts)
    return FamaScore(presence=presence, absence=absence, fama=presence and absence)


_ABSTAIN_MARKERS = (
    "don't know", "do not know", "don't have", "no reliable memory",
    "not sure", "cannot answer", "can't answer", "unknown", "no memory",
)


def _looks_like_abstention(answer: str) -> bool:
    low = answer.casefold()
    return any(m in low for m in _ABSTAIN_MARKERS)
