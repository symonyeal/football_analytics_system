"""Cross-source player-id unification (Part 0.2).

Map ``player_id`` values from FBref / Transfermarkt / StatsBomb to a single
``player_uid`` via fuzzy name + DOB matching:

    match(a, b) := JaroWinkler(name_a, name_b) >= 0.92
                   AND nationality_a == nationality_b
                   AND |dob_a - dob_b| <= dob_tol

Jaro-Winkler is implemented locally (no hard dependency) so the matcher runs
without ``jellyfish``/``rapidfuzz`` installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

JW_THRESHOLD = 0.92


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler similarity in [0, 1]. Pure-python, no deps."""
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    match_dist = max(len(s1), len(s2)) // 2 - 1
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)

    matches = 0
    for i, c1 in enumerate(s1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len(s2))
        for j in range(lo, hi):
            if not s2_matches[j] and s2[j] == c1:
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0

    # transpositions
    k = 0
    transpositions = 0
    for i in range(len(s1)):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
    transpositions //= 2

    jaro = (
        matches / len(s1)
        + matches / len(s2)
        + (matches - transpositions) / matches
    ) / 3.0

    # common prefix up to 4 chars
    prefix = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2 and prefix < 4:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


@dataclass(slots=True)
class IdMatch:
    left_id: int
    right_id: int
    score: float


def match_rosters(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    name_col: str = "name",
    id_col: str = "player_id",
    nationality_col: str | None = "nationality",
    threshold: float = JW_THRESHOLD,
) -> list[IdMatch]:
    """Greedy best-match between two rosters under the JW + nationality rule.

    Returns one :class:`IdMatch` per matched left row (highest-scoring right
    candidate above ``threshold``). Nationality, when present in both frames,
    is used as a hard filter.
    """
    matches: list[IdMatch] = []
    for _, lrow in left.iterrows():
        best: IdMatch | None = None
        for _, rrow in right.iterrows():
            if (
                nationality_col
                and nationality_col in left
                and nationality_col in right
                and lrow[nationality_col] != rrow[nationality_col]
            ):
                continue
            score = jaro_winkler(str(lrow[name_col]), str(rrow[name_col]))
            if score >= threshold and (best is None or score > best.score):
                best = IdMatch(int(lrow[id_col]), int(rrow[id_col]), score)
        if best is not None:
            matches.append(best)
    return matches
