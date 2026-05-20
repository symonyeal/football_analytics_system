"""Functorial coherence statement for the analytics system (v3 Part E.6).

This module is deliberately small. It records the composability contract:
canonical match data is mapped to performance measures, and existing v1/v2
modules act as natural transformations that enrich the same entity spine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AnalyticsDiagram:
    """Named diagram from data objects to enriched performance entities."""

    source_category: str = "canonical match data and entity records"
    target_category: str = "performance measures and predictive distributions"
    functor: str = "F: MatchData -> PerformanceMeasures"
    natural_transformations: tuple[str, ...] = field(default_factory=lambda: (
        "graph/pass_network",
        "network_flow/xT",
        "milp/player_value_score",
        "performance/RAPM",
        "headtohead/matchup_prediction",
        "inference/FDR_insight",
    ))

    def statement(self) -> str:
        """Return the brief coherence statement used by docs/UI metadata."""
        transforms = ", ".join(self.natural_transformations)
        return (
            f"{self.functor} maps {self.source_category} to {self.target_category}; "
            f"{transforms} are composable enrichments of the same entity spine."
        )


def functorial_statement() -> str:
    """Public helper for the v3 coherence statement."""
    return AnalyticsDiagram().statement()
