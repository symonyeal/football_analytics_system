"""Product layer for ``fas``.

This package turns the mathematical engine in ``fas`` into a runnable
analytics product: a local-data-first data spine, a rich deterministic
synthetic fallback, a six-layer artifact materializer, and a loader the UI
reads without re-running models on every interaction.

Entry points:

- :func:`fas.product.build.product_build` — materialize all artifacts.
- :func:`fas.product.loader.load_product` — read artifacts for the UI.
- :func:`fas.product.synthetic.generate_league` — deterministic demo data.
"""

from __future__ import annotations

ARTIFACT_FILES = (
    "actions.parquet",
    "matches.parquet",
    "teams.parquet",
    "players.parquet",
    "match_artifacts.parquet",
    "player_artifacts.parquet",
    "team_artifacts.parquet",
    "matchup_artifacts.parquet",
    "insights.parquet",
    "pass_network_edges.parquet",
    "pass_clusters.parquet",
    "centralisation.parquet",
    "formations.parquet",
    "zone_flow.parquet",
    "xt_surface.parquet",
    "scorelines.parquet",
    "product_summary.json",
    "manifest.json",
)

__all__ = ["ARTIFACT_FILES"]
