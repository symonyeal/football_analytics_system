# Unified Mathematical Football Analytics System (`fas`)

`fas` is a Python package for football analytics experiments. It keeps the
math explicit, but the code stays practical: event data comes in through one
canonical action schema, and the modules build graph, flow, optimization,
performance, matchup, and inference outputs on top of it.

The project is intentionally modular. Most modules run on the core install.
Heavier libraries are kept behind extras so the package remains usable on newer
Python versions while wheels catch up.

## What Is Here

| Area | Main idea | Package |
|---|---|---|
| Data | Canonical StatsBomb-style action table | `fas.data` |
| Passing networks | Weighted directed graphs, centrality, min-cut pressing | `fas.graph` |
| Ball progression | xT, zone graphs, max-flow buildup | `fas.network_flow` |
| On-ball decisions | Pass value, set-piece optimization, EPV stubs | `fas.nlp` |
| Squad planning | Player value scores, squad MILP, robust scenarios | `fas.milp` |
| Tactical states | Decision lists, dual Boolean functions, formation chains | `fas.boolean` |
| Valuation | Cross-league adjustment, development curves, reports | `fas.valuation` |
| Entity spine | `MatchObject`, `PlayerSeason`, `TeamSeason`, `Matchup` | `fas.entities` |
| Performance | RAPM, skill, IRT, form, roles, scoring, pitch control | `fas.performance` |
| Matchups | Paired comparison, rankings, tensors, copulas, Hawkes | `fas.headtohead` |
| Inference | TDA, RMT, OT, MMD, causality, controlled insight text | `fas.inference` |

## Install

The project targets Python 3.11+. It has been run on Python 3.14, so optional
dependencies are split out where wheel support may lag.

```bash
pip install -e .
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[ml]"      # torch, lightgbm
pip install -e ".[gnn]"     # torch-geometric
pip install -e ".[nlp]"     # cvxpy, cyipopt, cma
pip install -e ".[scrape]"  # soccerdata, understatapi
pip install -e ".[db]"      # sqlalchemy, psycopg2
pip install -e ".[bayes]"   # pymc, numpyro where available
pip install -e ".[topo]"    # ripser, gudhi where available
pip install -e ".[ot]"      # POT
pip install -e ".[tensor]"  # tensorly
pip install -e ".[pp]"      # hawkeslib where available
```

Modules that need an optional extra either provide a small core fallback or
raise a clear error naming the extra.

## Quick Start

```bash
fas demo
fas demo --data data/processed/actions.parquet
fas version
```

The demo first looks for a canonical actions file under `data/`. If none is
there, it uses the synthetic fallback. The command now exercises the old core
pipeline and the v3 layer: pass network, xT, zone flow, player value scoring,
RAPM, skill summaries, roles, possession value, matchup models, insight checks,
and squad selection. By default it writes `data/processed/demo_summary.json`.

```python
from statsbombpy import sb
from fas.data.statsbomb import events_to_actions
from fas.graph import build_pass_network, centrality_table
from fas.network_flow import fit_xt, build_zone_graph, buildup_potency

events = sb.events(match_id=3788741)
actions = events_to_actions(events, match_id=3788741)

net = build_pass_network(actions, team_id=...)
centrality = centrality_table(net)
xt = fit_xt(actions)
flow = buildup_potency(build_zone_graph(actions, team_id=..., xt_model=xt))
```

## Implementation Status

Core modules covered by tests:

- Canonical action schema, StatsBomb loader, id matching, and SQL schema.
- Pass networks, centrality, entropy, temporal snapshots, and min-cut pressing.
- xT value iteration, 18-zone flow graphs, and min-cost max-flow buildup.
- Pass-value and set-piece optimizers.
- Bradley-Terry league strength, robust PCA, PVS, fair-value regression,
  squad MILP, robust MILP, and substitution MILP.
- Decision-list learning, dual Boolean functions, formation lattice, and
  formation Markov chains.
- Cross-league normalization, development curves, and scouting reports.
- v3 entity spine, performance models, head-to-head models, and inference tools
  with dependency-light fallbacks.
- `fas demo` runs the integrated local-data-first path and returns the entity
  spine plus a compact v3 summary.

Documented stubs:

- `fas.graph.gat_model`: GAT collective valuation, needs `[gnn]`.
- `fas.nlp.epv_unet`: EPV U-Net, needs `[ml]`.
- `fas.nlp.trajectory_opt`: direct-collocation trajectory optimizer, needs `[nlp]`.

## Layout

```text
src/fas/
  data/           canonical schema, loaders, id matching
  entities.py     MatchObject, PlayerSeason, TeamSeason, Matchup
  foundations/    point process, performance functional, context operators
  graph/          pass networks, centrality, min-cut pressing
  network_flow/   xT, zone graphs, max-flow buildup
  nlp/            EPV/pass-value/trajectory/set-piece modules
  milp/           player valuation and squad/substitution optimization
  boolean/        tactical decision lists and formation lattice
  valuation/      cross-league normalization and reports
  performance/    player/team performance models
  headtohead/     matchup and result models
  inference/      higher-order inference and insight extraction
  evaluation/     calibration and validation metrics
  examples/       offline synthetic pipeline

tests/            pytest coverage for core paths
docs/SPEC.md      compact spec-to-code map
docs/DECISIONS.md project decision log
docs/NEXT_PHASE_PROMPT.md prompt for the next implementation phase
```

## Data

The code is written around public football data sources: StatsBomb Open Data,
FBref, Transfermarkt-style valuation tables, understat xG, and historical
results/odds from football-data.co.uk. The offline test suite does not download
data.

## License

MIT.
