# Unified Mathematical Football Analytics System (`fas`)

A research-grade soccer analytics stack that integrates **graph theory**,
**network flow optimization**, **nonlinear programming**, **mixed-integer
linear programming (MILP)**, and **Boolean / dual functions** over publicly
available event data. It culminates in a cross-league player-valuation engine
and an optimal squad-selection framework.

This is the implementation of the *Unified Mathematical Football Analytics
System* specification (v1.0). See [`docs/SPEC.md`](docs/SPEC.md) for the full
mathematical formulation of every module.

---

## Why this exists

Most football analytics tools stop at descriptive per-90 statistics. This
system treats the game as a sequence of mathematical objects:

| Object | Mathematics | Module |
|---|---|---|
| Passing as a network | Weighted digraphs, centrality, min-cut | `fas.graph` |
| Ball progression | Network flow, max-flow / min-cost flow, xT value iteration | `fas.network_flow` |
| Decision-making on the ball | Constrained NLP, optimal control, projectile dynamics | `fas.nlp` |
| Squad construction | MILP, robust optimization, Benders / minimax-regret | `fas.milp` |
| Tactical states | Boolean functions, duality, the Boolean lattice | `fas.boolean` |
| Cross-league value | Bradley-Terry, robust PCA, Beta career curves | `fas.milp` + `fas.valuation` |

---

## Install

The project targets **Python 3.11+** (developed and run on 3.14). Core
dependencies are pure-Python / widely-wheeled; the heavy ML / global-solver
dependencies are split into optional extras because their wheels can lag on the
newest interpreters.

```bash
pip install -e .              # core: numpy, scipy, networkx, pulp, statsbombpy, ...
pip install -e ".[dev]"       # + pytest, ruff
pip install -e ".[ml]"        # + torch, lightgbm        (EPV U-Net, sub model)
pip install -e ".[gnn]"       # + torch-geometric         (GAT, Part 1.4)
pip install -e ".[nlp]"       # + cvxpy, cyipopt, cma      (suppression, trajectory)
pip install -e ".[scrape]"    # + soccerdata, understatapi (FBref / understat)
pip install -e ".[db]"        # + sqlalchemy, psycopg2     (PostgreSQL schema)
```

> On Python 3.14, `torch` / `torch-geometric` / `cyipopt` wheels may not yet be
> available. The modules that need them raise a clear `NotImplementedError`
> pointing to the extra; **every other module runs on the core install.**

## Quick start

```bash
fas demo          # runs the offline synthetic end-to-end pipeline (no data download)
fas version
```

```python
from statsbombpy import sb
from fas.data.statsbomb import events_to_actions
from fas.graph import build_pass_network, centrality_table, pressing_assignment
from fas.network_flow import fit_xt, build_zone_graph, buildup_potency

events  = sb.events(match_id=3788741)            # La Liga, Appendix B
actions = events_to_actions(events, match_id=3788741)

net   = build_pass_network(actions, team_id=...)  # weighted pass digraph
cents = centrality_table(net)                     # degree / betweenness / closeness / PageRank
xt    = fit_xt(actions)                            # Expected Threat surface (value iteration)
flow  = buildup_potency(build_zone_graph(actions, team_id=..., xt_model=xt))
```

---

## What is implemented vs. specified

**Fully implemented (core install, exercised by `fas demo` + tests):**

- **Part 0** — canonical action schema, StatsBomb loader, Jaro-Winkler id
  unification, PostgreSQL DDL.
- **Part 1** — pass-network construction, all four centralities (degree,
  Brandes betweenness, closeness, PageRank), clustering, network entropy,
  temporal snapshots / network velocity, **min-cut pressing assignment**.
- **Part 2** — **xT surface by value iteration**, 18-zone flow graph,
  **min-cost max-flow build-up potency**; suppression bilevel program via
  successive linearization + McCormick envelopes (needs `cvxpy`).
- **Part 3.2 / 3.4** — pass reward-risk optimizer (SLSQP multi-start),
  set-piece Magnus-force trajectory optimizer (RK4 + Nelder-Mead).
- **Part 4** — Bradley-Terry league strength, **robust PCA via ADMM**, PVS
  positional percentiles, fair-value WLS regression, **squad-selection MILP**
  (all constraints C1-C11 + formation choice), **minimax-regret robust MILP**,
  **substitution MILP**.
- **Part 5** — decision-list learner, **dual Boolean function**, formation
  Boolean lattice + Markov chain + stationary distribution.
- **Part 6** — three-layer cross-league normalization, Beta career curves,
  scouting-report generator.
- **Part 7** — ECE, ARI, log-value RMSE; synthetic end-to-end integration test.

**Documented stubs (need optional heavy deps, full formulation in docstrings):**

- **Part 1.4** Graph Attention Network (`fas.graph.gat_model`) — `[gnn]`.
- **Part 3.1** EPV U-Net (`fas.nlp.epv_unet`) — `[ml]`.
- **Part 3.3** trajectory direct collocation (`fas.nlp.trajectory_opt`) — `[nlp]`.

---

## Project layout

```
src/fas/
├── data/            schema, StatsBomb loader, id unification, schema.sql
├── graph/           pass_network, centrality, min_cut_pressing, gat_model
├── network_flow/    xt_surface, max_flow_buildup, defensive_suppression
├── nlp/             epv_unet, pass_value_nlp, trajectory_opt, set_piece_opt
├── milp/            player_valuation, squad_selection, robust_milp, substitution_milp
├── boolean/         pattern_recognition, formation_lattice
├── valuation/       cross_league_normalization, development_curves, scouting_report
├── evaluation/      metrics
└── examples/        synthetic_pipeline   (fas demo)
tests/               pytest unit + integration tests
docs/SPEC.md         full mathematical specification (v1.0)
```

## Data sources

StatsBomb Open Data, FBref (via `soccerdata`), Transfermarkt valuations,
understat xG, football-data.co.uk. All public. See `docs/SPEC.md` Part 0.

## License

MIT.
