# Unified Mathematical Football Analytics System — Specification (v1.0)

This document maps each part of the project specification to its
implementation in `src/fas/`, with the governing mathematical formulation
restated for each module. It is the canonical reference for the codebase.

---

## Part 0 — Data Infrastructure → `fas.data`

Canonical action schema
`A = (t, player_id, team_id, action_type, x_start, y_start, x_end, y_end, outcome, freeze_frame)`
on the standardized StatsBomb pitch `[0,120] × [0,80]`.

| Spec | Code |
|---|---|
| Action schema + validation | [`schema.py`](../src/fas/data/schema.py) — `Action`, `validate_actions` |
| StatsBomb → canonical mapping | [`statsbomb.py`](../src/fas/data/statsbomb.py) — `events_to_actions` |
| Cross-source id unification (Jaro-Winkler ≥ 0.92 + nationality) | [`unify_ids.py`](../src/fas/data/unify_ids.py) — `match_rosters` |
| PostgreSQL DDL | [`schema.sql`](../src/fas/data/schema.sql) |

---

## Part 1 — Graph Theory → `fas.graph`

Weighted directed pass network `G = (V, E, W)`, row-stochastic Markov matrix
`P[i,j] = W[i,j] / Σ_k W[i,k]`.

- **Degree** `C_D(i) = (d_in + d_out)/(2(n-1))`
- **Betweenness** `C_B(i) = Σ σ_st(i)/σ_st` (Brandes)
- **Closeness** on reciprocal-weight distances `d = 1/W`
- **PageRank** `π = πP` (stationary distribution, steady-state ball touch)
- **Entropy** `H(G) = -Σ P[i,j] log P[i,j]`
- **Min-cut pressing** — max-flow/min-cut (GK source → attacking-third sink)
  via preflow-push, yielding ranked `(presser, target, priority)` triples.

| Spec | Code |
|---|---|
| Network construction, entropy, temporal snapshots, network velocity | [`pass_network.py`](../src/fas/graph/pass_network.py) |
| All four centralities + clustering | [`centrality.py`](../src/fas/graph/centrality.py) |
| §1.3 Min-cut pressing assignment | [`min_cut_pressing.py`](../src/fas/graph/min_cut_pressing.py) |
| §1.4 GAT collective valuation (stub, needs `[gnn]`) | [`gat_model.py`](../src/fas/graph/gat_model.py) |

---

## Part 2 — Network Flow → `fas.network_flow`

18-zone pitch graph (6 bands × 3 thirds). Expected Threat by value iteration:

`xT(c) = s(c)·g(c) + m(c)·Σ_{c'} T(c→c')·xT(c')`, iterated to `‖Δ‖_∞ < 1e-6`.

Build-up potency = **min-cost max-flow** from defensive-third to attacking-third
zones (cost = −value, so min-cost = max-reward; successive shortest paths).
Defensive suppression = bilevel program, single-levelled by LP duality and
solved by successive linearization + McCormick envelopes.

| Spec | Code |
|---|---|
| §2.1 xT surface (value iteration) + per-action xT-added | [`xt_surface.py`](../src/fas/network_flow/xt_surface.py) |
| §2.1–2.2 zone graph + max-flow build-up | [`max_flow_buildup.py`](../src/fas/network_flow/max_flow_buildup.py) |
| §2.3 bilevel suppression (needs `[nlp]`/cvxpy) | [`defensive_suppression.py`](../src/fas/network_flow/defensive_suppression.py) |

---

## Part 3 — Nonlinear Programming → `fas.nlp`

| Spec | Code | Status |
|---|---|---|
| §3.1 EPV U-Net | [`epv_unet.py`](../src/fas/nlp/epv_unet.py) | stub, needs `[ml]` |
| §3.2 Pass value `V = R − K`, SQP multi-start + exclusion ellipsoids | [`pass_value_nlp.py`](../src/fas/nlp/pass_value_nlp.py) | implemented (scipy SLSQP) |
| §3.3 Trajectory optimal control, direct collocation | [`trajectory_opt.py`](../src/fas/nlp/trajectory_opt.py) | stub, needs `[nlp]`/IPOPT |
| §3.4 Set-piece Magnus-force trajectory | [`set_piece_opt.py`](../src/fas/nlp/set_piece_opt.py) | implemented (RK4 + Nelder-Mead) |

---

## Part 4 — MILP: Squad Selection & Valuation → `fas.milp`

**PVS pipeline (§4.1):** per-90 features → Bradley-Terry league strength
`λ_L = mean exp(β)` → Robust PCA `F = L + S` (ADMM/PCP) → embedding `z_i` →
positional percentile `PVS_i = Φ((z_i−μ_P)/σ_P)` → fair-value WLS regression on
`log(MarketValue)`.

**Squad MILP (§4.2):** maximize `Σ w_p PVS_i y_ip + γ·NetworkBonus(x)` subject
to coverage (C1), starting XI (C2), squad size (C3), role links (C4), budget
(C5), age (C6), quota (C7), formation choice with `δ_f` (C8), synergy (C9),
flexibility sets (C10), and the linearized `q_ij = x_i x_j` bonus (C11).

| Spec | Code |
|---|---|
| §4.1 Bradley-Terry, robust PCA, PVS, fair value | [`player_valuation.py`](../src/fas/milp/player_valuation.py) |
| §4.2 Squad MILP (C1–C11, PuLP/CBC) | [`squad_selection.py`](../src/fas/milp/squad_selection.py) |
| §4.3 Robust minimax-regret MILP (extensive form / Benders) | [`robust_milp.py`](../src/fas/milp/robust_milp.py) |
| §4.4 In-match substitution MILP | [`substitution_milp.py`](../src/fas/milp/substitution_milp.py) |

---

## Part 5 — Boolean & Dual Functions → `fas.boolean`

Decision-list learner for tactical events; **dual** `f^d(b) = ¬f(¬b)` gives the
complementary escape conditions. Formations as elements of the Boolean lattice
`(2^R, ⊆)`; transitions are Hasse edges (`|F △ F'| = 1`); fit a formation
Markov chain and extract its stationary distribution.

| Spec | Code |
|---|---|
| §5.1 Decision lists + dual function | [`pattern_recognition.py`](../src/fas/boolean/pattern_recognition.py) |
| §5.2 Boolean lattice + formation Markov chain | [`formation_lattice.py`](../src/fas/boolean/formation_lattice.py) |

---

## Part 6 — Cross-League Valuation → `fas.valuation`

Three-layer normalization: (1) within-league inverse-normal percentile,
(2) Bradley-Terry league factor with feature-specific shrinkage `α_k`,
(3) Beta-shaped career-curve projection to peak age. Scouting-report generator
renders the §6.3 format.

| Spec | Code |
|---|---|
| §6.2 Layers 1–2 normalization | [`cross_league_normalization.py`](../src/fas/valuation/cross_league_normalization.py) |
| §6.2 Layer 3 development curves | [`development_curves.py`](../src/fas/valuation/development_curves.py) |
| §6.3 Scouting report | [`scouting_report.py`](../src/fas/valuation/scouting_report.py) |

---

## Part 7 — Evaluation → `fas.evaluation`

ECE ≤ 0.02 (EPV calibration), next-goal AUC ≥ 0.72, formation ARI ≥ 0.70,
log-value RMSE ≤ 0.35, Spearman(PVS, minutes) ≥ 0.55. The offline synthetic
pipeline ([`examples/synthetic_pipeline.py`](../src/fas/examples/synthetic_pipeline.py),
run via `fas demo`) is the §7.2 integration test.

| Spec | Code |
|---|---|
| §7.1 ECE, ARI, log-value RMSE | [`metrics.py`](../src/fas/evaluation/metrics.py) |
| §7.2 end-to-end pipeline | [`synthetic_pipeline.py`](../src/fas/examples/synthetic_pipeline.py) + [`tests/test_pipeline.py`](../tests/test_pipeline.py) |

---

## Part 9 — Research Extensions (open problems)

1. Strong-duality solution + McCormick-tightness analysis of the pressing
   bilevel program (§2.3 / `defensive_suppression.py`).
2. EPV as a martingale; Doob decomposition.
3. Laplacian spectrum of pass networks — Fiedler value vs. pressing resilience.
4. Multi-objective squad Pareto front (quality / youth / budget / cohesion) by
   ε-constraint.
5. Stochastic EPV with Lévy jumps (goals, red cards) → optimal substitution
   stopping times.
6. Transfer-market efficiency test using the §4.1 fair-value model.

---

*Full prose formulations are in the original project prompt; this document is
the authoritative spec-to-code map.*
