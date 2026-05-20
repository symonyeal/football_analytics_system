# `fas` Spec Map

This document is a map from the project specification to the code. It is not a
second README and it does not repeat every derivation. The long-form prompt in
`football_analytics_phd_project.md` is the source for the mathematical detail;
this file shows where each piece lives.

## Shared Data Model

Canonical actions use the StatsBomb pitch, `[0, 120] x [0, 80]`, and the table

```text
(match_id, period, timestamp_ms, player_id, team_id, action_type,
 x_start, y_start, x_end, y_end, outcome)
```

Freeze-frame data is kept in object/JSON form when available.

| Spec area | Code |
|---|---|
| Action record and validation | [`src/fas/data/schema.py`](../src/fas/data/schema.py) |
| StatsBomb conversion | [`src/fas/data/statsbomb.py`](../src/fas/data/statsbomb.py) |
| Cross-source id matching | [`src/fas/data/unify_ids.py`](../src/fas/data/unify_ids.py) |
| SQL schema | [`src/fas/data/schema.sql`](../src/fas/data/schema.sql) |
| Entity spine | [`src/fas/entities.py`](../src/fas/entities.py) |
| Local action discovery for demos | [`src/fas/examples/synthetic_pipeline.py`](../src/fas/examples/synthetic_pipeline.py) |

## v1 Optimizers And Core Analytics

| Area | What it does | Code |
|---|---|---|
| Passing graph | Pass networks, entropy, snapshots, velocity | [`pass_network.py`](../src/fas/graph/pass_network.py) |
| Centrality | Degree, betweenness, closeness, PageRank, clustering | [`centrality.py`](../src/fas/graph/centrality.py) |
| Pressing cut | Min-cut pressing targets | [`min_cut_pressing.py`](../src/fas/graph/min_cut_pressing.py) |
| GAT stub | Graph-attention collective valuation | [`gat_model.py`](../src/fas/graph/gat_model.py) |
| xT | Expected Threat by value iteration | [`xt_surface.py`](../src/fas/network_flow/xt_surface.py) |
| Buildup flow | 18-zone graph and min-cost max-flow | [`max_flow_buildup.py`](../src/fas/network_flow/max_flow_buildup.py) |
| Suppression | Defensive flow suppression, optional `cvxpy` | [`defensive_suppression.py`](../src/fas/network_flow/defensive_suppression.py) |
| EPV stub | U-Net EPV model interface | [`epv_unet.py`](../src/fas/nlp/epv_unet.py) |
| Pass value | Reward-risk pass optimizer | [`pass_value_nlp.py`](../src/fas/nlp/pass_value_nlp.py) |
| Trajectory stub | Direct-collocation player movement | [`trajectory_opt.py`](../src/fas/nlp/trajectory_opt.py) |
| Set pieces | Magnus-force delivery optimizer | [`set_piece_opt.py`](../src/fas/nlp/set_piece_opt.py) |
| Player value | Bradley-Terry, robust PCA, PVS, fair value | [`player_valuation.py`](../src/fas/milp/player_valuation.py) |
| Squad MILP | Formation, budget, role, age, quota constraints | [`squad_selection.py`](../src/fas/milp/squad_selection.py) |
| Robust MILP | Bootstrap/scenario squad robustness | [`robust_milp.py`](../src/fas/milp/robust_milp.py) |
| Substitutions | In-match substitution optimizer | [`substitution_milp.py`](../src/fas/milp/substitution_milp.py) |
| Boolean patterns | Decision lists and dual functions | [`pattern_recognition.py`](../src/fas/boolean/pattern_recognition.py) |
| Formations | Boolean lattice and Markov chain | [`formation_lattice.py`](../src/fas/boolean/formation_lattice.py) |
| League adjustment | Cross-league feature normalization | [`cross_league_normalization.py`](../src/fas/valuation/cross_league_normalization.py) |
| Development | Beta-style career curves | [`development_curves.py`](../src/fas/valuation/development_curves.py) |
| Reports | Scouting report renderer | [`scouting_report.py`](../src/fas/valuation/scouting_report.py) |

## v3 Foundations

These modules give later models a common language. They are small on purpose.

| Spec area | Code |
|---|---|
| Marked point process and intensity surface | [`point_process.py`](../src/fas/foundations/point_process.py) |
| Performance functional, `Pi(e, W) = integral phi dV_e` | [`performance_functional.py`](../src/fas/foundations/performance_functional.py) |
| Context operators for strength, game state, venue, fatigue | [`context_ops.py`](../src/fas/foundations/context_ops.py) |
| Coherence statement for entity enrichment | [`coherence.py`](../src/fas/foundations/coherence.py) |

## v3 Performance Models

| Spec area | Code |
|---|---|
| Regularized adjusted plus-minus | [`rapm.py`](../src/fas/performance/rapm.py) |
| Empirical-Bayes skill posteriors | [`bayesian_skill.py`](../src/fas/performance/bayesian_skill.py) |
| 2PL item-response model | [`irt.py`](../src/fas/performance/irt.py) |
| Kalman form state | [`form_state.py`](../src/fas/performance/form_state.py) |
| NMF role discovery | [`roles_nmf.py`](../src/fas/performance/roles_nmf.py) |
| Dixon-Coles scoring model | [`team_scoring.py`](../src/fas/performance/team_scoring.py) |
| Possession MDP | [`possession_mdp.py`](../src/fas/performance/possession_mdp.py) |
| Pitch-control field | [`pitch_control.py`](../src/fas/performance/pitch_control.py) |
| Fisher-Rao style distribution | [`style_manifold.py`](../src/fas/performance/style_manifold.py) |

## v3 Head-To-Head Models

| Spec area | Code |
|---|---|
| Bradley-Terry-Davidson with covariates | [`paired_comparison.py`](../src/fas/headtohead/paired_comparison.py) |
| Massey, Colley, PageRank result rankings | [`network_ranking.py`](../src/fas/headtohead/network_ranking.py) |
| CP matchup tensor factorization | [`matchup_tensor.py`](../src/fas/headtohead/matchup_tensor.py) |
| Gaussian copula outcome simulation | [`copula_outcomes.py`](../src/fas/headtohead/copula_outcomes.py) |
| Hawkes momentum model | [`hawkes_momentum.py`](../src/fas/headtohead/hawkes_momentum.py) |

## v3 Inference And Insight

| Spec area | Code |
|---|---|
| Shape summaries from persistence features | [`tda_shape.py`](../src/fas/inference/tda_shape.py) |
| Marchenko-Pastur covariance cleaning | [`rmt_clean.py`](../src/fas/inference/rmt_clean.py) |
| Sinkhorn style distances and barycenters | [`ot_style.py`](../src/fas/inference/ot_style.py) |
| Kernel MMD tests and kernel ridge prediction | [`kernel_mmd.py`](../src/fas/inference/kernel_mmd.py) |
| Granger and transfer-entropy influence | [`causality.py`](../src/fas/inference/causality.py) |
| Bootstrap, FDR, Shapley, insight templates | [`insight_engine.py`](../src/fas/inference/insight_engine.py) |

## Evaluation

| Metric/tool | Code |
|---|---|
| ECE, ARI, log-value RMSE | [`metrics.py`](../src/fas/evaluation/metrics.py) |
| Offline integration pipeline | [`synthetic_pipeline.py`](../src/fas/examples/synthetic_pipeline.py) |
| CLI runner | [`cli.py`](../src/fas/cli.py) |
| Tests | [`tests/`](../tests) |

The package uses lightweight fallbacks for the v3 research modules. Optional
extras (`[bayes]`, `[topo]`, `[ot]`, `[tensor]`, `[pp]`) are available for
heavier backends when the local Python environment supports them.

## Next Phase

The next phase is to replace the synthetic fallback with checked-in or
downloaded public competition data, then persist the same entity outputs in a
small feature/model store. The current `fas demo` path is the acceptance shape:
local data is discovered, entities are materialized, v1/v3 metrics run, and a
summary is written for the UI or notebook layer to read.

## Product Layer (real-data-first)

| Concern | Code | Output |
|---|---|---|
| Data spine (priority: `--data` → local → real StatsBomb → synthetic) | [`ingest.py`](../src/fas/product/ingest.py) | `DataSpine` + `manifest.json` |
| Real StatsBomb ingestion (names, positions, formations) | [`ingest.load_statsbomb`](../src/fas/product/ingest.py) | canonical actions + metadata |
| Deterministic synthetic fallback | [`synthetic.py`](../src/fas/product/synthetic.py) | 6 matches, 4 teams, lineups |
| L1 events: xT, action value, zones, phases | [`artifacts.enrich_actions`](../src/fas/product/artifacts.py) | `actions.parquet` |
| L2/L3 space + networks + control | [`artifacts.build_match_layer`](../src/fas/product/artifacts.py) | edges, centralisation, zone-flow |
| Centralisation (Freeman on PageRank) | [`centralisation.py`](../src/fas/product/centralisation.py) | `centralisation.parquet` |
| L4 pass clustering (DBSCAN→KMeans) | [`clustering.py`](../src/fas/product/clustering.py) | `pass_clusters.parquet` |
| L4 formation inference (lineup + phase) | [`formation.py`](../src/fas/product/formation.py) | `formations.parquet` |
| L5 players/roles/PVS/form/value | [`artifacts.build_player_layer`](../src/fas/product/artifacts.py) | `player_artifacts.parquet` |
| L6 matchups (Dixon-Coles, BT, style, min-cut) | [`artifacts.build_matchup_layer`](../src/fas/product/artifacts.py) | `matchup_artifacts.parquet`, `scorelines.parquet` |
| Insight engine (FDR + exploratory cards) | [`artifacts.build_insights`](../src/fas/product/artifacts.py) | `insights.parquet` |
| Build orchestrator | [`build.py`](../src/fas/product/build.py) | all artifacts + `product_summary.json` |
| Artifact loader | [`loader.py`](../src/fas/product/loader.py) | `Product` bundle |
| UI: charts / Streamlit app / HTML report | [`ui/`](../src/fas/ui) | 6 views + static report |

Every long-format `*_artifacts` table carries the context contract: match,
team, opponent, competition, season, phase, minute window, model name/version,
sample size, data source, `is_synthetic`, and limitations.
