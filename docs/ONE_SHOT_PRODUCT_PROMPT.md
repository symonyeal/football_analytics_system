# One-Shot Product Prompt

Use this prompt to ask a coding agent to build the next full product pass of
the football analytics system.

---

You are working in this repository:

```text
C:\Users\sislam\OneDrive - Government of Yukon\Documents\GitHub\Codex\F U N\football_analytics_system
```

Build a complete, usable football analytics product on top of the existing
`fas` package. The product must preserve the current mathematical ambition of
the repo, but it must become practical: real or local data first, a coherent
entity spine, persisted artifacts, and a polished UI where every chart is tied
to exact football context rather than floating metrics.

The final result must be runnable by the user immediately. Do not finish with a
design-only answer, skeleton UI, or instructions that depend on the user finding
data elsewhere. The implementation must ensure there is enough local data or
generated fallback data in `data/processed/` for the user to run one command and
see populated charts, tables, insights, and data-quality status. At the end,
print the exact command(s) the user should run and what they should expect to
see.

Do not start from a blank app. Read the repo first and build with the existing
architecture:

- `README.md`
- `football_analytics_phd_project.md`
- `docs/SPEC.md`
- `docs/DECISIONS.md`
- `docs/NEXT_PHASE_PROMPT.md`
- `src/fas/**`
- `tests/**`

The current package already contains data schema validation, StatsBomb-style
actions, graph analytics, xT and flow, optimization, valuation, Boolean pattern
logic, performance models, head-to-head models, and inference tools. Treat those
as the engine. Your job is to connect them into a real data spine, feature
artifacts, and an interactive product.

## Product Thesis

Design the system around one principle:

> Football analytics should translate the continuous geometry of the game into
> contextual decisions, while still respecting the discrete event data that most
> public datasets actually provide.

Football is not only a list of passes, shots, and tackles. It is a dynamic field
of space, timing, pressure, role, opponent behavior, and tactical state. But do
not pretend we have perfect tracking data when we only have event data. The
system should build a bridge:

- discrete events become canonical actions;
- actions become possessions, pass networks, zones, phases, and role states;
- spatial patterns become graph, flow, topology, pitch-control, and style
  summaries;
- model outputs become validated insight cards with baselines, uncertainty,
  context, and sample size;
- the UI answers football questions, not "what charts can we draw?"

Use modern football analytics where it already exists. Do not recreate the wheel
for established concepts such as xG, xT, VAEP/OBV-style action value, pass
networks, centralisation, Elo/team strength, or pitch maps. Implement reliable
adapters and transparent fallbacks. When a mature external library or public
reference exists and fits the repo, use it or design an interface around it.
When the repo already has an implementation, use that first.

## Sources And Ideas To Fold In

Use the following ideas as product requirements, not as decorative citations:

- The existing `fas` spec frames the game as a multi-model mathematical system:
  probability for xG/projections, geometry for passes and space, mechanics for
  trajectories, game theory for pressing and opposition plans, graph theory for
  passing networks, topology for control of pitch regions, optimization for
  squad and substitution choices, and inference for trustworthy insights.
- The current repo has a canonical action schema:
  `(match_id, period, timestamp_ms, player_id, team_id, action_type, x_start,
  y_start, x_end, y_end, outcome)` on a StatsBomb `[0,120] x [0,80]` pitch.
- The current repo has a canonical entity spine:
  `MatchObject`, `PlayerSeason`, `TeamSeason`, and `Matchup`. All new product
  artifacts must attach to those records or to serializable views of those
  records.
- StatsBomb 360 line-breaking analysis shows the value of adding defensive
  context to event locations. If 360 freeze frames are available, detect
  line-breaking passes and pressure-aware actions; if not, expose the missing
  context explicitly and provide event-only approximations.
- Pass clustering work should be used for spatial style discovery: cluster pass
  vectors by start/end coordinates, mirror where tactically appropriate, and
  label clusters by football meaning only after inspecting their geometry.
- Formation visualization work should be rule-based and interpretable at first:
  use lineup positions to infer structures like `4-2-1-3`, map positions to
  stable StatsBomb pitch coordinates, and compare shapes across matches.
- Centralisation analysis should distinguish structure from control: a team can
  keep the same formation while moving influence from one hub to another. Track
  centralisation per match, per phase, and over time, excluding goalkeepers when
  appropriate and normalizing for minutes/sample size.
- FPL-style datasets are useful for discrete player context: price, ownership,
  minutes, form, xG/xA, gameweek snapshots, and Elo-adjusted fixtures. Treat
  them as supplementary context for player and matchup views, not as a
  replacement for event/spatial analysis.
- Edd Webster-style resource aggregation points to a practical ecosystem:
  StatsBomb, Wyscout/SPADL, Understat, FBref, Transfermarkt, ClubElo,
  football-data.co.uk, xT grids, EPV grids, and visualization libraries such as
  `mplsoccer`. Prefer these public sources and adapters.
- Graph-based sports analytics is a mature research area. Use graph
  representations deliberately: pass networks, result networks, influence
  networks, role relationships, and suppression/pressing cuts should each have
  clear semantics.
- "Beyond numbers and charts" is a hard requirement: every metric must be
  interpreted through role, opponent, game state, location, phase, pressure,
  fatigue, and tactical structure where available.

## Non-Negotiable Design Rules

1. Build the actual usable product, not a landing page.
2. The first screen must be an analytics workspace with real controls,
   summaries, and charts.
3. Every chart must show its context:
   match, team/player, competition/season, minute range, sample size, data
   source, filters, baseline, and uncertainty where available.
4. Every generated insight must include:
   metric, effect direction, baseline, confidence interval or credible interval
   where available, p/q value if a hypothesis scan was used, and a short
   football interpretation.
5. Do not show naked rankings without context. A player leaderboard must state
   position group, minutes threshold, league/team context, and whether the score
   is raw, adjusted, or model-derived.
6. Do not imply tracking-data precision from event-only data. If freeze frames
   are missing, use "event-only" labels and hide or degrade 360-specific
   analytics gracefully.
7. Keep new heavy dependencies optional. Preserve the existing lightweight
   fallback philosophy.
8. Preserve the CLI contract:

```bash
fas demo
fas demo --data data/processed/actions.parquet
fas demo --no-summary
```

9. Add a UI command or documented launch command that works locally without a
   network download when synthetic fallback data is the only data available.
10. Create or materialize all required demo/product data artifacts during the
    build path. The user should not have to manually create files before seeing
    the UI.
11. The final response must tell the user exactly what to run, including the
    artifact-build command and the UI command or local URL.
12. Tests must pass without network access.

## Recommended Implementation Shape

Keep the Python package as the source of truth and add a focused UI layer.

Preferred approach:

- Add `fas product-build` or `fas materialize` for artifact generation.
- Add `fas ui` or a documented UI launch script.
- The artifact command must be idempotent and must populate `data/processed/`
  with a complete small demo dataset if real local data is unavailable.
- The UI command must auto-build missing artifacts or fail with a clear message
  that includes the exact artifact-build command.
- Use Streamlit for the first product UI unless the repo already contains a
  different frontend stack by the time you start. Streamlit is appropriate here
  because the project is Python-native and analytics-heavy.
- Use `plotly` for interactive charts and `mplsoccer`/matplotlib for pitch
  maps. If adding these, put them under an optional `ui` extra where possible.
- Persist product artifacts under `data/processed/`:
  `actions.parquet`, `matches.parquet`, `players.parquet`, `teams.parquet`,
  `matchups.parquet`, `insights.parquet`, and `product_summary.json`.
- Build a small artifact loader that the UI can read without re-running all
  models on every page interaction.

If Streamlit is not available and cannot be installed, still create the artifact
generation path and a static HTML report fallback under `data/processed/report/`.

## Data Spine

Implement a local-data-first ingestion layer.

Data priority:

1. User-provided canonical file via `--data`.
2. Existing `data/processed/actions.parquet`.
3. Downloaded or locally cached StatsBomb Open Data sample.
4. Checked-in small sample if available.
5. Synthetic fallback, clearly labelled.

There must always be a visible dataset after setup. If real public data cannot
be downloaded because network access is unavailable, automatically generate a
rich deterministic synthetic dataset large enough to populate every product
view. This fallback must include:

- at least 3 matches;
- at least 2 teams;
- at least 18 players per team;
- passes, carries, shots, pressures, tackles/interceptions, and set-piece-like
  actions;
- match metadata and scores;
- basic lineup/formation metadata;
- player role/position metadata;
- enough repeated pass patterns to show pass clusters;
- enough phase variation to show centralisation and style drift;
- enough player samples to show player dashboards and insight cards;
- enough result rows to show matchup forecasts.

Write this generated dataset to normal product artifacts, not to hidden
in-memory objects. The UI should clearly badge it as deterministic synthetic
demo data.

Ingest at least one real public match set if network/cache permits. Prefer
StatsBomb Open Data because the repo already targets StatsBomb coordinates and
has `statsbombpy` in core dependencies.

The ingestion layer must:

- convert events into the canonical action schema;
- preserve match/team/player metadata;
- preserve lineup data for formation inference;
- preserve 360 freeze-frame availability flags;
- write deterministic Parquet artifacts;
- include a manifest with source, extraction time, competition, season, match
  IDs, row counts, and limitations;
- validate coordinates and action types;
- normalize all IDs used in downstream artifacts.

If supplementing with FPL/FBref/Understat/ClubElo-style tables:

- keep adapters optional;
- store source-specific raw columns separately from canonical derived fields;
- record join quality and unmatched entities;
- never silently merge players by name only. Use name, date of birth if
  available, nationality/team, and fuzzy score.

## Mathematical Product Layers

Organize the product into six layers. Each layer should have persisted outputs,
tests, and UI surfaces.

### Layer 1: Event And Possession Context

Compute:

- canonical actions;
- possessions and phases;
- shot table and xG if available or approximated;
- xT added per pass/carry;
- action value timeline;
- game state: score, period, minute, home/away, phase;
- opponent/team IDs;
- pressure/360 availability flags;
- sample-size metadata.

Charts:

- match event timeline;
- shot map with xG labels;
- xT added timeline;
- action value by phase;
- possession sequence explorer.

Every chart should be filterable by team, player, action type, period, minute
range, outcome, phase, and game state where available.

### Layer 2: Geometry And Space

Compute:

- pitch zone occupancy and action density;
- xT surface and zone transitions;
- line-breaking pass detection when 360 data exists;
- event-only progressive pass/carry approximations otherwise;
- pitch-control surface from freeze frames if available, or last-known/action
  location approximations clearly labelled;
- compactness, width, depth, and convex-hull style summaries;
- topology/shape summaries already available in `fas.inference.tda_shape`.

Charts:

- pitch heatmaps;
- pass maps;
- carry maps;
- xT surface;
- zone flow arrows;
- pitch-control overlay;
- line-breaking pass map;
- shape compactness over phases.

Principle:

Geometry charts must answer "where did control, threat, or progression live?"
They should not be generic heatmaps without a comparison baseline.

### Layer 3: Graphs, Networks, And Control

Use existing graph modules:

- `build_pass_network`
- `centrality_table`
- `network_entropy`
- `phase_snapshots`
- `network_velocity`
- `pressing_assignment`

Add or expose:

- centralisation index per match and phase;
- hub identity drift over time;
- decentralisation/complementarity view;
- result network rankings where match results exist;
- influence network from causality module where time series support it.

Charts:

- pass network on pitch using average action locations;
- phase-by-phase pass network comparison;
- centralisation trend;
- PageRank/betweenness role table;
- network velocity chart;
- pressing min-cut targets.

Interpretation:

Show that formation and control are different. A shape may stay stable while
the main hub changes from a midfielder to a forward, or while influence becomes
more evenly distributed.

### Layer 4: Style, Roles, And Tactical Pattern Discovery

Use existing modules:

- `fit_roles_nmf`
- `team_style_distribution`
- `fisher_rao_distance`
- `sinkhorn_distance`
- `mmd_permutation_test`
- Boolean decision lists and formation lattice.

Add pass clustering:

- cluster passes by `(x_start, y_start, x_end, y_end)` and optional features
  such as length, angle, pressure, phase, and outcome;
- support mirroring so left/right equivalents can be compared;
- prefer density-based clustering for pattern discovery when enough data exists;
- label clusters with readable geometry such as "left half-space switch",
  "central wall pass", "deep diagonal", "wide progression";
- show noise/unclustered passes honestly.

Add formation logic:

- infer starting formation from lineup positions;
- map StatsBomb positions to stable pitch coordinates;
- infer phase formations from average player action locations as a separate,
  lower-confidence signal;
- compare formation, pass clusters, and centralisation across matches/seasons.

Charts:

- pass cluster atlas;
- style distance matrix;
- style drift timeline;
- role membership matrix;
- formation comparison pitch;
- formation Markov chain;
- pattern rule cards.

Interpretation:

Make the UI explain whether a pattern is spatial, relational, or control-based.
For example: "Barcelona's shape remained similar, but control became less
centralized and the hub moved higher."

### Layer 5: Player, Squad, And Recruitment Intelligence

Use existing modules:

- `player_value_scores`
- `robust_pca`
- `low_rank_embedding`
- `fair_value_regression`
- `fit_hierarchical_skill`
- `fit_irt_2pl`
- `kalman_form`
- `fit_rapm`
- `development_curves`
- `cross_league_normalization`
- `solve_squad`
- `solve_robust_squad`
- `solve_substitutions`

Build player views around role-contextual comparison:

- position/role-adjusted metrics;
- minutes thresholds;
- age curve and development projection;
- form state with uncertainty;
- skill posterior;
- RAPM/action contribution where data supports it;
- PVS and fair value;
- market value gap if valuation source exists;
- tactical fit to squad roles and formation constraints.

Charts:

- player profile dashboard;
- role percentile radar or bar profile;
- development curve;
- value vs market scatter;
- skill uncertainty interval;
- form trend;
- squad optimization board;
- robust squad scenario comparison.

Principle:

Never say "best player" without saying "best for what role, context, budget,
minutes threshold, and tactical need."

### Layer 6: Matchup, Forecast, And Decision Support

Use existing modules:

- Dixon-Coles scoring model;
- Bradley-Terry-Davidson paired comparison;
- Massey/Colley/PageRank result rankings;
- matchup tensor;
- Gaussian copula scoreline simulation;
- Hawkes momentum;
- MDP possession value;
- pressing suppression and min-cut targeting.

Build:

- matchup page for team vs team;
- scoreline distribution;
- win/draw/loss probability;
- style clash explanation;
- pressing targets;
- likely buildup corridors;
- set-piece optimization summary;
- substitution/squad what-if controls.

Charts:

- scoreline probability grid;
- outcome probability bars;
- matchup style radar;
- zone-flow comparison;
- pressing target network;
- momentum event timeline;
- what-if panel for lineups/substitutions.

Interpretation:

A forecast is not enough. Explain which model produced it, what data it used,
what assumptions it makes, and what tactical levers appear actionable.

## Insight Engine Requirements

Use `fas.inference.insight_engine` as the controlled language layer.

Every insight card should contain:

- title;
- entity;
- context;
- claim;
- evidence;
- method;
- uncertainty;
- sample size;
- comparison baseline;
- caveats;
- recommended next look.

Example format:

```text
Claim: Team 100 progressed more through the right half-space after minute 60.
Evidence: right-half-space xT share rose from 22% to 39% against its match
baseline, n=46 progressive actions, bootstrap 95% CI [+0.08, +0.24].
Method: xT zone attribution + phase split + bootstrap CI.
Caveat: event-only data; no freeze-frame pressure context.
Next look: inspect pass clusters 4 and 7 and the right-back/right-wing edge.
```

Use FDR control for scans across many players/teams/metrics. Do not present
uncontrolled multiple-comparison discoveries as findings. Mark them as
"exploratory" if they are useful but not statistically validated.

## UI Specification

Build a work-focused analytics app. It should feel like an internal club
analysis tool: dense, calm, readable, and fast.

Main navigation:

1. Match Workspace
2. Team Style
3. Player Intelligence
4. Matchup Lab
5. Recruitment And Squad
6. Data Quality

Global controls:

- data source selector;
- competition;
- season;
- match;
- team;
- opponent;
- player;
- minute range;
- phase;
- action type;
- data mode: real/local/synthetic;
- confidence level where applicable.

### View 1: Match Workspace

Purpose: understand one match from event, space, network, and momentum layers.

Required panels:

- match metadata strip;
- event/xT timeline;
- shot map;
- pass network;
- zone flow map;
- phase selector;
- top validated insights;
- data quality badge.

Interactions:

- clicking a phase updates every chart;
- clicking a player highlights them across pitch, network, and tables;
- clicking an insight applies its filters if possible.

### View 2: Team Style

Purpose: compare how a team plays across matches or seasons.

Required panels:

- formation comparison;
- centralisation trend;
- network entropy trend;
- pass cluster atlas;
- style distance matrix;
- style drift timeline;
- buildup corridor chart;
- tactical interpretation cards.

### View 3: Player Intelligence

Purpose: evaluate players in role and context.

Required panels:

- player summary and role;
- minutes/sample badge;
- role-adjusted metric profile;
- xT/EPV/action value contribution;
- graph influence;
- skill/form uncertainty;
- development curve;
- similar-role comparison;
- contextual insight cards.

### View 4: Matchup Lab

Purpose: prepare for an opponent.

Required panels:

- win/draw/loss and scoreline distribution;
- style clash summary;
- opponent pass network;
- min-cut pressing targets;
- zone-flow strengths and vulnerabilities;
- likely momentum/event profile;
- set-piece or substitution what-if where available.

### View 5: Recruitment And Squad

Purpose: connect analytics to roster decisions.

Required panels:

- player shortlist;
- role filters;
- value vs market chart;
- age/development projection;
- cross-league normalization explanation;
- squad optimizer;
- robust scenario comparison;
- constraints summary.

### View 6: Data Quality

Purpose: make trust visible.

Required panels:

- source manifest;
- row counts by table;
- missing data by field;
- 360/freeze-frame availability;
- ID join quality;
- synthetic fallback warning;
- model availability and optional dependency status;
- test/validation status.

## Visual Encoding Contract

Use consistent encodings:

- pitch coordinates always StatsBomb `[0,120] x [0,80]`;
- team attacking direction must be explicit and consistent;
- xT/action value uses a sequential color scale;
- uncertainty uses bands or intervals, not hidden tooltips only;
- selected team/player uses a stable highlight color;
- opponent uses a separate neutral color;
- failed actions use lower opacity or dashed strokes;
- sample size appears in subtitles or badges;
- event-only approximations have visible labels.

Avoid chart junk. Do not use decorative graphics. Use football-native visuals:
pitch maps, networks, timelines, matrices, distributions, and constrained
tables.

## Artifact Contract

Create a reproducible product artifact layer. At minimum:

```text
data/processed/
  actions.parquet
  match_artifacts.parquet
  player_artifacts.parquet
  team_artifacts.parquet
  matchup_artifacts.parquet
  insights.parquet
  product_summary.json
  manifest.json
```

After `python -m fas.cli product-build --no-download`, these files must exist
and contain non-empty data. If the command uses synthetic fallback data, the
manifest must say so plainly and include the random seed/generation parameters.

Each artifact must include enough IDs and context fields for the UI to filter
without guessing:

- `match_id`
- `team_id`
- `opponent_id` where applicable
- `player_id` where applicable
- `competition`
- `season`
- `period`
- `minute_start`
- `minute_end`
- `phase`
- `data_source`
- `model_name`
- `model_version` or implementation name
- `sample_size`
- `is_synthetic`
- `limitations`

## Testing And Acceptance

Add tests for:

- artifact generation from synthetic fallback;
- artifact generation from a small canonical actions file;
- manifest creation;
- UI data loader;
- centralisation calculation;
- pass clustering on toy data;
- formation inference on toy lineup data;
- insight cards include context, sample size, and method;
- all existing tests still pass.

Acceptance commands:

```bash
pytest
python -m fas.cli demo --no-summary
python -m fas.cli product-build --no-download
```

If you add a UI command:

```bash
python -m fas.cli ui
```

or document the exact Streamlit command:

```bash
streamlit run src/fas/ui/app.py
```

The UI must launch using synthetic fallback artifacts if no real data exists.

At the end of the implementation, run the acceptance commands that are feasible
in the environment. Then provide a short "Run It" section in the final response
with the exact commands, for example:

```bash
python -m fas.cli product-build --no-download
python -m fas.cli ui
```

If the UI starts a server, include the local URL. If it uses Streamlit, include
the `streamlit run ...` command as a fallback. State whether the displayed data
is real local data or deterministic synthetic demo data.

## Documentation Updates

Update:

- `README.md`
- `docs/SPEC.md`
- `docs/DECISIONS.md`
- `football_analytics_phd_project.md`

Document:

- how to build artifacts;
- how to run the UI;
- what data is used;
- what is synthetic vs real;
- what each view answers;
- which optional dependencies unlock richer models;
- what remains approximate without tracking or 360 data.

## Definition Of Done

The work is complete when:

1. The repo can generate product artifacts locally without network access.
2. The repo can use real local StatsBomb-style data when present.
3. The UI opens to an actual analytics workspace.
4. Running the documented commands produces visible, populated charts and
   tables immediately.
5. Every visible chart is tied to exact match/team/player context.
6. Every insight states evidence, baseline, uncertainty or validation status,
   sample size, and limitations.
7. The existing mathematical modules remain importable and tested.
8. Heavy dependencies remain optional or are justified.
9. Documentation explains the product clearly and plainly.
10. The final response tells the user what to run and what data they are seeing.
11. `pytest` passes.

Build this as a coherent football intelligence system. The goal is not to prove
that one metric is magic. The goal is to let event data, geometry, network
structure, model uncertainty, and tactical context speak together.
