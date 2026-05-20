# Decision Log

This is the project memory for implementation choices. It is meant to be short,
plain, and useful later.

At the end of each work session, Codex should update this file with:

- the decision made
- why it was made
- the main files affected
- any follow-up that should not be forgotten

Newest entries go first.

## 2026-05-20 - Require Runnable Product Data In The One-Shot Prompt

Decision: strengthen the one-shot product prompt so the implementation must
materialize enough local or deterministic synthetic data for the UI to show
populated charts immediately, and must tell the user exactly what to run.

Why: the next builder should not leave the user with a design, skeleton, or
data-hunting exercise. The product pass should end with runnable commands and a
visible analytics workspace.

Files affected:

- `docs/ONE_SHOT_PRODUCT_PROMPT.md`
- `docs/DECISIONS.md`

Follow-up: when implemented, verify that `product-build --no-download` creates
non-empty artifacts and that the UI can launch from those artifacts.

## 2026-05-20 - Add a One-Shot Product Prompt

Decision: add a self-contained product build prompt for the next pass of the
football analytics system.

Why: the project now needs a single coherent instruction set that ties the
existing mathematical modules to real/local data, contextual charts, validated
insights, and a proper UI without losing the repo's local-data-first contract.

Files affected:

- `docs/ONE_SHOT_PRODUCT_PROMPT.md`
- `docs/DECISIONS.md`

Follow-up: when the prompt is used, update the product artifact contract and UI
instructions to match the implementation that lands.

## 2026-05-20 - Make `fas demo` Reflect the v3 Stack

Decision: extend `fas demo` so it first looks for a local canonical actions
file under `data/`, then falls back to synthetic data. The same path now runs
the v1 core modules and a compact v3 set: entity spine, intensity, RAPM, skill,
IRT, form, roles, scoring, possession MDP, pitch control, matchup models,
Hawkes, style drift, shape, causality, and insight checks.

Why: the project should show the current work when it is run, even before a
real public dataset is materialized locally.

Files affected:

- `src/fas/cli.py`
- `src/fas/examples/synthetic_pipeline.py`
- `tests/test_pipeline.py`

Follow-up: add a real `data/processed/actions.parquet` ingestion pass, then keep
the same demo path as the acceptance check.

## 2026-05-20 - Record the Next Phase Prompt

Decision: add a dedicated next-phase prompt and append the same phase direction
to the main project document.

Why: the next pass should be easy to resume: materialize a small public dataset,
run the existing local-data-first path, and persist summary artifacts.

Files affected:

- `docs/NEXT_PHASE_PROMPT.md`
- `football_analytics_phd_project.md`
- `README.md`
- `docs/SPEC.md`

Follow-up: update these documents again after the real-data spine is in place.

## 2026-05-20 - Add a Standing Decision Log

Decision: keep project decisions in this file and update it at the end of each
process.

Why: the project is broad enough that choices about scope, dependencies, and
architecture can get lost in chat history. A small log makes the work easier to
audit without reading every diff.

Files affected:

- `docs/DECISIONS.md`
- `README.md`

Follow-up: update this document before the final response for future work on
this repository.

## 2026-05-20 - Keep the Documentation Understated

Decision: rewrite the README and spec map in a quieter style.

Why: the previous wording sounded promotional and repetitive. The docs should
explain what exists, where it lives, and what is still optional or stubbed.

Files affected:

- `README.md`
- `docs/SPEC.md`

Follow-up: keep future documentation in the same tone: concrete, direct, and
light on claims.

## 2026-05-20 - Treat v3 as an Extension of the Entity Spine

Decision: route v3 outputs through `MatchObject`, `PlayerSeason`,
`TeamSeason`, and `Matchup` rather than creating a parallel data model.

Why: the existing plan already identified the entity spine as the integration
point. Reusing it lets performance, matchup, valuation, and UI work consume the
same records without glue code.

Files affected:

- `src/fas/entities.py`
- `src/fas/foundations/`
- `src/fas/performance/`
- `src/fas/headtohead/`
- `src/fas/inference/`

Follow-up: when adding new metrics, include an `enrich(...)` adapter unless
there is a clear reason not to.

## 2026-05-20 - Prefer Core Fallbacks for Research Modules

Decision: implement v3 models with NumPy, SciPy, pandas, networkx, and
scikit-learn fallbacks first. Keep heavier libraries as optional extras.

Why: the project is being run on a new Python version where some scientific
packages may lag. Core fallbacks keep tests and demos usable.

Files affected:

- `pyproject.toml`
- `src/fas/performance/`
- `src/fas/headtohead/`
- `src/fas/inference/`

Follow-up: if optional backends are added later, preserve the current public
interfaces.

## 2026-05-20 - Make the Coherence Layer Explicit but Small

Decision: add a small `foundations/coherence.py` module with a concise
functorial statement instead of weaving category-theory language through the
runtime code.

Why: the prompt asked for a coherence statement, but it should clarify the
system rather than make ordinary code harder to read.

Files affected:

- `src/fas/foundations/coherence.py`
- `src/fas/foundations/__init__.py`
- `tests/test_foundations_v3.py`

Follow-up: keep this module as metadata/documentation unless a real runtime use
appears.

## 2026-05-20 - Keep Tests Broad but Lightweight

Decision: add focused tests for each v3 package and keep them synthetic.

Why: the repo should verify interfaces and numerical sanity without depending
on network downloads or large datasets.

Files affected:

- `tests/test_foundations_v3.py`
- `tests/test_performance_v3.py`
- `tests/test_headtohead_v3.py`
- `tests/test_inference_v3.py`

Follow-up: when real-data DAG work starts, add separate integration tests that
can be skipped cleanly when data is unavailable.

## 2026-05-20 - Fix Paired-Comparison Outcome Parsing for Pandas Strings

Decision: use `pd.api.types.is_numeric_dtype(...)` in the Davidson paired
comparison parser instead of `np.issubdtype(...)`.

Why: pandas nullable string dtypes are not always interpretable as NumPy dtypes.
The old check failed under the current environment.

Files affected:

- `src/fas/headtohead/paired_comparison.py`

Follow-up: prefer pandas dtype helpers when handling pandas extension dtypes.

## 2026-05-20 - Product Layer: Real-Data-First Spine, Artifacts, and UI

Decision: add a `fas.product` package (data spine, deterministic synthetic
generator, six-layer artifact materializer, loader) and a `fas.ui` package
(Streamlit workspace + static HTML report). `fas product-build` defaults to
pulling REAL StatsBomb Open Data (PSG, Ligue 1 2022/23); `--no-download` forces
a deterministic synthetic fallback for offline/CI.

Why: the engine was complete but not runnable as a product. The product turns
discrete events into a coherent entity spine, persists artifacts under
`data/processed/`, and surfaces context-rich charts and FDR-controlled insights.
The user requirement was explicit: use real data wherever available.

Files affected:

- `src/fas/product/{synthetic,ingest,centralisation,formation,clustering,artifacts,build,loader}.py`
- `src/fas/ui/{charts,app,report}.py`
- `src/fas/cli.py` (`product-build`, `ui`, `report` subcommands)
- `tests/test_product.py`, `tests/conftest.py`

Follow-up: add FBref/Understat/ClubElo adapters for supplementary player and
fixture context; add 360 freeze-frame ingestion to unlock line-breaking and
pitch-control precision.

## 2026-05-20 - Fix StatsBomb Loader for statsbombpy Flattened Columns

Decision: make `events_to_actions` read `statsbombpy`'s flattened columns
(`pass_end_location`, `pass_outcome`, `shot_outcome`, ...) in addition to the
raw nested-JSON format, and harden `build_pass_network` / `build_zone_graph`
against empty inputs.

Why: `statsbombpy` flattens nested event objects into columns, so the original
nested-dict-only parser silently produced `x_end = NaN` for every pass — which
zeroed xT, emptied zone graphs, and broke real-data ingestion. Empty windows
(e.g. a half with no events) also crashed the graph builders.

Files affected:

- `src/fas/data/statsbomb.py`
- `src/fas/graph/pass_network.py`
- `src/fas/network_flow/max_flow_buildup.py`

Follow-up: add a small fixture-based loader test that exercises both nested and
flattened event shapes.

## 2026-05-20 - Broaden Real Ingestion to All Leagues / Eras + Polished UI

Decision: make `product-build` default to a broad real-data sample spanning
*every* StatsBomb competition, season, and team (no team filter), round-robined
across competition-seasons for league/era breadth and capped by
`--sb-max-matches` (0 = all). Matchups are computed only within a
competition+season cohort to avoid an O(T^2) cross product and meaningless
cross-league pairings. The Streamlit UI gained a professional dark theme and
global Competition / Season scope filters.

Why: the product should not be fixated on one club. Users want all teams, all
leagues, and all available timeframes — historical (e.g. 1962 World Cup, Ajax's
European Cup three-peat) through current (2024 Copa America, 2023/24 Bundesliga).

Files affected:

- `src/fas/product/ingest.py` (`_select_matches` round-robin, `load_statsbomb`)
- `src/fas/product/artifacts.py` (cohort-bounded matchups)
- `src/fas/product/build.py`, `src/fas/cli.py` (scope flags)
- `src/fas/ui/{app.py,theme.py,charts.py}` (theme + scope filters)

Follow-up: add disk caching of downloaded events to speed repeat builds; add a
cross-league normalization toggle when leagues are mixed in one shortlist.
