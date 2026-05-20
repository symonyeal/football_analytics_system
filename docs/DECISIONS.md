# Decision Log

This is the project memory for implementation choices. It is meant to be short,
plain, and useful later.

At the end of each work session, Codex should update this file with:

- the decision made
- why it was made
- the main files affected
- any follow-up that should not be forgotten

Newest entries go first.

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
