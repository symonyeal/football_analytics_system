# Next Phase Prompt

Use this prompt for the next implementation pass.

## Goal

Move from a runnable local-data-first demo to a small real-data spine. The
command-line path should stay simple:

```bash
fas demo
fas demo --data data/processed/actions.parquet
```

When a local canonical actions file exists, the project should use it. When it
does not, the synthetic fallback should still run and make that clear.

## Work To Do

1. Add a public-data ingestion pass for one small competition or match set.
   Prefer StatsBomb Open Data if available in the environment. If network access
   is not available, document the expected local file path and keep the fallback.
2. Materialize canonical actions into `data/processed/actions.parquet`.
3. Build `MatchObject`, `PlayerSeason`, `TeamSeason`, and `Matchup` records from
   those actions.
4. Run the existing v1 modules and v3 modules from the same path:
   pass networks, xT, flow, PVS, RAPM, skill, IRT, form, roles, scoring,
   possession MDP, pitch control, matchup models, TDA/RMT/OT/MMD/causality, and
   FDR-controlled insights.
5. Persist a small summary artifact under `data/processed/` for notebooks or a
   first UI layer to read.
6. Keep every new dependency optional unless it is already in the core install.
7. Update `README.md`, `docs/SPEC.md`, `football_analytics_phd_project.md`, and
   `docs/DECISIONS.md` at the end.

## Acceptance Check

`fas demo` should print the data source, old core outputs, v3 outputs, and squad
MILP status without requiring a download. If real local data exists, the printed
source should name that file. If not, it should say `synthetic fallback`.

Run:

```bash
pytest
python -m fas.cli demo --no-summary
```

Both should pass.
