"""fas — Unified Mathematical Football Analytics System.

A research-grade soccer analytics stack built on public event data
(StatsBomb open data, FBref, Transfermarkt). Modules:

    fas.data          canonical action schema + loaders (Part 0)
    fas.graph         pass-network graph theory (Part 1)
    fas.network_flow  pitch-zone flow, xT, max-flow build-up (Part 2)
    fas.nlp           EPV / pass-value / trajectory / set-piece NLP (Part 3)
    fas.milp          valuation + squad-selection MILP (Part 4)
    fas.boolean       tactical Boolean functions + formation lattice (Part 5)
    fas.valuation     cross-league normalization + scouting (Part 6)
    fas.evaluation    calibration, sensitivity, end-to-end tests (Part 7)
    fas.entities      canonical MatchObject / PlayerSeason / TeamSeason spine
    fas.foundations   v3 point-process and performance-measure foundations
    fas.performance   v3 player/team performance inference
    fas.headtohead    v3 matchup and predictive-distribution models
    fas.inference     v3 higher-order inference and insight extraction

See README.md for full mathematical formulations.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
