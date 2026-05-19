"""Scouting report generator (Part 6.3)."""

from __future__ import annotations

from dataclasses import dataclass


def _bar(pct: float, width: int = 10) -> str:
    filled = int(round(max(0.0, min(1.0, pct)) * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {int(round(pct*100))}th pctl"


@dataclass(slots=True)
class ScoutingReport:
    name: str
    club: str
    league: str
    age: int
    contract_expiry: str
    fair_value_eur: float
    market_value_eur: float
    pvs_attack: float
    pvs_defend: float
    xt_added_pctl: float
    centrality_pctl: float
    role_fit: float
    pair_synergy: float
    epv_added_90: float
    pagerank: float
    recommendation: str
    ceiling_eur: float

    @property
    def valuation_gap_pct(self) -> float:
        if self.market_value_eur <= 0:
            return 0.0
        return (self.fair_value_eur - self.market_value_eur) / self.market_value_eur * 100.0


def render_report(r: ScoutingReport) -> str:
    """Render a :class:`ScoutingReport` to the Part 6.3 text format."""
    gap = r.valuation_gap_pct
    sign = "+" if gap >= 0 else ""
    return f"""SCOUTING REPORT: {r.name}
Current Club / League: {r.club} / {r.league}
Age: {r.age} | Contract Expiry: {r.contract_expiry}

VALUATION:
  Fair Market Value:    EUR {r.fair_value_eur/1e6:.1f}m   (model predicted)
  Transfermarkt Value:  EUR {r.market_value_eur/1e6:.1f}m   (observed)
  Valuation Gap:        {sign}{gap:.0f}%   ({'undervalued' if gap > 0 else 'overvalued'})

PERFORMANCE PROFILE (league-adjusted percentiles):
  Attacking Contribution:  {_bar(r.pvs_attack)}
  Defensive Contribution:  {_bar(r.pvs_defend)}
  Build-up (xT added/90):  {_bar(r.xt_added_pctl)}
  Network centrality:      {_bar(r.centrality_pctl)}

TACTICAL FIT:
  Role compatibility:  {r.role_fit:.2f}
  Pairwise synergy with squad:  {r.pair_synergy:.2f}

KEY MATHEMATICAL HIGHLIGHTS:
  EPV added/90: {r.epv_added_90:.2f}
  PageRank in pass network: {r.pagerank:.2f}

RECOMMENDATION: {r.recommendation} at EUR {r.ceiling_eur/1e6:.1f}m ceiling
"""
