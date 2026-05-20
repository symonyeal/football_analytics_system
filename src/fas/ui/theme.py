"""Shared visual theme for the Streamlit workspace (professional dark UI)."""

from __future__ import annotations

CSS = """
<style>
:root {
  --fas-bg:#0d1117; --fas-panel:#161b22; --fas-border:#222b36;
  --fas-text:#e6edf3; --fas-muted:#8b97a6; --fas-accent:#3b82f6; --fas-gold:#f5b301;
}
.stApp { background: var(--fas-bg); }
section[data-testid="stSidebar"] { background:#0a0e14; border-right:1px solid var(--fas-border); }
.fas-hero {
  background: linear-gradient(120deg,#10243a 0%, #0d1117 60%);
  border:1px solid var(--fas-border); border-radius:14px;
  padding:18px 24px; margin-bottom:14px;
}
.fas-hero h1 { margin:0; font-size:26px; letter-spacing:.3px; color:var(--fas-text); }
.fas-hero p { margin:4px 0 0; color:var(--fas-muted); font-size:13px; }
.fas-pill {
  display:inline-block; padding:3px 11px; border-radius:999px; font-size:11.5px;
  font-weight:600; margin-right:6px;
}
.fas-pill.real { background:#0c2f1c; color:#5fe39b; border:1px solid #1d6b43; }
.fas-pill.synthetic { background:#3a2f12; color:#ffcf6b; border:1px solid #7a5e1f; }
.fas-pill.local { background:#13283f; color:#7cc0ff; border:1px solid #2b5d8f; }
div[data-testid="stMetric"] {
  background:var(--fas-panel); border:1px solid var(--fas-border);
  border-radius:12px; padding:12px 14px;
}
div[data-testid="stMetricLabel"] { color:var(--fas-muted); }
div[data-testid="stMetricValue"] { color:var(--fas-text); font-size:24px; }
h2, h3 { color:var(--fas-text); }
.fas-section {
  border-left:3px solid var(--fas-accent); padding-left:10px; margin:18px 0 6px;
  color:var(--fas-text); font-weight:700; font-size:18px;
}
.fas-card {
  background:var(--fas-panel); border:1px solid var(--fas-border);
  border-radius:12px; padding:14px 16px; margin:8px 0;
}
.fas-insight { border-left:3px solid var(--fas-accent); }
.fas-insight.exploratory { border-left-color:var(--fas-gold); }
.stDataFrame { border:1px solid var(--fas-border); border-radius:10px; }
.stTabs [data-baseweb="tab"] { font-size:14px; }
</style>
"""


def hero(title: str, subtitle: str, mode: str) -> str:
    cls = {"real": "real", "synthetic": "synthetic", "local": "local"}.get(mode, "local")
    label = {"real": "REAL DATA", "synthetic": "SYNTHETIC DEMO",
             "local": "LOCAL DATA"}.get(mode, mode.upper())
    return (f'<div class="fas-hero"><h1>⚽ {title}</h1>'
            f'<p><span class="fas-pill {cls}">{label}</span>{subtitle}</p></div>')
