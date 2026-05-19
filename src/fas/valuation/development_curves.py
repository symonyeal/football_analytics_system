"""Player development curves (Part 6.2, Layer 3).

Beta-shaped career arc: f(age) = f_peak * Beta_pdf_scaled(age; a_p, b_p), with
peak at age a_p/(a_p+b_p). We fit (a_p, b_p, f_peak) per player (or per position
group) by nonlinear least squares, then project current performance to peak.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import beta as beta_dist

AGE_MIN, AGE_MAX = 16.0, 38.0


def beta_career_curve(age, a_p, b_p, f_peak):
    """Scaled Beta career arc over the playing-age window [16, 38]."""
    u = np.clip((np.asarray(age, dtype=float) - AGE_MIN) / (AGE_MAX - AGE_MIN), 1e-6, 1 - 1e-6)
    pdf = beta_dist.pdf(u, a_p, b_p)
    peak_u = (a_p - 1) / (a_p + b_p - 2) if (a_p > 1 and b_p > 1) else 0.5
    peak_pdf = beta_dist.pdf(np.clip(peak_u, 1e-6, 1 - 1e-6), a_p, b_p)
    return f_peak * pdf / (peak_pdf + 1e-12)


def fit_curve(ages, values):
    """Fit (a_p, b_p, f_peak) by least squares; returns the parameter tuple."""
    p0 = [3.0, 3.0, float(np.max(values)) if len(values) else 1.0]
    try:
        popt, _ = curve_fit(
            beta_career_curve, ages, values, p0=p0,
            bounds=([1.01, 1.01, 0], [15, 15, np.inf]), maxfev=5000,
        )
        return tuple(popt)
    except Exception:
        return tuple(p0)


def project_to_peak(current_age, current_value, params):
    """Scale a current observation up to its modeled career peak (Layer 3)."""
    a_p, b_p, f_peak = params
    cur = beta_career_curve(current_age, a_p, b_p, f_peak)
    if cur <= 1e-9:
        return current_value
    return float(current_value * f_peak / cur)
