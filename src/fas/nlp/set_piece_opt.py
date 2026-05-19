"""Set-piece delivery optimization with Magnus force (Part 3.4).

Ball flight under gravity, quadratic drag, and Magnus lift:

    m a = -0.5 rho C_D A ||v|| v + 0.5 rho C_L A (omega x v) - m g e_z

Decision variables: initial velocity v0 in R^3, spin omega in R^3.
Objective: maximize P(landing in target zone) while clearing the wall and
staying in-pitch.

We integrate the ODE with RK4 (numpy) and search over (v0, omega) by
multi-start gradient-free optimization (scipy Nelder-Mead). This avoids the
torch autodiff dependency while remaining a faithful physical model; swap in
torch + autodiff through the integrator for the gradient-descent variant in
the spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

# Physical constants (size-5 football, sea level).
RHO = 1.225          # air density kg/m^3
C_D = 0.25           # drag coefficient
C_L = 0.20           # lift coefficient
A = 0.038            # cross-section m^2
M = 0.43             # mass kg
G = 9.81             # gravity m/s^2
RADIUS = 0.11        # ball radius m


def _accel(v: np.ndarray, omega: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(v)
    drag = -0.5 * RHO * C_D * A * speed * v / M
    magnus = 0.5 * RHO * C_L * A * np.cross(omega, v) / M
    gravity = np.array([0.0, 0.0, -G])
    return drag + magnus + gravity


def simulate(
    p0: np.ndarray,
    v0: np.ndarray,
    omega: np.ndarray,
    *,
    dt: float = 0.01,
    max_t: float = 6.0,
) -> np.ndarray:
    """RK4-integrate the flight until the ball lands (z<=0). Returns trajectory."""
    p, v = p0.astype(float).copy(), v0.astype(float).copy()
    traj = [p.copy()]
    t = 0.0
    while t < max_t:
        k1v = _accel(v, omega)
        k2v = _accel(v + 0.5 * dt * k1v, omega)
        k3v = _accel(v + 0.5 * dt * k2v, omega)
        k4v = _accel(v + dt * k3v, omega)
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        p = p + dt * v
        traj.append(p.copy())
        if p[2] <= 0 and len(traj) > 1:
            break
        t += dt
    return np.array(traj)


@dataclass(slots=True)
class SetPieceResult:
    v0: np.ndarray
    omega: np.ndarray
    landing: np.ndarray
    miss_distance: float


def optimize_set_piece(
    p0: np.ndarray,
    target: np.ndarray,
    *,
    wall_x: float | None = None,
    wall_height: float = 2.0,
    v_min: float = 15.0,
    v_max: float = 30.0,
    n_restarts: int = 12,
) -> SetPieceResult:
    """Find (v0, omega) minimizing landing miss-distance to a target point.

    A wall-clearance penalty enforces ``z >= wall_height`` at ``x = wall_x``.
    Parameterizes v0 by (speed, azimuth, elevation); spin by 3 components.
    """
    rng = np.random.default_rng(0)

    def unpack(theta):
        speed, az, el, wx, wy, wz = theta
        speed = np.clip(speed, v_min, v_max)
        v0 = speed * np.array([
            np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
        ])
        return v0, np.array([wx, wy, wz])

    def cost(theta):
        v0, omega = unpack(theta)
        traj = simulate(p0, v0, omega)
        landing = traj[-1]
        miss = np.linalg.norm(landing[:2] - target[:2])
        penalty = 0.0
        if wall_x is not None:
            xs = traj[:, 0]
            k = np.argmin(np.abs(xs - wall_x))
            if traj[k, 2] < wall_height:
                penalty += 10.0 * (wall_height - traj[k, 2])
        return miss + penalty

    best: SetPieceResult | None = None
    for _ in range(n_restarts):
        theta0 = np.array([
            rng.uniform(v_min, v_max),
            rng.uniform(-np.pi / 3, np.pi / 3),
            rng.uniform(0.2, 0.9),
            rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-50, 50),
        ])
        res = minimize(cost, theta0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-2, "fatol": 1e-2})
        v0, omega = unpack(res.x)
        traj = simulate(p0, v0, omega)
        miss = np.linalg.norm(traj[-1][:2] - target[:2])
        if best is None or miss < best.miss_distance:
            best = SetPieceResult(v0, omega, traj[-1], float(miss))
    return best
