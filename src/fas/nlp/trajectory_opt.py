"""Player trajectory optimization via direct collocation (Part 3.3) — STUB.

Optimal control: state q(t)=(x,y), control u(t)=(vx,vy), dynamics q'=u.

    min_u  alpha * int ||u||^2 dt  -  beta * Phi(q(T))
    s.t.   q' = u, q(0)=q0, ||u|| <= v_max(t), q in pitch,
           ||q - q_j|| >= r_j  (collision avoidance),
    Phi(q(T)) = sum_z softmax(-||q(T)-z||^2/sigma^2) * xT(z).

Discretize (dt=0.1s, N=30), RK4 dynamics, solve with IPOPT (cyipopt).
Fatigue (Banister): v_max(t)=v_base*exp(-k_f*S), S' = -tau_s*S + ||u||^2.

Requires cyipopt (pip install fas[nlp]). scipy SLSQP can serve as a fallback
collocation solver for short horizons.
"""

from __future__ import annotations


def optimize_trajectory(*args, **kwargs):
    raise NotImplementedError(
        "trajectory_opt requires cyipopt (pip install fas[nlp]); see docstring "
        "for the full direct-collocation formulation."
    )
