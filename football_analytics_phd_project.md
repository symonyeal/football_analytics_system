# UNIFIED MATHEMATICAL FOOTBALL ANALYTICS SYSTEM
## End-to-End Project Specification for a PhD Mathematician
### Version 1.0 — Full Technical Prompt

---

> **Scope**: This document specifies a complete, research-grade football analytics system that integrates Graph Theory, Mixed-Integer Linear Programming (MILP), Nonlinear Programming (NLP), Network Flow optimization, Boolean/Dual functions, and statistical learning — all grounded in publicly available data. The system culminates in a cross-league player valuation engine and optimal squad selection framework. Every module is written to PhD mathematical standards with full problem formulations.

---

## PART 0 — DATA INFRASTRUCTURE

### 0.1 Primary Data Sources

```
StatsBomb Open Data      https://github.com/statsbomb/open-data
  └── /data/events/        JSON event streams (~3.5M events across competitions)
  └── /data/lineups/       Player lineups + positions per match
  └── /data/matches/       Match metadata (score, competition, season)
  └── /data/three-sixty/   360° freeze-frame positional data (select competitions)

Roboflow Sports          https://github.com/roboflow/sports
  └── Soccer pitch keypoint detection model (YOLO-based)
  └── Player tracking + jersey number recognition
  └── Ball detection from video frames
  └── Homography estimation for pitch coordinate mapping

Edd Webster Aggregation  https://github.com/eddwebster/football_analytics
  └── Transfermarkt player valuations (scraped/linked)
  └── FBref statistical summaries (per-90 metrics)
  └── Wyscout / SPADL-format action data
  └── FIFA player ratings (cross-league comparison baseline)

Supplementary Public Sources:
  └── understat.com        xG/xGA per shot (scrape via understatapi)
  └── football-data.co.uk  Historical match results + betting odds
  └── sofifa.com           Physical attributes, position flexibility scores
```
https://github.com/matiasmascioto/awesome-soccer-analytics
https://github.com/jalapic/engsoccerdata
https://github.com/jokecamp/FootballData
https://github.com/DimaKudosh/pydfs-lineup-optimizer
https://github.com/felipeall/transfermarkt-api


### 0.2 Data Schema Unification

Design a canonical **action schema** `A = (t, player_id, team_id, action_type, x_start, y_start, x_end, y_end, outcome, freeze_frame)` where:

- `(x, y) ∈ [0,120] × [0,80]` — standardized StatsBomb pitch coordinates
- `action_type ∈ {pass, carry, shot, dribble, pressure, tackle, interception, clearance, ...}`
- `freeze_frame`: sparse matrix `F ∈ ℝ^{22×4}` of (x, y, vx, vy) for each visible player at action timestamp
- `outcome ∈ {0,1}` — binary success indicator

Build a PostgreSQL schema with:
```sql
CREATE TABLE actions (
    action_id     BIGSERIAL PRIMARY KEY,
    match_id      INT REFERENCES matches(match_id),
    period        SMALLINT,
    timestamp_ms  INT,
    player_id     INT REFERENCES players(player_id),
    team_id       INT REFERENCES teams(team_id),
    action_type   VARCHAR(32),
    x_start       FLOAT,  y_start FLOAT,
    x_end         FLOAT,  y_end   FLOAT,
    outcome       BOOLEAN,
    freeze_json   JSONB
);
```

Cross-league normalization: map all player_ids from FBref / Transfermarkt / StatsBomb to a unified `player_uid` using fuzzy name + DOB matching (Jaro-Winkler distance threshold 0.92 + nationality filter).

---

## PART 1 — GRAPH THEORY MODULE

### 1.1 Pass Network Construction

For each match `m` and time window `[t_1, t_2]`, construct a **weighted directed multigraph**:

```
G_m = (V, E, w)
V = {v_i : i is a player who made or received ≥1 pass in window}
E = {(v_i, v_j, k) : player i completed pass k to player j}
w : E → ℝ₊,   w(i,j) = count of completed passes from i to j
```

Aggregate to a simple weighted digraph by `W[i,j] = Σ_k w(i,j,k)`.
Normalize: `P[i,j] = W[i,j] / Σ_k W[i,k]`  (row-stochastic, i.e., a Markov transition matrix).

**Compute for each match and each 15-minute phase:**

**Centrality measures (implement all four):**

1. **Degree centrality**: `C_D(v_i) = (d_in(v_i) + d_out(v_i)) / (2(n-1))`

2. **Betweenness centrality**: 
   `C_B(v_i) = Σ_{s≠i≠t} σ_{st}(v_i) / σ_{st}`
   where `σ_{st}` = number of shortest paths from s to t, `σ_{st}(v_i)` = those passing through i.
   Use Brandes algorithm O(VE) for computation.

3. **Closeness centrality** (on weighted graph, using reciprocal-weight distances):
   `C_C(v_i) = (n-1) / Σ_{j≠i} d(v_i, v_j)`
   where `d(v_i, v_j)` = shortest path in graph with edge weight `1/W[i,j]`.

4. **PageRank** (stationary distribution of the Markov matrix P):
   `π = πP`,   solved as left eigenvector for eigenvalue 1.
   Interpretation: `π(v_i)` = probability the ball is at player i in steady state → direct measure of ball-touch dominance.

**Structural metrics:**

- **Clustering coefficient**: `C(v_i) = (# triangles through v_i) / (# connected triples at v_i)`
  Team average `C̄ = (1/n) Σ C(v_i)` measures passing circularity.

- **Edge connectivity** `λ(G)`: minimum edge cut separating any two nodes.
  Solve via max-flow (Ford-Fulkerson on unit-capacity copy). Directly quantifies how many passing lanes must be blocked to disconnect the team's circulation.

- **Network entropy**: `H(G) = -Σ_{i,j} P[i,j] log P[i,j]`
  High entropy → unpredictable, distributed passing. Low entropy → predictable, centralized.

### 1.2 Temporal Pass Network Analysis

Model match dynamics as a **sequence of snapshots** `{G_m^{(1)}, G_m^{(2)}, ..., G_m^{(K)}}` at K=6 phases of 15 minutes each.

Define **network velocity**: `ΔG^{(k)} = ||A^{(k)} - A^{(k-1)}||_F / ||A^{(k-1)}||_F`
(Frobenius norm of adjacency matrix change, normalized). High `ΔG` indicates tactical shift.

Define **formation fingerprint** as the vector of sorted player positions (by x-coordinate mean):
`φ_m = sort({x̄_i : i ∈ V})` — this is a 10-dimensional vector (excluding GK) for clustering formations.

Use k-means (k=8, initialized with known formation archetypes: 4-4-2, 4-3-3, 4-2-3-1, 3-5-2, etc.) to cluster `{φ_m}` across all matches. Identify formation transitions mid-match as a change in cluster assignment between consecutive phases.

### 1.3 Opponent Vulnerability Mapping via Min-Cut

For each opponent team `T_opp`, compute their **passing network G_opp** from the last 5 matches.

Formulate the **min-cut problem** to find the optimal pressing locations:

```
Given G_opp = (V, E, W):
  Source s = goalkeeper node
  Sink   t = {attacking third nodes}
  Capacity c(i,j) = W[i,j] (pass frequency as flow capacity)

Solve: min-cut = min_{S⊂V, s∈S, t∉S} Σ_{i∈S, j∉S} c(i,j)
```

The **min-cut edges** reveal exactly which player connections to disrupt through targeted pressing. Implemented via Push-Relabel algorithm, O(V²√E).

Output: **Pressing assignment** — a ranked list of (pressing player, target player, priority score) triples to present to the coaching staff.

### 1.4 Graph Neural Network for Collective Valuation

Build a **Graph Attention Network (GAT)** over each possession sequence:

- Node features `h_i ∈ ℝ^d`: player coordinates, velocity, role encoding, in-possession flag
- Edge features `e_{ij} ∈ ℝ^p`: Euclidean distance, angular separation, pass frequency in match
- Attention: `α_{ij} = softmax_j(LeakyReLU(a^T [Wh_i ∥ Wh_j ∥ e_{ij}]))`
- Message passing: `h_i' = σ(Σ_j α_{ij} W h_j)`

**Target**: predict `ΔEPV` (change in Expected Possession Value) after each action.
Loss: `L = Σ_actions (ΔEPV_predicted - ΔEPV_true)² + λ||θ||²`

This produces a **player graph contribution score** — each player's mean `Σ α_{ij}` attention weight received, aggregated over a season, is their collective tactical importance beyond event-level statistics.

---

## PART 2 — NETWORK FLOW MODULE

### 2.1 Pitch Zone Flow Model

Partition the pitch into a **directed zone graph** `Z = (U, F)`:

```
Zones U = {u_1, ..., u_18}  (6 vertical bands × 3 horizontal thirds)
Directed edges F: (u_i → u_j) if a ball progression from zone i to j occurred ≥ 5× in dataset
Flow capacity cap(u_i, u_j) = # successful progressive actions (passes/carries) from i to j
Flow value val(u_i, u_j) = mean xT(x_end, y_end) − xT(x_start, y_start) per action
```

**Expected Threat (xT) surface** — fit a 16×12 grid model:

`xT(x,y) = P(shoot|x,y) · P(goal|shoot,x,y) + P(move|x,y) · Σ_{(x',y')} P(x',y'|move,x,y) · xT(x',y')`

Solve via value iteration:
`xT^{(k+1)}(x,y) = P_shot(x,y) · G(x,y) + P_move(x,y) · Σ_{x',y'} T(x',y'|x,y) · xT^{(k)}(x',y')`

until `||xT^{(k+1)} - xT^{(k)}||_∞ < ε = 10^{-6}`.

### 2.2 Max-Flow Build-Up Analysis

For each team in each match, compute **offensive max-flow** from defensive zones to attacking zones:

```
min-cost max-flow:
  Source: defensive third zones (u_1,...,u_6) with supply = # possessions originating there
  Sink:   attacking third zones (u_13,...,u_18)
  Cost:   −val(u_i, u_j)  (negative value = reward, so min-cost = max-reward)
  Capacity: cap(u_i, u_j) as above
```

Solve with **Successive Shortest Paths (SSP) algorithm**.

Interpretation: The maximum flow value is the team's **theoretical build-up potency** per possession. The flow decomposition reveals the most productive corridors (left flank, central channel, right flank) used in ball progression.

**Cross-match comparison**: Build a 30×30 matrix `Φ` where `Φ[m1, m2]` = Wasserstein distance between flow distributions of two matches. Cluster via spectral clustering on `Φ` to identify the team's tactical variants across a season.

### 2.3 Defensive Flow Suppression

Model the **defending team's objective** as minimizing the opponent's max-flow:

For a given opponent passing network `G_opp` and our team's available pressing energy budget `B`:

```
Variables:  r_{ij} ∈ [0,1]  — fraction of edge (i,j) capacity we suppress via pressing
Objective:  min  MaxFlow(G_opp with capacities cap(i,j)·(1 - r_{ij}))
Subject to: Σ_{ij} cost(i,j) · r_{ij} ≤ B     (pressing energy budget)
            r_{ij} ∈ [0,1]  ∀(i,j) ∈ E_opp
```

This is a **bilevel optimization problem** (upper: choose r; lower: opponent solves max-flow). Reformulate using LP duality on the inner max-flow:

Inner max-flow LP dual gives: `MaxFlow = min Σ_{ij} cap(i,j)(1-r_{ij}) · y_{ij}`
where `y_{ij}` are the dual cut indicator variables. Substituting, the bilevel program becomes a single-level bilinear program, solvable by McCormick envelope relaxation or successive linearization.

---

## PART 3 — NONLINEAR PROGRAMMING MODULE

### 3.1 Expected Possession Value (EPV) Surface Model

Model EPV as a function of the full game state:

```
EPV(s) = P(team scores next | state s) − P(opponent scores next | state s)
```

where state `s = (x_ball, y_ball, x_1,...,x_22, y_1,...,y_22, t, scoreline)`.

**Architecture**: U-Net Convolutional Neural Network operating on a spatial grid representation:

- Input: `I ∈ ℝ^{C×H×W}` where H=80, W=120, C=channels:
  - C1: attacking team player density (kernel-smoothed Gaussian heatmap)
  - C2: defending team player density  
  - C3: ball position indicator
  - C4: player velocity vectors (2 channels per team → 4 total)
  - C5: time remaining, score differential (broadcast as constant channels)
- U-Net encoder: 4 downsampling blocks (Conv → BatchNorm → ReLU → MaxPool)
- U-Net decoder: 4 upsampling blocks with skip connections
- Output head: scalar EPV via global average pooling → FC → sigmoid (scaled to [−1,1])

**Training**:
- Labels: terminal match outcome (+1 = team scored, −1 = conceded) discounted back through possession: `EPV_label(s_t) = γ^{T-t} · outcome`, where T = next goal/half-end, γ = 0.99
- Loss: `L(θ) = Σ_t (EPV_θ(s_t) − EPV_label(s_t))² + λ_1||θ||² + λ_2 TV(EPV_θ)`
  where `TV` = total variation regularizer enforcing spatial smoothness
- Calibration: Expected Calibration Error (ECE) on held-out World Cup data ≤ 0.02

### 3.2 Pass Value NLP: Reward-Risk Decomposition

For a ball-carrier at position `(x_0, y_0)` considering a pass to target zone `(x_T, y_T)`:

**Pass reward** (expected EPV gain if completed):
`R(x_T, y_T) = EPV(s | ball at (x_T, y_T)) − EPV(s_0)`

**Pass risk** (expected EPV loss if intercepted):
`K(x_T, y_T) = P(intercept | x_0, y_0, x_T, y_T) · (EPV(s_0) − EPV(s | opponent has ball at interception point))`

**Pass value**: `V(x_T, y_T) = R(x_T, y_T) − K(x_T, y_T)`

To find the **optimal pass target**, solve:
```
max_{(x_T, y_T) ∈ Ω}  V(x_T, y_T)
  subject to:
    ||(x_T, y_T) − (x_0, y_0)||₂ ≤ d_max          (max pass distance)
    P(complete | x_0,y_0,x_T,y_T) ≥ p_min           (minimum completion probability)
    (x_T, y_T) ∉ Ω_blocked                          (blocked by defenders, modeled as exclusion zones)
```

`Ω_blocked` is a union of ellipsoidal defender zones (non-convex). Solve with:
- Sequential Quadratic Programming (SQP) for smooth objective
- Multiple restarts from a grid of initial points
- Final global search via CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

### 3.3 Player Trajectory Optimization

Model each player's optimal movement as a **continuous-time optimal control problem**:

**State**: `q(t) = (x(t), y(t))` — player position on pitch
**Control**: `u(t) = (v_x(t), v_y(t))` — velocity vector
**Dynamics**: `q̇(t) = u(t)`

**Objective**: Move from current position `q(0) = q_0` to a target coverage zone `Q_target` while:
1. Minimizing energy expenditure: `∫₀ᵀ ||u(t)||² dt`
2. Maximizing pitch control coverage at time T: `∫_{pitch} P(q(T) covers z) dz`

**Formulation**:
```
min_{u(·)}  α · ∫₀ᵀ ||u(t)||² dt  −  β · Φ(q(T))
  subject to:
    q̇(t) = u(t),               q(0) = q_0
    ||u(t)||₂ ≤ v_max(t)       (speed limit — function of fatigue model)
    q(t) ∈ [0,120]×[0,80]      (pitch boundary)
    ||q(t) − q_j(t)||₂ ≥ r_j  ∀j ∈ teammates  (collision avoidance)
```

where `Φ(q(T)) = Σ_z softmax(-||q(T) - z||² / σ²) · xT(z)` — differentiable pitch control metric.

Discretize with time step Δt = 0.1s, N = 30 steps (3-second horizon).
Solve with **Direct Collocation**: parameterize u(t) as piecewise-constant, apply Runge-Kutta-4 for dynamics, then pass to IPOPT solver.

**Fatigue model** (Banister impulse-response):
```
v_max(t) = v_base · exp(−k_f · S(t))
Ṡ(t)     = −τ_s · S(t) + ||u(t)||²     (stamina depletion)
```
Parameters `(k_f, τ_s)` estimated per-player from GPS tracking data (or FBref sprint/distance stats as proxy).

### 3.4 Set Piece Delivery Optimization

For a corner or free kick delivery, model the ball trajectory as a **projectile with Magnus force**:

```
m·ẍ = −½ρC_D·A·||ẋ||·ẋ + ½ρC_L·A·(ω × ẋ)
```

where ρ = air density, C_D = drag coefficient, C_L = lift coefficient, A = cross-section, ω = spin vector.

**Decision variables**:
- `v_0 ∈ ℝ³` — initial velocity (speed, azimuth angle, elevation angle)
- `ω ∈ ℝ³` — spin vector

**Objective**: Maximize probability that ball lands in a target delivery zone `Z_target ⊂ ℝ³`, avoiding the defender wall, while conditioning on target player expected arrival position from the trajectory optimization above (Section 3.3).

```
max_{v_0, ω}  P(ball(T_land) ∈ Z_target) · P(attacker arrives before defender | v_0, ω)
  subject to:
    ||v_0||₂ ∈ [v_min, v_max]              (physical kicking range)
    ball clears the wall: z(t_wall) ≥ h_wall  (explicit wall clearance constraint)
    ball stays in pitch: (x(t), y(t)) ∈ [0,120]×[0,80]  ∀t ∈ [0, T_land]
```

Solve with **multi-start gradient descent** using automatic differentiation (PyTorch) through the ODE integrator.

---

## PART 4 — MILP MODULE: SQUAD SELECTION & VALUATION

### 4.1 Player Valuation: Composite Mathematical Score

Before optimization, compute a **universal Player Value Score (PVS)** for each player `i` across leagues.

**Step 1 — Per-90 Feature Vector** (sourced from FBref + StatsBomb):

```
f_i = [xG_90, xA_90, npxG_90, progressive_passes_90, progressive_carries_90,
       pressures_90, tackles_won_90, interceptions_90, aerial_won_pct,
       passing_completion_pct, live_ball_reception_90, shot_creation_90,
       pass_EPV_added_90, carry_EPV_added_90, defensive_EPV_reduced_90,
       betweenness_centrality_mean, pagerank_mean, graph_contribution_GAT_mean]
```

The last 3 features come from your Graph Theory module (Part 1).

**Step 2 — League Difficulty Adjustment**:

Fit a **Bradley-Terry model** across leagues using European competition results:
```
P(team i beats team j) = exp(β_i) / (exp(β_i) + exp(β_j))
```
MLE of `β` via logistic regression on all European match outcomes. 
League strength factor: `λ_L = (1/|teams_L|) Σ_{i ∈ L} exp(β_i)`

Adjust all per-90 metrics: `f_i_adj = f_i · (λ_L / λ_ref)^α`
where `λ_ref` = Premier League strength (reference), `α ∈ (0,1)` is a shrinkage exponent fit by cross-validation.

**Step 3 — Dimensionality Reduction via Robust PCA**:

Solve the **Robust PCA** decomposition: `F = L + S`
where `F = [f_1_adj, ..., f_n_adj]^T` (player × feature matrix),
`L` = low-rank signal, `S` = sparse outlier matrix.

Optimization:
```
min_{L,S}  ||L||_* + λ||S||₁
  subject to  L + S = F
```
(nuclear norm + L1 regularization, solved by ADMM)

Retain top-K principal components of `L` (Kaiser criterion: eigenvalue > 1).
Projected player vector: `z_i = U^T f_i_adj ∈ ℝ^K`.

**Step 4 — Positional Peer Comparison**:

For each position group `P ∈ {GK, CB, FB, CM, AM, W, ST}`:
```
PVS_i = Φ((z_i − μ_P) / σ_P)
```
where `Φ` = standard normal CDF, `μ_P, σ_P` = mean and std of `z` within position peer group.
Result: `PVS_i ∈ (0,1)` — percentile rank within positional peer group, league-adjusted.

**Step 5 — Market Value Calibration**:

Fit a **nonlinear regression** against Transfermarkt valuations:
```
log(MarketValue_i) = θ_0 + θ_1·PVS_i + θ_2·Age_i + θ_3·Age_i² + θ_4·PVS_i·Age_i + ε_i
```
Estimated via WLS (weighted least squares, weights = data recency). 
Use this as a **fair value model**: players with `log(MarketValue) << predicted` are undervalued targets.

### 4.2 Squad Selection MILP

**Decision variables**:

```
x_i ∈ {0,1}          — player i is in the squad (i = 1,...,N candidates)
y_{ip} ∈ {0,1}        — player i is assigned to role p  (p ∈ {GK,LCB,RCB,LB,RB,CDM,CM,CAM,LW,RW,ST})
z_i ∈ {0,1}          — player i starts (subset of selected)
s_i ∈ {0,1}          — player i is on the bench (x_i = z_i + s_i)
```

**Objective** (maximize squad quality):

```
max  Σ_i Σ_p w_p · PVS_i · y_{ip}  +  γ · NetworkBonus(x)
```

where:
- `w_p` = positional importance weights (estimated from league-average goal contribution by position)
- `NetworkBonus(x) = Σ_{i,j ∈ selected} compat_{ij}` = sum of pairwise style-compatibility scores (cosine similarity of `f_i_adj` and `f_j_adj` for positionally adjacent roles) — rewards cohesive, compatible squads

**Constraints**:

```
(C1) Positional coverage:
     Σ_i y_{i,GK} = 1,   Σ_i y_{i,ST} ≥ 1,   Σ_i y_{i,CB} ≥ 2,  ...  (one per role)

(C2) Starting XI:
     Σ_i z_i = 11

(C3) Squad size:
     Σ_i x_i = S   (S ∈ {18,23,25} as specified)

(C4) Role-selection link:
     Σ_p y_{ip} ≤ x_i   ∀i         (can only assign role if selected)
     Σ_p y_{ip} ≤ z_i   ∀i         (starting role implies starting)

(C5) Budget constraint:
     Σ_i x_i · Wage_i ≤ W_cap      (wage bill cap)
     Σ_i x_i · FairValue_i ≤ T_budget  (transfer budget)

(C6) Age distribution (squad building):
     Σ_i x_i · 𝟙[Age_i ≤ 23] ≥ A_min    (minimum young players)
     Σ_i x_i · 𝟙[Age_i ≥ 30] ≤ A_max    (maximum aging players)

(C7) League quota (e.g., homegrown):
     Σ_i x_i · 𝟙[player i is homegrown] ≥ H_min

(C8) Formation compatibility:
     Σ_i y_{i,CB} = n_CB(f),   Σ_i y_{i,CM} = n_CM(f),  ...
     where n_p(f) is the count of role p in formation f ∈ {4-3-3, 4-2-3-1, 3-5-2}
     Formation choice: introduce binary variable δ_f ∈ {0,1}, Σ_f δ_f = 1
     Then: n_p = Σ_f δ_f · n_p(f)  [linearize with big-M]

(C9) Synergy constraints (player pairs):
     y_{i,LW} + y_{j,ST} ≤ 1 + compat_flag_{ij}  — can block incompatible pairings

(C10) Positional flexibility:
     Each player i has a feasibility set Roles_i ⊆ {all roles}
     y_{ip} = 0 ∀p ∉ Roles_i
     (Roles_i estimated from StatsBomb/FBref position data + cosine similarity to role archetype)

(C11) NetworkBonus linearization:
     NetworkBonus(x) = Σ_{i<j} compat_{ij} · q_{ij}
     q_{ij} ≤ x_i,   q_{ij} ≤ x_j,   q_{ij} ≥ x_i + x_j − 1,   q_{ij} ∈ {0,1}
     (standard linearization of x_i · x_j)
```

**Solver**: PuLP with CBC backend (open-source), or Gurobi/CPLEX academic license.
Report: optimality gap, solve time, sensitivity analysis on budget and age constraints.

### 4.3 Robust MILP under Uncertainty

Player performance is uncertain. Introduce **scenario-based robustness**:

Let `ξ^(k)` be scenario k (k = 1,...,K), with `PVS_i^(k)` drawn from a bootstrap distribution over last 3 seasons.

**Robust objective** (minimax regret):
```
min_x  max_k  [OPT(ξ^(k)) − Objective(x, ξ^(k))]
```

Reformulate as: introduce `r_k ≥ 0` (regret in scenario k):
```
min_{x,r}  max_k r_k   (equivalently: min θ  s.t. r_k ≤ θ ∀k)
  s.t.  r_k ≥ OPT(ξ^(k)) − Σ_i Σ_p w_p · PVS_i^(k) · y_{ip} − γ·NetworkBonus(x)  ∀k
        all original MILP constraints
```

Solve with **Benders decomposition**: master problem controls (x, y, z), K sub-problems evaluate regret.

### 4.4 In-Match Substitution MILP

At time T in a match with current scoreline and estimated remaining EPV differential:

**Decision**: which player to substitute off, which substitute to bring on, and at which minute.

```
Variables:
  δ_{out,i,t} ∈ {0,1}  — player i subbed off at minute t
  δ_{in,j,t} ∈ {0,1}   — substitute j brought on at minute t
  (t ∈ {T, T+5, ..., 85})

Objective:
  max  Σ_{t,i,j} P(EPV_improvement | subbing j for i at t, current state) · δ_{in,j,t}

Constraints:
  Σ_{t,i} δ_{out,i,t} ≤ 3 − subs_used          (remaining substitutions)
  Σ_t δ_{out,i,t} ≤ 1 ∀i                        (each player at most once off)
  Σ_t δ_{in,j,t} ≤ 1 ∀j                         (each substitute at most once on)
  δ_{in,j,t} ≤ δ_{out,i,t}  for compatible role pairs (j replaces someone in role of i)
  Σ_{t'≤t} Σ_j δ_{in,j,t'} = Σ_{t'≤t} Σ_i δ_{out,i,t'}  (balance of subs)
```

`P(EPV_improvement | ...)` estimated from historical substitution outcomes using a trained gradient boosted model (LightGBM) conditioned on: minute, scoreline, substitute's fatigue, role mismatch.

---

## PART 5 — BOOLEAN & DUAL FUNCTIONS MODULE

### 5.1 Tactical Pattern Recognition via Boolean Functions

Represent each **game state** as a Boolean vector `b ∈ {0,1}^d` with features:

```
b_1 = 𝟙[team in high press]
b_2 = 𝟙[opponent GK has ball]
b_3 = 𝟙[wingers positioned wide (x < 15 or x > 105)]
b_4 = 𝟙[opponent defensive line high (mean CB y > 50)]
b_5 = 𝟙[ball in opponent half]
b_6 = 𝟙[numerical advantage in central zone]
...  (extend to d=20 binary features)
```

**Learn a Boolean function** `f: {0,1}^d → {0,1}` that predicts `P(turnover within 3 actions) ≥ 0.4`:

Use **Decision List learning** (Rivest 1987): 
```
f(b) = if b_3 ∧ b_2 → 1
        elif b_1 ∧ b_4 ∧ b_6 → 1
        elif ¬b_5 ∧ b_2 → 0
        else → 0
```

Enumerate decision lists by RIPPER (Repeated Incremental Pruning to Produce Error Reduction) on StatsBomb pressure data.

**Dual function analysis**: For a Boolean function `f` in DNF (Disjunctive Normal Form),
the **dual function** is: `f^d(b) = ¬f(¬b_1, ..., ¬b_d)`

If `f` represents "pressing trap triggered", then `f^d` represents the **complementary condition** that guarantees the pressing trap *cannot* be triggered — directly giving the defensive escape conditions the opponent must achieve.

This gives coaching staff a bidirectional tactical tool: trigger conditions AND escape conditions from the same model.

### 5.2 Formation Switch Detection via Boolean Lattice

Model all possible formations as elements of a **Boolean lattice** `(2^R, ⊆)` where R = set of roles.

A formation `F` is a subset of roles: e.g., `4-3-3 = {GK, LCB, RCB, LB, RB, CDM, LCM, RCM, LW, RW, ST}`.

Partial order: `F ≤ F'` iff `F ⊆ F'` (F is a "sub-formation" of F' — fewer attacking commitments).

**Formation transition detection**: given spatial player positions at times t and t+Δ, detect if the team has moved up or down the lattice (i.e., committed more/fewer players forward). A transition `F_t → F_{t+Δ}` is a Hasse diagram edge if `|F_{t+Δ} \ F_t| = 1` (one role changed).

Use this to build a **formation transition graph** per team and season: nodes = formation clusters (from Section 1.2), edges = observed transitions, edge weight = transition frequency. Fit a discrete Markov chain `T_form` and extract its stationary distribution to characterize a team's tactical flexibility.

---

## PART 6 — CROSS-LEAGUE PLAYER VALUATION SYSTEM

### 6.1 Full Valuation Pipeline

```
StatsBomb Open Data
  + FBref scrape (via soccerdata library)
  + Transfermarkt valuations
  → Canonical action schema (Part 0)
        │
        ▼
[Graph Module]         →  betweenness, pagerank, GAT score  (per-season)
[Network Flow Module]  →  xT-added per action, build-up contribution
[EPV Module]           →  EPV_added per 90 (pass + carry + press)
[xG/xA baseline]       →  from StatsBomb data / understat
        │
        ▼
Feature vector f_i  (18 features, Part 4.1 Step 1)
        │
        ▼
League adjustment  (Bradley-Terry model, Part 4.1 Step 2)
        │
        ▼
Robust PCA → z_i ∈ ℝ^K  (Part 4.1 Step 3)
        │
        ▼
PVS_i = positional percentile rank  (Part 4.1 Step 4)
        │
        ▼
Fair value regression  (Part 4.1 Step 5)
        │
        ▼
MILP Squad Selection  (Part 4.2)
        │
        ▼
Robust MILP with Benders Decomposition  (Part 4.3)
        │
        ▼
OUTPUT: optimal squad + formation + role assignments
        + top-10 undervalued targets per position
        + sensitivity report on budget/age constraints
```

### 6.2 Cross-League Generalization

**Problem**: A player in the Eredivisie (Netherlands) has different raw statistics than a Premier League player of equal quality. Naïve comparison is invalid.

**Solution** — Three-layer normalization:

**Layer 1 — Within-league percentile**: 
`f̃_i = Φ^{-1}(rank(f_i | league L) / (n_L + 1))`
Convert each metric to a within-league z-score (inverse normal transform).

**Layer 2 — Bradley-Terry league adjustment**: 
Multiply by `(λ_L / λ_Premier)^α_k` for each feature k, where `α_k` is feature-specific shrinkage (attacking metrics: α ≈ 0.8; defensive metrics: α ≈ 0.6 — defensive stats are more league-context dependent).

**Layer 3 — Age-adjusted projection**: 
Fit player development curves using **Gamma process regression**:
`f_i(age) = f_peak · Beta(age; a_p, b_p)` — Beta-shaped career arc, peak at age `a_p/(a_p+b_p)`.
Project each player to peak performance for forward-looking valuation.

### 6.3 Scouting Report Generator

For each candidate player `i` and target club `C`, output:

```
SCOUTING REPORT: [Player Name]
Current Club / League: [...]
Age: [...] | Contract Expiry: [...]

VALUATION:
  Fair Market Value:    €Xm   (model predicted)
  Transfermarkt Value:  €Ym   (observed)
  Valuation Gap:        +Z%   (undervalued if positive)

PERFORMANCE PROFILE (league-adjusted percentiles):
  Attacking Contribution:  PVS_attack  [███████░░░] 72nd pctl
  Defensive Contribution:  PVS_defend  [████████░░] 80th pctl
  Build-up (xT added/90):  [██████░░░░] 58th pctl
  Network centrality:      [█████████░] 88th pctl

TACTICAL FIT SCORE with target formation [4-3-3]:
  Role compatibility:  [role cosine similarity]
  Pairwise synergy with existing squad:  [mean compat_{ij}]

KEY MATHEMATICAL HIGHLIGHTS:
  EPV added/90: X.XX  (Top 15% of positional peers, league-adj.)
  Progressive pass EPV: X.XX
  Defensive flow suppression contribution: X.XX
  PageRank in team pass network: 0.XX (top/bottom quartile)

RECOMMENDATION: [Buy / Monitor / Pass] at €Xm ceiling
```

---

## PART 7 — EVALUATION FRAMEWORK

### 7.1 Module-Level Evaluation Metrics

**Graph Module**:
- Predictive validity: Does `C_B` (betweenness) predict player absence impact? Measure `ΔWin%` when top-betweenness player is absent. Hypothesis: `ΔWin% ∝ C_B`.
- Tactical detection: Formation clustering purity (vs. manually labeled formations from coaching staff / football-reference) — Adjusted Rand Index ≥ 0.70.

**EPV Model**:
- Calibration: ECE ≤ 0.02 on held-out competitions
- Discrimination: AUC ≥ 0.72 for next-goal prediction
- OJN-Pass-EPV Benchmark: correctly identify higher-value state ≥ 78% (matching SOTA)

**MILP Solver**:
- Optimality gap: ≤ 0.5% within 5 minutes (CBC solver)
- Feasibility rate: 100% on all constraint combinations tested
- Sensitivity analysis: report shadow prices for budget, age, and formation constraints

**Valuation Model**:
- RMSE on log(MarketValue): ≤ 0.35 log-points (leave-one-season-out CV)
- Rank correlation (Spearman) between PVS and minutes played: ≥ 0.55
- Cross-league transfer prediction: among players transferred between top-5 leagues in 2023/24, does the model's "undervalued" flag predict overperformance at new club? Compute hit rate ≥ 55%.

### 7.2 System Integration Test

**End-to-end pipeline test on a concrete task**:

*Task*: Given a target club (e.g., a mid-table Premier League club), a transfer budget of €80m, and a wage cap of £3m/week:
1. Download all StatsBomb open data for relevant competitions
2. Compute full feature vectors for all N candidates across 5 leagues
3. Run league adjustment and PVS computation
4. Solve MILP to identify optimal squad reinforcements (top 3 positions)
5. Run robust MILP with 50 bootstrap scenarios
6. Output scouting report for top 5 targets

Measure: wall-clock time for full pipeline, solver optimality gap, and manual review by a domain expert (score ≥ 7/10 plausibility rating).

---

## PART 8 — IMPLEMENTATION STACK

### 8.1 Technology Choices

```
Language:        Python 3.11+
Data:            pandas, polars (fast OLAP), psycopg2, sqlalchemy
StatsBomb:       statsbombpy  (pip install statsbombpy)
Graph:           networkx, torch-geometric (PyTorch Geometric for GAT)
Optimization:    PuLP + CBC (open),  scipy.optimize (NLP),  IPOPT via cyipopt,  cvxpy (convex relaxations)
ML:              PyTorch (EPV U-Net), scikit-learn (PCA, clustering), lightgbm (sub models)
Visualization:   matplotlib, mplsoccer (pitch plots), plotly (interactive dashboards)
Computer Vision: ultralytics (YOLOv8), roboflow SDK (player tracking)
Solvers:         HiGHS (via scipy.optimize.milp) for open-source MILP;  
                 Gurobi/CPLEX academic license for production
```

### 8.2 Recommended Project Structure

```
football_analytics_system/
│
├── data/
│   ├── raw/          ← StatsBomb JSON, FBref CSVs
│   ├── processed/    ← canonical action schema, Parquet format
│   └── external/     ← Transfermarkt valuations, FIFA ratings
│
├── modules/
│   ├── graph/
│   │   ├── pass_network.py        ← G_m construction, centrality, entropy
│   │   ├── min_cut_pressing.py    ← push-relabel min-cut
│   │   └── gat_model.py           ← PyTorch Geometric GAT
│   │
│   ├── network_flow/
│   │   ├── xT_surface.py          ← value iteration for xT grid
│   │   ├── max_flow_buildup.py    ← SSP algorithm on zone graph
│   │   └── defensive_suppression.py  ← bilevel program, McCormick
│   │
│   ├── nlp/
│   │   ├── epv_unet.py            ← U-Net EPV model (PyTorch)
│   │   ├── pass_value_nlp.py      ← SQP + CMA-ES pass optimizer
│   │   ├── trajectory_opt.py      ← IPOPT direct collocation
│   │   └── set_piece_opt.py       ← Magnus trajectory NLP
│   │
│   ├── milp/
│   │   ├── player_valuation.py    ← PVS pipeline, Bradley-Terry, rPCA
│   │   ├── squad_selection.py     ← MILP formulation (PuLP)
│   │   ├── robust_milp.py         ← Benders decomposition
│   │   └── substitution_milp.py   ← in-match sub optimization
│   │
│   └── boolean/
│       ├── pattern_recognition.py ← RIPPER, Boolean functions, dual
│       └── formation_lattice.py   ← Boolean lattice, Markov chain
│
├── valuation/
│   ├── cross_league_normalization.py   ← 3-layer normalization
│   ├── development_curves.py           ← Gamma process regression
│   └── scouting_report.py             ← report generator
│
├── evaluation/
│   ├── epv_calibration.py
│   ├── milp_sensitivity.py
│   └── end_to_end_test.py
│
├── notebooks/                    ← Jupyter notebooks for exploration
├── tests/                        ← pytest unit tests for each module
└── README.md
```

### 8.3 Key Mathematical Libraries and Functions

```python
# Pass network + centrality
import networkx as nx
G = nx.DiGraph()
G.add_weighted_edges_from(pass_triples)
bc = nx.betweenness_centrality(G, weight='weight', normalized=True)
pr = nx.pagerank(G, weight='weight', alpha=0.85)
ec = nx.edge_connectivity(G)

# MILP (PuLP)
from pulp import *
prob = LpProblem("squad_selection", LpMaximize)
x = {i: LpVariable(f"x_{i}", cat='Binary') for i in players}
y = {(i,p): LpVariable(f"y_{i}_{p}", cat='Binary') for i in players for p in roles}

# EPV U-Net (PyTorch)
import torch
from torch import nn
class EPV_UNet(nn.Module): ...  # 4-level encoder-decoder with skip connections

# NLP trajectory optimization (cyipopt / scipy)
from scipy.optimize import minimize
result = minimize(energy_coverage_obj, u0, method='SLSQP',
                  jac=grad_obj, constraints=constraints, bounds=bounds)

# Graph Attention Network (PyTorch Geometric)
from torch_geometric.nn import GATConv
class TacticalGAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(in_channels=d, out_channels=64, heads=4)
        self.conv2 = GATConv(64*4, out_channels=1, heads=1)

# Bradley-Terry model
from sklearn.linear_model import LogisticRegression
# Design matrix: X[k, i] = +1, X[k, j] = -1 for match k (i vs j)
# Fit: beta = LogisticRegression(fit_intercept=False).fit(X, outcomes).coef_
```

---

## PART 9 — RESEARCH EXTENSIONS (PhD-Level Contributions)

The following represent **open problems** where this system can produce original research:

1. **Bilevel Optimization for Pressing**: Rigorous solution of the defensive flow suppression bilevel program (Section 2.3) via strong duality. Characterize when the McCormick relaxation is tight.

2. **EPV as a Martingale**: Prove or disprove that a properly calibrated EPV model is a martingale under the true data-generating process. Explore its Doob decomposition into predictable + innovation components.

3. **Graph Spectrum of Formations**: Study the Laplacian eigenvalue structure `L = D − A` of pass networks. Conjecture: the Fiedler value (second-smallest eigenvalue of L) correlates with team resilience to pressing. Test on 5 seasons of StatsBomb data.

4. **Multi-Objective MILP Pareto Front**: Squad selection is inherently multi-objective (quality vs. youth vs. budget vs. cohesion). Compute the full Pareto frontier via ε-constraint method and characterize the trade-off curves analytically.

5. **Stochastic EPV with Lévy Jumps**: The EPV process has discontinuities at goals, red cards, and injuries. Model as `dEPV_t = μ_t dt + σ_t dW_t + ΔN_t` (Lévy process). Estimate parameters from StatsBomb data and derive closed-form optimal stopping times for substitution decisions.

6. **Transfer Market Efficiency Test**: Use your fair value model (Section 4.1) to test whether the football transfer market is informationally efficient in the semi-strong sense: do observed transfer fees systematically differ from model fair values in a predictable direction?

---

## APPENDIX A — Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| `G_m = (V,E,w)` | Weighted directed pass network for match m |
| `P` | Row-stochastic Markov transition matrix of pass network |
| `π` | Stationary distribution of P (PageRank) |
| `λ(G)` | Edge connectivity of G |
| `xT(x,y)` | Expected Threat at pitch coordinate (x,y) |
| `EPV(s)` | Expected Possession Value at game state s |
| `PVS_i` | Player Value Score (positional percentile, league-adjusted) |
| `x_i ∈ {0,1}` | MILP binary selection variable |
| `y_{ip} ∈ {0,1}` | MILP role assignment variable |
| `compat_{ij}` | Pairwise style compatibility score between players i,j |
| `λ_L` | League strength factor from Bradley-Terry model |
| `z_i ∈ ℝ^K` | Player embedding from Robust PCA |
| `f^d` | Dual Boolean function of f |
| `F_form` | Formation as subset of roles (Boolean lattice element) |
| `Φ` | Standard normal CDF |

---

## APPENDIX B — StatsBomb Data Quick Reference

```python
from statsbombpy import sb

# List available competitions
comps = sb.competitions()

# Get matches for a competition/season
matches = sb.matches(competition_id=11, season_id=90)  # La Liga 2020/21

# Get full event stream for one match
events = sb.events(match_id=3788741)
# Returns DataFrame with columns:
#   id, index, period, timestamp, minute, second, type, possession,
#   possession_team, play_pattern, team, player, position,
#   location [x,y], duration, under_pressure, related_events,
#   + type-specific nested dicts (pass, shot, carry, etc.)

# Get 360° freeze frames (select competitions only)
frames = sb.frames(match_id=3788741)
# Returns: freeze_frame (list of {location, player, team, actor})

# Useful competition IDs:
#   2  = UEFA Champions League
#   11 = La Liga
#   37 = La Liga (Women's)
#   49 = FIFA World Cup
#   55 = UEFA Euro

# Parse pass data
passes = events[events.type == 'Pass'].copy()
passes['x_start'] = passes.location.apply(lambda l: l[0])
passes['y_start'] = passes.location.apply(lambda l: l[1])
passes['x_end']   = passes.pass.apply(lambda p: p.get('end_location',[None,None])[0])
passes['y_end']   = passes.pass.apply(lambda p: p.get('end_location',[None,None])[1])
passes['completed'] = passes.pass.apply(lambda p: p.get('outcome') is None)
```

---

*End of Project Specification — Version 1.0*

*This document is self-contained. A PhD mathematician / data scientist should be able to implement each module independently following the formulations above, using the referenced libraries and data sources.*

---
---

# UNIFIED MATHEMATICAL FOOTBALL ANALYTICS SYSTEM
## Version 2.0 — From Modules to a Coherent, Interactive Product
### Follow-up Specification: Status, Coherence Architecture, UI, and the Road Ahead

> **Why this addendum exists.** Version 1.0 specified six mathematically rigorous *solvers* (graph, flow, NLP, MILP, Boolean, valuation). They have now been **implemented as the `fas` Python package** (see `src/fas/`, `docs/SPEC.md`). But a pile of solvers is not a *system*. Version 2.0 specifies the **connective tissue** — the canonical entity model, feature store, orchestration DAG, serving API, and **interactive UI** — that turns the modules into one coherent instrument a human can actually *drive*. It also opens **new mathematical frontiers** (uncertainty propagation, optimal transport, game-theoretic equilibria, causal counterfactuals) that only become reachable once the pieces are wired together.

---

## PART 10 — IMPLEMENTATION STATUS (Phase 0 Complete)

### 10.0 Spec → Code reconciliation

The `fas` package implements the v1.0 spec on a core, pure-Python install
(`numpy/scipy/networkx/pulp/scikit-learn/pandas`). Heavy ML / global-solver
dependencies are isolated as optional extras (`[ml] [gnn] [nlp]`).

| Spec | Module | State | Notes |
|---|---|---|---|
| 0 — schema, loader, id-unify, DDL | `fas.data` | ✅ done | Jaro-Winkler matcher is dependency-free |
| 1.1–1.3 — networks, 4 centralities, entropy, min-cut | `fas.graph` | ✅ done | Brandes + PageRank power-iteration + preflow-push |
| 1.4 — GAT collective valuation | `fas.graph.gat_model` | 🟡 stub | needs `[gnn]`; centralities serve as interim feature |
| 2.1–2.2 — xT value iteration, max-flow build-up | `fas.network_flow` | ✅ done | min-cost max-flow via SSP |
| 2.3 — bilevel suppression | `…defensive_suppression` | 🟡 partial | McCormick + successive linearization, needs `cvxpy` |
| 3.1 — EPV U-Net | `fas.nlp.epv_unet` | 🟡 stub | needs `[ml]` (torch) |
| 3.2 — pass reward-risk optimizer | `…pass_value_nlp` | ✅ done | SLSQP multi-start + ellipsoidal exclusion |
| 3.3 — trajectory collocation | `…trajectory_opt` | 🟡 stub | needs `[nlp]` (cyipopt) |
| 3.4 — set-piece Magnus trajectory | `…set_piece_opt` | ✅ done | RK4 + Nelder-Mead |
| 4.1 — Bradley-Terry, rPCA, PVS, fair value | `…player_valuation` | ✅ done | rPCA via ADMM/PCP |
| 4.2 — squad MILP (C1–C11) | `…squad_selection` | ✅ done | formation choice + linearized bonus |
| 4.3 — robust minimax-regret MILP | `…robust_milp` | ✅ done | extensive form; Benders is the scale path |
| 4.4 — substitution MILP | `…substitution_milp` | ✅ done | gain table injected |
| 5.1–5.2 — Boolean fns, dual, lattice | `fas.boolean` | ✅ done | decision-list learner + dual involution |
| 6.2–6.3 — 3-layer norm, dev curves, scouting | `fas.valuation` | ✅ done | Beta career arc fit |
| 7.1–7.2 — metrics, end-to-end test | `fas.evaluation`, `examples` | ✅ done | `fas demo` runs offline synthetic pipeline |

**Verification gate met:** 19 passing tests; `fas demo` runs graph → xT →
max-flow → PVS → MILP and returns an *Optimal* squad with a chosen formation.

### 10.1 What "Phase 0" deliberately left out

The implemented stack runs on **synthetic data** and **point estimates**. It has
no persistence, no learned EPV/GAT models, no way to *see* its own output, and
no thread that makes module B consume module A automatically. Those four gaps
define the rest of this document.

---

## PART 11 — THE COHERENCE PROBLEM AND ITS ARCHITECTURE

### 11.0 The thesis

> Six solvers become one system when they all read from, and write to, a single
> **canonical entity model**, are sequenced by one **orchestration DAG**, share a
> versioned **feature/model store**, and are exposed through one **serving API**
> that an **interactive UI** drives. Coherence is an architecture problem first
> and a math problem second — but the architecture *unlocks* new math (Part 14).

```
                    ┌──────────────────────────────────────────────┐
                    │                INTERACTIVE UI                  │
                    │  Match Lab · Pressing Planner · Squad Builder  │
                    │     Scouting Hub · Match Theatre               │
                    └───────────────▲───────────────┬───────────────┘
                                    │ JSON / WS      │ controls
                    ┌───────────────┴───────────────▼───────────────┐
                    │            SERVING API (FastAPI)               │
                    │  /network /xt /flow /valuation /optimize/solve │
                    └───────────────▲───────────────┬───────────────┘
                                    │                │ warm-start re-solve
        ┌───────────────────────────┴────────────────▼──────────────┐
        │             ORCHESTRATION DAG (Prefect/Dagster)            │
        │ ingest → canonicalize → graph/flow/epv → valuation → MILP  │
        └───▲──────────────▲───────────────▲───────────────▲─────────┘
            │              │               │               │
   ┌────────┴───┐  ┌───────┴──────┐ ┌──────┴──────┐ ┌──────┴───────┐
   │  CANONICAL │  │   FEATURE     │ │   MODEL      │ │   ARTIFACT   │
   │  ENTITIES  │  │   STORE       │ │   REGISTRY   │ │   STORE      │
   │ Match/     │  │ (Parquet,     │ │ xT, EPV,     │ │ reports,     │
   │ Player/    │  │  versioned)   │ │ GAT, β_BT,   │ │ solutions,   │
   │ TeamSeason │  │               │ │ dev-curves   │ │ Pareto sets  │
   └────────────┘  └───────────────┘ └─────────────┘ └──────────────┘
```

### 11.1 The canonical entity model — the spine

Today each module passes raw DataFrames around. Introduce three immutable,
serializable records that every module reads and enriches. *This is the single
most important change* — it is what makes PVS feed the MILP feed the scouting
report without glue code each time.

```python
# src/fas/entities.py  (NEW)
@dataclass(frozen=True, slots=True)
class MatchObject:
    match_id: int
    actions: pd.DataFrame                # canonical schema (Part 0)
    pass_networks: dict[int, PassNetwork]    # team_id -> network
    centrality: dict[int, pd.DataFrame]      # team_id -> per-player table
    xt_added: pd.Series                       # action_id -> Δ xT
    zone_flow: dict[int, BuildupResult]
    epv_timeline: pd.DataFrame | None         # filled once EPV model exists
    meta: MatchMeta

@dataclass(frozen=True, slots=True)
class PlayerSeason:
    player_uid: int
    league: str
    minutes: int
    features_90: pd.Series        # the 18-vector (Part 4.1 Step 1)
    graph_features: pd.Series     # betweenness/pagerank/GAT (Part 1)
    epv_added_90: float
    pvs: float                    # Part 4.1 Step 4
    pvs_distribution: np.ndarray  # bootstrap samples (Part 4.3 / Part 14)
    fair_value_eur: float
    market_value_eur: float
    dev_curve: tuple              # (a_p, b_p, f_peak)  Part 6.2 Layer 3

@dataclass(frozen=True, slots=True)
class TeamSeason:
    team_id: int
    bt_strength: float            # β from Bradley-Terry
    formation_markov: np.ndarray  # Part 5.2 stationary chain
    squad: list[int]              # player_uids
```

**Contract:** every module gains a `*.enrich(obj) -> obj` adapter. The DAG is
then literally a fold over these adapters. Modules never again re-derive a pass
network or re-fit an xT surface that a sibling already produced.

### 11.2 Feature store & model registry

- **Feature store** — `data/processed/features/{entity}/{version}.parquet`,
  keyed by `player_uid`/`match_id`, with a `lineage.json` recording source
  competitions, code git-SHA, and parameter hash. Enables leave-one-season-out
  CV (Part 7) without recomputation and makes the UI instant (reads, not solves).
- **Model registry** — `artifacts/models/{name}/{version}/` for the **xT grid**,
  **EPV U-Net weights**, **GAT weights**, **Bradley-Terry β-vector**, **fair-value
  θ**, **per-position dev-curve params**. A tiny `registry.json` maps
  `name -> active version`. Everything downstream pins a version, so a scouting
  report is always reproducible.

### 11.3 Orchestration: the analytics DAG

```
ingest_statsbomb ─┐
ingest_fbref ─────┼─► canonicalize ─► unify_ids ─► build_match_objects
ingest_tmkt ──────┘                                      │
                                                         ▼
                          ┌──────────── graph_features (Part 1)
                          ├──────────── flow_features  (Part 2: xT, build-up)
                          ├──────────── epv_features    (Part 3.1 once trained)
                          └──────────── boolean_patterns (Part 5)
                                                         │
                                                         ▼
                              league_adjust (Bradley-Terry, Part 4.1/6.2)
                                                         │
                                                         ▼
                              build_PlayerSeason ─► PVS ─► fair_value
                                                         │
                                          ┌──────────────┼───────────────┐
                                          ▼              ▼               ▼
                                   squad_MILP      robust_MILP      scouting_reports
                                    (Part 4.2)      (Part 4.3)        (Part 6.3)
```

Implement with **Prefect** (lightweight, pure-Python, good local dev) or
**Dagster** (stronger asset lineage). Each node caches to the feature store;
re-runs are incremental on changed inputs only.

---

## PART 12 — UI & INTERACTIVE DISPLAY

### 12.0 Design principles

1. **Show the math, don't hide it.** Every visual encodes a specific quantity
   (node size = PageRank, edge width = pass volume, heat = xT). A hover reveals
   the number and its formula.
2. **Optimization-in-the-loop.** The squad and substitution MILPs are fast
   enough to **re-solve live** as the user moves a slider. The UI is a control
   surface over a solver, not a static dashboard.
3. **Two-tier build.** Ship research value immediately on a Python-native tier;
   graduate to a product tier only where interactivity demands it.

### 12.1 Architecture: two tiers

| | **Tier A — Research console** | **Tier B — Product** |
|---|---|---|
| Frontend | **Streamlit** or **Plotly Dash** | **Next.js + React + TypeScript** |
| Pitch viz | `mplsoccer` (static) + Plotly (interactive) | **D3 / deck.gl** custom canvas |
| Graph viz | `networkx` + Plotly | `react-force-graph` / `sigma.js` |
| Backend | in-process `fas` calls | **FastAPI** + Pydantic, `fas` as lib |
| Live solve | synchronous (CBC < 1s) | **WebSocket** push, warm-started CBC/HiGHS |
| Audience | analyst / researcher, today | club staff, scouts, demo |
| Effort | days | weeks |

**Recommendation:** build **Tier A first** (it is ~80% of the analytical value
for ~20% of the effort and validates every view), then port the two
"killer" interactive views (Squad Builder, Match Theatre) to Tier B.

### 12.2 The five core views

#### View 1 — Match Lab (Parts 1 & 2 made visible)

```
┌─────────────────────────────────────────────────────────────┐
│ Match: Barcelona 3–0 Sevilla   [La Liga 2020/21]   ▣ Home ▢ Away│
├───────────────────────────────┬─────────────────────────────┤
│        PASS NETWORK            │      EXPECTED THREAT (xT)    │
│   (nodes sized by PageRank,    │   16×12 heatmap, possession  │
│    placed at avg position,     │   corridors overlaid as      │
│    edges = pass volume)        │   max-flow ribbons           │
│         ⊙───⊙                  │   ░░▒▒▓▓██  → goal            │
│        ╱ ╲ ╱ ╲                 │                              │
│      ⊙   ⊙   ⊙                 │   build-up potency: 0.48     │
│                                │   edge connectivity λ: 3     │
├───────────────────────────────┴─────────────────────────────┤
│ Phase ◀ [00:00 ━━━●━━━━━━━ 90:00] ▶   entropy H(G): 5.04      │
│ network velocity ΔG between phases: ▁▃▇▂▁  (spike @ 60' = sub)│
└─────────────────────────────────────────────────────────────┘
```
Interactions: scrub the **phase slider** (6×15-min snapshots) → network and xT
animate; click a node → that player's centrality time-series; toggle teams.

#### View 2 — Pressing Planner (Part 1.3 min-cut)

```
┌─────────────────────────────────────────────────────────────┐
│ Opponent: Sevilla (last 5 matches aggregated)                │
│  GK ●──────●──────●  ← MIN-CUT edges highlighted RED          │
│        ╲    │    ╱     these are the lanes to press           │
│         ●──●──●                                               │
│  Ranked pressing assignment:                                  │
│   1. press #6 Fernando → cut to #20 (priority 14.2) ████████  │
│   2. press #25 Jordán  → cut to #10 (priority  9.1) █████     │
│   3. press #16 Navas   → cut to #7  (priority  6.4) ███       │
│  Pressing energy budget B: [────●──── 60%]  (re-solve)        │
└─────────────────────────────────────────────────────────────┘
```
Moving the **budget slider** re-runs the §2.3 suppression program and re-ranks.

#### View 3 — Squad Builder (Part 4 — the flagship)

```
┌──────────────────────────────────────────────────────────────┐
│ FORMATION: ▼ 4-3-3   Budget €[──●──── 80m]  Wage £[──●─ 3m/wk] │
│ Young ≤23 ≥ [3]   Old ≥30 ≤ [5]   Homegrown ≥ [4]   γ cohesion │
├────────────────────────────────┬─────────────────────────────┤
│        PITCH (optimal XI)       │   BENCH / SQUAD (S=23)       │
│            ⓖⓚ                   │   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○    │
│      ⓛⓑ ⓒⓑ ⓒⓑ ⓡⓑ              │                              │
│        ⓒⓜ ⓒⓜ ⓒⓜ               │   Objective:   10.88         │
│      ⓛⓦ   ⓢⓣ   ⓡⓦ             │   Opt. gap:    0.0%          │
│                                 │   Solve:       0.3s          │
├────────────────────────────────┴─────────────────────────────┤
│ SENSITIVITY (shadow prices):  +€1m budget → +0.04 quality      │
│                               −1 young-quota → +0.11 quality    │
│ PARETO FRONT  quality ▲   ●you-are-here                        │
│               │   ●  ●                                          │
│               │ ●        navigate ◀ youth ─── budget ▶          │
│               └──────────────────► cohesion                     │
└──────────────────────────────────────────────────────────────┘
```
**The loop:** any slider/formation change → API `/optimize/solve` with a
**warm start** from the current solution → CBC/HiGHS returns in <1s → the XI
**diffs** (players in green/out red). The **Pareto explorer** (Part 9.4 / Part
14.2) lets the user walk the quality-youth-budget-cohesion trade surface.

#### View 4 — Scouting Hub (Part 6.3)

```
┌─────────────────────────────────────────────────────────────┐
│ ◍ Player radar (8 league-adj. percentile axes)   FAIR VALUE  │
│        attack ▲                                   €42m model  │
│      build  ╱│╲  defend                           €28m market │
│           ╱  │  ╲                                 GAP +50% ✅  │
│   centrality──┼──aerial          RECOMMENDATION: BUY ≤ €38m   │
│           ╲  │  ╱                                            │
│      press  ╲│╱  passing      Similar players (OT distance):  │
│        carry ▼                  1. … 2. … 3. …               │
│ Filter: pos[CM] age≤[24] league[any] gap≥[+20%]  → ranked    │
│ Undervalued targets table ▼ (top 10 by valuation gap)        │
└─────────────────────────────────────────────────────────────┘
```

#### View 5 — Match Theatre (Parts 3 & 4.4 — live/replay)

```
┌─────────────────────────────────────────────────────────────┐
│ EPV timeline ▁▂▅▇▅▃▂▁▃▇█▅ ← goals as ▲ jumps (Lévy, Part 14.6)│
│ minute ◀ [━━━●━━━━━━] 67'  scoreline 1-1                       │
├────────────────────────────┬────────────────────────────────┤
│ SUBSTITUTION OPTIMIZER      │ SET-PIECE DESIGNER (Part 3.4)   │
│ best move: #11 OFF, #19 ON  │  corner → Magnus trajectory     │
│ at 72'  ΔEPV +0.08          │  target zone ▣  wall ▢          │
│ subs left: 2  [apply]       │  v₀,ω solved → landing ✕        │
├────────────────────────────┴────────────────────────────────┤
│ PASS VALUE map (Part 3.2): for ball-carrier at ●, best target │
│  shows V=R−K surface, optimal pass ➤, risk ellipses (defenders)│
└─────────────────────────────────────────────────────────────┘
```

### 12.3 Visual encoding contract (math → pixels)

| Quantity | Module | Encoding |
|---|---|---|
| PageRank π(v) | 1.1 | node **radius** |
| Pass volume W[i,j] | 1.1 | edge **width** + opacity |
| Betweenness C_B | 1.1 | node **halo** thickness |
| Min-cut edges | 1.3 | edge **red**, dashed |
| xT(x,y) | 2.1 | pitch **heat** (sequential colormap) |
| Max-flow corridor | 2.2 | **ribbon** width = flow units |
| Network entropy H | 1.1 | gauge / sparkline |
| EPV(s) | 3.1 | **timeline** area + goal jumps |
| PVS percentile | 4.1 | radar **axis** + bar fill |
| Valuation gap | 4.1/6 | **diverging** color (green=undervalued) |
| Pareto point | 9.4 | scatter on trade-off surface |
| Bootstrap PVS dist. | 4.3/14 | **violin** / band on radar |

### 12.4 Latency budget (why live solve is feasible)

The squad MILP on ~200 candidates solves to optimality in **<1s with CBC** and
**<200ms with HiGHS** when **warm-started** from the incumbent. Strategy:
keep the model object resident in the API process; on a control change, mutate
only the changed RHS/objective coefficients and re-`solve()` with the previous
basis. Substitution and pressing models are smaller still. Only the **robust
MILP (K scenarios)** and **Pareto sweep** run async with a WebSocket progress
push.

---

## PART 13 — REMAINING WORK: PHASED ROADMAP

### Phase 1 — The real data spine (unlocks everything)
- StatsBomb ingest at scale (`statsbombpy` → canonical Parquet, all open comps).
- FBref per-90 scrape via `soccerdata`; Transfermarkt values via `transfermarkt-api`.
- Run the **Jaro-Winkler id-unification** for real; build the `player_uid` table.
- Stand up the **PostgreSQL** schema (`schema.sql`) or DuckDB for local dev.
- **Deliverable:** populated `MatchObject`/`PlayerSeason` records for ≥3 leagues.
- **Acceptance:** `fas demo --real` runs the pipeline on a real competition.

### Phase 2 — Learned models (turns stubs into science)
- **EPV U-Net** (Part 3.1): build state tensors, train, register weights;
  hit **ECE ≤ 0.02, AUC ≥ 0.72** (Part 7.1). Requires `[ml]`.
- **GAT** (Part 1.4): possession graphs → ΔEPV; export per-player attention as
  the `graph_contribution` feature. Requires `[gnn]`.
- **LightGBM substitution model** (Part 4.4): historical sub outcomes → gain table.
- **Trajectory collocation** (Part 3.3): wire IPOPT via `cyipopt`.
- **Acceptance:** model registry holds versioned, calibrated EPV + GAT; PVS
  feature vector is fully populated (no synthetic placeholders).

### Phase 3 — Coherence layer
- Implement `fas.entities` + `*.enrich()` adapters on every module.
- Feature store + model registry + lineage.
- Orchestration DAG (Prefect).
- **Acceptance:** changing one upstream parameter incrementally re-materializes
  only affected downstream assets; a scouting report is reproducible from SHAs.

### Phase 4 — UI (Tier A then B)
- Tier A Streamlit/Dash for all five views.
- FastAPI serving layer; port Squad Builder + Match Theatre to Tier B with
  warm-started live solve over WebSocket.
- **Acceptance:** end-to-end demo (Part 7.2) driven entirely from the UI; squad
  re-solves <1s on slider change; domain-expert plausibility ≥ 7/10.

### Phase 5 — Research extensions (Part 9 → Part 14, productionized)
See Part 14. Each becomes a notebook + a module + (where relevant) a UI affordance.

### Effort / dependency summary

```
Phase 1 ─► Phase 2 ─► Phase 3 ─► Phase 4 ─► Phase 5
 data       models     wiring      UI         research
 (core)     ([ml]/[gnn])(prefect)  (web)      (cvxpy/torch)
```

---

## PART 14 — NEW MATHEMATICAL FRONTIERS (deepening the contribution)

These extend Part 9 and are *enabled* by the coherence layer — they require
modules to exchange **distributions and structure**, not just point scalars.

### 14.1 Uncertainty propagation end-to-end (Bayesian / conformal)
Replace point PVS with **calibrated predictive distributions** (conformal
prediction on the fair-value residuals; Bayesian posterior on Bradley-Terry β).
Propagate them through to the **robust MILP** (Part 4.3) so the scenarios
`ξ^(k)` are *principled posterior draws*, not bootstrap heuristics. UI: violin
bands on every PVS. *Contribution:* coverage-guaranteed squad robustness.

### 14.2 Multi-objective Pareto front (ε-constraint, navigable)
Compute the full **quality / youth / budget / cohesion** Pareto frontier via the
ε-constraint method (sweep ε on three objectives, solve the MILP at each grid
point), store the frontier as an artifact, and let the UI **navigate** it.
*Contribution:* analytic characterization of the trade-off surface’s shape and
its supporting prices.

### 14.3 Optimal transport for cross-league style (deepens Part 6)
Model each player as a **distribution over action types/zones**; compute
**Wasserstein distances** between players and **Wasserstein barycenters** per
league. League adjustment becomes an **OT map** from one league’s style measure
to the reference league’s — a more faithful normalization than scalar shrinkage.
*Contribution:* "style translation" — what a player’s output *becomes* in a new
league, not just a rescale.

### 14.4 Graph signal processing on pass networks (deepens Part 9.3)
Treat player metrics as **signals on the pass graph**; apply **spectral filters**
of the Laplacian `L = D − A`. Test the **Fiedler-value ↔ pressing-resilience**
conjecture across 5 seasons; design a **low-pass tactical filter** that denoises
possession value. *Contribution:* a spectral theory of team structure.

### 14.5 Game-theoretic pressing equilibria (deepens Part 2.3)
Recast the suppression bilevel as a **Stackelberg game** (defender leads,
attacker’s max-flow follows) and compute equilibrium pressing strategies; study
mixed strategies over pressing triggers (ties to the Part 5 Boolean traps).
*Contribution:* equilibrium pressing policy with regret bounds.

### 14.6 Streaming / online inference for live matches
**Recursive Bayesian xT** and a **Kalman-style EPV tracker** updated per event,
so Match Theatre runs **live**, not just on replays. Model EPV jumps at goals /
red cards as a **Lévy process** (Part 9.5) and derive **optimal substitution
stopping times**. *Contribution:* real-time decision support with explicit jump
risk.

### 14.7 Causal counterfactuals for transfers
A **do-calculus / potential-outcomes** layer: estimate the counterfactual win
contribution of *adding player i to club C* (adjusting for confounders via the
pass-network structure as the adjustment set). Connects valuation (Part 4/6) to
**actual outcomes**, and powers a UI "what-if signing" simulation.
*Contribution:* moving from correlation (PVS) to causal squad impact.

---

## PART 15 — DEFINITION OF DONE (system-level)

The system is "v2.0 complete" when a user can, **in the UI, in one session**:

1. Pick a real competition; the DAG materializes all entities (Phase 1–3).
2. Open **Match Lab** and scrub a real match’s pass network + xT + flow.
3. Open **Pressing Planner**, set a budget, and get min-cut press targets.
4. Open **Squad Builder**, set €80m / £3m-wk / age & homegrown quotas, pick a
   formation, and watch the **MILP re-solve live** with sensitivity + a
   navigable **Pareto front** and **uncertainty bands**.
5. Open **Scouting Hub**, filter to undervalued CMs ≤24, and export a
   reproducible **scouting report** (pinned model versions).
6. Open **Match Theatre** and get an EPV-optimal substitution and a set-piece
   delivery solution.

…with the Part 7 metric gates met (**ECE ≤ 0.02, AUC ≥ 0.72, ARI ≥ 0.70,
log-value RMSE ≤ 0.35**), squad re-solve **<1s**, and a domain-expert
plausibility score **≥ 7/10**.

---

*End of Project Specification — Version 2.0*

*Version 1.0 specified the mathematics. Version 2.0 specifies how the mathematics
becomes a coherent, interactive instrument — and the new mathematics that the
instrument, in turn, makes possible.*
