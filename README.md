# DUJ v0.2 — Minimal Stability Stress Test (Empirical First-Passage Scaffold)

DUJ v0.2 is a minimal, reproducible instrument for empirically characterizing first-passage (collapse) behavior in a bounded corrective dynamical system under persistent external pressure. This repository does not claim novelty in control theory. Its purpose is to provide a clean, executable baseline for measuring how collapse probability and time-to-collapse vary as persistent pressure approaches and exceeds corrective capacity.

## Model

State:
- `x_t ∈ ℝ` (stability margin)

Dynamics:
- `x_{t+1} = x_t + clip(−k x_t, −α, α) − μ + ε_t`
- `ε_t ~ Normal(0, σ²)`

Absorbing collapse:
- Stop trial at first `t` such that `|x_t| > θ`.

Control parameter:
- `ρ = μ / α` (swept).

## What this measures

For each `ρ`, DUJ v0.2 estimates:
- `P(collapse by T_max)`
- Conditional mean and median time-to-collapse
- Survival fraction at `T_max`
- Direction of collapse (diagnostic)

## Quickstart

```python
import numpy as np
from duj_v02 import sweep_rho, plot_collapse_probability, plot_time_to_collapse_mean, plot_trajectories

rho_grid = np.round(np.arange(0.2, 2.0 + 1e-9, 0.05), 2)
out = sweep_rho(
    rho_grid=rho_grid,
    alpha=1.0, k=1.0, theta=10.0, sigma=0.05,
    t_max=500, n_trials=300, seed=123
)

plot_collapse_probability(out, save_path="collapse_prob.png")
plot_time_to_collapse_mean(out, save_path="mean_ttc.png")
plot_trajectories(alpha=1.0, k=1.0, theta=10.0, sigma=0.05,
                  t_max=500, rhos=[0.5, 1.0, 1.5], seed=42)
