from __future__ import annotations

"""
DUJ v0.2 — Minimal Empirical Stability Stress Test

Pure NumPy + matplotlib reference implementation (simulation + plots).

Core principles:
- Absorbing collapse: stop on first |x_t| > theta
- Right-censoring: store t_max for non-collapsed trials; conditional TTC stats exclude censored
- Separation: simulation functions are independent from plotting helpers
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Data container
# =========================

@dataclass(frozen=True)
class SweepResult:
    rho_grid: np.ndarray
    mu_grid: np.ndarray
    n_trials: int
    t_max: int

    collapsed_count: np.ndarray
    collapse_prob: np.ndarray
    survival_frac: np.ndarray

    collapse_times_all: np.ndarray

    mean_ttc_conditional: np.ndarray
    median_ttc_conditional: np.ndarray

    collapse_pos_count: np.ndarray
    collapse_neg_count: np.ndarray


# =========================
# Core simulation
# =========================

def run_trial(
    *,
    mu: float,
    alpha: float,
    k: float,
    sigma: float,
    theta: float,
    t_max: int,
    rng: np.random.Generator,
    x0: float = 0.0
) -> Tuple[bool, int, int, np.ndarray]:

    if t_max <= 0:
        raise ValueError("t_max must be positive.")
    if alpha <= 0 or theta <= 0 or k <= 0:
        raise ValueError("alpha, theta, and k must be > 0.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if mu < 0:
        raise ValueError("mu must be >= 0.")

    x = float(x0)
    traj = [x]

    for t in range(t_max):
        a_t = float(np.clip(-k * x, -alpha, alpha))
        eps = float(rng.normal(0.0, sigma)) if sigma > 0 else 0.0

        x = x + a_t - mu + eps
        traj.append(x)

        if abs(x) > theta:
            direction = 1 if x > 0 else -1
            return True, t, direction, np.array(traj, dtype=float)

    return False, t_max, 0, np.array(traj, dtype=float)


def sweep_rho(
    *,
    rho_grid: np.ndarray,
    alpha: float,
    k: float,
    theta: float,
    sigma: float,
    t_max: int,
    n_trials: int,
    seed: Optional[int] = 0
) -> SweepResult:

    rho_grid = np.asarray(rho_grid, dtype=float)
    if rho_grid.ndim != 1 or rho_grid.size == 0:
        raise ValueError("rho_grid must be a 1D non-empty array.")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if t_max <= 0:
        raise ValueError("t_max must be positive.")

    mu_grid = rho_grid * float(alpha)
    rng = np.random.default_rng(seed)

    R = rho_grid.size
    collapse_times_all = np.full((R, n_trials), t_max, dtype=int)
    collapsed_count = np.zeros(R, dtype=int)
    collapse_pos_count = np.zeros(R, dtype=int)
    collapse_neg_count = np.zeros(R, dtype=int)

    for i, mu in enumerate(mu_grid):
        for j in range(n_trials):
            collapsed, t_c, direction, _ = run_trial(
                mu=float(mu),
                alpha=float(alpha),
                k=float(k),
                sigma=float(sigma),
                theta=float(theta),
                t_max=int(t_max),
                rng=rng,
            )

            if collapsed:
                collapse_times_all[i, j] = int(t_c)
                collapsed_count[i] += 1

                if direction > 0:
                    collapse_pos_count[i] += 1
                elif direction < 0:
                    collapse_neg_count[i] += 1

    collapse_prob = collapsed_count / float(n_trials)
    survival_frac = 1.0 - collapse_prob

    mean_ttc_cond = np.full(R, np.nan, dtype=float)
    median_ttc_cond = np.full(R, np.nan, dtype=float)

    for i in range(R):
        times = collapse_times_all[i]
        collapsed_mask = times < t_max
        if np.any(collapsed_mask):
            ttc = times[collapsed_mask].astype(float)
            mean_ttc_cond[i] = float(np.mean(ttc))
            median_ttc_cond[i] = float(np.median(ttc))

    return SweepResult(
        rho_grid=rho_grid,
        mu_grid=mu_grid,
        n_trials=int(n_trials),
        t_max=int(t_max),
        collapsed_count=collapsed_count,
        collapse_prob=collapse_prob,
        survival_frac=survival_frac,
        collapse_times_all=collapse_times_all,
        mean_ttc_conditional=mean_ttc_cond,
        median_ttc_conditional=median_ttc_cond,
        collapse_pos_count=collapse_pos_count,
        collapse_neg_count=collapse_neg_count,
    )


# =========================
# Plotting utilities
# =========================

def _maybe_savefig(save_path: Optional[str], dpi: int = 300) -> None:
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")


def plot_collapse_probability(
    out: SweepResult,
    *,
    title: str = "DUJ v0.2 — Collapse Probability vs ρ",
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> None:

    plt.figure()
    plt.plot(out.rho_grid, out.collapse_prob, marker="o")
    plt.xlabel("ρ = μ / α")
    plt.ylabel(f"P(collapse by T_max={out.t_max})")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    _maybe_savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_time_to_collapse_mean(
    out: SweepResult,
    *,
    title: str = "DUJ v0.2 — Mean Time-to-Collapse (Conditional) vs ρ",
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> None:

    plt.figure()
    plt.plot(out.rho_grid, out.mean_ttc_conditional, marker="o")
    plt.xlabel("ρ = μ / α")
    plt.ylabel("E[t_collapse | collapse] (steps)")
    plt.title(title)
    _maybe_savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_time_to_collapse_median(
    out: SweepResult,
    *,
    title: str = "DUJ v0.2 — Median Time-to-Collapse (Conditional) vs ρ",
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> None:

    plt.figure()
    plt.plot(out.rho_grid, out.median_ttc_conditional, marker="o")
    plt.xlabel("ρ = μ / α")
    plt.ylabel("median(t_collapse | collapse)")
    plt.title(title)
    _maybe_savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_overlay_collapse_probability(
    results: List[Tuple[str, SweepResult]],
    *,
    title: str = "DUJ v0.2 — Collapse Probability vs ρ (Overlay)",
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> None:

    plt.figure()
    for label, out in results:
        plt.plot(out.rho_grid, out.collapse_prob, marker="o", label=label)
    plt.xlabel("ρ = μ / α")
    plt.ylabel("P(collapse by T_max)")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    _maybe_savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()
