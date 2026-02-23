import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from duj_v02 import run_trial, sweep_rho

st.set_page_config(page_title="DUJ v0.2 Demo", layout="centered")

st.title("DUJ v0.2 — Minimal Stability Stress Test")
st.caption("Interactive exploration of collapse under bounded corrective capacity and persistent pressure.")

if "sigma" not in st.session_state:
    st.session_state["sigma"] = None

with st.sidebar:
    st.header("Parameters")

    alpha = st.number_input("α (max corrective capacity)", min_value=0.01, value=1.0, step=0.1)
    k = st.number_input("k (responsiveness gain)", min_value=0.01, value=1.0, step=0.1)
    theta = st.number_input("θ (collapse threshold)", min_value=0.1, value=10.0, step=1.0)

    if st.session_state["sigma"] is None:
        st.session_state["sigma"] = 0.05 * float(alpha)

    col1, col2 = st.columns([2, 1])

    with col1:
        sigma = st.number_input(
            "σ (noise std dev)",
            min_value=0.0,
            value=float(st.session_state["sigma"]),
            step=0.01
        )

    with col2:
        if st.button("Reset σ=0.05α"):
            st.session_state["sigma"] = 0.05 * float(alpha)
            sigma = float(st.session_state["sigma"])

    t_max = st.number_input("T_max (timesteps)", min_value=10, value=500, step=50)
    n_trials = st.number_input("Trials per ρ", min_value=10, value=250, step=50)

    st.divider()
    st.subheader("Live ρ slider (single run)")
    rho_live = st.slider("ρ = μ / α", min_value=0.2, max_value=2.0, value=1.0, step=0.05)

    seed = st.number_input("Seed", min_value=0, value=0, step=1)

st.session_state["sigma"] = float(sigma)

# =====================
# Live single trajectory
# =====================

st.subheader("Live trajectory (single trial)")

mu_live = float(rho_live) * float(alpha)
rng = np.random.default_rng(int(seed))

collapsed, t_c, direction, traj = run_trial(
    mu=mu_live,
    alpha=float(alpha),
    k=float(k),
    sigma=float(sigma),
    theta=float(theta),
    t_max=int(t_max),
    rng=rng
)

fig1 = plt.figure()
t = np.arange(traj.size)
plt.plot(t, traj)
plt.axhline(float(theta), linestyle="--")
plt.axhline(-float(theta), linestyle="--")

if collapsed:
    plt.axvline(int(t_c), linestyle="--")

plt.xlabel("t (steps)")
plt.ylabel("x_t (stability margin)")
plt.title(f"ρ={rho_live:.2f} — " + ("COLLAPSED" if collapsed else "SURVIVED"))

st.pyplot(fig1)
plt.close(fig1)

# =====================
# Sweep block
# =====================

st.subheader("Sweep (collapse probability + conditional time-to-collapse)")

run_sweep = st.button("Run sweep over ρ ∈ [0.2, 2.0]")

if run_sweep:

    rho_grid = np.round(np.arange(0.2, 2.0 + 1e-9, 0.05), 2)

    out = sweep_rho(
        rho_grid=rho_grid,
        alpha=float(alpha),
        k=float(k),
        theta=float(theta),
        sigma=float(sigma),
        t_max=int(t_max),
        n_trials=int(n_trials),
        seed=int(seed),
    )

    fig2 = plt.figure()
    plt.plot(out.rho_grid, out.collapse_prob, marker="o")
    plt.xlabel("ρ = μ / α")
    plt.ylabel(f"P(collapse by T_max={out.t_max})")
    plt.title("Collapse Probability vs ρ")
    plt.ylim(-0.05, 1.05)
    st.pyplot(fig2)
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(out.rho_grid, out.mean_ttc_conditional, marker="o")
    plt.xlabel("ρ = μ / α")
    plt.ylabel("E[t_collapse | collapse] (steps)")
    plt.title("Mean Time-to-Collapse (Conditional)")
    st.pyplot(fig3)
    plt.close(fig3)

    st.subheader("Directional collapse counts")
    st.write({
        "positive_direction": int(out.collapse_pos_count.sum()),
        "negative_direction": int(out.collapse_neg_count.sum())
    })
