"""Compare reward functions for a cart-pole swing-up task.

Goal state: x = 0 (cart centred), theta = 0 (pole upright).

Rewards:
  1. Cosine           — 4·(cos(θ)+1) / (x²+1) − 8  [Lorentzian, range [−8, 0]]
  2. Sum-sqr          — -(x² + θ²)                  [current CartPole-BT style]
  3. Sq. dist.        — -‖pole_tip − goal_tip‖²      [squared tip distance]
  4. Weighted sum-sqr — -(x² + L²·θ²)               [q2=L², matches sq. dist.]
"""

import os
import numpy as np
import matplotlib.pyplot as plt

L = 2.0  # pole length (m), matching CartPole-BT
N = 500  # number of plot points
X_MAX = 4.0  # full-range cart position limit (m)
THETA_MAX_DEG = 360.0  # full-range angle limit (degrees)

# --- Reward functions (higher = better; goal at x=0, theta=0) ---------------


def r_sumsqr(x, theta):
    """Negative sum of squared deviations from goal (original CartPole-BT)."""
    return -(x**2 + theta**2)


def r_sinsqr(x, theta):
    """Respects that theta is a circular variable."""
    return -(x**2) - 4 * L**2 * np.sin(theta / 2) ** 2


def r_weighted_sinsqr(x, theta):
    """Like r_sumsqr near origin; respects that theta is circular."""
    return -(x**2) - 4 * np.sin(theta / 2) ** 2


def r_sqrdist(x, theta):
    """Negative squared distance from pole tip to goal position."""
    tip_x = x + L * np.sin(theta)
    tip_y = L * np.cos(theta)
    return -(tip_x**2 + (tip_y - L) ** 2)


def r_weighted_sumsqr(x, theta):
    # q2 = L² matches the curvature of r_sqrdist at the origin
    return -(x**2) - L**2 * theta**2


rewards = {
    "Sum-squared diff.": r_sumsqr,
    "Weighted sin-squared": r_weighted_sinsqr,
    # "Sin-squared": r_sinsqr,
    # "Squared distance": r_sqrdist,
    # "Weighted sum-sqr. diff.": r_weighted_sumsqr,
}
colors = {
    "Sum-squared diff.": "tab:blue",
    "Weighted sin-squared": "tab:orange",
    "Sin-squared": "tab:green",
    "Squared distance": "tab:red",
    "Weighted sum-sqr. diff.": "tab:purple",
}

# --- Figure generation ------------------------------------------------------


def nice_degree_ticks(theta_max_deg, max_ticks=9):
    """Tick positions at round degree intervals within ±theta_max_deg."""
    nice_steps = [1, 2, 5, 10, 15, 30, 45, 90, 180]
    for step in nice_steps:
        if round(2 * theta_max_deg / step) + 1 <= max_ticks:
            break
    lo = np.ceil(-theta_max_deg / step) * step
    return np.arange(lo, theta_max_deg + 0.01 * step, step)


def make_figure(scale, out_path, ylim=(None, None)):
    x_max = X_MAX * scale
    theta_max_deg = THETA_MAX_DEG * scale
    theta_max = np.radians(theta_max_deg)

    def add_curves(ax, x_axis, x_vals, theta_vals, annot=None):
        for name, func in rewards.items():
            ax.plot(
                x_axis,
                func(x_vals, theta_vals),
                label=name,
                color=colors[name],
                lw=1.8,
            )
        ax.axvline(0.0, color="k", lw=0.6, ls=":", alpha=0.5)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        if annot:
            ax.text(
                0.5,
                0.5,
                annot,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.5,
                color="0.40",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.8
                ),
            )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey="row")
    fig.suptitle(
        "Reward function comparison\n"
        f"Pole length L = {L} m,  goal: x = 0, θ = 0 (upright)  |  "
        f"range: x ∈ [±{x_max} m],  θ ∈ [±{theta_max_deg:.0f}°]",
        fontsize=10,
    )

    # (i) theta varies, x = 0
    thetas = np.linspace(-theta_max, theta_max, N)
    ax = axes[0, 0]
    add_curves(ax, np.degrees(thetas), np.zeros(N), thetas)
    ax.set_xticks(nice_degree_ticks(theta_max_deg))
    ax.set_xlabel("θ (degrees)")
    ax.set_ylabel("Reward")
    ax.set_title("(i)  θ varies,  x = 0")

    # (ii) x varies, theta = 0
    xs = np.linspace(-x_max, x_max, N)
    ax = axes[0, 1]
    add_curves(ax, xs, xs, np.zeros(N))
    ax.set_xlabel("x (m)")
    ax.set_title("(ii)  x varies,  θ = 0")

    # (iii) both vary, same sign: x = z*x_max, theta = z*theta_max
    z = np.linspace(-1.0, 1.0, N)
    annot_iii = (
        f"x = $z$ × {x_max} m\nθ = $z$ × {{theta_max_deg:.0f}}°\n(same sign)"
    )
    ax = axes[1, 0]
    add_curves(ax, z, z * x_max, z * theta_max, annot=annot_iii)
    ax.set_xlabel("$z$")
    ax.set_ylabel("Reward")
    ax.set_title("(iii)  x and θ vary with same sign")

    # (iv) both vary, opposite sign: x = z*x_max, theta = -z*theta_max
    annot_iv = (
        f"x = $z$ × {x_max} m\n"
        f"θ = −$z$ × {theta_max_deg:.0f}°\n(opposite sign)"
    )
    ax = axes[1, 1]
    add_curves(ax, z, z * x_max, -z * theta_max, annot=annot_iv)
    ax.set_xlabel("$z$")
    ax.set_title("(iv)  x and θ vary with opposite sign")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


# --- Generate figures --------------------------------------------------------

make_figure(
    scale=30 / 360,
    out_path="results/compare_rewards_zoom.png",
    ylim=(-0.4, None),
)
make_figure(
    scale=1.0, out_path="results/compare_rewards.png", ylim=(-30, None)
)
plt.show()
