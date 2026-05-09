import pathlib
import pickle
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tueplots import bundles

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from algos.common.violation_depth_profiler import ViolationDepthProfiler

plt.rcParams.update(bundles.icml2024())
plt.rcParams["lines.dash_joinstyle"] = "round"
plt.rcParams["lines.dash_capstyle"] = "round"
plt.rcParams["lines.solid_joinstyle"] = "round"
plt.rcParams["lines.solid_capstyle"] = "round"
plt.rcParams["figure.constrained_layout.use"] = False
plt.rcParams["figure.autolayout"] = False

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
with open("data/safety_gym/iqm.pkl", "rb") as f:
    iqm_scores, iqm_cis, iqm_scores_env, iqm_cis_env = pickle.load(f)

# ---------------------------------------------------------------------------
# Style — using our defined color palette
# ---------------------------------------------------------------------------
algorithms = ["mpo_cat", "sac_cat_naive_tuning", "wcsac", "mpo", "mpo_pid"]

label_map = {
    "mpo_cat": "VT-MPO (ours)",
    "sac_cat_naive_tuning": "AS-SAC (ours)",
    "mpo": "MPO",
    "mpo_pid": "MPO-PID",
    "wcsac": "WCSAC",
}
color_map = {
    "mpo_cat": "#5e81b5",       # ourblue
    "sac_cat_naive_tuning": "#eb6235",  # ourred
    "mpo": "#e19c24",           # ourorange
    "mpo_pid": "#8778b3",       # ourviolet
    "wcsac": "#8fb032",         # ourgreen
}
marker_map = {
    "mpo_cat": "o",
    "sac_cat_naive_tuning": "x",
    "mpo": "s",
    "mpo_pid": "^",
    "wcsac": "+",
}

x = np.arange(0, 11) * 0.1

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_curves(means, intervals, ax):
    for algo in algorithms:
        c = color_map[algo]
        ax.plot(x, means[algo], label=label_map[algo],
                marker=marker_map[algo], markersize=2.5, linewidth=0.7,
                color=c, zorder=3)
        if intervals is not None:
            ax.fill_between(x, intervals[algo][0], intervals[algo][1],
                            alpha=0.2, color=c, zorder=2)
    ax.grid(True, linewidth=0.3, alpha=0.7)


# ---------------------------------------------------------------------------
# Violation depth profiles
# ---------------------------------------------------------------------------
PROFILER_STEP = 499999
profiler_dirs = {
    "SafetyAntVelocity":
        "runs/profiler/SafetyAntVelocity-v1__sac_profiler__0__1777669630",
    "SafetyCarButton":
        "runs/profiler/SafetyCarButton1-v0__sac_profiler__0__1777669304",
}
profiler_colors = {
    "SafetyAntVelocity": "#5e81b5",
    "SafetyCarButton":  "#eb6235",
}

fig, axs = plt.subplots(1, 4, figsize=(5.5, 2.0))

# --- Left group: Aggregate IQM curves ---
plot_curves(iqm_scores["Reward"], iqm_cis["Reward"], axs[0])
axs[0].set_title("\\small Reward")
axs[0].set_ylabel("IQM Average Return")

plot_curves(iqm_scores["Cost"], iqm_cis["Cost"], axs[1])
axs[1].set_title("\\small Cost")
axs[1].axhline(25, color="black", ls=(0, (3, 3)), lw=0.5, alpha=0.7, zorder=10)
axs[1].set_ylim(0, 130)

for ax in axs[:2]:
    ax.set_xlim(-0.02, 1.02)

# --- Right group: one panel per environment ---
subtitles = ["\\small Single-Scale", "\\small Multi-Scale"]
for ax, (env_name, run_dir), subtitle in zip(axs[2:], profiler_dirs.items(), subtitles):
    with open(f"{run_dir}/profiler_step_{PROFILER_STEP}.pkl", "rb") as f:
        profiler = pickle.load(f)
    profiler.plot(ax=ax, label=env_name, color=profiler_colors[env_name],
                  show_fit=True, log_scale=True, b_max=30)
    _, _, fit = profiler.estimate()
    tau = fit["tau_linear"]
    r2  = fit["r_squared_linear"]
    ax.set_title(f"{subtitle}\n$\\tau$={tau:.2f}")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.legend().remove()

# y-label on the leftmost right-group panel; push it outward so it clears axs[1]
axs[2].set_ylabel(r"$\Omega^\pi(b)$", labelpad=0)

plt.subplots_adjust(bottom=0.30, top=0.62, wspace=0.25, left=0.08, right=0.98,)

# Extra gap between the two groups: shift axs[2] and axs[3] right
GROUP_GAP = 0.06  # extra figure-fraction to add between axs[1] and axs[2]
for ax in axs[2:]:
    pos = ax.get_position()
    ax.set_position([pos.x0 + GROUP_GAP, pos.y0, pos.width, pos.height])

# ---------------------------------------------------------------------------
# Group titles with LaTeX curly brace
# ---------------------------------------------------------------------------
fig.canvas.draw()


def add_group_title(label, ax_left, ax_right):
    bb_l = ax_left.get_position()
    bb_r = ax_right.get_position()
    x_center = (bb_l.x0 + bb_r.x1) / 2
    fig_w = fig.get_size_inches()[0]
    span_inches = (bb_r.x1 - bb_l.x0) * fig_w
    span_cm = span_inches * 2.54

    brace_y = 0.74
    label_y = 0.80
    fig.text(x_center, brace_y,
             r"$\overbrace{\hspace{" + f"{span_cm:.1f}" + r"cm}}$",
             ha="center", va="bottom", fontsize=9)
    fig.text(x_center, label_y, label,
             ha="center", va="bottom", fontsize=9, fontweight="bold")


add_group_title("Aggregate Performance", axs[0], axs[1])
add_group_title("Violation Depth Profiles (SAC)", axs[2], axs[3])

# x-labels centered under each group
left_group_center = (axs[0].get_position().x0 + axs[1].get_position().x1) / 2
right_group_center = (axs[2].get_position().x0 + axs[3].get_position().x1) / 2

fig.text(left_group_center, 0.2, "Environment steps (millions)",
         ha="center", va="top")
fig.text(right_group_center, 0.2, "Violation depth $b$",
         ha="center", va="top")

from matplotlib.lines import Line2D

# Left group legend
legend_handles = [
    Line2D([0], [0], color=color_map[a], marker=marker_map[a], markersize=3,
           lw=0.7, label=label_map[a])
    for a in algorithms
] + [
    Line2D([0], [0], color="black", ls=(0, (3, 3)), lw=0.5, label="Constraint threshold"),
]
leg1 = fig.legend(handles=legend_handles, loc="lower center",
                  bbox_to_anchor=(left_group_center, -0.05), ncol=3)
fig.add_artist(leg1)  # keep it when the second legend is added

# Right group legend
right_handles = [
    Line2D([0], [0], color=profiler_colors[env], lw=1.2, label=env)
    for env in profiler_dirs
] + [
    Line2D([0], [0], color="gray", ls="--", lw=0.8, label="Exp. fit"),
]
fig.legend(handles=right_handles, loc="lower center",
           bbox_to_anchor=(right_group_center, -0.05), ncol=2)

plt.savefig("safety_gym.pdf", dpi=600, bbox_inches="tight")
print("Saved figures/safety_gym.pdf")
