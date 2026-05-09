import os
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from algos.common.violation_depth_profiler import *

ENVS = {
    "SafetyAntVelocity":  "runs/profiler/SafetyAntVelocity-v1__sac_profiler__0__1777669630",
    "SafetyCarButton":   "runs/profiler/SafetyCarButton1-v0__sac_profiler__0__1777669304",
    "SafetyRacecarCircle": "runs/profiler/SafetyRacecarCircle1-v0__sac_profiler__0__1777719082",
}
STEPS = {
    "SafetyAntVelocity":  [19999, 99999, 499999],
    "SafetyCarButton":   [19999, 99999, 499999],
    "SafetyRacecarCircle": [19999, 99999, 499999]
}
COLORS = {"SafetyAntVelocity": "tab:blue", "SafetyCarButton": "tab:orange", "SafetyRacecarCircle": "tab:green"}

MAX_DEPTH = 30

n_envs = len(ENVS)
n_steps = max(len(s) for s in STEPS.values())
fig, axes = plt.subplots(n_envs, n_steps, figsize=(5 * n_steps, 4 * n_envs), sharey='row')

for row, (env_name, run_dir) in enumerate(ENVS.items()):
    for col, step in enumerate(STEPS[env_name]):
        ax = axes[row, col]
        if not os.path.exists(f"{run_dir}/profiler_step_{step}.pkl"):
            continue
        with open(f"{run_dir}/profiler_step_{step}.pkl", "rb") as f:
            profiler = pickle.load(f)
        profiler.plot(ax=ax, label=env_name, color=COLORS[env_name], show_fit=True, b_max=MAX_DEPTH)
        if row == 0:
            ax.set_title(f"Step {step + 1}")
        if col == 0:
            ax.set_ylabel(f"{env_name}\n$\\Omega^\\pi(b)$")
        else:
            ax.set_ylabel("")

plt.suptitle("Violation depth profile over training", y=1.02)
plt.tight_layout()
plt.savefig("violation_profiles_b30.png", dpi=150, bbox_inches='tight')
plt.show()