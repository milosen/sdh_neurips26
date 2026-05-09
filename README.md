# SDH: Stochastic Decision Horizons

Code accompanying the paper **"Stochastic Decision Horizons for Constrained Reinforcement Learning"** (NeurIPS 2026 submission).

SDH transforms a cost-constrained MDP into a modified MDP where each step's reward and continuation discount are attenuated by `α = exp(−λ·c)`, where `c` is the per-step cost and `λ` controls sharpness. This links early termination of the return to survival probability and to the *violation depth profile* of the environment and yields principled safety guarantees without explicit Lagrangian multipliers or episode termination.

## Algorithms

| Script | Description |
|---|---|
| `algos/as_sac.py` | **AS-SAC** — Absorbing-State Soft Actor-Critic (SDH applied to SAC, our method) |
| `algos/vt_mpo.py` | **VT-MPO** — Virtual-Termination Maximum a Posteriori Policy Optimisation with n-step returns (SDH applied to MPO, our method) |
| `algos/wcsac.py` | WCSAC baseline |
| `algos/sac_pid.py` | SAC-PID baseline |
| `algos/mpo_pid.py` | MPO-PID baseline |
| `algos/mpo.py` | Standard MPO baseline |

The core SDH logic lives in `sdh.py`. Environment wrappers, `make_env`, and logging utilities are in `utils.py`. Plot scripts are in `plots/`.

## Installation

Requires Python 3.10. We recommend [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Running experiments

All scripts accept `--env-id` and `--seed` flags. The main SDH hyperparameters are `--cost-lambda-start`, `--cost-lambda-end` (linearly scheduled over 500k steps), and `--alive-reward`.

**AS-SAC (our method):**
```bash
python algos/as_sac.py --env-id SafetyPointGoal1-v0 --seed 1
python algos/as_sac.py --env-id SafetyHalfCheetahVelocity-v1 --seed 1
```

**VT-MPO (our method):**
```bash
python algos/vt_mpo.py --env-id SafetyPointGoal1-v0 --seed 1
python algos/vt_mpo.py --env-id SafetyHalfCheetahVelocity-v1 --seed 1
```

**Baselines:**
```bash
python algos/wcsac.py   --env-id SafetyPointGoal1-v0 --seed 1
python algos/sac_pid.py --env-id SafetyPointGoal1-v0 --seed 1
python algos/mpo_pid.py --env-id SafetyPointGoal1-v0 --seed 1
python algos/mpo.py     --env-id SafetyPointGoal1-v0 --seed 1
```

TensorBoard logs and CSV files are written to `runs/`.

## Environments

Experiments use [Safety Gymnasium](https://safety-gymnasium.readthedocs.io/). The environments tested in the paper are:

- `SafetyPointGoal1-v0`
- `SafetyPointPush1-v0`
- `SafetyRacecarCircle1-v0`
- `SafetyCarButton1-v0`
- `SafetyHopperVelocity-v1`
- `SafetyHumanoidVelocity-v1`
- `SafetyHalfCheetahVelocity-v1`
- `SafetyAntVelocity-v1`

## Hyperparameter tuning

```bash
python tune.py --algo sac --n-trials 50 --env-id SafetyPointGoal1-v0
python tune.py --algo mpo --n-trials 50 --env-id SafetyPointGoal1-v0
```

## Violation depth profiling

The `ViolationDepthProfiler` class in `violation_depth_profiler.py` estimates the environment's violation depth profile Ω^π(b) from rollout data. AS-SAC saves profiler snapshots to `runs/profiler/` during training; `plot_violation_depth_profiles.py` generates the corresponding figures.

## Plotting

`plot_main.py` reproduces the main results figure from the paper using pre-computed IQM scores stored in `data/safety_gym/iqm.pkl`. The data file is available separately (see paper appendix for the download link).

## Cluster submission (SLURM / Apptainer)

The `mpcdf/` directory contains a template submit script for SLURM clusters using Apptainer. Adapt `mpcdf/submit_container.sh` to your cluster and container path, then use `submit_all_as_sac.sh` or `submit_all_sac.sh` to launch the full experimental sweep.

## License

MIT
