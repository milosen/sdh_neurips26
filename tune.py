"""Hyperparameter tuning for SAC and MPO with SDH using Optuna.

Usage:
    python tune.py --algo sac --n-trials 50 --env-id SafetyPointGoal1-v0
    python tune.py --algo mpo --n-trials 50 --env-id SafetyPointGoal1-v0
"""
import argparse
import glob
import os
import subprocess
import sys

import optuna
from optuna.samplers import TPESampler


RUNS_DIR = "runs/ten_mil"
TUNE_TIMESTEPS = 300_000


def read_repeated_header_csv(path: str) -> list[dict]:
    """Parse CSVs where dump_csv() writes a header + row on every call."""
    rows = []
    fieldnames = None
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if parts[0] == "step":
                fieldnames = parts
                continue
            if fieldnames is None:
                continue
            row = dict(zip(fieldnames, parts))
            rows.append(row)
    return rows


def objective_from_logs(run_dir: str, cost_limit: float) -> float:
    csv_path = os.path.join(run_dir, "logs.csv")
    if not os.path.exists(csv_path):
        raise optuna.TrialPruned()

    rows = read_repeated_header_csv(csv_path)
    eval_rows = [r for r in rows if "evaluation/episodic_return" in r and r["evaluation/episodic_return"]]
    if len(eval_rows) < 3:
        raise optuna.TrialPruned()

    last = eval_rows[-3:]
    avg_return = sum(float(r["evaluation/episodic_return"]) for r in last) / len(last)
    avg_cost = sum(float(r["evaluation/episodic_cost"]) for r in last) / len(last)
    penalty = max(0.0, avg_cost - cost_limit)
    return avg_return - 100*penalty


def run_trial(algo: str, env_id: str, seed: int, trial_num: int, extra_args: list[str]) -> str:
    exp_name = f"tune_{trial_num:04d}"
    cmd = [
        sys.executable, f"{algo}.py",
        "--env-id", env_id,
        "--exp-name", exp_name,
        "--seed", str(seed),
        "--total-timesteps", str(TUNE_TIMESTEPS),
        "--cuda", "True",
        "--track", "False",
        "--capture-video", "False",
    ] + extra_args

    print(f"[trial {trial_num}] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        raise optuna.TrialPruned()

    pattern = os.path.join(RUNS_DIR, f"{env_id}__{exp_name}__{seed}__*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise optuna.TrialPruned()
    return matches[-1]


def suggest_sac(trial: optuna.Trial) -> list[str]:
    cost_lambda_start = trial.suggest_float("cost_lambda_start", 0.0, 0.5)
    cost_lambda_end = trial.suggest_float("cost_lambda_end", cost_lambda_start, 2.0)
    alive_reward = trial.suggest_float("alive_reward", 0.0, 1.0, log=False)
    sdh_dual_update = trial.suggest_categorical("sdh_dual_update", [True, False])
    alpha = trial.suggest_float("alpha", 0.001, 0.5, log=True)
    q_lr = trial.suggest_float("q_lr", 1e-4, 1e-3, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.02, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    return [
        "--cost-lambda-start", str(cost_lambda_start),
        "--cost-lambda-end", str(cost_lambda_end),
        "--alive-reward", str(alive_reward),
        "--sdh-dual-update", str(sdh_dual_update),
        "--alpha", str(alpha),
        "--q-lr", str(q_lr),
        "--tau", str(tau),
        "--batch-size", str(batch_size),
    ]


def suggest_mpo(trial: optuna.Trial) -> list[str]:
    cost_lambda_start = trial.suggest_float("cost_lambda_start", 0.0, 0.5)
    cost_lambda_end = trial.suggest_float("cost_lambda_end", cost_lambda_start, 2.0)
    alive_reward = trial.suggest_float("alive_reward", 0.0, 1.0)
    sdh_dual_update = trial.suggest_categorical("sdh_dual_update", [True, False])
    policy_q_lr = trial.suggest_float("policy_q_lr", 5e-5, 5e-4, log=True)
    dual_lr = trial.suggest_float("dual_lr", 1e-3, 1e-1, log=True)
    epsilon_non_parametric = trial.suggest_float("epsilon_non_parametric", 0.001, 0.1, log=True)
    epsilon_parametric_mu = trial.suggest_float("epsilon_parametric_mu", 0.0005, 0.05, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    return [
        "--cost-lambda-start", str(cost_lambda_start),
        "--cost-lambda-end", str(cost_lambda_end),
        "--alive-reward", str(alive_reward),
        "--sdh-dual-update", str(sdh_dual_update),
        "--policy-q-lr", str(policy_q_lr),
        "--dual-lr", str(dual_lr),
        "--epsilon-non-parametric", str(epsilon_non_parametric),
        "--epsilon-parametric-mu", str(epsilon_parametric_mu),
        "--batch-size", str(batch_size),
    ]


def make_objective(algo: str, env_id: str, cost_limit: float, seed: int):
    trial_counter = [0]

    def objective(trial: optuna.Trial) -> float:
        n = trial_counter[0]
        trial_counter[0] += 1

        if algo == "sac":
            extra_args = suggest_sac(trial)
        elif algo == "mpo":
            extra_args = suggest_mpo(trial)
        else:
            raise ValueError(f"Unknown algo: {algo}")

        extra_args += ["--cost-limit", str(cost_limit)]
        run_dir = run_trial(algo, env_id, seed, n, extra_args)
        return objective_from_logs(run_dir, cost_limit)

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["sac", "mpo"], required=True)
    parser.add_argument("--env-id", default="SafetyPointGoal1-v0")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--cost-limit", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel Optuna workers")
    args = parser.parse_args()

    storage = f"sqlite:///tune_{args.algo}.db"
    study_name = f"{args.algo}_{args.env_id}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        load_if_exists=True,
    )
    study.optimize(
        make_objective(args.algo, args.env_id, args.cost_limit, args.seed),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )

    print("\n=== Best trial ===")
    best = study.best_trial
    print(f"  value: {best.value:.4f}")
    print(f"  params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
