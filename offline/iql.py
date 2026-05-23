"""
Implicit Q-Learning (IQL) – unconstrained offline RL baseline.

Trained on DSRL (Datasets for Safe RL) offline datasets and evaluated online
against Safety-Gymnasium environments. Costs are recorded during evaluation but
not optimised, making this a pure reward-maximisation baseline.

Reference:  Kostrikov et al., "Offline RL with Implicit Q-Learning", ICLR 2022
Datasets:   https://www.offline-saferl.org/
Install:    pip install dsrl

Algorithm (one gradient step per iteration):
  V  ← expectile regression:  L_V  = E[L_τ(stop_grad(Q̂_min) - V(s))]
  Q  ← TD regression:         L_Q  = E[(r + γ(1−done)·V(s') − Q(s,a))²]  (×2)
  π  ← advantage-weighted BC: L_π  = E[exp(β·stop_grad(A)) · ‖π(s)−a‖²]
  Q̂  ← Polyak EMA of Q

L_τ(u) = |τ − 𝟙(u<0)| · u²   (asymmetric L2, estimates τ-expectile of Q)
A(s,a) = Q̂_min(s,a) − V(s)    (advantage estimate)
done   = terminal flag only;  timeouts do NOT block bootstrapping
"""
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass

try:
    import dsrl  # noqa: F401 – registers DSRL offline environments with gymnasium
except ImportError as _e:
    raise ImportError(
        "dsrl is not installed. Install the offline extras:\n"
        "  uv sync --extra offline\n"
        "  # or: pip install 'sdh[offline]'  /  pip install dsrl\n"
        "Note: dsrl pulls in pybullet (C++ build) for BulletSafetyGym support."
    ) from _e
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from algos.common.utils import SummaryWriter


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """name of this experiment (used in run directory)"""
    seed: int = 0
    """random seed"""
    cuda: bool = True
    """enable CUDA"""
    torch_deterministic: bool = True
    """set torch.backends.cudnn.deterministic"""

    # Environment
    env_id: str = "OfflineAntVelocityGymnasium-v0"
    """DSRL offline environment ID.
    Full list: https://github.com/liuzuxin/DSRL"""

    # Training schedule
    total_steps: int = 1_000_000
    """total number of gradient update steps"""
    batch_size: int = 256
    """minibatch size sampled from the offline dataset"""
    log_freq: int = 1_000
    """how often to write training scalars (steps)"""
    eval_freq: int = 20_000
    """how often to run online evaluation (steps)"""
    eval_episodes: int = 10
    """number of episodes per evaluation"""

    # IQL hyperparameters
    gamma: float = 0.99
    """discount factor"""
    expectile: float = 0.7
    """V-network expectile τ (higher → more optimistic upper bound on Q).
    Paper uses 0.7 for locomotion, 0.9 for AntMaze."""
    beta: float = 3.0
    """AWR temperature β for the actor update.
    Higher β → more greedy extraction; paper uses 3.0 for locomotion."""
    awr_clip: float = 100.0
    """hard cap on advantage weights exp(β·A) to prevent instability"""
    lr: float = 3e-4
    """learning rate shared by V, Q and actor optimisers"""
    tau: float = 0.005
    """Polyak EMA rate for target Q networks (same convention as as_sac.py)"""

    # Network architecture
    hidden_dim: int = 256
    n_layers: int = 2

    # Dataset pre-processing
    normalize_reward: bool = False
    """shift & scale rewards to [0, 1] using dataset min/max"""


# ── Networks ──────────────────────────────────────────────────────────────────

def build_mlp(in_dim: int, out_dim: int, hidden_dim: int = 256,
              n_layers: int = 2) -> nn.Sequential:
    """MLP with LayerNorm + Tanh on hidden layers, plain Linear on the output."""
    dims = [in_dim] + [hidden_dim] * n_layers + [out_dim]
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.extend([nn.LayerNorm(b), nn.Tanh()])
    return nn.Sequential(*layers)


class VNet(nn.Module):
    """State-value function V(s) → ℝ."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.net = build_mlp(obs_dim, 1, hidden_dim, n_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class TwinQ(nn.Module):
    """Twin Q-network Q₁, Q₂(s, a) → ℝ (clipped double Q)."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.q1 = build_mlp(obs_dim + act_dim, 1, hidden_dim, n_layers)
        self.q2 = build_mlp(obs_dim + act_dim, 1, hidden_dim, n_layers)

    def both(self, obs: torch.Tensor,
             act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.both(obs, act)
        return torch.minimum(q1, q2)


class Actor(nn.Module):
    """Deterministic actor π(s) → a ∈ (−1, 1)^d (tanh-squashed).

    The AWR loss (MSE to dataset actions weighted by advantages) works with
    deterministic actors and avoids the log-prob mode-seeking issue of
    stochastic AWR on out-of-distribution actions.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.net = build_mlp(obs_dim, act_dim, hidden_dim, n_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(env_id: str, device: torch.device,
                 normalize_reward: bool = False) -> tuple[dict, gym.Space, gym.Space]:
    """Load a DSRL offline dataset into pre-allocated GPU/CPU tensors."""
    env = gym.make(env_id)
    raw = env.get_dataset()
    env.close()

    def t(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(arr, dtype=dtype, device=device)

    obs  = t(raw["observations"])
    act  = t(raw["actions"])
    rew  = t(raw["rewards"])
    nobs = t(raw["next_observations"])
    cost = t(raw["costs"])
    # Only true episode terminations block bootstrapping; timeouts do not.
    done = t(raw["terminals"])

    if normalize_reward:
        rew = (rew - rew.min()) / (rew.max() - rew.min() + 1e-8)

    print(f"[dataset] {obs.shape[0]:,} transitions | "
          f"obs_dim={obs.shape[1]} act_dim={act.shape[1]} | "
          f"r∈[{rew.min():.3f}, {rew.max():.3f}] "
          f"c∈[{cost.min():.3f}, {cost.max():.3f}]")

    data = dict(obs=obs, act=act, rew=rew, nobs=nobs, cost=cost, done=done)
    env2 = gym.make(env_id)   # re-open to read spaces
    obs_space, act_space = env2.observation_space, env2.action_space
    env2.close()
    return data, obs_space, act_space


def sample_batch(data: dict, batch_size: int) -> dict:
    idx = torch.randint(0, data["obs"].shape[0], (batch_size,))
    return {k: v[idx] for k, v in data.items()}


# ── IQL loss helpers ──────────────────────────────────────────────────────────

def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """L_τ(u) = |τ − 𝟙(u < 0)| · u²  (asymmetric L2 expectile regression)."""
    weight = torch.where(diff >= 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(actor: Actor, env_id: str,
             n_episodes: int, device: torch.device) -> tuple[float, float]:
    """Roll out the deterministic policy online and return mean return & cost."""
    env = gym.make(env_id)
    actor.eval()
    returns, costs = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret = ep_cost = 0.0
        done = False
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                action = actor(obs_t)[0].cpu().numpy()
            obs, rew, terminated, truncated, info = env.step(action)
            ep_ret  += float(rew)
            ep_cost += float(info.get("cost", 0.0))
            done = terminated or truncated
        returns.append(ep_ret)
        costs.append(ep_cost)
    env.close()
    actor.train()
    return float(np.mean(returns)), float(np.mean(costs))


# ── Training ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    log_dir  = f"offline/runs/{run_name}"
    writer   = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    data, obs_space, act_space = load_dataset(args.env_id, device,
                                              args.normalize_reward)
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]

    # ── Networks ──────────────────────────────────────────────────────────────
    vnet   = VNet(obs_dim,          args.hidden_dim, args.n_layers).to(device)
    qnet   = TwinQ(obs_dim, act_dim, args.hidden_dim, args.n_layers).to(device)
    qnet_t = TwinQ(obs_dim, act_dim, args.hidden_dim, args.n_layers).to(device)
    qnet_t.load_state_dict(qnet.state_dict())
    for p in qnet_t.parameters():
        p.requires_grad_(False)
    actor  = Actor(obs_dim, act_dim, args.hidden_dim, args.n_layers).to(device)

    v_opt = torch.optim.Adam(vnet.parameters(),  lr=args.lr)
    q_opt = torch.optim.Adam(qnet.parameters(),  lr=args.lr)
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

    start_time = time.time()

    for step in range(1, args.total_steps + 1):
        b = sample_batch(data, args.batch_size)

        # ── V update: expectile regression onto stop-grad(Q̂_min) ─────────────
        with torch.no_grad():
            q_min = qnet_t.min(b["obs"], b["act"])
        v      = vnet(b["obs"])
        v_loss = expectile_loss(q_min - v, args.expectile)

        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()

        # ── Q update: TD regression with V(s') as bootstrap target ───────────
        with torch.no_grad():
            v_next    = vnet(b["nobs"])
            td_target = b["rew"] + args.gamma * (1.0 - b["done"]) * v_next
        q1, q2 = qnet.both(b["obs"], b["act"])
        q_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        # ── Actor update: advantage-weighted behavioural cloning ──────────────
        with torch.no_grad():
            adv     = qnet_t.min(b["obs"], b["act"]) - vnet(b["obs"])
            weights = (args.beta * adv).exp().clamp(max=args.awr_clip)
        pred   = actor(b["obs"])
        a_loss = (weights * (pred - b["act"]).pow(2).sum(-1)).mean()

        a_opt.zero_grad()
        a_loss.backward()
        a_opt.step()

        # ── Target Q Polyak update ────────────────────────────────────────────
        for p_src, p_tgt in zip(qnet.parameters(), qnet_t.parameters()):
            p_tgt.data.copy_(args.tau * p_src.data + (1 - args.tau) * p_tgt.data)

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.log_freq == 0:
            writer.add_scalar("losses/v_loss",     v_loss.item(),      step)
            writer.add_scalar("losses/q_loss",     q_loss.item() / 2,  step)
            writer.add_scalar("losses/actor_loss", a_loss.item(),      step)
            writer.add_scalar("losses/mean_v",     v.mean().item(),    step)
            writer.add_scalar("losses/mean_q_min", q_min.mean().item(), step)
            writer.add_scalar("losses/mean_adv",   adv.mean().item(),  step)
            writer.add_scalar("charts/SPS",
                              int(step / (time.time() - start_time)), step)
            writer.dump_csv()

        # ── Online evaluation ─────────────────────────────────────────────────
        if step % args.eval_freq == 0:
            ep_ret, ep_cost = evaluate(actor, args.env_id,
                                       args.eval_episodes, device)
            print(f"step={step:>7d}  return={ep_ret:.2f}  cost={ep_cost:.2f}")
            writer.add_scalar("evaluation/episodic_return", ep_ret,  step)
            writer.add_scalar("evaluation/episodic_cost",   ep_cost, step)
            writer.dump_csv()

    writer.close()
