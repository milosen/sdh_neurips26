# Portions of the code are adapted from Safety Starter Agents and Spinning Up, released by OpenAI under the MIT license.
# WCSAC: Worst-Case Soft Actor-Critic with CVaR constraint, implemented in the CleanRL style of sac_pid.py
import csv
import os
import random
import time
from dataclasses import dataclass
from typing import NamedTuple, Optional

from safety_gymnasium import wrappers
import gymnasium as gym
import numpy as np
from safety_gymnasium.vector import SafetyAsyncVectorEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from scipy.stats import norm

from buffers import CostReplayBuffer, ReplayBuffer, ReplayBufferSamples
from utils import SummaryWriter, make_env


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SafetyHalfCheetahVelocity-v1"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.0
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""

    # WCSAC-specific arguments
    cost_limit: float = 25.0
    """undiscounted episode cost limit"""
    cl: float = 0.9
    """CVaR confidence level (between 0 and 1); lower = more risk-averse"""
    beta_lr: float = 5e-4
    """learning rate for the dual variable (Lagrange multiplier) beta"""
    max_ep_len: int = 1000
    """maximum episode length, used to convert cost_limit to discounted constraint"""
    damp_scale: float = 0.0
    """scale for the damping term in the policy loss (0 = disabled)"""

# ──────────────────────────────────────────────────────────────────────────────
# Networks  (same architecture as sac_pid.py)
# ──────────────────────────────────────────────────────────────────────────────

class SoftQNetwork(nn.Module):
    """Standard Q-network shared by reward critics and the mean cost critic."""
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        act_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CostVarianceNetwork(nn.Module):
    """
    Variance critic V[G_c | s, a].
    Output is passed through softplus so it is always positive.
    """
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        act_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # softplus ensures variance estimate > 0
        return F.softplus(self.fc3(x))


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """Squashed Gaussian actor — identical to sac_pid.py."""
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        act_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/sac_wcsac/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # ── Seeding ──────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environments ─────────────────────────────────────────────────────────
    envs = SafetyAsyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    eval_envs = SafetyAsyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # ── CVaR constant  pdf_cdf = φ(Φ⁻¹(cl)) / cl  ───────────────────────────
    # This scalar converts std-dev of cost return into the CVaR excess above the mean.
    pdf_cdf = float(norm.pdf(norm.ppf(args.cl)) / args.cl)

    # Discounted cost constraint (from undiscounted cost_limit)
    cost_constraint = (
        args.cost_limit
        * (1 - args.gamma ** args.max_ep_len)
        / (1 - args.gamma)
        / args.max_ep_len
    )
    print(f"CVaR pdf/cdf factor: {pdf_cdf:.4f}")
    print(f"Discounted cost constraint: {cost_constraint:.4f}")

    # ── Networks ─────────────────────────────────────────────────────────────
    actor = Actor(envs).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    # Reward critics (clipped double-Q)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)

    # Cost mean critic
    qfc = SoftQNetwork(envs).to(device)
    qfc_target = SoftQNetwork(envs).to(device)
    qfc_target.load_state_dict(qfc.state_dict())
    qc_optimizer = optim.Adam(qfc.parameters(), lr=args.q_lr)

    # Cost variance critic  (NEW in WCSAC)
    qfc_var = CostVarianceNetwork(envs).to(device)
    qfc_var_target = CostVarianceNetwork(envs).to(device)
    qfc_var_target.load_state_dict(qfc_var.state_dict())
    qc_var_optimizer = optim.Adam(qfc_var.parameters(), lr=args.q_lr)

    # Lagrange multiplier beta (log-parameterised, projected positive via softplus)
    log_beta = torch.zeros(1, requires_grad=True, device=device)
    beta_optimizer = optim.Adam([log_beta], lr=args.beta_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # ── Replay Buffer ─────────────────────────────────────────────────────────
    envs.single_observation_space.dtype = np.float32
    rb = CostReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    ep_cost = 0.0

    # ── Main training loop ───────────────────────────────────────────────────
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):

        # ── Action selection ──────────────────────────────────────────────────
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # ── Environment step ──────────────────────────────────────────────────
        next_obs, rewards, costs, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ep_cost = info["episode"]["c"]
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_cost={ep_cost}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_cost", ep_cost, global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # ── Store transition ─────────────────────────────────────────────────
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos, costs)
        obs = next_obs

        # ── Learning ──────────────────────────────────────────────────────────
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            beta = F.softplus(log_beta).item()   # always positive

            # ── Critic update ────────────────────────────────────────────────
            with torch.no_grad():
                next_actions, next_log_pi, _ = actor.get_action(data.next_observations)

                # Reward targets (clipped double-Q + entropy)
                qf1_next = qf1_target(data.next_observations, next_actions)
                qf2_next = qf2_target(data.next_observations, next_actions)
                min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next.view(-1)

                # Cost mean target
                qfc_next = qfc_target(data.next_observations, next_actions)
                next_qc_value = data.costs.flatten() + (1 - data.dones.flatten()) * args.gamma * qfc_next.view(-1)

                # Cost variance target
                # Var[G_c] = c² + 2γ·c·E[G_c'] + γ²·(Var[G_c'] + E[G_c']²) − E[G_c]²
                # We use current qfc as E[G_c] (already computed before the update).
                qfc_var_next = qfc_var_target(data.next_observations, next_actions)
                qfc_current = qfc(data.observations, data.actions)   # E[G_c] at current (s,a)
                c = data.costs.flatten()
                qc_next_mean = qfc_next.view(-1)
                next_qc_var_value = (
                    c ** 2
                    + 2 * args.gamma * c * qc_next_mean
                    + args.gamma ** 2 * (qfc_var_next.view(-1) + qc_next_mean ** 2)
                    - qfc_current.view(-1) ** 2
                )
                # Clamp to avoid numerical issues with the Bellman residual
                next_qc_var_value = next_qc_var_value.clamp(min=1e-8)

            # Reward Q-losses
            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), max_norm=1.0)
            q_optimizer.step()

            # Cost mean Q-loss
            qfc_a_values = qfc(data.observations, data.actions).view(-1)
            qfc_loss = F.mse_loss(qfc_a_values, next_qc_value)

            qc_optimizer.zero_grad()
            qfc_loss.backward()
            torch.nn.utils.clip_grad_norm_(qfc.parameters(), max_norm=1.0)
            qc_optimizer.step()

            # Cost variance Q-loss
            # We use the Itô-style loss from WCSAC paper:
            # L = 0.5 * mean( Var + target - 2*sqrt(Var * target) )
            # which is equivalent to 0.5 * mean( (sqrt(Var) - sqrt(target))² )
            qfc_var_a_values = qfc_var(data.observations, data.actions).view(-1).clamp(min=1e-6)
            qfc_var_loss = 0.5 * F.mse_loss(
                qfc_var_a_values.sqrt(),
                next_qc_var_value.sqrt(),
            )

            qc_var_optimizer.zero_grad()
            qfc_var_loss.backward()
            torch.nn.utils.clip_grad_norm_(qfc_var.parameters(), max_norm=1.0)
            qc_var_optimizer.step()

            # ── Actor update ─────────────────────────────────────────────────
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)

                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    qfc_pi = qfc(data.observations, pi).view(-1)
                    qfc_var_pi = qfc_var(data.observations, pi).view(-1).clamp(min=1e-6)

                    # CVaR cost estimate: mean + pdf_cdf * std
                    cvar_cost = (qfc_pi + pdf_cdf * qfc_var_pi.sqrt()).clamp(max=1e3)

                    # Optional damping term (penalises constraint slack in policy loss)
                    if args.damp_scale > 0.0:
                        with torch.no_grad():
                            qfc_mean_val = qfc(data.observations, data.actions).view(-1)
                            qfc_var_val = qfc_var(data.observations, data.actions).view(-1).clamp(min=1e-6)
                        damp = args.damp_scale * (cost_constraint - qfc_mean_val - pdf_cdf * qfc_var_val.sqrt()).mean()
                        effective_beta = max(beta - damp.item(), 0.0)
                    else:
                        effective_beta = beta

                    actor_loss = (alpha * log_pi - min_qf_pi + effective_beta * cvar_cost).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                    actor_optimizer.step()

                    # Entropy auto-tuning
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # ── Beta (Lagrange multiplier) update ────────────────────────────
            # beta_loss = beta * (cost_constraint − CVaR_cost)
            # We maximise over beta ↔ minimise its negation.
            with torch.no_grad():
                qfc_det = qfc(data.observations, data.actions).view(-1)
                qfc_var_det = qfc_var(data.observations, data.actions).view(-1).clamp(min=1e-6)
                cvar_det = qfc_det + pdf_cdf * qfc_var_det.sqrt()

            beta_loss = -F.softplus(log_beta) * (cost_constraint - cvar_det).mean()
            beta_optimizer.zero_grad()
            beta_loss.backward()
            torch.nn.utils.clip_grad_norm_([log_beta], max_norm=1.0)
            beta_optimizer.step()
            with torch.no_grad():
                log_beta.clamp_(min=-10.0, max=5.0)  # keeps beta in (~0, ~150)

            # ── Target network updates ────────────────────────────────────────
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qfc.parameters(), qfc_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qfc_var.parameters(), qfc_var_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % 1000 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/qfc_values", qfc_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qfc_loss", qfc_loss.item(), global_step)
                writer.add_scalar("losses/qfc_var_values", qfc_var_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qfc_var_loss", qfc_var_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/beta", F.softplus(log_beta).item(), global_step)
                writer.add_scalar("losses/beta_loss", beta_loss.item(), global_step)
                writer.add_scalar("losses/cvar_cost", cvar_det.mean().item(), global_step)
                writer.add_scalar("losses/cost_constraint", cost_constraint, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                writer.dump_csv()

        # ── Periodic evaluation ───────────────────────────────────────────────
        eval_every = 10000
        if (global_step + 1) % eval_every == 0:
            eval_nb = 10
            eval_obs, _ = eval_envs.reset(seed=args.seed)
            eval_episodic_return = np.zeros((eval_nb,))
            eval_episodic_cost = np.zeros((eval_nb,))
            eval_episodic_length = np.zeros((eval_nb,))

            for e in range(eval_nb):
                eval_done = False
                while not eval_done:
                    with torch.no_grad():
                        # Use the deterministic mean action for evaluation
                        _, _, det_action = actor.get_action(torch.Tensor(eval_obs).to(device))
                        det_action = det_action.cpu()

                    eval_obs, _, _, eval_terminated, eval_truncated, eval_infos = eval_envs.step(
                        det_action.numpy().clip(-1, 1)
                    )
                    eval_done = np.logical_or(eval_terminated, eval_truncated)

                    if "final_info" in eval_infos:
                        for eval_info in eval_infos["final_info"]:
                            if eval_info is None:
                                continue
                            print(f"eval={e}, episodic_return={eval_info['episode']['r']}, episodic_cost={eval_info['episode']['c']}")
                            eval_episodic_return[e] = eval_info["episode"]["r"]
                            eval_episodic_cost[e] = eval_info["episode"]["c"]
                            eval_episodic_length[e] = eval_info["episode"]["l"]

            writer.add_scalar("evaluation/episodic_return", eval_episodic_return.mean(), global_step)
            writer.add_scalar("evaluation/episodic_cost", eval_episodic_cost.mean(), global_step)
            writer.add_scalar("evaluation/episodic_length", eval_episodic_length.mean(), global_step)
            writer.dump_csv()

    envs.close()
    writer.close()