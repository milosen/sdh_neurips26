# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
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

from buffers import CostReplayBuffer
from utils import make_env, SummaryWriter


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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
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
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""

    cost_limit: float = 25.0
    kp: float = 0.01
    ki: float = 0.01
    kd: float = 0.1


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@dataclass
class PIDConfig:
    kp: float
    ki: float
    kd: float

    # Optional stability helpers
    integral_limit: Optional[float] = 10.   # anti-windup
    output_min: float = 0.0                  # λ ≥ 0
    output_max: Optional[float] = 10.

    # derivative smoothing (important in RL noise)
    smoothing: float = 0.0  # 0=no smoothing, ~0.9 strong


class PIDController:
    """
    PID controller specialized for CMDP Lagrange multipliers.

    Usage:
        pid = PIDController(target=d, init_value=0.0, config=PIDConfig(...))

        lambda_t = pid.update(measured_constraint)
    """

    def __init__(
        self,
        target: float,
        config: PIDConfig,
    ):
        self.target = target
        self.cfg = config

        # internal state
        self.integral = 0.0
        self.error = 0.0
        self.derivative = 0.0

    def update(self, cost_return: float) -> float:
        """
        Update PID controller.

        Parameters
        ----------
        measurement : float
            Observed constraint value J_C(π)

        Returns
        -------
        float
            Updated lambda (dual variable)
        """

        # constraint error
        error = cost_return - self.target

        # ----- Integral term -----
        integral = np.clip(self.integral + error, 0, self.cfg.integral_limit).item() # integral_limit for anti-windup
        self.integral = self.cfg.smoothing * self.integral + (1 - self.cfg.smoothing) * integral

        # ----- Derivative term -----
        derivative = np.clip(error - self.error, 0, None).item()
        self.derivative = self.cfg.smoothing * self.derivative + (1 - self.cfg.smoothing) * derivative

        self.error = error

        # ----- PID update -----
        pid_lambda = self.cfg.kp * error + self.cfg.ki * self.integral + self.cfg.kd * derivative

        # ----- projection -----
        pid_lambda = np.clip(pid_lambda, self.cfg.output_min, self.cfg.output_max).item()

        return pid_lambda


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
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
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/sac_pid/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = SafetyAsyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    eval_envs = SafetyAsyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    qfc = SoftQNetwork(envs).to(device)
    qfc_target = SoftQNetwork(envs).to(device)
    qc_optimizer = optim.Adam(list(qfc.parameters()), lr=args.q_lr)

    pid = PIDController(
        target=args.cost_limit,
        config=PIDConfig(
            kp=0.005,      
            ki=0.0005,   
            kd=0.05,     
            integral_limit=5.0, 
            output_max=5.0,
            smoothing=0.5,
        ),
    )

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

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

    ep_cost = 0
    lambda_c = 0

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, costs, terminations, truncations, infos = envs.step(actions)
        #costs = infos["cost"] if "cost" in infos else np.array([info["cost"] for info in infos["final_info"]])

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ep_cost = info["episode"]["c"]
                    lambda_c = pid.update(ep_cost.item())
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_cost", info["episode"]["c"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos, costs)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                qfc_next_target = qfc_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                next_qc_value = data.costs.flatten() + (1 - data.dones.flatten()) * args.gamma * qfc_next_target.view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            qfc_a_values = qfc(data.observations, data.actions).view(-1)
            qfc_loss = F.mse_loss(qfc_a_values, next_qc_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            qc_optimizer.zero_grad()
            qfc_loss.backward()
            qc_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    qfc_pi = qfc(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi + lambda_c*qfc_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qfc.parameters(), qfc_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                writer.dump_csv()

        eval_every = 5000
        if (global_step + 1) % eval_every == 0:
            eval_nb = 10
            eval_obs, _ = eval_envs.reset(seed=args.seed)
            eval_episodic_return = np.zeros((eval_nb,))
            eval_episodic_cost_return = np.zeros((eval_nb,))
            eval_episodic_length = np.zeros((eval_nb,))
            for e in range(eval_nb):
                eval_done = False
                while not eval_done:
                    with torch.no_grad():
                        taus, _ = actor(torch.Tensor(eval_obs).to(device))
                        taus = taus.cpu()

                    eval_obs, _, _, eval_terminated, eval_truncated, eval_infos = eval_envs.step(taus.numpy().clip(-1, 1))
                    eval_done = np.logical_or(eval_terminated, eval_truncated)

                    if "final_info" in eval_infos:
                        for eval_info in eval_infos["final_info"]:
                            # Skip the envs that are not done
                            if eval_info is None:
                                continue

                            print(f"eval={e}, episodic_return={eval_info['episode']['r']}")
                            eval_episodic_return[e] = eval_info["episode"]["r"]
                            eval_episodic_cost_return[e] = eval_info["episode"]["c"]
                            eval_episodic_length[e] = eval_info["episode"]["l"]
            writer.add_scalar("evaluation/episodic_return", eval_episodic_return.mean(), global_step)
            writer.add_scalar("evaluation/episodic_cost", eval_episodic_cost_return.mean(), global_step)
            writer.add_scalar("evaluation/episodic_length", eval_episodic_length.mean(), global_step)
            writer.dump_csv()

    envs.close()
    writer.close()
