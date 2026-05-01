"""Wrapper that tracks the cumulative rewards and episode lengths."""
import csv
import time
from collections import deque
from typing import Optional

from safety_gymnasium import wrappers
from torch.utils.tensorboard import SummaryWriter as TBSummaryWriter
import numpy as np
import gymnasium as gym
import numpy as np
from safety_gymnasium.bases.base_task import MechanismConf
MechanismConf.randomize_layout = False

import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper
from gymnasium.spaces import Box, Discrete


class FixedResetSeed(gym.Wrapper):
    """Terminates the episode when cumulative cost exceeds a budget.
 
    This creates a hard-budget viability task: the agent has a finite
    cost budget, and the episode ends when it is exhausted. The SDH
    theory predicts this should improve SDH performance relative to
    the standard (no-termination) Safety Gymnasium setup.
 
    Args:
        env: The base environment.
        budget: Maximum cumulative cost before termination.
        truncate: If True, set `truncated=True` instead of `terminated=True`
                  on budget exhaustion. Use True if the termination is an
                  artificial time limit rather than a genuine terminal state.
    """
 
    def __init__(self, env: gym.Env, seed: int = 0):
        super().__init__(env)
        self.seed = seed
 
    def reset(self, **kwargs):
        kwargs.update({"seed": self.seed})
        return self.env.reset(**kwargs)


class CumulativeCostObservation(gym.Wrapper):
    """Terminates the episode when cumulative cost exceeds a budget.
 
    This creates a hard-budget viability task: the agent has a finite
    cost budget, and the episode ends when it is exhausted. The SDH
    theory predicts this should improve SDH performance relative to
    the standard (no-termination) Safety Gymnasium setup.
 
    Args:
        env: The base environment.
        budget: Maximum cumulative cost before termination.
        truncate: If True, set `truncated=True` instead of `terminated=True`
                  on budget exhaustion. Use True if the termination is an
                  artificial time limit rather than a genuine terminal state.
    """
 
    def __init__(self, env: gym.Env, budget: float = 25.0, truncate: bool = False):
        super().__init__(env)
        self.budget = budget
        self.observation_space = Box(shape=(self.observation_space.shape[0] + 2,), low=-np.inf, high=np.inf)
        self._cumulative_cost = 0.0
 
    def reset(self, **kwargs):
        self._cumulative_cost = 0.0
        obs, info = self.env.reset(**kwargs)
        info["cumulative_cost"] = self._cumulative_cost
        info["cost_budget_remaining"] = max(0.0, self.budget - self._cumulative_cost)
        #for a in [obs, np.array(info["cumulative_cost"]), np.array(info["cost_budget_remaining"])]:
        #    print(a.shape)
        obs = np.concatenate([obs, np.array([info["cumulative_cost"]]), np.array([info["cost_budget_remaining"]])])
        return obs, info
 
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        #cost = info.get("cost", 0.0)
        self._cumulative_cost += cost
 
        # Track cumulative cost in info for logging
        info["cumulative_cost"] = self._cumulative_cost
        info["cost_budget_remaining"] = max(0.0, self.budget - self._cumulative_cost)
        
        obs = np.concatenate([obs, np.array([info["cumulative_cost"]]), np.array([info["cost_budget_remaining"]])])
 
        return obs, reward, cost, terminated, truncated, info


class RecordCostEpisodeStatistics(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards, costs, and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        gym.Wrapper.__init__(self, env)

        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_cost_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.cost_return_queue = deque(maxlen=deque_size)
        self.violation_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32
        )
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_cost_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_violations = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            costs,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_cost_returns += costs
        self.episode_violations += (costs > 0)
        self.episode_lengths += 1
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "c": np.where(dones, self.episode_cost_returns, 0.0),
                    "v": np.where(dones, self.episode_violations, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.cost_return_queue.extend(self.episode_cost_returns[dones])
            self.violation_queue.extend(self.episode_violations[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_cost_returns[dones] = 0
            self.episode_violations[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            costs,
            terminations,
            truncations,
            infos,
        )


class CumulativeCostTermination(gym.Wrapper):
    """Terminates the episode when cumulative cost exceeds a budget.
 
    This creates a hard-budget viability task: the agent has a finite
    cost budget, and the episode ends when it is exhausted. The SDH
    theory predicts this should improve SDH performance relative to
    the standard (no-termination) Safety Gymnasium setup.
 
    Args:
        env: The base environment.
        budget: Maximum cumulative cost before termination.
        truncate: If True, set `truncated=True` instead of `terminated=True`
                  on budget exhaustion. Use True if the termination is an
                  artificial time limit rather than a genuine terminal state.
    """
 
    def __init__(self, env: gym.Env, budget: float = 25.0, truncate: bool = False):
        super().__init__(env)
        self.budget = budget
        self.truncate = truncate
        self._cumulative_cost = 0.0
 
    def reset(self, **kwargs):
        self._cumulative_cost = 0.0
        return self.env.reset(**kwargs)
 
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        #cost = info.get("cost", 0.0)
        self._cumulative_cost += cost
 
        # Track cumulative cost in info for logging
        info["cumulative_cost"] = self._cumulative_cost
        info["cost_budget_remaining"] = max(0.0, self.budget - self._cumulative_cost)
 
        if self._cumulative_cost >= self.budget and not terminated:
            if self.truncate:
                truncated = True
            else:
                terminated = True
            info["cost_termination"] = True
 
        return obs, reward, cost, terminated, truncated, info
 
 
class StochasticCostTermination(gym.Wrapper):
    """Terminates probabilistically based on per-step cost severity.
 
    At each step with cost c > 0, the episode terminates with probability
    1 - exp(-beta * c). This makes the environment's true dynamics match
    the SDH generative model: the SDH with parameter lambda is then a
    *correct* model of the environment, not an approximation.
 
    The SDH theory predicts this should be the setting where the SDH
    performs best, since the method is exactly matched to the data-
    generating process.
 
    Args:
        env: The base environment.
        beta: Termination sharpness. Higher beta = more aggressive termination.
              When beta matches the SDH's lambda, the SDH is an exact model.
    """
 
    def __init__(self, env: gym.Env, beta: float = 1.0):
        super().__init__(env)
        self.beta = beta
        self._rng = np.random.default_rng()
 
    def reset(self, *, seed=None, **kwargs):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self.env.reset(seed=seed, **kwargs)
 
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        cost = info.get("cost", 0.0)
 
        if cost > 0.0 and not terminated:
            # Survival probability = exp(-beta * cost), matching SDH alpha
            survival_prob = np.exp(-self.beta * cost)
            if self._rng.random() > survival_prob:
                terminated = True
                info["cost_termination"] = True
                info["termination_cost"] = cost
 
        return obs, reward, cost, terminated, truncated, info
 
 
class InstantCostTermination(gym.Wrapper):
    """Terminates immediately on any nonzero cost.
 
    This is the lambda -> infinity limit of the SDH: any violation is
    fatal. Equivalent to StochasticCostTermination with beta -> infinity,
    or CumulativeCostTermination with budget = 0.
 
    This is the hardest setting for learning (shortest episodes in early
    training) but the strongest safety guarantee.
 
    Args:
        env: The base environment.
    """
 
    def __init__(self, env: gym.Env):
        super().__init__(env)
 
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        # cost = info.get("cost", 0.0)
 
        if cost > 0.0 and not terminated:
            terminated = True
            info["cost_termination"] = True
 
        return obs, reward, cost, terminated, truncated, info
 
 
# ---------------------------------------------------------------------------
# Fixed layout wrapper: pins hazard + goal positions, re-samples agent only
# ---------------------------------------------------------------------------

class FixedLayoutWrapper(gym.Wrapper):
    """Fixes hazard and goal positions while re-sampling only the agent spawn each reset.

    On the first reset the full random layout is built (optionally with a specific
    layout_seed so you can reproducibly pick a layout where hazards block the path).
    After that, every non-agent object's placement box in the random_generator is
    replaced with a point-tight box, so subsequent builds always place those objects
    at the same coordinates.  The agent placement is left untouched, so it is drawn
    uniformly from the arena on every reset.

    Args:
        env: The base Safety Gymnasium environment (Builder or wrapped).
        layout_seed: Seed used for the very first reset to fix the layout.
                     None uses whatever seed was passed to env.reset().
    """

    def __init__(self, env: gym.Env, layout_seed: int | None = None):
        super().__init__(env)
        self._layout_seed = layout_seed
        self._layout_fixed = False

    def reset(self, **kwargs):
        if not self._layout_fixed:
            seed_kwargs = dict(kwargs)
            if self._layout_seed is not None:
                seed_kwargs['seed'] = self._layout_seed
            obs, info = self.env.reset(**seed_kwargs)
            self._freeze_non_agent_placements()
            self._layout_fixed = True
            # Second reset: re-draw agent position with caller's original seed
            obs, info = self.env.reset(**kwargs)
        else:
            obs, info = self.env.reset(**kwargs)
        return obs, info

    def _freeze_non_agent_placements(self):
        """Replace every non-agent placement with a point box at its current position."""
        task = self.unwrapped.task
        rg = task.random_generator
        layout = task.world_info.layout
        for name, xy in layout.items():
            if name == 'agent' or name not in rg.placements:
                continue
            keepout = rg.placements[name][1]
            eps = keepout + 1e-9
            rg.placements[name] = ([(xy[0] - eps, xy[1] - eps, xy[0] + eps, xy[1] + eps)], keepout)


# ---------------------------------------------------------------------------
# Updated make_env with termination mode support
# ---------------------------------------------------------------------------

def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    run_name,
    fixed_reset_seed=False,
    termination_mode="none",
    termination_kwargs=None,
    fixed_layout=False,
    layout_seed=None,
):
    """Create a (possibly termination-wrapped) Safety Gymnasium environment.

    Args:
        env_id: Gymnasium / Safety Gymnasium environment ID.
        seed: Random seed.
        idx: Environment index (for video recording).
        capture_video: Whether to record video.
        run_name: Run name for video directory.
        termination_mode: One of:
            "none"        - standard Safety Gymnasium (no termination on cost)
            "cumulative"  - terminate when cumulative cost exceeds budget
            "stochastic"  - terminate probabilistically proportional to cost
            "instant"     - terminate on any nonzero cost
        termination_kwargs: Dict of keyword arguments passed to the wrapper.
            For "cumulative": {"budget": 25.0, "truncate": False}
            For "stochastic": {"beta": 1.0}
            For "instant":    {} (no arguments)
        fixed_layout: If True, fix hazard and goal positions after the first reset
                      while re-sampling the agent spawn each episode.
        layout_seed: Seed for the initial layout when fixed_layout=True.
                     Use this to reproducibly pick a layout where hazards block
                     the path between the agent spawn region and the goal.
    """
    if termination_kwargs is None:
        termination_kwargs = {}

    def thunk():
        if "Safety" in env_id:
            import safety_gymnasium
            env = safety_gymnasium.make(env_id)
        else:
            env = gym.make(env_id)

        if fixed_layout:
            env = FixedLayoutWrapper(env, layout_seed=layout_seed)

        # Apply termination wrapper before episode statistics
        if termination_mode == "cumulative":
            env = CumulativeCostTermination(env, **termination_kwargs)
        elif termination_mode == "stochastic":
            env = StochasticCostTermination(env, **termination_kwargs)
        elif termination_mode == "instant":
            env = InstantCostTermination(env)
        elif termination_mode != "none":
            raise ValueError(f"Unknown termination_mode: {termination_mode}")

        env = RecordCostEpisodeStatistics(env)
        if fixed_reset_seed:
            env = FixedResetSeed(env, seed=seed)
        if capture_video:
            if idx == 0:
                env = safety_gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}")

        return env

    return thunk


class LinearSchedule:
    """Linearly interpolates a value from start_value to stop_value between steps start and stop."""

    def __init__(self, start: int, stop: int, start_value: float, stop_value: float):
        self.start = start
        self.stop = stop
        self.start_value = start_value
        self.stop_value = stop_value

    def value(self, step: int) -> float:
        if step <= self.start:
            return self.start_value
        if step >= self.stop:
            return self.stop_value
        t = (step - self.start) / (self.stop - self.start)
        return self.start_value + t * (self.stop_value - self.start_value)


class SummaryWriter(TBSummaryWriter):
    def __init__(self, log_dir=None, comment="", purge_step=None, max_queue=10, flush_secs=120, filename_suffix=""):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.csv_dict = {}

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False):
        self.csv_dict.update({"step": global_step, tag: scalar_value, 'walltime': walltime})
        return super().add_scalar(tag, scalar_value, global_step, walltime, new_style, double_precision)
    
    def dump_csv(self):
        with open(f'{self.log_dir}/logs.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_dict.keys())

            writer.writeheader()
            writer.writerow(self.csv_dict)
        
        self.csv_dict = {}
