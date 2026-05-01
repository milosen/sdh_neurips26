import numpy as np
import torch
from collections import deque

from torch import optim


def log_mean_exp(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Numerically stable log-mean-exp: log(mean(exp(x)))."""
    x_max = x.max(dim=dim, keepdim=True).values
    return x_max.squeeze(dim) + (x - x_max).exp().mean(dim=dim).log()


class SDH:
    def __init__(self,
                 cost_lambda,
                 cost_limit: float,
                 cost_lambda_schedule=None,
                 dual_lr=1e-4,
                 survival_rho: float = None,
                 gamma=0.99,
                 ep_cost_buffer_capacity=5,
                 cost_buffer_capacity=10000,
                 alive_reward=0,
                 dual_updates=False,
                 device='cpu') -> None:
        self.cost_limit = cost_limit
        self.cost_lambda_schedule = cost_lambda_schedule
        self.per_step_cost_limit = cost_limit * (1 - gamma)
        self.lam = torch.tensor(cost_lambda)
        self.gamma = gamma
        self._ep_costs = deque(maxlen=ep_cost_buffer_capacity)
        self._costs = deque(maxlen=cost_buffer_capacity)
        self.alive_reward = alive_reward
        self.dual_updates = dual_updates
        self.dual_lr = dual_lr
        # Desired survival probability. From SDH theory: rho = 1 / (1 + d*(1-gamma))
        self.survival_rho = survival_rho if survival_rho is not None else 1.0 / (1.0 + cost_limit * (1 - gamma))
        if self.dual_updates:
            init_eta = float(np.log(max(float(alive_reward), 1e-4)))
            self.log_eta = torch.tensor([init_eta], requires_grad=True, device=device)
            self.eta_optimizer = optim.Adam([self.log_eta], lr=dual_lr)
        
    def alpha(self, costs):
        #return np.exp(-self.lam*np.clip(costs-self.per_step_cost_limit, 0, None))
        return np.exp(-self.lam*costs)

    def compute_rewards_and_continuations(self, rewards, costs):
        alpha = self.alpha(costs)

        # reward attenuation
        attenuated_rewards = alpha*(rewards + self.alive_reward)

        # beta calculation
        next_continuation = self.gamma*alpha

        return attenuated_rewards, next_continuation
    
    def update(self, p_surv: float, step: int):
        if self.dual_updates:
            self.update_alive_reward(p_surv)
        if self.cost_lambda_schedule is not None:
            self.update_lambda(step)

    def update_alive_reward(self, p_surv: float):
        """Update the dual variable eta and set alive_reward = eta = exp(eta).

        Minimises L(eta) = exp(eta) * (sg[p_surv] - rho) via gradient descent.
        When p_surv < rho the gradient is negative so eta increases, raising
        alive_reward and incentivising the agent to survive more.
        """
        eta_loss = self.log_eta.exp() * (p_surv - self.survival_rho)
        self.eta_optimizer.zero_grad()
        eta_loss.backward()
        self.eta_optimizer.step()
        self.alive_reward = self.log_eta.exp().item()

    def update_lambda(self, step: int):
        self.lam = torch.tensor(self.cost_lambda_schedule.value(step), dtype=torch.float32)
