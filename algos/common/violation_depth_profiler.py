"""
Estimate the violation depth profile Omega_pi(b) from rollout data.

Usage:
    # Collect rollouts and store per-step costs
    profiles = ViolationDepthProfiler(gamma=0.99)
    for episode in rollouts:
        profiles.add_episode(episode_costs)  # list/array of per-step c(s,a)

    # Plot
    profiles.plot()

    # Or get raw data
    b_grid, omega, fit = profiles.estimate()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class ViolationDepthProfiler:

    def __init__(self, gamma: float = 0.99, b_points: int = 200):
        self.gamma = gamma
        self.b_points = b_points
        self.episodes = []

    def add_episode(self, costs: np.ndarray):
        """Add one episode's per-step costs c(s_t, a_t)."""
        self.episodes.append(np.asarray(costs, dtype=np.float64))

    def estimate(self, b_max: float = None, b_fit_max: float = None):
        """Estimate the violation depth profile.

        Returns:
            b_grid: array of violation depth thresholds
            omega: estimated Omega_pi(b) at each threshold
            fit: dict with single-scale fit parameters
        """
        if not self.episodes:
            raise ValueError("No episodes added")

        # Compute cumulative violations for each episode
        all_cum_costs = []
        all_discounts = []
        for costs in self.episodes:
            T = len(costs)
            cum_costs = np.cumsum(costs)
            discounts = self.gamma ** np.arange(T)
            all_cum_costs.append(cum_costs)
            all_discounts.append(discounts)

        # Determine b range
        max_cum = max(c[-1] for c in all_cum_costs if len(c) > 0)
        if b_max is None:
            b_max = max_cum * 1.1
        b_grid = np.linspace(0, b_max, self.b_points)

        # Estimate Omega(b) = E[sum_t gamma^t 1{C_t >= b}]
        omega = np.zeros(self.b_points)
        N = len(self.episodes)
        for cum_costs, discounts in zip(all_cum_costs, all_discounts):
            for i, b in enumerate(b_grid):
                omega[i] += np.sum(discounts * (cum_costs >= b))
        omega /= N

        # Fit single-scale exponential: Omega(b) = A * exp(-b / tau)
        fit = self._fit_exponential(b_grid, omega, b_fit_max=b_fit_max)

        return b_grid, omega, fit

    def _fit_exponential(self, b_grid, omega, b_fit_max: float = None):
        """Fit Omega(b) = A * exp(-b/tau) and compute goodness of fit."""
        _nan_result = {"tau_linear": np.nan, "A_linear": np.nan, "r_squared_linear": 0.0,
                       "tau_nonlinear": np.nan, "A_nonlinear": np.nan, "r_squared_nonlinear": 0.0}

        # Exclude b=0: omega[0] = E[discounted episode length] ≈ 1/(1-gamma), which is
        # a degenerate boundary value (all timesteps count), not part of the exponential tail.
        # Including it inflates A and ruins the fit for the actual decay region.
        mask = (omega > 1e-10) & (b_grid > 0)
        if b_fit_max is not None:
            mask &= (b_grid <= b_fit_max)
        if mask.sum() < 3:
            return _nan_result

        b_fit = b_grid[mask]
        log_omega = np.log(omega[mask])

        # Drop any non-finite log values (can occur at the tail near the threshold)
        finite = np.isfinite(log_omega)
        b_fit = b_fit[finite]
        log_omega = log_omega[finite]
        if len(b_fit) < 3:
            return _nan_result

        # Linear regression on log(Omega) = log(A) - b/tau
        try:
            coeffs = np.polyfit(b_fit, log_omega, 1)
        except (np.linalg.LinAlgError, ValueError):
            return _nan_result
        slope, intercept = coeffs
        tau = -1.0 / slope if slope < 0 else np.inf
        A = np.exp(intercept)

        # R^2
        log_pred = intercept + slope * b_fit
        ss_res = np.sum((log_omega - log_pred) ** 2)
        ss_tot = np.sum((log_omega - log_omega.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Also try nonlinear fit for robustness (use the same finite-filtered arrays)
        omega_fit = omega[mask][finite]
        try:
            def exp_model(b, A, tau):
                return A * np.exp(-b / tau)

            popt, _ = curve_fit(
                exp_model, b_fit, omega_fit,
                p0=[A, tau], maxfev=5000,
                bounds=([0, 1e-6], [np.inf, np.inf])
            )
            A_nl, tau_nl = popt
            pred_nl = exp_model(b_fit, A_nl, tau_nl)
            ss_res_nl = np.sum((omega_fit - pred_nl) ** 2)
            ss_tot_nl = np.sum((omega_fit - omega_fit.mean()) ** 2)
            r_squared_nl = 1 - ss_res_nl / ss_tot_nl if ss_tot_nl > 0 else 0.0
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            A_nl, tau_nl, r_squared_nl = A, tau, r_squared

        return {
            "tau_linear": tau,
            "A_linear": A,
            "r_squared_linear": r_squared,
            "tau_nonlinear": tau_nl,
            "A_nonlinear": A_nl,
            "r_squared_nonlinear": r_squared_nl,
        }

    def plot(self, ax=None, label=None, color=None, b_max=None,
             show_fit=True, log_scale=True):
        """Plot the violation depth profile with optional exponential fit."""
        b_grid, omega, fit = self.estimate(b_max=b_max, b_fit_max=b_max)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        # Plot empirical profile
        mask = omega > 1e-10
        plot_label = label or "Empirical"
        ax.plot(b_grid[mask], omega[mask], '-', label=plot_label,
                color=color, linewidth=2)

        # Plot exponential fit
        if show_fit and fit["tau_nonlinear"] < np.inf:
            tau = fit["tau_nonlinear"]
            A = fit["A_nonlinear"]
            r2 = fit["r_squared_nonlinear"]
            fitted = A * np.exp(-b_grid / tau)
            fit_label = (f"Exp. fit: $\\tau$={tau:.2f}, "
                         f"$R^2$={r2:.3f}")
            ax.plot(b_grid[mask], fitted[mask], '--', color=color,
                    alpha=0.7, label=fit_label)
        
        #if b_fit_max is not None:
        #    ax.set_xlim((None, b_fit_max))

        if log_scale:
            ax.set_yscale('log')
        ax.set_xlabel('Violation depth $b$')
        ax.set_ylabel('$\\Omega^\\pi(b)$')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def is_single_scale(self, threshold: float = 0.95) -> bool:
        """Check if the profile is approximately single-scale.

        Returns True if the log-linear R^2 exceeds the threshold.
        """
        _, _, fit = self.estimate()
        return fit["r_squared_linear"] >= threshold


def compare_environments(env_profiles: dict, save_path: str = None):
    """Plot profiles from multiple environments side by side.

    Args:
        env_profiles: dict mapping env_name -> ViolationDepthProfiler
        save_path: optional path to save the figure
    """
    n = len(env_profiles)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for ax, (name, profiler), color in zip(axes, env_profiles.items(), colors):
        profiler.plot(ax=ax, label=name, color=color)
        _, _, fit = profiler.estimate()
        r2 = fit["r_squared_linear"]
        tau = fit["tau_linear"]
        single = "Yes" if r2 > 0.95 else "No"
        ax.set_title(f"{name}\n$\\tau$={tau:.2f}, "
                     f"$R^2$={r2:.3f}, Single-scale: {single}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Example usage with synthetic data
# =========================================================================

if __name__ == "__main__":

    np.random.seed(42)

    # --- Viability-type environment (single scale) ---
    viability_profiler = ViolationDepthProfiler(gamma=0.99)
    for _ in range(500):
        T = 200
        # Cost is mostly zero with occasional bursts that decay quickly
        costs = np.zeros(T)
        if np.random.random() < 0.3:
            onset = np.random.randint(10, 100)
            duration = np.random.geometric(0.3)
            magnitude = np.random.exponential(0.5)
            costs[onset:onset + duration] = magnitude
        viability_profiler.add_episode(costs)

    # --- Budget-type environment (multi-scale) ---
    budget_profiler = ViolationDepthProfiler(gamma=0.99)
    for _ in range(500):
        T = 200
        # Cost at every step, drawn from a mixture of two scales
        if np.random.random() < 0.5:
            costs = np.random.exponential(0.1, size=T)
        else:
            costs = np.random.exponential(2.0, size=T)
        budget_profiler.add_episode(costs)

    # --- Navigation-type environment (heavy-tailed / multi-scale) ---
    navigation_profiler = ViolationDepthProfiler(gamma=0.99)
    for _ in range(500):
        T = 200
        # Sparse but large violations interspersed with long safe stretches
        costs = np.zeros(T)
        n_violations = np.random.poisson(3)
        for _ in range(n_violations):
            t = np.random.randint(0, T)
            costs[t] = np.random.pareto(1.5) + 1  # heavy-tailed
        navigation_profiler.add_episode(costs)

    compare_environments({
        "Viability (Hyfydy-like)": viability_profiler,
        "Budget (Safety Gym)": budget_profiler,
        "Navigation (PointButton-like)": navigation_profiler,
    }, save_path="violation_profiles.png")
