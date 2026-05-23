"""
Compares CMDP and SDH occupancy structures.

  • Two-state MDP  – exact polytopes via dense policy grid
  • 5×5 Gridworld  – state-occupancy heatmaps, navigation obstacle

SDH occupancy derivation
------------------------
Standard CMDP (constant discount γ):
    d_CMDP = (1-γ)(I - γ P_π)⁻¹ μ_π

SDH replaces γ with a state-action-dependent effective discount
    γ_eff(s,a) = γ · exp(-λ · c(s,a))

The SDH occupancy satisfies the modified Bellman flow
    d_SDH(s',a') = μ_π(s',a') + Σ_{s,a} γ_eff(s,a)·P_π(s',a'|s,a)·d_SDH(s,a)

In matrix form (P_π columns indexed by source (s,a)):
    d_SDH = (I - P_π_flat @ diag(γ_eff))⁻¹ μ_π

Scaled by (1-γ) to match env.d_pi convention; recovers env.d_pi at λ=0.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "../../hpo/plots"))

import numpy as np
import torch as th
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from einops import einsum, rearrange

from mdp import (FiniteCMDP, get_policy_grid, P_TWO_STATE, p_grid_action)
from grid import plot_state_function

# ── style ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8,  "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})

GAMMA      = 0.85
LAMS       = [0.5, 2.0, 5.0]
LAM_COLORS = ["#EE854A", "#D65F5F", "#8B2252"]

# ── SDH occupancy ─────────────────────────────────────────────────────────────

def sdh_d_pi(env: FiniteCMDP, policy: th.Tensor,
             c_sa: th.Tensor, lam: th.Tensor) -> th.Tensor:
    """(1-γ)-scaled SDH occupancy.  Recovers env.d_pi exactly at λ=0."""
    n_sa = env.n_states * env.n_actions

    geff = (env.gamma * th.exp(-lam * c_sa.float())).reshape(-1)  # (S·A,)

    P_pi     = einsum(policy.float(), env.P.float(),
                      "b aa ss, ss s a -> b ss aa s a")
    P_pi_flat = rearrange(P_pi, "b ss aa s a -> b (ss aa) (s a)")

    mod_P = P_pi_flat * geff[None, None, :]

    I_ = th.eye(n_sa, dtype=th.float64).unsqueeze(0)
    SR = th.linalg.inv(I_ - mod_P.double()).float()

    mu_pi   = einsum(env.mu.float(), policy.float(), "s, b a s -> b s a")
    mu_flat = rearrange(mu_pi, "b s a -> b (s a)")

    d_flat = th.einsum("bij,bj->bi", SR, mu_flat)
    d_sdh  = rearrange(d_flat, "b (s a) -> b s a",
                       s=env.n_states, a=env.n_actions)
    return (1.0 - env.gamma) * d_sdh


# ── convex-hull fill ──────────────────────────────────────────────────────────

def fill_hull(ax, xs, ys, color, alpha_fill=0.30, alpha_edge=0.85,
              lw=1.6, ls="-", label=None, zorder=2):
    pts = np.stack([xs, ys], axis=1)
    try:
        hull = ConvexHull(pts)
        v    = np.append(hull.vertices, hull.vertices[0])
        ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                color=color, alpha=alpha_fill, zorder=zorder)
        ax.plot(pts[v, 0], pts[v, 1],
                color=color, alpha=alpha_edge, lw=lw, ls=ls,
                label=label, zorder=zorder + 1)
    except Exception:
        ax.scatter(xs, ys, color=color, s=1, alpha=0.3, label=label)


# ═══════════════════════════════════════════════════════════════════════
# Two-state MDP
# ═══════════════════════════════════════════════════════════════════════
# P_TWO_STATE: action a0 = stay,  action a1 = switch states
# Cost: c(s0, a0) = 1  (staying in state 0 costs)
# Policy format: policy[batch, action, state]
#
# 4 deterministic extreme policies and their CMDP occupancies in
# (d(s0,a0), d(s1,a0)) projection:
#   D1  pi(a0|·)=1             → (0.50,  0.50)   [both states with a0]
#   D2  pi(a1|·)=1             → (0.00,  0.00)   [never uses a0]
#   D3  pi(a0|s0)=pi(a1|s1)=1 → (0.925, 0.00)   [absorbs into costly (s0,a0)]
#   D4  pi(a1|s0)=pi(a0|s1)=1 → (0.00,  0.925)  [absorbs into (s1,a0), safe]
# ═══════════════════════════════════════════════════════════════════════

# Deterministic corner policies  (batch=1, n_actions=2, n_states=2)
CORNERS = {
    "D1": th.tensor([[[1., 1.], [0., 0.]]]),   # a0 always
    "D2": th.tensor([[[0., 0.], [1., 1.]]]),   # a1 always
    "D3": th.tensor([[[1., 0.], [0., 1.]]]),   # a0@s0, a1@s1
    "D4": th.tensor([[[0., 1.], [1., 0.]]]),   # a1@s0, a0@s1
}

def two_state_section(axs):
    env = FiniteCMDP(P=P_TWO_STATE, gamma=GAMMA)
    env.mu = th.tensor([0.5, 0.5])

    c_sa = th.zeros(2, 2); c_sa[0, 0] = 1.0   # cost only at (s0, a0)
    b    = 0.30                                 # CMDP constraint threshold

    grid   = get_policy_grid(100)              # (10000, 2, 2)
    d_cmdp = env.d_pi(grid).detach()

    xc = d_cmdp[:, 0, 0].numpy()   # d(s0, a0) – high-cost coordinate
    yc = d_cmdp[:, 1, 0].numpy()   # d(s1, a0)

    d_sdh = {lam: sdh_d_pi(env, grid, c_sa, th.tensor(lam)).detach()
             for lam in LAMS}

    # auto-scale: cover the full polytope (x,y up to ~0.93)
    pad = 0.04
    lim = (max(xc.max(), yc.max()) + pad,) * 2

    def _ax_fmt(ax, xlabel=True, ylabel=True):
        ax.set_xlim(-pad, lim[0])
        ax.set_ylim(-pad, lim[1])
        ax.set_xticks([0, 0.25, 0.5, 0.75])
        ax.set_yticks([0, 0.25, 0.5, 0.75])
        if xlabel:
            ax.set_xlabel(r"$d(s_0,\,a_0)$  [costly]")
        if ylabel:
            ax.set_ylabel(r"$d(s_1,\,a_0)$")

    # corner occupancies for annotation
    def _corner_xy(pi, sdh_lam=None):
        if sdh_lam is None:
            d = env.d_pi(pi).detach()
        else:
            d = sdh_d_pi(env, pi, c_sa, th.tensor(sdh_lam)).detach()
        return d[0, 0, 0].item(), d[0, 1, 0].item()

    def _annotate_corners(ax, sdh_lam=None):
        for name, pi in CORNERS.items():
            x, y = _corner_xy(pi, sdh_lam)
            ax.plot(x, y, "k.", ms=5, zorder=10)
            offset = {"D1": (-0.04, 0.02), "D2": (0.02, -0.04),
                      "D3": (0.02, 0.02),  "D4": (-0.06, 0.02)}[name]
            ax.annotate(name, (x, y), xytext=(x+offset[0], y+offset[1]),
                        fontsize=7.5, color="k")

    # ── Panel A: CMDP polytope – feasible region bright, infeasible dim ──
    ax = axs["A"]
    # Full polytope (ghost – infeasible region)
    fill_hull(ax, xc, yc, "#4878D0",
              alpha_fill=0.08, alpha_edge=0.22, lw=1.0, ls="--", zorder=1)
    # Feasible sub-polytope (bright)
    feas_mask = xc <= b
    fill_hull(ax, xc[feas_mask], yc[feas_mask], "#4878D0",
              alpha_fill=0.42, alpha_edge=0.88, lw=1.6,
              label="CMDP (feasible)", zorder=2)
    ax.axvline(b, color="#D65F5F", lw=1.4, ls="--",
               label=rf"$c^\top d={b}$")
    _annotate_corners(ax)
    _ax_fmt(ax)
    ax.set_title(r"\textbf{CMDP polytope}")
    ax.legend(loc="upper right", framealpha=0.8)

    # ── Panel C: SDH polytopes ──────────────────────────────────────
    ax = axs["C"]
    fill_hull(ax, xc, yc, "#4878D0",
              alpha_fill=0.07, alpha_edge=0.28, lw=1.0, ls="--",
              label=r"CMDP ($\lambda=0$)", zorder=1)
    ax.axvline(b, color="#D65F5F", lw=1.0, ls="--", alpha=0.45)

    for lam, col in zip(LAMS, LAM_COLORS):
        d  = d_sdh[lam]
        xs = d[:, 0, 0].numpy()
        ys = d[:, 1, 0].numpy()
        fill_hull(ax, xs, ys, col, alpha_fill=0.28, alpha_edge=0.9, lw=1.4,
                  label=rf"SDH $\lambda={lam}$", zorder=3)
    # annotate how corners move under largest lambda
    _annotate_corners(ax)                              # CMDP corners (black dots)
    lam_big = LAMS[-1]
    for name, pi in CORNERS.items():
        x0, y0 = _corner_xy(pi)
        x1, y1 = _corner_xy(pi, lam_big)
        if abs(x1 - x0) + abs(y1 - y0) > 0.01:      # only draw if corner moves
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color="#8B2252",
                                        lw=1.0, shrinkA=3, shrinkB=3))
    _ax_fmt(ax)
    ax.set_title(r"\textbf{SDH polytopes}")
    ax.legend(loc="upper right", framealpha=0.8, fontsize=7.5)

    # ── Panel D: total SDH mass vs λ ────────────────────────────────
    ax   = axs["D"]
    lams = th.linspace(0.0, 10.0, 80)
    for pi, label, color, ls in [
        (CORNERS["D3"], r"D3: $\pi(a_0|s_0)=\pi(a_1|s_1)=1$ (costly)", "#8B2252", "-"),
        (th.ones(1,2,2)*0.5, r"uniform $\pi$",                          "#555555", ":"),
        (CORNERS["D4"], r"D4: $\pi(a_1|s_0)=\pi(a_0|s_1)=1$ (safe)",   "#4878D0", "--"),
    ]:
        masses = [sdh_d_pi(env, pi, c_sa, lam).sum().item() for lam in lams]
        ax.plot(lams.numpy(), masses, color=color, ls=ls, lw=1.5, label=label)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\sum_{s,a} d_\mathrm{SDH}(s,a)$")
    ax.set_title(r"\textbf{SDH total occupancy mass}")
    ax.legend(loc="upper right", framealpha=0.8, fontsize=7.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.05)


# ═══════════════════════════════════════════════════════════════════════
# 5×5 Gridworld – navigation obstacle
# ═══════════════════════════════════════════════════════════════════════
#
#  Grid layout (state = row*5 + col, row 0 = top):
#
#   ·  ·  G  ·  ·          G = implicit goal region (top row)
#   ·  ·  ·  ·  ·
#   ·  ·  ·  ·  ·
#   ·  X  X  X  ·          X = obstacle (states 16,17,18 – row 3, cols 1-3)
#   ·  ·  S  ·  ·          S = start    (state 22     – row 4, col 2)
#
#  The obstacle sits one row above the start, forcing the agent to route
#  left (via col 0) or right (via col 4) to reach the upper half.
#  Obstacle states carry cost 1 on all actions.
# ═══════════════════════════════════════════════════════════════════════

def gridworld_section(axs):
    n   = 5
    P   = p_grid_action(n, n).float()
    env = FiniteCMDP(P=P, gamma=GAMMA)

    # Start at bottom-centre (row 4, col 2)
    env.mu      = th.zeros(n * n)
    env.mu[22]  = 1.0

    # Obstacle: horizontal bar at row 3, cols 1-3  → states 16, 17, 18
    obstacle = {16, 17, 18}
    c_sa = th.zeros(n * n, 4)
    for s in obstacle:
        c_sa[s] = 1.0

    # Adjacency for plot_state_function
    adj = env.P.max(dim=-1).values.numpy()     # (S, S)

    # Two extreme policy thetas:
    #   theta_up    – unconstrained, goes straight toward goal (high cost)
    #   theta_route – tightly constrained, routes via left col (low cost)
    # p_grid_action actions: 0=left, 1=right, 2=up, 3=down
    theta_up = th.zeros(1, 4, n * n)
    theta_up[:, 2, :] = 2.5

    theta_route = th.zeros(1, 4, n * n)
    theta_route[:, 2, :] = 3.0
    for s in [24, 23, 22, 21]:          # row 4: steer left to reach col 0
        theta_route[:, 0, s] = 6.0
    for s in [19, 18, 17, 16]:          # row 3 obstacle row: escape left
        theta_route[:, 0, s] = 5.0
    theta_route[:, 2, 15] = 6.0         # col 0, row 3: up (clear of obstacle)

    # 4 constraint levels: interpolate theta between unconstrained and routing
    ALPHAS   = [0.0, 0.5, 0.75, 1.0]
    LAM_HI   = 2.0    # row 3
    LAM_LO   = 0.5    # row 4

    pi_list = [th.softmax((1 - a) * theta_up + a * theta_route, dim=1)
               for a in ALPHAS]

    d_cmdp_list  = [env.d_pi(pi).detach()                                for pi in pi_list]
    d_sdh_hi     = [sdh_d_pi(env, pi, c_sa, th.tensor(LAM_HI)).detach() for pi in pi_list]
    d_sdh_lo     = [sdh_d_pi(env, pi, c_sa, th.tensor(LAM_LO)).detach() for pi in pi_list]

    costs = [(c_sa * d[0]).sum().item() for d in d_cmdp_list]

    # Shared colour ceiling across all 12 occupancy panels
    vmax = max(d[0].sum(-1).max().item()
               for d in d_cmdp_list + d_sdh_hi + d_sdh_lo)

    # ── Panel E: gridworld layout ─────────────────────────────────
    ax = axs["E"]
    cost_grid = c_sa[:, 0].reshape(n, n).numpy()
    ax.imshow(cost_grid, cmap="Reds", vmin=0, vmax=1.5, origin="upper")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_title(r"\textbf{Navigation task}")
    ax.text(2, 4, r"\textbf{S}", ha="center", va="center",
            color="#333333", fontsize=11)
    ax.text(2, 0, r"\textbf{G}", ha="center", va="center",
            color="#333333", fontsize=11)
    for s in obstacle:
        r, c = s // n, s % n
        ax.text(c, r, r"\textbf{X}", ha="center", va="center",
                color="white", fontsize=9)
    ax.set_xlabel(r"$\longleftarrow$ tighter $b$ constraint")

    # ── Panels F–I: CMDP occupancy, constraint tightens left→right ─
    for ax_key, d, cost in zip(["F", "G", "H", "I"], d_cmdp_list, costs):
        ax  = axs[ax_key]
        rho = d[0].sum(-1).reshape(n, n).numpy()
        im  = plot_state_function(rho, adj, ax=ax, cmap="Blues")
        im.set_clim(0, vmax)
        ax.set_title(r"\textbf{CMDP}")
        plt.colorbar(im, ax=ax, shrink=0.78)
        ax.set_xlabel(rf"$c^\top d = {cost:.2f}$", fontsize=8)

    # ── Panels J–M: SDH(λ=LAM_HI) occupancy ────────────────────────
    for ax_key, d, cost in zip(["J", "K", "L", "M"], d_sdh_hi, costs):
        ax    = axs[ax_key]
        rho_s = d[0].sum(-1).reshape(n, n).numpy()
        im_s  = plot_state_function(rho_s, adj, ax=ax, cmap="Oranges")
        im_s.set_clim(0, vmax)
        ax.set_title(rf"\textbf{{SDH}} ($\lambda={LAM_HI}$)")
        plt.colorbar(im_s, ax=ax, shrink=0.78)
        total = d[0].sum().item()
        ax.set_xlabel(rf"mass $={total:.2f}$, $c^\top d={cost:.2f}$", fontsize=8)

    # ── Panels N–Q: SDH(λ=LAM_LO) occupancy ────────────────────────
    for ax_key, d, cost in zip(["N", "O", "P", "Q"], d_sdh_lo, costs):
        ax    = axs[ax_key]
        rho_s = d[0].sum(-1).reshape(n, n).numpy()
        im_s  = plot_state_function(rho_s, adj, ax=ax, cmap="Oranges")
        im_s.set_clim(0, vmax)
        ax.set_title(rf"\textbf{{SDH}} ($\lambda={LAM_LO}$)")
        plt.colorbar(im_s, ax=ax, shrink=0.78)
        total = d[0].sum().item()
        ax.set_xlabel(rf"mass $={total:.2f}$, $c^\top d={cost:.2f}$", fontsize=8)


# ═══════════════════════════════════════════════════════════════════════
# Assemble
# ═══════════════════════════════════════════════════════════════════════

def main():
    fig = plt.figure(figsize=(16, 14.0), layout="constrained")
    axs = fig.subplot_mosaic(
        "AACCDD\nEEFGHI\n..JKLM\n..NOPQ",
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.0, 1.0],
                     "width_ratios":  [1, 1, 1, 1, 1, 1]},
    )

    two_state_section(axs)
    gridworld_section(axs)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "compare_occupancy.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
