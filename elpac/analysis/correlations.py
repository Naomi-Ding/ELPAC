"""Stage-dependent correlations rho_c(r, r'; s) and liquid association (paper C.8, C.9).

Liquid association is the stage derivative of ρ_c(r, r'; s). We provide a
closed-form variant (``liquid_association_closed_form``) derived in paper.7 via ``∂ν/∂s`` and the chain rule, and retain the finite-difference
helper as ``liquid_association_finite_diff`` / ``liquid_association_trajectory``
for backward compatibility.
"""

from __future__ import annotations

import numpy as np


def stage_dependent_correlation(
    A: np.ndarray,                  # (R, K)
    nu: np.ndarray,                 # (K,)
    D: np.ndarray,                  # (R,)
    r: int, r_prime: int,
) -> float:
    """rho_c(r, r'; s) = sum_k a_rk a_{r'k} nu_k / sqrt((D_r + sum_k a_rk^2 nu_k)(D_{r'} + sum_k a_{r'k}^2 nu_k))."""
    num = float(np.sum(A[r] * A[r_prime] * nu))
    var_r = float(D[r] + np.sum(A[r] ** 2 * nu))
    var_rp = float(D[r_prime] + np.sum(A[r_prime] ** 2 * nu))
    denom = float(np.sqrt(var_r * var_rp) + 1e-12)
    return num / denom


def correlation_trajectory(
    A: np.ndarray,                  # (R, K)
    w_ckp: np.ndarray,              # (K, P)
    D: np.ndarray,                  # (R,)
    psi_grid: np.ndarray,           # (G, P)
    r: int, r_prime: int,
) -> np.ndarray:
    """Return rho_c(r, r'; s) evaluated on a grid of stages. Shape (G,)."""
    G = psi_grid.shape[0]
    out = np.zeros(G, dtype=np.float64)
    for g in range(G):
        log_nu = w_ckp @ psi_grid[g]
        nu_g = np.exp(np.clip(log_nu, -8.0, 8.0))
        out[g] = stage_dependent_correlation(A, nu_g, D, r, r_prime)
    return out


def liquid_association_finite_diff(
    A: np.ndarray,
    w_ckp: np.ndarray,
    D: np.ndarray,
    psi_grid: np.ndarray,
    grid: np.ndarray,               # (G,) the s values
    r: int, r_prime: int,
) -> np.ndarray:
    """LA(r, r'; s) = d/ds rho_c(r, r'; s) via finite difference on the grid."""
    rho = correlation_trajectory(A, w_ckp, D, psi_grid, r, r_prime)
    # Central difference on interior, forward/backward on edges
    la = np.zeros_like(rho)
    la[1:-1] = (rho[2:] - rho[:-2]) / (grid[2:] - grid[:-2])
    la[0] = (rho[1] - rho[0]) / (grid[1] - grid[0])
    la[-1] = (rho[-1] - rho[-2]) / (grid[-1] - grid[-2])
    return la


# Backward-compatible alias (finite-difference variant).
liquid_association_trajectory = liquid_association_finite_diff


# ---------------------------------------------------------------------------
# A2: Closed-form liquid association (the paper /.7, C.9)
# ---------------------------------------------------------------------------

def liquid_association_closed_form(
    A: np.ndarray,                  # (R, K)
    nu_c_s: np.ndarray,             # (K,) ν_c(s) at this stage
    nu_c_s_deriv: np.ndarray,       # (K,) ν'_c(s) at this stage
    D: np.ndarray,                  # (R,)
    r: int, r_prime: int,
    eps: float = 1e-12,
) -> float:
    """Exact d ρ_c(r, r'; s)/ds at a single stage (paper C.9 closed form).

 Let N(s) = Σ_k a_{rk} a_{r'k} ν_k(s),
 Vr(s) = d_r + Σ_k a_{rk}^2 ν_k(s),
 Vp(s) = d_{r'} + Σ_k a_{r'k}^2 ν_k(s),
 so that ρ = N / √(Vr · Vp).

 Their derivatives:
 N'(s) = Σ_k a_{rk} a_{r'k} ν'_k(s)
 Vr'(s) = Σ_k a_{rk}^2 ν'_k(s)
 Vp'(s) = Σ_k a_{r'k}^2 ν'_k(s)

 Quotient/chain rule (let W = √(Vr Vp)):
 dρ/ds = N' / W − N · (Vr' Vp + Vr Vp') / (2 W Vr Vp)
 = N'/W − ρ · (Vr'/Vr + Vp'/Vp) / 2

 This is what paper.7 + C.9 give; no finite-difference needed.
 """
    A = np.asarray(A, dtype=np.float64)
    nu = np.asarray(nu_c_s, dtype=np.float64)
    nup = np.asarray(nu_c_s_deriv, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    a_r = A[r]
    a_rp = A[r_prime]
    N = float(np.sum(a_r * a_rp * nu))
    N_prime = float(np.sum(a_r * a_rp * nup))
    Vr = float(D[r] + np.sum(a_r ** 2 * nu))
    Vp = float(D[r_prime] + np.sum(a_rp ** 2 * nu))
    Vr_prime = float(np.sum(a_r ** 2 * nup))
    Vp_prime = float(np.sum(a_rp ** 2 * nup))
    Vr = max(Vr, eps)
    Vp = max(Vp, eps)
    W = float(np.sqrt(Vr * Vp))
    rho = N / (W + eps)
    return N_prime / (W + eps) - 0.5 * rho * (Vr_prime / Vr + Vp_prime / Vp)


def liquid_association_closed_form_trajectory(
    A: np.ndarray,                  # (R, K)
    w_ckp: np.ndarray,               # (K, P)
    D: np.ndarray,                  # (R,)
    psi_grid: np.ndarray,           # (G, P) — ψ_p(s) on grid
    psi_deriv_grid: np.ndarray,     # (G, P) — ψ'_p(s) on grid
    r: int, r_prime: int,
) -> np.ndarray:
    """Trajectory of the closed-form LA on a grid. Returns shape (G,)."""
    G = psi_grid.shape[0]
    out = np.zeros(G, dtype=np.float64)
    for g in range(G):
        log_nu = w_ckp @ psi_grid[g]
        nu_g = np.exp(np.clip(log_nu, -8.0, 8.0))
        log_nu_prime = w_ckp @ psi_deriv_grid[g]
        nu_prime_g = nu_g * log_nu_prime
        out[g] = liquid_association_closed_form(A, nu_g, nu_prime_g, D, r, r_prime)
    return out
