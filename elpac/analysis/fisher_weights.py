"""Fisher information weights w^FI_{c,m}(s) (paper App. B.3, Eqs. 31-32).

Quantifies which modality is most informative about disease stage at each point
along the stage axis, using the fitted (theta_c(s), Sigma_c(s)).

Two variants:
 - ``fisher_weights_at_stage`` — diagonal SNR-style; legacy fast summary.
 - ``fisher_weight_blockwise_exact`` — exact per-modality marginal Fisher (Eq. 31);
 real-data default. Uses Sylvester-Woodbury on the full (R, R) block and then
 reduces per modality via marginal Fisher information of ``Y_m ~ N(θ'_m, Σ_{mm})``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from elpac.utils.covariance import (
    JITTER_DEFAULT,
    POSITIVE_FLOOR,
    safe_solve_psd,
)


def sigma_diagonal(A: np.ndarray, nu: np.ndarray, D: np.ndarray) -> np.ndarray:
    """[Sigma(s)]_rr = D_r + sum_k A_rk^2 nu_k — returns shape (R,)."""
    return D + (A ** 2) @ nu


def fisher_weights_at_stage(
    theta_prime: np.ndarray,    # (R,) d theta / ds at this stage
    A: np.ndarray,              # (R, K)
    nu: np.ndarray,             # (K,)
    D: np.ndarray,              # (R,)
    modality_of_r: np.ndarray,  # (R,) int modality index in [0, M)
    M: int,                     # number of modalities
) -> np.ndarray:
    """Return w^FI_{c, m}(s) for m in range(M). Shape (M,).

 Each biomarker r contributes (theta'_r)^2 / [Sigma(s)]_rr to its modality.
 Weights are normalized so they sum to 1 across modalities.
 """
    sigma_diag = sigma_diagonal(A, nu, D)
    fisher_r = (theta_prime ** 2) / np.clip(sigma_diag, 1e-12, None)   # (R,)
    weights = np.zeros(M, dtype=np.float64)
    for r in range(len(fisher_r)):
        weights[modality_of_r[r]] += fisher_r[r]
    total = weights.sum()
    if total > 0:
        return weights / total
    return weights


def fisher_weights_trajectory(
    A: np.ndarray,                  # (R, K)
    w_ckp: np.ndarray,              # (K, P)
    D: np.ndarray,                  # (R,)
    psi_grid: np.ndarray,           # (G, P)
    theta_prime_grid: np.ndarray,   # (G, R) d theta_c(s)/ds on the grid
    modality_of_r: np.ndarray,      # (R,)
    M: int,
) -> np.ndarray:
    """Return w^FI on a grid. Shape (G, M)."""
    G = psi_grid.shape[0]
    out = np.zeros((G, M), dtype=np.float64)
    for g in range(G):
        log_nu = w_ckp @ psi_grid[g]
        nu_g = np.exp(np.clip(log_nu, -8.0, 8.0))
        out[g] = fisher_weights_at_stage(theta_prime_grid[g], A, nu_g, D, modality_of_r, M)
    return out


# ---------------------------------------------------------------------------
# A1: Exact per-modality marginal Fisher weights (paper.10.2)
# ---------------------------------------------------------------------------

def _block_slice(bnd: np.ndarray, m: int) -> slice:
    return slice(int(bnd[m]), int(bnd[m + 1]))


def _marginal_fisher_information_block(
    theta_prime_m: np.ndarray,       # (R_m,)
    Sigma_mm: np.ndarray,            # (R_m, R_m) — marginal Σ_c(s) for modality m
    Sigma_mm_prime: np.ndarray,      # (R_m, R_m) — marginal Σ'_c(s) for modality m
) -> float:
    """Exact marginal Fisher information for a single modality's block (C.10.2).

 I_m(s) = θ'_m^T Σ_mm^{-1} θ'_m
 + ½ tr(Σ_mm^{-1} Σ'_mm Σ_mm^{-1} Σ'_mm).

 Uses Cholesky-based solves (no explicit inversion).
 """
    Rm = Sigma_mm.shape[0]
    eye_m = np.eye(Rm, dtype=np.float64)
    # Σ_mm^{-1} via Cholesky solve; also return Σ_mm^{-1} Σ'_mm by solving against Σ'_mm
    Sigma_inv_Sigma_prime = safe_solve_psd(Sigma_mm, Sigma_mm_prime, jitter=JITTER_DEFAULT)
    # θ'_m^T Σ_mm^{-1} θ'_m
    x = safe_solve_psd(Sigma_mm, theta_prime_m, jitter=JITTER_DEFAULT)
    mean_term = float(theta_prime_m @ x)
    # tr((Σ^{-1} Σ')^2) -- this equals tr(Σ^{-1} Σ' Σ^{-1} Σ')
    cov_term = float(np.einsum("ij,ji->", Sigma_inv_Sigma_prime, Sigma_inv_Sigma_prime))
    return mean_term + 0.5 * cov_term


def fisher_weight_blockwise_exact(
    A_c: np.ndarray,                 # (R, K)
    nu_c_s: np.ndarray,              # (K,)
    nu_c_s_deriv: np.ndarray,        # (K,) ν'_c(s)
    D_c: np.ndarray,                 # (R,)
    modality_boundaries: np.ndarray, # (M+1,) int — block ranges [bnd[m], bnd[m+1])
    theta_prime: np.ndarray | None = None,   # (R,) d θ_c(s)/ds; zero-vector if None
    l1_normalize: bool = True,
) -> np.ndarray:
    """Exact per-modality marginal Fisher-information weights (paper C.10.2).

 Each modality m owns biomarker indices ``[bnd[m], bnd[m+1])`` (contiguous).
 Marginal information ``I_m(s)`` is computed on the (R_m, R_m) block of
 ``Σ = A diag(ν) A^T + D`` and its stage derivative
 ``Σ' = A diag(ν') A^T``. Full model Σ is formed only on the sub-block
 via row/column picking; no explicit full R×R inversion is required.

 Complexity: O(R_m² · K + R_m³) per modality; overall O(R·K² + Σ_m R_m³)
 where the cubic block-inversion term replaces a full O(R³) inverse.
 For the paper scale (R ≈ 20, K ≤ 6, M ≤ 4), this is negligible.

 Returns an array of shape ``(M,)`` — L1-normalized across modalities
 when ``l1_normalize=True``, else raw marginal Fisher information I_m(s).
 """
    A_c = np.asarray(A_c, dtype=np.float64)
    nu = np.asarray(nu_c_s, dtype=np.float64)
    nu_prime = np.asarray(nu_c_s_deriv, dtype=np.float64)
    D = np.asarray(D_c, dtype=np.float64)
    bnd = np.asarray(modality_boundaries, dtype=np.int64)
    R, K = A_c.shape
    if theta_prime is None:
        theta_prime = np.zeros(R, dtype=np.float64)
    else:
        theta_prime = np.asarray(theta_prime, dtype=np.float64)
    if bnd.ndim != 1 or bnd[0] != 0 or bnd[-1] != R:
        raise ValueError(
            f"modality_boundaries must start at 0 and end at R={R}; got {bnd.tolist()}"
        )
    M = bnd.size - 1

    D_safe = np.clip(D, POSITIVE_FLOOR, None)

    I_values = np.zeros(M, dtype=np.float64)
    for m in range(M):
        sl = _block_slice(bnd, m)
        A_m = A_c[sl]                                            # (R_m, K)
        D_m = D_safe[sl]
        theta_prime_m = theta_prime[sl]
        # Marginal Σ_mm(s) = A_m diag(ν) A_m^T + diag(D_m)
        Sigma_mm = (A_m * nu) @ A_m.T + np.diag(D_m)
        # Σ'_mm(s) = A_m diag(ν') A_m^T
        Sigma_mm_prime = (A_m * nu_prime) @ A_m.T
        I_values[m] = _marginal_fisher_information_block(
            theta_prime_m, Sigma_mm, Sigma_mm_prime
        )

    if l1_normalize:
        total = float(np.sum(np.abs(I_values)))
        if total > 0:
            return I_values / total
    return I_values


def fisher_weight_blockwise_exact_trajectory(
    A_c: np.ndarray,                 # (R, K)
    w_ckp: np.ndarray,               # (K, P)
    D_c: np.ndarray,                 # (R,)
    psi_grid: np.ndarray,            # (G, P)
    psi_deriv_grid: np.ndarray,      # (G, P) — derivatives of ψ_p(s) on grid
    modality_boundaries: np.ndarray, # (M+1,)
    theta_prime_grid: np.ndarray | None = None,  # (G, R), optional
) -> np.ndarray:
    """Trajectory of exact blockwise Fisher weights. Returns (G, M)."""
    G = psi_grid.shape[0]
    M = modality_boundaries.size - 1
    out = np.zeros((G, M), dtype=np.float64)
    for g in range(G):
        log_nu = w_ckp @ psi_grid[g]
        nu_g = np.exp(np.clip(log_nu, -8.0, 8.0))
        # ν'(s) = ν(s) · d log ν / ds = ν(s) · (w @ ψ'(s)) [since log ν = w ψ]
        log_nu_prime = w_ckp @ psi_deriv_grid[g]
        nu_prime_g = nu_g * log_nu_prime
        theta_prime = None if theta_prime_grid is None else theta_prime_grid[g]
        out[g] = fisher_weight_blockwise_exact(
            A_c, nu_g, nu_prime_g, D_c,
            modality_boundaries,
            theta_prime=theta_prime,
            l1_normalize=True,
        )
    return out
