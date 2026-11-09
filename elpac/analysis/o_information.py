"""Closed-form O-information, TC, DTC on the staged factor model.

Formulas use the standard-Gaussian definitions from Rosas et al. (2019):
 TC(Σ) = −½ log|R(Σ)| where R is the correlation matrix
 DTC(Σ) = ½ log|Σ| + ½ Σ_r log [Σ^{-1}]_rr
 Ω(Σ) = TC(Σ) − DTC(Σ)

Complexity O(R·K² + K³) per stage evaluation (sub-second for R≤200, K≤10).
Everything is computed from the fitted (A_c, ν_c, D_c).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from elpac.utils.covariance import (
    JITTER_DEFAULT,
    POSITIVE_FLOOR,
    safe_log,
    safe_slogdet,
    safe_solve_psd,
)


def _build_M(A: np.ndarray, D: np.ndarray) -> np.ndarray:
    """M = A^T diag(D)^{-1} A (K, K)."""
    D_safe = np.clip(D, POSITIVE_FLOOR, None)
    return A.T @ (A / D_safe[:, None])


def total_correlation(A: np.ndarray, nu: np.ndarray, D: np.ndarray) -> float:
    """TC(s) = -0.5 log|D| - 0.5 log|I_K + diag(nu) M| + 0.5 sum_r log(D_r + sum_k A_rk^2 nu_k)."""
    K = A.shape[1]
    M = _build_M(A, D)
    F = nu[:, None] * M                      # diag(nu) M
    # log-det of (I + F) via sign-safe slogdet (I + F is PSD for PSD M, nu >= 0)
    _sign, logdet_IpF = safe_slogdet(np.eye(K) + F, require_positive=True)
    marg = D + (A ** 2) @ nu                 # (R,)
    return float(
        -0.5 * np.sum(safe_log(D))
        - 0.5 * logdet_IpF
        + 0.5 * np.sum(safe_log(marg))
    )


def dual_total_correlation(A: np.ndarray, nu: np.ndarray, D: np.ndarray, eps: float = 1e-8) -> float:
    """Standard Gaussian DTC (Rosas 2019): DTC(Σ) = ½ log|Σ| + ½ Σ_r log [Σ^{-1}]_rr.

 Closed-form via Sylvester + Woodbury (O(R·K² + K³)):
 log|Σ| = log|D| + log|I_K + diag(ν) M| (Sylvester)
 [Σ^{-1}]_rr = 1/D_r − (A_r / D_r)^T G (A_r / D_r) (Woodbury; G = (diag(ν)^{-1} + M)^{-1})

 Uses Cholesky-based solves instead of ``np.linalg.inv`` (A4 hardening).
 """
    K = A.shape[1]
    M = _build_M(A, D)
    D_safe = np.clip(D, POSITIVE_FLOOR, None)
    nu_safe = np.clip(nu, eps, None)
    # G(s) = (diag(nu)^{-1} + M)^{-1} — solve instead of invert for stability.
    inner = np.diag(1.0 / nu_safe) + M
    # We need A_scaled @ G @ A_scaled.T diag; compute X = G @ A_scaled.T via solve.
    A_scaled = A / D_safe[:, None]                              # (R, K)
    X = safe_solve_psd(inner, A_scaled.T, jitter=JITTER_DEFAULT)  # (K, R) = G A_scaled^T
    inv_diag = (1.0 / D_safe) - np.einsum("rk,kr->r", A_scaled, X)
    # log|Σ| = log|D| + log|I_K + diag(ν) M| (Sylvester identity)
    _sign_IpF, logdet_IpF = safe_slogdet(np.eye(K) + nu[:, None] * M, require_positive=True)
    logdet_Sigma = float(np.sum(safe_log(D_safe)) + logdet_IpF)
    return float(0.5 * logdet_Sigma + 0.5 * np.sum(safe_log(inv_diag)))


def o_information(A: np.ndarray, nu: np.ndarray, D: np.ndarray) -> float:
    """Omega(s) = TC(s) - DTC(s). Positive = redundancy, negative = synergy."""
    return total_correlation(A, nu, D) - dual_total_correlation(A, nu, D)


def o_information_trajectory(
    A: np.ndarray,                  # (R, K)
    w_ckp: np.ndarray,              # (K, P)
    D: np.ndarray,                  # (R,)
    psi_grid: np.ndarray,           # (G, P) activation basis on evaluation grid
) -> np.ndarray:
    """Compute Omega(s) on a grid of stages. Returns shape (G,)."""
    G = psi_grid.shape[0]
    omega = np.zeros(G, dtype=np.float64)
    for g in range(G):
        log_nu = w_ckp @ psi_grid[g]       # (K,)
        nu_g = np.exp(np.clip(log_nu, -8.0, 8.0))
        omega[g] = o_information(A, nu_g, D)
    return omega


def phase_transition_crossings(omega_trajectory: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Find stages where Omega(s) changes sign. Returns an array of crossing s values.

 Handles exact-zero crossings by attributing the crossing to grid[i] only
 (not both [i] and [i+1]) to avoid double-counting.
 """
    crossings = []
    for i in range(len(omega_trajectory) - 1):
        a = omega_trajectory[i]
        b = omega_trajectory[i + 1]
        if a == 0.0:
            # If we already recorded this index as a crossing from the previous
            # iteration, skip. Otherwise record if the NEXT segment has a
            # different sign from the PREVIOUS segment.
            prev_sign = np.sign(omega_trajectory[i - 1]) if i > 0 else np.sign(b)
            next_sign = np.sign(b)
            if prev_sign != next_sign and prev_sign != 0 and next_sign != 0:
                crossings.append(float(grid[i]))
            continue
        if b == 0.0:
            # Let the next iteration handle the sign comparison
            continue
        if a * b < 0 and abs(a - b) > 1e-12:
            # Linear interpolation for the zero-crossing
            t = -a / (b - a)
            s_star = float(grid[i] + t * (grid[i + 1] - grid[i]))
            crossings.append(s_star)
    return np.asarray(crossings)


def wald_statistic_at_phase_transition(
    omega_samples: np.ndarray,      # (B, G) bootstrap samples of Omega(s) on the grid
    s_star_idx: int,                # grid index at the detected phase transition
    n: int,                         # sample size used in the VI fit
) -> Tuple[float, float]:
    """Compute the Wald statistic T_n(s*) = sqrt(n) * Omega_hat / sigma_hat at a grid point.

 Omega_hat is the mean across bootstrap samples, sigma_hat is the std.
 Returns (T_n, p_value_two_sided).
 """
    from scipy import stats
    col = omega_samples[:, s_star_idx]
    omega_hat = float(np.mean(col))
    sigma_hat = float(np.std(col) + 1e-12)
    T_n = float(np.sqrt(n)) * omega_hat / sigma_hat
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(T_n))))
    return T_n, p


# ---------------------------------------------------------------------------
# A3: Delta-method confidence band for Omega(s) (paper.13 Prop 3)
# ---------------------------------------------------------------------------

def omega_gradient_vec_sigma(Sigma: np.ndarray) -> np.ndarray:
    """Closed-form gradient ``∇_{vec Σ} Ω(Σ)`` evaluated at a single Σ.

 Rosas 2019 definitions:
 TC(Σ) = −½ log|Σ| + ½ Σ_r log Σ_rr
 DTC(Σ) = ½ log|Σ| + ½ Σ_r log [Σ^{-1}]_rr
 Ω(Σ) = TC − DTC = −log|Σ| + ½ Σ_r log Σ_rr − ½ Σ_r log [Σ^{-1}]_rr.

 Matrix-form gradients (flattened row-major to match ``vec = ravel('C')``):
 ∂(log|Σ|)/∂Σ = Σ^{-1} (symmetric)
 ∂(Σ_r log Σ_rr)/∂Σ = diag(1/Σ_rr)
 ∂(Σ_r log [Σ^{-1}]_rr)/∂Σ = − Σ^{-1} diag(1 / [Σ^{-1}]_rr) Σ^{-1}

 The last identity follows from ``∂ vec(Σ^{-1}) / ∂ vec(Σ) = −Σ^{-1} ⊗ Σ^{-1}``
 (for symmetric Σ) applied to the r-th diagonal entry of Σ^{-1}.

 Returns a flat vector of length ``R²`` (row-major), for use with
 ``V_Σ ∈ R^{R² × R²}`` consistent with ``vec = ravel('C')``.
 """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    R = Sigma.shape[0]
    # Use safe solve-based inverse (symmetric PSD)
    eye = np.eye(R)
    Sigma_inv = safe_solve_psd(Sigma, eye, jitter=JITTER_DEFAULT)
    Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)                  # re-symmetrize
    diag_Sigma = np.clip(np.diag(Sigma), POSITIVE_FLOOR, None)
    diag_Prec = np.clip(np.diag(Sigma_inv), POSITIVE_FLOOR, None)

    # Matrix-form gradient
    grad_logdet = Sigma_inv                                      # ∂ log|Σ|/∂Σ
    grad_diag_logSig = np.diag(1.0 / diag_Sigma)                 # ∂ Σ_r log Σ_rr /∂Σ
    # ∂ Σ_r log [Σ^{-1}]_rr / ∂Σ = - Σ^{-1} diag(1/[Σ^{-1}]_rr) Σ^{-1}
    grad_diag_logPrec = -Sigma_inv @ np.diag(1.0 / diag_Prec) @ Sigma_inv

    # Ω = -log|Σ| + ½ Σ log Σ_rr - ½ Σ log [Σ^{-1}]_rr
    grad_Omega = (
        -grad_logdet
        + 0.5 * grad_diag_logSig
        - 0.5 * grad_diag_logPrec
    )
    return grad_Omega.ravel(order="C")


def omega_delta_method_ci(
    Sigma_c_s: np.ndarray,               # (R, R) estimated covariance at stage s
    dSigma_dvec: np.ndarray | None,      # (R^2, P_psi) Jacobian ∂ vec(Σ)/∂ ψ;
                                         # set to None to pass V_Σ directly
    V_Sigma: np.ndarray,                 # (P_psi, P_psi) asymp covariance of ψ, OR
                                         # (R^2, R^2) asymp covariance of vec(Σ) if dSigma_dvec is None
    n: int,                              # effective sample size n in the paper
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """Delta-method confidence interval for Ω(Σ) at one stage (paper C.13 Prop 3).

 ``V_Sigma`` (and ``V_psi`` after chain rule, if ``dSigma_dvec`` given) must
 be the *finite-n* covariance of the estimator — i.e. Cov(vec Σ̂) at the
 observed sample size. The parametric-bootstrap path (recommended under VI)
 returns exactly this. No additional
 /√n scaling is applied.

 Variance of Ω̂:
 σ²_Ω(s) = ∇Ω(Σ)^T Cov(vec Σ̂) ∇Ω(Σ)

 CI at level 1−α:
 Ω̂(s) ± z_{α/2} · σ_Ω(s)

 The ``n`` argument is accepted for API compatibility but unused in the
 width (the finite-n scale is already baked into V_Sigma). Earlier code
 divided by √n a second time, producing CIs √n too narrow at n~1000 and
 guaranteeing coverage failure.

 Returns ``(Omega_hat, ci_lo, ci_hi, sigma_Omega)``.
 """
    from scipy import stats

    Sigma = np.asarray(Sigma_c_s, dtype=np.float64)
    R = Sigma.shape[0]
    if Sigma.shape != (R, R):
        raise ValueError(f"Sigma_c_s must be square; got {Sigma.shape}")

    # Omega_hat via correlation-matrix determinant (robust direct computation)
    eye_R = np.eye(R)
    Sigma_inv = safe_solve_psd(Sigma, eye_R, jitter=JITTER_DEFAULT)
    Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)
    _sS, logdet_Sigma = safe_slogdet(Sigma, require_positive=True)
    diag_Sigma = np.clip(np.diag(Sigma), POSITIVE_FLOOR, None)
    diag_Prec = np.clip(np.diag(Sigma_inv), POSITIVE_FLOOR, None)
    # Omega = -log|Σ| + ½ Σ_r log Σ_rr - ½ Σ_r log [Σ^{-1}]_rr
    Omega_hat = float(
        -logdet_Sigma
        + 0.5 * np.sum(safe_log(diag_Sigma))
        - 0.5 * np.sum(safe_log(diag_Prec))
    )

    grad_vec = omega_gradient_vec_sigma(Sigma)    # (R^2,)

    V_Sigma = np.asarray(V_Sigma, dtype=np.float64)
    if dSigma_dvec is None:
        if V_Sigma.shape != (R * R, R * R):
            raise ValueError(
                f"When dSigma_dvec is None, V_Sigma must be ({R*R},{R*R}); got {V_Sigma.shape}"
            )
        var_Omega = float(grad_vec @ V_Sigma @ grad_vec)
    else:
        J = np.asarray(dSigma_dvec, dtype=np.float64)
        if J.shape[0] != R * R:
            raise ValueError(
                f"dSigma_dvec first dim must be {R*R} (vec Σ); got {J.shape}"
            )
        if V_Sigma.shape != (J.shape[1], J.shape[1]):
            raise ValueError(
                f"V_Sigma must be ({J.shape[1]},{J.shape[1]}); got {V_Sigma.shape}"
            )
        # Chain rule: Cov(vec Σ̂) = J V_ψ J^T
        JtGrad = J.T @ grad_vec                   # (P_psi,)
        var_Omega = float(JtGrad @ V_Sigma @ JtGrad)

    var_Omega = max(var_Omega, 0.0)
    sigma_Omega = float(np.sqrt(var_Omega))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    # V_Sigma is finite-n Cov(vec Σ̂), so sigma_Omega is already the CI scale.
    # The `n` argument is retained for API compatibility but not used here.
    del n  # explicitly acknowledge the unused arg
    half_width = z * sigma_Omega
    return Omega_hat, Omega_hat - half_width, Omega_hat + half_width, sigma_Omega
