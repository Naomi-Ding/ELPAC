"""Efficient Gaussian log-pdf evaluation via Sylvester + Woodbury.

Implements the closed-form identities from the paper (App. B.3):

 Sigma_c(s) = A_c diag(nu_c(s)) A_c^T + diag(D_c) (R x R)

 log |Sigma_c(s)| = sum_r log D_{c,r} + log |I_K + diag(nu) M|
 Sigma_c(s)^{-1} = diag(D_c)^{-1}
 - diag(D_c)^{-1} A_c [diag(nu)^{-1} + M]^{-1} A_c^T diag(D_c)^{-1}

where M = A_c^T diag(D_c)^{-1} A_c is a K x K matrix. This turns R x R
operations into K x K operations, an O(R^3) -> O(R*K^2 + K^3) win.

The Gaussian log-pdf is evaluated on the observed coordinates only, using
mask-based selection rather than pre-slicing the matrices (JAX-friendly).

Numerical-stability helpers (A4, Audit 3):
 - safe_cholesky: Cholesky with 1e-10 diagonal jitter fallback.
 - safe_solve_psd: SPD solve via Cholesky (preferred over np.linalg.inv).
 - safe_slogdet: slogdet that asserts sign > 0 for PSD matrices.
 - safe_log: log(x) clipped at a positive floor.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "build_M_matrix",
    "gaussian_logpdf_sylvester_woodbury",
    "sigma_diagonal",
    "safe_cholesky",
    "safe_solve_psd",
    "safe_slogdet",
    "safe_log",
    "POSITIVE_FLOOR",
    "JITTER_DEFAULT",
]


POSITIVE_FLOOR = 1e-12      # minimum value for arguments to log()
JITTER_DEFAULT = 1e-10      # default diagonal jitter for near-singular PSD matrices


def safe_log(x: np.ndarray, floor: float = POSITIVE_FLOOR) -> np.ndarray:
    """log(x) with x clipped to [floor, inf). No warnings on non-positive input."""
    x_arr = np.asarray(x, dtype=np.float64)
    return np.log(np.clip(x_arr, floor, None))


def safe_cholesky(
    matrix: np.ndarray,
    jitter: float = JITTER_DEFAULT,
    max_tries: int = 6,
) -> np.ndarray:
    """Cholesky factor with progressively larger diagonal jitter on failure.

 Adds `jitter * I` then `10 * jitter * I` etc. up to `max_tries` attempts.
 Raises ``np.linalg.LinAlgError`` if all attempts fail.
 """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"safe_cholesky expects a square matrix; got shape {mat.shape}")
    n = mat.shape[0]
    eye = np.eye(n, dtype=np.float64)
    cur_jitter = jitter
    last_err: Exception | None = None
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(mat + cur_jitter * eye)
        except np.linalg.LinAlgError as exc:
            last_err = exc
            cur_jitter *= 10.0
    raise np.linalg.LinAlgError(
        f"safe_cholesky failed after {max_tries} jitter escalations (last jitter={cur_jitter:.3e}): {last_err}"
    )


def safe_solve_psd(
    matrix: np.ndarray,
    rhs: np.ndarray,
    jitter: float = JITTER_DEFAULT,
) -> np.ndarray:
    """Solve ``matrix @ x = rhs`` for SPD ``matrix`` via Cholesky.

 Replaces ``np.linalg.inv(matrix) @ rhs`` usages with a numerically
 stable triangular-solve pair. ``rhs`` may be a vector or matrix.
 """
    L = safe_cholesky(matrix, jitter=jitter)
    # Solve L y = rhs, then L^T x = y
    y = np.linalg.solve(L, rhs)
    x = np.linalg.solve(L.T, y)
    return x


def safe_slogdet(
    matrix: np.ndarray,
    require_positive: bool = True,
    jitter: float = 0.0,
) -> Tuple[float, float]:
    """Sign-safe ``slogdet``.

 Returns ``(sign, logabsdet)``. If ``require_positive`` is True, raises
 ``ValueError`` when ``sign <= 0``. When the first attempt returns a
 non-positive sign, a tiny diagonal jitter (scaled by matrix norm) is
 added and the determinant is recomputed once before giving up. The
 default ``jitter=0`` keeps the log-determinant compatible
 with unguarded ``np.linalg.slogdet`` on well-conditioned inputs.
 """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"safe_slogdet expects a square matrix; got shape {mat.shape}")
    if jitter > 0:
        eye = np.eye(mat.shape[0], dtype=np.float64)
        mat = mat + jitter * eye
    sign, logdet = np.linalg.slogdet(mat)
    if require_positive and sign <= 0:
        # One retry with jitter proportional to matrix norm, to handle borderline-PD cases
        scale = float(np.linalg.norm(mat, ord=np.inf)) + 1.0
        eye = np.eye(mat.shape[0], dtype=np.float64)
        sign, logdet = np.linalg.slogdet(mat + JITTER_DEFAULT * scale * eye)
        if sign <= 0:
            raise ValueError(
                f"safe_slogdet: expected positive-definite matrix but slogdet sign={sign}"
            )
    return float(sign), float(logdet)


def build_M_matrix(A: jnp.ndarray, D: jnp.ndarray) -> jnp.ndarray:
    """Compute M = A^T diag(D)^{-1} A (K x K)."""
    # A : (R, K), D : (R,)
    A_scaled = A / D[:, None]  # (R, K)
    return A.T @ A_scaled      # (K, K)


def sigma_diagonal(A: jnp.ndarray, nu: jnp.ndarray, D: jnp.ndarray) -> jnp.ndarray:
    """Diagonal of Sigma = A diag(nu) A^T + diag(D): [D_r + sum_k A_rk^2 nu_k]."""
    # A : (R, K), nu : (K,), D : (R,)
    return D + (A ** 2) @ nu


def gaussian_logpdf_sylvester_woodbury(
    residual: jnp.ndarray,    # (R,) y - mean, on the full R-dim space
    mask: jnp.ndarray,        # (R,) binary, 1 = observed
    A: jnp.ndarray,           # (R, K) loading matrix for this subtype
    nu: jnp.ndarray,          # (K,) activations at this (c, s)
    D: jnp.ndarray,           # (R,) idiosyncratic noise per biomarker
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Log N(residual[mask] | 0, Sigma_c(s)[mask, mask]) via Sylvester + Woodbury.

 Vectorized — no Python control flow on R_obs. Uses the masking trick:
 `A_masked = A * mask[:, None]`, `D_masked = D * mask + (1 - mask)`, so that
 rows corresponding to missing coordinates contribute zero to the sums and
 the cross-products in a numerically safe way.

 Returns a scalar log-density.
 """
    # Mask out missing coordinates by zeroing the residual and the loading rows.
    r = residual * mask                              # (R,)
    A_m = A * mask[:, None]                          # (R, K)
    # For the diagonal D contribution, replace missing D entries with 1 (so log(1) = 0)
    # and r^2 / 1 = 0 for missing (already zeroed by mask).
    D_safe = jnp.where(mask > 0.5, D, jnp.ones_like(D))  # (R,)

    # M_obs = A_m^T diag(D_obs)^{-1} A_m, only over observed rows
    A_scaled = A_m / D_safe[:, None]                 # (R, K)
    M_obs = A_m.T @ A_scaled                          # (K, K)

    K = A.shape[1]

    # Determinant: log|Sigma_obs| = sum_{r in O} log D_r + log|I_K + diag(nu) M_obs|
    log_det_D = jnp.sum(mask * jnp.log(D_safe))
    nu_safe = jnp.clip(nu, min=eps)
    F = (nu_safe[:, None] * M_obs)                    # (K, K), = diag(nu) M
    I_plus_F = jnp.eye(K) + F
    sign, log_det_ipf = jnp.linalg.slogdet(I_plus_F)
    # B2 fix: if sign <= 0, I_plus_F is not PD — return -inf so this
    # non-PD sample is rejected cleanly rather than silently producing
    # a garbage LL. (sign should be +1 for a valid PD factor covariance.)
    log_det_sigma = jnp.where(
        sign > 0,
        log_det_D + log_det_ipf,
        jnp.array(-jnp.inf),
    )

    # Woodbury for quadratic form:
    # r^T Sigma^{-1} r = r^T D^{-1} r - r^T D^{-1} A_m (diag(nu)^{-1} + M_obs)^{-1} A_m^T D^{-1} r
    rDinv = r / D_safe                                # (R,)
    u = A_m.T @ rDinv                                  # (K,) = A_m^T D^{-1} r
    # B2 fix: add a small fixed-scale jitter (JITTER_DEFAULT · I_K) to inner
    # before solving so near-singular matrices don't produce NaN gradients.
    # Mirrors safe_cholesky's jitter strategy in JAX-compatible form
    # (no Python control flow, so jitter is always added but kept tiny).
    inner_raw = jnp.diag(1.0 / nu_safe) + M_obs       # (K, K)
    inner_jitter = JITTER_DEFAULT * jnp.eye(K)
    inner = inner_raw + inner_jitter                   # (K, K), jittered
    # Solve inner @ x = u
    solved = jnp.linalg.solve(inner, u)                # (K,)
    quad = jnp.sum(rDinv * r) - jnp.dot(u, solved)

    R_obs = jnp.sum(mask)
    const = R_obs * jnp.log(2.0 * jnp.pi)
    # B2 fix: propagate -inf from non-PD log_det_sigma through to the return.
    return jnp.where(
        sign > 0,
        -0.5 * (const + log_det_sigma + quad),
        jnp.array(-jnp.inf),
    )
