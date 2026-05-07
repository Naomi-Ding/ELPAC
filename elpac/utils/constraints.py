"""Parameter constraints and projections for the inv_trajectory v2 model.

Two critical constraints:

 1. Lower-triangular A_c with positive diagonal (S1):
 make_lower_triangular_A(a_lower, a_diag, R, K) -> (R, K)

 2. Per-factor normalization of activation coefficients (S4):
 project_per_factor_normalization(w_ckp) -> w_ckp - row_mean

 This is applied to the VARIATIONAL MEAN of w_{c,k,p} after every
 Phase-2 gradient update. Enforces (1/P) sum_p w_{c,k,p} = 0 for each
 (c, k), eliminating the K-dimensional scale ambiguity (the paper).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "make_lower_triangular_A",
    "make_lower_triangular_A_direct",
    "project_per_factor_normalization",
]


def make_lower_triangular_A(
    a_flat: jnp.ndarray,   # (R*K,) — unconstrained entries; upper triangle ignored
    a_diag_raw: jnp.ndarray,  # (K,) — unconstrained; softplus-ed for positive diagonal
    R: int,
    K: int,
) -> jnp.ndarray:
    """Build an (R, K) lower-triangular matrix with positive diagonal.

 The parameter layout is intentionally redundant-and-masked rather than
 fully packed: this makes the NumPyro trace shape stable across calls.

 - a_flat: unconstrained reals of shape (R, K); upper-triangle entries
 (r < k) are masked to zero.
 - a_diag_raw: unconstrained reals of shape (K,); the diagonal a_{k,k} is
 set to softplus(a_diag_raw[k]) > 0.

 Returns (R, K) with upper triangle = 0 and diagonal > 0.
 """
    A_full = a_flat.reshape(R, K)
    # Zero the upper triangle (entries with r < k)
    row_idx = jnp.arange(R)[:, None]
    col_idx = jnp.arange(K)[None, :]
    lt_mask = (row_idx >= col_idx).astype(A_full.dtype)
    A_masked = A_full * lt_mask
    # Replace the diagonal with softplus of a_diag_raw
    diag_vals = jax.nn.softplus(a_diag_raw)  # (K,)
    # The diagonal entries live at positions (k, k) for k in 0..K-1; assuming R >= K.
    diag_idx = jnp.arange(K)
    A_final = A_masked.at[diag_idx, diag_idx].set(diag_vals)
    return A_final


def make_lower_triangular_A_direct(
    a_flat: jnp.ndarray,        # (R*K,) or (R,K) — entries; upper-triangle masked
    a_diag_pos: jnp.ndarray,    # (K,) — ALREADY POSITIVE diagonal (no softplus)
    R: int,
    K: int,
) -> jnp.ndarray:
    """Like make_lower_triangular_A but takes an already-positive diagonal.

 audit (): the legacy `make_lower_triangular_A` applies
 softplus to a Normal-sampled `a_diag_raw`, which pins the diagonal at
 softplus(0)≈0.693 when shrinkage tightens (a_prior_std → 0). This breaks
 K_eff pruning under MGP/ARD priors — the diagonal can never reach 0.

 Paper the paper (line 864) actually specifies
 a_{c,k,k} ~ HalfNormal(ψ_{c,k}^{-1/2})
 sampled directly. This helper supports that paper-faithful path: the
 caller draws a_diag_pos from HalfNormal(prior_std) and we plug it in
 without a redundant softplus.
 """
    A_full = a_flat.reshape(R, K)
    row_idx = jnp.arange(R)[:, None]
    col_idx = jnp.arange(K)[None, :]
    lt_mask = (row_idx >= col_idx).astype(A_full.dtype)
    A_masked = A_full * lt_mask
    diag_idx = jnp.arange(K)
    A_final = A_masked.at[diag_idx, diag_idx].set(a_diag_pos)
    return A_final


def project_per_factor_normalization(w_ckp: jnp.ndarray) -> jnp.ndarray:
    """Subtract per-factor mean from activation-spline coefficients.

 Input shape: (C, K, P). For each (c, k), subtract the mean over p.
 Enforces (1/P) sum_p w_{c,k,p} = 0 — the paper / (S4).
 """
    mean_per_factor = jnp.mean(w_ckp, axis=-1, keepdims=True)  # (C, K, 1)
    return w_ckp - mean_per_factor
