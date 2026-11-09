"""B-spline and Hermite utilities for the ELPAC model.

Provides:
 - build_bspline_eval_matrix: precompute B_q(tau_j) matrix
 - eval_bspline_basis: evaluate basis at arbitrary s
 - second_difference_penalty: K_B matrix for smoothness prior
 - cubic_hermite_interpolate: within-shell trajectory evaluation

All functions return jnp arrays with consistent dtypes. Splines use
uniform interior knots on [0, 1] with cubic order (degree 3).

Symbol map (paper notation):
 B_q(tau_j) : velocity basis (Q=6 default)
 psi_p(s) : activation basis (P=5 default)
 K_B : second-difference penalty matrix (P x P)
 Hermite : piecewise cubic trajectory interpolation between shells
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline

__all__ = [
    "build_uniform_bspline_basis",
    "build_bspline_eval_matrix",
    "eval_bspline_basis",
    "eval_bspline_basis_derivative",
    "second_difference_penalty",
    "cubic_hermite_interpolate",
    "find_shell_index",
    "build_ispline_eval_matrix",
    "eval_ispline_basis",
]


def build_uniform_bspline_basis(n_basis: int, degree: int = 3) -> BSpline:
    """Build a uniform B-spline basis on [0, 1] with `n_basis` functions.

 Uses clamped knot vector:
 [0]*(degree+1) + interior + [1]*(degree+1)
 which gives `n_basis` basis functions of `degree` order on [0, 1].

 If `n_basis < degree + 1`, the degree is automatically reduced to
 `n_basis - 1` so the basis is still well-defined (e.g., P=3 becomes
 a degree-2 quadratic spline, P=2 becomes linear).
 """
    if n_basis < degree + 1:
        degree = max(1, n_basis - 1)
    n_interior = n_basis - degree - 1
    if n_interior > 0:
        interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    else:
        interior = np.array([])
    knots = np.concatenate([
        np.zeros(degree + 1),
        interior,
        np.ones(degree + 1),
    ])
    # A B-spline object with identity coefficients lets us call it at arbitrary s,
    # but we mainly use it to derive the design matrix below.
    coeffs = np.eye(n_basis)
    return BSpline(knots, coeffs, degree, extrapolate=False)


def _eval_bspline_design(n_basis: int, s_grid: np.ndarray, degree: int = 3) -> np.ndarray:
    """Evaluate the design matrix rows = s values, cols = basis function index."""
    bs = build_uniform_bspline_basis(n_basis, degree=degree)
    # bs(s) returns shape (len(s), n_basis) because coeffs is I_{n_basis}
    s_clipped = np.clip(s_grid, 0.0, 1.0)
    vals = bs(s_clipped)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    # Handle the boundary at s=1: BSpline typically returns 0 at the rightmost
    # knot because the basis is half-open. Fill in the limit.
    boundary = (s_clipped >= 1.0 - 1e-12)
    if boundary.any():
        # The last basis function equals 1 at s=1, all others 0
        vals[boundary] = 0.0
        vals[boundary, -1] = 1.0
    return vals


def build_bspline_eval_matrix(n_basis: int, n_eval: int, degree: int = 3) -> jnp.ndarray:
    """Build B_q(s_grid) matrix of shape (n_eval, n_basis) on a uniform grid.

 Used for precomputed shell-boundary velocity basis: n_eval = J+1.
 """
    s_grid = np.linspace(0.0, 1.0, n_eval)
    design = _eval_bspline_design(n_basis, s_grid, degree=degree)
    return jnp.asarray(design, dtype=jnp.float32)


def eval_bspline_basis(n_basis: int, s: float | np.ndarray, degree: int = 3) -> jnp.ndarray:
    """Evaluate cubic B-spline basis at a single s (or 1D array of s values).

 Returns shape (n_basis,) for scalar s, or (len(s), n_basis) for array s.
 """
    s_arr = np.atleast_1d(np.asarray(s))
    design = _eval_bspline_design(n_basis, s_arr, degree=degree)
    if np.isscalar(s):
        return jnp.asarray(design[0], dtype=jnp.float32)
    return jnp.asarray(design, dtype=jnp.float32)


def eval_bspline_basis_derivative(n_basis: int, s: float | np.ndarray, degree: int = 3) -> jnp.ndarray:
    """First derivative of the cubic B-spline basis at s.

 Used for the exact derivative of nu_k(s) (paper C.7) during post-fit analysis.
 """
    bs = build_uniform_bspline_basis(n_basis, degree=degree)
    dbs = bs.derivative(nu=1)
    s_arr = np.atleast_1d(np.asarray(s))
    s_clipped = np.clip(s_arr, 0.0, 1.0)
    vals = dbs(s_clipped)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isscalar(s):
        return jnp.asarray(vals[0], dtype=jnp.float32)
    return jnp.asarray(vals, dtype=jnp.float32)


def second_difference_penalty(P: int, ridge: float = 1e-4) -> jnp.ndarray:
    """Build the second-difference penalty matrix K_B = D2^T D2 + ridge*I.

 Guarantees positive-definiteness via the ridge term; without it the
 constant functions are in the null space of D2^T D2.
 """
    if P < 3:
        return jnp.eye(P, dtype=jnp.float32)
    D2 = np.zeros((P - 2, P), dtype=np.float32)
    for i in range(P - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    K_B = D2.T @ D2 + ridge * np.eye(P, dtype=np.float32)
    return jnp.asarray(K_B)


def find_shell_index(s: jnp.ndarray, J: int) -> jnp.ndarray:
    """Map continuous stage s in [0, 1] to enclosing shell index j in [0, J-1].

 Returns the integer j such that tau_{j} <= s < tau_{j+1} (or J-1 at s=1).
 Gradients through this function are zero (index is a control-flow choice)
 but that's fine because the Hermite interpolant below is continuously
 differentiable in s within each shell.
 """
    j = jnp.floor(s * J).astype(jnp.int32)
    j = jnp.clip(j, 0, J - 1)
    return j


def _build_ispline_design(
    n_basis: int,
    s_grid: np.ndarray,
    degree: int = 2,
) -> np.ndarray:
    """Evaluate an I-spline design matrix of shape (len(s_grid), n_basis).

 I-splines (Ramsay 1988) are integrals of M-splines. A standard construction
 is: take a uniform B-spline basis of ``degree+1`` on [0, 1] and integrate
 each basis function from 0. The resulting functions are monotone
 non-decreasing on [0, 1], start at 0 when ``s=0``, and reach 1 at ``s=1``
 (after normalization). A non-negative linear combination is therefore a
 monotone non-decreasing function on [0, 1] with chi_c(0) = 0.

 Implementation: we take a B-spline of order ``degree+1`` with ``n_basis+1``
 basis functions on [0, 1] (so its antiderivative has ``n_basis`` columns
 after dropping the constant mode), construct the antiderivative symbolically
 via ``BSpline.antiderivative()``, evaluate it at s_grid, subtract the
 value at 0 so each column starts at 0, and take columns 1..n_basis (dropping
 the constant antiderivative column). Normalization is done per-column by
 dividing by the value at s=1 so each basis ends at 1.
 """
    if n_basis < 1:
        raise ValueError(f"n_basis must be >= 1 for I-spline; got {n_basis}")
    if degree < 1:
        raise ValueError(f"degree must be >= 1 for I-spline; got {degree}")
    # Build a B-spline of degree `degree` with (n_basis) basis functions, then
    # integrate. After integration the order increases by 1 and the number of
    # basis functions is preserved (the antiderivative design matrix has the
    # same column count as the B-spline design at the lower degree).
    bs = build_uniform_bspline_basis(n_basis, degree=degree)
    # Antiderivative of a BSpline with identity coefficient matrix returns a
    # BSpline whose call at s yields the integral of each original basis from
    # the left knot to s.
    anti = bs.antiderivative(nu=1)
    s_clipped = np.clip(np.asarray(s_grid, dtype=np.float64), 0.0, 1.0)
    vals = anti(s_clipped)                              # (n_eval, n_basis)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    # Pin chi(0) = 0 by subtracting the value at 0 per column.
    vals0 = anti(np.array([0.0]))                       # (1, n_basis)
    vals0 = np.nan_to_num(vals0, nan=0.0, posinf=0.0, neginf=0.0)
    vals = vals - vals0
    # Normalize each column so its value at s=1 equals 1 (standard I-spline
    # normalization). Columns whose endpoint integral is ~0 (should not happen
    # for a clamped uniform B-spline with n_basis>=1) are left unnormalized to
    # avoid divide-by-zero.
    vals1 = anti(np.array([1.0])) - vals0               # (1, n_basis)
    denom = np.where(np.abs(vals1) > 1e-12, vals1, 1.0)
    vals = vals / denom
    # Clip to [0, 1] for numerical safety — antiderivative can overshoot by ~1e-16.
    vals = np.clip(vals, 0.0, 1.0)
    return vals.astype(np.float64)


def build_ispline_eval_matrix(n_basis: int, n_eval: int, degree: int = 2) -> jnp.ndarray:
    """Build I-spline basis design matrix on a uniform grid of [0, 1].

 Returns shape (n_eval, n_basis). Each column is a monotone non-decreasing
 I-spline basis function starting at 0 (s=0) and normalized to 1 (s=1). A
 non-negative linear combination of columns is therefore a monotone
 non-decreasing function on [0, 1] with f(0) = 0, as required for the
 cognition link chi_c(s).
 """
    s_grid = np.linspace(0.0, 1.0, n_eval)
    design = _build_ispline_design(n_basis, s_grid, degree=degree)
    return jnp.asarray(design, dtype=jnp.float32)


def eval_ispline_basis(n_basis: int, s: float | np.ndarray, degree: int = 2) -> jnp.ndarray:
    """Evaluate I-spline basis at a single s (or 1D array of s values).

 Returns shape (n_basis,) for scalar s, or (len(s), n_basis) for array s.
 """
    s_arr = np.atleast_1d(np.asarray(s))
    design = _build_ispline_design(n_basis, s_arr, degree=degree)
    if np.isscalar(s):
        return jnp.asarray(design[0], dtype=jnp.float32)
    return jnp.asarray(design, dtype=jnp.float32)


def cubic_hermite_interpolate(
    theta_left: jnp.ndarray,   # (R,) state at shell boundary j
    theta_right: jnp.ndarray,  # (R,) state at shell boundary j+1
    v_left: jnp.ndarray,       # (R,) velocity at shell boundary j
    v_right: jnp.ndarray,      # (R,) velocity at shell boundary j+1
    u: jnp.ndarray,            # scalar in [0, 1], local coordinate within shell
    delta_j: float = 1.0,      # width of the shell
) -> jnp.ndarray:
    """Evaluate the cubic Hermite interpolant within a single shell.

 Returns shape (R,). The Hermite basis functions (h00, h10, h01, h11)
 are the standard cubic Hermite basis functions.
 """
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return (
        h00 * theta_left
        + h10 * delta_j * v_left
        + h01 * theta_right
        + h11 * delta_j * v_right
    )
