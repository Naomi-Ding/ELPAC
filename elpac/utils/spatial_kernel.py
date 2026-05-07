"""Shared modality spatial kernel utility (Step 5 paper-compliance).

Paper the paper Steps 7, 11, 13 specify per-modality spatial covariances Σ^(m)
that enter the onset-time and velocity-basis priors:

 η_{c, R_m} ~ N(t̃_{c,m} · 1_{|R_m|}, σ_η^2 · Σ^(m)) (Step 11)
 β_{c, R_m, q} ~ N(0, σ_{β,m,q}^2 · Σ^(m)) (Step 13)

Paper the paper specifies the four kernel families:

 | Modality | Σ^(m) family |
 |----------|--------------------------------------------------------|
 | Amyloid | RBF (Gaussian squared-exponential) with lengthscale ℓ_A|
 | Tau | (L^(Tau) + εI)^{-1} (connectome-informed). |
 | | In elpac we use the Laplacian / Matérn-½ |
 | | exponential kernel as a surrogate (no connectome yet). |
 | Struct. | CAR model (I - ρ_N W_N)^{-1} with nearest-neighbor W_N |
 | Blood | I (no spatial structure, scalar features) |

Paired fix: this utility is imported by BOTH `src/simulate.py` (truth
generation) and `src/model.py` (prior specification) so the exact same
covariance matrix enters both sides of the model-vs-truth comparison.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def exp_kernel_matrix(R_m: int, length_scale: float = 0.3,
                       nugget: float = 1e-3) -> np.ndarray:
    """R_m × R_m exponential (Matérn-½) covariance on a 1-D index grid.

 The grid index i ∈ {0,..., R_m-1} is normalized by R_m to place the
 biomarkers on [0, 1); pairwise distance is |i - j| / R_m. This keeps
 the covariance scale comparable across modalities of different sizes
 (|R_m| varies across A/T/N/B).

 Parameters
 ----------
 R_m : int
 Number of biomarkers in modality m.
 length_scale : float
 Exponential kernel length scale in normalized-index units.
 Default 0.3 means a pair at distance 0.3 has correlation e^{-1} ≈ 0.37.
 nugget : float
 Diagonal ridge added for numerical Cholesky stability. Default 1e-3.

 Returns
 -------
 Sigma : (R_m, R_m) np.float32 symmetric PSD matrix
 Sigma[i, j] = exp(-|i - j| / (R_m · length_scale))
 Sigma[i, i] = 1 + nugget

 Notes
 -----
 - R_m == 0: returns (0, 0) empty array — caller must skip.
 - R_m == 1: returns [[1 + nugget]] — degenerate but well-defined.
 """
    if R_m <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    idx = np.arange(R_m, dtype=np.float32)
    # Pairwise index distances normalized by R_m so the scale is [0, 1).
    d = np.abs(idx[:, None] - idx[None, :]) / float(max(R_m, 1))
    K = np.exp(-d / float(length_scale)).astype(np.float32)
    K = K + float(nugget) * np.eye(R_m, dtype=np.float32)
    # Enforce symmetry to within floating-point roundoff.
    K = 0.5 * (K + K.T)
    return K


def rbf_kernel_matrix(R_m: int, length_scale: float = 0.3,
                      nugget: float = 1e-3) -> np.ndarray:
    """R_m × R_m RBF (Gaussian / squared-exponential) covariance on a 1-D grid.

 K[i, j] = exp(-0.5 · ((i - j) / (R_m · length_scale))^2) + nugget·δ_ij

 Paper the paper: amyloid uses an RBF kernel (diffuse cortical deposition).
 """
    if R_m <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    idx = np.arange(R_m, dtype=np.float32)
    # Signed normalized distance so the squared-exponential is on the [0, 1) scale.
    d = (idx[:, None] - idx[None, :]) / float(max(R_m, 1))
    K = np.exp(-0.5 * (d / float(length_scale)) ** 2).astype(np.float32)
    K = K + float(nugget) * np.eye(R_m, dtype=np.float32)
    K = 0.5 * (K + K.T)
    return K


def laplacian_kernel_matrix(R_m: int, length_scale: float = 0.3,
                            nugget: float = 1e-3) -> np.ndarray:
    """Laplacian / Matérn-½ (exponential) kernel — alias for `exp_kernel_matrix`.

 Paper the paper: tau uses a connectome-informed graph Laplacian
 inverse. In elpac we surrogate with the 1-D exponential kernel
 (same intra-modality locality structure without requiring a connectome).
 """
    return exp_kernel_matrix(R_m, length_scale, nugget)


def car_kernel_matrix(R_m: int, rho: float = 0.5,
                      nugget: float = 1e-3) -> np.ndarray:
    """R_m × R_m CAR covariance with 1-D nearest-neighbor adjacency.

 K = (I − ρ · W_norm)^{-1} + nugget·I

 where W is the 1-D nearest-neighbor adjacency (W[i, j] = 1 iff |i−j|=1)
 and W_norm is row-normalized (each row sums to 1 at interior nodes and
 to 1/2 reduces correctly via the row-sum; endpoint rows normalize with
 their single neighbor to row-sum 1, see below).

 Paper the paper: structural modality uses a CAR prior with ρ_N ≈ 0.9.
 The default here is ρ=0.5 for numerical headroom; callers (simulator +
 model) pass the paper value via `build_per_modality_kernels(rho_N=...)`.

 Parameters
 ----------
 R_m : int
 Number of biomarkers (R_N in the paper).
 rho : float
 CAR coefficient in (0, 1). Must be < 1 / ‖W_norm‖_2 for positivity;
 for row-normalized 1-D nearest-neighbor W_norm, ‖W_norm‖_2 = 1 so
 rho ∈ [0, 1) is the valid range. rho=0.5 gives moderate coherence,
 rho=0.9 gives strong spatial smoothing (paper default).
 nugget : float
 Diagonal ridge for Cholesky stability.
 """
    if R_m <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    if R_m == 1:
        return (1.0 + nugget) * np.eye(1, dtype=np.float32)
    W = np.zeros((R_m, R_m), dtype=np.float32)
    for i in range(R_m):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < R_m - 1:
            W[i, i + 1] = 1.0
    # Row-normalize so each row sums to 1 (interior rows divide by 2, endpoint
    # rows divide by 1). This is the standard row-stochastic normalization
    # used with CAR (Besag, 1974).
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # defensive; shouldn't happen for R_m ≥ 2
    W_norm = (W / row_sums).astype(np.float32)
    I = np.eye(R_m, dtype=np.float32)
    # K = (I - rho · W_norm)^{-1}. Stable for rho in [0, 0.99]. At rho=1 the
    # system is singular (diffusion to stationary null-space).
    K = np.linalg.inv(I - float(rho) * W_norm).astype(np.float32)
    K = 0.5 * (K + K.T)  # enforce symmetry
    K = K + float(nugget) * I
    return K


def identity_kernel_matrix(R_m: int, nugget: float = 1e-3) -> np.ndarray:
    """R_m × R_m identity covariance — no spatial structure.

 Paper the paper: blood biomarkers have Σ^(B) = I because they are
 scalar features and not spatially registered.
 """
    if R_m <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    return (1.0 + float(nugget)) * np.eye(R_m, dtype=np.float32)


def build_per_modality_kernels(
    R_A: int,
    R_T: int,
    R_N: int,
    R_blood: int,
    length_scale_A: float = 0.3,
    length_scale_T: float = 0.3,
    rho_N: float = 0.9,   # Paper the paper value (was 0.5 in Step 5b; updated to paper-compliant)
    nugget: float = 1e-3,
    # Backward-compat: if caller passes a single `length_scale=`, use it as a
    # shared default for A and T (and ignore for N/blood which don't use ℓ).
    length_scale: float | None = None,
    # Step 5c (): pluggable external kernels. When provided, the
    # four paper-family kernels (RBF / Laplacian / CAR / Identity) are BYPASSED
    # and `external_K` is used instead (after shape validation + nugget ridge).
    # None preserves the prior behavior.
    external_K: Optional[List[np.ndarray]] = None,
) -> list:
    """Return the list of four Σ^(m) matrices in canonical modality order.

 Order: [Σ^(A) (RBF), Σ^(T) (Laplacian), Σ^(N) (CAR), Σ^(B) (Identity)].

 Paper the paper:
 - Amyloid (A): RBF (Gaussian squared-exponential).
 - Tau (T): Laplacian / Matérn-½ exponential kernel (surrogate for the
 paper's connectome-informed graph-Laplacian inverse).
 - Structural (N): CAR (conditional autoregressive, nearest-neighbor).
 - Blood (B): Identity (scalar features, no spatial registration).

 Parameters
 ----------
 R_A, R_T, R_N, R_blood : int
 Number of biomarkers in each modality (may be 0 → empty slot).
 length_scale_A : float
 RBF lengthscale for amyloid (normalized-index units).
 length_scale_T : float
 Laplacian lengthscale for tau (normalized-index units).
 rho_N : float
 CAR coefficient for structural (default 0.5; the paper default 0.9).
 nugget : float
 Diagonal ridge shared across all four kernels.
 length_scale : float, optional
 Backward-compat shim: if provided (old call-sites), override both
 `length_scale_A` and `length_scale_T` with this shared value so old
 `build_per_modality_kernels(..., length_scale=0.3)` calls keep working.
 external_K : list of 4 np.ndarray, optional
 Step 5c: pluggable per-modality kernels. When provided, must be a list
 of 4 symmetric PSD matrices with shapes (R_A, R_A), (R_T, R_T),
 (R_N, R_N), (R_blood, R_blood). The paper-family kernels are then
 ignored; only the `nugget` ridge is applied on top of the input K.
 Use case: real-data flexibility where Σ^(m) comes from atlas-based
 region distances or connectome adjacency, not 1-D biomarker-index.
 None (default) preserves the prior behavior.

 Returns
 -------
 list of four (R_m, R_m) np.ndarray. Zero-sized modalities get a (0, 0)
 empty array — callers must skip those slots.

 Examples
 --------
 With external_K, returned kernels match input (modulo nugget):
 >>> R_A, R_T, R_N, R_B = 5, 5, 6, 4
 >>> K_fake = [np.eye(R) * 2.0 for R in (R_A, R_T, R_N, R_B)]
 >>> out = build_per_modality_kernels(R_A, R_T, R_N, R_B, external_K=K_fake)
 >>> # out[m] should be K_fake[m] + nugget * I
 >>> bool(np.allclose(out[0], K_fake[0] + 1e-3 * np.eye(R_A)))
 True
 """
    if external_K is not None:
        assert len(external_K) == 4, (
            f"external_K must have 4 kernels (A, T, N, blood); "
            f"got {len(external_K)}"
        )
        expected_R = [R_A, R_T, R_N, R_blood]
        for m, (K, R_exp) in enumerate(zip(external_K, expected_R)):
            K_arr = np.asarray(K)
            assert K_arr.shape == (R_exp, R_exp), (
                f"external_K[{m}] shape {K_arr.shape} != expected "
                f"({R_exp}, {R_exp})"
            )
        out = []
        for K in external_K:
            K_arr = np.asarray(K, dtype=np.float32)
            if K_arr.shape[0] == 0:
                out.append(K_arr)
            else:
                K_out = K_arr + float(nugget) * np.eye(K_arr.shape[0],
                                                         dtype=np.float32)
                # Enforce symmetry to within floating-point roundoff.
                K_out = 0.5 * (K_out + K_out.T)
                out.append(K_out.astype(np.float32))
        return out

    if length_scale is not None:
        length_scale_A = float(length_scale)
        length_scale_T = float(length_scale)
    return [
        rbf_kernel_matrix(R_A, length_scale_A, nugget),
        laplacian_kernel_matrix(R_T, length_scale_T, nugget),
        car_kernel_matrix(R_N, rho_N, nugget),
        identity_kernel_matrix(R_blood, nugget),
    ]


def build_per_modality_kernels_single_ls(
    R_A: int, R_T: int, R_N: int, R_blood: int,
    length_scale: float = 0.3,
    nugget: float = 1e-3,
) -> list:
    """DEPRECATED: single-length-scale variant (exponential for all 4 modalities).

 Kept for pre-paper-kernel backward compat with any caller that expected
 the old "exponential everywhere" behavior. New code should call
 `build_per_modality_kernels` with its per-modality arguments.
 """
    return [exp_kernel_matrix(R, length_scale, nugget)
            for R in (R_A, R_T, R_N, R_blood)]
