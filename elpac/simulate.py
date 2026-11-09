"""Synthetic data generator for elpac.

Matches the paper's generative process with the simplifications needed
for a tractable small-scale ground truth:

 - 4 modalities {A, T, N, B}, R biomarkers partitioned arbitrarily.
 - Shell discretization with J boundaries; velocity computed at each boundary
 via B-spline basis, sigmoid onset, modality-specific sign transform, and
 Bernoulli region gate.
 - Reverse-time integration from terminal state mu_c to shell j=0, soft
 origin pulled toward shared Theta_0.
 - Stage-varying factor covariance Sigma_c(s) = A_c diag(nu_c(s)) A_c^T + D_c
 with lower-triangular positive-diagonal A_c and per-factor normalized w_ckp.

The simulator respects the same constraints the model enforces:
 * lower-triangular A with positive diagonal
 * sign-constrained velocity (softplus / -softplus)
 * per-factor (1/P) sum_p w_ckp = 0
 * non-constant nu_k(s) — uses a random non-trivial w_ckp
 * anchor subjects near s=0.95, controls near s=0.01

Returns a SyntheticTruth dataclass with every quantity needed for the
recovery metrics in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .utils.bspline import (
    build_bspline_eval_matrix,
    eval_bspline_basis,
    eval_ispline_basis,
    cubic_hermite_interpolate,
    find_shell_index,
)
from .utils.constraints import (
    make_lower_triangular_A,
    project_per_factor_normalization,
)
from .utils.spatial_kernel import build_per_modality_kernels


# Modality codes A = amyloid, T = tau, N = structural, B = blood
MODALITY_CODES = ("A", "T", "N", "B")


@dataclass
class SyntheticShape:
    """All shape parameters for a synthetic run."""
    n: int = 120
    C: int = 2
    K: int = 3
    J: int = 4
    P: int = 5
    Q: int = 4
    R_A: int = 3
    R_T: int = 3
    R_N: int = 3
    R_blood: int = 2
    p_x: int = 4
    p_w: int = 3

    @property
    def R(self) -> int:
        return self.R_A + self.R_T + self.R_N + self.R_blood

    @property
    def modality_of_r(self) -> np.ndarray:
        """Per-biomarker modality code index in {0:A, 1:T, 2:N, 3:B}."""
        return np.concatenate([
            np.full(self.R_A, 0, dtype=np.int32),
            np.full(self.R_T, 1, dtype=np.int32),
            np.full(self.R_N, 2, dtype=np.int32),
            np.full(self.R_blood, 3, dtype=np.int32),
        ])


@dataclass
class SyntheticTruth:
    """Container for ground-truth quantities used by recovery metrics."""
    shape: SyntheticShape
    stages: np.ndarray            # (n,)
    subtypes: np.ndarray          # (n,) int
    anchor_type: np.ndarray       # (n,) int, 0 = general, 1 = anchor_c1,..., C = NC
    A_c: np.ndarray               # (C, R, K)
    w_ckp: np.ndarray             # (C, K, P) (per-factor normalized)
    D_c: np.ndarray               # (C, R)
    theta_c_shells: np.ndarray    # (C, J+1, R)
    v_c_shells: np.ndarray        # (C, J+1, R)
    beta_velocity: np.ndarray     # (C, R, Q) raw velocity basis coefficients
    onset_times: np.ndarray       # (C, R)
    sharpness: np.ndarray         # (C, R)
    delta_region: np.ndarray      # (C, R) binary
    alpha_c_truth: np.ndarray     # (C,) subtype prior intercepts
    beta_c_sub_truth: np.ndarray  # (C, p_x) subtype prior slopes
    f_c_age_truth: np.ndarray     # (C,) stage-prior age effect
    h_c_truth: np.ndarray         # (C, p_w) stage-prior covariate effect
    sigma_c_stage_truth: np.ndarray  # (C,) stage-prior std
    X: np.ndarray                 # (n, p_x)
    W: np.ndarray                 # (n, p_w) subset of X used in stage prior
    Y: np.ndarray                 # (n, R)
    mask: np.ndarray              # (n, R) binary
    C_cog: np.ndarray             # (n,) cognitive score (NaN if missing)
    ages: np.ndarray              # (n,)
    Theta_0: np.ndarray           # (R,) shared healthy reference
    phase_transition_s_star: float | None = None  # for recovery metrics
    # Ground-truth global pathological severity G*_i
    # computed from the TRUE θ*_c trajectory. Shape (n,). See
    # `compute_G_star_path_from_truth` for the formula. Optional for backwards
    # compatibility with truth objects pickled before.
    G_star_truth: np.ndarray | None = field(default=None)
    # Manuscript-aligned aliases for the gate-layer truth fields. Alias the
    # existing `onset_times`, `sharpness`, `beta_velocity` arrays under the
    # names used in the paper; not re-derived.
    t_cr_true: np.ndarray | None = field(default=None)        # (C, R) — = onset_times
    omega_cr_true: np.ndarray | None = field(default=None)    # (C, R) — = sharpness
    beta_cr_q_true: np.ndarray | None = field(default=None)   # (C, R, Q) — = beta_velocity


def _sample_monotone_velocity_coeffs(rng: np.random.Generator, C: int, R: int, Q: int, modality_of_r: np.ndarray) -> np.ndarray:
    """Draw velocity basis coefficients `beta_{c,r,q}` with per-subtype
 modality signatures reflecting real clinical AD subtypes:

 - c=0 "typical AD": strong amyloid+tau+atrophy, moderate blood
 - c=1 "hippocampal-led": moderate amyloid+tau, strong atrophy
 - c=2 "amyloid-sparing": strong amyloid+blood, mild tau/atrophy
 - c>=3 fallback random

 Each subtype has its own (modality -> velocity-magnitude) signature so
 the integrated trajectories end at modality-dependent distinct endpoints.
 The per-region, per-basis beta coefficients are still sampled with noise
 on top of the subtype-modality mean.

 Signs are applied INSIDE the model via softplus/-softplus; these raw
 beta values are unconstrained reals. Scale is compatible with the
 model's Normal(0, 2.0) prior — posterior can recover truth exactly.
 """
    # Per-subtype modality weights (shape (C, 4)): [Amyloid, Tau, Structural, Blood]
    # These are pre-softplus raw values; softplus compresses dynamic range,
    # so strongly negative values give near-zero velocity and positive
    # values give proportional positive velocity. Designed around the
    # clinical spectrum: SuperAger / Normal ager / Typical AD / Hippocampal AD.
    # SuperAger and Normal ager are AD-negative (near-zero A/T/blood) and
    # differ primarily in the extent of structural atrophy.
    clinical_signatures = np.array([
        [-3.0, -3.0, -2.5, -2.5],   # c0: SuperAger (all near zero; preserved cognition)
        [-2.5, -2.5,  0.0, -1.5],   # c1: Normal ager (mild atrophy only)
        [ 2.5,  2.5,  2.0,  1.5],   # c2: Typical AD (all modalities strongly affected)
        [ 1.5,  2.0,  2.5,  1.0],   # c3: Hippocampal-led AD (atrophy dominates)
    ], dtype=np.float32)
    if C > clinical_signatures.shape[0]:
        clinical_signatures = np.vstack([
            clinical_signatures,
            np.full((C - clinical_signatures.shape[0], 4), 1.8, dtype=np.float32),
        ])
    # Small noise on each subtype's signature (so replicate runs differ)
    clinical_signatures = clinical_signatures[:C] + 0.15 * rng.normal(size=(C, 4)).astype(np.float32)

    # Broadcast modality weight -> per-region weight via modality_of_r
    modality_weight = clinical_signatures[:, modality_of_r]   # (C, R)

    # beta = subtype-modality mean (broadcast over all q so the subtype
    # signature is present at every shell, not just where B[j, 0] is large)
    # plus small per-q noise for natural shell-wise variation. Noise std
    # is intentionally small (0.25) so the subtype signature dominates;
    # modality gates and onset smoothing in _velocity_at_shell_boundaries
    # still produce stage-dependent curves.
    subtype_mean = modality_weight[:, :, None]                 # (C, R, 1)
    noise = rng.normal(loc=0.0, scale=0.25, size=(C, R, Q)).astype(np.float32)
    beta = np.broadcast_to(subtype_mean, (C, R, Q)).astype(np.float32) + noise
    return beta


def _modality_velocity_sign(modality: int, blood_index: int | None = None, is_ab_ratio: bool = False) -> int:
    """Return +1 for softplus (accumulating) or -1 for -softplus (atrophy / Aβ42/40)."""
    # A, T = accumulation -> softplus(+)
    # N = atrophy -> -softplus
    # Blood: p-tau217 (+), Aβ42/40 (-); for simplicity, first blood index -> (-), rest (+)
    if modality == 0 or modality == 1:
        return +1
    if modality == 2:
        return -1
    if modality == 3:
        return -1 if is_ab_ratio else +1
    return +1


def _build_modality_signs(shape: SyntheticShape) -> np.ndarray:
    """(R,) array of +1/-1 signs for modality velocity transforms."""
    signs = np.ones(shape.R, dtype=np.float32)
    m_idx = shape.modality_of_r
    for r in range(shape.R):
        if m_idx[r] == 2:   # structural
            signs[r] = -1.0
        elif m_idx[r] == 3:  # blood
            # convention: first blood index = Aβ42/40 (-), rest (+)
            if r == shape.R_A + shape.R_T + shape.R_N:
                signs[r] = -1.0
            else:
                signs[r] = +1.0
    return signs


def _velocity_at_shell_boundaries(
    B_shell: np.ndarray,          # (J+1, Q)
    beta: np.ndarray,             # (C, R, Q)
    onset_times: np.ndarray,      # (C, R)
    sharpness: np.ndarray,        # (C, R)
    delta_region: np.ndarray,     # (C, R)
    tau_grid: np.ndarray,         # (J+1,)
    mod_signs: np.ndarray,        # (R,)
) -> np.ndarray:
    """Compute V_{c,j,r} on the shell grid."""
    # beta @ B_shell^T gives raw velocity coefficients of shape (C, R, J+1).
    # Transpose to (C, J+1, R)
    raw = np.einsum("crq,jq->cjr", beta, B_shell)                 # (C, J+1, R)
    # Apply modality sign via softplus. B fix: np.logaddexp is
    # numerically stable for large |raw|, unlike np.log1p(np.exp(raw)) which
    # overflows for raw ≳ 37 (exp(37) ≈ 1e16).
    softplus_raw = np.logaddexp(0.0, raw)                         # softplus
    velocity_mag = mod_signs[None, None, :] * softplus_raw        # (C, J+1, R)
    # Sigmoid onset gate
    onset_arg = (tau_grid[None, :, None] - onset_times[:, None, :]) / np.clip(sharpness[:, None, :], 1e-3, None)
    onset = 1.0 / (1.0 + np.exp(-onset_arg))                      # (C, J+1, R)
    # Region inclusion gate
    V = delta_region[:, None, :] * onset * velocity_mag           # (C, J+1, R)
    return V.astype(np.float32)


def compute_G_path_shared_origin_from_truth(
    theta_c_shells: np.ndarray,
    stages: np.ndarray,
    subtypes: np.ndarray,
    shape: SyntheticShape,
    theta_0_shared: np.ndarray,
) -> np.ndarray:
    """G^path computed against a SHARED healthy origin Theta_0 instead of
 per-subtype θ_c(0). Sibling of compute_G_star_path_from_truth.

 Motivation (added): the canonical G^path subtracts each
 subtype's own learned shell start θ_c(0). When the model uses a soft
 origin prior (Omega_0_soft_std=0.5), per-subtype θ_c(0) drifts and the
 displacement metric inherits that drift, masking real stage information
 in cross-subtype severity comparisons. This variant subtracts the
 *single* shared Theta_0 (the trained shared healthy reference parameter)
 so all subtypes are measured against a common origin. Numerically
 equivalent to canonical G^path under a hard-pinned origin
 (theta_c(0) ≡ Theta_0); diverges from canonical when the soft origin
 prior lets per-subtype θ_c(0) drift away from the shared Theta_0.

 Mathematically:
 Φ_c^{path,shared}(s) = (1/|M|) Σ_m ‖θ_c^std(s) − Theta_0^std‖_m

 Parameters identical to compute_G_star_path_from_truth except for
 `theta_0_shared`: (R,) array, the shared origin reference. Caller passes
 Theta_0 from posterior_full_real.npz.

 Returns: (n,) float32 array, same shape contract as the canonical version.
 """
    theta = np.asarray(theta_c_shells, dtype=np.float64)
    theta_0 = np.asarray(theta_0_shared, dtype=np.float64)
    C, J1, R = theta.shape
    if theta_0.shape != (R,):
        raise ValueError(
            f"theta_0_shared shape {theta_0.shape} != ({R},)"
        )
    bnd = [0,
           shape.R_A,
           shape.R_A + shape.R_T,
           shape.R_A + shape.R_T + shape.R_N,
           shape.R_A + shape.R_T + shape.R_N + shape.R_blood]
    # Standardize using the same (mu, sigma) as the canonical version so the
    # two variants are directly comparable on the standardized scale.
    baseline = theta[:, 0, :]
    mu = baseline.mean(axis=0)
    sigma = baseline.std(axis=0, ddof=0)
    fallback_scale = np.maximum(np.abs(theta).max(axis=(0, 1)) - np.abs(theta).min(axis=(0, 1)),
                                 1e-6)
    sigma_safe = np.where(sigma > 1e-8, sigma, fallback_scale)

    theta_std = (theta - mu[None, None, :]) / sigma_safe[None, None, :]
    theta_0_std = (theta_0 - mu) / sigma_safe                     # (R,) — shared origin in std-space
    disp_std = theta_std - theta_0_std[None, None, :]             # (C, J+1, R)

    phi = np.zeros((C, J1), dtype=np.float64)
    n_modalities = 0
    for m_i in range(4):
        lo, hi = bnd[m_i], bnd[m_i + 1]
        if hi <= lo:
            continue
        per_m_norm = np.linalg.norm(disp_std[:, :, lo:hi], axis=-1)
        phi = phi + per_m_norm
        n_modalities += 1
    if n_modalities == 0:
        raise RuntimeError("SyntheticShape has zero biomarkers across all modalities")
    phi = phi / float(n_modalities)

    max_terminal = float(np.maximum(phi[:, -1].max(), 1e-8))
    phi_norm = phi / max_terminal

    s_grid = np.linspace(0.0, 1.0, J1)
    s_i = np.asarray(stages, dtype=np.float64)
    c_i = np.asarray(subtypes, dtype=np.int64)
    G_star = np.empty(s_i.shape[0], dtype=np.float64)
    for c in range(C):
        idx = np.where(c_i == c)[0]
        if idx.size == 0:
            continue
        G_star[idx] = np.interp(s_i[idx], s_grid, phi_norm[c])
    return G_star.astype(np.float32)


def compute_G_star_path_from_truth(
    theta_c_shells: np.ndarray,
    stages: np.ndarray,
    subtypes: np.ndarray,
    shape: SyntheticShape,
) -> np.ndarray:
    """Ground-truth pathological severity G*_i on the true simulator trajectory.

 Per-subtype trajectory pathology summary:

 Φ_c^path(s) = (1/|M|) Σ_m ‖θ_c^std(s) − θ_c^std(0)‖_m

 where `θ_c^std` is θ_c standardized per-modality, with the per-modality baseline
 mean μ_m and baseline std σ_m computed ACROSS SUBTYPES at s=0 (the anchor):

 μ_m = mean_c mean_{r∈M_m} θ_c(0, r)
 σ_m = std_c mean_{r∈M_m} θ_c(0, r) (pooled across the subtype axis)

 `‖·‖_m` is the Euclidean norm taken over biomarkers belonging to modality m.
 After normalization by `max_c Φ_c^path(1)` the output is in [0, 1].

 Interpolates Φ_c^path(s) to the per-subject true stage S*_i and selects the
 subject's true subtype c*_i.

 Parameters
 ----------
 theta_c_shells : (C, J+1, R) true mean trajectory on the shell grid.
 stages : (n,) per-subject true stage S*_i in [0, 1].
 subtypes : (n,) per-subject true subtype c*_i.
 shape : SyntheticShape with modality boundaries R_A, R_T, R_N, R_blood.

 Returns
 -------
 G_star : (n,) float32 array in [0, 1] (monotone in S*_i within each subtype).
 """
    theta = np.asarray(theta_c_shells, dtype=np.float64)        # (C, J+1, R)
    C, J1, R = theta.shape
    # Modality boundaries (R_A, R_T, R_N, R_blood)
    bnd = [0,
           shape.R_A,
           shape.R_A + shape.R_T,
           shape.R_A + shape.R_T + shape.R_N,
           shape.R_A + shape.R_T + shape.R_N + shape.R_blood]
    # Per-modality per-biomarker standardization parameters computed ACROSS
    # SUBTYPES at s=0. For each modality m and each biomarker r∈M_m:
    # μ_{m,r} = mean_c θ_c(0, r)
    # σ_{m,r} = std_c θ_c(0, r) (pooled over the C subtypes)
    # ddof=0 so the population std is well-defined even for C=1 (→ σ=0 → fallback).
    baseline = theta[:, 0, :]                                    # (C, R)
    mu = baseline.mean(axis=0)                                   # (R,)
    sigma = baseline.std(axis=0, ddof=0)                         # (R,)
    # For tiny or zero σ (e.g. C=1 or degenerate shared anchor), fall back to
    # the global scale of the trajectory in that biomarker so the normalization
    # is still well-defined and dimension-free.
    fallback_scale = np.maximum(np.abs(theta).max(axis=(0, 1)) - np.abs(theta).min(axis=(0, 1)),
                                 1e-6)                           # (R,)
    sigma_safe = np.where(sigma > 1e-8, sigma, fallback_scale)   # (R,)

    theta_std = (theta - mu[None, None, :]) / sigma_safe[None, None, :]  # (C, J+1, R)
    disp_std = theta_std - theta_std[:, 0:1, :]                  # (C, J+1, R)
    # Per-modality Euclidean norm at each shell, averaged over modalities
    phi = np.zeros((C, J1), dtype=np.float64)
    n_modalities = 0
    for m_i in range(4):
        lo, hi = bnd[m_i], bnd[m_i + 1]
        if hi <= lo:
            continue                                             # empty modality
        per_m_norm = np.linalg.norm(disp_std[:, :, lo:hi], axis=-1)  # (C, J+1)
        phi = phi + per_m_norm
        n_modalities += 1
    if n_modalities == 0:
        raise RuntimeError("SyntheticShape has zero biomarkers across all modalities")
    phi = phi / float(n_modalities)                              # (C, J+1), Φ_c^path(s) unnormalized

    # Normalize by the max terminal (s=1) displacement across subtypes so the
    # scale is in [0, 1].
    max_terminal = float(np.maximum(phi[:, -1].max(), 1e-8))
    phi_norm = phi / max_terminal                                # (C, J+1)

    # Per-subject G*_i via linear interpolation of Φ_{c*_i}^path(·) at S*_i.
    s_grid = np.linspace(0.0, 1.0, J1)                           # (J+1,)
    s_i = np.asarray(stages, dtype=np.float64)                   # (n,)
    c_i = np.asarray(subtypes, dtype=np.int64)                   # (n,)
    G_star = np.empty(s_i.shape[0], dtype=np.float64)
    for c in range(C):
        idx = np.where(c_i == c)[0]
        if idx.size == 0:
            continue
        G_star[idx] = np.interp(s_i[idx], s_grid, phi_norm[c])
    return G_star.astype(np.float32)


def simulate_synthetic(
    shape: SyntheticShape,
    seed: int = 202,
    n_anchor_per_subtype: int = 15,
    n_nc: int = 20,
    noise_scale: float = 1.0,
    missing_rate: float = 0.2,
    phase_transition_s_star: float = 0.5,
    include_cognition: bool = False,
    cognition_noise: float = 0.3,
    cognition_model: str = "ispline",  # pair with ModelConfig.chi_c_basis; legacy callers must pass cognition_model="polynomial" explicitly.
    ispline_n_basis: int = 3,
    ispline_degree: int = 2,
    use_rich_nu: bool = False,
    nu_amplitude_scale: float = 2.0,
    backward_generation: bool = True,
    cognition_missing_rate: float = 0.2,
    mu_c_noise_std: float = 0.05,
    theta_0_prior_std: float = 0.3,
    # When True, draw
    # onset_times and beta_velocity jointly per modality from
    # N(center, sigma_eta^2 Σ^(m)) and N(mean, sigma_beta^2 Σ^(m))
    # respectively, with Σ^(m) the per-modality kernel from
    # `build_per_modality_kernels` (length_scale=`spatial_length_scale`, default 0.3). False
    # uses the legacy iid Normal draws.
    use_spatial_sim: bool = True,
    spatial_length_scale: float = 0.3,
    sigma_eta_spatial: float = 0.3,
    sigma_beta_spatial: float = 0.25,
    # Optional externally-provided per-modality kernels
    # [K_A, K_T, K_N, K_blood]. When provided AND use_spatial_sim=True, the
    # paper-family kernels (RBF / Laplacian / CAR / Identity) are replaced by
    # these user-supplied kernels. Useful when
    # Σ^(m) comes from an atlas / connectome / adjacency graph rather than a
    # 1-D biomarker-index grid. Shapes must match (R_A, R_A), (R_T, R_T),
    # (R_N, R_N), (R_blood, R_blood). None (default) uses the built-in
    # kernel families.
    external_kernels: Optional[List[np.ndarray]] = None,
    # Draw w_ckp from the model's smoothness prior (paper Eq. 15).
    # When True, replace the deterministic 6-shape template for w_ckp with random
    # draws from the model's prior:
    # tau_k ~ Half-Normal(sigma_tau) (per subtype, per factor)
    # w_{c,k} ~ N(0, tau_k^2 (K_B + eps_B I_P)^{-1}) (P-vec)
    # then apply per-factor zero-mean projection. The model's K_B smoothness
    # prior (use_paper_w_ckp_smoothness) matches this generative draw.
    # False uses the 6-shape template (recovery-test interpretability).
    w_ckp_paper_prior: bool = True,
    w_ckp_sigma_tau: float = 0.1,
    w_ckp_eps_B: float = 1e-3,
    # Project the deterministic 3-shape / 6-shape w_ckp
    # templates through the K_B smoother. Each template is replaced with
    # w_smooth = (I + lambda * K_B)^{-1} * w_template
    # then optionally rescaled to preserve L2 norm per (c, k). Result: smooth
    # (prior-consistent shape) BUT high-amplitude (recoverable signal).
    # Default 0.0 = no smoothing.
    w_ckp_kb_smooth_strength: float = 0.0,
    w_ckp_kb_smooth_preserve_amplitude: bool = True,
    # When True, draw A_c diagonal
    # from HalfNormal(scale=sim_halfnormal_diag_scale) directly, matching
    # the model's use_paper_halfnormal_diag=True prior. Default True keeps
    # simulator and model priors paired.
    # Scale convention: default 1.0 matches the model's default a_prior_std
    # for MGP with κ=1 / ARD with α=1. If the model sweep tunes shrinkage
    # strength, the paired simulator draw stays at this fixed scale so that
    # recovery is evaluated against a consistent truth magnitude.
    # Callers that need an legacy softplus(Normal(0.5, 0.2)) truth —
    # e.g. tests, calibration JSONs, misspec sweeps validated on earlier
    # data — must pass sim_halfnormal_diag=False explicitly.
    sim_halfnormal_diag: bool = True,
    sim_halfnormal_diag_scale: float = 1.0,
) -> SyntheticTruth:
    """Generate one synthetic dataset matching the paper's generative process.

 ``use_rich_nu``: when True, replace the 3-shape
 (rise/fall/mid-peak, amp ≈ ±0.9) w_ckp generator with a 6-shape template
 bank (logistic rise/fall, sharp Gaussian peak, early bimodal, late
 plateau, tanh step) at amplitude ``nu_amplitude_scale`` (default 2.0),
 plus subtype-specific emphasis weights. Default False preserves the
 legacy behavior.

 ``backward_generation``: when True, switch to the paper's reverse-time
 (endpoint-anchored) construction:
 1. Sample ``mu_c`` from an anchor-signature-centered prior (analog of
 the model's ``mu_c ~ N(bar_Y_c, tilde_Omega_c)``), using the
 clinical-signature-derived endpoint as the anchor-mean surrogate.
 2. Pin ``Theta_{c,J} = mu_c``.
 3. Backward-integrate ``theta_{c,j-1} = theta_{c,j} - Delta * V[c, j]``.
 ``theta_{c,0}`` is whatever the recursion produces.
 4. Sample ``Theta_0 ~ N(0, theta_0_prior_std)`` as a SEPARATE reference
 point. The soft-origin factor is then a
 non-trivial prior PULL rather than an equality.
 Default True; False uses the legacy forward-from-Theta_0 path.

 ``cognition_missing_rate``: fraction of subjects whose
 ``C_cog`` is randomly set to NaN (exercises model's ``C_cog_mask`` path).
 Default 0.2 matches real-data expectation. Applied AFTER observation
 (modality-block) missingness so the two mechanisms are independent.

 ``cognition_model="polynomial"`` makes the simulator's cognition surface match the model's
 cubic chi_c form. Callers that need the legacy path must pass
 ``cognition_model="linear"`` AND ``backward_generation=False`` AND
 ``cognition_missing_rate=0.0`` explicitly.

 ``cognition_model="ispline"``: sample per-
 subtype I-spline coefficients ``chi_c_raw_true`` of shape (C, ispline_n_basis),
 apply softplus to enforce nonneg, and combine with the I-spline basis
 evaluated at each subject's stage. The resulting chi_c(s) is a monotone
 non-decreasing function on [0, 1] with chi_c(0) = 0. This branch MUST be
 paired with the model's ``chi_c_basis="ispline"`` — the SLURM driver emits
 a warning when only one side is set.

 ``mu_c_noise_std`` (backward mode only): std of Gaussian noise added to
 the clinical-signature-implied endpoint when sampling mu_c. Ignored when
 ``backward_generation=False``.

 ``theta_0_prior_std`` (backward mode only): std of the Theta_0 prior
 N(0, std). In legacy mode Theta_0 is drawn from N(0, 0.3) (hardcoded);
 in backward mode this parameter controls the prior std for experimental
 clarity. Default 0.3 matches legacy.

 ``external_kernels``: optional list of 4 np.ndarray
 per-modality kernels [K_A, K_T, K_N, K_blood] to use IN PLACE of the
 paper's modality-specific kernels (RBF / Laplacian / CAR / Identity).
 Only consumed when ``use_spatial_sim=True``. Shapes must be
 (R_A, R_A), (R_T, R_T), (R_N, R_N), (R_blood, R_blood). Each kernel is
 used for BOTH the onset-time MVN and the velocity-basis
 MVN. None (default) uses the built-in kernel
 families. When ``external_kernels`` is provided, the caller
 is responsible for ensuring the same bundle is plumbed into the
 ``model.ModelConfig.external_kernels_path`` so simulator + model use
 matched per-modality kernels.
 """
    rng = np.random.default_rng(seed)
    n, C, K, J, P, Q = shape.n, shape.C, shape.K, shape.J, shape.P, shape.Q
    R = shape.R
    modality_of_r = shape.modality_of_r
    mod_signs = _build_modality_signs(shape)

    # --- Shared healthy origin ----------------------------------------
    # LEGACY (backward_generation=False): all subtypes share a single healthy
    # origin Theta_0 exactly at stage s=0 (theta_shells[c, 0, :] == Theta_0 for
    # every c), and diverge forward along distinct per-subtype trajectories.
    # mu_c is then defined AFTER V is built as
    # mu_c = Theta_0 + sum_j Delta_j V_{c, j}. This forces the soft-origin
    # factor to be exactly (trivially) satisfied by the truth.
    #
    # BACKWARD (backward_generation=True): sample mu_c FIRST from the anchor-
    # signature-centered prior, pin theta_{c,J} = mu_c, backward-integrate,
    # and sample Theta_0 separately. theta_{c,0} is whatever the recursion
    # produces — the soft-origin factor is then a non-trivial prior pull.
    Theta_0 = rng.normal(loc=0.0, scale=theta_0_prior_std, size=R).astype(np.float32)

    # --- Trajectory parameters ----------------------------------------
    # Region inclusion: most regions included with high prob, a few excluded.
    pi_region = rng.uniform(0.7, 0.98, size=(C, 4))                 # per subtype, per modality
    delta_region = np.zeros((C, R), dtype=np.float32)
    for c in range(C):
        for r in range(R):
            m = modality_of_r[r]
            delta_region[c, r] = float(rng.uniform() < pi_region[c, m])

    # Onset times t_{c,r} with soft cascade: amyloid earlier, tau mid, structural late
    onset_center_by_modality = np.array([0.25, 0.40, 0.55, 0.35], dtype=np.float32)
    if not use_spatial_sim:
        # LEGACY iid Normal per (c, r). Preserved.
        onset_times = np.zeros((C, R), dtype=np.float32)
        for c in range(C):
            for r in range(R):
                m = modality_of_r[r]
                center = onset_center_by_modality[m] + 0.1 * c  # offset by subtype
                onset_times[c, r] = np.clip(center + rng.normal(scale=0.08), 0.05, 0.95)
    else:
        # Per-modality MVN draw.
        # Build Σ^(m) per modality via shared utility. If
        # `external_kernels` is supplied, the built-in kernels are
        # bypassed and the user-supplied bundle is used instead.
        mod_sizes = [shape.R_A, shape.R_T, shape.R_N, shape.R_blood]
        Sigma_per_mod = build_per_modality_kernels(
            shape.R_A, shape.R_T, shape.R_N, shape.R_blood,
            length_scale=spatial_length_scale, nugget=1e-3,
            external_K=external_kernels,
        )
        # Scaled covariance = sigma_eta^2 * Σ^(m). Factor each once (per modality),
        # reuse across c for each draw.
        mod_L_list = []
        for m_i in range(4):
            Sm = Sigma_per_mod[m_i]
            if Sm.shape[0] == 0:
                mod_L_list.append(None)
                continue
            Lm = np.linalg.cholesky(
                (sigma_eta_spatial ** 2) * Sm + 1e-6 * np.eye(Sm.shape[0], dtype=np.float32)
            ).astype(np.float32)
            mod_L_list.append(Lm)

        # Compute modality block slices [lo, hi)
        mod_bounds = []
        lo = 0
        for m_i in range(4):
            hi = lo + mod_sizes[m_i]
            mod_bounds.append((lo, hi))
            lo = hi

        onset_times = np.zeros((C, R), dtype=np.float32)
        for c in range(C):
            for m_i in range(4):
                lo, hi = mod_bounds[m_i]
                if hi <= lo:
                    continue
                center = onset_center_by_modality[m_i] + 0.1 * c
                Lm = mod_L_list[m_i]
                eps = rng.normal(size=hi - lo).astype(np.float32)
                draw = center + (Lm @ eps).astype(np.float32)
                onset_times[c, lo:hi] = np.clip(draw, 0.05, 0.95)
    sharpness = np.abs(rng.normal(loc=0.15, scale=0.03, size=(C, R))).astype(np.float32)

    beta_velocity = _sample_monotone_velocity_coeffs(rng, C, R, Q, modality_of_r)
    if use_spatial_sim:
        # The spatial prior on beta_velocity. The
        # original _sample_monotone_velocity_coeffs returns a (C, R, Q) array
        # with per-(c, m, q) modality signature means + iid noise. Here we
        # REPLACE the iid noise with a spatially correlated draw per (c, m, q):
        # beta[c, R_m, q] ~ N(signature_mean, sigma_beta^2 Σ^(m))
        # The subtype-signature means (set by _sample_monotone_velocity_coeffs)
        # are retained; only the iid noise layer is swapped for the MVN layer.
        # We do this by re-computing the means and adding fresh correlated noise.
        # Use the SAME rng stream so downstream draws remain deterministic given
        # seed, but understand this changes when the flag flips.
        clinical_signatures = np.array([
            [-3.0, -3.0, -2.5, -2.5],
            [-2.5, -2.5,  0.0, -1.5],
            [ 2.5,  2.5,  2.0,  1.5],
            [ 1.5,  2.0,  2.5,  1.0],
        ], dtype=np.float32)
        if C > clinical_signatures.shape[0]:
            clinical_signatures = np.vstack([
                clinical_signatures,
                np.full((C - clinical_signatures.shape[0], 4), 1.8, dtype=np.float32),
            ])
        # Small perturbation to per-subtype signature (same scale as
        # _sample_monotone_velocity_coeffs but uses a fresh rng draw to avoid
        # double-consuming the stream). Recovery of the signature is what
        # matters; per-seed bit equality inside the spatial branch does NOT
        # need to match the legacy path.
        clinical_signatures = (clinical_signatures[:C]
                                + 0.15 * rng.normal(size=(C, 4)).astype(np.float32))
        modality_weight = clinical_signatures[:, modality_of_r]   # (C, R)
        subtype_mean = modality_weight[:, :, None]                # (C, R, 1)

        # Build Σ^(m) per modality with sigma_beta_spatial scale.
        # Thread `external_kernels` through so both onset and velocity draws
        # use the SAME per-modality kernel bundle (matches model side).
        mod_sizes_b = [shape.R_A, shape.R_T, shape.R_N, shape.R_blood]
        Sigma_beta_list = build_per_modality_kernels(
            shape.R_A, shape.R_T, shape.R_N, shape.R_blood,
            length_scale=spatial_length_scale, nugget=1e-3,
            external_K=external_kernels,
        )
        mod_L_beta_list = []
        for m_i in range(4):
            Sm = Sigma_beta_list[m_i]
            if Sm.shape[0] == 0:
                mod_L_beta_list.append(None)
                continue
            Lm = np.linalg.cholesky(
                (sigma_beta_spatial ** 2) * Sm + 1e-6 * np.eye(Sm.shape[0], dtype=np.float32)
            ).astype(np.float32)
            mod_L_beta_list.append(Lm)
        mod_bounds_b = []
        lo = 0
        for m_i in range(4):
            hi = lo + mod_sizes_b[m_i]
            mod_bounds_b.append((lo, hi))
            lo = hi

        beta_velocity = np.broadcast_to(subtype_mean, (C, R, Q)).astype(np.float32).copy()
        for c in range(C):
            for m_i in range(4):
                lo, hi = mod_bounds_b[m_i]
                if hi <= lo:
                    continue
                Lm = mod_L_beta_list[m_i]
                for q in range(Q):
                    eps = rng.normal(size=hi - lo).astype(np.float32)
                    noise = (Lm @ eps).astype(np.float32)
                    beta_velocity[c, lo:hi, q] = beta_velocity[c, lo:hi, q] + noise

    # Precompute B-spline design matrix for velocity basis at shell boundaries
    B_shell_jnp = build_bspline_eval_matrix(n_basis=Q, n_eval=J + 1)
    B_shell = np.asarray(B_shell_jnp)                                # (J+1, Q)
    tau_grid = np.linspace(0.0, 1.0, J + 1).astype(np.float32)
    V = _velocity_at_shell_boundaries(
        B_shell, beta_velocity, onset_times, sharpness, delta_region, tau_grid, mod_signs
    )                                                                # (C, J+1, R)
    delta_j = 1.0 / J

    # Forward vs backward integration branch.
    theta_shells = np.zeros((C, J + 1, R), dtype=np.float32)
    if not backward_generation:
        # LEGACY forward path — theta[c, 0] == Theta_0, mu_c emerges as endpoint.
        theta_shells[:, 0, :] = Theta_0[None, :]
        for j in range(1, J + 1):
            theta_shells[:, j, :] = theta_shells[:, j - 1, :] + delta_j * V[:, j, :]
        mu_c = theta_shells[:, -1, :].copy()   # (C, R) — integrated endpoints
    else:
        # BACKWARD (endpoint-anchored) generation.
        # Sample mu_c FIRST from an anchor-signature-centered
        # prior. The clinical-signature-implied endpoint (integrated from zero
        # with the same velocity profile) serves as the anchor-mean surrogate
        # ``m_c`` — this is the simulator's stand-in for the model's bar_Y_c
        # (which is computed from anchor observations AFTER generation). Small
        # Gaussian noise ``mu_c_noise_std`` plays the role of tilde_Omega_c.
        mu_c_mean = np.zeros((C, R), dtype=np.float32)
        for j in range(1, J + 1):
            mu_c_mean += delta_j * V[:, j, :]                        # (C, R)
        mu_c = (mu_c_mean +
                rng.normal(loc=0.0, scale=mu_c_noise_std,
                           size=(C, R)).astype(np.float32))           # (C, R)
        # Pin terminal state to mu_c.
        theta_shells[:, -1, :] = mu_c
        # Deterministic backward recursion.
        for j in range(J, 0, -1):
            theta_shells[:, j - 1, :] = theta_shells[:, j, :] - delta_j * V[:, j, :]
        # Theta_0 is already sampled above as an independent reference point
        # (N(0, theta_0_prior_std)), so the soft-origin factor
        # p(theta[c,0] | Theta_0) is a non-trivial prior pull, not an equality.

    # --- Factor covariance --------------------------------------------
    # Build per-subtype lower-triangular positive-diagonal A_c
    A_c = np.zeros((C, R, K), dtype=np.float32)
    for c in range(C):
        A_raw = rng.normal(loc=0.0, scale=0.4, size=(R, K)).astype(np.float32)
        A_raw = np.tril(A_raw)
        if sim_halfnormal_diag:
            # diag ~ HalfNormal(scale) (paper Sec. 2.4). Matches the
            # model's --use-paper-halfnormal-diag prior.
            diag_vals = np.abs(rng.normal(
                loc=0.0, scale=float(sim_halfnormal_diag_scale), size=K
            ).astype(np.float32))
        else:
            # Legacy: softplus(Normal(0.5, 0.2)) → diag ∈ ~[0.85, 1.6].
            diag_vals = np.log(1.0 + np.exp(rng.normal(loc=0.5, scale=0.2, size=K).astype(np.float32)))
        A_raw[np.arange(K), np.arange(K)] = diag_vals
        A_c[c] = A_raw

    # Activation coefficients w_ckp. Factor shapes are assigned deterministically:
    # k=0 late-rising, k=1 early-rising, k>=2 mid-stage peaks. The simulator does
    # NOT apply shrinkage ordering; the metric's factor-permutation search
    # accommodates arbitrary post-fit factor ordering.
    #
    # When `use_rich_nu=True`, replace the 3-shape bank
    # (rise/fall/mid-peak, amp ≈ ±0.9) with a 6-shape template bank (logistic
    # rise/fall, sharp Gaussian peak, early bimodal, late plateau, tanh step)
    # at amplitude `nu_amplitude_scale` (default 2.0), plus subtype-specific
    # emphasis weights to strengthen the stage-varying covariance signal.
    w_ckp = np.zeros((C, K, P), dtype=np.float32)
    knot_positions = np.linspace(0.0, 1.0, P)
    if w_ckp_paper_prior:
        # Draw w_{c,k} ~ N(0, tau_k^2 (K_B + eps_B I_P)^{-1})
        # with tau_k ~ Half-Normal(w_ckp_sigma_tau). This replaces the hand-crafted
        # shape templates with the paper's smoothness-prior generative draw on the
        # simulator side, paired with the model's --use-paper-w-ckp-smoothness flag.
        from elpac.utils.bspline import second_difference_penalty as _sd_pen
        K_B_mat = np.asarray(_sd_pen(P, ridge=0.0))                      # (P, P)
        W_prec = K_B_mat + float(w_ckp_eps_B) * np.eye(P)                # SPD
        W_cov = np.linalg.inv(W_prec)
        L_W = np.linalg.cholesky(0.5 * (W_cov + W_cov.T))                # P x P
        for c in range(C):
            for k in range(K):
                # |Normal(0, sigma_tau)| -> Half-Normal(sigma_tau)
                tau_k = float(abs(rng.normal(loc=0.0, scale=float(w_ckp_sigma_tau))))
                z = rng.normal(size=P).astype(np.float64)                # iid standard
                base = tau_k * (L_W @ z)
                base = base - base.mean()                                 # (S4) zero-mean
                w_ckp[c, k, :] = base.astype(np.float32)
    elif not use_rich_nu:
        for c in range(C):
            for k in range(K):
                if k == 0:
                    base = -0.9 + 1.8 * knot_positions + 0.2 * c
                elif k == 1:
                    base = 0.9 - 1.8 * knot_positions - 0.1 * c
                else:
                    peak_offset = 0.4 + (k - 2) * 0.05
                    base = 0.8 * np.exp(-((knot_positions - 0.5) ** 2) / 0.08) - peak_offset
                base = base + rng.normal(scale=0.15, size=P)
                base = base - base.mean()
                w_ckp[c, k, :] = base.astype(np.float32)
    else:
        amp = float(nu_amplitude_scale)
        # 6 distinct shape templates (monotone rise/fall, sharp Gaussian peak,
        # early bimodal, late plateau, tanh step). Each lambda returns a 1D
        # array of the same shape as `knot_positions`.
        shape_templates = [
            # k=0: monotone rise (logistic)
            lambda s: 1.0 * (2.0 / (1.0 + np.exp(-6.0 * (s - 0.5))) - 1.0),
            # k=1: monotone fall
            lambda s: -1.0 * (2.0 / (1.0 + np.exp(-6.0 * (s - 0.5))) - 1.0),
            # k=2: sharp mid-peak Gaussian
            lambda s: 1.3 * np.exp(-((s - 0.5) ** 2) / 0.04) - 0.6,
            # k=3: early bimodal (peaks at 0.2 and 0.8)
            lambda s: 0.9 * (np.exp(-((s - 0.2) ** 2) / 0.02)
                              + np.exp(-((s - 0.8) ** 2) / 0.02)) - 0.45,
            # k=4: late plateau (rises at s=0.3, plateaus)
            lambda s: 1.2 * np.where(s > 0.3, 1.0 - np.exp(-3.0 * (s - 0.3)), 0.0) - 0.6,
            # k=5: step transition
            lambda s: 1.2 * np.tanh(8.0 * (s - 0.5)),
        ]
        # Subtype-specific emphasis matrix (C, K). For C≥3, K≥3 each subtype
        # emphasizes a different subset of factors. Smaller/zero weights
        # deactivate factors for a subtype, creating cross-subtype contrast
        # that strengthens the M2 > M1 signal.
        if K <= len(shape_templates) and C >= 3 and K >= 3:
            subtype_weights = np.ones((C, K), dtype=np.float32)
            subtype_weights[0] = np.array(
                [1.0, 0.8, 1.0] + [0.3] * (K - 3), dtype=np.float32)  # c=0 emphasizes k=0, k=2
            subtype_weights[1] = np.array(
                [0.8, 1.0, 0.3] + [1.0] * (K - 3), dtype=np.float32)  # c=1 emphasizes k=1, late
            late_extras = [0.5, 1.0, 0.0][: max(0, K - 3)]
            subtype_weights[2] = np.array(
                [0.3, 1.0, 1.0] + late_extras, dtype=np.float32)      # c=2 emphasizes k=1, k=2
        else:
            subtype_weights = np.ones((C, K), dtype=np.float32)
        for c in range(C):
            for k in range(K):
                template = shape_templates[k % len(shape_templates)]
                base = amp * float(subtype_weights[c, k]) * template(knot_positions)
                base = base + rng.normal(scale=0.1, size=P).astype(np.float32)
                base = base - base.mean()   # per-factor zero-mean
                w_ckp[c, k, :] = base.astype(np.float32)

    # K_B-smooth the deterministic templates so they
    # match the smoothness prior in SHAPE while retaining
    # (optionally, via rescale) the original amplitude for recoverable signal.
    # Applied AFTER the template / rich_nu branches, AFTER the smoothness-prior
    # branch (which bypasses templates entirely). No-op when
    # w_ckp_kb_smooth_strength == 0.
    if float(w_ckp_kb_smooth_strength) > 0.0 and not w_ckp_paper_prior:
        from elpac.utils.bspline import second_difference_penalty as _sd_pen
        K_B_mat = np.asarray(_sd_pen(P, ridge=0.0))                      # (P, P)
        smoother = np.linalg.inv(
            np.eye(P) + float(w_ckp_kb_smooth_strength) * K_B_mat
        )                                                                  # (P, P)
        for c in range(C):
            for k in range(K):
                orig = w_ckp[c, k].astype(np.float64)
                smooth = smoother @ orig                                   # (P,)
                if bool(w_ckp_kb_smooth_preserve_amplitude):
                    orig_norm = np.linalg.norm(orig)
                    smooth_norm = np.linalg.norm(smooth)
                    if smooth_norm > 1e-8:
                        smooth = smooth * (orig_norm / smooth_norm)
                smooth = smooth - smooth.mean()                           # (S4) zero-mean
                w_ckp[c, k, :] = smooth.astype(np.float32)

    # Idiosyncratic noise D_c
    log_D = rng.normal(loc=-1.2, scale=0.3, size=(C, R)).astype(np.float32)
    D_c = np.exp(log_D) * noise_scale

    # --- Forward generative process: X -> c | X -> s | c, X -> Y -----
    # Paper model dependence:
    # (1) Sample covariates X = [age_norm, sex, APOE4, edu] with wide span
    # (2) Sample subtype c | X via softmax(alpha_c + X @ beta_c^T)
    # (3) Sample stage s | c, X via logit(s) ~ N(f_c*age + h_c^T W, sigma_c)
    # Anchors are NOT a separately generated cohort. Instead, after running
    # the generative process on N_pool >> n subjects we POST-SELECT the
    # highest-stage subjects per subtype as anchors (they represent the
    # clinical endpoints) and the lowest-stage subjects as NC.

    # Truth hyperparameters for subtype assignment
    alpha_c_truth = np.zeros(C, dtype=np.float32)
    beta_c_sub_truth = np.zeros((C, shape.p_x), dtype=np.float32)
    if C >= 1:
        beta_c_sub_truth[0] = np.array([ 1.5, 0.0, -2.5,  2.0] + [0.0] * (shape.p_x - 4))[:shape.p_x]  # SuperAger
    if C >= 2:
        beta_c_sub_truth[1] = np.array([ 0.5, 0.0, -0.5,  0.0] + [0.0] * (shape.p_x - 4))[:shape.p_x]  # Normal
    if C >= 3:
        beta_c_sub_truth[2] = np.array([ 0.3, 0.0,  2.5, -1.5] + [0.0] * (shape.p_x - 4))[:shape.p_x]  # AD
    if C >= 4:
        beta_c_sub_truth[3] = np.array([-1.5, 0.0,  1.5, -0.5] + [0.0] * (shape.p_x - 4))[:shape.p_x]  # EOAD
    if C > 4:
        beta_c_sub_truth[4:] = rng.normal(size=(C - 4, shape.p_x), scale=0.5).astype(np.float32)

    f_c_truth_vec = np.array([0.3, 0.6, 1.1, 1.4], dtype=np.float32)[:C]
    h_c_truth = np.zeros((C, shape.p_w), dtype=np.float32)
    if shape.p_w >= 2:
        h_c_truth[:, 1] = np.array([-0.2, 0.3, 0.9, 0.7])[:C]
    sigma_c_truth = np.array([1.3, 1.3, 1.3, 1.3], dtype=np.float32)[:C]

    def _softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    # Oversample pool so anchor post-selection finds enough high-stage subjects
    N_pool = max(n * 6, 3000)
    pool_ages = rng.uniform(40.0, 95.0, size=N_pool).astype(np.float32)
    pool_X = np.zeros((N_pool, shape.p_x), dtype=np.float32)
    pool_X[:, 0] = (pool_ages - 70.0) / 10.0
    pool_X[:, 1] = rng.integers(0, 2, size=N_pool).astype(np.float32)
    pool_X[:, 2] = (rng.uniform(size=N_pool) < 0.30).astype(np.float32)
    if shape.p_x >= 4:
        pool_X[:, 3] = rng.normal(size=N_pool, scale=1.0).astype(np.float32)

    # Subtype | X
    pool_logits = alpha_c_truth[None, :] + pool_X @ beta_c_sub_truth.T   # (N_pool, C)
    pool_probs = _softmax(pool_logits)
    u = rng.uniform(size=N_pool)
    cumu = np.cumsum(pool_probs, axis=1)
    pool_subtypes = (u[:, None] < cumu).argmax(axis=1).astype(np.int32)

    # Stage | c, X
    pool_W = pool_X[:, 1:1 + shape.p_w]
    pool_mean_logit = (
        f_c_truth_vec[pool_subtypes] * pool_X[:, 0]
        + np.einsum("nk,nk->n", h_c_truth[pool_subtypes], pool_W)
    )
    pool_logit_s = rng.normal(loc=pool_mean_logit, scale=sigma_c_truth[pool_subtypes]).astype(np.float32)
    pool_stages = 1.0 / (1.0 + np.exp(-pool_logit_s))

    # Post-select subjects:
    # - per-subtype top-k-highest stage → anchor_c
    # - globally lowest-k stage → NC (regardless of true subtype)
    # - random remaining → general cohort (hidden labels)
    selected_global = np.zeros(N_pool, dtype=bool)
    final_idx = []
    final_anchor_type = []
    for c in range(C):
        mask_c = (pool_subtypes == c) & (~selected_global)
        if mask_c.sum() < n_anchor_per_subtype:
            raise RuntimeError(f"Not enough subjects in subtype {c} for anchors")
        candidates = np.where(mask_c)[0]
        top_k = candidates[np.argsort(-pool_stages[candidates])[:n_anchor_per_subtype]]
        for t in top_k:
            selected_global[t] = True
            final_idx.append(int(t))
            final_anchor_type.append(int(c))

    nc_remaining = np.where(~selected_global)[0]
    if len(nc_remaining) < n_nc:
        raise RuntimeError("Not enough pool subjects for NC selection")
    nc_sorted = nc_remaining[np.argsort(pool_stages[nc_remaining])[:n_nc]]
    for t in nc_sorted:
        selected_global[t] = True
        final_idx.append(int(t))
        final_anchor_type.append(int(C))

    n_general = n - (C * n_anchor_per_subtype) - n_nc
    gen_remaining = np.where(~selected_global)[0]
    if len(gen_remaining) < n_general:
        raise RuntimeError("Not enough pool subjects for general cohort")
    gen_choice = rng.choice(gen_remaining, size=n_general, replace=False)
    for t in gen_choice:
        selected_global[t] = True
        final_idx.append(int(t))
        final_anchor_type.append(-1)

    final_idx = np.array(final_idx, dtype=np.int64)
    # Assemble final cohort
    anchor_type = np.array(final_anchor_type, dtype=np.int32)
    subtypes = pool_subtypes[final_idx].astype(np.int32)
    stages = pool_stages[final_idx].astype(np.float32)
    stages = np.clip(stages, 0.02, 0.98)
    ages = pool_ages[final_idx]
    X = pool_X[final_idx]
    W = X[:, 1:1 + shape.p_w]

    # Observation
    Y = np.zeros((n, R), dtype=np.float32)
    C_cog = np.full(n, np.nan, dtype=np.float32)
    psi_eval_cache = {}   # cache basis at each s to reduce repeat work

    # Per-subtype cubic polynomial cognition coefficients (only used if
    # cognition_model='polynomial'). Softplus-positive by construction, so
    # chi_c(s) is monotone increasing in s for each subtype.
    if include_cognition and cognition_model == "polynomial":
        # Design: chi_c(1) = sum_j softplus(eta_{c,j}) ≈ 16 to match the linear
        # baseline's dynamic range (baseline = 2 + 16*s). Softplus(x) ≈ x for
        # x >> 0, so target eta_j ≈ 5-6 per coefficient sums to ~16-18.
        # Per-subtype variation lets the model distinguish subtypes via chi_c.
        chi_c_raw_true = rng.normal(loc=np.array([5.0, 5.0, 5.5]), scale=0.3,
                                    size=(C, 3)).astype(np.float32)
        # np.logaddexp(0, x) is numerically stable softplus.
        chi_c_true = np.logaddexp(0.0, chi_c_raw_true)  # (C, 3), softplus-positive
    elif include_cognition and cognition_model == "ispline":
        # chi_c is a monotone I-spline with nonneg
        # coefficients. Sample per-subtype softplus-pre-image so chi_c is
        # always monotone non-decreasing with chi_c(0) = 0. Design the mean
        # loc so sum of coefficients ~ 16 to match the polynomial dynamic
        # range (softplus(x) ≈ x for x >> 0; target 16 / ispline_n_basis per
        # coefficient).
        target = 16.0 / float(ispline_n_basis)
        # Per-subtype offsets (0, 0, +0.5) to preserve across-subtype contrast
        c_offset = np.array([0.0, 0.0, 0.5] + [0.0] * max(0, C - 3),
                            dtype=np.float32)[:C]
        chi_c_raw_true = (
            rng.normal(loc=target + c_offset[:, None], scale=0.3,
                       size=(C, ispline_n_basis)).astype(np.float32)
        )
        # np.logaddexp(0, x) is numerically stable softplus.
        chi_c_true = np.logaddexp(0.0, chi_c_raw_true)  # (C, n_basis), softplus-positive
    else:
        chi_c_true = None

    # Precompute I-spline basis evaluations per unique stage (only if needed)
    ispline_eval_cache: dict = {}

    for i in range(n):
        c_i = int(subtypes[i])
        s_i = float(stages[i])
        # Evaluate mean trajectory theta(s_i) via cubic Hermite within shell
        j = min(int(s_i * J), J - 1)
        u = (s_i - tau_grid[j]) / delta_j
        theta_i = cubic_hermite_interpolate(
            jnp.asarray(theta_shells[c_i, j]),
            jnp.asarray(theta_shells[c_i, j + 1]),
            jnp.asarray(V[c_i, j]),
            jnp.asarray(V[c_i, j + 1]),
            jnp.asarray(u),
            delta_j=float(delta_j),
        )
        theta_i = np.asarray(theta_i)
        # Evaluate activations nu_k(s_i)
        if s_i not in psi_eval_cache:
            psi_eval_cache[s_i] = np.asarray(eval_bspline_basis(P, s_i))
        psi_vec = psi_eval_cache[s_i]                                # (P,)
        nu_vec = np.exp(np.clip(w_ckp[c_i] @ psi_vec, -8.0, 8.0))   # (K,)
        # Sigma = A diag(nu) A^T + diag(D)
        A_scaled = A_c[c_i] * np.sqrt(nu_vec)[None, :]                # (R, K)
        Sigma = A_scaled @ A_scaled.T + np.diag(D_c[c_i])             # (R, R)
        # Sample
        L = np.linalg.cholesky(Sigma + 1e-4 * np.eye(R))
        eps = rng.normal(size=R).astype(np.float32)
        Y[i] = theta_i + L @ eps

        # Cognitive observation (optional). CDR-SB convention: higher = worse.
        if include_cognition:
            if cognition_model == "polynomial":
                # Matches model's generative process: per-subtype cubic polynomial
                # E[C_i] = gamma_0 + gamma_X^T X_i + chi_{c_i}(s_i),
                # where chi_c(s) = sum_j softplus(eta_{c,j}) * s^j has softplus-positive coeffs.
                s_basis = np.array([s_i, s_i ** 2, s_i ** 3], dtype=np.float32)
                chi_s = float(chi_c_true[c_i] @ s_basis)
                baseline = 2.0 + 0.3 * X[i, 2] + chi_s
            elif cognition_model == "ispline":
                # chi_c is a monotone I-spline. Evaluate the
                # I-spline basis at s_i (cached) and dot with the softplus-positive
                # coefficients. chi_c(s) is monotone non-decreasing with chi_c(0)=0.
                if s_i not in ispline_eval_cache:
                    ispline_eval_cache[s_i] = np.asarray(
                        eval_ispline_basis(ispline_n_basis, s_i, degree=ispline_degree)
                    )
                is_basis = ispline_eval_cache[s_i]
                chi_s = float(chi_c_true[c_i] @ is_basis)
                baseline = 2.0 + 0.3 * X[i, 2] + chi_s
            else:
                # Linear baseline (original): E[C_i] = 2 + 16*s_i + 0.3*X[i,2]
                baseline = 2.0 + 16.0 * s_i + 0.3 * X[i, 2]
            C_cog[i] = baseline + rng.normal(scale=cognition_noise)

    # Missingness
    mask = np.ones((n, R), dtype=np.float32)
    if missing_rate > 0.0:
        # Block missingness by modality: each subject has independent prob
        # of losing an entire modality at rate `missing_rate`.
        mod_boundaries = [0, shape.R_A, shape.R_A + shape.R_T, shape.R_A + shape.R_T + shape.R_N, R]
        for i in range(n):
            for m in range(4):
                if rng.uniform() < missing_rate:
                    start, end = mod_boundaries[m], mod_boundaries[m + 1]
                    mask[i, start:end] = 0.0
                    Y[i, start:end] = 0.0  # zero-fill missing positions

    # Cognition missingness. Independently set a fraction of
    # cognitive scores to NaN — exercises the model's C_cog_mask branch. Only
    # meaningful when cognition was observed in the first place; NaNs from
    # `include_cognition=False` already dominate. The model downstream reads
    # `np.isfinite(truth.C_cog)` as the mask (see inference.py:231), so NaN is
    # the propagation contract — no separate mask field is required.
    if include_cognition and cognition_missing_rate > 0.0:
        drop = rng.uniform(size=n) < cognition_missing_rate
        C_cog[drop] = np.nan

    # Ground-truth global pathological severity G*_i.
    # Computed on the true θ*_c trajectory:
    # per-modality standardized Euclidean displacement from the s=0 anchor,
    # averaged over modalities, normalized by the max terminal displacement.
    G_star_truth = compute_G_star_path_from_truth(
        theta_c_shells=theta_shells,
        stages=stages,
        subtypes=subtypes,
        shape=shape,
    )

    return SyntheticTruth(
        shape=shape,
        stages=stages,
        subtypes=subtypes,
        anchor_type=anchor_type,
        A_c=A_c,
        w_ckp=w_ckp,
        D_c=D_c,
        theta_c_shells=theta_shells,
        v_c_shells=V,
        beta_velocity=beta_velocity,
        onset_times=onset_times,
        sharpness=sharpness,
        delta_region=delta_region,
        alpha_c_truth=alpha_c_truth,
        beta_c_sub_truth=beta_c_sub_truth,
        f_c_age_truth=f_c_truth_vec,
        h_c_truth=h_c_truth,
        sigma_c_stage_truth=sigma_c_truth,
        X=X,
        W=W,
        Y=Y,
        mask=mask,
        C_cog=C_cog,
        ages=ages,
        Theta_0=Theta_0,
        phase_transition_s_star=phase_transition_s_star,
        G_star_truth=G_star_truth,
        # Paper.1 alias fields: same arrays under manuscript-aligned names.
        t_cr_true=onset_times,
        omega_cr_true=sharpness,
        beta_cr_q_true=beta_velocity,
    )
