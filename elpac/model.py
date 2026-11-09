"""NumPyro model for elpac.

Implements the generative process:

 Sigma_c(s) = A_c diag(nu_c(s)) A_c^T + diag(D_c)

with:
 - lower-triangular positive-diagonal A_c with multiplicative gamma shrinkage,
 - nu_{c,k}(s) = exp(sum_p w_{c,k,p} psi_p(s)) with per-factor normalized w,
 - diagonal idiosyncratic noise D_c with log-Normal prior,
 - cubic Hermite trajectory between B-spline-generated shell velocities,
 - Concrete (relaxed Bernoulli) region gates delta_{c,r},
 - manual marginalization over discrete subtype c_i (via logsumexp),
 - imaging observation with Sylvester/Woodbury Gaussian log-pdf,
 - optional monotone cognition submodel (CDR-SB convention: +chi, later stage -> higher score).

The model has TWO MODES controlled by `phase`:
 phase = "phase1" : forces A_c = 0 and w = 0 (equivalent); Sigma_c = diag(D_c).
 phase = "phase2" : full stage-varying factor covariance.

Amortization via a flax recognition network is done OUTSIDE the numpyro model
(see src/inference.py); `numpyro.plate("subjects",...)` here uses local
variational sites emitted by an AutoGuide by default, and the custom amortized
guide overrides those sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from .utils.bspline import (
    build_bspline_eval_matrix,
    build_ispline_eval_matrix,
    cubic_hermite_interpolate,
    second_difference_penalty,
)
from .utils.constraints import make_lower_triangular_A, make_lower_triangular_A_direct
from .utils.covariance import gaussian_logpdf_sylvester_woodbury
from .utils.spatial_kernel import build_per_modality_kernels


@dataclass
class ModelConfig:
    """Shape + hyperparameter container for model.py.

 This is the same set of dimensions the simulator uses, plus a few
 hyperparameters.

 Selected fields:

 - `w_anchor`: likelihood weight on anchor subjects.
 - `eps_pinning`: epsilon_S, std of the anchor / healthy-reference stage
 priors on the logit scale.
 - `d_min`: floor on the idiosyncratic noise variances D_c.
 - `delta_shrink_shape_k1`, `delta_shrink_shape`: Gamma shapes a_1 (k=1)
 and a_2 (k>=2) of the multiplicative-gamma shrinkage on loadings.
 - `theta0_prior_std_override`: None -> NC-sample std diag (Omega^NC);
 float -> scalar std (default 0.5).
 - `mu_c_prior_std_override`: None -> anchor-sample std per subtype diag
 (tilde_Omega_c); float -> scalar std (default 0.5).
 """
    n: int
    C: int
    K: int
    J: int
    P: int
    Q: int
    R_A: int
    R_T: int
    R_N: int
    R_blood: int
    p_x: int
    p_w: int
    w_anchor: float = 5.0
    gumbel_tau: float = 1.0 # Annealed externally per SVI step
    use_cognition: bool = False
    log_d_prior_mean: float = -1.0
    log_d_prior_std: float = 1.0
    tau_smoothness_scale: float = 1.0
    tau_k_scale: float = 0.1 # HalfNormal scale for tau_k shrinkage prior
 # Benchmarked: 0.3 gives better activation recovery than 0.6 or 1.0
 # (activation_sup 0.33 vs 0.50 vs 0.53 at paper scale with low_rank guide)
    delta_shrink_shape: float = 1.5 # Gamma shape a_2 for kappa_{c,k>=2}
 # prior on delta_shrink (Bhattacharya-Dunson). The paper sets a_2 = 1.5 for
 # k >= 2.
 # Shape <= 0 blows up silently inside numpyro (NaN losses many layers down)
 # because psi = cumprod(Gamma(shape, 1)) drives a_flat's scale to 1/sqrt(NaN).
 # Validate at construction so the failure is loud.
 #
 # `delta_shrink_shape_k1` is a separate shape a_1 for kappa_{c,1} (k=1),
 # so it can be set independently of a_2.
    delta_shrink_shape_k1: float = 1.5
    eps_pinning: float = 0.1
    d_min: float = 1e-4
    Omega_0_soft_std: float = 0.5
    theta0_prior_std_override: Optional[float] = 0.5 # None = NC sample std; float = scalar
    mu_c_prior_std_override: Optional[float] = 0.5 # None = anchor sample std; float = scalar
    nc_subtype_idx: int = -1 # -1 = origin subjects identified by anchor_type==C (control_label path);
 # >=0 = that subtype index is treated as NC origin (S≈0, Theta_0 prior)

 # When True:
 # - sigma_c_stage prior switches from log-Normal (TransformedDistribution(
 # Normal(0, 1), Exp)) to Half-Cauchy(0, 1).
 # - The [0.5, 1.5] training-stability clamp on sigma_c_stage in the
 # structured-stage likelihood path is removed (the
 # unclamped σ_c as the std of logit(S_i | c=c)).
 # Both fixes MUST toggle together — the clamp was a workaround for the
 # log-Normal tail, and the Half-Cauchy posterior is expected to land
 # inside [0.5, 1.5] organically (median 1.0).
 # Default True. Callers that need the legacy behavior must pin
 # use_paper_sigma_c=False explicitly.
    use_paper_sigma_c: bool = True

 # When True (paper Eq. 15):
 # w_{c,k,:} ~ N(0, tau_k^2 (K_B + eps_B I)^{-1})
 # where K_B is the P×P second-difference penalty matrix (approximate
 # Gaussian-random-walk prior on the activation coefficients). Implements
 # smoothness between adjacent B-spline knots so nu_c(s) varies smoothly.
 # The sample-site name switches from `w_ckp_raw` (iid Normal) to
 # `w_ckp_raw_std` (iid standard Normal pre-Cholesky); `w_ckp` remains
 # the deterministic per-factor-normalized output consumed downstream.
 # Default True. Same legacy pinning rule as use_paper_sigma_c above.
    use_paper_w_ckp_smoothness: bool = True
    eps_B: float = 1e-3 # ridge for K_B + eps_B I invertibility

 # When True, replaces the iid per-(c, r) onset prior
 # onset_times[c, r] ~ sigmoid(Normal(0, 1))
 # with modality-level onset centers and a soft cascade A < T < N ordering:
 # t̃_{c,A} ~ N(μ_A, σ_m²) (anchored at μ_A)
 # t̃_{c,T} = t̃_{c,A} + softplus(raw_T - t̃_{c,A})
 # t̃_{c,N} = t̃_{c,T} + softplus(raw_N - t̃_{c,T}) (strict A < T < N)
 # t̃_{c,B} ~ N(μ_B, σ_m²) (blood UNconstrained)
 # onset_times[c, r] = sigmoid(t̃_{c, m(r)} + η_{c, r})
 # with η ~ N(0, cascade_eta_sigma²) the per-biomarker deviation. Blood
 # center is NOT constrained to be after N.
 # New sample sites: `t_ATN_raw` (C, 3), `t_tilde_B` (C,), `onset_eta`
 # (C, R). `onset_times` becomes a `numpyro.deterministic` site so
 # downstream consumers (extract_posterior_full / recovery metrics) still
 # see it under the same name.
 # Default True. Legacy callers should pass ``use_paper_cascade_prior=False``.
    use_paper_cascade_prior: bool = True
 # Cognition submodel χ_c(s):
 # "ispline" — monotone I-spline with non-negative coefficients (paper default).
 # "polynomial" — softplus-polynomial in [s, s², s³] (legacy).
 # Must be paired with the matching simulator setting.
    chi_c_basis: str = "ispline"
    chi_c_ispline_n_basis: int = 3 # Number of I-spline basis functions
    chi_c_ispline_degree: int = 2 # Polynomial degree of the underlying M-spline
 # Onset-center priors: μ_A=0.3, μ_T=0.5, μ_N=0.7, σ_m=0.15.
    cascade_mu_A: float = 0.3 # modality-A onset center
    cascade_mu_T: float = 0.5 # modality-T onset center
    cascade_mu_N: float = 0.7 # modality-N onset center
    cascade_mu_B: float = 0.35 # blood unconstrained; matches simulator's 0.35
    cascade_sigma_m: float = 0.15 # per-modality std σ_m around center
    cascade_eta_sigma: float = 0.05 # per-biomarker deviation η_{c,r} std

 # When True, replaces the iid `onset_times[c, r] ~ sigmoid(N(0, 1))`
 # and `beta_vel[c, r, q] ~ N(0, 2)` priors with modality-block multivariate
 # Normals (onset handled in pre-sigmoid η space, velocity in raw space):
 # η_{c, R_m} ~ N(t_tilde_{c,m} · 1, sigma_eta^2 · Σ^(m))
 # beta_{c, R_m, q} ~ N(0, sigma_beta^2 · Σ^(m))
 # Σ^(m) is the per-modality kernel from `build_per_modality_kernels`
 # (RBF / Laplacian / CAR / Identity; length scale `spatial_length_scale`, default 0.3). This is PAIRED
 # with the simulator flag `use_spatial_sim` — enabling one without the
 # other will over-shrink an unstructured truth (or fail to capture
 # structured truth). Default True. Legacy callers must pass both
 # False together to restore the iid priors.
    use_paper_spatial_priors: bool = True
    spatial_length_scale: float = 0.3 # Exponential kernel length scale
    spatial_nugget: float = 1e-3 # Ridge added to Σ^(m) for Cholesky
    sigma_eta_spatial: float = 0.3 # Scale on onset-time MVN prior (σ_η)
    sigma_beta_spatial: float = 2.0 # Scale on velocity-coeff MVN prior
 # (matches the legacy iid Normal(0, 2.0)
 # on beta_vel so marginal variance is
 # preserved as length_scale → 0).

 # External kernel path. When non-None, the model
 # loads per-modality kernels [K_A, K_T, K_N, K_blood] from this NPZ file
 # and uses them in place of the built-in RBF / Laplacian / CAR / Identity
 # kernels. The NPZ must contain keys 'K_A', 'K_T', 'K_N', 'K_blood' with
 # square matrices of shapes (R_A, R_A), (R_T, R_T), (R_N, R_N),
 # (R_blood, R_blood). Only consumed when `use_paper_spatial_priors=True`.
 # Storing a PATH (not the array) keeps ModelConfig JSON-serializable for
 # paper-fit reproducibility — ModelConfig instances are serialized into
 # sidecar JSONs, so embedding raw (potentially large) arrays would defeat
 # that purpose.
 # Default None preserves the prior paper-kernel behavior.
 # PAIRED: if set, the simulator should receive the same NPZ-loaded list
 # via `simulate_synthetic(..., external_kernels=[...])`.
    external_kernels_path: Optional[str] = None

 # When True, replaces the default beta_c_sub ~ N(0, 1) subtype-coefficient
 # prior with the clinically-motivated AD-specific prior on the age column
 # of X (EOAD/LOAD/HA), per the paper. Other coefficients keep N(0, 1).
 # Only applied when cfg.p_x == 4 (age, sex, apoe, educ); otherwise falls
 # back to the default N(0, 1) prior.
    use_informative_age_prior: bool = False

    def __post_init__(self):
        if self.delta_shrink_shape <= 0:
            raise ValueError(
                f"delta_shrink_shape must be > 0 (Gamma shape); "
                f"got {self.delta_shrink_shape}"
            )
        if self.delta_shrink_shape_k1 <= 0:
            raise ValueError(
                f"delta_shrink_shape_k1 must be > 0 (Gamma shape); "
                f"got {self.delta_shrink_shape_k1}"
            )
        if self.eps_pinning <= 0:
            raise ValueError(f"eps_pinning must be > 0; got {self.eps_pinning}")
        if self.nc_subtype_idx != -1 and not (0 <= self.nc_subtype_idx < self.C):
            raise ValueError(
                f"nc_subtype_idx must be -1 or in [0, C-1]; got {self.nc_subtype_idx} with C={self.C}"
            )
 # Semantic safeguard — Θ₀ empirical prior must come from a healthy
 # startpoint cohort, not from a subtype's disease endpoint anchors. When
 # nc_subtype_idx>=0, the prior is routed through subtype c's anchor rows,
 # which are by construction disease endpoints (anchor_type==c at S≈0.95).
 # For real-data C=3 runs with a separate NC startpoint cohort
 # (control_label set or --nc-startpoint enabled), use nc_subtype_idx=-1
 # (default) so the model reads Θ₀ from anchor_type==C rows.
        if self.nc_subtype_idx >= 0:
            import warnings
            warnings.warn(
                f"nc_subtype_idx={self.nc_subtype_idx} routes Θ₀ empirical prior through "
                f"subtype c={self.nc_subtype_idx}'s anchor subjects. Paper semantics require "
                f"Θ₀ to represent a healthy startpoint, not a disease endpoint. "
                f"For real-data C=3 runs with a separate NC startpoint cohort, use "
                f"nc_subtype_idx=-1 (default).",
                UserWarning, stacklevel=2,
            )
        if self.d_min <= 0:
            raise ValueError(f"d_min must be > 0; got {self.d_min}")
        if self.Omega_0_soft_std <= 0:
            raise ValueError(
                f"Omega_0_soft_std must be > 0; got {self.Omega_0_soft_std}"
            )
        if self.eps_B <= 0:
            raise ValueError(f"eps_B must be > 0; got {self.eps_B}")
        if self.cascade_sigma_m <= 0:
            raise ValueError(
                f"cascade_sigma_m must be > 0; got {self.cascade_sigma_m}"
            )
        if self.cascade_eta_sigma <= 0:
            raise ValueError(
                f"cascade_eta_sigma must be > 0; got {self.cascade_eta_sigma}"
            )
        if self.chi_c_basis not in ("polynomial", "ispline"):
            raise ValueError(
                f"chi_c_basis must be 'polynomial' or 'ispline'; "
                f"got {self.chi_c_basis!r}"
            )
        if self.chi_c_ispline_n_basis < 1:
            raise ValueError(
                f"chi_c_ispline_n_basis must be >= 1; "
                f"got {self.chi_c_ispline_n_basis}"
            )
 # structured_stage: if True, use per-(subject, subtype) logit_S with shape
 # (n, C) — the subtype-conditional stage prior of the paper:
 # logit(S_i | c=c) ~ N(f_c(Age_i) + h_c^T W_i, sigma_c^2)
    structured_stage: bool = True
    joint_cog_marginalization: bool = False # If True, jointly marginalize
 # imaging + cognition via logsumexp (matches paper Eq 3). If False (default),
 # use conditional-expectation lower bound E_{p(c|Y,X)}[log p(C|c,S)].
    train_q_c: bool = False # Train encoder subtype head via Dirichlet prior
    wider_encoder: bool = False # Use wider encoder (256,256,128) instead of (128,128)
    anchor_ce_supervision: bool = False # Add supervised CE loss on anchor q_c
    anchor_ce_weight: float = 1.0 # Weight for supervised anchor CE loss (mean-reduced)
 # Backward-compat alias: enhanced_encoder = all three (wider + CE + longer training)
    enhanced_encoder: bool = False
    q_c_anchor_concentration: float = 100.0 # Dirichlet peak for anchors (higher = stronger)
    forward_time: bool = False # M2-forward ablation: integrate forward from Theta_0
 # instead of backward from Theta_J. Breaks anchor-endpoint identifiability;
 # expected to lower stage_rho vs. the reverse-time baseline.

 # ---------------------------------------------------------------------
 # Stage E1.B weak_dep fixes (user decision #119). Defaults preserve the
 # existing generative process.
 #
 # psi_shrink_scale: multiplier applied to the psi_shrink vector that
 # sets the prior scale for a_flat / a_diag_raw (see model.py lines
 # 432-440). psi_shrink_scale<1 multiplies psi_shrink by 1/scale**2 (via
 # division of the std by sqrt(scale)), producing TIGHTER shrinkage on
 # the factor loadings. Set to 0.25 (≡ std scaled by sqrt(0.25)=0.5)
 # to cut the prior std in half on a_flat columns. NOTE: this only
 # tightens the spherical a_flat prior; delta_shrink itself is NOT
 # touched so Bhattacharya-Dunson multiplicative ordering is preserved.
 #
 # A_frobenius_penalty: if > 0, adds a numpyro.factor soft penalty that
 # subtracts A_frobenius_penalty * sum_c ||A_c||_F^2 from the log-joint.
 # This acts like a Gaussian ridge on the full A_c matrix and INTERACTS
 # with the existing Bhattacharya-Dunson prior on (a_flat, a_diag_raw).
 # When both shrinkages are active the effective prior on a_flat is
 # N(0, 1/sqrt(psi_shrink * psi_shrink_scale + 2*penalty)); the penalty
 # shrinks equally across K (no column ordering) so it can be combined
 # with psi_shrink_scale (which preserves the multiplicative ordering
 # per k) without canceling it.
    psi_shrink_scale: float = 1.0
    A_frobenius_penalty: float = 0.0

 # ---------------------------------------------------------------------
 # Phase 1a ARD prior on A_c columns (user decision, Apr 20 2026).
 # When use_ard_on_A=True, the Bhattacharya-Dunson multiplicative shrinkage
 # (psi_shrink) on (a_flat, a_diag_raw) is REPLACED by per-(C, K)
 # automatic-relevance-determination precisions alpha_{c,k} ~ Gamma(a, b).
 # The prior on a_flat / a_diag_raw columns becomes
 # a_flat[c, r, k] ~ N(0, 1/sqrt(alpha[c, k]))
 # a_diag_raw[c, k] ~ N(0, 1/sqrt(alpha[c, k]))
 # which at K_fit > K_truth drives alpha[c, k>=K_truth] to large values
 # (column collapse). psi_shrink_scale / A_frobenius_penalty are ignored
 # when use_ard_on_A=True (the ARD prior supersedes them).
 # Defaults keep the existing psi_shrink path.
    use_ard_on_A: bool = False
    ard_a_shape: float = 1.0
    ard_b_rate: float = 1.0
 # a_{c,k,k} ~ HalfNormal(ψ_{c,k}^{-1/2}) (paper Sec. 2.4)
 # sampled DIRECTLY (no Normal+softplus). Callers that need the legacy
 # behavior (posterior `keep_sites` lists that hard-code `a_diag_raw`)
 # must pass `use_paper_halfnormal_diag=False` explicitly.
    use_paper_halfnormal_diag: bool = True

    @property
    def R(self) -> int:
        return self.R_A + self.R_T + self.R_N + self.R_blood

    @property
    def modality_of_r(self) -> np.ndarray:
        return np.concatenate([
            np.full(self.R_A, 0, dtype=np.int32),
            np.full(self.R_T, 1, dtype=np.int32),
            np.full(self.R_N, 2, dtype=np.int32),
            np.full(self.R_blood, 3, dtype=np.int32),
        ])


def _modality_sign_vector(cfg: ModelConfig) -> jnp.ndarray:
    """(+1 / -1) sign vector applied inside softplus for velocity transforms.

 Convention: A, T = +; N = -; first blood = - (Aβ42/40), rest = +.
 """
    signs = np.ones(cfg.R, dtype=np.float32)
    boundaries = [
        (0, cfg.R_A),
        (cfg.R_A, cfg.R_A + cfg.R_T),
        (cfg.R_A + cfg.R_T, cfg.R_A + cfg.R_T + cfg.R_N),
        (cfg.R_A + cfg.R_T + cfg.R_N, cfg.R),
    ]
 # structural -> -1
    start, end = boundaries[2]
    signs[start:end] = -1.0
 # blood first entry -> -1 (Aβ42/40 convention)
    b_start, b_end = boundaries[3]
    if b_end > b_start:
        signs[b_start] = -1.0
    return jnp.asarray(signs, dtype=jnp.float32)


def _velocity_from_shell_boundaries(
    B_shell: jnp.ndarray, # (J+1, Q)
    beta: jnp.ndarray, # (C, R, Q)
    onset_times: jnp.ndarray, # (C, R)
    sharpness: jnp.ndarray, # (C, R)
    delta_region: jnp.ndarray, # (C, R) in [0, 1] (relaxed Bernoulli)
    tau_grid: jnp.ndarray, # (J+1,)
    mod_signs: jnp.ndarray, # (R,) +1/-1
) -> jnp.ndarray:
    """Compute V_{c,j,r} at all shell boundaries."""
 # (C, R, J+1)
    raw = jnp.einsum("crq,jq->cjr", beta, B_shell)
    softplus_raw = jax.nn.softplus(raw)
    velocity_mag = mod_signs[None, None, :] * softplus_raw # (C, J+1, R)
    onset_arg = (tau_grid[None, :, None] - onset_times[:, None, :]) / jnp.clip(sharpness[:, None, :], 1e-3, None)
    onset = jax.nn.sigmoid(onset_arg) # (C, J+1, R)
    V = delta_region[:, None, :] * onset * velocity_mag # (C, J+1, R)
    return V


def _reverse_time_integration(theta_J: jnp.ndarray, V: jnp.ndarray, J: int) -> jnp.ndarray:
    """Scan Theta_{c,j-1} = Theta_{c,j} - (1/J) * V_{c,j} from j=J down to 1.

 theta_J : (C, R) — terminal state
 V : (C, J+1, R) — velocities at shell boundaries
 Returns : (C, J+1, R)
 """
    delta_j = 1.0 / J

 # We use lax.scan over descending shell indices.
 # carry = theta at current shell j; xs = V at the same j (used for step j -> j-1)
    def step(theta_j, V_j):
        theta_prev = theta_j - delta_j * V_j
        return theta_prev, theta_j # output the current (j) state

 # Scan from j = J down to 1; the velocities used to step from j to j-1 are V[:, j, :].
    V_rev = V[:, ::-1, :] # (C, J+1, R) reversed over j
 # We want to consume V_{c,J}, V_{c,J-1},..., V_{c,1} (length J steps)
    V_used = V_rev[:, :-1, :] # drop V[:, 0, :] because it's never used as the "from" step
 # Scan
    def scan_fn(carry, v_j):
        theta_prev = carry - delta_j * v_j
        return theta_prev, carry

 # Move leading "C" dim to the end for per-subject vectorization
 # Actually easier: transpose so the scan axis is the first, then retranspose.
    V_used_t = jnp.transpose(V_used, (1, 0, 2)) # (J, C, R)
    _, outs_t = jax.lax.scan(scan_fn, theta_J, V_used_t) # outs_t: (J, C, R)
 # Need to include the terminal state as well; final theta_{c,0} is the last carry.
    final, _ = jax.lax.scan(scan_fn, theta_J, V_used_t)
 # The "outputs" are the BEFORE-step states, i.e. theta_{c,J}, theta_{c,J-1},..., theta_{c,1}.
 # Stack with final theta_{c,0} to get full (J+1, C, R) in reverse order.
    shells_rev = jnp.concatenate([outs_t, final[None, :, :]], axis=0) # (J+1, C, R)
    shells = shells_rev[::-1] # (J+1, C, R) in forward j order
    shells = jnp.transpose(shells, (1, 0, 2)) # (C, J+1, R)
    return shells


def _forward_time_integration(theta_0: jnp.ndarray, V: jnp.ndarray, J: int) -> jnp.ndarray:
    """Scan Theta_{c,j} = Theta_{c,j-1} + (1/J) * V_{c,j} from j=1 up to J.

 theta_0 : (C, R) — origin state (shared healthy root, broadcast over C)
 V : (C, J+1, R)
 Returns : (C, J+1, R)
 """
    delta_j = 1.0 / J
    V_used = V[:, 1:, :] # (C, J, R)
    V_used_t = jnp.transpose(V_used, (1, 0, 2)) # (J, C, R)

    def scan_fn(carry, v_j):
        theta_next = carry + delta_j * v_j
        return theta_next, carry

    final, outs_t = jax.lax.scan(scan_fn, theta_0, V_used_t) # outs_t: (J, C, R) = theta_0.. theta_{J-1}
    shells_t = jnp.concatenate([outs_t, final[None, :, :]], axis=0) # (J+1, C, R)
    shells = jnp.transpose(shells_t, (1, 0, 2)) # (C, J+1, R)
    return shells


def _eval_trajectory_at_stage(
    theta_shells: jnp.ndarray, # (C, J+1, R)
    v_shells: jnp.ndarray, # (C, J+1, R)
    tau_grid: jnp.ndarray, # (J+1,)
    c_idx: int, # scalar subtype
    s: jnp.ndarray, # scalar stage in [0, 1]
    J: int,
) -> jnp.ndarray:
    """Cubic Hermite evaluation at (c, s) -> (R,). Uses `jnp.take` for shell lookup."""
    delta_j = 1.0 / J
    s_clipped = jnp.clip(s, 0.0, 1.0 - 1e-6)
    j_int = jnp.clip(jnp.floor(s_clipped * J).astype(jnp.int32), 0, J - 1)
    u = (s_clipped - j_int.astype(jnp.float32) / J) * J
    theta_left = theta_shells[c_idx, j_int, :]
    theta_right = theta_shells[c_idx, j_int + 1, :]
    v_left = v_shells[c_idx, j_int, :]
    v_right = v_shells[c_idx, j_int + 1, :]
    return cubic_hermite_interpolate(theta_left, theta_right, v_left, v_right, u, delta_j)


def _eval_activation_nu(
    w_ckp: jnp.ndarray, # (C, K, P)
    psi_s: jnp.ndarray, # (P,) basis at stage s
) -> jnp.ndarray:
    """nu_{c,k}(s) = exp(sum_p w_{c,k,p} psi_p(s)) — (C, K)."""
    g = jnp.einsum("ckp,p->ck", w_ckp, psi_s)
    return jnp.exp(jnp.clip(g, -8.0, 8.0))


def _imaging_log_lik_for_subject(
    y_i: jnp.ndarray, # (R,)
    mask_i: jnp.ndarray, # (R,)
    theta_i: jnp.ndarray, # (R,) mean trajectory at (c, s)
    A_c: jnp.ndarray, # (R, K)
    nu: jnp.ndarray, # (K,)
    D_c: jnp.ndarray, # (R,)
) -> jnp.ndarray:
    """Gaussian log-likelihood on observed coordinates, Sylvester/Woodbury."""
    residual = y_i - theta_i
    return gaussian_logpdf_sylvester_woodbury(residual, mask_i, A_c, nu, D_c)


def build_numpyro_model(cfg: ModelConfig, phase: str = "phase2"):
    """Return a NumPyro model closure with config captured.

 Phase 1 forces A_c = 0 (via w sampling) — equivalent to diagonal Sigma_c.
 Phase 2 uses the full stage-varying factor covariance.
 """
    assert phase in {"phase1", "phase2"}, phase
    R, C, K, J, P, Q = cfg.R, cfg.C, cfg.K, cfg.J, cfg.P, cfg.Q
    mod_signs = _modality_sign_vector(cfg)
    B_shell = build_bspline_eval_matrix(n_basis=Q, n_eval=J + 1)
    tau_grid = jnp.linspace(0.0, 1.0, J + 1)

 # precompute the I-spline design
 # matrix on the same (J+1)-point grid as `psi_basis_cache`, used for the
 # cognition submodel chi_c(s) when cfg.chi_c_basis == "ispline".
 # Shape (J+1, chi_c_ispline_n_basis). Earlier (default "polynomial")
 # keeps this cache unused.
    chi_basis_cache = build_ispline_eval_matrix(
        n_basis=cfg.chi_c_ispline_n_basis,
        n_eval=J + 1,
        degree=cfg.chi_c_ispline_degree,
    ) # (J+1, n_basis)

 # Precompute the Cholesky
 # scale factor for the smoothness prior
 # w_{c,k,:} ~ N(0, tau_k^2 (K_B + eps_B I)^{-1})
 # by building K_B + eps_B I, inverting, and factoring. Sampled via
 # reparameterization: w = tau_k * L @ w_std with w_std ~ N(0, I_P) and
 # L L^T = (K_B + eps_B I)^{-1}. Earlier (flag False) keeps iid Normal.
    if getattr(cfg, "use_paper_w_ckp_smoothness", False):
 # Pass ridge=0.0 so the only ridge is cfg.eps_B (avoid double
 # ridging the helper's default 1e-4).
        K_B = second_difference_penalty(P, ridge=0.0) # (P, P)
        W_prec = K_B + float(cfg.eps_B) * jnp.eye(P) # (P, P), SPD
        W_cov = jnp.linalg.inv(W_prec) # (P, P)
        W_cov_scale = jnp.linalg.cholesky(W_cov) # (P, P) lower-tri L
    else:
        W_cov_scale = None

 # Precompute Cholesky factors L_eta^(m), L_beta^(m) for per-modality MVN priors:
 # eta_{c, R_m} ~ N(mean, sigma_eta^2 · Σ^(m))
 # beta_{c, R_m, q} ~ N(0, sigma_beta^2 · Σ^(m))
 # Σ^(m) is the per-modality kernel from `build_per_modality_kernels`
 # (RBF / Laplacian / CAR / Identity).
 # Modalities with zero biomarkers get `None` slots; callers must skip.
    if getattr(cfg, "use_paper_spatial_priors", False):
 # If cfg.external_kernels_path is set, load the NPZ bundle
 # once at model-construction time (cached as a list of np.ndarray).
 # Keys: 'K_A', 'K_T', 'K_N', 'K_blood' — shapes validated inside
 # build_per_modality_kernels via the external_K assertion path.
        _ext_path = getattr(cfg, "external_kernels_path", None)
        if _ext_path is not None:
            _bundle = np.load(_ext_path)
            _ext_K = [
                np.asarray(_bundle["K_A"], dtype=np.float32),
                np.asarray(_bundle["K_T"], dtype=np.float32),
                np.asarray(_bundle["K_N"], dtype=np.float32),
                np.asarray(_bundle["K_blood"], dtype=np.float32),
            ]
        else:
            _ext_K = None
        Sigma_mods_np = build_per_modality_kernels(
            cfg.R_A, cfg.R_T, cfg.R_N, cfg.R_blood,
            length_scale=float(cfg.spatial_length_scale),
            nugget=float(cfg.spatial_nugget),
            external_K=_ext_K,
        )
        sigma_eta = float(cfg.sigma_eta_spatial)
        sigma_beta = float(cfg.sigma_beta_spatial)
        L_eta_mods = []
        L_beta_mods = []
        for Sm in Sigma_mods_np:
            if Sm.shape[0] == 0:
                L_eta_mods.append(None)
                L_beta_mods.append(None)
                continue
            Sm_j = jnp.asarray(Sm)
            L_eta_mods.append(
                jnp.linalg.cholesky((sigma_eta ** 2) * Sm_j
                                    + 1e-6 * jnp.eye(Sm_j.shape[0]))
            )
            L_beta_mods.append(
                jnp.linalg.cholesky((sigma_beta ** 2) * Sm_j
                                    + 1e-6 * jnp.eye(Sm_j.shape[0]))
            )
 # Modality block boundaries [lo, hi) — host-side ints, safe to use
 # as Python-level static slicing indices inside the model closure.
        _mod_sizes = (cfg.R_A, cfg.R_T, cfg.R_N, cfg.R_blood)
        _mod_bounds = []
        _lo = 0
        for _sz in _mod_sizes:
            _mod_bounds.append((_lo, _lo + _sz))
            _lo += _sz
    else:
        L_eta_mods = None
        L_beta_mods = None
        _mod_bounds = None

    def model(
        Y: jnp.ndarray, # (n, R)
        mask: jnp.ndarray, # (n, R)
        X: jnp.ndarray, # (n, p_x)
        W: jnp.ndarray, # (n, p_w)
        ages_norm: jnp.ndarray, # (n,)
        anchor_type: jnp.ndarray, # (n,) int; -1 = general, 0..C-1 = anchor_c, C = NC
        psi_basis_cache: jnp.ndarray, # (n, P) — basis evaluated at s=0.5 as an
 # implicit init; recognition nets override S_i
        gumbel_tau: Optional[jnp.ndarray] = None, # scalar, relaxed-Bernoulli temperature
        C_cog: Optional[jnp.ndarray] = None, # (n,) NaN where missing
        C_cog_mask: Optional[jnp.ndarray] = None, # (n,)
        ipw_weights: Optional[jnp.ndarray] = None, # (n,) per-subject IPW anchor weights
    ):
 # Default to cfg.gumbel_tau if not provided (backward compat)
        if gumbel_tau is None:
            gumbel_tau = jnp.float32(cfg.gumbel_tau)
        n = Y.shape[0]

 # ---------- Data-driven prior centers ----------
 # Theta_0 ~ N(mu^NC, Omega^NC), mu_c ~ N(bar_Y_c, tilde_Omega_c).
 # Empirical means and stds computed inline from the data (anchor_type
 # selects which rows are NC / anchor_c). The
 # prior covariance as the NC / anchor SAMPLE VARIANCE (diagonal).
        _nc_code = int(cfg.nc_subtype_idx) if cfg.nc_subtype_idx >= 0 else C
        is_nc_mask = (anchor_type == _nc_code).astype(jnp.float32)[:, None]
        nc_weight = is_nc_mask * mask
        nc_sum = jnp.sum(nc_weight, axis=0) # (R,)
        mu_NC_emp = jnp.sum(Y * nc_weight, axis=0) / (nc_sum + 1e-6)
        mu_NC_emp = jnp.where(jnp.isfinite(mu_NC_emp), mu_NC_emp, 0.0)
 # NC sample variance per coordinate (diagonal of Omega^NC).
 # Fall back to 1.0 where there are no NC observations (nc_sum<2).
        nc_var_raw = (
            jnp.sum(nc_weight * (Y - mu_NC_emp[None, :]) ** 2, axis=0)
            / jnp.maximum(nc_sum - 1.0, 1.0)
        )
        nc_std_emp = jnp.where(
            (nc_sum >= 2.0) & jnp.isfinite(nc_var_raw) & (nc_var_raw > 0.0),
            jnp.sqrt(jnp.maximum(nc_var_raw, 1e-6)),
            jnp.ones(R),
        )

        bar_Y_c_list = []
        anchor_std_list = []
        for c_idx in range(C):
            is_anc_c = (anchor_type == c_idx).astype(jnp.float32)[:, None]
            anc_weight = is_anc_c * mask
            anc_sum = jnp.sum(anc_weight, axis=0) # (R,)
            bar = jnp.sum(Y * anc_weight, axis=0) / (anc_sum + 1e-6)
            bar = jnp.where(jnp.isfinite(bar), bar, 0.0)
            bar_Y_c_list.append(bar)
            anc_var_raw = (
                jnp.sum(anc_weight * (Y - bar[None, :]) ** 2, axis=0)
                / jnp.maximum(anc_sum - 1.0, 1.0)
            )
            anc_std = jnp.where(
                (anc_sum >= 2.0) & jnp.isfinite(anc_var_raw) & (anc_var_raw > 0.0),
                jnp.sqrt(jnp.maximum(anc_var_raw, 1e-6)),
                jnp.ones(R),
            )
            anchor_std_list.append(anc_std)
        bar_Y_c = jnp.stack(bar_Y_c_list, axis=0)
        anchor_std_c = jnp.stack(anchor_std_list, axis=0) # (C, R)

 # Theta_0 ~ N(mu^NC, Omega^NC), mu_c ~ N(bar_Y_c, tilde_Omega_c).
 # Omega^NC = NC sample variance (diagonal); tilde_Omega_c = per-subtype
 # anchor sample variance (diagonal). When an override is supplied (e.g.
 # `theta0_prior_std_override=0.5`), use a scalar std instead — this path
 # reproduces the legacy behavior.
        if cfg.theta0_prior_std_override is not None:
            theta0_std = float(cfg.theta0_prior_std_override) * jnp.ones(R)
        else:
            theta0_std = nc_std_emp # (R,)
        if cfg.mu_c_prior_std_override is not None:
            mu_c_std = float(cfg.mu_c_prior_std_override) * jnp.ones((C, R))
        else:
            mu_c_std = anchor_std_c # (C, R)

        Theta_0 = numpyro.sample(
            "Theta_0",
            dist.Normal(mu_NC_emp, theta0_std).to_event(1),
        )
        mu_c = numpyro.sample(
            "mu_c",
            dist.Normal(bar_Y_c, mu_c_std).to_event(2),
        )

 # Subtype assignment parameters
        alpha_c = numpyro.sample(
            "alpha_c", dist.Normal(jnp.zeros(C), 1.0).to_event(1)
        )
        if cfg.use_informative_age_prior and cfg.p_x == 4 and C == 3:
 # Informative AD-specific prior on age coefficient (X column 0 =
 # age_norm) for the canonical 3-subtype (EOAD, LOAD, HA) layout.
 # EOAD: by definition age <65 → β_age strongly negative
 # LOAD: late-onset disease → β_age mildly positive
 # HA: healthy aging at any age → β_age mildly negative
            prior_mean = jnp.array([
                [-2.0, 0.0, 0.0, 0.0], # EOAD
                [+1.5, 0.0, 0.0, 0.0], # LOAD
                [-0.5, 0.0, 0.0, 0.0], # HA
            ])
            prior_std = jnp.array([
                [0.5, 1.0, 1.0, 1.0], # EOAD: tight age, loose other
                [0.5, 1.0, 1.0, 1.0], # LOAD
                [0.5, 1.0, 1.0, 1.0], # HA
            ])
            beta_c_sub = numpyro.sample(
                "beta_c_sub",
                dist.Normal(prior_mean, prior_std).to_event(2),
            )
        else:
            beta_c_sub = numpyro.sample(
                "beta_c_sub",
                dist.Normal(jnp.zeros((C, cfg.p_x)), 1.0).to_event(2),
            )

 # Stage prior params
        h_c = numpyro.sample(
            "h_c", dist.Normal(jnp.zeros((C, cfg.p_w)), 1.0).to_event(2)
        )
        f_c_age = numpyro.sample(
            "f_c_age", dist.Normal(jnp.zeros(C), 1.0).to_event(1)
        )
        if cfg.use_paper_sigma_c:
 # Half-Cauchy(0, 1) per subtype.
 # Median ≈ 1.0, heavy tails. Expanded to a vector of length C.
            sigma_c_stage = numpyro.sample(
                "sigma_c_stage",
                dist.HalfCauchy(scale=1.0).expand([C]).to_event(1),
            )
        else:
 # Earlier default: log-Normal via TransformedDistribution.
 # Preserved; paired with [0.5, 1.5] clamp in the
 # likelihood path below.
            sigma_c_stage = numpyro.sample(
                "sigma_c_stage",
                dist.TransformedDistribution(
                    dist.Normal(jnp.zeros(C), 1.0), dist.transforms.ExpTransform()
                ).to_event(1),
            )

 # ---------- Trajectory parameters (per subtype) ----------
        pi_region = numpyro.sample(
            "pi_region", dist.Beta(2.0 * jnp.ones((C, 4)), 1.0).to_event(2)
        )
 # delta_{c,r}: relaxed Bernoulli at learned pi_region; temperature
 # is now a MODEL INPUT so the SVI loop can anneal it per-step.
        mod_idx = jnp.asarray(cfg.modality_of_r) # (R,) int indices 0..3
        pi_per_r = pi_region[:, mod_idx] # (C, R)
        delta_region = numpyro.sample(
            "delta_region",
            dist.RelaxedBernoulli(
                temperature=gumbel_tau,
                probs=pi_per_r,
            ).to_event(2),
        )

 # ---------- Onset times ----------
 # Flag-gated: when `use_paper_cascade_prior` is True, sample modality-
 # level onset centers t̃_{c,m} with a SOFT cascade A < T < N ordering
 # (blood unconstrained), plus per-biomarker deviation η_{c,r}. The
 # observed `onset_times` site becomes a deterministic pass-through
 # sigmoid(t̃_{c, m(r)} + η_{c, r}) so downstream consumers still see
 # it by the same name.
 # Flag off: legacy iid sigmoid-Normal prior.
 #
 # When `use_paper_spatial_priors` is True, the per-biomarker η draw is a
 # per-modality MVN with Σ^(m) covariance (per-modality kernel on the
 # 1-D biomarker-index grid). Applies in both cascade-on and cascade-off
 # paths below — in the cascade path the MVN replaces the iid η_{c,r};
 # in the non-cascade path the MVN replaces the sigmoid(Normal(0, 1))
 # direct draw and we use the legacy modality centers as the MVN mean.
        if getattr(cfg, "use_paper_cascade_prior", False):
 # SOFT ordering: each modality center is sampled INDEPENDENTLY from a
 # Gaussian centered at its prior mean, with μ_A < μ_T < μ_N encouraging
 # but NOT enforcing A→T→N ordering. The previous cumulative-softplus
 # reparameterization enforced STRICT A < T < N (hard ordering), which
 # contradicts the paper's "encouraging but not enforcing" language.
 #
 # Paper form (soft):
 # t̃_{c,A} ~ N(μ_A, σ_m²) independently
 # t̃_{c,T} ~ N(μ_T, σ_m²) independently
 # t̃_{c,N} ~ N(μ_N, σ_m²) independently
 # The ordering A→T→N is encouraged by μ_A=0.3 < μ_T=0.5 < μ_N=0.7
 # but not mechanically enforced. This
 # matches the paper exactly.
 #
 # Legacy strict form (use_paper_cascade_prior=False): no cascade at all
 # (legacy iid sigmoid-Normal prior). The cumulative-softplus form
 # is removed — it was never the legacy; it was a misimplementation of
 # the paper's soft form under the True branch.
            raw_ATN = numpyro.sample(
                "t_ATN_raw",
                dist.Normal(
                    jnp.broadcast_to(
                        jnp.array([
                            cfg.cascade_mu_A, # μ_A = 0.3
                            cfg.cascade_mu_T, # μ_T = 0.5
                            cfg.cascade_mu_N, # μ_N = 0.7
                        ]),
                        (C, 3),
                    ),
                    cfg.cascade_sigma_m, # σ_m (shared per the paper; default 0.15)
                ).to_event(2),
            )
 # Each column is an independent Gaussian draw — no cumulative-softplus
 # coupling. Ordering is encouraged (not enforced) by the prior means.
            t_tilde_ATN = raw_ATN # (C, 3), columns = [t̃_A, t̃_T, t̃_N]

 # Blood center: per-subtype prior, UNconstrained.
            t_tilde_B = numpyro.sample(
                "t_tilde_B",
                dist.Normal(
                    jnp.full((C,), cfg.cascade_mu_B),
                    cfg.cascade_sigma_m,
                ).to_event(1),
            )
            t_tilde = jnp.concatenate(
                [t_tilde_ATN, t_tilde_B[:, None]], axis=-1
            ) # (C, 4)

 # Per-biomarker deviation.
            if getattr(cfg, "use_paper_spatial_priors", False):
 # Per-modality MVN for eta_{c, R_m}, with zero mean
 # (modality centers are carried by t_tilde above). Each
 # modality block is sampled as its own sample site so the
 # overall `onset_eta` vector has the correct modality-block
 # covariance structure.
                eta_blocks = []
                for m_i in range(4):
                    lo, hi = _mod_bounds[m_i]
                    if hi <= lo:
                        continue
                    Lm = L_eta_mods[m_i]
                    site_name = f"onset_eta_mod{m_i}"
                    eps_m = numpyro.sample(
                        site_name,
                        dist.Normal(
                            jnp.zeros((C, hi - lo)), 1.0
                        ).to_event(2),
                    )
 # eta_m = eps_m @ L_m^T is (C, R_m) — drawing from
 # N(0, σ_η^2 Σ^(m)) via the Cholesky L_m of that covariance.
                    eta_m = jnp.einsum("cr,sr->cs", eps_m, Lm)
 # Scale factor: rescale so the marginal std of each entry
 # is cfg.cascade_eta_sigma (the per-biomarker
 # variability); the spatial Cholesky factor's marginal is
 # cfg.sigma_eta_spatial by construction. Dividing restores
 # that variance and leaves the correlation structure
 # intact, so the spatial prior is a correlation-only
 # modification of the iid eta draw.
                    eta_m = eta_m * (cfg.cascade_eta_sigma / jnp.maximum(
                        float(cfg.sigma_eta_spatial), 1e-6))
                    eta_blocks.append((lo, hi, eta_m))
 # Assemble (C, R) from the modality blocks.
                eta_cr = jnp.zeros((C, R))
                for lo, hi, eta_m in eta_blocks:
                    eta_cr = eta_cr.at[:, lo:hi].set(eta_m)
 # Expose as deterministic for downstream consumers.
                numpyro.deterministic("onset_eta", eta_cr)
            else:
                eta_cr = numpyro.sample(
                    "onset_eta",
                    dist.Normal(jnp.zeros((C, R)), cfg.cascade_eta_sigma).to_event(2),
                )

            mod_of_r = jnp.asarray(cfg.modality_of_r) # (R,)
            onset_raw = t_tilde[:, mod_of_r] + eta_cr # (C, R)
            onset_times = numpyro.deterministic(
                "onset_times", jax.nn.sigmoid(onset_raw)
            )
        else:
            if getattr(cfg, "use_paper_spatial_priors", False):
 # Without cascade: sample η_{c, R_m} ~ N(mu_m · 1,
 # sigma_eta^2 Σ^(m)) per modality, then onset_times =
 # sigmoid(η). Use the modality centers as mu_m
 # (same values as the cascade defaults — these are the paper's
 # soft A<T<N onset centers, plus a blood center at 0.35).
                mu_m_vec = jnp.array([
                    cfg.cascade_mu_A, cfg.cascade_mu_T,
                    cfg.cascade_mu_N, cfg.cascade_mu_B,
                ], dtype=jnp.float32)
 # Invert sigmoid to get the prior mean in pre-sigmoid η space:
 # onset_center is a probability in (0, 1); we want η s.t.
 # sigmoid(η) = onset_center. Use logit.
                mu_m_eta = jnp.log(mu_m_vec / (1.0 - mu_m_vec)) # (4,)
                eta_blocks = []
                for m_i in range(4):
                    lo, hi = _mod_bounds[m_i]
                    if hi <= lo:
                        continue
                    Lm = L_eta_mods[m_i]
 # Sample standard Normal then affine-transform by L:
 # η_m = mu_m_eta + L @ eps
 # (equivalent to drawing from N(mu_m_eta·1, σ_η^2 Σ^(m)).)
                    eps_m = numpyro.sample(
                        f"onset_eta_mod{m_i}",
                        dist.Normal(jnp.zeros((C, hi - lo)), 1.0).to_event(2),
                    )
                    eta_m_center = mu_m_eta[m_i] * jnp.ones((C, hi - lo))
                    corr_noise = jnp.einsum("cr,sr->cs", eps_m, Lm) # (C, R_m)
                    eta_blocks.append((lo, hi, eta_m_center + corr_noise))
                eta_cr = jnp.zeros((C, R))
                for lo, hi, eta_m in eta_blocks:
                    eta_cr = eta_cr.at[:, lo:hi].set(eta_m)
                onset_times = numpyro.deterministic(
                    "onset_times", jax.nn.sigmoid(eta_cr)
                )
            else:
                onset_times = numpyro.sample(
                    "onset_times",
                    dist.TransformedDistribution(
                        dist.Normal(jnp.zeros((C, R)), 1.0),
                        dist.transforms.SigmoidTransform(),
                    ).to_event(2),
                )
        sharpness = numpyro.sample(
            "sharpness",
            dist.TransformedDistribution(
                dist.Normal(-1.5 * jnp.ones((C, R)), 0.3),
                dist.transforms.ExpTransform(),
            ).to_event(2),
        )
 # beta_vel prior widened to 2.0 so the model can learn large velocities
 # matching the simulator's `V = V * (target_scale / vel_norm)` scaling.
 # At prior std 0.5, softplus(beta_vel @ B) saturates around small values;
 # at 2.0 the posterior can reach velocity norms comparable to truth.
        if getattr(cfg, "use_paper_spatial_priors", False):
 # beta_{c, R_m, q} ~ N(0, sigma_beta^2 Σ^(m))
 # per modality per (c, q). Sample standard-Normal residuals, rotate
 # by the modality Cholesky, and assemble (C, R, Q).
            beta_blocks = []
            for m_i in range(4):
                lo, hi = _mod_bounds[m_i]
                if hi <= lo:
                    continue
                Lm_b = L_beta_mods[m_i]
                eps_b = numpyro.sample(
                    f"beta_vel_mod{m_i}",
                    dist.Normal(jnp.zeros((C, hi - lo, Q)), 1.0).to_event(3),
                )
 # eps_b: (C, R_m, Q); beta_m[c, r, q] = sum_s L[r, s] eps[c, s, q]
                beta_m = jnp.einsum("rs,csq->crq", Lm_b, eps_b)
                beta_blocks.append((lo, hi, beta_m))
            beta_vel = jnp.zeros((C, R, Q))
            for lo, hi, beta_m in beta_blocks:
                beta_vel = beta_vel.at[:, lo:hi, :].set(beta_m)
            numpyro.deterministic("beta_vel", beta_vel)
        else:
            beta_vel = numpyro.sample(
                "beta_vel",
                dist.Normal(jnp.zeros((C, R, Q)), 2.0).to_event(3),
            )

 # Terminal state Theta_{c,J} ~ N(mu_c, small). Tight std (0.1) pins
 # Theta_J at mu_c, so mu_c is strongly identified via anchor observations.
        Theta_J = numpyro.sample(
            "Theta_J",
            dist.Normal(mu_c, 0.1).to_event(2),
        )

 # ---------- Factor covariance parameters ----------
        if phase == "phase2":
            if cfg.use_ard_on_A:
 # Phase 1a ARD branch: per-(C, K) precision alpha[c, k] with
 # Gamma(a, b) prior. Each column of A_c gets its own std, so
 # the model can drive alpha[c, k>=K_truth] large and collapse
 # the redundant columns. No Bhattacharya-Dunson ordering.
                alpha_ard = numpyro.sample(
                    "alpha_ard",
                    dist.Gamma(
                        jnp.full((C, K), cfg.ard_a_shape),
                        cfg.ard_b_rate,
                    ).to_event(2),
                )
                a_prior_std = 1.0 / jnp.sqrt(alpha_ard) # (C, K)
 # NOTE: a_flat_prior_std is computable post-hoc from alpha_ard
 # (= 1/sqrt(alpha)); we don't emit it as a deterministic site
 # because AutoGuideList + handlers.block rejects deterministic
 # sites inside sub-guides. Downstream diagnostics should load
 # alpha_ard from the NPZ sidecar.

                a_flat = numpyro.sample(
                    "a_flat",
                    dist.Normal(
                        jnp.zeros((C, R, K)),
                        a_prior_std[:, None, :], # (C, 1, K) broadcast
                    ).to_event(3),
                )
 # Direct HalfNormal on diagonal so shrinkage actually drives diag → 0.
                if getattr(cfg, "use_paper_halfnormal_diag", False):
                    a_diag_pos = numpyro.sample(
                        "a_diag_pos",
                        dist.HalfNormal(a_prior_std).to_event(2),
                    )
                    a_diag_raw = None
                else:
                    a_diag_raw = numpyro.sample(
                        "a_diag_raw",
                        dist.Normal(jnp.zeros((C, K)), a_prior_std).to_event(2),
                    )
                    a_diag_pos = None
            else:
 # Multiplicative gamma shrinkage (Bhattacharya & Dunson 2011).
 # Split shapes:
 # kappa_{c,1} ~ Gamma(delta_shrink_shape_k1, 1) # first column
 # kappa_{c,k>=2} ~ Gamma(delta_shrink_shape, 1) # later columns
 # Both default to 1.5; set delta_shrink_shape_k1=2.0 for a_1=2.0.
                shape_vec = jnp.concatenate([
                    jnp.full((1,), cfg.delta_shrink_shape_k1),
                    jnp.full((max(K - 1, 0),), cfg.delta_shrink_shape),
                ]) # (K,)
                shape_ck = jnp.broadcast_to(shape_vec[None, :], (C, K)) # (C, K)
                delta_shrink = numpyro.sample(
                    "delta_shrink",
                    dist.Gamma(shape_ck, 1.0).to_event(2),
                )
                psi_shrink = jnp.cumprod(delta_shrink, axis=1) # (C, K)
 # E1.B fix 1: multiply the psi_shrink "precision" by 1/psi_shrink_scale
 # so that psi_shrink_scale < 1 *tightens* the prior on a_flat
 # (smaller scale -> smaller std). Default cfg.psi_shrink_scale=1.0
 # leaves the std unchanged, preserving existing behaviour.
                a_prior_std = jnp.sqrt(cfg.psi_shrink_scale) / jnp.sqrt(psi_shrink) # (C, K)

 # A_c: lower-triangular with positive diagonal, per-subtype
                a_flat = numpyro.sample(
                    "a_flat",
                    dist.Normal(
                        jnp.zeros((C, R, K)),
                        a_prior_std[:, None, :], # (C, 1, K) broadcast
                    ).to_event(3),
                )
 # Direct HalfNormal on diagonal.
                if getattr(cfg, "use_paper_halfnormal_diag", False):
                    a_diag_pos = numpyro.sample(
                        "a_diag_pos",
                        dist.HalfNormal(a_prior_std).to_event(2),
                    )
                    a_diag_raw = None
                else:
                    a_diag_raw = numpyro.sample(
                        "a_diag_raw",
                        dist.Normal(jnp.zeros((C, K)), a_prior_std).to_event(2),
                    )
                    a_diag_pos = None

 # Shrinkage prior on tau_k prevents w_ckp_raw from drifting to
 # arbitrarily large values during long SVI runs. cfg.tau_k_scale
 # controls the HalfNormal scale (default 0.1). Benchmarked via
 # coordinate sweep: tau ∈ {0.1,0.2,0.3,0.5,0.8} all give 5/7 gates;
 # tau=0.1 gives best factor metrics (activation=0.227, cov_fro=0.253).
            tau_k = numpyro.sample(
                "tau_k",
                dist.HalfNormal(cfg.tau_k_scale * jnp.ones(K)).to_event(1),
            )
 # Activation coefficients w_{c,k,p}: per-factor normalized by construction.
 # We sample in an unconstrained space and apply the projection as a
 # deterministic transformation inside the model. The KL is still well-
 # defined because the projection is linear and the prior has zero mean.
            if cfg.use_paper_w_ckp_smoothness:
 # Paper Eq. 15: w_{c,k,:} ~ N(0, tau_k^2 (K_B + eps_B I)^{-1}).
 # Reparameterize: sample w_std ~ N(0, I_P) then scale by tau_k * L,
 # where L = chol((K_B + eps_B I)^{-1}) was precomputed above.
 # Site name renamed to `w_ckp_raw_std` so downstream consumers
 # that look for `w_ckp_raw` (legacy path) don't accidentally
 # pick up the pre-Cholesky draw.
                w_std = numpyro.sample(
                    "w_ckp_raw_std",
                    dist.Normal(0.0, 1.0).expand([C, K, P]).to_event(3),
                )
 # Apply L @ w_std per (c, k), then scale by tau_k[k]:
 # w_ckp_raw[c, k, p] = tau_k[k] * sum_q L[p, q] * w_std[c, k, q]
 # (NOT emitted as a deterministic — AutoGuideList rejects sites
 # exposed as deterministic in blocked sub-guides. Downstream
 # consumers read `w_ckp` which is the per-factor-normalized
 # deterministic emitted below.)
                w_ckp_raw = tau_k[None, :, None] * jnp.einsum(
                    "pq,ckq->ckp", W_cov_scale, w_std
                )
            else:
 # Earlier default: iid Normal per knot, no smoothness coupling.
                w_ckp_raw = numpyro.sample(
                    "w_ckp_raw",
                    dist.Normal(
                        jnp.zeros((C, K, P)),
                        tau_k[None, :, None] * jnp.ones((C, K, P)),
                    ).to_event(3),
                )
 # Per-factor normalization (1^T w_{c,k} = 0)
            w_ckp = w_ckp_raw - jnp.mean(w_ckp_raw, axis=-1, keepdims=True)
            numpyro.deterministic("w_ckp", w_ckp)

 # Build A_c per subtype. When
 # use_paper_halfnormal_diag=True, the diagonal was sampled directly
 # from HalfNormal (already positive) — use the _direct helper that
 # skips the softplus transform.
            if getattr(cfg, "use_paper_halfnormal_diag", False):
                def build_A_one(a_flat_c, a_diag_c):
                    return make_lower_triangular_A_direct(
                        a_flat_c.reshape(-1), a_diag_c, R, K
                    )
                A_c = jax.vmap(build_A_one)(a_flat, a_diag_pos) # (C, R, K)
            else:
                def build_A_one(a_flat_c, a_diag_c):
                    return make_lower_triangular_A(
                        a_flat_c.reshape(-1), a_diag_c, R, K
                    )
                A_c = jax.vmap(build_A_one)(a_flat, a_diag_raw) # (C, R, K)
            numpyro.deterministic("A_c", A_c)

 # E1.B fix 2: soft Frobenius-norm penalty on A_c. Adds
 # log_joint += -penalty * sum_c ||A_c||_F^2
 # i.e. a Gaussian ridge that shrinks every entry of A_c equally.
 # Interacts additively with the Bhattacharya-Dunson prior on
 # (a_flat, a_diag_raw); default A_frobenius_penalty=0.0 makes
 # this a no-op. When > 0 the effective Gaussian precision on
 # A_c entries is (psi_shrink / psi_shrink_scale) + 2*penalty.
            if cfg.A_frobenius_penalty > 0.0:
                fro_sq = jnp.sum(A_c * A_c) # sum over (c, r, k) == sum_c ||A_c||_F^2
                numpyro.factor(
                    "A_frobenius_penalty",
                    -cfg.A_frobenius_penalty * fro_sq,
                )
        else:
 # Phase 1: diagonal-only. A_c = 0 in effect.
            A_c = jnp.zeros((C, R, K))
            w_ckp = jnp.zeros((C, K, P))

 # Idiosyncratic noise D_c
        log_D = numpyro.sample(
            "log_D",
            dist.Normal(
                cfg.log_d_prior_mean * jnp.ones((C, R)),
                cfg.log_d_prior_std,
            ).to_event(2),
        )
        D_c = jnp.exp(log_D) + cfg.d_min # positivity floor d_min

 # ---------- Trajectory construction ----------
        V = _velocity_from_shell_boundaries(
            B_shell, beta_vel, onset_times, sharpness, delta_region, tau_grid, mod_signs
        ) # (C, J+1, R)
        if cfg.forward_time:
            theta_0_c = jnp.broadcast_to(Theta_0[None, :], (C, R)) # (C, R)
            theta_shells = _forward_time_integration(theta_0_c, V, J) # (C, J+1, R)
            numpyro.factor(
                "soft_terminal",
                dist.Normal(Theta_J, cfg.Omega_0_soft_std).log_prob(theta_shells[:, -1, :]).sum(),
            )
        else:
            theta_shells = _reverse_time_integration(Theta_J, V, J) # (C, J+1, R)
            numpyro.factor(
                "soft_origin",
                dist.Normal(Theta_0[None, :], cfg.Omega_0_soft_std).log_prob(theta_shells[:, 0, :]).sum(),
            )

 # Cognition params
        if cfg.use_cognition:
            gamma_0 = numpyro.sample("gamma_0", dist.Normal(0.0, 1.0))
            gamma_X = numpyro.sample(
                "gamma_X", dist.Normal(jnp.zeros(cfg.p_x), 1.0).to_event(1)
            )
            log_sigma_C = numpyro.sample("log_sigma_C", dist.Normal(0.0, 1.0))
            sigma_C_cog = jnp.exp(log_sigma_C) + 1e-3
 # Chi_c: positive coefficients so +chi_c(s) is monotone increasing
 # (later stage -> higher CDR-SB score = worse cognition).
 #
 # Two bases:
 # "polynomial" (default): softplus-positive polynomial in
 # [s, s^2, s^3]. Monotone for nonneg coefficients but only
 # cubic-expressive.
 # "ispline": monotone I-spline with nonneg coefficients.
 # chi_c(0) = 0 by construction; strict monotone non-decreasing.
 # Sample site name stays "chi_c_raw" in both paths so the amortized
 # guide / posterior-extraction sites see the same name; the SHAPE
 # differs (C, 3) polynomial vs (C, n_basis) I-spline.
 # Traced end-to-end for cfg.chi_c_basis=="ispline":
 # 1. build_ispline_eval_matrix → chi_basis_cache (J+1, n_basis) precomputed above.
 # 2. chi_c_raw sampled (C, n_basis) from N(0,1); chi_c = softplus(chi_c_raw) >= 0.
 # 3. chi_basis_at(s) interpolates chi_basis_cache at subject's stage → (n_basis,).
 # 4. chi_s_cn = einsum(chi_c, s_basis_cn) produces monotone non-decreasing chi_c(s).
 # 5. heldout.py uses eval_ispline_basis for the same evaluation at test time.
 # All links verified intact; flag correctly produces I-spline behavior at fit time.
            if cfg.chi_c_basis == "ispline":
                n_chi_basis = int(cfg.chi_c_ispline_n_basis)
                chi_c_raw = numpyro.sample(
                    "chi_c_raw",
                    dist.Normal(jnp.zeros((C, n_chi_basis)), 1.0).to_event(2),
                )
            else:
                chi_c_raw = numpyro.sample(
                    "chi_c_raw",
                    dist.Normal(jnp.zeros((C, 3)), 1.0).to_event(2),
                )
            chi_c = jax.nn.softplus(chi_c_raw) # (C, n_basis), nonneg → monotone non-decreasing

 # ---------- Per-subject log-likelihood via MANUAL mixture marginalization ----------
 # We build a (C, n) log-likelihood matrix explicitly and use logsumexp with
 # the subtype prior to get the marginal log-likelihood. This avoids the
 # numpyro enumeration machinery (which is finicky with vmap-over-plate).
        logit_anchor_mean = jnp.log(0.95 / 0.05)
        logit_nc_mean = jnp.log(0.01 / 0.99)
        eps_pinning = cfg.eps_pinning # epsilon_S (default 0.1)

        is_anchor = (anchor_type >= 0) & (anchor_type < C)
        is_nc = anchor_type == C
        anchor_c = jnp.where(is_anchor, anchor_type, 0) # valid index, masked below

 # Per-subject logit-stage prior. Two modes:
 # structured_stage=False: one logit_S per subject, subtype-agnostic.
 # Backward-compatible legacy mode.
 # structured_stage=True : one logit_S per (subject, subtype), shape (n, C).
 # Structured VI with subtype-conditional stage posterior.
        if cfg.structured_stage:
 # Build (n, C) prior mean: for general subjects, the subtype-specific
 # prior m_{i,c} = f_c(Age_i) + h_c^T W_i per the paper. For anchors,
 # pin ONLY the true-c stage to the boundary; wrong-c stages get the
 # subtype-specific general prior (so the KL penalty is minimal —
 # wrong-c doesn't appear in the log-lik anyway because the subtype
 # softmax log-prior is -inf there, and a well-specified general
 # prior lets q(S | wrong-c) collapse with ~zero KL).
            general_logit_mean_c = (
                ages_norm[:, None] * f_c_age[None, :] # (n, C)
                + W @ h_c.T # (n, C)
            )
 # Pinning masks:
 # - Anchor i with true c=c*: pin ONLY (i, c*) at logit(0.95)
 # Wrong-c entries follow the general prior (~0 KL since likelihood
 # weight on wrong-c is -inf via log_prior_adj)
 # - NC i: pin ALL c entries at logit(0.01) because NC is at origin
 # regardless of the (unobserved) subtype label
            c_range = jnp.arange(C) # (C,)
            anchor_c_expand = anchor_c[:, None] # (n, 1)
            is_true_c_anchor = (c_range[None, :] == anchor_c_expand) & is_anchor[:, None] # (n, C)
            is_nc_col = jnp.broadcast_to(is_nc[:, None], (n, C)) # (n, C) all columns for NC

            mean_logit_S_c = jnp.where(
                is_true_c_anchor, logit_anchor_mean,
                jnp.where(is_nc_col, logit_nc_mean, general_logit_mean_c),
            )
 # Per-subtype std per the paper: sigma_c. sigma_c_stage is
 # already positive (log-Normal default; Half-Cauchy when
 # cfg.use_paper_sigma_c=True).
            if cfg.use_paper_sigma_c:
 # no clamp; σ_c flows directly.
                sigma_c_positive = sigma_c_stage[None, :] # (1, C)
            else:
 # Earlier training-stability clamp to [0.5, 1.5]. Paper
 # never specifies this; retained for preservation
 # when cfg.use_paper_sigma_c=False.
                sigma_c_positive = jnp.clip(sigma_c_stage, 0.5, 1.5)[None, :] # (1, C)
            std_logit_S_c = jnp.where(
                is_true_c_anchor | is_nc_col,
                eps_pinning,
                jnp.broadcast_to(sigma_c_positive, (n, C)),
            )
            with numpyro.plate("subjects", n):
                logit_S_c = numpyro.sample(
                    "logit_S",
                    dist.Normal(mean_logit_S_c, std_logit_S_c).to_event(1),
                )
            S_i_c = jax.nn.sigmoid(logit_S_c) # expected (n, C)
 # Be robust to shape — if a guide substitutes a 1D array, broadcast it
            if S_i_c.ndim == 1:
                S_i_c = jnp.broadcast_to(S_i_c[:, None], (n, C))
            S_i = S_i_c[:, 0] # scalar stage for convenience
        else:
            general_logit_mean = f_c_age[0] * ages_norm # (n,) simplified shared age effect
            mean_logit_S = jnp.where(
                is_anchor, logit_anchor_mean,
                jnp.where(is_nc, logit_nc_mean, general_logit_mean),
            )
            std_logit_S = jnp.where(
                is_anchor | is_nc, eps_pinning, jax.nn.softplus(sigma_c_stage.mean())
            )
            with numpyro.plate("subjects", n):
                logit_S = numpyro.sample(
                    "logit_S",
                    dist.Normal(mean_logit_S, std_logit_S),
                )
            S_i = jax.nn.sigmoid(logit_S) # (n,)
            S_i_c = None # not used

 # Subtype prior (n, C)
        subtype_logits = alpha_c[None, :] + X @ beta_c_sub.T # (n, C)
        log_prior_c = jax.nn.log_softmax(subtype_logits, axis=-1) # (n, C)

 # Build psi(s) on the grid — we'll index by the correct per-(i, c) stage
        def psi_at_stage(s_val):
            s_cap = jnp.clip(s_val, 0.0, 1.0 - 1e-6)
            j_float = s_cap * J
            j_lo = jnp.clip(jnp.floor(j_float).astype(jnp.int32), 0, J - 1)
            frac = j_float - j_lo.astype(jnp.float32)
            psi_lo_ = psi_basis_cache[j_lo] # (P,)
            psi_hi_ = psi_basis_cache[j_lo + 1] # (P,)
            return psi_lo_ + frac * (psi_hi_ - psi_lo_) # (P,)

 # theta_c(s) at a single (c_idx, s) point
        def theta_at_cs(c_idx, s_val):
            theta_shells_c = theta_shells[c_idx] # (J+1, R)
            v_shells_c = V[c_idx] # (J+1, R)
            s_clipped = jnp.clip(s_val, 0.0, 1.0 - 1e-6)
            j_idx = jnp.clip(jnp.floor(s_clipped * J).astype(jnp.int32), 0, J - 1)
            u = (s_clipped - j_idx.astype(jnp.float32) / J) * J
            th_l = theta_shells_c[j_idx]
            th_r = theta_shells_c[j_idx + 1]
            v_l = v_shells_c[j_idx]
            v_r = v_shells_c[j_idx + 1]
            return cubic_hermite_interpolate(th_l, th_r, v_l, v_r, u, delta_j=1.0 / J)

        if cfg.structured_stage:
 # Use S_i_c (shape (n, C)): for each subject and subtype, evaluate theta
 # at that subtype's specific stage, and likewise nu(s_{i,c}).
            def per_c_block(c_idx):
                s_vec = S_i_c[:, c_idx] # (n,)
 # psi at each subject's per-subtype stage
                psi_ic = jax.vmap(psi_at_stage)(s_vec) # (n, P)
 # theta at (c, s_{i,c}) for each subject
                theta_ic = jax.vmap(lambda s: theta_at_cs(c_idx, s))(s_vec) # (n, R)
                if phase == "phase2":
                    nu_ic = jnp.exp(jnp.clip(
                        jnp.einsum("kp,np->nk", w_ckp[c_idx], psi_ic), -8.0, 8.0
                    )) # (n, K)
                else:
                    nu_ic = jnp.ones((n, K))
                return theta_ic, nu_ic
            theta_all, nu_all = jax.vmap(per_c_block)(jnp.arange(C)) # (C, n, R), (C, n, K)
        else:
 # Legacy path: single S_i per subject, shared across c
            psi_S = jax.vmap(psi_at_stage)(S_i) # (n, P)
            if phase == "phase2":
                g_cnk = jnp.einsum("ckp,np->cnk", w_ckp, psi_S)
                nu_all = jnp.exp(jnp.clip(g_cnk, -8.0, 8.0)) # (C, n, K)
            else:
                nu_all = jnp.ones((C, n, K))
            theta_all = jax.vmap(
                lambda c_idx: jax.vmap(lambda s: theta_at_cs(c_idx, s))(S_i)
            )(jnp.arange(C)) # (C, n, R)

 # Per-(c, i) log-likelihood matrix
        def loglik_one_subject(y_row, mask_row, theta_row, A_mat, nu_row, D_row):
            return _imaging_log_lik_for_subject(y_row, mask_row, theta_row, A_mat, nu_row, D_row)

        def loglik_for_c(c_idx):
            theta_c = theta_all[c_idx] # (n, R)
            nu_c_i = nu_all[c_idx] # (n, K)
            A_mat = A_c[c_idx] # (R, K)
            D_vec = D_c[c_idx] # (R,)
            return jax.vmap(
                lambda y_row, m_row, th_row, nu_row: loglik_one_subject(
                    y_row, m_row, th_row, A_mat, nu_row, D_vec
                )
            )(Y, mask, theta_c, nu_c_i) # (n,)

        log_lik_matrix = jax.vmap(loglik_for_c)(jnp.arange(C)) # (C, n)
        log_lik_matrix = log_lik_matrix.T # (n, C)

 # Anchor-constrained marginalization: if i is an anchor, force c = anchor_c[i]
 # by adding a large negative offset to all other subtypes' log-prior.
        large_neg = jnp.float32(-1e6)
        anchor_onehot = jax.nn.one_hot(anchor_c, C) # (n, C)
        anchor_mask = is_anchor[:, None].astype(jnp.float32) # (n, 1)
        log_prior_adj = jnp.where(
            anchor_mask > 0,
            jnp.log(anchor_onehot + 1e-30), # -inf for non-anchor subtypes
            log_prior_c,
        )

 # Log-marginal over c_i: (n,)
        log_joint = log_lik_matrix + log_prior_adj # (n, C)
        if ipw_weights is not None:
            w_tilde = ipw_weights
        else:
 # Design note: NC subjects (anchor_type == C) receive w_tilde=1.0
 # (general-pool weight) rather than w_anchor. Rationale: NC is already
 # tightly constrained by the stage-prior pinning at logit(0.01) with
 # eps_pinning std; upweighting the imaging term with w_anchor would
 # double-count this constraint and effectively force Θ₀ identification
 # through NC biomarker means rather than the generative trajectory
 # structure. Design decision confirmed by user.
 # Here `is_anchor` = anchor_type in [0, C-1] (endpoint subtypes only),
 # so NC rows (anchor_type == C) correctly land in the 1.0 branch.
            w_tilde = jnp.where(is_anchor, cfg.w_anchor, 1.0)

 # Encoder subtype head training (distillation).
 # Guide samples q_c ~ Delta(softmax(logit_c)). Model places a Dirichlet
 # prior peaked at the model's own posterior p(c|Y,X,Psi) on general
 # subjects, and at the true anchor label for anchors. This provides
 # gradient to the encoder for every subject (not just anchors).
        if cfg.train_q_c:
            alpha_base = jnp.ones(C)
            anchor_onehot_qc = jax.nn.one_hot(jnp.clip(anchor_type, 0, C - 1), C)
 # Target for general subjects: softmax of log_joint, stop-gradient
 # (avoid gradient loop through the encoder back into log_joint).
            model_post_target = jax.lax.stop_gradient(jax.nn.softmax(log_joint, axis=-1))
            alpha_general = alpha_base + cfg.q_c_anchor_concentration * model_post_target
            alpha_anchor = alpha_base + cfg.q_c_anchor_concentration * anchor_onehot_qc
            alpha_per_subject = jnp.where(
                is_anchor[:, None],
                alpha_anchor,
                alpha_general,
            )
            with numpyro.plate("subjects", n):
                q_c_site = numpyro.sample("q_c", dist.Dirichlet(alpha_per_subject))

 # Optional supervised CE on anchors for stronger encoder training.
 # Mean-reduced over anchors to avoid gradient magnitude overwhelming stage head.
            if cfg.anchor_ce_supervision or cfg.enhanced_encoder:
                log_q_c = jnp.log(q_c_site + 1e-12) # (n, C)
                true_c_onehot = jax.nn.one_hot(jnp.clip(anchor_type, 0, C - 1), C) # (n, C)
                anchor_ce = -jnp.sum(true_c_onehot * log_q_c, axis=-1) # (n,)
                anchor_ce_masked = jnp.where(is_anchor, anchor_ce, 0.0)
                n_anchors = jnp.maximum(jnp.sum(is_anchor.astype(jnp.float32)), 1.0)
                mean_ce = jnp.sum(anchor_ce_masked) / n_anchors # scalar
                numpyro.factor(
                    "anchor_ce_supervision",
                    -cfg.anchor_ce_weight * mean_ce,
                )

 # Cognition log-likelihood per (c, i)
        cog_log_cn = None
        if cfg.use_cognition and C_cog is not None and C_cog_mask is not None:
 # chi_c has shape (C, n_chi_basis); compute (C, n) cognition effect.
 # +chi gives later stage -> higher C_i (CDR-SB convention: higher = worse).
            if cfg.chi_c_basis == "ispline":
 # Evaluate I-spline basis at each
 # subject's stage via linear interpolation over the precomputed
 # (J+1, n_basis) design grid (same pattern as psi_at_stage above).
 # Monotone non-decreasing since chi_c = softplus(chi_c_raw) >= 0.
                def chi_basis_at(s_val):
                    s_cap = jnp.clip(s_val, 0.0, 1.0 - 1e-6)
                    j_float = s_cap * J
                    j_lo = jnp.clip(jnp.floor(j_float).astype(jnp.int32), 0, J - 1)
                    frac = j_float - j_lo.astype(jnp.float32)
                    b_lo = chi_basis_cache[j_lo] # (n_basis,)
                    b_hi = chi_basis_cache[j_lo + 1] # (n_basis,)
                    return b_lo + frac * (b_hi - b_lo) # (n_basis,)

                if S_i_c is not None:
 # Per-(c, i) I-spline basis at S_i_c[i, c]: (C, n, n_basis)
                    s_basis_cn = jax.vmap(
                        lambda s_col: jax.vmap(chi_basis_at)(s_col)
                    )(S_i_c.T) # (C, n, n_basis)
                    chi_s_cn = jnp.einsum("ck,cnk->cn", chi_c, s_basis_cn)
                else:
                    s_basis = jax.vmap(chi_basis_at)(S_i) # (n, n_basis)
                    chi_s_cn = jnp.einsum("ck,nk->cn", chi_c, s_basis)
            else:
 # Polynomial branch (legacy).
                if S_i_c is not None:
 # structured_stage: per-(c, i) stage basis
                    s_basis_cn = jnp.stack([S_i_c.T, S_i_c.T**2, S_i_c.T**3], axis=-1) # (C, n, 3)
                    chi_s_cn = jnp.einsum("ck,cnk->cn", chi_c, s_basis_cn) # (C, n)
                else:
                    s_basis = jnp.stack([S_i, S_i**2, S_i**3], axis=-1) # (n, 3)
                    chi_s_cn = jnp.einsum("ck,nk->cn", chi_c, s_basis) # (C, n)
            base_mean = gamma_0 + X @ gamma_X # (n,)
            mean_cog_cn = base_mean[None, :] + chi_s_cn # (C, n)
            cog_log_cn = dist.Normal(mean_cog_cn, sigma_C_cog).log_prob(C_cog[None, :]) # (C, n)

        if cfg.joint_cog_marginalization and cog_log_cn is not None:
 # Option A: joint marginalization — logsumexp over imaging + cognition
 # log p(Y,C|Ψ) = logsumexp_c [log p(c|X) + log p(Y|c,S,Ψ) + log p(C|c,S,X)]
 # Only add cognition for subjects with observed C_cog
            log_joint_full = log_joint + (cog_log_cn * C_cog_mask[None, :]).T # (n, C)
            log_marginal = jax.nn.logsumexp(log_joint_full, axis=-1) # (n,)
            numpyro.factor(
                "joint_log_lik",
                jnp.sum(w_tilde * log_marginal),
            )
        else:
 # Option B (default): imaging-only marginalization + conditional cognition
            log_marginal = jax.nn.logsumexp(log_joint, axis=-1) # (n,)
            numpyro.factor(
                "imaging_log_lik",
                jnp.sum(w_tilde * log_marginal),
            )
            if cog_log_cn is not None:
 # Weight cognition by posterior responsibility q(c|Y,X)
                log_resp = log_joint - log_marginal[:, None] # (n, C)
                resp = jnp.exp(log_resp).T # (C, n)
                cog_log_marg = jnp.sum(resp * cog_log_cn, axis=0) # (n,)
                numpyro.factor(
                    "cognition_log_lik",
                    jnp.sum(cog_log_marg * C_cog_mask),
                )

    return model
