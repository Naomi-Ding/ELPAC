"""Held-out marginal log-likelihood computation for model comparison.

The paper's M2 > M1 > M0 claim is on *held-out* marginal log-likelihood.
This module provides:
  - train_test_split: deterministic 80/20 split by subject index
  - compute_heldout_marginal_ll: fit on train, evaluate LL on test subjects
    using the fitted global parameters

The key observation is that global parameters (A_c, D_c, w_ckp, trajectory params)
are shared across train and test. Per-subject latents (c_i, S_i) are re-inferred
for the test subjects using the VI guide, then the log-likelihood is computed
and averaged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .inference import (
    TrainConfig,
    _build_model_inputs,
    extract_posterior_means,
    run_full_pipeline,
)
from .model import ModelConfig, build_numpyro_model
from .simulate import SyntheticTruth



def _reconstruct_theta_shells(posterior: Dict[str, Any], V: np.ndarray, J: int,
                              cfg: ModelConfig) -> np.ndarray:
    """Rebuild (C, J+1, R) theta shells from posterior, respecting cfg.forward_time.

    Reverse (default): anchored at Theta_J, integrate backward to shell 0.
    Forward (M2-fwd): anchored at Theta_0, integrate forward to shell J.
    Must mirror src/model.py:{_reverse_time_integration, _forward_time_integration}.
    """
    C, _, R = V.shape
    J_steps = V.shape[1] - 1
    assert J_steps == J
    dj = 1.0 / J
    theta_shells = np.zeros((C, J + 1, R), dtype=np.float32)
    if getattr(cfg, "forward_time", False):
        Theta_0 = np.asarray(posterior["Theta_0"])
        theta_shells[:, 0, :] = Theta_0[None, :]
        for j in range(1, J + 1):
            theta_shells[:, j, :] = theta_shells[:, j - 1, :] + dj * V[:, j, :]
    else:
        Theta_J = np.asarray(posterior["Theta_J"])
        theta_shells[:, -1, :] = Theta_J
        for j in range(J, 0, -1):
            theta_shells[:, j - 1, :] = theta_shells[:, j, :] - dj * V[:, j, :]
    return theta_shells



def train_test_split(truth: SyntheticTruth, test_frac: float = 0.2, seed: int = 42):
    """Split a SyntheticTruth into train and test MiniTruths.

    Stratified by anchor_type so both splits have anchors, controls, general.
    Returns (truth_train, truth_test, idx_train, idx_test).
    """
    C = truth.shape.C
    rng = np.random.default_rng(seed)
    is_anchor = (truth.anchor_type >= 0) & (truth.anchor_type < C)
    is_nc = truth.anchor_type == C
    is_general = truth.anchor_type < 0

    def _split(mask):
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_frac))
        return idx[n_test:], idx[:n_test]

    anchor_train, anchor_test = _split(is_anchor)
    nc_train, nc_test = _split(is_nc)
    general_train, general_test = _split(is_general)

    idx_train = np.concatenate([anchor_train, nc_train, general_train])
    idx_test = np.concatenate([anchor_test, nc_test, general_test])

    def _subset(t, idx):
        new_shape = replace(t.shape, n=len(idx))
        return replace(
            t, shape=new_shape,
            Y=t.Y[idx], mask=t.mask[idx], X=t.X[idx], W=t.W[idx],
            ages=t.ages[idx], stages=t.stages[idx], subtypes=t.subtypes[idx],
            anchor_type=t.anchor_type[idx], C_cog=t.C_cog[idx],
        )

    return _subset(truth, idx_train), _subset(truth, idx_test), idx_train, idx_test



def compute_marginal_ll_from_posterior(
    cfg: ModelConfig,
    posterior: Dict[str, Any],
    params: Dict[str, Any],
    truth: SyntheticTruth,
    phase: str = "phase2",
) -> float:
    """Compute the mean per-subject marginal log-likelihood on `truth` using the
    posterior means from `posterior` and `params` (typically from a training run
    on a DIFFERENT subset of subjects).

    Rebuilds the per-(c, i) log-lik matrix from the global parameters and
    marginalizes over c via logsumexp with the subject-specific log-prior.
    """
    from .utils.bspline import build_bspline_eval_matrix, eval_bspline_basis
    from .utils.covariance import gaussian_logpdf_sylvester_woodbury

    # Model dimensions from cfg (not truth) — critical for misspec sweeps
    C = cfg.C
    R = truth.shape.R   # data dimensionality — always from truth
    K = cfg.K
    J = cfg.J

    onset_times = np.asarray(posterior["onset_times"])
    sharpness = np.asarray(posterior["sharpness"])
    delta_region = np.asarray(posterior["delta_region"])
    beta_vel = np.asarray(posterior["beta_vel"])
    log_D = np.asarray(posterior["log_D"])
    D_c = np.exp(log_D) + 1e-4
    alpha_c = np.asarray(posterior["alpha_c"])
    beta_c_sub = np.asarray(posterior["beta_c_sub"])
    if phase == "phase2" and "A_c" in posterior:
        A_c = np.asarray(posterior["A_c"])
        w_ckp = np.asarray(posterior["w_ckp"])
    else:
        A_c = np.zeros((C, R, K))
        w_ckp = np.zeros((C, K, cfg.P))

    B_shell = np.asarray(build_bspline_eval_matrix(n_basis=cfg.Q, n_eval=J + 1))
    tau_grid = np.linspace(0.0, 1.0, J + 1).astype(np.float32)

    mod_signs = np.ones(R, dtype=np.float32)
    sb = truth.shape.R_A + truth.shape.R_T
    se = sb + truth.shape.R_N
    mod_signs[sb:se] = -1.0
    if truth.shape.R_blood > 0:
        mod_signs[se] = -1.0

    raw = np.einsum("crq,jq->cjr", beta_vel, B_shell)
    softplus_raw = np.logaddexp(0.0, raw)  # numerically stable softplus (B1)
    vmag = mod_signs[None, None, :] * softplus_raw
    onset_arg = (tau_grid[None, :, None] - onset_times[:, None, :]) / np.clip(sharpness[:, None, :], 1e-3, None)
    onset_g = 1.0 / (1.0 + np.exp(-onset_arg))
    V = delta_region[:, None, :] * onset_g * vmag

    theta_shells = _reconstruct_theta_shells(posterior, V, J, cfg)
    dj = 1.0 / J

    # Re-infer S_i from held-out biomarkers via test-time latent optimization:
    # for each held-out subject, marginalize over a stage grid and pick the
    # MAP S given the trained global parameters and the observed Y_i. This
    # is a proper predictive likelihood (NOT the prior proxy that previously
    # used `f_c_age[0] * age_norm`, which is wrong under structured_stage and
    # ignored the biomarker evidence entirely).
    n = len(truth.Y)
    ages_norm = ((truth.ages - 70.0) / 10.0).astype(np.float32)
    f_c_age = np.asarray(posterior.get("f_c_age", np.zeros(C)))
    h_c = np.asarray(posterior.get("h_c", np.zeros((C, truth.W.shape[1]))))
    sigma_c = np.asarray(posterior.get("sigma_c_stage", np.ones(C)))
    is_anchor = (truth.anchor_type >= 0) & (truth.anchor_type < C)
    is_nc = truth.anchor_type == C

    # Stage grid for argmax over S_i per (i, c). Coarser is fine — predictive
    # LL is averaged over subjects, not point-estimated for any single one.
    stage_grid = np.linspace(0.05, 0.95, 19).astype(np.float32)

    def _log_lik_at_stage(c: int, s: float, residual_obs: np.ndarray,
                          A_obs: np.ndarray, D_obs: np.ndarray, K_: int) -> float:
        """Per-subject Gaussian log-pdf using observed-row Woodbury."""
        if phase == "phase2":
            psi_s = np.asarray(eval_bspline_basis(cfg.P, s))
            nu_ci = np.exp(np.clip(w_ckp[c] @ psi_s, -8.0, 8.0))
            rDinv = residual_obs / D_obs
            u_vec = A_obs.T @ rDinv
            M_c = A_obs.T @ (A_obs / D_obs[:, None])
            inner = np.diag(1.0 / np.clip(nu_ci, 1e-6, None)) + M_c
            try:
                solved = np.linalg.solve(inner, u_vec)
            except np.linalg.LinAlgError:
                return float("nan")
            quad = float(np.sum(rDinv * residual_obs) - u_vec @ solved)
            I_plus_F = np.eye(K_) + np.diag(nu_ci) @ M_c
            sign, logdet_ipf = np.linalg.slogdet(I_plus_F)
            if sign <= 0 or not np.isfinite(logdet_ipf):
                return float("nan")
            log_det_sigma = float(np.sum(np.log(D_obs))) + float(logdet_ipf)
        else:
            rDinv = residual_obs / D_obs
            quad = float(np.sum(rDinv * residual_obs))
            log_det_sigma = float(np.sum(np.log(D_obs)))
        R_obs = float(len(residual_obs))
        return -0.5 * (R_obs * np.log(2 * np.pi) + log_det_sigma + quad)

    # Pre-compute trajectory at each stage in the grid for each subtype
    from .utils.bspline import cubic_hermite_interpolate as _hermite
    theta_grid = np.zeros((C, len(stage_grid), R), dtype=np.float32)
    for c in range(C):
        for gi, s in enumerate(stage_grid):
            j_int = min(int(s * J), J - 1)
            u = (s - j_int / J) * J
            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            h01 = -2 * u**3 + 3 * u**2
            h11 = u**3 - u**2
            theta_grid[c, gi] = (
                h00 * theta_shells[c, j_int] + h10 * dj * V[c, j_int]
                + h01 * theta_shells[c, j_int + 1] + h11 * dj * V[c, j_int + 1]
            )

    # The predictive LL requires marginalization over S:
    # log p(Y_i | c, X_i) = log ∫ p(Y_i | c, s, Ψ) p(s | c, X_i) ds
    # Approximated via Riemann sum on uniform `stage_grid` (Δs=0.05).
    # (Previously took argmax-over-s — profile likelihood — which inflates M2's
    # LL relative to M1 because M2 has more stage-dependent flexibility to
    # exploit the argmax-S choice.)
    S_i_per_c = np.zeros((n, C), dtype=np.float32)      # MAP-s per (i, c) (reporting only)
    W_obs = truth.W
    # Accumulate log_q_per_c_s_i[i, c, g] = log p(Y_i | c, s_g, Ψ) + log p(s_g | c, X_i)
    # for GENERAL (held-out unlabelled) subjects. Anchor/NC rows are evaluated
    # at their pinned stage (0.95 / 0.01) directly and stored in anchor_ll[i, c].
    # NC pin 0.01 matches the fit-time stage prior logit_nc_mean = log(0.01/0.99)
    # used in model.py.
    log_q_per_c_s_i = np.full((n, C, len(stage_grid)), -np.inf, dtype=np.float64)
    anchor_ll = np.full((n, C), -np.inf, dtype=np.float64)  # LL at pinned-s for anchor/NC
    for i in range(n):
        obs_i = truth.mask[i] > 0.5
        if is_anchor[i] or is_nc[i]:
            s_pin = 0.95 if is_anchor[i] else 0.01
            S_i_per_c[i, :] = s_pin
            # Evaluate LL directly at the pinned stage for each c.
            j_int = min(int(s_pin * J), J - 1)
            u = (s_pin - j_int / J) * J
            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            h01 = -2 * u**3 + 3 * u**2
            h11 = u**3 - u**2
            for c in range(C):
                theta_i_c = (
                    h00 * theta_shells[c, j_int] + h10 * dj * V[c, j_int]
                    + h01 * theta_shells[c, j_int + 1] + h11 * dj * V[c, j_int + 1]
                )
                resid_obs = (truth.Y[i] - theta_i_c)[obs_i]
                A_obs = A_c[c][obs_i]
                D_obs = D_c[c][obs_i]
                ll = _log_lik_at_stage(c, s_pin, resid_obs, A_obs, D_obs, K)
                if np.isfinite(ll):
                    anchor_ll[i, c] = ll
            continue
        for c in range(C):
            mean_logit = f_c_age[c] * ages_norm[i] + float(h_c[c] @ W_obs[i])
            sigma_c_safe = max(float(sigma_c[c]), 0.1)
            for gi, s in enumerate(stage_grid):
                A_obs = A_c[c][obs_i]
                D_obs = D_c[c][obs_i]
                resid_obs = (truth.Y[i] - theta_grid[c, gi])[obs_i]
                ll = _log_lik_at_stage(c, float(s), resid_obs, A_obs, D_obs, K)
                if not np.isfinite(ll):
                    continue   # leave -inf for non-PSD / solve-fail cells
                # Logit-normal density on
                # stage s has the full log-pdf
                # log p(s | c, X) = -log σ - 0.5 log 2π - log[s(1-s)]
                # - 0.5 ((logit s - μ) / σ)^2
                # The paper's stage prior: s ~ LogitNormal(μ_c, σ_c); p(s | c, X_i) is
                # the target predictive density so the normalizer must be
                # included. Compute logit_s and the Jacobian from
                # the SAME clipped value so boundary behaviour stays consistent
                # if the grid is ever extended toward {0, 1}.
                s_safe = float(np.clip(s, 1e-12, 1.0 - 1e-12))
                logit_s = np.log(s_safe / (1.0 - s_safe))
                lp_stage = (
                    -np.log(sigma_c_safe)
                    - 0.5 * np.log(2.0 * np.pi)
                    - np.log(s_safe * (1.0 - s_safe))
                    - 0.5 * ((logit_s - mean_logit) / sigma_c_safe) ** 2
                )
                log_q_per_c_s_i[i, c, gi] = ll + lp_stage
            # MAP-s per c preserved for reporting (used only for S_i_per_c field):
            if np.any(np.isfinite(log_q_per_c_s_i[i, c])):
                S_i_per_c[i, c] = float(stage_grid[int(np.argmax(log_q_per_c_s_i[i, c]))])
            else:
                S_i_per_c[i, c] = 0.5
    S_i = S_i_per_c[:, 0]  # placeholder; unused downstream

    # Marginalize over s per (i, c): log_lik_matrix[i, c] = log ∫ exp(log_q) ds.
    # Numerically-stable logsumexp with log(Δs) Riemann-sum weight.
    stage_grid_step = float(stage_grid[1] - stage_grid[0])
    log_ds = float(np.log(stage_grid_step))
    max_over_s = np.max(log_q_per_c_s_i, axis=-1, keepdims=True)          # (n, C, 1)
    safe_max = np.where(np.isfinite(max_over_s), max_over_s, 0.0)
    sumexp = np.sum(np.exp(log_q_per_c_s_i - safe_max), axis=-1)           # (n, C)
    with np.errstate(divide="ignore"):
        log_lik_matrix = np.log(sumexp) + safe_max.squeeze(-1) + log_ds    # (n, C)
    # Overwrite anchor/NC rows with their pinned-stage LL (no marginalization).
    anchor_or_nc = is_anchor | is_nc
    log_lik_matrix[anchor_or_nc] = anchor_ll[anchor_or_nc]

    # Log prior over subtype given X
    subtype_logits = alpha_c[None, :] + truth.X @ beta_c_sub.T
    log_prior_c = subtype_logits - np.log(np.sum(np.exp(subtype_logits), axis=-1, keepdims=True))
    is_anchor_np = (truth.anchor_type >= 0) & (truth.anchor_type < C)
    log_prior_adj = log_prior_c.copy()
    for i in range(n):
        if is_anchor_np[i]:
            log_prior_adj[i] = -1e6
            log_prior_adj[i, int(truth.anchor_type[i])] = 0.0

    log_joint = log_lik_matrix + log_prior_adj
    log_joint_shifted = log_joint - np.max(log_joint, axis=-1, keepdims=True)
    log_marginal = np.log(np.sum(np.exp(log_joint_shifted), axis=-1)) + np.max(log_joint, axis=-1)
    # Persist per-subject log-marginal LL
    # so downstream aggregators can compute paired-subject bootstrap CIs.
    # The scalar return
    # stays for backward compat; the per-subject array is attached as an
    # attribute on the return value.
    heldout_mean = float(np.mean(log_marginal))
    try:
        # Attach per-subject LL array + general-only mask for paired bootstrap.
        # Use a plain float subclass (no __slots__, so __dict__ allows attr set).
        import numpy as _np
        class _HeldoutLLResult(float):
            """Behaves as float in arithmetic but carries per-subject arrays."""
            pass
        result = _HeldoutLLResult(heldout_mean)
        result.per_subject_ll = _np.asarray(log_marginal, dtype=_np.float64)
        result.is_general = _np.asarray((~is_anchor) & (~is_nc), dtype=bool)
        # Test-subject posterior internals — these are the by-products already
        # computed above; exposing them lets the caller save them to the sidecar
        # so downstream analyses (age-strat, cohort-strat subtype metrics) don't
        # have to re-run inference on test subjects.
        # Shapes: log_lik_matrix (n_test, C), r_ic (n_test, C), S_i_per_c (n_test, C).
        r_ic = _np.exp(log_joint - log_marginal[:, None])
        r_ic = r_ic / _np.clip(r_ic.sum(axis=1, keepdims=True), 1e-12, None)
        map_c = _np.argmax(r_ic, axis=1)
        S_map = S_i_per_c[_np.arange(len(map_c)), map_c]
        S_post_mean = _np.sum(r_ic * S_i_per_c, axis=1)
        result.log_lik_matrix = _np.asarray(log_lik_matrix, dtype=_np.float64)
        result.log_joint = _np.asarray(log_joint, dtype=_np.float64)
        result.r_ic = _np.asarray(r_ic, dtype=_np.float32)
        result.S_i_per_c = _np.asarray(S_i_per_c, dtype=_np.float32)
        result.S_map = _np.asarray(S_map, dtype=_np.float32)
        result.S_post_mean = _np.asarray(S_post_mean, dtype=_np.float32)
        return result
    except Exception as _exc:
        # Surface silent data loss so a regression doesn't hide behind a
        # backward-compat fallback ( WARNING 2).
        import sys as _sys
        print(f"[heldout] WARN: per-subject LL persistence disabled: "
              f"{type(_exc).__name__}: {_exc}", file=_sys.stderr, flush=True)
        return heldout_mean



