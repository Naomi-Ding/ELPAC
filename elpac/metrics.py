"""Recovery metrics and calibration for elpac.

Scores a fitted posterior against simulator ground truth (subtype accuracy and
ARI, stage Spearman rho and MAE, mean-trajectory error, loading Procrustes
distance, activation recovery, ECE, Brier, marginal log-likelihood). Called by
run_synthetic.py at the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import jax
import numpy as np
import yaml

from .model import ModelConfig
from .simulate import SyntheticShape, simulate_synthetic
from .inference import TrainConfig, run_full_pipeline, run_phase, _build_model_inputs, extract_posterior_means, auto_loc_to_init_values


def _varimax(A: np.ndarray, max_iter: int = 500, tol: float = 1e-6) -> tuple:
    """Varimax rotation of a (R, K) loading matrix.

 Returns (A_rot, rotation_matrix) where A_rot = A @ rotation_matrix has
 maximally simple structure (columns with loadings close to 0 or +/-1).
 """
    R_, K_ = A.shape
    rotation = np.eye(K_, dtype=A.dtype)
    d = 0.0
    for _ in range(max_iter):
        Lambda = A @ rotation
        u, s, vt = np.linalg.svd(
            A.T @ (Lambda ** 3 - Lambda @ np.diag(np.sum(Lambda ** 2, axis=0)) / R_),
            full_matrices=False,
        )
        new_rotation = u @ vt
        d_new = np.sum(s)
        if abs(d_new - d) < tol:
            break
        rotation = new_rotation
        d = d_new
    return A @ rotation, rotation


def _sign_correct_factors(
    A_post: np.ndarray,
    w_post: np.ndarray,
    A_truth: np.ndarray,
) -> tuple:
    """Flip signs of factors so the largest absolute loading matches truth sign.

 The lower-triangular constraint fixes most rotation freedom but NOT sign
 flips (diagonal element could be positive in both A and -A after rotation).
 For each factor k, if the dot product A_post[:, k]. A_truth[:, k] < 0,
 negate A_post[:, k] and w_post[k, :].

 Returns (A_corrected, w_corrected).
 """
    A_out = A_post.copy()
    w_out = w_post.copy()
    K_ = A_out.shape[1]
    for k in range(K_):
        if np.dot(A_out[:, k], A_truth[:, k]) < 0:
            A_out[:, k] *= -1.0
            w_out[k, :] *= -1.0
    return A_out, w_out


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _shape_from_cfg(cfg_dict: Dict[str, Any], key: str) -> SyntheticShape:
    ms = cfg_dict["testing"]["smoke_test"]["small_shape"] if key == "small-scale" else cfg_dict["model_shape"]
    # Use simulation overrides for the synthetic case
    sim = cfg_dict.get("simulation", {})
    if key == "synthetic":
        return SyntheticShape(
            n=sim.get("n_sim", 200),
            C=sim.get("true_C", 3),
            K=sim.get("true_K", 3),
            J=sim.get("true_J", 10),
            P=ms["P"],
            Q=ms["Q"],
            R_A=7, R_T=7, R_N=6, R_blood=0,   # 20 total for synthetic
            p_x=ms["p_x"],
            p_w=ms["p_w"],
        )
    return SyntheticShape(
        n=ms["n"], C=ms["C"], K=ms["K"], J=ms["J"], P=ms["P"], Q=ms["Q"],
        R_A=ms["R_A"], R_T=ms["R_T"], R_N=ms["R_N"], R_blood=ms["R_blood"],
        p_x=ms["p_x"], p_w=ms["p_w"],
    )


def _model_cfg_from_shape(shape: SyntheticShape, w_anchor: float, gumbel_tau: float) -> ModelConfig:
    return ModelConfig(
        n=shape.n, C=shape.C, K=shape.K, J=shape.J, P=shape.P, Q=shape.Q,
        R_A=shape.R_A, R_T=shape.R_T, R_N=shape.R_N, R_blood=shape.R_blood,
        p_x=shape.p_x, p_w=shape.p_w, w_anchor=w_anchor, gumbel_tau=gumbel_tau,
        use_cognition=False,
    )


def _subtype_posterior_from_trace_for_metric(truth, post, params, phase: str) -> np.ndarray:
    """Compute per-subject (n, C) subtype posterior from fitted post/params.

 Always rebuilds the per-(c, i) log-likelihood matrix. q_c from the
 amortized guide is NOT used (currently untrained — anchor supervision
 disabled in amortized_guide.py).
 """
    import numpy as _np

    # NOTE: q_c from the amortized guide is currently UNTRAINED (anchor
    # supervision disabled in amortized_guide.py). Always use LL-based
    # assignment for subtype posterior computation.
    from .utils.bspline import build_bspline_eval_matrix, eval_bspline_basis
    from .utils.bspline import cubic_hermite_interpolate

    C = truth.shape.C
    R = truth.shape.R
    J = truth.shape.J
    n = len(truth.Y)
    # Infer K from posterior so K_model can exceed K_truth without
    # crashing the zero-filled fallback. In phase1 neither w_ckp nor A_c is
    # present; the fallback K doesn't matter (A_c and w_ckp are zeros, so
    # factor terms drop out of the log-likelihood) but needs to be self-consistent.
    if phase == "phase2" and "w_ckp" in post:
        K = int(_np.asarray(post["w_ckp"]).shape[1])
    elif phase == "phase2" and "A_c" in post:
        K = int(_np.asarray(post["A_c"]).shape[2])
    else:
        K = truth.shape.K

    # Read logit_S from post (guide-agnostic) with params fallback.
    if "logit_S" in post:
        v = _np.asarray(post["logit_S"])
        if _np.all((v >= 0) & (v <= 1)):
            logit_S_mean = _np.log(_np.clip(v, 1e-6, 1 - 1e-6) /
                                   _np.clip(1 - v, 1e-6, 1 - 1e-6))
        else:
            logit_S_mean = v
    else:
        logit_S_mean = _np.asarray(
            params.get("logit_S_auto_loc",
                       params.get("logit_S_loc", _np.zeros(n)))
        )
    if logit_S_mean.ndim == 2:
        S_i_c_grid = 1.0 / (1.0 + _np.exp(-logit_S_mean))   # (n, C)
    else:
        S_i_c_grid = _np.tile(1.0 / (1.0 + _np.exp(-logit_S_mean))[:, None], (1, C))

    Theta_J = _np.asarray(post["Theta_J"])
    onset_times = _np.asarray(post["onset_times"])
    sharpness = _np.asarray(post["sharpness"])
    delta_region = _np.asarray(post["delta_region"])
    beta_vel = _np.asarray(post["beta_vel"])
    log_D = _np.asarray(post["log_D"])
    D_c = _np.exp(log_D) + 1e-4
    alpha_c = _np.asarray(post["alpha_c"])
    beta_c_sub = _np.asarray(post["beta_c_sub"])
    # Read A_c / w_ckp independently so a partial posterior (one present, other
    # missing — should never happen in practice, but be defensive) doesn't get
    # silently zero-filled when real values are available. When both are
    # missing (phase1), zero-fill at the K we inferred above.
    if phase == "phase2" and "A_c" in post:
        A_c = _np.asarray(post["A_c"])
    else:
        A_c = _np.zeros((C, R, K))
    if phase == "phase2" and "w_ckp" in post:
        w_ckp = _np.asarray(post["w_ckp"])
    else:
        P_fallback = truth.shape.P
        w_ckp = _np.zeros((C, K, P_fallback))
    # P sensitivity: psi must be evaluated at the inference's P
    # (= w_ckp.shape[-1]), NOT truth.shape.P. When --model-p override is
    # in effect, these differ; when no override, w_ckp.shape[-1] ==
    # truth.shape.P (=5 default) so the basis is bit-identical.
    P_inf = int(w_ckp.shape[-1])

    # Infer Q from the posterior beta_vel shape, NOT from truth.shape.Q,
    # to support model misspecification sweeps where model Q != truth Q.
    Q_model = beta_vel.shape[2]
    B_shell = _np.asarray(build_bspline_eval_matrix(n_basis=Q_model, n_eval=J + 1))
    tau_grid = _np.linspace(0.0, 1.0, J + 1).astype(_np.float32)
    mod_signs = _np.ones(R, dtype=_np.float32)
    sb = truth.shape.R_A + truth.shape.R_T
    se = sb + truth.shape.R_N
    mod_signs[sb:se] = -1.0
    if truth.shape.R_blood > 0:
        mod_signs[se] = -1.0

    raw = _np.einsum("crq,jq->cjr", beta_vel, B_shell)
    softplus_raw = _np.log1p(_np.exp(raw))
    vmag = mod_signs[None, None, :] * softplus_raw
    onset_arg = (tau_grid[None, :, None] - onset_times[:, None, :]) / _np.clip(sharpness[:, None, :], 1e-3, None)
    onset_g = 1.0 / (1.0 + _np.exp(-onset_arg))
    V = delta_region[:, None, :] * onset_g * vmag

    theta_shells = _np.zeros((C, J + 1, R), dtype=_np.float32)
    theta_shells[:, -1, :] = Theta_J
    dj = 1.0 / J
    for j in range(J, 0, -1):
        theta_shells[:, j - 1, :] = theta_shells[:, j, :] - dj * V[:, j, :]

    log_lik_matrix = _np.zeros((n, C), dtype=_np.float64)
    for c in range(C):
        A = A_c[c]
        D_vec = D_c[c]
        for i in range(n):
            s = float(S_i_c_grid[i, c])
            j_int = min(int(s * J), J - 1)
            u = (s - j_int / J) * J
            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            h01 = -2 * u**3 + 3 * u**2
            h11 = u**3 - u**2
            theta_i_c = (
                h00 * theta_shells[c, j_int] + h10 * dj * V[c, j_int]
                + h01 * theta_shells[c, j_int + 1] + h11 * dj * V[c, j_int + 1]
            )
            obs = truth.mask[i] > 0.5
            A_obs = A[obs]                                       # (R_obs, K)
            D_obs = D_vec[obs]                                   # (R_obs,)
            residual_obs = (truth.Y[i] - theta_i_c)[obs]
            rDinv = residual_obs / D_obs
            if phase == "phase2":
                psi_s = _np.asarray(eval_bspline_basis(P_inf, s))
                nu_ci = _np.exp(_np.clip(w_ckp[c] @ psi_s, -8.0, 8.0))
                u_vec = A_obs.T @ rDinv
                M_c = A_obs.T @ (A_obs / D_obs[:, None])         # per-subject masked
                inner = _np.diag(1.0 / _np.clip(nu_ci, 1e-6, None)) + M_c
                try:
                    solved = _np.linalg.solve(inner, u_vec)
                except _np.linalg.LinAlgError:
                    solved = _np.zeros_like(u_vec)
                quad = float(_np.sum(rDinv * residual_obs) - u_vec @ solved)
                I_plus_F = _np.eye(K) + _np.diag(nu_ci) @ M_c
                log_det_sigma = float(_np.sum(_np.log(D_obs))) + float(_np.log(abs(_np.linalg.det(I_plus_F)) + 1e-12))
            else:
                quad = float(_np.sum(rDinv * residual_obs))
                log_det_sigma = float(_np.sum(_np.log(D_obs)))
            R_obs = float(obs.sum())
            log_lik_matrix[i, c] = -0.5 * (R_obs * _np.log(2 * _np.pi) + log_det_sigma + quad)

    subtype_logits = alpha_c[None, :] + truth.X @ beta_c_sub.T
    log_prior_c = subtype_logits - _np.log(_np.sum(_np.exp(subtype_logits), axis=-1, keepdims=True))
    is_anchor_np = (truth.anchor_type >= 0) & (truth.anchor_type < C)
    log_prior_adj = log_prior_c.copy()
    for i in range(n):
        if is_anchor_np[i]:
            log_prior_adj[i] = -1e6
            log_prior_adj[i, int(truth.anchor_type[i])] = 0.0

    log_joint = log_lik_matrix + log_prior_adj
    log_joint -= _np.max(log_joint, axis=-1, keepdims=True)
    resp = _np.exp(log_joint)
    resp /= _np.sum(resp, axis=-1, keepdims=True)
    return resp


def _reconstruct_V_from_post(truth, post) -> np.ndarray:
    """Rebuild velocity tensor V (C, J+1, R) from posterior means — mirrors model.py."""
    from .utils.bspline import build_bspline_eval_matrix

    C = truth.shape.C
    R = truth.shape.R
    J = truth.shape.J

    onset_times = np.asarray(post["onset_times"])
    sharpness = np.asarray(post["sharpness"])
    delta_region = np.asarray(post["delta_region"])
    beta_vel = np.asarray(post["beta_vel"])

    # Infer Q from posterior beta_vel, not truth.shape.Q (model Q may differ)
    Q = beta_vel.shape[2]
    B_shell = np.asarray(build_bspline_eval_matrix(n_basis=Q, n_eval=J + 1))
    tau_grid = np.linspace(0.0, 1.0, J + 1).astype(np.float32)

    mod_signs = np.ones(R, dtype=np.float32)
    sb = truth.shape.R_A + truth.shape.R_T
    se = sb + truth.shape.R_N
    mod_signs[sb:se] = -1.0
    if truth.shape.R_blood > 0:
        mod_signs[se] = -1.0

    raw = np.einsum("crq,jq->cjr", beta_vel, B_shell)
    # np.logaddexp(0, x) is numerically stable softplus.
    softplus_raw = np.logaddexp(0.0, raw)
    vmag = mod_signs[None, None, :] * softplus_raw
    onset_arg = (tau_grid[None, :, None] - onset_times[:, None, :]) / np.clip(sharpness[:, None, :], 1e-3, None)
    onset_g = 1.0 / (1.0 + np.exp(-onset_arg))
    return delta_region[:, None, :] * onset_g * vmag   # (C, J+1, R)


def _reconstruct_theta_shells_from_post(truth, post, forward_time: bool = False) -> np.ndarray:
    """Rebuild theta_c_shells (C, J+1, R) from the posterior means.

 Uses Theta_J (reverse anchor) or Theta_0 (forward anchor), beta_vel,
 onset_times, sharpness, delta_region to recompute V and integrate.
 Reverse path is the default and matches ``_reverse_time_integration``
 in ``src/model.py``; forward path matches ``_forward_time_integration``
 (used by the M2_forward ablation, ``cfg.forward_time=True``).
 Returns (C, J+1, R).
 """
    import numpy as _np
    from .utils.bspline import build_bspline_eval_matrix

    C = truth.shape.C
    R = truth.shape.R
    J = truth.shape.J

    Theta_J = _np.asarray(post["Theta_J"])
    onset_times = _np.asarray(post["onset_times"])
    sharpness = _np.asarray(post["sharpness"])
    delta_region = _np.asarray(post["delta_region"])
    beta_vel = _np.asarray(post["beta_vel"])

    # Infer Q from posterior beta_vel, not truth.shape.Q (model Q may differ)
    Q = beta_vel.shape[2]
    B_shell = _np.asarray(build_bspline_eval_matrix(n_basis=Q, n_eval=J + 1))
    tau_grid = _np.linspace(0.0, 1.0, J + 1).astype(_np.float32)

    mod_signs = _np.ones(R, dtype=_np.float32)
    sb = truth.shape.R_A + truth.shape.R_T
    se = sb + truth.shape.R_N
    mod_signs[sb:se] = -1.0
    if truth.shape.R_blood > 0:
        mod_signs[se] = -1.0

    raw = _np.einsum("crq,jq->cjr", beta_vel, B_shell)
    softplus_raw = _np.log1p(_np.exp(raw))
    vmag = mod_signs[None, None, :] * softplus_raw
    onset_arg = (tau_grid[None, :, None] - onset_times[:, None, :]) / _np.clip(sharpness[:, None, :], 1e-3, None)
    onset_g = 1.0 / (1.0 + _np.exp(-onset_arg))
    V = delta_region[:, None, :] * onset_g * vmag

    theta_shells = _np.zeros((C, J + 1, R), dtype=_np.float32)
    dj = 1.0 / J
    if forward_time:
        # Forward integration from Theta_0, matches model.py::_forward_time_integration.
        # Under forward mode Theta_J is a soft-constrained latent, so anchoring on
        # Theta_J would give reconstructed shells that disagree with the model's output.
        Theta_0 = _np.asarray(post["Theta_0"])
        if Theta_0.ndim == 1:
            Theta_0 = _np.broadcast_to(Theta_0, (C, R))
        theta_shells[:, 0, :] = Theta_0
        for j in range(1, J + 1):
            theta_shells[:, j, :] = theta_shells[:, j - 1, :] + dj * V[:, j, :]
    else:
        theta_shells[:, -1, :] = Theta_J
        for j in range(J, 0, -1):
            theta_shells[:, j - 1, :] = theta_shells[:, j, :] - dj * V[:, j, :]
    return theta_shells


def _compute_subtype_and_ll_metrics(truth, post, params, phase: str) -> Dict[str, Any]:
    """Compute subtype ARI and in-sample marginal log-likelihood.

 Rebuilds the (n, C) log-joint matrix from posterior means. Subtype c_hat is
 the argmax of q(c_i | Y_i, X_i). Marginal LL is sum_i logsumexp_c (log_lik +
 log_prior) averaged over subjects. These metrics support model comparison
 (held-out LL correlates tightly with the in-sample marginal LL computed here
 when the models have similar parameter counts per subject).
 """
    from sklearn.metrics import adjusted_rand_score
    import numpy as _np

    C = truth.shape.C
    R = truth.shape.R
    J = truth.shape.J
    n = truth.shape.n
    # Model K may differ from truth K when the inference shape
    # is over-specified relative to the simulator. Infer model K from the
    # posterior when available (phase2 carries w_ckp / A_c), fall back to
    # truth.K in phase1 where neither exists (same dim as a zero-padded
    # fallback). Downstream allocations (nu_all, A_c fallback) use this K.
    if phase == "phase2" and "w_ckp" in post:
        K = int(_np.asarray(post["w_ckp"]).shape[1])
    elif phase == "phase2" and "A_c" in post:
        K = int(_np.asarray(post["A_c"]).shape[2])
    else:
        K = truth.shape.K

    # Posterior means of global parameters
    Theta_J = _np.asarray(post.get("Theta_J", _np.zeros((C, R))))
    onset_times = _np.asarray(post.get("onset_times", _np.full((C, R), 0.5)))
    sharpness = _np.asarray(post.get("sharpness", _np.full((C, R), 0.1)))
    delta_region = _np.asarray(post.get("delta_region", _np.ones((C, R))))
    beta_vel = _np.asarray(post.get("beta_vel", _np.zeros((C, R, truth.shape.Q))))
    log_D = _np.asarray(post.get("log_D", _np.full((C, R), -1.0)))
    D_c = _np.exp(log_D) + 1e-4
    alpha_c = _np.asarray(post.get("alpha_c", _np.zeros(C)))
    beta_c_sub = _np.asarray(post.get("beta_c_sub", _np.zeros((C, truth.X.shape[1]))))

    # Reconstruct velocity V and theta_shells
    # Infer Q from posterior beta_vel, not truth.shape.Q (model Q may differ)
    from .utils.bspline import build_bspline_eval_matrix
    Q_model = beta_vel.shape[2]
    B_shell = _np.asarray(build_bspline_eval_matrix(n_basis=Q_model, n_eval=J + 1))
    tau_grid = _np.linspace(0.0, 1.0, J + 1).astype(_np.float32)

    mod_signs = _np.ones(R, dtype=_np.float32)
    sb = truth.shape.R_A + truth.shape.R_T
    se = sb + truth.shape.R_N
    mod_signs[sb:se] = -1.0
    if truth.shape.R_blood > 0:
        mod_signs[se] = -1.0

    raw = _np.einsum("crq,jq->cjr", beta_vel, B_shell)
    softplus_raw = _np.log1p(_np.exp(raw))
    vmag = mod_signs[None, None, :] * softplus_raw
    onset_arg = (tau_grid[None, :, None] - onset_times[:, None, :]) / _np.clip(sharpness[:, None, :], 1e-3, None)
    onset_g = 1.0 / (1.0 + _np.exp(-onset_arg))
    V = delta_region[:, None, :] * onset_g * vmag        # (C, J+1, R)

    theta_shells = _np.zeros((C, J + 1, R), dtype=_np.float32)
    theta_shells[:, -1, :] = Theta_J
    dj = 1.0 / J
    for j in range(J, 0, -1):
        theta_shells[:, j - 1, :] = theta_shells[:, j, :] - dj * V[:, j, :]

    # Extract per-subject stages — guide-agnostic via post first, then params.
    if "logit_S" in post:
        v = _np.asarray(post["logit_S"])
        if _np.all((v >= 0) & (v <= 1)):
            logit_S_mean = _np.log(_np.clip(v, 1e-6, 1 - 1e-6) /
                                   _np.clip(1 - v, 1e-6, 1 - 1e-6))
        else:
            logit_S_mean = v
    elif "logit_S_auto_loc" in params:
        logit_S_mean = _np.asarray(params["logit_S_auto_loc"])
    elif "logit_S_loc" in params:
        logit_S_mean = _np.asarray(params["logit_S_loc"])
    else:
        print(" [WARNING] logit_S not found in post or params; defaulting to zeros (all stages = 0.5)")
        logit_S_mean = _np.zeros(len(truth.Y))
    # Structured stage: shape (n, C). For subtype posterior computation we
    # need ONE stage per subject per c; use the structured values directly.
    # For the scalar S_i used by the non-structured path, take the mean over c.
    if logit_S_mean.ndim == 2:
        S_i_c = 1.0 / (1.0 + _np.exp(-logit_S_mean))   # (n, C)
        S_i = S_i_c.mean(axis=1)                         # (n,) for scalar metrics
        structured = True
    else:
        S_i_c = None
        S_i = 1.0 / (1.0 + _np.exp(-logit_S_mean))
        structured = False

    # Activation nu(S_i) — for phase1 this is nu=1; for phase2 use w_ckp
    if phase == "phase2" and "w_ckp" in post:
        w_ckp = _np.asarray(post["w_ckp"])          # (C, K, P)
        from .utils.bspline import eval_bspline_basis
        P = w_ckp.shape[2]  # infer from posterior, not truth (robust to misspec)
        if structured and S_i_c is not None:
            # Per-subtype stages: nu_all[c, i, k] uses S_i_c[i, c]
            nu_all = _np.ones((C, n, K))
            for c in range(C):
                psi_c = _np.stack([_np.asarray(eval_bspline_basis(P, float(S_i_c[i, c])))
                                   for i in range(n)], axis=0)  # (n, P)
                g_nk = _np.einsum("kp,np->nk", w_ckp[c], psi_c)
                nu_all[c] = _np.exp(_np.clip(g_nk, -8.0, 8.0))
        else:
            psi_S = _np.stack([_np.asarray(eval_bspline_basis(P, float(s))) for s in S_i], axis=0)  # (n, P)
            g_cnk = _np.einsum("ckp,np->cnk", w_ckp, psi_S)
            nu_all = _np.exp(_np.clip(g_cnk, -8.0, 8.0))   # (C, n, K)
    else:
        nu_all = _np.ones((C, n, K))

    # A_c: for phase2, extract from post; for phase1 it's zero
    if phase == "phase2" and "A_c" in post:
        A_c = _np.asarray(post["A_c"])
    else:
        A_c = _np.zeros((C, R, K))

    # Per-(c, i) log-likelihood via Sylvester/Woodbury (pure numpy version).
    # M = A[O_i]^T D[O_i]^-1 A[O_i] MUST be computed per-subject using only
    # observed biomarkers — otherwise the Woodbury determinant / quadratic
    # terms include loadings from unobserved rows, biasing the marginal LL
    # (and subtype responsibilities) in any missing-data setting.
    log_lik_matrix = _np.zeros((n, C), dtype=_np.float64)
    for c in range(C):
        A = A_c[c]
        D = D_c[c]
        for i in range(n):
            obs = truth.mask[i] > 0.5                 # (R,) bool
            A_obs = A[obs]                            # (R_obs, K)
            D_obs = D[obs]                            # (R_obs,)
            A_scaled_obs = A_obs / D_obs[:, None]
            M = A_obs.T @ A_scaled_obs                # (K, K) — per-subject now
            # Use per-subtype stage when available (structured_stage=True)
            s = float(S_i_c[i, c]) if (structured and S_i_c is not None) else float(S_i[i])
            j_int = min(int(s * J), J - 1)
            u = (s - j_int / J) * J
            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            h01 = -2 * u**3 + 3 * u**2
            h11 = u**3 - u**2
            theta_i_c = (
                h00 * theta_shells[c, j_int] + h10 * dj * V[c, j_int]
                + h01 * theta_shells[c, j_int + 1] + h11 * dj * V[c, j_int + 1]
            )
            residual_obs = (truth.Y[i] - theta_i_c)[obs]   # (R_obs,)
            rDinv = residual_obs / D_obs
            u_vec = A_obs.T @ rDinv                         # (K,)
            nu_ci = nu_all[c, i]
            inner = _np.diag(1.0 / _np.clip(nu_ci, 1e-6, None)) + M
            try:
                solved = _np.linalg.solve(inner, u_vec)
            except _np.linalg.LinAlgError:
                solved = _np.zeros_like(u_vec)
            quad = float(_np.sum(rDinv * residual_obs) - u_vec @ solved)
            log_det_D = float(_np.sum(_np.log(D_obs)))
            # For phase1 (A=0), the factor log-det is zero
            if phase == "phase1":
                log_det_sigma = log_det_D
            else:
                log_det_sigma = log_det_D + float(
                    _np.log(_np.abs(_np.linalg.det(_np.eye(K) + _np.diag(nu_ci) @ M)) + 1e-12)
                )
            R_obs = float(obs.sum())
            log_lik_matrix[i, c] = -0.5 * (R_obs * _np.log(2 * _np.pi) + log_det_sigma + quad)

    # Log-prior (n, C) and anchor override
    subtype_logits = alpha_c[None, :] + truth.X @ beta_c_sub.T
    log_prior_c = subtype_logits - _np.log(_np.sum(_np.exp(subtype_logits), axis=-1, keepdims=True))
    is_anchor = (truth.anchor_type >= 0) & (truth.anchor_type < C)
    large_neg = -1e6
    log_prior_adj = log_prior_c.copy()
    for i in range(n):
        if is_anchor[i]:
            log_prior_adj[i] = large_neg
            log_prior_adj[i, int(truth.anchor_type[i])] = 0.0

    log_joint = log_lik_matrix + log_prior_adj
    log_joint -= _np.max(log_joint, axis=-1, keepdims=True)  # stabilize
    log_marginal = _np.log(_np.sum(_np.exp(log_joint), axis=-1)) + _np.max(log_lik_matrix + log_prior_adj, axis=-1)

    # Subtype posterior from log-likelihood matrix (primary path;
    # q_c from amortized guide is currently untrained)
    resp = _np.exp(log_joint)
    resp /= _np.sum(resp, axis=-1, keepdims=True)
    c_hat = _np.argmax(resp, axis=-1)

    # Subtype metrics — compute on GENERAL subjects only:
    # - Anchors: fixed labels, trivially match, so including them inflates ARI
    # - NC: truth.subtypes is a RANDOM assignment in the simulator (no true c)
    # The only meaningful test is on general subjects where c_i is a true latent.
    is_general = truth.anchor_type < 0                                    # (n,)
    ari_general = None
    accuracy_general = None
    n_general = int(is_general.sum())
    if n_general >= 2:
        c_true_gen = truth.subtypes[is_general]
        c_hat_gen = c_hat[is_general]
        try:
            ari_general = float(adjusted_rand_score(c_true_gen, c_hat_gen))
        except (ValueError, TypeError) as e:
            print(f"  [metrics] ARI computation failed: {e}")
            ari_general = None
        # Classification accuracy on general subjects (subject to label-permutation)
        # Best accuracy under any label permutation:
        from itertools import permutations
        best_acc = 0.0
        best_perm = tuple(range(C))
        for perm in permutations(range(C)):
            perm_arr = _np.asarray(perm)
            mapped = perm_arr[c_hat_gen]
            acc = float(_np.mean(mapped == c_true_gen))
            if acc > best_acc:
                best_acc = acc
                best_perm = perm
        accuracy_general = best_acc

    # Also compute ARI over non-NC (for backward compat) for comparison
    non_nc = truth.anchor_type != C
    if non_nc.sum() > 0:
        ari_non_nc = float(adjusted_rand_score(truth.subtypes[non_nc], c_hat[non_nc]))
    else:
        ari_non_nc = None

    # Mean per-subject marginal LL
    mean_ll = float(_np.mean(log_marginal))

    # Subtype posterior entropy: H[p(c_i | Y_i, Ψ)] = -Σ_c resp[i,c] log resp[i,c]
    # Low entropy = confident assignment; high entropy = uncertain
    resp_safe = _np.clip(resp, 1e-10, 1.0)
    entropy_per_subject = -_np.sum(resp_safe * _np.log(resp_safe), axis=-1)  # (n,)
    max_entropy = _np.log(C)  # uniform distribution entropy

    # ---------- Calibration: ECE and Brier score on general subjects ----------
    ece = None
    brier = None
    if n_general >= 10:
        resp_gen = resp[is_general]
        c_true_gen = truth.subtypes[is_general]
        conf_gen = _np.max(resp_gen, axis=-1)
        c_pred_gen = _np.argmax(resp_gen, axis=-1)
        # Use best permutation for accuracy
        c_pred_mapped = _np.array([best_perm[c] for c in c_pred_gen]) if best_perm else c_pred_gen

        # ECE: bin by confidence, measure |accuracy - confidence| per bin
        n_bins = 10
        bin_edges = _np.linspace(0, 1, n_bins + 1)
        ece_sum = 0.0
        for b in range(n_bins):
            mask_b = (conf_gen >= bin_edges[b]) & (conf_gen < bin_edges[b + 1])
            if b == n_bins - 1:  # include right edge
                mask_b |= (conf_gen == bin_edges[b + 1])
            if mask_b.sum() > 0:
                acc_b = float(_np.mean(c_pred_mapped[mask_b] == c_true_gen[mask_b]))
                conf_b = float(_np.mean(conf_gen[mask_b]))
                ece_sum += mask_b.sum() * abs(acc_b - conf_b)
        ece = float(ece_sum / n_general)

        # Brier score: mean over subjects of sum_c (resp[i,c] - 1{c=c_true})^2
        one_hot = _np.zeros_like(resp_gen)
        one_hot[_np.arange(n_general), c_true_gen] = 1.0
        brier = float(_np.mean(_np.sum((resp_gen - one_hot) ** 2, axis=-1)))

    # ---------- Held-out LL (approximate: evaluate on general subjects only) ----------
    # General subjects' LL is a proxy for held-out performance since anchors
    # are heavily supervised and would inflate the in-sample LL
    heldout_ll_general = float(_np.mean(log_marginal[is_general])) if n_general > 0 else None

    return {
        "subtype_ari": ari_general,                         # canonical: general-only ARI
        "subtype_ari_general_only": ari_general,            # explicit same value
        "subtype_accuracy_general": accuracy_general,       # best-permutation accuracy
        "n_general_subjects": n_general,
        "subtype_ari_non_nc": ari_non_nc,                    # legacy (includes anchors)
        "mean_subject_marginal_ll": mean_ll,
        "marginal_ll_general_only": heldout_ll_general,     # proxy for held-out LL
        "n_subjects": int(n),
        # Calibration
        "ece": ece,
        "brier_score": brier,
        # Subtype posterior distribution: exact p(c_i | Y_i, Ψ) via Bayes' rule
        "subtype_posterior_entropy_mean": float(_np.mean(entropy_per_subject)),
        "subtype_posterior_entropy_general": float(_np.mean(entropy_per_subject[is_general])) if n_general > 0 else None,
        "subtype_posterior_entropy_anchor": float(_np.mean(entropy_per_subject[~is_general & (truth.anchor_type < C)])) if (~is_general & (truth.anchor_type < C)).sum() > 0 else None,
        "subtype_posterior_max_entropy": float(max_entropy),
        "subtype_posterior_mean_confidence": float(_np.mean(_np.max(resp, axis=-1))),
    }


def extract_posterior_full(truth, post, params, phase: str,
                           model_K: int | None = None,
                           truth_test=None,
                           forward_time: bool = False) -> Dict[str, np.ndarray]:
    """Gather per-subject + global posterior arrays needed by Tranche E.

 Returns a dict of numpy arrays ready for `np.savez_compressed` sidecar
 persistence. Downstream consumers (global staging, Wald phase-transition
 test, liquid-association peaks, Fisher-weight recovery) pick the fields
 they need. Keys:

 r_ic (n, C) posterior responsibilities p(c_i | Y_i, Ψ)
 S_i (n,) sigmoid(logit_S) collapsed across subtypes
 S_i_c (n, C) structured per-subtype stage (if structured_stage
 is active; otherwise broadcast S_i)
 theta_c_grid (C, J+1, R) reverse-time mean-trajectory shells
 V_c_grid (C, J+1, R) velocity shells (same grid)
 Theta_0 (R,) shared healthy-origin anchor
 Theta_J (C, R) subtype disease endpoints
 D_c (C, R) diagonal biomarker noise variances
 A_c (C, R, K) factor loadings (zeros if phase==phase1)
 nu_ci (C, n, K) per-(c, i) factor strengths at S_i_c
 w_ckp (C, K, P) B-spline coefficients for nu(s)
 alpha_c (C,) subtype prior logits
 Y (n, R) observed biomarkers (from truth)
 mask (n, R) observation mask
 anchor_type (n,) anchor flag (>=0 anchor, -1 general, ==C NC)
 subtypes_true (n,) simulator ground-truth c (for validation only)
 stages_true (n,) simulator ground-truth s (for validation only)
 C_cog (n,) CDR-SB ground truth (NaN if missing)
 log_marginal (n,) per-subject marginal log-likelihood (pre-normalization)
 log_marginal_per_c (n, C) per-subject per-subtype log-likelihood (pre-anchor-lock)
 ages_raw (n,) per-subject age in years (imputed; removes median bias)
 chi_c_raw (C, B) raw cognition spline coefs (if use_cognition)
 chi_c (C, B) softplus(chi_c_raw) — non-neg spline weights (if use_cognition)
 gamma_0 scalar cognition intercept (if use_cognition)
 gamma_X (p_x,) cognition covariate weights (if use_cognition)
 log_sigma_C scalar log cognition noise std (if use_cognition)
 onset_times (C, R) mechanistic gate midpoints (if in posterior)
 sharpness (C, R) mechanistic gate widths (if in posterior)
 Y_test (n_te, R) held-out biomarkers (if truth_test provided)
 mask_test (n_te, R) held-out observation mask (if truth_test provided)
 anchor_type_test (n_te,) held-out anchor flags (if truth_test provided)
 W_test (n_te, p) held-out covariates W (if truth_test provided)
 X_test (n_te, p) held-out covariates X (if truth_test provided)
 C_cog_test (n_te,) held-out CDR-SB (if truth_test provided)
 peak_memory_mb scalar peak resident memory at extraction time (MB, Linux)
 """
    import resource as _resource
    from .utils.bspline import eval_bspline_basis

    C = truth.shape.C
    R = truth.shape.R
    J = truth.shape.J
    n = int(truth.Y.shape[0])
    # Infer K from posterior to support K_model ≠ K_truth. The
    # extract_posterior_full sidecar stores the MODEL posterior, so K should
    # match the model's factor count (not truth's). For phase1 / M0 where the
    # posterior carries no factor sites, use the caller-supplied model_K
    # (cfg.K) so A_c / w_ckp zero-placeholders have the model-level K shape
    # rather than the truth-level K; this keeps sidecar shapes uniform across
    # variants at fixed --model-k.
    if phase == "phase2" and "w_ckp" in post:
        K = int(np.asarray(post["w_ckp"]).shape[1])
    elif phase == "phase2" and "A_c" in post:
        K = int(np.asarray(post["A_c"]).shape[2])
    elif model_K is not None:
        K = int(model_K)
    else:
        K = truth.shape.K

    # Trajectory + velocity shells (reuse verified helpers)
    theta_shells = _reconstruct_theta_shells_from_post(truth, post, forward_time=forward_time)   # (C, J+1, R)
    V_shells = _reconstruct_V_from_post(truth, post)                  # (C, J+1, R)

    # Global parameters with safe defaults (small-scale robustness)
    Theta_0 = np.asarray(post.get("Theta_0", np.zeros(R)))
    Theta_J = np.asarray(post.get("Theta_J", np.zeros((C, R))))
    log_D = np.asarray(post.get("log_D", np.full((C, R), -1.0)))
    D_c = np.exp(log_D) + 1e-4
    alpha_c = np.asarray(post.get("alpha_c", np.zeros(C)))
    beta_c_sub = np.asarray(post.get("beta_c_sub", np.zeros((C, truth.X.shape[1]))))

    # Stage extraction (mirrors _compute_subtype_and_ll_metrics; guide-agnostic)
    if "logit_S" in post:
        v = np.asarray(post["logit_S"])
        if np.all((v >= 0) & (v <= 1)):
            logit_S_mean = np.log(np.clip(v, 1e-6, 1 - 1e-6) /
                                  np.clip(1 - v, 1e-6, 1 - 1e-6))
        else:
            logit_S_mean = v
    elif "logit_S_auto_loc" in params:
        logit_S_mean = np.asarray(params["logit_S_auto_loc"])
    elif "logit_S_loc" in params:
        logit_S_mean = np.asarray(params["logit_S_loc"])
    else:
        logit_S_mean = np.zeros(n)

    if logit_S_mean.ndim == 2:
        S_i_c = 1.0 / (1.0 + np.exp(-logit_S_mean))           # (n, C)
        S_i = S_i_c.mean(axis=1)
        structured = True
    else:
        S_i = 1.0 / (1.0 + np.exp(-logit_S_mean))             # (n,)
        S_i_c = np.broadcast_to(S_i[:, None], (n, C)).copy()
        structured = False

    # Per-(c, i, k) factor strengths at each subject's (subtype-specific) stage
    if phase == "phase2" and "w_ckp" in post:
        w_ckp = np.asarray(post["w_ckp"])                      # (C, K, P)
        A_c = np.asarray(post.get("A_c", np.zeros((C, R, K))))
        P_sp = w_ckp.shape[2]
        nu_ci = np.ones((C, n, K), dtype=np.float64)
        for c in range(C):
            stages_c = S_i_c[:, c] if structured else S_i
            psi_c = np.stack(
                [np.asarray(eval_bspline_basis(P_sp, float(s))) for s in stages_c],
                axis=0,
            )                                                   # (n, P_sp)
            g_nk = np.einsum("kp,np->nk", w_ckp[c], psi_c)
            nu_ci[c] = np.exp(np.clip(g_nk, -8.0, 8.0))
    else:
        w_ckp = np.zeros((C, K, 5), dtype=np.float64)
        A_c = np.zeros((C, R, K), dtype=np.float64)
        nu_ci = np.ones((C, n, K), dtype=np.float64)

    # Per-subject log-likelihood + responsibilities (Woodbury form, observed-only)
    dj = 1.0 / J
    log_lik = np.zeros((n, C), dtype=np.float64)
    for c in range(C):
        A = A_c[c]
        D = D_c[c]
        for i in range(n):
            obs = truth.mask[i] > 0.5
            A_obs = A[obs]
            D_obs = D[obs]
            s = float(S_i_c[i, c]) if structured else float(S_i[i])
            j_int = min(int(s * J), J - 1)
            u = (s - j_int / J) * J
            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            h01 = -2 * u**3 + 3 * u**2
            h11 = u**3 - u**2
            theta_i_c = (
                h00 * theta_shells[c, j_int] + h10 * dj * V_shells[c, j_int]
                + h01 * theta_shells[c, j_int + 1] + h11 * dj * V_shells[c, j_int + 1]
            )
            residual_obs = (truth.Y[i] - theta_i_c)[obs]
            rDinv = residual_obs / D_obs
            A_scaled_obs = A_obs / D_obs[:, None]
            M = A_obs.T @ A_scaled_obs
            u_vec = A_obs.T @ rDinv
            nu_c_i = nu_ci[c, i]
            inner = np.diag(1.0 / np.clip(nu_c_i, 1e-6, None)) + M
            try:
                solved = np.linalg.solve(inner, u_vec)
            except np.linalg.LinAlgError:
                solved = np.zeros_like(u_vec)
            quad = float(np.sum(rDinv * residual_obs) - u_vec @ solved)
            log_det_D = float(np.sum(np.log(D_obs)))
            if phase == "phase1":
                log_det_sigma = log_det_D
            else:
                log_det_sigma = log_det_D + float(
                    np.log(np.abs(np.linalg.det(np.eye(K) + np.diag(nu_c_i) @ M)) + 1e-12)
                )
            R_obs = float(obs.sum())
            log_lik[i, c] = -0.5 * (R_obs * np.log(2 * np.pi) + log_det_sigma + quad)

    # Log-prior + anchor override
    subtype_logits = alpha_c[None, :] + truth.X @ beta_c_sub.T
    log_prior_c = subtype_logits - np.log(np.sum(np.exp(subtype_logits), axis=-1, keepdims=True))
    is_anchor = (truth.anchor_type >= 0) & (truth.anchor_type < C)
    large_neg = -1e6
    log_prior_adj = log_prior_c.copy()
    for i in range(n):
        if is_anchor[i]:
            log_prior_adj[i] = large_neg
            log_prior_adj[i, int(truth.anchor_type[i])] = 0.0

    log_joint = log_lik + log_prior_adj
    log_joint_max = np.max(log_joint, axis=-1, keepdims=True)
    log_joint_shifted = log_joint - log_joint_max
    exp_shifted = np.exp(log_joint_shifted)
    norm = np.sum(exp_shifted, axis=-1, keepdims=True)
    resp = exp_shifted / norm
    log_marginal = (log_joint_max + np.log(norm)).reshape(n)

    C_cog_raw = getattr(truth, "C_cog", None)
    if C_cog_raw is not None and np.asarray(C_cog_raw).shape == (n,):
        C_cog_out = np.asarray(C_cog_raw, dtype=np.float32)
    else:
        C_cog_out = np.full(n, np.nan, dtype=np.float32)

    # Truth trajectory shells — required for the global pathology score G* on
    # true simulator parameters. Present on synthetic truth objects
    # (SyntheticTruth fields are `theta_c_shells` and `v_c_shells`); absent on
    # real data.
    # Fall back to NaN so the sidecar schema is identical across synthetic and
    # real runs and downstream `scripts/global_staging.py` can rely on the key
    # being present. Shape-mismatch is a developer bug (extractor/simulator
    # drift), not a data condition — emit a loud stderr warning but still NaN-
    # fill so the rest of the sidecar survives.
    def _truth_trajectory_array(attr_name: str, target_shape, label: str) -> np.ndarray:
        raw = getattr(truth, attr_name, None)
        if raw is None:
            return np.full(target_shape, np.nan, dtype=np.float32)
        try:
            arr = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError) as e:
            print(f"[extract_posterior_full] {label}: np.asarray failed "
                  f"({type(e).__name__}: {e}); writing NaN", file=sys.stderr, flush=True)
            return np.full(target_shape, np.nan, dtype=np.float32)
        if arr.shape != target_shape:
            print(f"[extract_posterior_full] {label}: shape mismatch "
                  f"truth={arr.shape} vs post={target_shape}; writing NaN "
                  f"(simulator/extractor drift?)", file=sys.stderr, flush=True)
            return np.full(target_shape, np.nan, dtype=np.float32)
        return arr

    theta_c_shells_true_out = _truth_trajectory_array(
        "theta_c_shells", theta_shells.shape, "theta_c_shells_true")
    V_c_shells_true_out = _truth_trajectory_array(
        "v_c_shells", V_shells.shape, "V_c_shells_true")

    # Paper.1 gate-layer truth (mechanistic-vs-effective onset scatter):
    # t_{c,r} (gate midpoint), ω_{c,r} (gate width), β_{c,r,q} (post-onset
    # basis coefficients). Present on `SyntheticTruth` as the aliases
    # `t_cr_true`, `omega_cr_true`, `beta_cr_q_true` (added) and
    # persisted into the NPZ sidecar so the scatter aggregator can use the
    # mechanistic t_cr definition. Additive — skipped cleanly if absent (old
    # simulator output or real data): we emit the key only when the truth
    # object exposes it.
    out: Dict[str, np.ndarray] = {
        "r_ic":               resp.astype(np.float32),
        "log_marginal":       log_marginal.astype(np.float32),
        "S_i":                S_i.astype(np.float32),
        "S_i_c":              S_i_c.astype(np.float32),
        "theta_c_grid":       theta_shells.astype(np.float32),
        "V_c_grid":           V_shells.astype(np.float32),
        "Theta_0":            Theta_0.astype(np.float32),
        "Theta_J":            Theta_J.astype(np.float32),
        "D_c":                D_c.astype(np.float32),
        "A_c":                A_c.astype(np.float32),
        "nu_ci":              nu_ci.astype(np.float32),
        "w_ckp":              w_ckp.astype(np.float32),
        "alpha_c":            alpha_c.astype(np.float32),
        "beta_c_sub":         beta_c_sub.astype(np.float32),
        "Y":                  np.asarray(truth.Y, dtype=np.float32),
        "mask":               np.asarray(truth.mask, dtype=np.uint8),
        "anchor_type":        np.asarray(truth.anchor_type, dtype=np.int32),
        "subtypes_true":      np.asarray(truth.subtypes, dtype=np.int32),
        "stages_true":        np.asarray(truth.stages, dtype=np.float32),
        "C_cog":              C_cog_out,
        "theta_c_shells_true": theta_c_shells_true_out,
        "V_c_shells_true":    V_c_shells_true_out,
    }

    # Additive (Phase 1a): ARD column precisions / shrinkage globals. Only
    # written when present in the posterior dict — keeps sidecar schema
    # backward compatible for pre-ARD runs.
    if "alpha_ard" in post:
        out["alpha_ard"] = np.asarray(post["alpha_ard"], dtype=np.float32)
    if "delta_shrink" in post:
        out["delta_shrink"] = np.asarray(post["delta_shrink"], dtype=np.float32)

    # Additive: paper.1 gate-layer truth fields. Only written when
    # present on the truth object (synthetic runs). Absent on real data runs
    # (no aliases on a non-SyntheticTruth) — in which case the key is simply
    # not added, preserving backward compatibility for sidecar consumers.
    t_cr_true_val = getattr(truth, "t_cr_true", None)
    if t_cr_true_val is not None:
        out["t_cr_true"] = np.asarray(t_cr_true_val, dtype=np.float32)
    omega_cr_true_val = getattr(truth, "omega_cr_true", None)
    if omega_cr_true_val is not None:
        out["omega_cr_true"] = np.asarray(omega_cr_true_val, dtype=np.float32)
    beta_cr_q_true_val = getattr(truth, "beta_cr_q_true", None)
    if beta_cr_q_true_val is not None:
        out["beta_cr_q_true"] = np.asarray(beta_cr_q_true_val, dtype=np.float32)
    # residual-regression diagnostic: synthetic W covariates (age, sex, APOE, ICV).
    # Present on SyntheticTruth (simulate.py:106, assigned at simulate.py:900); absent on
    # real-data truth objects — guard with getattr.
    W_val = getattr(truth, "W", None)
    if W_val is not None:
        out["W"] = np.asarray(W_val, dtype=np.float32)

    # -----------------------------------------------------------------------
    # Extended fields (6 groups) — additive / backward-compatible
    # -----------------------------------------------------------------------

    # log_marginal_per_c — raw (n, C) per-subtype log-likelihood
    # computed by the Woodbury loop above (pre-anchor-lock, pre-prior).
    # Enables exact held-out subtype eval without re-running inference.
    out["log_marginal_per_c"] = log_lik.astype(np.float32)

    # ages_raw — per-subject age in years (from truth.ages, imputed).
    # Removes the uniform median fill that posthoc_subtype_eval currently uses.
    ages_val = getattr(truth, "ages", None)
    if ages_val is not None:
        out["ages_raw"] = np.asarray(ages_val, dtype=np.float32)

    # Cognition model parameters — present only when use_cognition=True
    # (i.e., the relevant numpyro sites were sampled). Additive: absent if the
    # run used use_cognition=False (distinguishable via d.files membership).
    if "chi_c_raw" in post:
        chi_c_raw_arr = np.asarray(post["chi_c_raw"], dtype=np.float32)
        out["chi_c_raw"] = chi_c_raw_arr
        out["chi_c"] = np.log1p(np.exp(chi_c_raw_arr.astype(np.float64))).astype(np.float32)
    if "gamma_0" in post:
        out["gamma_0"] = np.asarray(post["gamma_0"], dtype=np.float32)
    if "gamma_X" in post:
        out["gamma_X"] = np.asarray(post["gamma_X"], dtype=np.float32)
    if "log_sigma_C" in post:
        out["log_sigma_C"] = np.asarray(post["log_sigma_C"], dtype=np.float32)

    # Mechanistic parameters — onset_times / sharpness.
    # Listed in SHARED_PHASE1_TO_PHASE2_SITES; expected in phase2 posterior.
    if "onset_times" in post:
        out["onset_times"] = np.asarray(post["onset_times"], dtype=np.float32)
    if "sharpness" in post:
        out["sharpness"] = np.asarray(post["sharpness"], dtype=np.float32)

    # Held-out truth — saves a CSV reload in downstream analysis.
    # Only written when truth_test is supplied by the caller.
    if truth_test is not None:
        try:
            out["Y_test"] = np.asarray(truth_test.Y, dtype=np.float32)
            out["mask_test"] = np.asarray(truth_test.mask, dtype=np.uint8)
            out["anchor_type_test"] = np.asarray(truth_test.anchor_type, dtype=np.int32)
            _W_test = getattr(truth_test, "W", None)
            if _W_test is not None:
                out["W_test"] = np.asarray(_W_test, dtype=np.float32)
            _X_test = getattr(truth_test, "X", None)
            if _X_test is not None:
                out["X_test"] = np.asarray(_X_test, dtype=np.float32)
            _C_cog_test_raw = getattr(truth_test, "C_cog", None)
            if _C_cog_test_raw is not None:
                out["C_cog_test"] = np.asarray(_C_cog_test_raw, dtype=np.float32)
        except Exception as _e:
            print(f"[extract_posterior_full] truth_test fields: skipped ({type(_e).__name__}: {_e})",
                  file=sys.stderr, flush=True)

    # Peak resident memory (Linux: ru_maxrss is in KB; macOS: bytes).
    # Recorded at extractor exit — cumulative process peak for the full run.
    try:
        _ru = _resource.getrusage(_resource.RUSAGE_SELF)
        _kb = float(_ru.ru_maxrss)
        import platform as _platform
        _mb = _kb / 1024.0 if _platform.system() != "Darwin" else _kb / (1024.0 * 1024.0)
        out["peak_memory_mb"] = np.array(_mb, dtype=np.float32)
    except Exception:
        pass  # non-critical; omit rather than fail

    return out


def _compute_recovery_metrics(truth, post, params, phase: str,
                              forward_time: bool = False) -> Dict[str, Any]:
    """Compute synthetic recovery metrics from posterior means.

 NOTE: at small-scale with only a few hundred SVI steps, we cannot meet the
 tight full-scale thresholds. These metrics are reported as-is.

 ``forward_time=True`` routes the internal theta-shell reconstruction through
 forward integration from ``post["Theta_0"]`` so the permutation-matching
 stage agrees with the ``M2_forward`` ablation's model direction.
 """
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score
    from scipy.linalg import orthogonal_procrustes
    from itertools import permutations as _iter_perms

    metrics: Dict[str, Any] = {}

    C_shape = truth.shape.C

    # --- Subtype label permutation alignment (fix from diagnostic) ---
    # Subtype labels in a mixture model are identifiable only up to permutation.
    # All per-subtype metrics (trajectory, loading, activation, Theta_J) must
    # apply the same permutation before comparing post-index c to truth-index c.
    # We compute the best permutation by minimizing the sum of per-subtype
    # trajectory L1 distances (robust to signed differences).
    try:
        theta_post_shells = _reconstruct_theta_shells_from_post(truth, post, forward_time=forward_time)  # (C, J+1, R)
    except (KeyError, ValueError, np.linalg.LinAlgError) as e:
        print(f"  [metrics] theta_shells reconstruction failed: {type(e).__name__}: {e}")
        theta_post_shells = None

    best_perm = tuple(range(C_shape))   # identity fallback
    if theta_post_shells is not None and theta_post_shells.shape == truth.theta_c_shells.shape:
        # Cost matrix cost[i, j] = mean |theta_post_shells[j] - truth.theta_c_shells[i]|
        # (i.e., aligning post-subtype j to truth-subtype i)
        cost = np.zeros((C_shape, C_shape), dtype=np.float64)
        for i in range(C_shape):
            for j in range(C_shape):
                cost[i, j] = float(np.mean(np.abs(theta_post_shells[j] - truth.theta_c_shells[i])))
        # Brute-force permutation search (C! = 24 for C=4)
        best_cost = np.inf
        for perm in _iter_perms(range(C_shape)):
            perm_cost = sum(cost[i, perm[i]] for i in range(C_shape))
            if perm_cost < best_cost:
                best_cost = perm_cost
                best_perm = perm
    metrics["best_subtype_permutation"] = list(best_perm)

    # Apply best_perm to theta_post_shells: theta_post_aligned[i] = theta_post[perm[i]]
    # so that index i in the aligned tensor corresponds to truth index i.
    perm_list = list(best_perm)
    if theta_post_shells is not None:
        theta_post_aligned = theta_post_shells[perm_list]

        # Sup-norm metric: sup_{s, c, r} |theta_c^post(s) - theta_c^true(s)|
        # evaluated at INTERIOR stages where subjects actually live. Using shell
        # boundaries (including j=0 = origin) conflates the metric with Theta_0
        # identifiability drift (the model can fit the data with shifted Theta_0
        # and smaller velocities). The paper's trajectory is the function
        # theta_c: [0,1] -> R^R; we evaluate it at 20 points on [0.05, 0.95]
        # via cubic Hermite interpolation between shells.
        from .utils.bspline import cubic_hermite_interpolate
        J_ = truth.shape.J
        grid_stages = np.linspace(0.05, 0.95, 20)
        dj_ = 1.0 / J_

        # Reconstruct V_post (C, J+1, R) for interpolation
        V_post = _reconstruct_V_from_post(truth, post)     # (C, J+1, R)
        V_post_aligned = V_post[perm_list]

        # Truth V is available directly
        V_true = truth.v_c_shells                           # (C, J+1, R)

        def _interp_traj(theta_shells, V_shells, s_val):
            j_int = min(int(s_val * J_), J_ - 1)
            u = (s_val - j_int / J_) * J_
            out = np.empty((theta_shells.shape[0], theta_shells.shape[-1]))
            for c in range(theta_shells.shape[0]):
                out[c] = np.asarray(cubic_hermite_interpolate(
                    theta_shells[c, j_int], theta_shells[c, j_int + 1],
                    V_shells[c, j_int], V_shells[c, j_int + 1],
                    float(u), delta_j=float(dj_),
                ))
            return out    # (C, R)

        # Build (C, G, R) tensor at interior stages
        theta_post_grid = np.stack(
            [_interp_traj(theta_post_aligned, V_post_aligned, s) for s in grid_stages], axis=1
        )  # (C, G, R)
        theta_true_grid = np.stack(
            [_interp_traj(truth.theta_c_shells, V_true, s) for s in grid_stages], axis=1
        )  # (C, G, R)

        diff = theta_post_grid - theta_true_grid
        metrics["mean_trajectory_sup_norm"] = float(np.max(np.abs(diff)))
        metrics["mean_trajectory_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
        metrics["mean_trajectory_per_subtype_sup_norm"] = [
            float(np.max(np.abs(diff[c, :, :]))) for c in range(diff.shape[0])
        ]
        # Legacy shell-boundary diagnostics (useful for debugging identifiability)
        shell_diff = theta_post_aligned - truth.theta_c_shells
        metrics["mean_trajectory_shell_sup_norm"] = float(np.max(np.abs(shell_diff)))
        metrics["mean_trajectory_per_shell_mae"] = [
            float(np.mean(np.abs(shell_diff[:, j, :]))) for j in range(shell_diff.shape[1])
        ]
    else:
        metrics["mean_trajectory_error"] = "reconstruction failed"

    if "Theta_J" in post:
        theta_J_post = np.asarray(post["Theta_J"])[perm_list]
        theta_J_true = truth.theta_c_shells[:, -1, :]
        metrics["theta_J_mae"] = float(np.mean(np.abs(theta_J_post - theta_J_true)))

    # Stage correlation: read logit_S directly from the constrained posterior
    # dict (`post`), which works for ALL guide types (AutoNormal, AutoLowRank,
    # amortized) — extract_posterior_means already handles guide-specific
    # extraction. The old `logit_S_auto_loc` lookup only worked for AutoNormal.
    # Need to convert sigmoid-space S back to logit space for the structured
    # branch's grid math; if post["logit_S"] looks like a sigmoid-space stage
    # already (in [0, 1]), invert it.
    logit_S_post = None
    if "logit_S" in post:
        v = np.asarray(post["logit_S"])
        if np.all((v >= 0) & (v <= 1)):
            # Already in stage space, convert to logit
            logit_S_post = np.log(np.clip(v, 1e-6, 1 - 1e-6) /
                                  np.clip(1 - v, 1e-6, 1 - 1e-6))
        else:
            logit_S_post = v
    elif params is not None and "logit_S_auto_loc" in params:
        logit_S_post = np.asarray(params["logit_S_auto_loc"])
    elif params is not None and "logit_S_loc" in params:
        logit_S_post = np.asarray(params["logit_S_loc"])

    if logit_S_post is not None:
        logit_S_mean = logit_S_post
        C_shape = truth.shape.C
        is_anchor_mask = (truth.anchor_type >= 0) & (truth.anchor_type < C_shape)
        is_nc_mask = truth.anchor_type == C_shape
        if logit_S_mean.ndim == 2:
            # Structured (n, C) posterior. Per-subject stage is computed via the
            # FULL subtype posterior (responsibility-weighted stage), which is
            # both the Bayes-optimal point estimate and consistent with how a
            # downstream user would query "what stage is subject i at?":
            # hat_S_i = E_{q(c|Y_i,X_i)} [E_{q(S|c,Y_i,X_i)} S_{i,c}]
            # = sum_c resp[i,c] * sigmoid(logit_S_mean[i,c])
            n = len(truth.stages)
            try:
                resp = _subtype_posterior_from_trace_for_metric(truth, post, params, phase)
            except (KeyError, ValueError, np.linalg.LinAlgError) as e:
                print(f"  [CRITICAL WARNING] Subtype posterior computation FAILED: {e}")
                print(f"  Falling back to uniform responsibilities -- stage metrics will be unreliable")
                resp = np.ones((n, C_shape)) / C_shape

            # Override: anchors use one-hot on their known c; NC use one-hot on
            # c=0 (since NC stage is pinned at c=0 in the structured prior).
            resp_eff = resp.copy()
            for i in range(n):
                if is_anchor_mask[i]:
                    resp_eff[i] = 0.0
                    resp_eff[i, int(truth.anchor_type[i])] = 1.0
                elif is_nc_mask[i]:
                    resp_eff[i] = 0.0
                    resp_eff[i, 0] = 1.0

            S_c_grid = 1.0 / (1.0 + np.exp(-logit_S_mean))         # (n, C)
            # Use MAP-c (argmax) to get a sharper point estimate than the
            # responsibility-weighted one; weighted blurs when resp ≈ 0.5/0.5.
            c_map = np.argmax(resp_eff, axis=1)
            S_mean = S_c_grid[np.arange(n), c_map]
            metrics["stage_metric_uses_per_subj_c"] = True
            metrics["stage_metric_aggregation"] = "MAP_c"
        else:
            S_mean = 1.0 / (1.0 + np.exp(-logit_S_mean))
            metrics["stage_metric_uses_per_subj_c"] = False
        rho, _ = spearmanr(S_mean, truth.stages)
        metrics["stage_spearman_rho"] = float(rho) if np.isfinite(rho) else None
        metrics["stage_mae"] = float(np.mean(np.abs(S_mean - truth.stages)))
    else:
        metrics["stage_spearman_rho"] = None

    # Subtype ARI + held-out log-likelihood — computed from the posterior trace.
    # The model's mixture marginalization produces log_joint[i, c] which we can
    # rebuild from posterior means and the input data.
    try:
        subtype_metrics = _compute_subtype_and_ll_metrics(truth, post, params, phase)
        metrics.update(subtype_metrics)
    except (KeyError, ValueError, NameError, np.linalg.LinAlgError) as e:
        import traceback
        print(f"  [metrics] Subtype/LL metrics failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        metrics["subtype_ari_error"] = f"{type(e).__name__}: {e}"
        metrics["subtype_ari"] = None

    # Factor loading and activation recovery metrics (Phase 2 only).
    # Subtype permutation already applied via perm_list.
    #
    # KEY INSIGHT: The pair (A, w) has rotation ambiguity: for any orthogonal R,
    # (A*R, R^T*w) does NOT give the same likelihood because nu_k(s) = exp(w_k @ psi(s))
    # is inside a diagonal, not a quadratic form. The invariance is:
    # Sigma(s) = A diag(nu(s)) A^T + D
    # which is invariant to SIGNED PERMUTATIONS of the factor index (not general rotations).
    #
    # However, SVI with a mean-field or low-rank guide can still converge to a
    # rotated basis because the ELBO landscape has approximate symmetries near
    # the lower-triangular constraint boundary.
    #
    # Resolution strategy:
    # (a) loading_procrustes: varimax pre-rotation + sign correction + column
    # permutation + Procrustes. This resolves the rotation ambiguity for
    # the loading matrix by finding the closest orthogonal alignment.
    # (b) activation_sup_norm: compare at the COVARIANCE level. For each stage s,
    # compute Sigma_post(s) = A_post diag(nu_post(s)) A_post^T + D_post and
    # Sigma_true(s) = A_true diag(nu_true(s)) A_true^T + D_true, then report
    # the relative Frobenius norm of the difference. This is INVARIANT to
    # any rotation of (A, w) and measures what the model actually predicts.
    # (c) As a secondary diagnostic, also report the per-factor activation
    # metric after varimax+sign+perm alignment (may be large due to
    # residual rotation, but useful for debugging).
    if phase == "phase2" and "A_c" in post and "w_ckp" in post:
        from .utils.bspline import eval_bspline_basis

        A_post_s = np.asarray(post["A_c"])[perm_list]
        w_ckp_s = np.asarray(post["w_ckp"])[perm_list]
        log_D_post = np.asarray(post["log_D"])[perm_list] if "log_D" in post else None
        R_ = A_post_s.shape[1]
        K_ = A_post_s.shape[2]
        K_truth = int(truth.A_c.shape[-1])
        # K_ (model) may differ from K_truth when the inference
        # shape is over-specified relative to the simulator. Procrustes /
        # activation_sup_norm / best-factor-permutation require K_ == K_truth
        # (they align a (R, K_) subspace to a (R, K_truth) subspace via an
        # orthogonal rotation — undefined when dims differ). cov_frobenius,
        # correlation_sup_norm, and omega_sup_norm are all covariance-level
        # (R × R) and rotation-invariant, so they still work under K_ != K_truth.
        K_matches = (K_ == K_truth)

        grid = np.linspace(0.05, 0.95, 21)
        # P-sensitivity: when --model-p override is in effect, the
        # inference w_ckp_s has shape (C, K_, P_post) where P_post may differ
        # from truth.shape.P (=5 default). Build separate psi grids per side
        # so the (post, truth) comparison evaluates each at its own basis.
        P_post = int(w_ckp_s.shape[-1])
        psi_grid = np.stack([np.asarray(eval_bspline_basis(P_post, s)) for s in grid], axis=0)
        if P_post == truth.shape.P:
            psi_grid_truth = psi_grid
        else:
            psi_grid_truth = np.stack(
                [np.asarray(eval_bspline_basis(truth.shape.P, s)) for s in grid], axis=0
            )

        procrustes_per_c = []
        activation_sup_per_c = []
        cov_frob_per_c = []
        factor_perms_per_c = []
        for c in range(A_post_s.shape[0]):
            if K_matches:
                # Step 1: Varimax pre-rotation to find the most interpretable basis.
                A_varimax, R_v = _varimax(A_post_s[c])
                w_varimax = R_v.T @ w_ckp_s[c]                              # (K, P)

                # Step 2: Sign correction + column permutation + Procrustes for loading metric.
                best_fp = list(range(K_))
                best_residual = np.inf
                best_R = np.eye(K_)
                for fperm in _iter_perms(range(K_)):
                    fperm_list = list(fperm)
                    A_permuted = A_varimax[:, fperm_list]
                    A_signed, _ = _sign_correct_factors(
                        A_permuted, w_varimax[fperm_list], truth.A_c[c]
                    )
                    try:
                        R_rot, _ = orthogonal_procrustes(A_signed, truth.A_c[c])
                    except (np.linalg.LinAlgError, ValueError):
                        R_rot = np.eye(K_)
                    residual = float(np.linalg.norm(A_signed @ R_rot - truth.A_c[c]))
                    if residual < best_residual:
                        best_residual = residual
                        best_fp = fperm_list
                        best_R = R_rot
                factor_perms_per_c.append(best_fp)
                denom = np.linalg.norm(truth.A_c[c]) + 1e-6
                procrustes_per_c.append(float(best_residual / denom))
            else:
                # K mismatch: loading-level metrics undefined; mark None and
                # fall through to the K-invariant covariance metrics below.
                factor_perms_per_c.append(list(range(K_)))
                procrustes_per_c.append(None)

            # Step 3: Covariance-level activation metric (rotation-invariant,
            # K-invariant — both sides build their own (R, R) Sigma from their
            # own K). Works even when K_ != K_truth.
            D_post_c = np.exp(log_D_post[c]) if log_D_post is not None else np.zeros(A_post_s.shape[1])
            D_true_c = truth.D_c[c] if hasattr(truth, 'D_c') else np.zeros(A_post_s.shape[1])
            cov_frob_stages = []
            for g in range(len(grid)):
                psi_s_post = psi_grid[g]                                        # P_post
                psi_s_true = psi_grid_truth[g]                                  # truth.shape.P
                nu_post_s = np.exp(np.clip(w_ckp_s[c] @ psi_s_post, -8.0, 8.0))  # (K_,)
                Sigma_post = A_post_s[c] @ np.diag(nu_post_s) @ A_post_s[c].T + np.diag(D_post_c)
                nu_true_s = np.exp(np.clip(truth.w_ckp[c] @ psi_s_true, -8.0, 8.0))  # (K_truth,)
                Sigma_true = truth.A_c[c] @ np.diag(nu_true_s) @ truth.A_c[c].T + np.diag(D_true_c)
                frob_diff = np.linalg.norm(Sigma_post - Sigma_true, 'fro')
                frob_denom = np.linalg.norm(Sigma_true, 'fro') + 1e-6
                cov_frob_stages.append(float(frob_diff / frob_denom))
            cov_frob_per_c.append(float(np.mean(cov_frob_stages)))

            # Step 4: Per-factor activation diagnostic (K-matched only).
            if K_matches:
                w_permuted = w_varimax[best_fp]
                _, w_signed = _sign_correct_factors(
                    A_varimax[:, best_fp], w_permuted, truth.A_c[c]
                )
                w_rotated = best_R.T @ w_signed
                # per-side psi grids (post uses P_post; truth uses
                # truth.shape.P). When P_post == truth.shape.P, both are the
                # same array so this is bit-exact for legacy P_post = 5.
                nu_post = np.exp(np.clip(np.einsum("kp,gp->gk", w_rotated, psi_grid), -8.0, 8.0))
                nu_true = np.exp(np.clip(np.einsum("kp,gp->gk", truth.w_ckp[c], psi_grid_truth), -8.0, 8.0))
                rel_err = np.max(np.abs(nu_post - nu_true), axis=0) / (np.max(np.abs(nu_true), axis=0) + 1e-6)
                activation_sup_per_c.append(float(np.mean(rel_err)))
            else:
                activation_sup_per_c.append(None)

        def _safe_nanmean(xs):
            """np.nanmean but treats Python None as NaN and returns NaN on all-None lists."""
            arr = np.array([np.nan if v is None else v for v in xs], dtype=float)
            if not np.isfinite(arr).any():
                return float("nan")
            return float(np.nanmean(arr))

        metrics["loading_procrustes_per_subtype"] = procrustes_per_c
        metrics["loading_procrustes_mean"] = _safe_nanmean(procrustes_per_c)
        metrics["activation_sup_norm_per_subtype"] = activation_sup_per_c
        metrics["activation_sup_norm_relative_mean"] = _safe_nanmean(activation_sup_per_c)
        metrics["cov_frobenius_per_subtype"] = cov_frob_per_c
        metrics["cov_frobenius_mean"] = _safe_nanmean(cov_frob_per_c)
        metrics["best_factor_permutations"] = factor_perms_per_c

        # --- correlation_sup_norm: compare correlation matrices post vs truth ---
        try:
            from .analysis.correlations import correlation_trajectory as _corr_traj
            corr_sup_per_c = []
            R_ = A_post_s.shape[1]
            for c in range(A_post_s.shape[0]):
                D_post_c = np.exp(log_D_post[c]) if log_D_post is not None else np.zeros(R_)
                D_true_c = truth.D_c[c] if hasattr(truth, 'D_c') else np.zeros(R_)
                max_diff = 0.0
                for r in range(R_):
                    for rp in range(r + 1, R_):
                        rho_post = _corr_traj(A_post_s[c], w_ckp_s[c], D_post_c, psi_grid, r, rp)
                        rho_true = _corr_traj(truth.A_c[c], truth.w_ckp[c], D_true_c, psi_grid, r, rp)
                        max_diff = max(max_diff, float(np.max(np.abs(rho_post - rho_true))))
                corr_sup_per_c.append(max_diff)
            metrics["correlation_sup_norm_per_subtype"] = corr_sup_per_c
            metrics["correlation_sup_norm"] = float(np.max(corr_sup_per_c))
        except (KeyError, ValueError) as e:
            print(f"  [metrics] correlation_sup_norm failed: {type(e).__name__}: {e}")
            metrics["correlation_sup_norm"] = None

        # --- omega_sup_norm: compare O-information trajectory post vs truth ---
        try:
            from .analysis.o_information import o_information_trajectory as _omega_traj
            omega_sup_per_c = []
            for c in range(A_post_s.shape[0]):
                D_post_c = np.exp(log_D_post[c]) if log_D_post is not None else np.zeros(R_)
                D_true_c = truth.D_c[c] if hasattr(truth, 'D_c') else np.zeros(R_)
                omega_post = _omega_traj(A_post_s[c], w_ckp_s[c], D_post_c, psi_grid)
                omega_true = _omega_traj(truth.A_c[c], truth.w_ckp[c], D_true_c, psi_grid)
                omega_sup_per_c.append(float(np.max(np.abs(omega_post - omega_true))))
            metrics["omega_sup_norm_per_subtype"] = omega_sup_per_c
            metrics["omega_sup_norm"] = float(np.max(omega_sup_per_c))
        except (KeyError, ValueError) as e:
            print(f"  [metrics] omega_sup_norm failed: {type(e).__name__}: {e}")
            metrics["omega_sup_norm"] = None

        # --- phase_transition_error: compare detected phase-transition s* vs truth ---
        try:
            from .analysis.o_information import (
                o_information_trajectory as _omega_traj_pt,
                phase_transition_crossings,
            )
            if truth.phase_transition_s_star is not None:
                # Use subtype c=0 (typical AD) for phase transition detection
                D_post_c0 = np.exp(log_D_post[0]) if log_D_post is not None else np.zeros(R_)
                omega_post_c0 = _omega_traj_pt(A_post_s[0], w_ckp_s[0], D_post_c0, psi_grid)
                crossings = phase_transition_crossings(omega_post_c0, grid)
                if len(crossings) > 0:
                    # Pick the crossing closest to truth
                    best_idx = int(np.argmin(np.abs(crossings - truth.phase_transition_s_star)))
                    metrics["phase_transition_s_star_post"] = float(crossings[best_idx])
                    metrics["phase_transition_error"] = float(
                        np.abs(crossings[best_idx] - truth.phase_transition_s_star)
                    )
                else:
                    metrics["phase_transition_s_star_post"] = None
                    metrics["phase_transition_error"] = None
                metrics["phase_transition_s_star_truth"] = float(truth.phase_transition_s_star)
                metrics["phase_transition_n_crossings"] = len(crossings)
            else:
                metrics["phase_transition_error"] = None
        except (KeyError, ValueError) as e:
            print(f"  [metrics] phase_transition_error failed: {type(e).__name__}: {e}")
            metrics["phase_transition_error"] = None

        # --- fisher_weights_mae: compare Fisher info weights post vs truth ---
        try:
            from .analysis.fisher_weights import fisher_weights_trajectory as _fw_traj
            modality_of_r = truth.shape.modality_of_r
            M_ = int(modality_of_r.max()) + 1
            J_ = truth.shape.J
            dj_ = 1.0 / J_
            fw_mae_per_c = []
            for c in range(A_post_s.shape[0]):
                D_post_c = np.exp(log_D_post[c]) if log_D_post is not None else np.zeros(R_)
                D_true_c = truth.D_c[c] if hasattr(truth, 'D_c') else np.zeros(R_)
                # Compute theta_prime on the grid via finite differences on
                # interpolated theta trajectory (post and truth).
                def _theta_prime_on_grid(theta_shells, v_shells):
                    """Compute d theta / ds at each grid point via Hermite derivative."""
                    from .utils.bspline import cubic_hermite_interpolate
                    G_ = len(grid)
                    theta_grid = np.zeros((G_, theta_shells.shape[-1]))
                    for g_idx in range(G_):
                        s_val = grid[g_idx]
                        j_int = min(int(s_val * J_), J_ - 1)
                        u = (s_val - j_int / J_) * J_
                        theta_grid[g_idx] = np.asarray(cubic_hermite_interpolate(
                            theta_shells[j_int], theta_shells[j_int + 1],
                            v_shells[j_int], v_shells[j_int + 1],
                            float(u), delta_j=float(dj_),
                        ))
                    # Finite difference for derivative
                    tp = np.zeros_like(theta_grid)
                    tp[1:-1] = (theta_grid[2:] - theta_grid[:-2]) / (grid[2:] - grid[:-2])[:, None]
                    tp[0] = (theta_grid[1] - theta_grid[0]) / (grid[1] - grid[0])
                    tp[-1] = (theta_grid[-1] - theta_grid[-2]) / (grid[-1] - grid[-2])
                    return tp  # (G, R)

                # Post: use permutation-aligned theta/V shells
                tp_post = _theta_prime_on_grid(
                    theta_post_aligned[c] if theta_post_shells is not None else truth.theta_c_shells[c],
                    V_post_aligned[c] if theta_post_shells is not None else truth.v_c_shells[c],
                )
                tp_true = _theta_prime_on_grid(truth.theta_c_shells[c], truth.v_c_shells[c])

                fw_post = _fw_traj(A_post_s[c], w_ckp_s[c], D_post_c, psi_grid, tp_post, modality_of_r, M_)
                fw_true = _fw_traj(truth.A_c[c], truth.w_ckp[c], D_true_c, psi_grid, tp_true, modality_of_r, M_)
                fw_mae_per_c.append(float(np.mean(np.abs(fw_post - fw_true))))
            metrics["fisher_weights_mae_per_subtype"] = fw_mae_per_c
            metrics["fisher_weights_mae"] = float(np.mean(fw_mae_per_c))
        except (KeyError, ValueError) as e:
            print(f"  [metrics] fisher_weights_mae failed: {type(e).__name__}: {e}")
            metrics["fisher_weights_mae"] = None

    # ---------- K_eff: effective number of factors ----------
    # K_eff is determined by the Bhattacharya-Dunson
    # multiplicative gamma shrinkage on loadings A_c, NOT by tau_k (which is
    # the activation-smoothness hyperparameter on w_{c,k}). We use
    # loading column norms ||A_c[:, k]||_2 relative to each subtype's max.
    #
    # Procedure: for each subtype c, compute column norms of A_c, normalize
    # by the max column norm (so the largest column has relative norm 1).
    # Count columns exceeding 10% of the max — these are "effective" factors.
    # K_eff is the median over subtypes.
    if "A_c" in post:
        A_c_post = np.asarray(post["A_c"])                       # (C, R, K)
        col_norms = np.linalg.norm(A_c_post, axis=1)              # (C, K)
        # Relative K_eff: columns exceeding 10% of max column norm per subtype.
        # Paper-principled (matches MGP shrinkage ordering) but INSENSITIVE to
        # absolute magnitude (can't detect weak-but-equal columns).
        max_per_c = np.max(col_norms, axis=-1, keepdims=True)
        col_norms_rel = col_norms / np.where(max_per_c > 1e-8, max_per_c, 1.0)
        k_eff_per_c = np.sum(col_norms_rel > 0.1, axis=-1)        # (C,)
        k_eff = int(np.median(k_eff_per_c))
        # Absolute K_eff: columns with ||A_c[:,k]||_2 > τ_abs=0.5
        # (addresses weak-dep detection — relative metric always
        # returns K_eff=K when all columns have similar magnitude, even if all
        # weak). τ_abs=0.5 roughly corresponds to "coupling strength detectable
        # above typical noise" given paper's D_c ~ exp(N(-1, 1)) prior.
        k_eff_per_c_abs = np.sum(col_norms > 0.5, axis=-1)         # (C,)
        k_eff_absolute = int(np.median(k_eff_per_c_abs))
        # Composite K_eff: ||A_c[:,k]||_2 · max_s ν_c,k(s) > 0.5. Couples
        # loading magnitude with activation amplitude — detects columns that
        # are weak by EITHER criterion. Primary metric for
        # negative controls (so the
        # summary carries this field on every M0/M1/M2 record).
        k_eff_composite = -1
        k_eff_composite_max = -1
        try:
            from elpac.utils.bspline import eval_bspline_basis as _eval_bs
            if "w_ckp" in post:
                w_ckp_post = np.asarray(post["w_ckp"])
                _stage_grid = np.linspace(0.05, 0.95, 19)
                _psi_grid = np.array(
                    [_eval_bs(int(truth.shape.P), float(s)) for s in _stage_grid])
                _nu_grid = np.exp(np.clip(
                    np.einsum('ckp,gp->ckg', w_ckp_post, _psi_grid), -8.0, 8.0
                ))
                _max_nu = _nu_grid.max(axis=-1)                        # (C, K)
                _composite = col_norms * _max_nu                       # (C, K)
                _k_eff_per_c_composite = np.sum(_composite > 0.5, axis=-1)
                k_eff_composite = int(np.median(_k_eff_per_c_composite))
                k_eff_composite_max = int(_k_eff_per_c_composite.max())
                metrics["composite_A_max_nu"] = _composite.tolist()
        except Exception as _exc:
            print(f"[warn] composite K_eff failed: {_exc}")
        metrics["k_eff"] = k_eff                                   # primary: relative median-C
        metrics["k_eff_absolute"] = k_eff_absolute                 # absolute median-C
        metrics["k_eff_composite"] = k_eff_composite               # composite median-C
        metrics["k_eff_composite_max"] = k_eff_composite_max       # composite max-C
        metrics["k_eff_per_c"] = k_eff_per_c.tolist()
        metrics["k_eff_per_c_absolute"] = k_eff_per_c_abs.tolist()
        metrics["col_norms_relative"] = col_norms_rel.tolist()
        metrics["col_norms_absolute"] = col_norms.tolist()
        metrics["k_eff_matches_truth"] = int(k_eff == truth.shape.K)
        if "tau_k" in post:
            metrics["tau_k_values"] = np.asarray(post["tau_k"]).tolist()
    elif "tau_k" in post:
        # Legacy fallback only if A_c unavailable (shouldn't happen in phase 2).
        tau_k = np.asarray(post["tau_k"])
        k_eff = int(np.sum(tau_k > 0.1))
        metrics["k_eff"] = k_eff
        metrics["tau_k_values"] = tau_k.tolist()
        metrics["k_eff_matches_truth"] = int(k_eff == truth.shape.K)
        metrics["k_eff_source"] = "tau_k_fallback"

    # ---------- Reconstruction RMSE: ||Y - theta_c(s_hat)|| ----------
    # Measures how well the model predicts each subject's biomarkers
    try:
        n = len(truth.Y)
        J = truth.shape.J
        C = truth.shape.C
        if theta_post_shells is not None and "logit_S" in post:
            logit_S_val = np.asarray(post["logit_S"])
            # Get subtype assignments from the subtype metrics (already computed)
            c_recon = np.argmax(
                _subtype_posterior_from_trace_for_metric(truth, post, params, phase),
                axis=-1)

            if logit_S_val.ndim == 2:
                S_for_recon = 1.0 / (1.0 + np.exp(-logit_S_val))
                S_scalar = np.array([float(S_for_recon[i, c_recon[i]]) for i in range(n)])
            else:
                S_scalar = 1.0 / (1.0 + np.exp(-logit_S_val))

            dj = 1.0 / J
            recon_errors = []
            for i in range(n):
                c_i = int(np.clip(c_recon[i], 0, C - 1))
                s = float(np.clip(S_scalar[i], 0.0, 1.0 - 1e-6))
                j_int = min(int(s * J), J - 1)
                u = (s - j_int / J) * J
                h00 = 2*u**3 - 3*u**2 + 1; h10 = u**3 - 2*u**2 + u
                h01 = -2*u**3 + 3*u**2; h11 = u**3 - u**2
                Y_pred_i = (h00 * theta_post_shells[c_i, j_int] + h10 * dj * V_post[c_i, j_int]
                           + h01 * theta_post_shells[c_i, j_int+1] + h11 * dj * V_post[c_i, j_int+1])
                obs = truth.mask[i] > 0.5
                if obs.sum() > 0:
                    recon_errors.append(float(np.sqrt(np.mean((truth.Y[i][obs] - Y_pred_i[obs])**2))))
            if recon_errors:
                metrics["reconstruction_rmse"] = float(np.mean(recon_errors))
    except (KeyError, ValueError) as e:
        print(f"  [metrics] reconstruction_rmse failed: {type(e).__name__}: {e}")

    return metrics
