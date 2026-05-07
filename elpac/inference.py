"""Phased SVI training for elpac.

Implements:
 - Phase 1: mean-only (A_c = 0, diagonal D_c)
 - Phase 1.5: residual factor analysis for A_c initialization
 - Phase 2: full stage-varying factor covariance

and the amortized recognition network that replaces per-subject mean-field.
The guide is a custom callable that wraps AutoNormal for the globals and uses
a flax MLP for q(c_i, S_i | Y_i, X_i).

For S3 first-pass, we also support a *non-amortized* guide using
AutoNormal for everything (simpler, faster to debug). The `build_guide()`
function returns either variant.

Frozen-mean Phase 2 variant (, the paper).
------------------------------------------------------------------
`run_phase(..., frozen_mean_source=...)` and `run_full_pipeline(...,
frozen_mean_source=...)` support a mean-fixed Phase 2 ablation that holds the
Phase 1 posterior-median mean-trajectory parameters fixed (Theta_0, mu_c,
Theta_J, beta_vel, onset_times, sharpness, delta_region, pi_region) and
optimizes only the covariance module (A_c via a_flat/a_diag_raw, nu_c via
w_ckp_raw, D_c via log_D, plus the shrinkage hyper-parameters delta_shrink,
tau_k). `frozen_mean_source` may be either a path to an `.npz`/`.pkl` file
containing the Phase 1 posterior or a dict of arrays keyed by NumPyro site
name (the same format produced by `extract_posterior_means`).

Implementation: we use `numpyro.handlers.condition` on the model to mark
the frozen sites as observed at the Phase 1 medians; AutoNormal /
AutoLowRankMultivariateNormal sub-guides treat conditioned sites as
non-sampled (no variational params created). For the custom amortized and
composite guides that build sites via `numpyro.sample`, we wrap the guide
with `numpyro.handlers.block(hide=FROZEN_MEAN_SITES)` so those sites never
enter the ELBO trace. See test_frozen_mean.py for the A=0 -> M1 sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import numpyro
from numpyro.handlers import seed, trace, substitute
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoGuideList,
    AutoDelta,
)
from numpyro import handlers
from numpyro.infer.initialization import init_to_median, init_to_value, init_to_uniform

from .model import ModelConfig, build_numpyro_model
from .utils.bspline import eval_bspline_basis

# -----------------------------------------------------------------------------
# Frozen-mean Phase 2 variant (/ the paper)
# -----------------------------------------------------------------------------

# Sample sites that together determine the mean trajectory theta_c(s). Freezing
# these at their Phase 1 posterior-median values isolates the covariance-module
# contribution in the M2 vs M1 comparison.
FROZEN_MEAN_SITES: Tuple[str, ...] = (
    "Theta_0",
    "mu_c",
    "Theta_J",
    "beta_vel",
    "onset_times",
    "sharpness",
    "delta_region",
    "pi_region",
)


def _load_frozen_mean_values(
    source: Any,
    site_names: Tuple[str, ...] = FROZEN_MEAN_SITES,
) -> Dict[str, jnp.ndarray]:
    """Load posterior-median mean-trajectory values from a Phase 1 checkpoint.

 `source` may be:
 - a `dict` of arrays keyed by NumPyro sample-site name (as returned by
 `extract_posterior_means` / `_posterior_medians_via_guide`);
 - a path (str / Path) to an `.npz` file whose array names match the
 sample-site names;
 - a path to a `.pkl` / `.pickle` file containing such a dict.

 Returns a dict with only the requested `site_names`. Missing sites are
 silently omitted; the caller is responsible for asserting completeness
 (the ablation is meaningful only when every mean-trajectory site is
 actually frozen).
 """
    if isinstance(source, dict):
        raw = source
    else:
        from pathlib import Path as _Path
        path = _Path(str(source))
        if not path.exists():
            raise FileNotFoundError(
                f"frozen_mean_source path does not exist: {path}"
            )
        if path.suffix.lower() == ".npz":
            with np.load(str(path), allow_pickle=False) as npz:
                raw = {k: np.asarray(npz[k]) for k in npz.files}
        elif path.suffix.lower() in (".pkl", ".pickle"):
            import pickle
            with open(path, "rb") as f:
                raw = pickle.load(f)
            if not isinstance(raw, dict):
                raise TypeError(
                    f"frozen_mean_source pickle must contain a dict, got {type(raw)}"
                )
        else:
            raise ValueError(
                f"Unsupported frozen_mean_source extension: {path.suffix} "
                f"(expected .npz, .pkl, or .pickle)"
            )
    out: Dict[str, jnp.ndarray] = {}
    for name in site_names:
        if name in raw:
            out[name] = jnp.asarray(np.asarray(raw[name]))
    return out


def _distribution_median(fn, fallback_value):
    """Extract the median from a NumPyro distribution.

 For TransformedDistribution(Normal(loc, scale), transforms), the median
 is transform(loc) — NOT fn.mean which gives E[transform(X)] and is biased
 for nonlinear transforms like Exp. For plain Normal, mean == median == loc.

 Returns fallback_value for distributions that lack.mean (e.g. Unit from
 numpyro.factor sites).
 """
    if fn is None:
        return fallback_value
    if hasattr(fn, "base_dist") and hasattr(fn, "transforms"):
        try:
            median_val = fn.base_dist.mean # For Normal, mean == median == loc
            for t in fn.transforms:
                median_val = t(median_val)
            return median_val
        except (AttributeError, NotImplementedError, TypeError):
            pass # fall through to direct.mean
    try:
        return fn.mean
    except (AttributeError, NotImplementedError):
 # Unit distributions from numpyro.factor sites lack.mean — expected
        return fallback_value


# -----------------------------------------------------------------------------
# Flax recognition network for amortized (c_i, S_i) inference
# -----------------------------------------------------------------------------

import flax.linen as nn


class RecognitionNet(nn.Module):
    """Paper the paper recognition network.

 Input: [Y_masked, mask, X] shape (2R + p_x,)
 Output: subtype logits (C,) + per-c stage (mu, log_sigma)
 """
    C: int
    hidden: Tuple[int, ...] = (128, 128)
    dropout: float = 0.1

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        x = inputs
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
            if train:
                x = nn.Dropout(self.dropout, deterministic=False)(x)
        trunk = x
 # Subtype head
        subtype_logits = nn.Dense(self.C)(trunk) # (C,)
 # Stage head: per c, produce (mu, log_sigma)
 # Build (C, 2) output by concatenating a one-hot c code
        def stage_for_c(c_onehot):
            augmented = jnp.concatenate([trunk, c_onehot], axis=-1)
            return nn.Dense(2)(augmented)
        c_eye = jnp.eye(self.C)
        stage_out = jax.vmap(stage_for_c)(c_eye) # (C, 2)
        return subtype_logits, stage_out[:, 0], stage_out[:, 1]


# -----------------------------------------------------------------------------
# Phase 1 / Phase 2 training config
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    phase1_steps: int = 600
    phase2_steps: int = 600
    lr_phase1: float = 3e-3
    lr_phase2: float = 1e-3
    grad_clip: float = 5.0
    gumbel_tau_init: float = 1.0
    gumbel_tau_final: float = 0.1
    gumbel_anneal_steps: int = 400
    seed: int = 0
 # Default "lowrank_amortized" matches the paper's variational family:
 # AutoLowRankMultivariateNormal (rank r_g=10) for global parameters
 # combined with the amortized RecognitionMLP for per-subject (mu_S, tau_S, c).
    guide_type: str = "lowrank_amortized"
    log_every: int = 100


def _build_model_inputs(
    truth, gumbel_tau: float = 0.5, cfg=None, model_shape=None,
) -> Dict[str, jnp.ndarray]:
    """Build the input dict consumed by the NumPyro model and guides.

 ``cfg`` / ``model_shape`` (both optional, mutually consistent — pass
 one): when the inference shape differs from the truth shape (Option L:
 ``--model-k``/``--model-p``/``--model-j`` overrides on synthetic, 
 the paper P/J sensitivity sweep), pass the model-side P/J here so
 ``psi_basis_cache`` is built with the model's P (rows) and the model's
 J+1 (grid points). ``cfg`` (a ModelConfig) takes precedence; if absent,
 falls back to ``model_shape``; if both absent, uses ``truth.shape`` for
 bit-exact legacy behavior.

 Without this kwarg the P sweep crashes for P_3 / P_8 with a shape
 mismatch in ``model.py:1359 g_cnk = jnp.einsum("ckp,np->cnk", w_ckp,
 psi_S)`` (w_ckp is built with cfg.P; psi_S was built with truth.shape.P).
 The J sweep at J=16 silently clipped out-of-bounds psi indices when
 cfg.J > truth.shape.J because the cache had only truth.shape.J + 1 rows.
 """
    from .utils.bspline import eval_bspline_basis
    if cfg is not None:
        P_eff = cfg.P
        J_eff = cfg.J
    elif model_shape is not None:
        P_eff = model_shape.P
        J_eff = model_shape.J
    else:
        P_eff = truth.shape.P
        J_eff = truth.shape.J
    grid = np.linspace(0.0, 1.0, J_eff + 1)
    psi_cache = np.stack([
        np.asarray(eval_bspline_basis(P_eff, s)) for s in grid
    ], axis=0).astype(np.float32)
    C_cog = np.nan_to_num(truth.C_cog, nan=0.0).astype(np.float32)
    C_cog_mask = np.isfinite(truth.C_cog).astype(np.float32)
    ages_norm = ((truth.ages - 70.0) / 10.0).astype(np.float32)
    return dict(
        Y=jnp.asarray(truth.Y.astype(np.float32)),
        mask=jnp.asarray(truth.mask.astype(np.float32)),
        X=jnp.asarray(truth.X.astype(np.float32)),
        W=jnp.asarray(truth.W.astype(np.float32)),
        ages_norm=jnp.asarray(ages_norm),
        anchor_type=jnp.asarray(truth.anchor_type, dtype=jnp.int32),
        psi_basis_cache=jnp.asarray(psi_cache),
        gumbel_tau=jnp.asarray(float(gumbel_tau), dtype=jnp.float32),
        C_cog=jnp.asarray(C_cog),
        C_cog_mask=jnp.asarray(C_cog_mask),
    )


def _make_optimizer(lr: float, grad_clip: float) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=lr),
    )


def _mom_warmstart_init_values(
    truth,
    cfg: ModelConfig,
    alpha_d: float = 0.30,
    d_min: float = 1e-3,
    jitter_floor: float = 1e-2,
) -> Dict[str, jnp.ndarray]:
    """Method-of-moments warmstart for Phase-2 factor sites.

 For each subtype c, estimate Σ̂_c from anchor-cohort residuals (which
 target Ã Ã^T + D̂ at s ≈ 1 with Ã = A · diag(ν(1))^{1/2}); take a partial
 Cholesky of (Σ̂_c − D̂_c) to seed a lower-triangular A_c. Per-factor
 normalization (S4) and the activation coefficients self-adjust during SVI;
 initial w_ckp ≈ 0 so ν_c(s) ≈ 1 across stage at the start.

 Returns a dict of init values for sites: a_flat, a_diag_pos OR a_diag_raw
 (depends on cfg.use_paper_halfnormal_diag), log_D, and w_ckp_raw_std OR
 w_ckp_raw (depends on cfg.use_paper_w_ckp_smoothness).

 Default-OFF: this helper is only invoked when caller passes mom_warmstart
 =True to run_phase. When OFF, run_phase is identical to
 pre- previous behavior.
 """
    Y = np.asarray(truth.Y, dtype=np.float64)
    mask = np.asarray(truth.mask, dtype=np.float64)
    anchor_type = np.asarray(truth.anchor_type)
    C = int(cfg.C)
    R = int(cfg.R)
    K = int(cfg.K)
    P = int(cfg.P)

    A_c_init = np.zeros((C, R, K), dtype=np.float32)
    log_D_init = np.full((C, R), float(cfg.log_d_prior_mean), dtype=np.float32)
    skipped, svd_fallbacks = [], []

    for c in range(C):
        idx = np.where(anchor_type == c)[0]
        if len(idx) < K + 2:
            skipped.append((c, len(idx)))
            continue
        Y_c = Y[idx]
        mc = mask[idx]
        denom = np.maximum(mc.sum(axis=0), 1.0)
        mu_c = (Y_c * mc).sum(axis=0) / denom
        residuals = (Y_c - mu_c) * mc
        n_eff = max(mc.sum(axis=0).mean(), 1.0)
        Sigma_hat = (residuals.T @ residuals) / n_eff

        D_init = np.maximum(alpha_d * np.diag(Sigma_hat), d_min)
        Sigma_tilde = Sigma_hat - np.diag(D_init)
        Sigma_tilde = 0.5 * (Sigma_tilde + Sigma_tilde.T)

        eigvals = np.linalg.eigvalsh(Sigma_tilde)
        ev_min = float(eigvals.min())
        if ev_min < jitter_floor:
            Sigma_tilde = Sigma_tilde + (jitter_floor - ev_min) * np.eye(R)

        try:
            L = np.linalg.cholesky(Sigma_tilde)
        except np.linalg.LinAlgError:
            U, s, _ = np.linalg.svd(Sigma_tilde)
            L = U * np.sqrt(np.maximum(s, jitter_floor))
            svd_fallbacks.append((c, ev_min))
        A_c_init[c] = L[:, :K].astype(np.float32)
        log_D_init[c] = np.log(np.maximum(D_init - cfg.d_min, 1e-6)).astype(np.float32)

    if skipped:
        print(
            f"  [WARN] {len(skipped)} subtype(s) had < K+2={K+2} anchors "
            f"and were left at A_c=0: {skipped}"
        )
    if svd_fallbacks:
        print(
            f"  [WARN] {len(svd_fallbacks)} subtype(s) Cholesky fell back "
            f"to SVD (cols are top-K eigvecs, not lower-triangular): {svd_fallbacks}."
        )

    a_flat = jnp.asarray(A_c_init)
    diag_K = np.stack([np.diag(A_c_init[c, :K, :]) for c in range(C)])

    init: Dict[str, jnp.ndarray] = {
        "a_flat": a_flat,
        "log_D": jnp.asarray(log_D_init),
    }
    if getattr(cfg, "use_paper_halfnormal_diag", True):
        init["a_diag_pos"] = jnp.asarray(np.maximum(diag_K, 1e-3).astype(np.float32))
    else:
        diag_pos = np.maximum(diag_K, 1e-3).astype(np.float64)
        a_diag_raw = np.log(np.expm1(diag_pos))
        init["a_diag_raw"] = jnp.asarray(a_diag_raw.astype(np.float32))

    w_ckp_init = jnp.zeros((C, K, P), dtype=jnp.float32)
    if getattr(cfg, "use_paper_w_ckp_smoothness", True):
        init["w_ckp_raw_std"] = w_ckp_init
    else:
        init["w_ckp_raw"] = w_ckp_init

    return init


def run_phase(
    cfg: ModelConfig,
    train_cfg: TrainConfig,
    truth,
    phase: str,
    init_values: Optional[Dict[str, Any]] = None,
    init_scale: float = 0.1,
    init_params: Optional[Dict[str, Any]] = None,
    frozen_mean_source: Optional[Any] = None,
    random_init: bool = False,
    mom_warmstart: bool = False,
) -> Dict[str, Any]:
    """Run one phase of SVI training. Returns a dict with final params + loss.

 If `init_values` is provided (a dict mapping numpyro site name to jnp array),
 the AutoNormal guide's per-site `auto_loc` is initialized at those values via
 numpyro's `init_to_value`. Sites not present in the dict fall back to
 `init_to_median`. This is how we wire the Phase 1 -> Phase 2 warm start.

 frozen_mean_source : optional path (.npz /.pkl) or dict of arrays that
 contains a Phase 1 posterior (keyed by NumPyro sample-site name). When
 provided AND `phase == "phase2"`, the mean-trajectory sites listed in
 FROZEN_MEAN_SITES are clamped at their Phase 1 posterior-median values
 for the entire Phase 2 run; only the covariance module (A_c, nu_c, D_c)
 and its shrinkage hyper-parameters are optimized. Implements the
 mean-fixed covariance ablation (/ the paper). Silently
 ignored when `phase == "phase1"`. Returns the extra key `frozen_mean`
 in the output dict so callers can audit which values were frozen.
 """
    assert phase in {"phase1", "phase2"}
    model = build_numpyro_model(cfg, phase=phase)

 # ---- Frozen-mean Phase 2 variant (/ the paper) ----
 # Condition the model on Phase 1 posterior medians for the mean-trajectory
 # sites; block the guide from sampling those sites. Result: Phase 2 SVI
 # optimizes only the covariance module variational params.
    frozen_mean: Dict[str, jnp.ndarray] = {}
    if frozen_mean_source is not None and phase == "phase2":
        frozen_mean = _load_frozen_mean_values(frozen_mean_source)
        missing = [s for s in FROZEN_MEAN_SITES if s not in frozen_mean]
        if missing:
            print(
                f"  [frozen-mean] WARNING: {len(missing)} of "
                f"{len(FROZEN_MEAN_SITES)} mean-trajectory sites missing from "
                f"checkpoint: {missing}. Proceeding with partial freeze."
            )
        else:
            print(
                f"  [frozen-mean] Clamping {len(frozen_mean)} mean-trajectory "
                f"sites to supplied posterior medians "
                f"(sites: {sorted(frozen_mean.keys())})."
            )
 # condition handler turns the listed sites into observed sites in the
 # model trace, so AutoNormal / AutoLowRank sub-guides skip them.
        model = handlers.condition(model, data=frozen_mean)

 # If caller didn't pass init_values, compute a data-driven Phase 1 init:
 # Theta_0 = mu_NC_emp (mean of NC observations)
 # mu_c = bar_Y_c (mean of anchor_c observations per subtype)
 # Theta_J = bar_Y_c (terminal state near anchor endpoints)
 # log_D = log of per-region residual variance from anchors
 # This is the same data the model's prior CENTERS use, but as an init
 # it places the variational posterior mean directly at the empirical
 # estimate instead of at AutoNormal's init_to_uniform(0.5) default.
 # Phase 2 (with warm-start init_values from Phase 1) is unchanged.
    if (init_values is None or len(init_values) == 0) and phase == "phase1":
        import numpy as _np
        Y_np = _np.asarray(truth.Y, dtype=_np.float32)
        mask_np = _np.asarray(truth.mask, dtype=_np.float32)
        anchor_np = _np.asarray(truth.anchor_type)
        C_ = cfg.C
        R_ = cfg.R
        _nc_ref = int(getattr(cfg, 'nc_subtype_idx', -1))
        nc_ref_code = _nc_ref if _nc_ref >= 0 else C_
        nc_idx = _np.where(anchor_np == nc_ref_code)[0]
        if len(nc_idx) > 0:
            nc_m = mask_np[nc_idx]
            mu_NC_emp = (Y_np[nc_idx] * nc_m).sum(axis=0) / (nc_m.sum(axis=0) + 1e-6)
        else:
            mu_NC_emp = _np.zeros(R_, dtype=_np.float32)
        mu_NC_emp = _np.where(_np.isfinite(mu_NC_emp), mu_NC_emp, 0.0)
        bar_Y_c = _np.zeros((C_, R_), dtype=_np.float32)
        log_D_emp = _np.full((C_, R_), -1.0, dtype=_np.float32)
        for c in range(C_):
            idx = _np.where(anchor_np == c)[0]
            if len(idx) > 0:
                m = mask_np[idx]
                bar_Y_c[c] = (Y_np[idx] * m).sum(axis=0) / (m.sum(axis=0) + 1e-6)
                res = (Y_np[idx] - bar_Y_c[c]) * m
                var = res.var(axis=0) + 1e-3
                log_D_emp[c] = _np.log(var)
        init_values = {
            "Theta_0": jnp.asarray(mu_NC_emp),
            "mu_c": jnp.asarray(bar_Y_c),
            "Theta_J": jnp.asarray(bar_Y_c),
            "log_D": jnp.asarray(log_D_emp),
        }

 # MoM warmstart : seed Phase-2 factor sites from anchor-cohort
 # residual covariance. Two modes:
 # • Skip Phase 1 + MoM only — empty init_values + mom_warmstart=True;
 # MoM populates {a_flat, a_diag_pos/raw, log_D, w_ckp_raw_std/raw}.
 # • Phase 1 + MoM (no Phase 1.5) — non-empty init_values from Phase 1 +
 # mom_warmstart=True; MoM keys overwrite Phase-1 values for the same
 # sites; mean-trajectory + per-subject sites from Phase 1 are preserved.
 # When mom_warmstart=False (default), this block is a no-op and the legacy
 # init_values path runs unchanged — identical to pre- master.
    if mom_warmstart and phase == "phase2":
        mom_init = _mom_warmstart_init_values(truth, cfg)
        if init_values is None or len(init_values) == 0:
            init_values = mom_init
            print(
                f"  Seeded {sorted(init_values.keys())} from anchor-"
                f"cohort covariance (Cholesky of Σ̂_c − D̂_c)."
            )
        else:
            overlap = sorted(set(init_values.keys()) & set(mom_init.keys()))
            kept = sorted(set(init_values.keys()) - set(mom_init.keys()))
            init_values = {**init_values, **mom_init} # MoM keys win on overlap
            print(
                f"  Phase-1 + MoM merge: MoM overrode {len(overlap)} "
                f"site(s) {overlap}; kept {len(kept)} Phase-1 site(s) "
                f"({kept[:6]}{'...' if len(kept) > 6 else ''})."
            )
        if random_init:
            print(
                " [/mom] Note: random_init=True was set; MoM supersedes it. "
                "Phase 2 will use init_to_value(MoM), not init_to_uniform."
            )
        random_init = False

    if init_values is None or len(init_values) == 0:
        if random_init:
 # Truly random initialization per SVI seed (test for VI mode-collapse:
 # do different inits land in different basins?). Distinct from the
 # default `init_to_median` which is deterministic given the prior.
            init_fn = init_to_uniform(radius=2.0)
        else:
            init_fn = init_to_median()
    else:
 # Make sure every value is a jnp array (init_to_value is strict)
        init_values_j = {
            name: jnp.asarray(np.asarray(val)) for name, val in init_values.items()
        }
        init_fn = init_to_value(values=init_values_j)

    if train_cfg.guide_type == "amortized":
        from .amortized_guide import make_amortized_guide
        guide = make_amortized_guide(cfg, phase=phase)
    elif train_cfg.guide_type == "low_rank":
 # Low-rank Gaussian: captures the dominant (A_c, D_c, beta_vel, alpha_c)
 # correlations mean-field drops. Rank 10 is a reasonable default.
        guide = AutoLowRankMultivariateNormal(
            model, init_loc_fn=init_fn, init_scale=init_scale, rank=10,
        )
    elif train_cfg.guide_type == "lowrank_local":
 # Combined guide: AutoLowRank for GLOBALS (everything except logit_S)
 # + AutoNormal for the per-subject local logit_S.
 #
 # create_plates must be passed to each sub-guide (NOT just to
 # AutoGuideList) because each sub-guide has its own
 # `_prototype_frame_full_sizes` dict that gets queried during its
 # own `_setup_prototype`. Passing it only to AutoGuideList is
 # insufficient.
        def _create_plates_subjects(*args, **kwargs):
            return [numpyro.plate("subjects", cfg.n)]
        guide = AutoGuideList(model, create_plates=_create_plates_subjects)
 # Hide logit_S AND the deterministic sites w_ckp / A_c — NumPyro
 # AutoGuideList raises if deterministic sites appear in sub-guides.
        hide_from_global = ["logit_S", "w_ckp", "A_c", "q_c"]
 # the paper: when paper spatial priors are on, `onset_times`, `beta_vel`,
 # and (cascade-on branch) `onset_eta` become deterministic sites —
 # hide them from the AutoLowRank sub-guide as well.
        if getattr(cfg, "use_paper_spatial_priors", False):
            hide_from_global.extend(["onset_times", "beta_vel"])
            if getattr(cfg, "use_paper_cascade_prior", False):
                hide_from_global.append("onset_eta")
        elif getattr(cfg, "use_paper_cascade_prior", False):
 # the paper: cascade prior makes `onset_times` deterministic; hide it
 # from the AutoLowRank sub-guide so only the raw sample sites are
 # variationalised.
            hide_from_global.append("onset_times")
        guide.append(AutoLowRankMultivariateNormal(
            handlers.block(model, hide=hide_from_global),
            init_loc_fn=init_fn, init_scale=init_scale, rank=10,
        ))
 # For the per-subject logit_S piece, use a custom function guide
 # (mirror of make_amortized_logit_s_guide but without the encoder)
 # so we sidestep AutoNormal's `_prototype_frame_full_sizes` plate
 # lookup (which trips even with `create_plates` set when the
 # prototype trace runs the blocked model).
        from .amortized_guide import make_perSubject_logit_s_guide
        guide.append(make_perSubject_logit_s_guide(cfg, phase=phase))
    elif train_cfg.guide_type == "lowrank_amortized":
 # Principled combination per the paper + the paper:
 # • globals: low-rank multivariate Gaussian (captures cross-parameter
 # correlations that mean-field drops — needed for factor recovery)
 # • per-subject logit_S: amortized encoder MLP (shared weights across
 # subjects, scales to large cohorts, gives subject-specific
 # posterior means + uncertainties)
        from .amortized_guide import make_amortized_logit_s_guide
        def _create_plates_subjects_amort(*args, **kwargs):
            return [numpyro.plate("subjects", cfg.n)]
        guide = AutoGuideList(model, create_plates=_create_plates_subjects_amort)
 # Hide logit_S AND deterministic sites — NumPyro rejects deterministic
 # sites in AutoGuideList sub-guides.
        hide_lowrank_amort = ["logit_S", "w_ckp", "A_c", "q_c"]
        if getattr(cfg, "use_paper_spatial_priors", False):
            hide_lowrank_amort.extend(["onset_times", "beta_vel"])
            if getattr(cfg, "use_paper_cascade_prior", False):
                hide_lowrank_amort.append("onset_eta")
        elif getattr(cfg, "use_paper_cascade_prior", False):
 # the paper: the cascade prior makes `onset_times` a deterministic
 # site computed from (t_ATN_raw, t_tilde_B, onset_eta). Hide it
 # from the AutoLowRank sub-guide (deterministic sites are
 # rejected) so only the new raw sample sites are variationalised.
            hide_lowrank_amort.append("onset_times")
        guide.append(AutoLowRankMultivariateNormal(
            handlers.block(model, hide=hide_lowrank_amort),
            init_loc_fn=init_fn, init_scale=init_scale, rank=10,
        ))
 # Custom amortized guide handles the subjects plate via its own
 # `with numpyro.plate("subjects", n):` context — no create_plates arg
        guide.append(make_amortized_logit_s_guide(cfg, phase=phase))
    else:
        guide = AutoNormal(model, init_loc_fn=init_fn, init_scale=init_scale)

 # Frozen-mean Phase 2: block the mean-trajectory sites from the guide so
 # they never enter the ELBO trace. AutoNormal / AutoLowRank already skip
 # conditioned sites (handled above via handlers.condition on the model),
 # but the custom amortized / composite guides emit explicit
 # `numpyro.sample` calls for the same site names, so we must block them.
    if frozen_mean:
        guide = handlers.block(guide, hide=list(frozen_mean.keys()))
    lr = train_cfg.lr_phase1 if phase == "phase1" else train_cfg.lr_phase2
    optimizer = _make_optimizer(lr, train_cfg.grad_clip)

 # Plain Trace_ELBO (not TraceEnum_ELBO) — the model does manual mixture
 # marginalization over subtype c via logsumexp inside the likelihood, and
 # has no discrete sample sites (delta_region is RelaxedBernoulli).
 # TraceEnum_ELBO added unnecessary enumeration machinery that broke the
 # AutoLowRankMultivariateNormal guide + plate('subjects') combination.
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    inputs = _build_model_inputs(truth, cfg=cfg)
    rng = jax.random.PRNGKey(train_cfg.seed + (0 if phase == "phase1" else 1))

    svi_state = svi.init(rng, **inputs)
 # Warm-start encoder / numpyro.param weights from Phase 1 → Phase 2.
 # We mutate param values in the freshly-initialized optimizer state,
 # then reinitialize the optimizer to pick up the new values (momentum
 # and counters reset, which is fine at phase boundary).
    if (init_params is not None and phase == "phase2"
            and train_cfg.guide_type in ("amortized", "lowrank_amortized",
                                         "lowrank_local")):
        raw_params = dict(svi.optim.get_params(svi_state.optim_state))
        n_transferred = 0
        for key in init_params:
            if key.startswith("auto_"):
                continue
            if key not in raw_params:
                continue
            p1 = init_params[key]
            p2 = raw_params[key]
            if hasattr(p1, 'shape') and hasattr(p2, 'shape'):
                if p1.shape == p2.shape:
                    raw_params[key] = jnp.asarray(np.asarray(p1))
                    n_transferred += 1
            elif isinstance(p1, dict) and isinstance(p2, dict):
                try:
                    p1_flat = jax.tree_util.tree_leaves(p1)
                    p2_flat = jax.tree_util.tree_leaves(p2)
                    if (len(p1_flat) == len(p2_flat) and
                            all(getattr(a, 'shape', None) == getattr(b, 'shape', None)
                                for a, b in zip(p1_flat, p2_flat))):
                        raw_params[key] = p1
                        n_transferred += 1
                except (TypeError, ValueError) as e:
                    print(f"  [warm-start] Skipping '{key}': pytree comparison failed ({type(e).__name__}: {e})")
        if n_transferred > 0:
            new_optim_state = svi.optim.init(raw_params)
            svi_state = svi_state._replace(optim_state=new_optim_state)
            print(f"  Warm-started {n_transferred} param site(s) from Phase 1")
        elif len(init_params) > 0:
            print(f"  [warm-start] WARNING: transferred 0 params from {len(init_params)} candidates")

    steps = train_cfg.phase1_steps if phase == "phase1" else train_cfg.phase2_steps

    @jax.jit
    def update_fn(svi_state, inputs):
        return svi.update(svi_state, **inputs)

 # Gumbel temperature annealing: gumbel_tau is now a model input, so we
 # can update it each step by mutating the inputs dict. The JIT'd
 # update_fn receives inputs as a PyTree argument — new tau values are
 # traced as weakly-typed scalar arrays and don't require recompilation.
    loss_history = []
    params_history = []
    tau_init = float(train_cfg.gumbel_tau_init)
    tau_final = float(train_cfg.gumbel_tau_final)
    anneal_steps = max(int(train_cfg.gumbel_anneal_steps), 1)
    for step in range(steps):
        frac = min(step / anneal_steps, 1.0)
        tau_step = tau_init * (1 - frac) + tau_final * frac
        inputs["gumbel_tau"] = jnp.asarray(tau_step, dtype=jnp.float32)
        svi_state, loss = update_fn(svi_state, inputs)
        loss_history.append(float(loss))
        if step % train_cfg.log_every == 0:
            print(f"[{phase}] step {step:5d} | loss = {float(loss):10.3f} | tau = {tau_step:.3f}")

    final_params = svi.get_params(svi_state)

 # (S4) per-factor normalization of w_ckp is enforced at model.py:1080 via
 # `numpyro.deterministic("w_ckp", w_ckp_raw - mean(axis=-1))`. No post-hoc
 # projection on variational params is needed — all downstream consumers read
 # post["w_ckp"] (normalized).

    raw_params = svi.optim.get_params(svi_state.optim_state)

    return {
        "svi": svi,
        "params": final_params,
        "raw_params": raw_params,
        "loss_history": loss_history,
        "phase": phase,
        "frozen_mean": {k: np.asarray(v) for k, v in frozen_mean.items()},
    }


def auto_loc_to_init_values(
    params: Dict[str, Any],
    exclude_prefixes: Tuple[str, ...] = (),
) -> Dict[str, jnp.ndarray]:
    """Strip `_auto_loc` suffix from AutoNormal params -> init_values dict.

 NOTE: AutoNormal's auto_loc lives in the UNCONSTRAINED space for
 transformed distributions (e.g., sigma_c_stage is stored as the log).
 `init_to_value` expects values in the CONSTRAINED (sample) space. This
 function is ONLY safe for untransformed Normal sites. For the warm-start
 path, use `posterior_to_init_values` below, which reads constrained values
 from a trace.
 """
    out: Dict[str, jnp.ndarray] = {}
    for name, val in params.items():
        if not name.endswith("_auto_loc"):
            continue
        site = name[: -len("_auto_loc")]
        if any(site.startswith(p) for p in exclude_prefixes):
            continue
        out[site] = jnp.asarray(val)
    return out


def posterior_to_init_values(
    post: Dict[str, Any],
    keep_sites: Tuple[str, ...],
) -> Dict[str, jnp.ndarray]:
    """Extract constrained-space values from a posterior trace.

 Takes the dict returned by `extract_posterior_means` and keeps only the
 named sites. These values ARE in the constrained space (because
 extract_posterior_means runs a trace with the auto_loc substituted and
 the distribution's transforms applied). They are safe to pass to
 `init_to_value`.
 """
    out: Dict[str, jnp.ndarray] = {}
    for name in keep_sites:
        if name in post:
            out[name] = jnp.asarray(np.asarray(post[name]))
    return out


def extract_posterior_means(
    cfg: ModelConfig,
    svi,
    params: Dict[str, Any],
    inputs: Dict[str, jnp.ndarray],
    phase: str,
    frozen_mean: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract posterior means for all sample sites + deterministic quantities.

 Uses `guide.median(params)` which returns values in the CONSTRAINED (sample)
 space via the inverse transforms. These are safe to pass to `init_to_value`
 and to use for downstream analysis. Deterministic sites (A_c, w_ckp,
 theta_shells) are recovered by running a trace with the median values
 substituted.

 ``frozen_mean`` is the dict returned by ``run_phase(..., frozen_mean_source=...)["frozen_mean"]`` (/ the paper). When non-empty, those
 values are merged into the median dict so the downstream model retrace
 recomputes derived sites (theta_c_grid, A_c, w_ckp,...) under the frozen
 mean rather than against a fresh prior sample. Without this merge the guide
 has no variational params for the conditioned sites — `guide.median` returns
 nothing for them, the substitute handler doesn't override them, and the
 retrace draws prior samples, so the saved posterior loses the freeze. See
 `tests/test_frozen_mean.py::test_extract_preserves_frozen_mean`.
 """
    from numpyro.handlers import substitute as subs_handler, seed as seed_handler, trace as trace_handler

    model_fn = build_numpyro_model(cfg, phase=phase)
 # AutoGuide instances have.median(); custom function guides don't.
 # AutoGuideList has.median() but it delegates to each sub-guide; if any
 # sub-guide is a plain function (e.g., the amortized logit_S piece of
 # lowrank_amortized),.median() raises AttributeError. Detect this case
 # and fall back to a guide-trace approach.
    medians = None
    try:
        if hasattr(svi.guide, "median"):
            medians = svi.guide.median(params)
    except AttributeError as e:
        print(f"  [WARNING] guide.median() raised AttributeError: {e}")
        print(f"  Falling back to guide trace (sampled values, not medians)")
        medians = None
    if medians is None:
 # Fallback: run the guide under a seeded substitute, then extract
 # distribution MEANS (not samples) for Normal sites. This gives
 # deterministic medians for amortized/function guides that lack
 #.median(). For Normal, mean == median == mu_s.
        with seed_handler(rng_seed=0), subs_handler(data=params):
            guide_tr = trace_handler(svi.guide).get_trace(**inputs)
        medians = {}
        for name, site in guide_tr.items():
            if not hasattr(site.get("value"), "shape"):
                continue
            if site["type"] == "sample":
                medians[name] = _distribution_median(site.get("fn"), site["value"])
            elif site["type"] == "deterministic":
                medians[name] = site["value"]

 # Frozen-mean Phase 2 (): override the median dict for sites that
 # were `condition`-clamped during training. AutoLowRank / AutoNormal skip
 # conditioned sites, so they are absent from `guide.median` output;
 # substituting the frozen values here ensures the retrace below recomputes
 # derived sites (theta_c_grid, A_c,...) under the frozen mean.
    if frozen_mean:
        for name, val in frozen_mean.items():
            medians[name] = jnp.asarray(np.asarray(val))

 # Run a seeded trace with the constrained medians substituted to pick up
 # the deterministic sites (A_c, w_ckp).
    with seed_handler(rng_seed=0), subs_handler(data=medians):
        tr = trace_handler(model_fn).get_trace(**inputs)

    out = {}
    for name, site in tr.items():
        if site["type"] in {"sample", "deterministic"}:
            v = site.get("value")
            if v is not None and hasattr(v, "shape"):
                out[name] = np.asarray(v)

 # Pick up guide-only deterministic sites (e.g. q_c from the amortized
 # subtype head) that are NOT in the model trace.
 # WARNING: q_c is currently UNTRAINED (anchor supervision disabled).
 # Stored with _untrained suffix to prevent accidental use for metrics.
    if medians is not None:
        if "q_c" not in out and "q_c" in medians:
            v = medians["q_c"]
            if hasattr(v, "shape"):
                out["q_c_untrained"] = np.asarray(v)
    return out


# -----------------------------------------------------------------------------
# Phase 1.5 residual factor analysis (CPU numpy implementation)
# -----------------------------------------------------------------------------


def residual_factor_analysis(
    Y: np.ndarray, # (n, R)
    mask: np.ndarray, # (n, R)
    theta_hat: np.ndarray, # (n, R) per-subject mean-trajectory estimate from Phase 1
    c_hat: np.ndarray, # (n,) subtype assignment
    C: int,
    K: int,
    n_bins: int = 5,
    s_hat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run FA per (subtype, stage bin) and average the loadings.

 Returns initial A_c : (C, R, K). Uses `sklearn.decomposition.FactorAnalysis`.
 If there are too few subjects in a bin, skips that bin.
 """
    from sklearn.decomposition import FactorAnalysis

    R = Y.shape[1]
    A_init = np.zeros((C, R, K), dtype=np.float32)
    residuals = (Y - theta_hat) * mask # masked residuals

    for c in range(C):
        per_bin = []
        if s_hat is not None:
            bin_edges = np.quantile(
                s_hat[c_hat == c] if np.sum(c_hat == c) > 0 else np.linspace(0, 1, 2),
                np.linspace(0, 1, n_bins + 1),
            )
        else:
            bin_edges = np.linspace(0, 1, n_bins + 1)
        for b in range(n_bins):
            if s_hat is not None:
                mask_bin = (c_hat == c) & (s_hat >= bin_edges[b]) & (s_hat <= bin_edges[b + 1])
            else:
                mask_bin = c_hat == c
            r_bc = residuals[mask_bin]
            if r_bc.shape[0] < max(2 * K + 1, 5):
                continue
            try:
                fa = FactorAnalysis(n_components=K, max_iter=100)
                fa.fit(r_bc)
                per_bin.append(fa.components_.T) # (R, K)
            except (ValueError, np.linalg.LinAlgError) as e:
                print(f"  [Phase 1.5] FA failed for subtype {c}, bin {b}: {type(e).__name__}: {e}")
                continue
        if per_bin:
            A_init[c] = np.mean(per_bin, axis=0).astype(np.float32)
        else:
            print(f"  [Phase 1.5] WARNING: all bins failed for subtype {c}, using zero A_c[{c}]")
            A_init[c] = np.zeros((R, K), dtype=np.float32)
    return A_init


# -----------------------------------------------------------------------------
# End-to-end pipeline driver
# -----------------------------------------------------------------------------


def _posterior_medians_via_guide(
    cfg: ModelConfig, params: Dict[str, Any], truth, guide_type: str, phase: str,
    svi=None,
) -> Dict[str, np.ndarray]:
    """Extract constrained-space posterior medians from any guide type.

 Uses guide.median() when an SVI object is available (handles AutoNormal,
 AutoLowRank, AutoGuideList, and composite guides correctly). Falls back
 to _auto_loc stripping for backward compatibility when svi is not passed.
 """
    from numpyro.handlers import seed as seed_handler, trace as trace_handler, substitute as subst
    from .model import build_numpyro_model
    inputs = _build_model_inputs(truth, cfg=cfg)
    model_fn = build_numpyro_model(cfg, phase=phase)

    medians = None

 # Preferred path: use guide.median() which handles all guide types
    if svi is not None:
        try:
            if hasattr(svi.guide, "median"):
                medians = svi.guide.median(params)
        except AttributeError as e:
            print(f"  [WARNING] guide.median() raised AttributeError: {e}")
            medians = None
        if medians is None:
 # Fallback for function guides: extract distribution means
            with seed_handler(rng_seed=0), subst(data=params):
                guide_tr = trace_handler(svi.guide).get_trace(**inputs)
            medians = {}
            for name, site in guide_tr.items():
                if not hasattr(site.get("value"), "shape"):
                    continue
                if site["type"] == "sample":
                    medians[name] = _distribution_median(site.get("fn"), site["value"])
                elif site["type"] == "deterministic":
                    medians[name] = site["value"]
    elif guide_type == "amortized":
        from .amortized_guide import make_amortized_guide
        guide = make_amortized_guide(cfg, phase=phase)
        with seed_handler(rng_seed=0), subst(data=params):
            guide_tr = trace_handler(guide).get_trace(**inputs)
        medians = {
            name: site["value"]
            for name, site in guide_tr.items()
            if site["type"] == "sample"
        }
    else:
 # Legacy AutoNormal fallback: strip _auto_loc suffix
        medians = {
            name[: -len("_auto_loc")]: val
            for name, val in params.items() if name.endswith("_auto_loc")
        }

 # Run the model trace with medians substituted to get deterministic sites
    with seed_handler(rng_seed=0), subst(data=medians):
        tr = trace_handler(model_fn).get_trace(**inputs)
    return {name: np.asarray(site["value"])
            for name, site in tr.items()
            if site["type"] in ("sample", "deterministic")}


def _phase1_trajectories_at_subject_stages(
    cfg: ModelConfig, params: Dict[str, Any], truth,
    c_hat: np.ndarray, s_hat: np.ndarray, guide_type: str = "auto_normal",
    svi=None,
) -> np.ndarray:
    """Return theta_{c_hat[i]}(s_hat[i]) for every subject i — Phase 1's
 estimate of each subject's mean trajectory evaluated at the estimated
 subtype and stage. Used by Phase 1.5 residual FA so the A_c init is
 built from residuals of the true mean trajectory, not residuals from
 the terminal state only (which contaminates A_c with mean structure).

 Works with all guide types via _posterior_medians_via_guide. Pass svi=
 for correct low-rank/composite guide posterior extraction.
 """
    from .utils.bspline import build_bspline_eval_matrix

    post = _posterior_medians_via_guide(cfg, params, truth, guide_type, phase="phase1", svi=svi)
    Theta_J = np.asarray(post["Theta_J"])
    beta_vel = np.asarray(post["beta_vel"])
    onset = np.asarray(post["onset_times"])
    sharp = np.asarray(post["sharpness"])
    delta = np.asarray(post["delta_region"])

    C, R, Q, J = cfg.C, cfg.R, cfg.Q, cfg.J
    B_shell = np.asarray(build_bspline_eval_matrix(n_basis=Q, n_eval=J + 1))
    tau_grid = np.linspace(0.0, 1.0, J + 1).astype(np.float32)

    mod_signs = np.ones(R, dtype=np.float32)
    sb = cfg.R_A + cfg.R_T
    se = sb + cfg.R_N
    mod_signs[sb:se] = -1.0
    if cfg.R_blood > 0:
        mod_signs[se] = -1.0

    raw = np.einsum("crq,jq->cjr", beta_vel, B_shell)
 # Audit B1: np.logaddexp(0, x) is numerically stable softplus (matches
 # heldout.py + simulate.py migration).
    softplus_raw = np.logaddexp(0.0, raw)
    vmag = mod_signs[None, None, :] * softplus_raw
    onset_arg = (tau_grid[None, :, None] - onset[:, None, :]) / np.clip(sharp[:, None, :], 1e-3, None)
    onset_g = 1.0 / (1.0 + np.exp(-onset_arg))
    V = delta[:, None, :] * onset_g * vmag

    theta_shells = np.zeros((C, J + 1, R), dtype=np.float32)
    theta_shells[:, -1, :] = Theta_J
    dj = 1.0 / J
    for j in range(J, 0, -1):
        theta_shells[:, j - 1, :] = theta_shells[:, j, :] - dj * V[:, j, :]

    n = len(c_hat)
    theta_hat = np.zeros((n, R), dtype=np.float32)
    for i in range(n):
        c_i = int(c_hat[i])
        s = float(np.clip(s_hat[i], 0.0, 1.0 - 1e-6))
        j_int = min(int(s * J), J - 1)
        u = (s - j_int / J) * J
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        theta_hat[i] = (
            h00 * theta_shells[c_i, j_int] + h10 * dj * V[c_i, j_int]
            + h01 * theta_shells[c_i, j_int + 1] + h11 * dj * V[c_i, j_int + 1]
        )
    return theta_hat


def _subtype_posterior_from_trace(
    cfg: ModelConfig, params: Dict[str, Any], truth, guide_type: str = "auto_normal",
    svi=None,
) -> np.ndarray:
    """Compute q(c_i = c | Y_i, X_i) at Phase 1 posterior means.

 Works with all guide types via _posterior_medians_via_guide. Pass svi=
 for correct low-rank/composite guide posterior extraction.
 Recomputes log_joint and returns (n, C) responsibilities.
 """
    post = _posterior_medians_via_guide(cfg, params, truth, guide_type, phase="phase1", svi=svi)

    import jax.numpy as jnp

    Theta_J = np.asarray(post["Theta_J"]) # (C, R)
    beta_vel = np.asarray(post["beta_vel"]) # (C, R, Q)
    onset = np.asarray(post["onset_times"]) # (C, R)
    sharp = np.asarray(post["sharpness"]) # (C, R)
    delta = np.asarray(post["delta_region"]) # (C, R)
    log_D = np.asarray(post["log_D"]) # (C, R)
    logit_S_raw = np.asarray(post["logit_S"]) # (n,) or (n, C)
    structured_stage = logit_S_raw.ndim == 2
    if structured_stage:
        S_i_c = 1.0 / (1.0 + np.exp(-logit_S_raw)) # (n, C) per-subtype
        logit_S = logit_S_raw.mean(axis=1) # (n,) for downstream
    else:
        S_i_c = None
        logit_S = logit_S_raw
    alpha_c = np.asarray(post["alpha_c"])
    beta_c_sub = np.asarray(post["beta_c_sub"])

    D_c = np.exp(log_D)

 # Reconstruct theta_c_shells via reverse-time integration
    C, R, Q = cfg.C, cfg.R, cfg.Q
    J = cfg.J
    from .utils.bspline import build_bspline_eval_matrix
    B_shell = np.asarray(build_bspline_eval_matrix(n_basis=Q, n_eval=J + 1))
    tau_grid = np.linspace(0.0, 1.0, J + 1).astype(np.float32)

 # Modality signs for velocity transform
    mod_signs = np.ones(R, dtype=np.float32)
    sb = cfg.R_A + cfg.R_T
    se = sb + cfg.R_N
    mod_signs[sb:se] = -1.0
    if cfg.R_blood > 0:
        mod_signs[se] = -1.0

    raw = np.einsum("crq,jq->cjr", beta_vel, B_shell)
 # Audit B1: np.logaddexp(0, x) is numerically stable softplus (matches
 # heldout.py + simulate.py migration).
    softplus_raw = np.logaddexp(0.0, raw)
    vmag = mod_signs[None, None, :] * softplus_raw
    onset_arg = (tau_grid[None, :, None] - onset[:, None, :]) / np.clip(sharp[:, None, :], 1e-3, None)
    onset_g = 1.0 / (1.0 + np.exp(-onset_arg))
    V = delta[:, None, :] * onset_g * vmag # (C, J+1, R)

    theta_shells = np.zeros((C, J + 1, R), dtype=np.float32)
    theta_shells[:, -1, :] = Theta_J
    for j in range(J, 0, -1):
        theta_shells[:, j - 1, :] = theta_shells[:, j, :] - (1.0 / J) * V[:, j, :]

 # Compute theta_c(S_i_c) for each (c, i) via Hermite interpolation.
 # Use per-subtype stages when available (structured_stage=True).
    n = logit_S.shape[0]
    S_i_scalar = 1.0 / (1.0 + np.exp(-logit_S)) # (n,) fallback
    log_lik_matrix = np.zeros((n, C), dtype=np.float32)
    dj = 1.0 / J
    for c in range(C):
        for i in range(n):
            s = float(S_i_c[i, c]) if structured_stage else float(S_i_scalar[i])
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
 # Phase 1 log-lik: diagonal only, Sigma = diag(D_c)
            residual = truth.Y[i] - theta_i_c
            r_obs = residual * truth.mask[i]
            d_safe = np.where(truth.mask[i] > 0.5, D_c[c], 1.0)
            quad = np.sum(r_obs * r_obs / d_safe)
            logdet = np.sum(truth.mask[i] * np.log(d_safe))
            R_obs = np.sum(truth.mask[i])
            log_lik_matrix[i, c] = -0.5 * (R_obs * np.log(2 * np.pi) + logdet + quad)

 # Combine with prior: log_prior_c = log softmax(alpha_c + beta_c @ X_i)
    subtype_logits = alpha_c[None, :] + truth.X @ beta_c_sub.T
    log_prior_c = subtype_logits - np.log(np.sum(np.exp(subtype_logits), axis=-1, keepdims=True))

 # Anchor override
    is_anchor = (truth.anchor_type >= 0) & (truth.anchor_type < C)
    log_prior_adj = log_prior_c.copy()
    large_neg = -1e6
    for i in range(n):
        if is_anchor[i]:
            log_prior_adj[i, :] = large_neg
            log_prior_adj[i, truth.anchor_type[i]] = 0.0

    log_joint = log_lik_matrix + log_prior_adj
    log_joint -= np.max(log_joint, axis=-1, keepdims=True)
    resp = np.exp(log_joint)
    resp /= np.sum(resp, axis=-1, keepdims=True)
    return resp


def run_full_pipeline(
    cfg: ModelConfig,
    train_cfg: TrainConfig,
    truth,
    use_phase15: bool = True,
    frozen_mean_source: Optional[Any] = None,
    mom_warmstart: bool = False,
) -> Dict[str, Any]:
    """Drive Phase 1 -> Phase 1.5 -> Phase 2 with warm-start transfer.

 fix: extract Phase 1 auto_loc values, optionally run Phase 1.5
 residual factor analysis to initialize A_c, then run Phase 2 with
 init_to_value.

 frozen_mean_source : optional path or dict with Phase 1 posterior medians
 for the mean-trajectory sites. When provided, Phase 2 runs in the
 frozen-mean ablation mode (/ the paper) — the mean
 trajectory is held fixed and only the covariance module is trained.
 If `frozen_mean_source == "phase1_in_memory"`, the in-memory Phase 1
 posterior computed just above is used (convenient default for the
 single-process ablation driver).
 """
    import numpy as _np
    print("Phase 1: mean-only SVI")
    phase1_out = run_phase(cfg, train_cfg, truth, phase="phase1")
    print(f"Phase 1 final loss: {phase1_out['loss_history'][-1]:.2f}")

    inputs = _build_model_inputs(truth, cfg=cfg)
    phase1_post = extract_posterior_means(
        cfg, phase1_out["svi"], phase1_out["params"], inputs, phase="phase1"
    )

 # ---- Phase 1.5: residual factor analysis (optional) ---------------------
 # Build init_values from CONSTRAINED (median) values, NOT from raw auto_loc.
 # These are the same sample sites that Phase 2 shares with Phase 1.
    shared_sites = (
        "Theta_0", "mu_c", "alpha_c", "beta_c_sub", "h_c", "f_c_age",
        "sigma_c_stage", "pi_region", "delta_region", "onset_times",
        "sharpness", "beta_vel", "Theta_J", "log_D", "logit_S",
    )
    init_values = posterior_to_init_values(phase1_post, keep_sites=shared_sites)
 # logit_S lives inside the plate; its prior is Normal so constrained ==
 # unconstrained. If the trace path didn't include it, fall back to auto_loc.
    if "logit_S" not in init_values and "logit_S_auto_loc" in phase1_out["params"]:
        init_values["logit_S"] = jnp.asarray(phase1_out["params"]["logit_S_auto_loc"])
 # Amortized / composite guides produce logit_S from an encoder, not from a
 # per-subject param. Passing logit_S through init_to_value conflicts with
 # the encoder path and triggers an int32-in-grad error during optimizer.init.
    if train_cfg.guide_type in ("amortized", "lowrank_amortized"):
        init_values.pop("logit_S", None)

    if use_phase15:
        print("Phase 1.5: residual factor analysis for A_c initialization")
 # Compute c_hat, s_hat from Phase 1 posterior — guide-agnostic.
 # AutoNormal exposes logit_S_auto_loc; the amortized guide exposes
 # logit_S_loc (and produces stages via an encoder, not a per-subject
 # param). For amortized, fall back to phase1_post["logit_S"] which
 # is computed from the guide's sample/median path in
 # extract_posterior_means.
        resp = _subtype_posterior_from_trace(
            cfg, phase1_out["params"], truth, guide_type=train_cfg.guide_type,
            svi=phase1_out["svi"],
        )
        c_hat = _np.argmax(resp, axis=1)
        logit_S_mean = None
        for key in ("logit_S_auto_loc", "logit_S_loc"):
            if key in phase1_out["params"]:
                logit_S_mean = _np.asarray(phase1_out["params"][key])
                break
        if logit_S_mean is None and "logit_S" in phase1_post:
            logit_S_post = _np.asarray(phase1_post["logit_S"])
 # phase1_post stage may already be in [0,1]; if so, invert to logit
            if _np.all((logit_S_post >= 0) & (logit_S_post <= 1)):
                logit_S_mean = _np.log(
                    _np.clip(logit_S_post, 1e-6, 1 - 1e-6)
                    / _np.clip(1 - logit_S_post, 1e-6, 1 - 1e-6)
                )
            else:
                logit_S_mean = logit_S_post
        if logit_S_mean is None:
 # Last resort: use anchor/NC pinning + shared age effect
            logit_S_mean = _np.zeros(cfg.n, dtype=_np.float32)
 # Handle structured_stage: logit_S has shape (n, C); pick the per-subject
 # estimate at each subject's MAP subtype.
        if logit_S_mean.ndim == 2:
            logit_S_1d = logit_S_mean[_np.arange(len(c_hat)), c_hat]
        else:
            logit_S_1d = logit_S_mean
        s_hat = 1.0 / (1.0 + _np.exp(-logit_S_1d))

 # Compute theta_hat per subject as Phase 1's full trajectory
 # theta_{c_hat[i]}(s_hat[i]), NOT the terminal state. Using only
 # Theta_J leaks mean-trajectory structure into the residuals passed
 # to FA and contaminates A_c_init with mean effects. The phased
 # protocol in the inference plan requires zero-mean residuals so
 # A_c captures covariance structure only.
        theta_hat = _phase1_trajectories_at_subject_stages(
            cfg, phase1_out["params"], truth, c_hat, s_hat,
            guide_type=train_cfg.guide_type,
            svi=phase1_out["svi"],
        )
        A_c_init = residual_factor_analysis(
            Y=truth.Y, mask=truth.mask, theta_hat=theta_hat,
            c_hat=c_hat, C=cfg.C, K=cfg.K, n_bins=3, s_hat=s_hat,
        )
 # Phase 2 parameterization: a_flat (C, R, K) unconstrained real. The
 # diagonal site depends on HalfNormal-direct flag. When flag on, the
 # model samples a_diag_pos ~ HalfNormal directly (fix);
 # when off, legacy a_diag_raw passes through softplus → ≈0.69. Pick
 # the correct site name so init_to_value actually lands — silent
 # skip of missing sites would leave the diag un-warm-started.
        init_values["a_flat"] = jnp.asarray(A_c_init.astype(_np.float32))
        if getattr(cfg, "use_paper_halfnormal_diag", False):
            init_values["a_diag_pos"] = 0.1 * jnp.ones((cfg.C, cfg.K))
        else:
            init_values["a_diag_raw"] = jnp.zeros((cfg.C, cfg.K))
        if getattr(cfg, "use_ard_on_A", False):
 # Phase 1a ARD: init alpha_ard at Gamma prior mean (a/b) so
 # prior std on A_c columns starts at 1/sqrt(a/b).
            alpha_init = float(cfg.ard_a_shape) / float(cfg.ard_b_rate)
            init_values["alpha_ard"] = alpha_init * jnp.ones((cfg.C, cfg.K))
        else:
 # delta_shrink: init to identity so psi_{c,k} = 1 (no shrinkage)
            init_values["delta_shrink"] = jnp.ones((cfg.C, cfg.K))
        init_values["tau_k"] = jnp.ones(cfg.K)
 # w ~ 0 so nu_{c,k}(s) ~ 1 initially — the paper recommendation.
 # Under Step 3 (paper K_B smoothness prior) the sample site is
 # `w_ckp_raw_std` (pre-Cholesky iid-standard-Normal); the prior version uses
 # `w_ckp_raw` directly. Init the correct site so the guide actually
 # starts at ν ≈ 1 — prior to this fix (Blocker 2,), only
 # `w_ckp_raw` was initialised, which is IGNORED when the paper-
 # smoothness flag is on, leaving the guide at the default noisy init.
        if getattr(cfg, "use_paper_w_ckp_smoothness", False):
            init_values["w_ckp_raw_std"] = jnp.zeros((cfg.C, cfg.K, cfg.P))
        else:
            init_values["w_ckp_raw"] = jnp.zeros((cfg.C, cfg.K, cfg.P))
        print(f"Phase 1.5 done. A_c_init frobenius: {_np.linalg.norm(A_c_init):.2f}")
    else:
 # No Phase 1.5: just provide reasonable initial values for the Phase 2-only sites
        init_values["a_flat"] = jnp.zeros((cfg.C, cfg.R, cfg.K))
        if getattr(cfg, "use_paper_halfnormal_diag", False):
            init_values["a_diag_pos"] = 0.1 * jnp.ones((cfg.C, cfg.K))
        else:
            init_values["a_diag_raw"] = jnp.zeros((cfg.C, cfg.K))
        if getattr(cfg, "use_ard_on_A", False):
            alpha_init = float(cfg.ard_a_shape) / float(cfg.ard_b_rate)
            init_values["alpha_ard"] = alpha_init * jnp.ones((cfg.C, cfg.K))
        else:
            init_values["delta_shrink"] = jnp.ones((cfg.C, cfg.K))
        init_values["tau_k"] = jnp.ones(cfg.K)
 # Same Blocker-2 fix: thread paper-smoothness flag to pick the right site.
        if getattr(cfg, "use_paper_w_ckp_smoothness", False):
            init_values["w_ckp_raw_std"] = jnp.zeros((cfg.C, cfg.K, cfg.P))
        else:
            init_values["w_ckp_raw"] = jnp.zeros((cfg.C, cfg.K, cfg.P))

    print("Phase 2: full factor covariance SVI with warm-start from Phase 1")
 # For amortized / composite guides, forward Phase 1's learned numpyro.param
 # state (encoder weights, per-subject locs, etc) so Phase 2 doesn't
 # reinitialize the encoder from scratch. AutoNormal / low_rank standalone
 # also benefit because their `auto_loc` / `auto_scale_tril` get warm-start.
 # Frozen-mean variant (): route Phase 1 posterior as the frozen
 # source when the caller passed `"phase1_in_memory"` or left it unset and
 # only mean-fix is desired. Explicit paths / dicts override.
    phase2_frozen = None
    if frozen_mean_source is not None:
        if isinstance(frozen_mean_source, str) and frozen_mean_source == "phase1_in_memory":
            phase2_frozen = {k: phase1_post[k] for k in FROZEN_MEAN_SITES if k in phase1_post}
        else:
            phase2_frozen = frozen_mean_source
    phase2_out = run_phase(
        cfg, train_cfg, truth, phase="phase2",
        init_values=init_values,
        init_params=phase1_out.get("raw_params"),
        frozen_mean_source=phase2_frozen,
        mom_warmstart=mom_warmstart,
    )
    print(f"Phase 2 final loss: {phase2_out['loss_history'][-1]:.2f}")
    phase2_post = extract_posterior_means(
        cfg, phase2_out["svi"], phase2_out["params"], inputs, phase="phase2",
        frozen_mean=phase2_out.get("frozen_mean") or None,
    )

    return {
        "phase1": phase1_out,
        "phase1_posterior": phase1_post,
        "phase2": phase2_out,
        "phase2_posterior": phase2_post,
    }
