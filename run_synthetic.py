#!/usr/bin/env python
"""Per-seed driver for the synthetic ELPAC reproduction.

Reproduces the headline synthetic experiment in the paper by fitting
ELPAC end-to-end on one synthetic cohort per seed:

    python run_synthetic.py --seeds 0
"""
import argparse, json, os, sys, time
from dataclasses import replace as _dc_replace
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax.numpy as jnp
from pathlib import Path
from elpac.simulate import simulate_synthetic, SyntheticShape
from elpac.model import ModelConfig
from elpac.inference import (TrainConfig, run_full_pipeline, run_phase,
                           _build_model_inputs, extract_posterior_means,
                           posterior_to_init_values)
from elpac.metrics import _compute_recovery_metrics

SHAPE = SyntheticShape(n=1000, C=3, K=3, J=8, P=5, Q=5,
                       R_A=5, R_T=5, R_N=6, R_blood=4, p_x=4, p_w=3)

GATES = {
    "stage_spearman_rho": ("higher", 0.90),
    "subtype_ari": ("higher", 0.85),
    "subtype_accuracy_general": ("higher", 0.85),
    "mean_trajectory_sup_norm": ("lower", 0.15),
    "mean_trajectory_rmse": ("lower", 0.15),
    "loading_procrustes_mean": ("lower", 0.15),
    "activation_sup_norm_relative_mean": ("lower", 0.20),
}


def count_gates(m):
    return sum(1 for k, (d, t) in GATES.items() if m.get(k) is not None and
               ((d == "higher" and m[k] > t) or (d == "lower" and m[k] < t)))


# ---------------------------------------------------------------------------
# Helper: filter guide variational parameters from svi.get_params() output.
# ---------------------------------------------------------------------------
def _collect_guide_params(params):
    """Return a dict of numeric guide variational parameters, filtered by key.

 Filters for keys whose names contain any of the canonical variational
 parameter suffixes used by the three guide families:

 AutoLowRankMultivariateNormal : ``*_cov_factor``, ``*_scale``
 AutoNormal : ``*_auto_loc``, ``*_auto_scale``
 AutoGuideList : union of the above across sub-guides

 Non-numeric values (ints, nested dicts, MLP pytrees, etc.) are silently
 skipped via a try/except around ``np.asarray``. Encoder MLP kernel/bias
 tensors do not match these substrings so they are excluded structurally
 as well.

 Keys are prefixed with ``guide_params/`` when written to the NPZ sidecar
 so they live in a distinct namespace from model-site arrays (A_c, D_c,
 nu_ci, w_ckp, …). Downstream eigenvalue analyses should load all keys
 whose names start with ``guide_params/``.
 """
    _TOKENS = ("_cov_factor", "_scale", "_auto_loc", "_auto_scale")
    out = {}
    if not isinstance(params, dict):
        return out
    for name, val in params.items():
        if not any(tok in name for tok in _TOKENS):
            continue
        try:
            arr = np.asarray(val)
            if arr.dtype.kind not in "biufc":
                continue
            out[f"guide_params/{name}"] = arr
        except Exception:
            pass
    return out



def gen_truth(include_cognition=True, cognition_model="linear",
              use_rich_nu=False, nu_amplitude_scale=2.0,
              backward_generation=False, cognition_missing_rate=0.0,
              use_spatial_sim=False, spatial_length_scale=0.3,
              external_kernels=None,
              w_ckp_paper_prior=False,
              w_ckp_sigma_tau=0.1, w_ckp_eps_B=1e-3,
              w_ckp_kb_smooth_strength=0.0,
              w_ckp_kb_smooth_preserve_amplitude=True,
              n=None, anchor_frac_override=None, nc_frac_override=None,
              simulator_seed: int = 42):
    """Generate one synthetic cohort using the paper's simulator.

    Optional overrides:
      ``n`` — replace the module-level SHAPE.n (=1000) with a custom size.
      ``anchor_frac_override`` — anchor fraction per subtype (default 0.10
        gives 100 anchors per subtype at n=1000).
      ``nc_frac_override`` — healthy-reference fraction (default 0.10).
      ``external_kernels`` — list of 4 per-modality covariance kernels;
        replaces the simulator's default RBF / Laplacian / CAR / Identity
        kernels when ``use_spatial_sim=True``.
    """
    # --- sample-size override ---
    if n is not None:
        shape = _dc_replace(SHAPE, n=n)
    else:
        shape = SHAPE
    if anchor_frac_override is not None:
        _raw_anchor = int(anchor_frac_override * shape.n / shape.C)
        if _raw_anchor < 1:
            raise ValueError(
                f"[gen_truth] n_anchor_per_subtype would be 0 "
                f"(n={shape.n}, anchor_frac={anchor_frac_override}, C={shape.C}). "
                f"Increase n or anchor_frac, or reduce C.")
        n_anchor_per_subtype = _raw_anchor
    else:
        n_anchor_per_subtype = 100
    if nc_frac_override is not None:
        n_nc = max(10, int(nc_frac_override * shape.n))
    else:
        n_nc = 100
    return simulate_synthetic(shape, seed=simulator_seed, n_anchor_per_subtype=n_anchor_per_subtype, n_nc=n_nc,
                               missing_rate=0.1, phase_transition_s_star=0.5,
                               include_cognition=include_cognition,
                               cognition_noise=0.3,
                               cognition_model=cognition_model,
                               use_rich_nu=use_rich_nu,
                               nu_amplitude_scale=nu_amplitude_scale,
                               backward_generation=backward_generation,
                               cognition_missing_rate=cognition_missing_rate,
                               use_spatial_sim=use_spatial_sim,
                               spatial_length_scale=spatial_length_scale,
                               external_kernels=external_kernels,
                               w_ckp_paper_prior=w_ckp_paper_prior,
                               w_ckp_sigma_tau=w_ckp_sigma_tau,
                               w_ckp_eps_B=w_ckp_eps_B,
                               w_ckp_kb_smooth_strength=w_ckp_kb_smooth_strength,
                               w_ckp_kb_smooth_preserve_amplitude=w_ckp_kb_smooth_preserve_amplitude)



def run_curriculum(truth, m1_seed, m2_seed, use_cog=True, struct_T=True,
                   tau_k=0.1, w_anchor=5.0, guide_type="lowrank_amortized",
                   m2_steps=10000, use_phase15=False,
                   truth_train=None, truth_test=None,
                   joint_cog_marginalization=False,
                   train_q_c=False,
                   enhanced_encoder=False,
                   wider_encoder=False,
                   anchor_ce_supervision=False,
                   forward_time=False,
                   export_posterior=False,
                   save_loss_history=False,
                   shape=None,
                   log_d_prior_mean=-1.0,
                   delta_shrink_shape=1.5,
                   psi_shrink_scale=1.0,
                   use_ard_on_A=False,
                   ard_a_shape=1.0,
                   ard_b_rate=1.0,
                   # per the paper overrides. None -> ModelConfig default.
                   eps_pinning=None,
                   d_min=None,
                   omega_0_soft_std=None,
                   delta_shrink_shape_k1=None,
                   theta0_prior_std_override=None,
                   mu_c_prior_std_override=None,
                   # Stage 0a v2 (Commit C,): explicit sample-var
                   # priors mode. When True, force ModelConfig to use the
                   # the paper data-driven diagonals (theta0_prior_std_override
                   # = mu_c_prior_std_override = None), regardless of the
                   # per-axis CLI overrides above. This is the only CLI path
                   # that can reach the sample-var mode because passing CLI
                   # None is indistinguishable from "CLI not set" (which
                   # falls through to the ModelConfig scalar-0.5 default).
                   force_sample_var_priors=False,
                   # per the paper flag (σ_c: Half-Cauchy prior +
                   # drop [0.5, 1.5] clamp). False preserves behavior.
                   use_paper_sigma_c=False,
                   # per the paper flag (K_B second-difference
                   # smoothness prior on w_{c,k,p}, the paper + the paper).
                   # False preserves the prior behavior.
                   use_paper_w_ckp_smoothness=False,
                   eps_B=None,
                   # per the paper flag (soft cascade A<T<N on onset
                   # centers, the paper + the paper). False preserves 
                   # the prior behavior.
                   use_paper_cascade_prior=False,
                   # per the paper flag (per-modality MVN spatial
                   # priors on onset_times + beta_velocity, the paper Steps 7, 11, 13
                   # + the paper). False preserves the prior behavior.
                   use_paper_spatial_priors=False,
                   spatial_length_scale=0.3,
                   # Step 5c (): external kernel NPZ path. When set,
                   # the ModelConfig will load Σ^(m) from this file instead of
                   # building RBF/Laplacian/CAR/Identity kernels. Coupled with
                   # the simulator's `external_kernels` kwarg upstream.
                   external_kernels_path=None,
                   # per the paper (monotone I-spline chi_c per
                   # the paper). "polynomial" default preserves
                   # the prior behavior. "ispline" MUST be paired
                   # with simulator cognition_model="ispline".
                   chi_c_basis="polynomial",
                   chi_c_ispline_n_basis=3,
                   # fix: HalfNormal-direct diagonal on A_c
                   #. False preserves an earlier-version softplus
                   # parameterization (which pins A_kk at ≈0.693).
                   use_paper_halfnormal_diag=True,
                   # / : freeze Phase-1 posterior mean during
                   # Phase-2. "phase1_in_memory" uses the just-computed m1
                   # posterior; a file path loads from NPZ/pkl. None = off.
                   frozen_mean_source=None,
                   # -style random-init test: skip Phase 1 + Phase 1.5
                   # warm-start entirely; run Phase 2 directly with
                   # init_to_uniform(radius=2.0) per the SVI seed. Tests
                   # whether VI lands in the same basin from random inits
                   # (deterministic mode-collapse) or different basins
                   # (multimodal landscape). Incompatible with
                   # frozen_mean_source (no m1 posterior to freeze from).
                   skip_phase1_warmstart=False,
                   # opt-in: seed Phase 2 factor sites from method-of-
                   # moments factor analysis on anchor-cohort residual
                   # covariance. False (default) = legacy path, bit-for-
                   # bit identical to pre- master.
                   mom_warmstart=False):
    """Run the M1 warm-start curriculum + M2 fine-tune. Returns
 ``(metrics_dict, loss, wall_s)``.
 """
    train_truth = truth_train if truth_train is not None else truth
    n_train = train_truth.Y.shape[0]
    sh = shape if shape is not None else SHAPE

    # Build the the paper override kwargs (only pass fields that are explicitly
    # set; otherwise ModelConfig's paper-the paper defaults apply).
    step1_kwargs = {}
    if eps_pinning is not None:
        step1_kwargs["eps_pinning"] = float(eps_pinning)
    if d_min is not None:
        step1_kwargs["d_min"] = float(d_min)
    if omega_0_soft_std is not None:
        step1_kwargs["Omega_0_soft_std"] = float(omega_0_soft_std)
    if delta_shrink_shape_k1 is not None:
        step1_kwargs["delta_shrink_shape_k1"] = float(delta_shrink_shape_k1)
    # force_sample_var_priors wins over per-axis floats: passing it True
    # always lands theta0/mu_c on the paper sample-variance mode (None).
    if force_sample_var_priors:
        step1_kwargs["theta0_prior_std_override"] = None
        step1_kwargs["mu_c_prior_std_override"] = None
    else:
        if theta0_prior_std_override is not None:
            step1_kwargs["theta0_prior_std_override"] = float(theta0_prior_std_override)
        if mu_c_prior_std_override is not None:
            step1_kwargs["mu_c_prior_std_override"] = float(mu_c_prior_std_override)
    # per the paper flag. Only attach when True so the default
    # False branch preserves ModelConfig construction 
    # (the kwarg itself is unchanged on absent keys, but being explicit about
    # "only set when non-default" matches the pattern used by step1_kwargs).
    if use_paper_sigma_c:
        step1_kwargs["use_paper_sigma_c"] = True
    # per the paper flag. Only attach when True to keep the default
    # False branch identical to ModelConfig construction.
    if use_paper_w_ckp_smoothness:
        step1_kwargs["use_paper_w_ckp_smoothness"] = True
    if eps_B is not None:
        step1_kwargs["eps_B"] = float(eps_B)
    # per the paper flag. Only attach when True so the default False
    # branch preserves ModelConfig construction.
    if use_paper_cascade_prior:
        step1_kwargs["use_paper_cascade_prior"] = True
    # per the paper flag. Only attach when True so the default False
    # branch preserves ModelConfig construction.
    if use_paper_spatial_priors:
        step1_kwargs["use_paper_spatial_priors"] = True
        step1_kwargs["spatial_length_scale"] = float(spatial_length_scale)
    # Step 5c: thread the NPZ path into ModelConfig so build_numpyro_model
    # loads the external kernels at construction time. Attached only when
    # non-None so the default the prior path stays identical.
    if external_kernels_path is not None:
        step1_kwargs["external_kernels_path"] = str(external_kernels_path)
    # per the paper. Only attach when non-default so 
    # preserved when chi_c_basis stays "polynomial".
    if chi_c_basis != "polynomial":
        step1_kwargs["chi_c_basis"] = str(chi_c_basis)
        step1_kwargs["chi_c_ispline_n_basis"] = int(chi_c_ispline_n_basis)
    # the paper HalfNormal-direct diagonal fix. Pass-through unconditional
    # after the default flip: if caller requests False, we MUST
    # force False into ModelConfig (class default is now True, so a missing
    # kwarg would silently fall through to HN).
    step1_kwargs["use_paper_halfnormal_diag"] = bool(use_paper_halfnormal_diag)

    cfg0 = ModelConfig(n=n_train, C=sh.C, K=sh.K, J=sh.J, P=sh.P, Q=sh.Q,
                       R_A=sh.R_A, R_T=sh.R_T, R_N=sh.R_N, R_blood=sh.R_blood,
                       p_x=sh.p_x, p_w=sh.p_w, w_anchor=w_anchor, gumbel_tau=0.5,
                       use_cognition=use_cog, structured_stage=False, tau_k_scale=tau_k,
                       log_d_prior_mean=log_d_prior_mean,
                       delta_shrink_shape=delta_shrink_shape,
                       joint_cog_marginalization=joint_cog_marginalization,
                       train_q_c=train_q_c, enhanced_encoder=enhanced_encoder,
                       wider_encoder=wider_encoder, anchor_ce_supervision=anchor_ce_supervision,
                       forward_time=forward_time,
                       psi_shrink_scale=psi_shrink_scale,
                       use_ard_on_A=use_ard_on_A,
                       ard_a_shape=ard_a_shape,
                       ard_b_rate=ard_b_rate,
                       **step1_kwargs)
    tc0 = TrainConfig(phase1_steps=8000, phase2_steps=10000, seed=m1_seed,
                      guide_type='low_rank', log_every=100000)
    t0 = time.time()
    if skip_phase1_warmstart:
        # -style random-init test: bypass Phase 1 + Phase 1.5 entirely.
        # Phase 2 will run with init_to_uniform per the m2_seed (different
        # initial point per seed). m1 is set to None and m1_post to {} so
        # downstream warm-start logic falls through to empty `iv` (which
        # triggers init_to_uniform in run_phase via random_init=True).
        if frozen_mean_source is not None:
            raise ValueError(
                "skip_phase1_warmstart is incompatible with frozen_mean_source: "
                "no m1 posterior available to freeze from."
            )
        m1 = None
        m1_post = {}
    else:
        m1 = run_full_pipeline(cfg0, tc0, train_truth, use_phase15=use_phase15)
        m1_post = m1['phase2_posterior']

    # a_diag_raw (softplus branch) and a_diag_pos (HalfNormal branch) are
    # mutually exclusive in the model trace; posterior_to_init_values
    # silently skips missing sites, so listing BOTH is the tolerant option.
    shared = ("Theta_0", "mu_c", "alpha_c", "beta_c_sub", "h_c", "f_c_age",
              "sigma_c_stage", "pi_region", "delta_region", "onset_times",
              "sharpness", "beta_vel", "Theta_J", "log_D",
              "a_flat", "a_diag_raw", "a_diag_pos", "delta_shrink", "tau_k",
              "alpha_ard",
              # the paper cascade-prior sites (present only when flag is on;
              # posterior_to_init_values silently skips missing keys).
              "t_ATN_raw", "t_tilde_B", "onset_eta")
    if skip_phase1_warmstart:
        # -style: no warm-start; let run_phase fall through to
        # init_to_uniform per the m2_seed.
        iv = {}
    else:
        iv = posterior_to_init_values(m1_post, keep_sites=shared)
        # Blocker 2 (): under Step 3 paper-smoothness, the Phase-2 sample
        # site is `w_ckp_raw_std` (not `w_ckp_raw`). Init the correct site so the
        # M2 warm-start actually starts at ν ≈ 1 — prior code only zeroed the
        # legacy site and silently left `w_ckp_raw_std` at its noisy guide init.
        if use_paper_w_ckp_smoothness:
            iv["w_ckp_raw_std"] = jnp.zeros((sh.C, sh.K, sh.P))
        else:
            iv["w_ckp_raw"] = jnp.zeros((sh.C, sh.K, sh.P))
        # the paper: when the cascade prior is on, `onset_times` is a DETERMINISTIC
        # pass-through of (t_ATN_raw, t_tilde_B, onset_eta). Drop it from iv so
        # init_to_value doesn't try to push a value into a non-sample site.
        if use_paper_cascade_prior:
            iv.pop("onset_times", None)
        if guide_type in ("lowrank_amortized", "lowrank_local"):
            iv.pop("logit_S", None)
        elif "logit_S" in iv and iv["logit_S"].ndim == 1 and struct_T:
            iv["logit_S"] = jnp.tile(iv["logit_S"][:, None], (1, sh.C))

    cfg_t = ModelConfig(n=n_train, C=sh.C, K=sh.K, J=sh.J, P=sh.P, Q=sh.Q,
                        R_A=sh.R_A, R_T=sh.R_T, R_N=sh.R_N, R_blood=sh.R_blood,
                        p_x=sh.p_x, p_w=sh.p_w, w_anchor=w_anchor, gumbel_tau=0.5,
                        use_cognition=use_cog, structured_stage=struct_T, tau_k_scale=tau_k,
                        log_d_prior_mean=log_d_prior_mean,
                        delta_shrink_shape=delta_shrink_shape,
                        joint_cog_marginalization=joint_cog_marginalization,
                        train_q_c=train_q_c, enhanced_encoder=enhanced_encoder,
                       wider_encoder=wider_encoder, anchor_ce_supervision=anchor_ce_supervision,
                       forward_time=forward_time,
                       psi_shrink_scale=psi_shrink_scale,
                       use_ard_on_A=use_ard_on_A,
                       ard_a_shape=ard_a_shape,
                       ard_b_rate=ard_b_rate,
                       **step1_kwargs)
    # the paper: pass `sh` so psi_basis_cache aligns with the inference-side
    # P/J rather than truth.shape.P/J. Without this, --model-p crashes with
    # a w_ckp/psi_S shape mismatch and --model-j silently clips out-of-bounds
    # psi indices when cfg.J > truth.shape.J.
    inputs = _build_model_inputs(train_truth, model_shape=sh)
    tc = TrainConfig(phase1_steps=0, phase2_steps=m2_steps, seed=m2_seed,
                     guide_type=guide_type, log_every=100000)
    # : resolve in-memory frozen-mean source strings to the actual m1
    # posterior dict. Two distinct sources are supported (the paper has both
    # readings; see ABLATION_FULL_CONFIGS[5] / [6]):
    # "m1_phase1" -> M1's Phase-1 mean-only posterior
    # "m1_phase2" -> M1's fully-converged (Phase-2) posterior
    # "phase1_in_memory"-> legacy alias preserved for back-compat → m1_phase2
    _frozen_mean_arg = frozen_mean_source
    if frozen_mean_source == "m1_phase1":
        _frozen_mean_arg = m1.get("phase1_posterior", {})
    elif frozen_mean_source in ("m1_phase2", "phase1_in_memory"):
        _frozen_mean_arg = m1.get("phase2_posterior", {})
    out = run_phase(cfg_t, tc, train_truth, phase="phase2", init_values=iv,
                    frozen_mean_source=_frozen_mean_arg,
                    random_init=skip_phase1_warmstart,
                    mom_warmstart=mom_warmstart)
    # : thread frozen_mean values into extract_posterior_means so the
    # saved posterior preserves the conditioned mean-trajectory sites instead
    # of falling back to prior samples (AutoLowRank skips conditioned sites,
    # leaving them absent from `guide.median()` output).
    post = extract_posterior_means(cfg_t, out["svi"], out["params"], inputs,
                                   phase="phase2",
                                   frozen_mean=out.get("frozen_mean") or None)
    # Recovery metrics on training set (uses truth labels for comparison).
    # : thread forward_time so the theta-shell reconstruction used for
    # subtype permutation matches the model's integration direction.
    m = _compute_recovery_metrics(train_truth, post, out["params"], phase="phase2",
                                  forward_time=cfg_t.forward_time)
    wall = time.time() - t0
    loss = float(out['loss_history'][-1])

    if save_loss_history:
        m["_loss_histories"] = {
            "m1_phase1":  [float(x) for x in m1["phase1"]["loss_history"]],
            "m1_phase2":  [float(x) for x in m1["phase2"]["loss_history"]],
            "refine_phase2": [float(x) for x in out["loss_history"]],
        }

    # Tranche E posterior sidecar: compact per-subject + global arrays for
    # downstream global staging, Wald phase-transition, liquid-association,
    # and Fisher-weight scripts. Stored under a reserved key so the caller
    # can pop it out before JSON serialization.
    if export_posterior:
        try:
            from elpac.metrics import extract_posterior_full
            m["_posterior_full"] = extract_posterior_full(
                train_truth, post, out["params"], phase="phase2",
                model_K=cfg_t.K, forward_time=cfg_t.forward_time)
            # Augment with guide variational parameters for posterior-covariance
            # eigenvalue analysis. Keys are prefixed ``guide_params/`` to avoid
            # collisions with model-site arrays.
            m["_posterior_full"].update(_collect_guide_params(out["params"]))
        except (KeyError, ValueError, AttributeError, TypeError,
                np.linalg.LinAlgError) as e:
            import traceback
            print(f"  [ERROR] extract_posterior_full failed: {type(e).__name__}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            m["_posterior_full_error"] = f"{type(e).__name__}: {e}"

    # Held-out NLL on the test partition (when truth_test is provided).
    if truth_test is not None:
        try:
            from elpac.heldout import compute_marginal_ll_from_posterior
            cfg_test = ModelConfig(
                n=truth_test.Y.shape[0], C=sh.C, K=sh.K, J=sh.J, P=sh.P, Q=sh.Q,
                R_A=sh.R_A, R_T=sh.R_T, R_N=sh.R_N, R_blood=sh.R_blood,
                p_x=sh.p_x, p_w=sh.p_w,
                forward_time=forward_time,
                **step1_kwargs,
            )
            heldout_ll = compute_marginal_ll_from_posterior(
                cfg_test, post, out["params"], truth_test, phase="phase2")
            m["heldout_ll"] = float(heldout_ll)
            m["heldout_nll_per_cell"] = float(-heldout_ll)
            train_ll = m.get("mean_subject_marginal_ll")
            if train_ll is not None:
                m["generalization_gap"] = float(train_ll - heldout_ll)
        except (KeyError, ValueError, np.linalg.LinAlgError) as e:
            print(f"  [ERROR] held-out NLL failed: {type(e).__name__}: {e}", flush=True)
            m["heldout_ll_error"] = f"{type(e).__name__}: {e}"

    return m, loss, wall


def sweep_main(seed, truth_full):
    """Fit ELPAC end-to-end on a stratified 80/20 train-test split.

 Mirrors the paper's synthetic protocol: train on 80% of subjects,
 evaluate recovery metrics on the train subset, and report held-out
 NLL/cell on the 20% test subset.
 """
    from elpac.heldout import train_test_split
    truth_train, truth_test, _, _ = train_test_split(truth_full, test_frac=0.2, seed=seed)
    m, loss, wall = run_curriculum(
        truth_full, m1_seed=seed, m2_seed=seed + 1000,
        use_cog=True, struct_T=True,
        truth_train=truth_train, truth_test=truth_test,
    )
    g = count_gates(m)
    return {"seed": seed, "gates": g, "loss": loss, "wall_s": wall, "metrics": m}



if __name__ == "__main__":
    import argparse
    import json
    import time
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Per-seed reproduction of the synthetic ELPAC experiment."
    )
    parser.add_argument("--seeds", required=True,
                        help="Comma-separated seed list, e.g. 0,1,2,3")
    parser.add_argument("--simulator-seed", type=int, default=42,
                        help="Seed for synthetic-cohort generation.")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: artifacts).")
    parser.add_argument("--include-cognition", action="store_true", default=True,
                        help="Include the cognition submodel in fitting.")
    parser.add_argument("--no-include-cognition", dest="include_cognition",
                        action="store_false")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    outdir = Path(args.outdir) if args.outdir else Path("artifacts")
    outdir.mkdir(parents=True, exist_ok=True)

    # Generate one shared synthetic truth (paper config: C=3, K=3, J=8, R=20,
    # N=1000). Anchor and stage priors use the values reported in the paper.
    truth_full = gen_truth(
        include_cognition=args.include_cognition,
        cognition_model="ispline",
        simulator_seed=args.simulator_seed,
    )

    all_results = []
    for seed in seeds:
        print(f"\n=== seed={seed} ===", flush=True)
        t0 = time.time()
        result = sweep_main(seed, truth_full)
        result["wall_s"] = time.time() - t0
        all_results.append(result)

    out_path = outdir / f"seeds_{seeds[0]}_{seeds[-1]}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o)
                  if hasattr(o, "__float__") else str(o))
    print(f"\nSaved {len(all_results)} result(s) to {out_path}")

