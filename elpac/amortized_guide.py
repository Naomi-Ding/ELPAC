"""Amortized recognition-network guide.

Provides a callable guide that replaces the per-subject `logit_S` auto_loc
with a flax MLP forward pass. Globals still use mean-field Gaussian via
numpyro.param. This module is self-contained and independent of AutoNormal.

The guide is switchable: `inference.run_phase` takes a `guide_type` argument
that selects between:
 - "auto_normal" (AutoNormal for everything; useful for debugging)
 - "amortized" (this module; enables single-forward-pass inference
 on new subjects)

The paper's guide (and the default in run_synthetic.py) is
"lowrank_amortized": a low-rank multivariate Gaussian over globals combined
with this recognition network for per-subject stage and subtype.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import flax.linen as nn

from .model import ModelConfig


class RecognitionMLP(nn.Module):
    """MLP that maps (Y_masked, mask, X) -> per-subject per-subtype (mu_s, log_sigma_s, logit_c).

 Returns:
 mu_s: (n, C) — stage location per subtype
 log_sigma_s: (n, C) — stage log-scale per subtype
 logit_c: (n, C) — unnormalized subtype logits for q(c_i | Y_i, X_i)

 When structured_stage=False, the caller should select the [:, 0, :] column
 for stage parameters. logit_c is always (n, C).
 """
    C: int = 2
    hidden: tuple = (128, 128)
    dropout: float = 0.1

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        x = inputs
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
            if train:
                x = nn.Dropout(self.dropout, deterministic=not train)(x)
        # Stage head: (2*C,) per subject -> reshape to (n, C, 2)
        out = nn.Dense(2 * self.C)(x)
        out = out.reshape(*out.shape[:-1], self.C, 2)
        mu_s = out[..., 0]
        log_sigma_s = out[..., 1]
        # Subtype head: separate Dense(C) producing (n, C) logits
        logit_c = nn.Dense(self.C, name="subtype_head")(x)
        return mu_s, log_sigma_s, logit_c


def make_amortized_guide(cfg: ModelConfig, phase: str = "phase2"):
    """Build a callable guide that amortizes logit_S via a flax MLP.

 Globals use mean-field Gaussian on unconstrained parameter spaces via
 numpyro.param, mirroring AutoNormal's behavior for the subset of sites
 the current model uses (all Normal or transform-of-Normal).
 """
    from numpyro.contrib.module import flax_module

    def guide(Y, mask, X, W, ages_norm, anchor_type, psi_basis_cache, gumbel_tau=None, C_cog=None, C_cog_mask=None, **kwargs):
        # gumbel_tau is consumed by the model only; guide accepts it to
        # keep its signature compatible with the input dict.
        n, R_obs = Y.shape
        C, R = cfg.C, cfg.R
        K, P, Q, J = cfg.K, cfg.P, cfg.Q, cfg.J
        p_x = cfg.p_x
        p_w = cfg.p_w

        # ------------------------------------------------------------------
        # Global sites: mean-field Gaussian on UNCONSTRAINED space
        # ------------------------------------------------------------------
        def reg(name: str, shape: tuple, init_loc: float = 0.0, init_scale: float = 0.1):
            loc = numpyro.param(f"{name}_loc", init_loc * jnp.ones(shape))
            scale = numpyro.param(
                f"{name}_scale",
                init_scale * jnp.ones(shape),
                constraint=dist.constraints.positive,
            )
            return loc, scale

        def sample_normal(name: str, shape: tuple, n_event: int, init_loc=0.0, init_scale=0.1):
            loc, scale = reg(name, shape, init_loc, init_scale)
            d = dist.Normal(loc, scale).to_event(n_event)
            return numpyro.sample(name, d)

        def sample_transformed(name: str, shape: tuple, n_event: int, transform, init_loc=0.0, init_scale=0.1):
            loc, scale = reg(name, shape, init_loc, init_scale)
            base = dist.Normal(loc, scale).to_event(n_event)
            td = dist.TransformedDistribution(base, transform)
            return numpyro.sample(name, td)

        sample_normal("Theta_0", (R,), 1)
        sample_normal("mu_c", (C, R), 2)
        sample_normal("alpha_c", (C,), 1)
        sample_normal("beta_c_sub", (C, p_x), 2)
        sample_normal("h_c", (C, p_w), 2)
        sample_normal("f_c_age", (C,), 1)
        sample_transformed("sigma_c_stage", (C,), 1, dist.transforms.ExpTransform())
        sample_transformed(
            "pi_region", (C, 4), 2,
            dist.transforms.SigmoidTransform(),
            init_loc=2.0,  # sigmoid(2.0) ~ 0.88 prior inclusion
        )
        sample_transformed(
            "delta_region", (C, R), 2,
            dist.transforms.SigmoidTransform(),
            init_loc=2.0,
        )
        # When the cascade prior is on, `onset_times` becomes a
        # deterministic site in the model. Register modality-level onset
        # centers + per-biomarker eta variates instead, matching the new
        # sample-site names (t_ATN_raw, t_tilde_B, onset_eta).
        # When the spatial prior is on, `onset_eta` is replaced
        # by per-modality residual sites `onset_eta_mod{m}`, and `beta_vel`
        # is replaced by per-modality residual sites `beta_vel_mod{m}`.
        _use_spatial = getattr(cfg, "use_paper_spatial_priors", False)
        if getattr(cfg, "use_paper_cascade_prior", False):
            sample_normal(
                "t_ATN_raw", (C, 3), 2,
                init_loc=0.0,          # loc is additive on top of the prior mean inside the model
            )
            sample_normal("t_tilde_B", (C,), 1, init_loc=0.0)
            if _use_spatial:
                mod_sizes = (cfg.R_A, cfg.R_T, cfg.R_N, cfg.R_blood)
                for m_i, R_m in enumerate(mod_sizes):
                    if R_m <= 0:
                        continue
                    sample_normal(f"onset_eta_mod{m_i}", (C, R_m), 2, init_scale=0.3)
            else:
                sample_normal("onset_eta", (C, R), 2, init_scale=0.05)
        else:
            if _use_spatial:
                mod_sizes = (cfg.R_A, cfg.R_T, cfg.R_N, cfg.R_blood)
                for m_i, R_m in enumerate(mod_sizes):
                    if R_m <= 0:
                        continue
                    sample_normal(f"onset_eta_mod{m_i}", (C, R_m), 2, init_scale=0.3)
            else:
                sample_transformed("onset_times", (C, R), 2, dist.transforms.SigmoidTransform())
        sample_transformed("sharpness", (C, R), 2, dist.transforms.ExpTransform(), init_loc=-1.5)
        if _use_spatial:
            mod_sizes = (cfg.R_A, cfg.R_T, cfg.R_N, cfg.R_blood)
            for m_i, R_m in enumerate(mod_sizes):
                if R_m <= 0:
                    continue
                sample_normal(f"beta_vel_mod{m_i}", (C, R_m, Q), 3, init_scale=0.3)
        else:
            sample_normal("beta_vel", (C, R, Q), 3)
        sample_normal("Theta_J", (C, R), 2)
        sample_normal("log_D", (C, R), 2, init_loc=cfg.log_d_prior_mean)

        if phase == "phase2":
            if getattr(cfg, "use_ard_on_A", False):
                # Phase 1a ARD branch: the model samples alpha_ard ~ Gamma in
                # place of delta_shrink. Mirror as log-Normal under ExpTransform
                # (same trick used elsewhere in this guide for positive sites).
                sample_transformed(
                    "alpha_ard", (C, K), 2, dist.transforms.ExpTransform()
                )
            else:
                sample_transformed("delta_shrink", (C, K), 2, dist.transforms.ExpTransform())
            sample_normal("a_flat", (C, R, K), 3, init_scale=0.3)
            # fix: HalfNormal-direct diagonal replaces the legacy
            # Normal+softplus. Under the flag, the model samples `a_diag_pos`
            # (positive) instead of `a_diag_raw` (unconstrained); the guide
            # must mirror or the ELBO trace will mismatch and the positive site
            # will have no variational factor. Use ExpTransform so the guide's
            # base Normal pushes through a positive bijector.
            if getattr(cfg, "use_paper_halfnormal_diag", False):
                sample_transformed("a_diag_pos", (C, K), 2, dist.transforms.ExpTransform())
            else:
                sample_normal("a_diag_raw", (C, K), 2)
            sample_transformed("tau_k", (K,), 1, dist.transforms.ExpTransform())
            # When the smoothness prior
            # is on the model samples `w_ckp_raw_std` (iid N(0, I_P) pre-
            # Cholesky) instead of `w_ckp_raw`. The guide must expose the
            # matching site; the loc/scale on the std variate still form a
            # mean-field Gaussian.
            if getattr(cfg, "use_paper_w_ckp_smoothness", False):
                sample_normal("w_ckp_raw_std", (C, K, P), 3, init_scale=0.3)
            else:
                sample_normal("w_ckp_raw", (C, K, P), 3, init_scale=0.3)

        if cfg.use_cognition:
            sample_normal("gamma_0", (), 0)
            sample_normal("gamma_X", (p_x,), 1)
            sample_normal("log_sigma_C", (), 0)
            # The chi_c_raw site shape depends on
            # cfg.chi_c_basis. Polynomial (default) keeps (C, 3); I-spline uses
            # (C, cfg.chi_c_ispline_n_basis).
            if getattr(cfg, "chi_c_basis", "polynomial") == "ispline":
                sample_normal(
                    "chi_c_raw",
                    (C, int(getattr(cfg, "chi_c_ispline_n_basis", 3))),
                    2,
                )
            else:
                sample_normal("chi_c_raw", (C, 3), 2)

        # ------------------------------------------------------------------
        # Per-subject amortized site: logit_S via RecognitionMLP
        # Output shape: (n, C, 2) per subject — matches the structured-stage
        # site shape (n, C) in the model. For models with structured_stage=False,
        # callers should use guide_type='auto_normal' instead.
        # ------------------------------------------------------------------
        _use_wider = getattr(cfg, "wider_encoder", False) or getattr(cfg, "enhanced_encoder", False)
        _hidden = (256, 256, 128) if _use_wider else (128, 128)
        encoder = flax_module(
            "encoder",
            RecognitionMLP(C=C, hidden=_hidden, dropout=0.1),
            input_shape=(2 * R + p_x,),
        )
        enc_inputs = jnp.concatenate([Y * mask, mask, X], axis=-1)   # (n, 2R + p_x)
        mu_s_logit_nc, log_sigma_s_logit_nc, logit_c = encoder(enc_inputs)   # each (n, C)
        sigma_s_logit_nc = jax.nn.softplus(log_sigma_s_logit_nc) + 1e-3

        with numpyro.plate("subjects", n):
            if cfg.structured_stage:
                # Structured: sample (n, C) joint with event dim C
                numpyro.sample(
                    "logit_S",
                    dist.Normal(mu_s_logit_nc, sigma_s_logit_nc).to_event(1),
                )
            else:
                # Legacy: collapse over C by selecting column 0
                numpyro.sample(
                    "logit_S",
                    dist.Normal(mu_s_logit_nc[:, 0], sigma_s_logit_nc[:, 0]),
                )
            # Expose amortized subtype posterior q(c_i | Y_i, X_i).
            # When train_q_c=True: sample site matched by Dirichlet prior in model.
            # When train_q_c=False: deterministic site (no gradient, untrained).
            q_c_val = jax.nn.softmax(logit_c, axis=-1)
            if cfg.train_q_c:
                numpyro.sample("q_c", dist.Delta(q_c_val).to_event(1))
            else:
                numpyro.deterministic("q_c", q_c_val)

    return guide


def make_perSubject_logit_s_guide(cfg: ModelConfig, phase: str = "phase2"):
    """Build a guide that uses PER-SUBJECT numpyro.param for logit_S — one
 independent (loc, scale) per subject, no encoder sharing.

 This is the 'local, non-amortized' variant. It's explicit about the
 per-subject latent and sidesteps AutoNormal's `_prototype_frame_full_sizes`
 plate-lookup issue that trips when composing AutoNormal with a blocked
 model under `AutoGuideList`.
 """
    def guide(Y, mask, X, W, ages_norm, anchor_type, psi_basis_cache, gumbel_tau=None, C_cog=None, C_cog_mask=None, **kwargs):
        n, _ = Y.shape
        C = cfg.C

        if cfg.structured_stage:
            loc = numpyro.param(
                "logit_S_local_loc",
                jnp.zeros((n, C)),
            )
            scale = numpyro.param(
                "logit_S_local_scale",
                0.1 * jnp.ones((n, C)),
                constraint=dist.constraints.positive,
            )
            with numpyro.plate("subjects", n):
                logit_S = numpyro.sample(
                    "logit_S",
                    dist.Normal(loc, scale).to_event(1),
                )
        else:
            loc = numpyro.param(
                "logit_S_local_loc",
                jnp.zeros(n),
            )
            scale = numpyro.param(
                "logit_S_local_scale",
                0.1 * jnp.ones(n),
                constraint=dist.constraints.positive,
            )
            with numpyro.plate("subjects", n):
                logit_S = numpyro.sample(
                    "logit_S",
                    dist.Normal(loc, scale),
                )
        return {"logit_S": logit_S}

    return guide




def make_amortized_logit_s_guide(cfg: ModelConfig, phase: str = "phase2"):
    """Build a guide that amortizes ONLY the per-subject logit_S site via a
 flax MLP.

 IMPORTANT: this guide MUST be composed inside an `AutoGuideList` using
 `handlers.block(model, expose=["logit_S"])` so it only acts on the
 logit_S site.
 """
    from numpyro.contrib.module import flax_module

    def guide(Y, mask, X, W, ages_norm, anchor_type, psi_basis_cache,
              gumbel_tau=None, C_cog=None, C_cog_mask=None, **kwargs):
        n, R_obs = Y.shape
        C = cfg.C
        p_x = cfg.p_x
        _use_wider = getattr(cfg, "wider_encoder", False) or getattr(cfg, "enhanced_encoder", False)
        _hidden = (256, 256, 128) if _use_wider else (128, 128)

        encoder = flax_module(
            "logit_s_encoder",
            RecognitionMLP(C=C, hidden=_hidden, dropout=0.1),
            input_shape=(2 * R_obs + p_x,),
        )
        enc_inputs = jnp.concatenate([Y * mask, mask, X], axis=-1)
        mu_s, log_sigma_s, logit_c = encoder(enc_inputs)
        sigma_s = jax.nn.softplus(log_sigma_s) + 1e-3

        with numpyro.plate("subjects", n):
            if cfg.structured_stage:
                logit_S = numpyro.sample(
                    "logit_S",
                    dist.Normal(mu_s, sigma_s).to_event(1),
                )
            else:
                logit_S = numpyro.sample(
                    "logit_S",
                    dist.Normal(mu_s[:, 0], sigma_s[:, 0]),
                )
            q_c_val = jax.nn.softmax(logit_c, axis=-1)
            if cfg.train_q_c:
                numpyro.sample("q_c", dist.Delta(q_c_val).to_event(1))
            else:
                numpyro.deterministic("q_c", q_c_val)

        return {"logit_S": logit_S}

    return guide
