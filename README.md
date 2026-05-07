# ELPAC: Endpoint-anchored Latent Progression And Coordination

Reference implementation accompanying the manuscript. The package bundles
the NumPyro generative model, the synthetic-cohort generator, the phased
SVI driver, and the per-seed reproduction script for the headline
synthetic experiment. The variational family follows the inference
section of the paper: a low-rank multivariate Gaussian on global
parameters paired with an amortized recognition network for per-subject
stage and subtype.

## Requirements

Python 3.11. Install via conda:

```
conda env create -f environment.yaml
conda activate elpac-demo
```

Or via pip:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `jax==0.4.30`, `numpyro==0.15.3`, `flax==0.8.5`,
`optax==0.2.3`. Production runs reported in the manuscript use a single
NVIDIA RTX Pro 6000 Blackwell GPU; this demo also runs on CPU.

## Reproducing the synthetic experiment

The reproduction entry point is `run_synthetic.py`. Default
configuration matches the paper's synthetic setup: C=3 subtypes, K=3
latent factors, J=8 trajectory shells, R=20 biomarkers across four
modalities, N=1000 subjects, 10% endpoint anchors per subtype, 10%
healthy reference, 60% unlabeled. The variational family is the
low-rank multivariate Gaussian (rank `r_g=10`) for globals plus the
amortized recognition MLP described in the paper (two hidden layers of
width 128, LayerNorm, SiLU, dropout 0.1) for the per-subject stage and
subtype. Concrete-relaxed Bernoulli region gates are annealed across the
phased SVI schedule (Phase 1 mean-only at lr `3e-3`; Phase 2
stage-varying covariance at lr `1e-3`; minibatch 256; grad-clip 5; Adam).

Run a single seed (~4 min on CPU):

```
python run_synthetic.py --seeds 0
```

Run multiple seeds:

```
python run_synthetic.py --seeds 0,1,2,3
```

For each seed the script (i) generates a synthetic cohort from the
simulator, (ii) draws a stratified 80/20 train–test split, (iii) fits
ELPAC end-to-end on the train partition, (iv) computes recovery metrics
(subtype accuracy, ARI, stage Spearman ρ and MAE, mean-trajectory RMSE,
loading Procrustes distance, activation recovery, ECE, Brier) on the
train cohort against simulator ground truth, and (v) computes the
held-out marginal log-likelihood per subject on the 20% test partition.
Results are written to `artifacts/seeds_<lo>_<hi>.json`.

This package bundles only the headline reproduction script. The
manuscript's other synthetic studies (covariance ablation, sample-size
sweep, robustness, sensitivity, guide ablation), the masked
leave-one-modality-out NLL evaluation, and the external baselines
(SuStaIn, MoPoE-VAE, HEALNet, etc.) are not included.

## Results

Reported synthetic-recovery numbers in the manuscript are means ± SE
across 100 random seeds; running `--seeds 0,...,99` and
aggregating the resulting JSONs reproduces those tables.

## Hyperparameters

The boundary-supervision priors use the values reported in the paper:
`s_A = 0.95`, `s_H = 0.01`, `ε_S = 0.1`. These are the defaults in
`elpac.model.ModelConfig` (`eps_pinning = 0.1`) and the simulator's
anchor-stage draws.

## Repository layout

```
code_submission/
├── elpac/                    # Importable package
│   ├── model.py              # NumPyro generative model
│   ├── simulate.py           # Synthetic-cohort generator
│   ├── inference.py          # Phased SVI driver
│   ├── amortized_guide.py    # Amortized recognition network
│   ├── heldout.py            # Stratified train/test split + held-out marginal LL
│   ├── metrics.py            # Recovery-metric helpers
│   ├── utils/                # B-spline / constraints / Sylvester-Woodbury / spatial-kernel
│   └── analysis/             # Correlations, O-information, Fisher weights
├── run_synthetic.py          # Single entry point (training + evaluation)
├── environment.yaml          # conda env: elpac-demo
├── requirements.txt          # pip fallback
└── LICENSE                   # MIT
```

## License

MIT — see `LICENSE`.
