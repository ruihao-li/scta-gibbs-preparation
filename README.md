# Numerical demonstration of SCTA local reduction

This repository contains the source code for the numerical demonstration in "Spectral Core–Tail Architecture for Locally Certified Gibbs-State Preparation": [paper-link].

It implements the open alternating-field graph-stabilizer chain, the scheduled first-order correction, the endpoint Gibbs-state comparisons, and the variational continuation through resonance and finite deformation.


## Scientific scope

The benchmark uses the core-frame Hamiltonian

$$
H_C(\lambda) = -\sum_i h_i Z_i + \lambda J \sum_i X_i X_{i+1}, \qquad h_i = \bar{h} + (-1)^i \delta.
$$

Here the site labels are one-based, the boundaries are open, $\bar{h} = 1.25$, $J = 1$, and the default staggering is $\delta = 0.25$.
Qubit 0 is the leftmost, most-significant tensor factor in every dense matrix and Pauli label.
The first-order circuit is evaluated as the exact two-layer unitary

$$
W = \exp(\lambda S_{\mathrm{odd}})\exp(\lambda S_{\mathrm{even}});
$$

no product-formula or truncated exponential is used.

The production workflow reproduces Figs. 3--5 in the manuscript.

The endpoint Gibbs-state calculations do not test the full interpolation-path response assumption used by the analytical certification result.


## Environment

The archived calculations used Python 3.10 with the exact package versions in `requirements.txt`.

A typical isolated setup is

```bash
cd numerics
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```


## Tests and inexpensive smoke runs

Run the unit suite before starting the production calculations:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The smoke modes exercise all execution pathways with $N = 4$ and write outside the production output tree:

```bash
.venv/bin/python run_benchmarks.py --smoke --output /tmp/scta-benchmark-smoke
.venv/bin/python run_variational.py --smoke --output /tmp/scta-variational-smoke
```

The variational smoke run retains the two deterministic and eight seeded random starts and includes exact resonance.

## Full reproduction

From this directory, run

```bash
.venv/bin/python run_benchmarks.py
.venv/bin/python run_variational.py
.venv/bin/python analyze_results.py
```

The first command writes the residual and thermal raw files under `results/benchmarks/raw/`.
The second command writes pointwise checkpoints under `results/variational/raw/` and is the computationally expensive stage.
The third command checks both output-schema versions, verifies the data-generation portion of the embedded benchmark configuration and the full embedded variational configuration, computes the reported fits and checks, writes `results/processed/`, and generates the three PDF figures under `figures/`.
The benchmark fit windows may therefore be changed and reanalyzed without regenerating the raw Hamiltonian data, whereas a variational-configuration change requires a new variational run.

The production variational configuration contains 20 unique physical parameter points and ten starts per point, for 200 optimizer runs.
The shared point $(\delta, \lambda) = (0.25, 0.02)$ is optimized once and recorded under both raw scan memberships.
For the cross-start robustness summary, an objective is called near-best when

$$
\lvert f - f_{\min} \rvert \leq \max\left(10^{-10}, 10^{-8}\lvert f_{\min} \rvert\right);
$$

these comparison tolerances do not enter the L-BFGS-B optimization.

If the variational calculation is interrupted, continue it with

```bash
.venv/bin/python run_variational.py --resume
```

To replace an existing deterministic run intentionally, pass `--overwrite` to `run_benchmarks.py`.
The variational runner never overwrites an existing checkpoint without `--resume`; choose a fresh `--output` directory for an independent rerun.

## Custom output locations

The two runners and the analyzer can share any run root without moving files manually:

```bash
RUN_ROOT=/path/to/run/results
PROCESSED_ROOT=/path/to/run/analysis/processed
FIGURE_ROOT=/path/to/run/analysis/figures
.venv/bin/python run_benchmarks.py --output "$RUN_ROOT/benchmarks"
.venv/bin/python run_variational.py --output "$RUN_ROOT/variational"
.venv/bin/python analyze_results.py --results-root "$RUN_ROOT" \
  --processed-output "$PROCESSED_ROOT" --figures-output "$FIGURE_ROOT"
```

