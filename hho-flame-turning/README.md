# hho-flame-turning

Python code and results for the article *Comparative Assessment of Six Metaheuristic Algorithms, Including Harris Hawks Optimization, for Surface Roughness and Material Removal Rate in Flame-Assisted Turning of Al/SiC Composites*.

The repository implements six population-based optimizers (DE, WOA, CS, TLBO, HHO, PSO), the three empirical power-law roughness models (dry, carburizing flame, oxidizing flame) and the analyses reported in the article. **No new machining experiments are included.** The roughness models come from the authors' earlier experimental work on a 15% SiC/Al 6061 composite.

## Problem

Variables: cutting speed `v` (34–94 m/min), feed `f` (0.113–0.249 mm/rev), depth of cut `d` (0.25–0.75 mm).

```
Ra_carburizing = 22.1005 v^-0.052 f^1.32  d^0.06
Ra_oxidizing   = 81.45   v^-0.356 f^1.2   d^0.106
Ra_dry         = 22.83   v^-0.028 f^1.28  d^0.06
MRR = v f d                       (cm^3/min)
F   = Ra_carburizing + Ra_oxidizing + Ra_dry + lambda / MRR     (lambda = 1)
```

Reference optimum (L-BFGS-B multi-start and 41^3 grid): `v = 94`, `f = 0.113`, `d = 0.3938`, `F* = 3.4034`.

Note on the oxidizing model: the feed exponent is 1.2 here. The earlier report gives 2.1 in its text and 1.2 in its optimization problem. Only 1.2 reproduces the earlier optimum and gives roughness values of the same order as the other two models.

## Layout

```
src/flameopt/models.py       models, bounds, objective
src/flameopt/algorithms.py   DE, WOA, CS, TLBO, HHO, PSO with a shared evaluation counter
scripts/run_all.py           every experiment of the article (resumable; --quick for a smoke test; --force to recompute)
scripts/make_figures.py      all figures from ./results
tests/test_models.py         unit tests (run with pytest)
results/                     run-level CSV files, summaries, Holm-adjusted p value matrices
figures/                     PNG figures used in the article
data/matlab_runs_original.csv  per-run records of an earlier MATLAB implementation (DE, CS, TLBO, HHO)
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py          # about 10-15 minutes
python scripts/make_figures.py
pytest -q
```

Seeds are fixed. Run `k` uses the same initial population for all algorithms (`default_rng(10000 + k)`), and the algorithm's own random stream is `default_rng(20000 + k)`.

## Protocols

* **A** – population 30, 20 iterations, 100 runs (as in the earlier work).
* **B** – population 30, budget of 1200 function evaluations, 100 runs.
* Success: `F - F* <= 0.01`. Statistics: Friedman test and paired Wilcoxon signed-rank tests with Holm correction; fitness is rounded to 1e-6 before ranking.
* Additional studies: iteration budget (10–100), population size (10–100), weight `lambda` (0.25–8), per-environment bi-objective analysis (Ra vs MRR), and Monte Carlo perturbation (±5%, ±10%) of all 12 model coefficients.

## Data note

`data/matlab_runs_original.csv` holds the 100 runs of an earlier MATLAB implementation. In the table from which it was taken, the `f` and `d` columns of the HHO block had been transposed; they were swapped back after recomputing `F` from the decision variables. The WOA block of that table contained only copies of `v` and is therefore not included.

## License

MIT (see `LICENSE`). Replace the copyright line with the names of the authors before publishing.

## Citation

Please cite the article. A DOI for this repository can be obtained by archiving a release on Zenodo.
