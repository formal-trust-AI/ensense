# ATVA26 glitch experiments

Reproduces the four artifacts of the ATVA26 glitch paper -- Figure 5, Figure 6,
Table 2, Table 3 -- by running Ensense (`src/sensitive.py --solver glitch`) across the
benchmark models in `models/`.

A **glitch** is a triple of inputs that agree on every feature but one, are ordered
along that feature, and whose outputs cross the decision boundary and then cross back,
with each consecutive step separated by at least the crossing gap. Its magnitude is

```
alpha = min(|f(x1) - f(x0)|, |f(x2) - f(x1)|) / dist(x0, x2)
```

-- how much output movement is bought per unit of input movement.

## The three problems

| Problem | Paper | Command shape |
|---|---|---|
| 1 | `TE_GLITCH(a, i)` -- fixed feature, feasibility | `--alpha A --features i` |
| 2 | `TE_GLITCH(a)` -- any feature, feasibility | `--alpha A` |
| 3 | `TE_GLITCH` -- any feature, maximise alpha | `--alpha 0` |

Problems 2 and 3 need the any-feature mode: with no `--features`, `glitch.py` adds a
binary selector so exactly one feature -- chosen by the solver -- may differ between
the three points.

## Running

```bash
python3 ATVA26/run_smoke.py                 # representative subset
python3 ATVA26/run_full.py  <result-dir> <workers> # every config, 3600s each
```

Both write per-job directories under the output root (default
`experiment_results/atva_{smoke,full}/`), each holding `results.csv` and `run.log`,
then print the RQ1/RQ2/RQ3 summaries and write `report.pdf` -- Figure 5, Figure 6,
Table 2, Table 3, in that order.

Safe to interrupt: a job whose directory already has `results.csv` is skipped on
re-run.

`ENSENSE_PYTHON=/path/to/python3` if gurobipy and xgboost live somewhere other than
the interpreter you launch with.

Individual pieces, against any finished results directory:

```bash
python3 ATVA26/analyze.py      <dir>
python3 ATVA26/tables.py       <dir>
python3 ATVA26/plot_figure5.py <dir> [out.png]
python3 ATVA26/plot_figure6.py <dir> [out.png] [timelimit]
python3 ATVA26/make_report.py  <dir> [out.pdf] [timelimit]
```


## Per-problem time limits

`run_smoke.py` gives Problem 3 a larger budget than Problems 1 and 2 (300s vs 30s)
rather than one flat limit. Measured on `diabetes`: a Problem 2 query returns in 0.13s
where the Problem 3 maximisation on the same model takes 15s (robust) and does not
prove optimality within 30s (unrobust). A flat budget reports P3 as timed out almost
everywhere and makes P1/P2 look trivially easy -- the wrong picture of the difficulty
ordering RQ3 reports. `run_full.py` keeps the paper's flat 3600s.

## Models

The self-trained grid is **discovered from `models/`**.

| dataset | configs |
|---|---|
| adult, breast_cancer, churn, pimadiabetes, spambase | t200, t300, t500 x d5, d6 |
| german_credit | t500, t800 x d5, d6 |

plus the 12 verification models (`robust`/`unrobust` x 6 datasets), for **46 configs,
all present**. Hardcoding any single grid would blank real configs out of the results:
five datasets ship t300 and only german_credit ships t800.

Tables 2 and 3 use the union of configs as their row set, so a dataset lacking a row
shows `--` there -- distinct from `TO` (ran, timed out) and `unsat` (proven no glitch
exists). Toy models (fewer than `MIN_TREES_FOR_GRID` = 100 trees, e.g. `adult_t2_d2`)
stay out of the row set.

Nothing here trains models. The boosters in `models/` are the ones the paper's numbers
come from; a retrained model has different trees, different split thresholds, and
therefore different glitch magnitudes, so it would not be comparable.

## Output layout

```
<output_dir>/
  p1_<model>_<config>_f<i>/  one per feature, Problem 1
  p2_<model>_<config>_a<A>/  one per alpha threshold, Problem 2
  p3_<model>_<config>/       one per config, Problem 3
      results.csv            sat / time / alpha / witness
      run.log                the full solver invocation and its output
  report.pdf
```

Each `results.csv` is one row that records both halves of the job -- what was asked
(`problem`, `feature`, `givenalpha`) and what came back (`sat`, `time`, `alpha`,
`threepoints`) -- so a results directory needs no separate index to be read.

The `sat` column reads `sat` (proven feasible, or proven optimal for Problem 3),
`unsat` (proven no glitch exists), `timelimit` (budget hit -- `threepoints`
distinguishes an incumbent from nothing), or `error`. A job directory with no
`results.csv` at all is a job that started and died; it is reported as `MISSING`
rather than dropped, so it cannot quietly shrink a denominator.

## Files

| File | Role |
|---|---|
| `common.py` | grid discovery, feature eligibility, job builders, subprocess runner, run-log-to-`results.csv` parser |
| `run_smoke.py` / `run_full.py` | the two sweeps, each ending in `report.pdf` |
| `analyze.py` | RQ1 prevalence, RQ2 robust vs unrobust, RQ3 scalability |
| `tables.py` | Table 2 (prevalence) and Table 3 (magnitudes) |
| `plot_figure5.py` | Figure 5, robust vs unrobust bar chart |
| `plot_figure6.py` | Figure 6, three-panel cactus plot |
| `make_report.py` | the four pages combined into `report.pdf` |

The analysis scripts read only the per-job `results.csv` files, so they work against
any results directory without re-running anything.
