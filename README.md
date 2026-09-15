# Ensense 0.2

A tool for **analysing the decision surface of tree ensemble models**. Ensense supports XGBoost, Random forest, and LGBM models.

It can check the following two properties :

- **Sensitivity.** Are there two inputs that differ only in the features you nominate, yet
  land on opposite sides of the decision boundary? A pair like that means the nominated
  features alone can flip the prediction (ICLR 2025, ICLR2026).
- **Glitches.** Are there small neighborhoods in the input space where the model's output
  abruptly oscillates with respect to small changes in the input? Ensense searches for them
  along a single feature: three inputs whose predictions cross the decision boundary and
  then cross back (ATVA 2026).

Both searches are exact: each answer is a witness you can re-run through the model, or a
proof that none exists.

## Quick start

Install dependencies and run a minimal analysis:


```bash
pip install -r requirements.txt
 
python ./src/main.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 2

python ./src/main.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json --features 5 --spec glitch --solver milp

```

That's it — this runs a sensitivity/glitch check on features 5 of the bundled breast-cancer model. The sections below explain how to go further

---
## Installation
 
### Python dependencies
 
```bash
pip install -r requirements.txt
```
 
### roundingsat
 
Required by the pseudo-Boolean (`pb`) solver:
 
```bash
cd ./utils
./installrounding.sh
```

## Running the docker image (alternative)

Build the image:
 
```bash
docker build -t sensitivity .
```

Run it (drops you into a command-line interface):
 
```bash
docker run -it sensitivity
```
Save the image to share it, and load it elsewhere:
 
```bash
docker save -o sensitivity.tar sensitivity:latest   # save
docker load -i sensitivity.tar                      # load
```
---
 
## Usage
 
### Command syntax

```bash
python ./src/main.py <model file> [options]
```
To see every available option:
 
```bash
python ./src/main.py -h
```

### Options
 
| Option | Purpose |
|---|---|
| `--features <list>` | Indices of the sensitive features to analyze (required). |
| `--alpha <float>` | Glitch magnitude to search for, with `--spec glitch --solver milp`. `0` maximises it. |
| `--spec <name>` | Specifications: `sens,glitch,monitor` . |
| `--solver <name>` | Solver backend: `pb,naive_smt,rounding,roundingsoplex,milp` . |
| `--output_gap <value pair>` | Output gap bounds, e.g. `0.2 0.8`. |
| `--precision <int>` | Numeric precision used by the solver. |
| `--timeout <int>` | Solver timeout in seconds. |
| `--details <csv>` | Feature names,type, operating ranges. |
| `--all_opt` | Ensense mode: optimized search. |
| `--prob` | Ensense mode: probabilistic / distribution-aware search. |
| `--compute_data_distance` | Compute distance between the nearest data point and the sensitivity pair. |
| `--data_file <csv>` | Training data, used for data-aware search. |
| `--in_distro_clauses <txt>` | cavity info(rule) file. |
| `--local_check_file <csv>` | Run local sensitivity search over the points in this file. |
| `--local_check_sample <float list>` | Run local sensitivity search around a single point. |
| `--perturb <value>` | Perturbation distance for local sensitivity search. |
|`--multiclass`      |    Enable support for multi-class models|
|`--truelabel` | Label of true class, required with `--multiclass`|
| `--otherlabel` | Label of other class, required with `--multiclass` |
 

### Why you may need a details file
 
A model file sometimes lacks information such as feature **names** and their valid **operating ranges**. The `--details` file supplies this missing metadata so that results are interpretable and well-bounded.

Details file has format 
```
feature,name,lb,ub,type
0,SALARY,10000,1000000,log
1,AGE,18,55,linear
2,HAS_CAR,0,1,bool
```

 
---
 
## Modes
 
### Global sensitivity search
 
The default mode. Pick the sensitive features and (optionally) tighten the search with output gap, precision, and a details file:
 
```bash
python ./src/main.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json \
  --features 2 5 --output_gap 0.2 0.8 --precision 400 --timeout 100 \
  --details models/dataset/breast_cancer/breast_cancer_details.csv --solver milp
```
 
### Local sensitivity search
 
Searches for sensitivity around a specific point, up to a given perturbation distance. Provide the sensitive features (`--features`), a point (`--local_check_sample` for a single point, or `--local_check_file` for many), and a perturbation bound (`--perturb`).
 
```bash
python ./src/main.py models/adult/adult_t200_d5.json \
  --features 11 --output_gap 0.2 0.8 --precision 400 --timeout 100 \
  --details models/dataset/adult/details.csv --solver milp --all_opt --prob \
  --compute_data_distance --data_file models/dataset/adult/train.csv \
  --in_distro_clauses outputs/cavities/learned-clauses_adult_t200_d5.txt \
  --local_check_sample 0.089 0.5625 0.09353 0.63 0.7 0.75 0.10 0.9 0.375 0.5 -0.4 0.4 1.45
```

 
### Data-aware search
 
First generate a file summarizing data patterns:
 
```bash
./src/learn-data.py \
  --model models/tree_verification_models/breast_cancer_robust/0004.resaved.json \
  --data models/dataset/breast_cancer/breast_cancer_train.csv \
  --output outputs/cavities/learned-clauses_breast_cancer \
  --details models/dataset/breast_cancer/breast_cancer_details.csv
```
 
Then pass that file to the search via `--in_distro_clauses`:
 
```bash
python ./src/main.py models/adult/adult_t200_d5.json \
  --features 11 --output_gap 0.2 0.8 --precision 400 --timeout 100 \
  --details models/dataset/adult/details.csv --solver milp --all_opt --prob \
  --compute_data_distance --data_file models/dataset/adult/train.csv \
  --in_distro_clauses outputs/cavities/learned-clauses_adult_t200_d5.txt
```


### Data-conformal sensitivity checking
 
Enables checking the distance between the neares training data and the sensitivity pair:
 
```bash
python ./src/main.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json \
  --features 7 --output_gap 0.2 0.8 --precision 400 --timeout 100 \
  --details models/dataset/breast_cancer/breast_cancer_details.csv --solver milp \
  --all_opt --prob --compute_data_distance \
  --data_file models/dataset/breast_cancer/breast_cancer_train.csv
```

### Glitch search
 
This is the first tool to find the glitches, which are small neighborhoods in the input space where the model's output abruptly
oscillates with respect to small changes in the input — a source of unreliable behaviour
in models with steep decision boundaries. Ensense searches for them along a single
feature: a **glitch** is a triple of inputs that agree on every feature except one, are
ordered along that feature, and whose outputs cross the decision boundary and then cross
back. Its magnitude is
 
```
alpha = min(|f(x1) - f(x0)|, |f(x2) - f(x1)|) / dist(x0, x2)
```
 
how much output movement the model buys per unit of input movement. Use `--spec glitch --solver milp`
with `--alpha`, and name the feature with `--features` or leave it to the solver:
 
```bash
# Is there a glitch of magnitude >= 1 along feature 2?
python ./src/main.py models/tree_verification_models/diabetes_robust/0020.resaved.json \
  --spec glitch --solver milp --alpha 1 --features 6 --timeout 60 \
  --details models/dataset/diabetes/diabetes_details.csv
 
# Is there one of magnitude >= 1 along any feature?
python ./src/main.py models/tree_verification_models/diabetes_robust/0020.resaved.json \
  --spec glitch --solver milp --alpha 1 --features 6 --timeout 60
 
# What is the largest glitch in this model, over any feature?
python ./src/main.py models/tree_verification_models/diabetes_robust/0020.resaved.json \
  --spec glitch --solver milp --alpha 0 --features 6 --timeout 300
 
# One fixed-feature query per feature, in turn
python ./src/main.py models/tree_verification_models/diabetes_robust/0020.resaved.json \
  --spec glitch --solver milp --alpha 1 --features 6 --all_single --timeout 60
```
 
Each run reports the glitch feature, the magnitude found, and the three input points,
and checks the witness against the model itself. A feature needs at least two distinct
splits to host a glitch; features below that are skipped.
 
Leaving out `--alpha` is the same as `--alpha 0`, so a bare `--spec glitch --solver milp` asks for
the largest glitch. The maximisation is markedly harder than the feasibility questions
and wants a longer `--timeout`.
 
`--output_gap` and `--prob` are not accepted by this solver: what counts as a glitch is
fixed by the definition above rather than configurable, and the magnitude is measured in
margin space.
 
---
 
## Reproducing prior work
 
Ensense bundles implementations that reproduce two earlier approaches. In every case the difference is just which flags you append — the model file and core options stay the same.
 
### Pseudo-Boolean (pb) tool
 
Reproduces *[Sensitivity Verification for Additive Decision Tree Ensembles](https://openreview.net/pdf?id=h0vC0fm1q7)*. Use `--solver pb`:
 
```bash
python ./src/main.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json \
  --features 2 5 --output_gap 0.2 0.8 --precision 400 --timeout 100 \
  --details models/dataset/breast_cancer/breast_cancer_details.csv --solver pb
```
 
### Evasion and Hardening of Tree Ensemble Classifiers
 
Our implementation of *[Evasion and Hardening of Tree Ensemble Classifiers](https://arxiv.org/abs/1509.07892)*. Use `--solver milp`:
 
```bash
python ./src/main.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json \
  --features 2 5 --output_gap 0.2 0.8 --precision 400 --timeout 100 \
  --details models/dataset/breast_cancer/breast_cancer_details.csv --solver milp
```
 
### Ensense
 
The full Ensense method from *[Data-Aware and Scalable Sensitivity Analysis for Decision Tree Ensembles](https://arxiv.org/abs/2602.07453)*. Add `--all_opt` (and optionally `--prob`, `--compute_data_distance`, `--data_file`, `--in_distro_clauses`) — see the **Modes** section above for the data-aware and distribution-aware variants.
 
```bash
python ./src/main.py models/tree_verification_models/breast_cancer_robust/0004.resaved.json \
  --features 2 5 --output_gap 0.2 0.8 --precision 400 --timeout 100 \
  --details models/dataset/breast_cancer/breast_cancer_details.csv --solver pb --all_opt
```

### Glitches in Decision Tree Ensemble Models (ATVA 2026)
 
Reproduces *[Glitches in Decision Tree Ensemble Models](https://arxiv.org/abs/2507.14492)*. Use `--spec glitch --solver milp` with `--alpha`, and either name the feature with `--features` or leave the choice to the solver. The paper's three problems map onto the flags directly:
 
```bash
# TE_GLITCH(alpha, i) -- is there a glitch of magnitude >= alpha along feature i?
python ./src/main.py models/tree_verification_models/diabetes_robust/0020.resaved.json \
  --spec glitch --solver milp --alpha 1 --features 2 --timeout 3600 \
  --details models/dataset/diabetes/diabetes_details.csv
 
# TE_GLITCH(alpha) -- is there one of magnitude >= alpha along any feature?
python ./src/main.py models/tree_verification_models/diabetes_robust/0020.resaved.json \
  --spec glitch --solver milp --alpha 1 --timeout 3600
 
# TE_GLITCH -- the largest glitch in the model, over any feature
python ./src/main.py models/tree_verification_models/diabetes_robust/0020.resaved.json \
  --spec glitch --solver milp --alpha 0 --timeout 3600
```
 
The paper's prevalence results sweep the first problem over every feature of a model, which `--all_single` does in one run:
 
```bash
python ./src/main.py models/tree_verification_models/diabetes_robust/0020.resaved.json \
  --spec glitch --solver milp --alpha 1 --all_single --timeout 3600
```
 
#### The paper's tables and figures
 
The commands above answer one query at a time. To rebuild the paper's own results —
Figure 5 (robust vs. unrobust), Figure 6 (the cactus plot), Table 2 (glitch prevalence)
and Table 3 (glitch magnitudes) — use the sweep harness in `ATVA26/`, which runs all
three problems across every bundled model and collects the answers:
 
```bash
python3 ATVA26/run_smoke.py                          # representative subset, ~30 min
python3 ATVA26/run_full.py experiment_results/full 4 # every model, 3600s per query
```
 
Both write one directory per query — holding its `results.csv` and `run.log` — then print
the RQ1/RQ2/RQ3 summaries and combine the four artifacts into `report.pdf`. Re-running
skips queries that already have a result, so a sweep is safe to interrupt. See
`ATVA26/README.md` for the per-problem time limits and the benchmark grid.

---
 
## Publications 

Sensitivity work
  - *[Data-Aware and Scalable Sensitivity Analysis for Decision Tree Ensembles](https://openreview.net/pdf?id=q8KqAvdfZK)*.
  - *[Sensitivity Verification for Additive Decision Tree Ensembles](https://openreview.net/pdf?id=h0vC0fm1q7)*

Glitch work
  - *[Glitches in Decision Tree Ensemble Models](https://arxiv.org/abs/2507.14492)*



