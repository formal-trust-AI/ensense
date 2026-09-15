#!/usr/bin/env python3

import concurrent.futures
import csv
import glob
import itertools
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)                           # .../xgboost
MAIN_PY = os.path.join(REPO_ROOT, "src", "main.py")
MODELS = os.path.join(REPO_ROOT, "models")

PYTHON = os.environ.get("ENSENSE_PYTHON", sys.executable)

SELF_TRAINED_DATASETS = ["adult", "breast_cancer", "churn", "german_credit", "pimadiabetes", "spambase"]


MIN_TREES_FOR_GRID = 100     
CONFIG_RE = re.compile(r"^t(\d+)_d(\d+)$")


def _config_sort_key(cfg):
    m = CONFIG_RE.match(cfg)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def discover_self_trained():
    found = {}
    for name in SELF_TRAINED_DATASETS:
        cfgs = []
        for path in glob.glob(os.path.join(MODELS, name, f"{name}_t*_d*.json")):
            cfg = os.path.basename(path)[len(name) + 1:-len(".json")]
            m = CONFIG_RE.match(cfg)
            if m and int(m.group(1)) >= MIN_TREES_FOR_GRID:
                cfgs.append(cfg)
        found[name] = sorted(set(cfgs), key=_config_sort_key)
    return found


SELF_TRAINED_BY_DATASET = discover_self_trained()
SELF_TRAINED_CONFIGS = sorted(
    {c for cfgs in SELF_TRAINED_BY_DATASET.values() for c in cfgs},
    key=_config_sort_key,
)

VERIFICATION_DATASETS = ["binary_mnist", "breast_cancer", "diabetes", "higgs", "ijcnn", "webspam"]
VERIFICATION_TYPES = ["robust", "unrobust"]

VERIFICATION_ITER = {
    "breast_cancer": "0004", "diabetes": "0020", "ijcnn": "0060",
    "webspam": "0100", "higgs": "0300", "binary_mnist": "1000",
}

TOP_K_RESTRICTED = {"binary_mnist", "webspam"}   # capped at top-40 features, as the paper does
TOP_K = 40

ALPHA_THRESHOLDS = [5, 25, 125, 625, 3125, 15625]


# ----------------------------------------------------------------------------
#  Configs
# ----------------------------------------------------------------------------

def verification_path(name, kind):
    return os.path.join(MODELS, "tree_verification_models",
                        f"{name}_{kind}", f"{VERIFICATION_ITER[name]}.resaved.json")


def self_trained_path(name, cfg):
    return os.path.join(MODELS, name, f"{name}_{cfg}.json")


def all_configs():
    
    configs = []
    for name in SELF_TRAINED_DATASETS:
        for cfg in SELF_TRAINED_BY_DATASET[name]:
            configs.append((name, cfg, self_trained_path(name, cfg)))
    for name, kind in itertools.product(VERIFICATION_DATASETS, VERIFICATION_TYPES):
        configs.append((name, kind, verification_path(name, kind)))
    return configs


def present(configs):
    """Split configs into (on disk, missing)."""
    have = [c for c in configs if os.path.isfile(c[2])]
    miss = [c for c in configs if not os.path.isfile(c[2])]
    return have, miss


# ----------------------------------------------------------------------------
#  Feature eligibility
# ----------------------------------------------------------------------------

def valid_features(modelpath, restrict_topk=False, top_k=TOP_K):
    
    import warnings
    warnings.filterwarnings("ignore")
    import xgboost as xgb

    b = xgb.Booster()
    b.load_model(modelpath)
    df = b.trees_to_dataframe()
    df = df[df["Feature"] != "Leaf"].copy()
    df["Split"] = df["Split"].round(6)

    per_feat = df[["Feature", "Split"]].drop_duplicates().groupby("Feature").size()
    feats = [f for f, n in per_feat.items() if n > 2]

    if restrict_topk:
        importance = b.get_score(importance_type="weight") or {}
        feats = sorted(feats, key=lambda f: -importance.get(f, 0))[:top_k]

    
    names = b.feature_names
    position = {name: i for i, name in enumerate(names)} if names else {}

    out = []
    for f in feats:
        f = str(f)
        if f in position:
            out.append(position[f])
        elif f.startswith("f") and f[1:].isdigit():
            out.append(int(f[1:]))
        else:
            raise ValueError(
                f"cannot map feature {f!r} of {os.path.basename(modelpath)} to an "
                "index: it is not in the model's feature_names and is not 'f<N>'")
    return sorted(out)


# ----------------------------------------------------------------------------
#  Running one job
# ----------------------------------------------------------------------------

POINT_RE = re.compile(r"^# Glitch point \d+:\s*(\[.*\])\s*$", re.M)


def parse_run_log(text):
    
    out = {"sat": "?", "alpha": "", "time": "", "maxfeat": "", "threepoints": "[]"}

    if m := re.search(r"# Error : (.*)", text):
        out["sat"] = "error"
        out["maxfeat"] = m.group(1).strip()[:80]
        return out

    if m := re.search(r"# Glitch feature : (\d+)", text):
        out["maxfeat"] = m.group(1)
        out["sat"] = "timelimit" if "time limit reached" in text else "sat"
        if a := re.search(r"# Magnitude alpha : ([\d.eE+-]+)", text):
            out["alpha"] = a.group(1)
        if t := re.search(r"# Time : ([\d.eE+-]+)\s*$", text, re.M):
            out["time"] = t.group(1)
        pts = POINT_RE.findall(text)
        if pts:
            out["threepoints"] = json.dumps(pts)
        return out

    if "# No glitch" in text:
        out["sat"] = "unsat"
        if t := re.search(r"# Time : ([\d.eE+-]+) seconds", text):
            out["time"] = t.group(1)
        return out

    if m := re.search(r"Timeout on .* after ([\d.eE+-]+) seconds", text):
        out["sat"] = "timelimit"
        out["time"] = m.group(1)
        return out

    return out


def write_results_csv(path, job, parsed):
    
    row = {
        "problem": job["problem"],
        "feature": job["features"][0] if job.get("features") else "",
        "givenalpha": job["alpha"],
        "alpha": parsed["alpha"],
        "time": parsed["time"],
        "maxfeat": parsed["maxfeat"],
        "modelname": job["modelname"],
        "modeltype": job["modeltype"],
        "sat": parsed["sat"],
        "prob": "",
        "threepoints": parsed["threepoints"],
    }
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def build_command(job, timelimit, one_thread):
    cmd = [PYTHON, MAIN_PY, job["modelpath"], "--spec", "glitch","--solver", "milp",
           "--alpha", str(job["alpha"]),
           "--timeout", str(timelimit),
           "--verbosity", "0"]
    if job.get("features"):
        cmd += ["--features"] + [str(f) for f in job["features"]]
    if one_thread:
        cmd += ["--n_jobs", "2"]
    return cmd


def resolve_timelimit(timelimit, job):
    """timelimit is either one number for every job, or {problem: seconds}."""
    if isinstance(timelimit, dict):
        return int(timelimit[job["problem"]])
    return int(timelimit)


def run_job(job, timelimit, outputroot, skip_if_done=True, one_thread=True, wall_slack=300):
    timelimit = resolve_timelimit(timelimit, job)
    outdir = os.path.join(outputroot, job["tag"])
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, "results.csv")
    if skip_if_done and os.path.isfile(csv_path):
        return {"tag": job["tag"], "returncode": 0, "wall_time": 0.0,
                "outdir": outdir, "skipped": True}

    cmd = build_command(job, timelimit, one_thread)
    logpath = os.path.join(outdir, "run.log")
    t0 = time.time()
    try:
        with open(logpath, "w") as f:
            f.write("$ " + " ".join(cmd) + "\n\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                timeout=timelimit + wall_slack).returncode
    except subprocess.TimeoutExpired:
        rc = -9
    wall = time.time() - t0

    with open(logpath, errors="replace") as f:
        text = f.read()
    parsed = parse_run_log(text)
    if rc == -9 and parsed["sat"] == "?":
        parsed["sat"] = "timelimit"
    write_results_csv(csv_path, job, parsed)

    return {"tag": job["tag"], "returncode": rc, "wall_time": wall,
            "outdir": outdir, "sat": parsed["sat"]}


def run_all(jobs, timelimit, outputroot, max_workers=2, skip_if_done=True):
    os.makedirs(outputroot, exist_ok=True)
    one_thread = max_workers > 1
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_job, j, timelimit, outputroot, skip_if_done, one_thread): j
                for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            j = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"tag": j["tag"], "returncode": -1, "wall_time": None, "error": str(e)}
            if r.get("skipped"):
                status = "SKIP(already done)"
            elif r.get("returncode") == 0:
                status = f"OK/{r.get('sat')}"
            else:
                status = f"FAIL(rc={r.get('returncode')})"
            print(f"[{status}] {r['tag']}  ({r.get('wall_time') or 0:.1f}s)", flush=True)
            results.append(r)
    return results


# ----------------------------------------------------------------------------
#  Job builders
# ----------------------------------------------------------------------------

def build_problem1_jobs(configs, alpha=1, topk_overrides=None):
    """Problem 1: TE_GLITCH(alpha, i) -- one job per eligible feature per config."""
    topk_overrides = topk_overrides or {}
    jobs = []
    for name, cfg, path in configs:
        k = topk_overrides.get(name)
        restrict = name in TOP_K_RESTRICTED or k is not None
        feats = valid_features(path, restrict_topk=restrict, top_k=k or TOP_K)
        for feat in feats:
            jobs.append({
                "modelname": name, "modeltype": cfg, "modelpath": path,
                "features": [feat], "alpha": alpha, "problem": 1,
                "tag": f"p1_{name}_{cfg}_f{feat}",
            })
    return jobs


def build_problem2_jobs(configs, alphas=ALPHA_THRESHOLDS):
    """Problem 2: TE_GLITCH(alpha) -- any feature, one job per (config, alpha)."""
    jobs = []
    for name, cfg, path in configs:
        for a in alphas:
            jobs.append({
                "modelname": name, "modeltype": cfg, "modelpath": path,
                "features": None, "alpha": a, "problem": 2,
                "tag": f"p2_{name}_{cfg}_a{a}",
            })
    return jobs


def build_problem3_jobs(configs):
    """Problem 3: TE_GLITCH (max) -- any feature, maximise alpha, one job per config."""
    return [{
        "modelname": name, "modeltype": cfg, "modelpath": path,
        "features": None, "alpha": 0, "problem": 3,
        "tag": f"p3_{name}_{cfg}",
    } for name, cfg, path in configs]


def report_missing(missing):
    if not missing:
        return
    print(f"\n{len(missing)} config(s) not on disk -- they will show as '--' in the tables:")
    for name, cfg, path in missing:
        print(f"  {name:<16}{cfg:<10}{os.path.relpath(path, REPO_ROOT)}")
    print()
