#!/usr/bin/env python3
"""Parallelized version of run_test_pca.py using multiprocessing.

Usage identical to run_test_pca.py, with an extra --workers flag:
    python3 run_test_pca_parallel.py --workers 64 --modelname adult --modeltype t500_d6
"""

import argparse
import ast
import atexit
import itertools
import json
import multiprocessing
import os
import re
import subprocess
import tempfile
import traceback
import collections
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from run_test import benchmarks as DEFAULT_BENCHMARKS


modelTrees = {
    "binary_mnist": "1000",
    "breast_cancer": "0004",
    "cod-rna": "0080",
    "covtype": "0080",
    "diabetes": "0020",
    "fashion": "0200",
    "higgs": "0300",
    "ijcnn": "0060",
    "ori_mnist": "0200",
    "webspam": "0100",
}

modelFeature = {
    "adult": 15,
    "churn": 21,
    "pimadiabetes": 9,
    "spambase": 58,
    "winequality_red": 11,
    "iris": 4,
    "german_credit": 20,
}

multimodel = ["covtype", "fashion", "ori_mnist"]


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

DEFAULT_OUTPUT_GAPS = ["0.4,0.6", "0.3,0.7"]

VARIANT_SPECS = [
    {
        "name": "plain",
        "label": "no_prob_no_clause_no_pca",
        "use_prob": False,
        "use_clause": False,
        "use_pca": False,
    },
    {
        "name": "prob",
        "label": "prob",
        "use_prob": True,
        "use_clause": False,
        "use_pca": False,
    },
    {
        "name": "clause",
        "label": "clause",
        "use_prob": False,
        "use_clause": True,
        "use_pca": False,
    },
    {
        "name": "pca_only",
        "label": "pca_only",
        "use_prob": False,
        "use_clause": False,
        "use_pca": True,
    },
    {
        "name": "pca_prob",
        "label": "pca_prob",
        "use_prob": True,
        "use_clause": False,
        "use_pca": True,
    },
    {
        "name": "pca_clause",
        "label": "pca_clause",
        "use_prob": False,
        "use_clause": True,
        "use_pca": True,
    },
    {
        "name": "pca_prob_clause",
        "label": "pca_prob_clause",
        "use_prob": True,
        "use_clause": True,
        "use_pca": True,
    },
]


def arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run MILP witness-search variants with/without PCA, probability objective, "
            "and learned clauses, then store raw metrics and pairwise comparisons. "
            "(Parallelized with multiprocessing.)"
        )
    )
    parser.add_argument("--modelname", default="", help="model name")
    parser.add_argument("--modeltype", default="", help="model type")
    parser.add_argument("--no_run", action="store_true", help="only print generated commands")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N feature/output-gap cases",
    )
    parser.add_argument("--show-output", action="store_true", help="print solver stdout/stderr")
    parser.add_argument(
        "--feature",
        type=int,
        default=None,
        help="run only one feature index",
    )
    parser.add_argument(
        "--output-gaps",
        nargs="+",
        default=DEFAULT_OUTPUT_GAPS,
        help="space-separated output-gap pairs such as 0.4,0.6 0.3,0.7",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="number of parallel worker processes (default: 64)",
    )
    return parser.parse_args()


def parse_output_gap_pairs(raw_pairs):
    gap_pairs = []
    for raw in raw_pairs:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) != 2:
            raise SystemExit(
                f"Invalid --output-gaps entry '{raw}'. Expected values like 0.4,0.6"
            )
        lgap = float(parts[0])
        ugap = float(parts[1])
        gap_pairs.append((lgap, ugap))
    return gap_pairs


def clean_output(text):
    return ANSI_ESCAPE_RE.sub("", text)


def parse_float_list(raw_text):
    tokens = raw_text.replace(",", " ").split()
    return np.array([float(token) for token in tokens], dtype=float)


def parse_bracket_array(stdout, prefix):
    cleaned = clean_output(stdout)
    pattern = rf"{re.escape(prefix)}\s*\[(.*?)\]"
    match = re.search(pattern, cleaned, re.S)
    if not match:
        return None
    return parse_float_list(match.group(1))


def parse_equals_array(stdout, prefix):
    cleaned = clean_output(stdout)
    pattern = rf"{re.escape(prefix)}\s*=\s*(\[[^\]]*\])"
    match = re.search(pattern, cleaned, re.S)
    if not match:
        return None
    return np.array(ast.literal_eval(match.group(1)), dtype=float)


def point_norms_to_centroid(point, centroid):
    diff = point - centroid
    return {
        "linf": float(np.max(np.abs(diff))),
        "l1": float(np.sum(np.abs(diff))),
        "l2": float(np.linalg.norm(diff)),
    }


_CENTROID_CACHE = {}


def _normalized_name(value):
    if pd.isna(value):
        return ""
    text = str(value)
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def dataset_metadata(data_file, detail_file):
    cache_key = (ROOT_DIR / data_file, ROOT_DIR / detail_file)
    if cache_key in _CENTROID_CACHE:
        return _CENTROID_CACHE[cache_key]

    data_path = ROOT_DIR / data_file
    detail_path = ROOT_DIR / detail_file

    df = pd.read_csv(data_path)
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    dataset_columns = [_normalized_name(col) for col in df.columns.tolist()]
    centroid = df.mean(axis=0, numeric_only=True).to_numpy(dtype=float)

    details_df = pd.read_csv(detail_path)
    detail_name_to_feature = {}
    for _, row in details_df.iterrows():
        name_key = _normalized_name(row["name"])
        feature_idx = int(row["feature"])
        if name_key:
            detail_name_to_feature[name_key] = feature_idx
        detail_name_to_feature[str(feature_idx)] = feature_idx
        detail_name_to_feature[f"f{feature_idx}"] = feature_idx

    dataset_to_model_indices = []
    missing_cols = []
    for col_name in dataset_columns:
        if col_name not in detail_name_to_feature:
            missing_cols.append(col_name)
            continue
        dataset_to_model_indices.append(detail_name_to_feature[col_name])

    if missing_cols:
        raise ValueError(
            f"Could not map dataset columns {missing_cols} through detail file {detail_file}"
        )

    metadata = {
        "centroid": centroid,
        "dataset_columns": dataset_columns,
        "dataset_to_model_indices": dataset_to_model_indices,
    }
    _CENTROID_CACHE[cache_key] = metadata
    return metadata


_PCA_TEMP_FILES = []


def _cleanup_pca_temp_files():
    for path in _PCA_TEMP_FILES:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(_cleanup_pca_temp_files)


def prepare_pca_data_file(data_file, detail_file):
    """Create a PCA-compatible data CSV with columns renamed from human-readable
    names (e.g. 'age', 'workclass') to model-index names ('f0', 'f1', ...).

    The milp.py PCA code expects feature names to match either 'f<idx>' or
    plain digit strings.  Datasets in the modelFeature group (adult, churn, etc.)
    have human-readable column headers, so we use details.csv to build the
    name -> f<feature_index> mapping and write a temp file with renamed columns.

    If the columns are already in f<idx> format, the original path is returned
    unchanged (no temp file created).
    """
    data_path = ROOT_DIR / data_file
    detail_path = ROOT_DIR / detail_file

    with open(data_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    orig_cols = header_line.split(",")

    # Check if columns already look like f0, f1, ... (skip label)
    non_label_cols = [c.strip() for c in orig_cols if c.strip() != "label"]
    already_indexed = all(
        (c.startswith("f") and c[1:].isdigit()) or c.isdigit()
        for c in non_label_cols
    )
    if already_indexed:
        return data_file  # no renaming needed

    # Build name -> f<feature_index> mapping from details.csv
    details_df = pd.read_csv(detail_path)
    name_to_fidx = {}
    for _, row in details_df.iterrows():
        name_key = str(row["name"]).strip()
        feature_idx = int(row["feature"])
        name_to_fidx[name_key] = f"f{feature_idx}"

    # Build new header string
    new_cols = []
    for col in orig_cols:
        col_stripped = col.strip()
        if col_stripped == "label":
            new_cols.append(col_stripped)
        elif col_stripped in name_to_fidx:
            new_cols.append(name_to_fidx[col_stripped])
        else:
            # Fallback: if column name is itself a valid index, keep it
            if (col_stripped.startswith("f") and col_stripped[1:].isdigit()) or col_stripped.isdigit():
                new_cols.append(col_stripped)
            else:
                raise ValueError(
                    f"Cannot map column '{col_stripped}' to a feature index via {detail_file}"
                )

    new_header_line = ",".join(new_cols) + "\n"

    # Write to a temp file next to the original data file
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".csv",
        prefix=f"pca_{data_path.stem}_",
        dir=str(data_path.parent),
    )
    
    import shutil
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f_out:
        with open(data_path, "r", encoding="utf-8") as f_in:
            f_in.readline()  # skip original header
            f_out.write(new_header_line)
            shutil.copyfileobj(f_in, f_out)

    _PCA_TEMP_FILES.append(tmp_path)
    return str(Path(tmp_path).relative_to(ROOT_DIR))


def resolve_clause_file(modelname, modeltype):
    candidates = [
        ROOT_DIR / "outputs" / "output" / f"learned-clauses_{modelname}_{modeltype}.txt",
        ROOT_DIR / "outputs" / "outputs_old" / f"learned-clauses_{modelname}_{modeltype}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.relative_to(ROOT_DIR))
    return str(candidates[0].relative_to(ROOT_DIR))


def feature_choices_for_dataset(modelname, modeltype, benchmark_root):
    if modelname == "sm":
        return list(range(0, 20))
    if modelname in multimodel:
        feat_path = (
            benchmark_root
            / "tree_verification_models"
            / f"{modelname}_{modeltype}"
            / "feat_imp.json"
        )
        with open(feat_path, "r", encoding="utf-8") as handle:
            feat_file = json.load(handle)
        return [int(name[1:]) for name in list(feat_file.keys())[:100]]
    if modelname in modelTrees:
        details_path = benchmark_root / "dataset" / modelname / f"{modelname}_details.csv"
        no_feature = pd.read_csv(details_path, index_col=0).shape[0]
        return list(range(1, no_feature))
    if modelname in modelFeature:
        return list(range(0, modelFeature[modelname]))
    raise SystemExit(f"Could not determine feature list for {modelname}_{modeltype}")


def resolve_model_files(modelname, modeltype):
    benchmark_root = ROOT_DIR / "models"

    if modelname in modelTrees:
        trees = modelTrees[modelname]
        model_file = (
            benchmark_root
            / "tree_verification_models"
            / f"{modelname}_{modeltype}"
            / f"{trees}.resaved.json"
        )
        detail_file = benchmark_root / "dataset" / modelname / f"{modelname}_details.csv"
        data_file = benchmark_root / "dataset" / modelname / f"{modelname}_train.csv"
    else:
        model_file = benchmark_root / modelname / f"{modelname}_{modeltype}.json"
        detail_name = "details.csv" if modelname in modelFeature else f"{modelname}_details.csv"
        data_name = "train.csv" if modelname in modelFeature else f"{modelname}_train.csv"
        detail_file = benchmark_root / "dataset" / modelname / detail_name
        data_file = benchmark_root / "dataset" / modelname / data_name

    dataset_key = f"{modelname}_{modeltype}"
    log_file = CURRENT_DIR / "failed_datasets.log"
    
    def log_skip(msg):
        print(f"Warning: {msg}")
        with open(log_file, "a") as f:
            f.write(f"[{dataset_key}] {msg}\n")

    if not model_file.exists():
        log_skip(f"Missing model file: {model_file}")
        return None
    if not detail_file.exists():
        log_skip(f"Missing detail file: {detail_file}")
        return None
    if not data_file.exists():
        log_skip(f"Missing data file: {data_file}")
        return None

    try:
        featlist = feature_choices_for_dataset(modelname, modeltype, benchmark_root)
    except Exception as e:
        log_skip(f"Failed to determine feature list: {e}")
        return None
    clause_file = resolve_clause_file(modelname, modeltype)
    multi = modelname in multimodel

    data_file_rel = str(data_file.relative_to(ROOT_DIR))
    detail_file_rel = str(detail_file.relative_to(ROOT_DIR))

    # Create a PCA-compatible data file with columns renamed to f<idx> format
    pca_data_file = prepare_pca_data_file(data_file_rel, detail_file_rel)

    return (
        str(model_file.relative_to(ROOT_DIR)),
        featlist,
        detail_file_rel,
        data_file_rel,
        clause_file,
        multi,
        pca_data_file,
    )


def command_parts_for_variant(
    model_file,
    detail_file,
    data_file,
    clause_file,
    feature,
    lgap,
    ugap,
    variant,
    multi,
    pca_data_file="",
):
    parts = [
        "python3",
        "./src/sensitive.py",
        model_file,
        "--features",
        str(feature),
        "--output_gap",
        str(lgap),
        str(ugap),
        "--details",
        detail_file,
        "--all_opt",
        "--verbosity",
        "3",
        "--solver=milp",
    ]

    if multi:
        parts.extend(["--multiclass", "--truelabel", "1", "--otherlabel", "0"])

    if variant["use_prob"]:
        parts.extend(["--prob", "--data_file", data_file])

    if variant["use_clause"]:
        parts.extend(["--in_distro_clauses", clause_file])

    if variant["use_pca"]:
        # Use the PCA-compatible data file (columns renamed to f<idx> format)
        pca_file = pca_data_file if pca_data_file else data_file
        parts.extend(["--pca_data", pca_file, "--pca_d"])

    return parts


def experiment_cases(
    modelname,
    modeltype,
    model_file,
    featlist,
    detail_file,
    data_file,
    clause_file,
    gap_pairs,
    selected_feature=None,
    multi=False,
    pca_data_file="",
):
    if selected_feature is not None:
        available = {int(feature) for feature in featlist}
        if selected_feature not in available:
            raise SystemExit(
                f"Feature {selected_feature} is not available for {modelname}_{modeltype}"
            )
        feature_choices = [selected_feature]
    else:
        feature_choices = [int(feature) for feature in featlist]

    cases = []
    for feature, (lgap, ugap) in itertools.product(feature_choices, gap_pairs):
        case = {
            "feature": int(feature),
            "lgap": float(lgap),
            "ugap": float(ugap),
            "variants": [],
        }
        for variant in VARIANT_SPECS:
            parts = command_parts_for_variant(
                model_file=model_file,
                detail_file=detail_file,
                data_file=data_file,
                clause_file=clause_file,
                feature=feature,
                lgap=lgap,
                ugap=ugap,
                variant=variant,
                multi=multi,
                pca_data_file=pca_data_file,
            )
            case["variants"].append(
                {
                    **variant,
                    "command_parts": parts,
                    "command": " ".join(parts),
                }
            )
        cases.append(case)
    return cases


def run_command(command_parts, show_output=False):
    result = subprocess.run(
        command_parts,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
    )
    if show_output:
        print(f"[RUN] {' '.join(command_parts)}")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    return result


def extract_solution_points(stdout, prefer_pca_points, dataset_to_model_indices):
    point1 = None
    point2 = None
    point_source = "sensitive_samples"

    if prefer_pca_points:
        point1 = parse_equals_array(stdout, "[PCA] milp x (dataset dims)")
        point2 = parse_equals_array(stdout, "[PCA] milp x2 (dataset dims)")
        if point1 is not None and point2 is not None:
            point_source = "pca_milp_points"

    if point1 is None or point2 is None:
        point1 = parse_bracket_array(stdout, "Sensitive sample 1:")
        point2 = parse_bracket_array(stdout, "Sensitive sample 2:")
        if point1 is None or point2 is None:
            raise ValueError("Could not parse witness points from solver output")
        point1 = np.array([point1[idx] for idx in dataset_to_model_indices], dtype=float)
        point2 = np.array([point2[idx] for idx in dataset_to_model_indices], dtype=float)

    return point1, point2, point_source


def compute_metrics(point1, point2, centroid):
    point1_norms = point_norms_to_centroid(point1, centroid)
    point2_norms = point_norms_to_centroid(point2, centroid)
    return {
        "pair_l2": float(np.linalg.norm(point1 - point2)),
        "point1": point1_norms,
        "point2": point2_norms,
        "mean_centroid_linf": float((point1_norms["linf"] + point2_norms["linf"]) / 2.0),
        "mean_centroid_l1": float((point1_norms["l1"] + point2_norms["l1"]) / 2.0),
        "mean_centroid_l2": float((point1_norms["l2"] + point2_norms["l2"]) / 2.0),
    }


def variant_row(dataset_key, modelname, modeltype, case, variant, clause_file, result):
    return {
        "dataset": modelname,
        "modeltype": modeltype,
        "dataset_key": dataset_key,
        "feature": case["feature"],
        "lgap": case["lgap"],
        "ugap": case["ugap"],
        "variant": variant["name"],
        "variant_label": variant["label"],
        "use_prob": variant["use_prob"],
        "use_clause": variant["use_clause"],
        "use_pca": variant["use_pca"],
        "clause_file": clause_file if variant["use_clause"] else "",
        "command": variant["command"],
        "rc": result.returncode,
        "status": "ok",
        "error": "",
    }


def print_case_header(dataset_key, case):
    print(
        f"Running {dataset_key} | feature={case['feature']} | "
        f"gap=({case['lgap']}, {case['ugap']})"
    )


def print_variant_metrics(dataset_key, case, variant_name, metrics, printer=print):
    printer(
        f"{dataset_key} | feature={case['feature']} | gap=({case['lgap']}, {case['ugap']}) "
        f"| variant={variant_name}"
    )
    printer(f"  Pairwise L2 distance: {metrics['pair_l2']}")
    printer(f"  Point1 L1/L2 to centroid: {metrics['point1']['l1']} / {metrics['point1']['l2']}")
    printer(f"  Point2 L1/L2 to centroid: {metrics['point2']['l1']} / {metrics['point2']['l2']}")
    printer(
        f"  Mean centroid L1/L2: {metrics['mean_centroid_l1']} / {metrics['mean_centroid_l2']}"
    )


def build_pairwise_rows(raw_df):
    ok_df = raw_df[raw_df["status"] == "ok"].copy()
    pairwise_rows = []
    case_cols = ["dataset", "modeltype", "dataset_key", "feature", "lgap", "ugap"]

    for case_key, case_df in ok_df.groupby(case_cols, dropna=False):
        records = case_df.to_dict("records")
        for left, right in itertools.combinations(records, 2):
            pairwise_rows.append(
                {
                    "dataset": case_key[0],
                    "modeltype": case_key[1],
                    "dataset_key": case_key[2],
                    "feature": case_key[3],
                    "lgap": case_key[4],
                    "ugap": case_key[5],
                    "lhs_variant": left["variant"],
                    "rhs_variant": right["variant"],
                    "lhs_pair_l2": left["pair_l2"],
                    "rhs_pair_l2": right["pair_l2"],
                    "winner_pair_l2": (
                        left["variant"]
                        if left["pair_l2"] < right["pair_l2"]
                        else right["variant"]
                        if right["pair_l2"] < left["pair_l2"]
                        else "tie"
                    ),
                    "lhs_mean_centroid_l1": left["mean_centroid_l1"],
                    "rhs_mean_centroid_l1": right["mean_centroid_l1"],
                    "winner_mean_centroid_l1": (
                        left["variant"]
                        if left["mean_centroid_l1"] < right["mean_centroid_l1"]
                        else right["variant"]
                        if right["mean_centroid_l1"] < left["mean_centroid_l1"]
                        else "tie"
                    ),
                    "lhs_mean_centroid_l2": left["mean_centroid_l2"],
                    "rhs_mean_centroid_l2": right["mean_centroid_l2"],
                    "winner_mean_centroid_l2": (
                        left["variant"]
                        if left["mean_centroid_l2"] < right["mean_centroid_l2"]
                        else right["variant"]
                        if right["mean_centroid_l2"] < left["mean_centroid_l2"]
                        else "tie"
                    ),
                }
            )
    return pd.DataFrame(pairwise_rows)


def dataset_summary(dataset_key, rows):
    df = pd.DataFrame(rows)
    print(dataset_key)
    if df.empty:
        print("  No rows collected")
        return
    for variant_name, group in df.groupby("variant"):
        ok_count = int((group["status"] == "ok").sum())
        total_count = len(group)
        print(f"  {variant_name}: ok={ok_count}/{total_count}")


# ---------------------------------------------------------------------------
# Worker function — runs a single solver variant in a child process.
# Must be at module level so multiprocessing can pickle it.
# ---------------------------------------------------------------------------

def _worker_run_variant(task):
    """Execute one solver variant and return the result row dict.

    Designed to be called via ``multiprocessing.Pool.map``.  Every piece of
    information the worker needs is passed through the *task* dict so that
    no shared mutable state is required.
    """
    # --- base fields present in every result row ---
    row = {
        "dataset": task["modelname"],
        "modeltype": task["modeltype"],
        "dataset_key": task["dataset_key"],
        "feature": task["feature"],
        "lgap": task["lgap"],
        "ugap": task["ugap"],
        "variant": task["variant_name"],
        "variant_label": task["variant_label"],
        "use_prob": task["use_prob"],
        "use_clause": task["use_clause"],
        "use_pca": task["use_pca"],
        "clause_file": task["clause_file_rel"] if task["use_clause"] else "",
        "command": task["command"],
        "rc": None,
        "status": "ok",
        "error": "",
    }

    # --- skip if clause file is required but missing ---
    if task["use_clause"] and not Path(task["clause_path"]).exists():
        row["status"] = "missing_clause_file"
        row["error"] = f"Missing clause file: {task['clause_path']}"
        return row

    # --- run the solver subprocess ---
    res = subprocess.run(
        task["command_parts"],
        cwd=task["root_dir"],
        capture_output=True,
        text=True,
    )
    row["rc"] = res.returncode
    row["stdout"] = res.stdout
    row["stderr"] = res.stderr

    if res.returncode != 0:
        row["status"] = "command_failed"
        row["error"] = f"returncode={res.returncode}"
        return row

    # --- parse witness points and compute metrics ---
    try:
        centroid = np.array(task["centroid"])
        d2m = task["dataset_to_model_indices"]

        point1, point2, point_source = extract_solution_points(
            res.stdout,
            prefer_pca_points=task["use_pca"],
            dataset_to_model_indices=d2m,
        )
        if len(point1) != len(point2) or len(point1) != len(centroid):
            raise ValueError(
                "Dimension mismatch: "
                f"point1={len(point1)} point2={len(point2)} centroid={len(centroid)}"
            )
        metrics = compute_metrics(point1, point2, centroid)
    except Exception as exc:
        row["status"] = "parse_failed"
        row["error"] = str(exc)
        return row

    row.update(
        {
            "point_source": point_source,
            "point1_json": json.dumps(point1.tolist()),
            "point2_json": json.dumps(point2.tolist()),
            "pair_l2": metrics["pair_l2"],
            "point1_linf": metrics["point1"]["linf"],
            "point1_l1": metrics["point1"]["l1"],
            "point1_l2": metrics["point1"]["l2"],
            "point2_linf": metrics["point2"]["linf"],
            "point2_l1": metrics["point2"]["l1"],
            "point2_l2": metrics["point2"]["l2"],
            "mean_centroid_linf": metrics["mean_centroid_linf"],
            "mean_centroid_l1": metrics["mean_centroid_l1"],
            "mean_centroid_l2": metrics["mean_centroid_l2"],
        }
    )
    return row


# ---------------------------------------------------------------------------
# Helper to reconstruct the nested metrics dict for print_variant_metrics
# ---------------------------------------------------------------------------

def _metrics_from_row(row):
    return {
        "pair_l2": row["pair_l2"],
        "point1": {"linf": row["point1_linf"], "l1": row["point1_l1"], "l2": row["point1_l2"]},
        "point2": {"linf": row["point2_linf"], "l1": row["point2_l1"], "l2": row["point2_l2"]},
        "mean_centroid_linf": row["mean_centroid_linf"],
        "mean_centroid_l1": row["mean_centroid_l1"],
        "mean_centroid_l2": row["mean_centroid_l2"],
    }


def main():
    args = arguments()
    gap_pairs = parse_output_gap_pairs(args.output_gaps)
    benchmarks = list(DEFAULT_BENCHMARKS)
    num_workers = max(1, args.workers)

    if args.modelname == "" and args.modeltype == "":
        print("Running full PCA benchmark")
    elif args.modelname != "" and args.modeltype != "":
        benchmarks = [(args.modelname, args.modeltype)]
    else:
        raise SystemExit("Error: missing modelname or modeltype")

    print(f"Using {num_workers} parallel workers")

    # ---- Phase 1: Collect tasks for all datasets ----
    all_tasks = []
    dataset_cases = [] # store cases per dataset to reconstruct later

    for modelname, modeltype in benchmarks:
        dataset_key = f"{modelname}_{modeltype}"
        resolved = resolve_model_files(modelname, modeltype)
        if resolved is None:
            print(f"Skipping {dataset_key} due to missing files.\n")
            continue
        model_file, featlist, detail_file, data_file, clause_file, multi, pca_data_file = resolved
        
        metadata = dataset_metadata(data_file, detail_file)
        centroid = metadata["centroid"]
        dataset_to_model_indices = metadata["dataset_to_model_indices"]
        cases = experiment_cases(
            modelname=modelname,
            modeltype=modeltype,
            model_file=model_file,
            featlist=featlist,
            detail_file=detail_file,
            data_file=data_file,
            clause_file=clause_file,
            gap_pairs=gap_pairs,
            selected_feature=args.feature,
            multi=multi,
            pca_data_file=pca_data_file,
        )
        if args.limit is not None:
            cases = cases[: args.limit]

        dataset_cases.append({
            "dataset_key": dataset_key,
            "cases": cases,
            "modelname": modelname,
            "modeltype": modeltype,
            "clause_file": clause_file
        })

        if args.no_run:
            for case in cases:
                print_case_header(dataset_key, case)
                for variant in case["variants"]:
                    print(f"[{variant['name']}] {variant['command']}")
                print()
            continue

        for case in cases:
            for variant in case["variants"]:
                all_tasks.append({
                    "modelname": modelname,
                    "modeltype": modeltype,
                    "dataset_key": dataset_key,
                    "feature": case["feature"],
                    "lgap": case["lgap"],
                    "ugap": case["ugap"],
                    "variant_name": variant["name"],
                    "variant_label": variant["label"],
                    "use_prob": variant["use_prob"],
                    "use_clause": variant["use_clause"],
                    "use_pca": variant["use_pca"],
                    "command_parts": variant["command_parts"],
                    "command": variant["command"],
                    "clause_file_rel": clause_file,
                    "clause_path": str(ROOT_DIR / clause_file),
                    "root_dir": str(ROOT_DIR),
                    "centroid": centroid.tolist(),
                    "dataset_to_model_indices": dataset_to_model_indices,
                })

    if args.no_run:
        return

    raw_output = CURRENT_DIR / "run_test_pca_results_long.csv"
    if raw_output.exists():
        raw_output.unlink()

    # ---- Phase 2 & 3: Run and Stream Results in Parallel ----
    effective_workers = min(num_workers, max(1, len(all_tasks)))
    print(f"Submitting {len(all_tasks)} total tasks "
          f"across {effective_workers} workers ...\n")

    all_results = []
    if all_tasks:
        with multiprocessing.Pool(processes=effective_workers) as pool:
            for row in tqdm(pool.imap_unordered(_worker_run_variant, all_tasks), total=len(all_tasks), desc="Running PCA Tests"):
                all_results.append(row)
                
                # Immediately append to CSV
                pd.DataFrame([row]).to_csv(raw_output, mode='a', header=not raw_output.exists(), index=False)
                
                # Live console output using tqdm.write to preserve progress bar
                tqdm.write(f"[Done] {row['dataset_key']} | feat={row['feature']} | gap=({row['lgap']},{row['ugap']}) | var={row['variant']}")
                
                if args.show_output:
                    if row.get("stdout"): tqdm.write(row["stdout"])
                    if row.get("stderr"): tqdm.write(row["stderr"])

                if row["status"] == "missing_clause_file":
                    tqdm.write(f"    skipped: {row['error']}\n")
                elif row["status"] == "command_failed":
                    tqdm.write(f"    failed: {row['error']}\n")
                elif row["status"] == "parse_failed":
                    tqdm.write(f"    parse failed: {row['error']}\n")
                elif row["status"] == "ok":
                    case_mock = {"feature": row["feature"], "lgap": row["lgap"], "ugap": row["ugap"]}
                    print_variant_metrics(row["dataset_key"], case_mock, row["variant"], _metrics_from_row(row), printer=tqdm.write)
                    tqdm.write("")

    # ---- Phase 4: Write CSVs ----
    raw_df = pd.DataFrame(all_results)
    print(f"Raw results incrementally saved to {raw_output}")

    pairwise_df = build_pairwise_rows(raw_df)
    pairwise_output = CURRENT_DIR / "run_test_pca_pairwise.csv"
    pairwise_df.to_csv(pairwise_output, index=False)
    print(f"Pairwise comparison results saved to {pairwise_output}")


if __name__ == "__main__":
    main()
