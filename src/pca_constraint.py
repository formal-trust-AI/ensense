#!/usr/bin/env python3
"""
pca_constraint.py
-----------------
Loads a dataset CSV, fits PCA (n → d dims), computes the projection matrix
P = V(VᵀV)⁻¹Vᵀ in closed form, then:

  1. Computes ‖x - Px‖_∞  (L-infinity norm) for every training sample.
  2. Sets  epsilon = max_{x in X} ‖x - Px‖_∞  (worst-case training residual).
  3. Checks the hardcoded test point y:  ‖y - Py‖_∞ < epsilon

Usage
-----
    python3 utils/pca_constraint.py \
        --csv    models/dataset/diabetes/diabetes_train.csv \
        --d      3 \
        [--label_col label] \
        [--no_header] \
        [--center] \
        [--verbose]

    *** Edit the Y_TEST variable below to set your test point y. ***

Arguments
---------
  --csv        Path to the training dataset CSV (used to fit PCA & set epsilon)
  --d          Number of PCA components (target dimension)
  --label_col  Column name to drop as label (default: 'label'). Use '' to skip.
  --no_header  Flag: CSV has no header row (columns named f0, f1, ...)
  --center     Flag: center data (subtract training means) before PCA
  --verbose    Flag: print per-sample residuals
"""

# ===========================================================================
# ★  EDIT YOUR TEST POINT y HERE  ★
# Must have exactly p values (one per feature, after label column is dropped).
# Example below uses 8 values for the diabetes dataset (f1..f8).
# ===========================================================================
Y_TEST = [0.35, 0.74, 0.59, 0.35, 0.0, 0.50, 0.23, 0.48]
# ===========================================================================

import argparse
import sys
import numpy as np
import pandas as pd
from requests import options
import z3
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PCA + projection matrix
# ---------------------------------------------------------------------------

def auto_select_d(S: np.ndarray) -> int:
    """
    Automatically select number of PCA components d by detecting the **elbow point**.

    Strategy (in priority order):
      1. Detect a **drastic drop** in eigenvalues: if S[d] / S[d+1] > 10.0
         (a sudden cliff), stop at d.
      2. If no drastic drop found, use the minimum d where explained variance 
         >= 0.95.
      3. Fallback: use all components.

    Parameters
    ----------
    S                           : array of singular values (sorted descending)

    Returns
    -------
    d : int
        Recommended number of components.
    """
    eigenvalue_ratio_threshold = 10.0  # Hardcoded cliff detection threshold
    variance_threshold = 0.95          # Hardcoded variance threshold
    p = len(S)
    print("length of S is ", p)
    total_var = (S ** 2).sum()

    # Strategy 1: Detect drastic drop (elbow point with large ratio)
    # E.g., if first 4 eigenvalues are large and 5th is suddenly tiny
    for d in range(1, p - 1):
        ratio = S[d - 1] / S[d]
        if ratio > eigenvalue_ratio_threshold:
            return d

    # Strategy 2: Fall back to variance threshold if no drastic drop
    cumsum_var = np.cumsum(S ** 2)
    for d in range(1, p + 1):
        explained = cumsum_var[d - 1] / total_var
        if explained >= variance_threshold:
            return d

    # Fallback: use all components
    return p


def fit_pca(X: np.ndarray):
    """
    Fit PCA on X (n_samples × n_features, already centered if desired).
    Returns V: (n_features × d) matrix whose columns are the top-d eigenvectors.

    Uses SVD of X:  X = U Σ Vᵀ  →  V columns are right singular vectors.

    Parameters
    ----------
    X                           : (n, p) data matrix
    d                           : int; if 0, auto-select based on eigenvalue elbow

    Returns
    -------
    V       : (p, d_selected) matrix of eigenvectors
    S       : array of singular values (all p values)
    explained : float, fraction of variance explained by selected d
    d_selected : int, the d that was actually used (useful if d=0 was passed)
    """
    n, p = X.shape

    # Economy SVD: X = U S Vt,  Vt is (min(n,p) × p)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    d = auto_select_d(S)
    print(f"d= {d}")
    V = Vt[:d].T          # shape (p, d) — top-d right singular vectors
    explained = (S[:d] ** 2).sum() / (S ** 2).sum()
    return V, S, explained, d


def projection_matrix(V: np.ndarray) -> np.ndarray:
    """
    Closed-form orthogonal projection onto column space of V:
        P = V (VᵀV)⁻¹ Vᵀ

    When V has orthonormal columns (as from SVD), VᵀV = I, so P = VVᵀ.
    We compute the general form anyway for correctness.
    """
    VtV = V.T @ V                          # (d × d)
    VtV_inv = np.linalg.inv(VtV)          # (d × d)
    P = V @ VtV_inv @ V.T                  # (p × p)
    return P


def residuals(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Compute ‖x - Px‖_∞  (L-infinity norm) for every row x in X.
    X: (n × p),  P: (p × p)
    Returns: (n,) array of residual L-inf norms.
    """
    diff = X - X @ P.T
    return np.max(np.abs(diff), axis=1)

# ---------------------------------------------------------------------------
# High-level API for external callers (e.g. pb.py)
# ---------------------------------------------------------------------------

def compute_pca_params(csv_path: str,
                       label_col: str = "label",
                       no_header: bool = False,
                       center: bool = False,
                       verbose: bool = False):
    """
    One-call API: load CSV → fit PCA → return everything pb.py needs.

    Parameters
    ----------
    csv_path                    : path to training CSV
    d                           : number of PCA components; if 0, auto-select via elbow
    label_col                   : column to drop as label ('' to skip)
    no_header                   : True if CSV has no header row
    center                      : subtract training means before PCA
    verbose                     : print diagnostics

    Returns
    -------
    dict with keys:
        'ImP'       : (p, p) np.ndarray   — the matrix  I − P
        'epsilon'   : float                — max_{x∈X} ‖x − Px‖_∞
        'P'         : (p, p) np.ndarray   — projection matrix
        'p'         : int                  — number of features
        'd'         : int                  — number of components used (auto-selected if d=0)
        'explained' : float                — explained-variance ratio
        'mean'      : (p,) np.ndarray      — training means (for centering)
    """
    X, feature_names = load_csv(csv_path, label_col, no_header)
    n, p = X.shape

    mean = np.zeros(p)
    if center:
        mean = X.mean(axis=0)
        X = X - mean

    V, S, explained, d_used = fit_pca(X)
    P = projection_matrix(V)
    ImP = np.eye(p) - P

    resid = residuals(X, P)
    epsilon = 1.0*float(resid.max())

    print(f"[PCA] Auto-selected d={d_used}/{p}  explained={explained:.4f}  "
              f"epsilon={epsilon:.6f}")

    return {
        'ImP':       ImP,
        'epsilon':   epsilon,
        'P':         P,
        'p':         p,
        'd':         d_used,
        'explained': explained,
        'mean':      mean,
        'data_min': X.min(axis=0),
        'data_max': X.max(axis=0),
        'feature_names': feature_names,
    }

def add_pca_linear_constraints(x_vars, ImP, mean, eps):
    p = len(mean)
    cons = []
    for i in range(p):
        expr_terms = []
        for j in range(p):
            coeff = z3.RealVal(float(ImP[i, j]))
            centered = x_vars[j] - z3.RealVal(float(mean[j]))
            expr_terms.append(coeff * centered)
        expr = z3.Sum(expr_terms)
        cons.append(expr <= eps)
        cons.append(expr >= -eps)
    return cons


def evaluate_pca_row_values(point, ImP, mean):
    values = []
    p = len(mean)
    for row in range(p):
        comp = 0.0
        for col in range(p):
            comp += float(ImP[row][col]) * (float(point[col]) - float(mean[col]))
        values.append(comp)
    return values


def gen_pca_constraints(
    options, n_features, split_bit_map,
    split_sat_value_map, split_guard_map,
    op_range_list, vars1, vars2, precision, ens
):
    if not options.pca_data:
        return []

    pca = compute_pca_params(
        csv_path=options.pca_data,
        center=True,
        verbose=(options.verbosity > 0),
    )

    ImP = pca["ImP"]
    eps = float(pca["epsilon"])
    mean = pca["mean"]
    pca_feature_names = pca["feature_names"]

    def real_val(x):
        return z3.RealVal(repr(float(x)))
    #So the function converts each column name like "f3" into model index 3. That gives a list like:[1,2,3,4,5,6,7,8]
    def model_index_from_feature_name(fname):
        if isinstance(fname, str) and fname.startswith("f"):
            return int(fname[1:])
        raise ValueError(f"Unsupported PCA feature name: {fname}")

    pca_feature_indices = []
    for fname in pca_feature_names:
        pca_feature_indices.append(model_index_from_feature_name(fname))

    def feature_thresholds(model_idx):
        thresholds = []
        for name in split_bit_map[model_idx]:
            thresholds.append(split_guard_map[name])
        return thresholds

    def max_interval_width(model_idx):
        bounds = [float(op_range_list[model_idx][0])]
        bounds.extend(float(t) for t in feature_thresholds(model_idx))
        bounds.append(float(op_range_list[model_idx][1]))
        if len(bounds) < 2:
            return 0.0
        return max(bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1))

    # Current pb.py feature bits are monotone threshold bits with semantics:
    #   bit_t = True  iff x < t
    # So a left-endpoint interval reconstruction is:
    #   x_tilde = lower + sum_i (t_i - t_{i-1}) * [not bit_i]
    # returns a linear Z3 expression for each feature, built only from If(bit, 0, width) terms
    def reconstructed_feature_expr(model_idx, var_map):
        lower = float(op_range_list[model_idx][0])
        expr_terms = [real_val(lower)]
        prev = lower
        for bit_name in split_bit_map[model_idx]:
            threshold = float(split_guard_map[bit_name])
            width = threshold - prev
            prev = threshold
            if abs(width) <= 0.0:
                continue
            expr_terms.append(z3.If(var_map[bit_name], real_val(0.0), real_val(width)))
        return z3.Sum(expr_terms)

    recon1 = []
    recon2 = []
    quant_error = []
    for model_idx in pca_feature_indices:
        recon1.append(reconstructed_feature_expr(model_idx, vars1))
        recon2.append(reconstructed_feature_expr(model_idx, vars2))
        quant_error.append(max_interval_width(model_idx))

    # Row-wise slack is safer than a single global delta:
    # if |x_j - x_tilde_j| <= delta_j, then for row r of ImP we have
    # |sum_j ImP[r,j] (x_j - x_tilde_j)| <= sum_j |ImP[r,j]| delta_j.
    row_slack = []
    for row in range(len(pca_feature_indices)):
        slack = 0.0
        for col in range(len(pca_feature_indices)):
            slack += abs(float(ImP[row][col])) * quant_error[col]
        print("slack for row ", row, " is ", slack)
        row_slack.append(0.35*slack)

    cons = []
    #expr1 = sum_j ImP[row,j] * (recon1[j] - mean[j])
    # expr2 = sum_j ImP[row,j] * (recon2[j] - mean[j])
    # expr1 <= epsilon + row_slack[row]
    # expr1 >= -(epsilon + row_slack[row])
    # expr2 <= epsilon + row_slack[row]
    # expr2 >= -(epsilon + row_slack[row])
    for row in range(len(pca_feature_indices)):
        expr1_terms = []
        expr2_terms = []
        for col in range(len(pca_feature_indices)):
            coeff = real_val(float(ImP[row][col]))
            mean_term = real_val(float(mean[col]))
            expr1_terms.append(coeff * (recon1[col] - mean_term))
            expr2_terms.append(coeff * (recon2[col] - mean_term))
        expr1 = z3.Sum(expr1_terms)
        expr2 = z3.Sum(expr2_terms)

        eps_row = real_val(eps + row_slack[row])
        cons.append(expr1 <= eps_row)
        cons.append(expr1 >= -eps_row)
        cons.append(expr2 <= eps_row)
        cons.append(expr2 >= -eps_row)

    if options.verbosity > 0:
        print(f"[PCA-bit] mapped PCA features to model indices: {pca_feature_indices}")
        print(f"[PCA-bit] added {len(cons)} linear bit-level PCA constraints")

    return cons
# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csv(path: str, label_col: str, no_header: bool) -> np.ndarray:
    if no_header:
        df = pd.read_csv(path, header=None)
        df.columns = [f"f{i}" for i in range(df.shape[1])]
    else:
        df = pd.read_csv(path)

    if label_col and label_col in df.columns:
        df = df.drop(columns=[label_col])
        # print(f"  Dropped label column '{label_col}'")
    elif label_col:
        print(f"  Warning: label column '{label_col}' not found — skipping drop")

    # Drop any remaining non-numeric columns
    non_num = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        print(f"  Dropping non-numeric columns: {non_num}")
        df = df.drop(columns=non_num)

    X = df.values.astype(float)
    return X, df.columns.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PCA projection constraint (L-inf): epsilon = max‖x-Px‖_∞, check ‖y-Py‖_∞ < epsilon"
    )
    parser.add_argument("--csv",       required=True,
                        help="Training dataset CSV (fits PCA and sets epsilon)")
    parser.add_argument("--d",         required=True, type=int,
                        help="Number of PCA components (target dimension)")
    parser.add_argument("--label_col", default="label",
                        help="Column to drop as label (default: 'label'). Use '' to skip.")
    parser.add_argument("--no_header", action="store_true",
                        help="CSV has no header row")
    parser.add_argument("--center",    action="store_true",
                        help="Center data (subtract training means) before PCA")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print per-sample residuals")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Training CSV : {csv_path}")
    print(f"  PCA d        : {args.d}")
    print(f"  Norm         : L-infinity")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    X, feature_names = load_csv(str(csv_path), args.label_col, args.no_header)
    n, p = X.shape
    print(f"Loaded  : {n} samples × {p} features")
    print(f"Features: {feature_names}\n")

    if args.d >= p:
        print(f"Warning: d={args.d} >= p={p}. Residual will be 0 (perfect projection).")

    # ------------------------------------------------------------------
    # 2. Optionally center
    # ------------------------------------------------------------------
    mean = np.zeros(p)
    if args.center:
        mean = X.mean(axis=0)
        X = X - mean
        print(f"Data centered (mean subtracted).\n")

    # ------------------------------------------------------------------
    # 3. PCA — fit top-d components via SVD
    # ------------------------------------------------------------------
    V, singular_values, explained_var = fit_pca(X, args.d)
    print(f"PCA singular values (top {args.d}): {np.round(singular_values[:args.d], 4)}")
    print(f"Explained variance ratio           : {explained_var:.4f}  ({explained_var*100:.2f}%)\n")
    print(f"V shape (p × d)                    : {V.shape}")

    # ------------------------------------------------------------------
    # 4. Projection matrix P = V(VᵀV)⁻¹Vᵀ  (closed form)
    # ------------------------------------------------------------------
    P = projection_matrix(V)
    print(f"P shape (p × p)                    : {P.shape}")

    # Sanity check: P should be idempotent (P² = P) and symmetric
    idempotent_err = np.max(np.abs(P @ P - P))
    symmetric_err  = np.max(np.abs(P - P.T))
    print(f"Idempotency check  max|P²-P|       : {idempotent_err:.2e}  (should be ~0)")
    print(f"Symmetry check     max|P-Pᵀ|       : {symmetric_err:.2e}  (should be ~0)\n")

    # ------------------------------------------------------------------
    # 5. Compute residuals ‖x - Px‖ for all samples
    # ------------------------------------------------------------------
    resid = residuals(X, P)

    max_resid  = resid.max()
    mean_resid = resid.mean()
    min_resid  = resid.min()
    argmax     = resid.argmax()

    print(f"Training residual ‖x - Px‖_∞ statistics:")
    print(f"  min  : {min_resid:.6f}")
    print(f"  mean : {mean_resid:.6f}")
    print(f"  max  : {max_resid:.6f}  (sample index {argmax})")
    print()

    if args.verbose:
        print("Per-sample residuals:")
        for i, r in enumerate(resid):
            print(f"  [{i:5d}]  {r:.6f}")
        print()

    # ------------------------------------------------------------------
    # 6. epsilon = max over training set
    # ------------------------------------------------------------------
    epsilon = max_resid
    print(f"{'='*60}")
    print(f"  epsilon = max_x ‖x - Px‖_∞ = {epsilon:.6f}")
    print(f"  (worst training sample index: {argmax})")
    x_worst = X[argmax]
    print(f"  x worst  : {np.round(x_worst, 4)}")
    print(f"  Px worst : {np.round(P @ x_worst, 4)}")
    print(f"  x - Px   : {np.round(x_worst - P @ x_worst, 4)}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 7. Use hardcoded test point y (edit Y_TEST at the top of this file)
    # ------------------------------------------------------------------
    y = np.array(Y_TEST, dtype=float)
    if len(y) != p:
        print(f"Error: Y_TEST has {len(y)} values but model has {p} features. "
              f"Edit Y_TEST at the top of pca_constraint.py.",
              file=sys.stderr)
        sys.exit(1)

    # Apply same centering as training
    y_centered = y - mean

    y_proj   = P @ y_centered                # Py
    y_diff   = y_centered - y_proj           # y - Py
    y_resid  = np.max(np.abs(y_diff))        # ‖y - Py‖_∞

    print(f"Test sample y (user-supplied):")
    print(f"  y (raw)  : {np.round(y, 4)}")
    if args.center:
        print(f"  y (ctrd) : {np.round(y_centered, 4)}")
    print(f"  Py       : {np.round(y_proj, 4)}")
    print(f"  y - Py   : {np.round(y_diff, 4)}")
    print(f"  ‖y-Py‖_∞ : {y_resid:.6f}")
    print()

    # ------------------------------------------------------------------
    # 8. Constraint check: ‖y - Py‖_∞ < epsilon
    # ------------------------------------------------------------------
    print(f"{'='*60}")
    print(f"  Constraint : ‖y - Py‖_∞ < max_x ‖x - Px‖_∞")
    print(f"  ‖y - Py‖_∞ = {y_resid:.6f}")
    print(f"  epsilon    = {epsilon:.6f}")
    if y_resid < epsilon:
        print(f"  Result     : ✓  SATISFIED  ({y_resid:.6f} < {epsilon:.6f})")
    else:
        print(f"  Result     : ✗  VIOLATED   ({y_resid:.6f} >= {epsilon:.6f})")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 9. Residual distribution across training set
    # ------------------------------------------------------------------
    percentiles = [50, 75, 90, 95, 99, 100]
    print(f"Training residual ‖x-Px‖_∞ distribution:")
    for pct in percentiles:
        val = np.percentile(resid, pct)
        label = "max " if pct == 100 else f"p{pct:3d}"
        bar = "█" * int(val / epsilon * 30)
        print(f"  {label} : {val:.6f}  {bar}")
    print()


if __name__ == "__main__":
    main()
