#!/usr/bin/env python3

import ensemble
from ensemble import Interval
from utils import print_info, print_verbose
import utils
import milp
from milp import node_wrapper

from gurobipy import *
import numpy as np
import os
import random
import time
from joblib import Parallel, delayed
import tqdm

SIGN_EPS = 1e-7

N_POINTS = 3


class glitchSolver(object):
    """Three coupled copies of the milp.py encoding, plus the glitch shape."""

    def __init__(self, model, options=None):
        self.options = options
        self.model = model
        self.log_file = None
        self.local_sample = None
        self.local_range = None

        self._reject_unsupported()

        self.base_val = model.get_base_value()
        feats = sorted(set(options.features))
        if len(feats) > 1:
            utils.print_error("arguments",
                              f"a glitch is along one feature; got {feats}")
        self.glitch_feat = feats[0] if feats else None
        self.label = feats if feats else "any feature"
        self.q_feat = {}

        self.node_list = []
        self.leaf_v_list = []     # value of each leaf, flat across trees
        self.leaf_pos_list = []   # {treeid, nodeid} of each leaf
        self.leaf_count = [0]     # number of leaves in the first i trees
        self._walk_trees(model)


        # ------------------------------------------------------------------
        # Gurobi model and the three copies of the variables
        # ------------------------------------------------------------------
        self.env = Env(empty=True)
        self.env.setParam('OutputFlag', 0)
        self.env.start()
        self.m = Model("glitch", env=self.env)
        self.m.setParam("OutputFlag", 1 if self.options.verbosity > 6 else 0)
        self.m.setParam("Threads", 1 if self.options.n_jobs > 1 else 0)
        self.m.params.Seed = self.options.seed

        self._add_variables()
        self._add_threshold_order()
        self._add_leaf_sum_one()
        self._add_node_leaf_constraints()

        # ------------------------------------------------------------------
        # Glitch shape
        # ------------------------------------------------------------------
        self._couple_points()
        self.margins, self.margin_lb, self.margin_ub = self._add_margins()
        self.D, self.min_gap = self._add_input_distance()
        self.out_diff, self.out_vals = self._add_output_shape()
        self.alpha = self._add_magnitude()

        self.m.update()
        self._report_size()

    # ----------------------------------------------------------------------
    #  Option checking
    # ----------------------------------------------------------------------

    def _reject_unsupported(self):
        o = self.options
        if o.multiclass:
            utils.print_error("arguments",
                              "--multiclass is not supported")
        for flag, name in [(o.all_features, "--all_features"),
                           (o.pca, "--pca"),
                           (o.gowal, "--gowal"),
                           (o.compute_data_distance, "--compute_data_distance"),
                           (o.in_distro_clauses_file, "--in_distro_clauses"),
                           (o.anchor, "--anchor"),
                           (o.prob, "--prob"),
                           (o.output_gap is not None, "--output_gap")]:
            if flag:
                utils.print_error("arguments",
                                  f"{name} is not supported by --spec glitch")

    # ----------------------------------------------------------------------
    #  Tree walk  
    # ----------------------------------------------------------------------

    def _walk_trees(self, model):
        node_check = {}   # (attribute, threshold) -> index into node_list

        def new_dfs(tree, nodeid, treeid, cid, root=False):
            rows = tree[tree["Node"] == nodeid]
            if len(rows) != 1:
                utils.print_error('Bad trees', f'{len(rows)} rows for node {nodeid}')
            for idx, row in rows.iterrows():
                f = row["Feature"]
                if f == "Leaf":
                    self.leaf_v_list.append(row["Gain"])
                    self.leaf_pos_list.append({"treeid": treeid, "nodeid": nodeid})
                    return [len(self.leaf_v_list) - 1]

                attribute = row['Feature']
                threshold = row['Split']
                left_subtree = row["Yes"].split("-")[1]
                if left_subtree.isdigit(): left_subtree = int(left_subtree)
                right_subtree = row["No"].split("-")[1]
                if right_subtree.isdigit(): right_subtree = int(right_subtree)
                if type(attribute) == str:
                    attribute = int(attribute[1:])

                left_leaves = new_dfs(tree, left_subtree, treeid, cid, False)
                right_leaves = new_dfs(tree, right_subtree, treeid, cid, False)

                if (attribute, threshold) not in node_check:
                    self.node_list.append(
                        node_wrapper(treeid, nodeid, attribute, threshold,
                                     left_leaves, right_leaves, root)
                    )
                    node_check[(attribute, threshold)] = len(self.node_list) - 1
                else:
                    node_index = node_check[(attribute, threshold)]
                    self.node_list[node_index].add_leaves(
                        treeid, nodeid, left_leaves, right_leaves, root
                    )
                return left_leaves + right_leaves

        for i in range(model.n_trees * model.n_classes):
            c = i % model.n_classes
            tid = i // model.n_classes
            tree = model.trees[(model.trees["Tree"] == tid)
                               & (model.trees["class"] == c)]
            # the root gets the equality form of the node constraint below
            new_dfs(tree, model.get_root_name(), i, c, root=True)
            self.leaf_count.append(len(self.leaf_v_list))

        if model.n_trees * model.n_classes + 1 != len(self.leaf_count):
            utils.print_error("Bad calculation", "leaf count mismatch")

    # ----------------------------------------------------------------------
    #  Variables
    # ----------------------------------------------------------------------

    def _add_variables(self):
        n_nodes = len(self.node_list)
        n_leaves = len(self.leaf_v_list)
        self.P = [[None] * n_nodes for _ in range(N_POINTS)]
        self.L = [[None] * n_leaves for _ in range(N_POINTS)]

        for j in range(n_nodes):
            for k in range(N_POINTS):
                self.P[k][j] = self.m.addVar(vtype=GRB.BINARY, name=f"p{k}_{j}")

        for i in range(n_leaves):
            for k in range(N_POINTS):
                self.L[k][i] = self.m.addVar(lb=0, ub=1, name=f"l{k}_{i}")

        # p dictionary by attribute, {attr: [(threshold, [p0,p1,p2]), ...]}
        self.pdict = {}
        for j, node in enumerate(self.node_list):
            entry = (node.threshold, [self.P[k][j] for k in range(N_POINTS)], j)
            self.pdict.setdefault(node.attribute, []).append(entry)
        for key in self.pdict:
            self.pdict[key].sort(key=lambda tup: tup[0])

    # ----------------------------------------------------------------------
    #  Structural constraints
    # ----------------------------------------------------------------------

    def _add_threshold_order(self):
        """x < t_i  implies  x < t_{i+1}  along each feature."""
        for key, entries in self.pdict.items():
            for i in range(len(entries) - 1):
                for k in range(N_POINTS):
                    self.m.addConstr(
                        entries[i][1][k] <= entries[i + 1][1][k],
                        name=f"p_consis_attr{key}_{i}th_{k}",
                    )

    def _add_leaf_sum_one(self):
        for t in range(len(self.leaf_count) - 1):
            for k in range(N_POINTS):
                leaf_vars = [self.L[k][j]
                             for j in range(self.leaf_count[t], self.leaf_count[t + 1])]
                self.m.addConstr(
                    LinExpr([1] * len(leaf_vars), leaf_vars) == 1,
                    name=f"leaf_sum_one_for_tree{t}_{k}",
                )

    def _add_node_leaf_constraints(self):
        for j, node in enumerate(self.node_list):
            for idx, item in enumerate(node.leaves_lists):
                for k in range(N_POINTS):
                    p = self.P[k][j]
                    left_l = [self.L[k][i] for i in item[0]]
                    right_l = [self.L[k][i] for i in item[1]]
                    if len(item) == 3:      # this occurrence is a tree root
                        self.m.addConstr(
                            LinExpr([1] * len(left_l), left_l) - p == 0,
                            name=f"p{j}_root_left_{idx}_{k}",
                        )
                        self.m.addConstr(
                            LinExpr([1] * len(right_l), right_l) + p == 1,
                            name=f"p{j}_root_right_{idx}_{k}",
                        )
                    else:
                        self.m.addConstr(
                            LinExpr([1] * len(left_l), left_l) - p <= 0,
                            name=f"p{j}_left_{idx}_{k}",
                        )
                        self.m.addConstr(
                            LinExpr([1] * len(right_l), right_l) + p <= 1,
                            name=f"p{j}_right_{idx}_{k}",
                        )

    # ----------------------------------------------------------------------
    #  Coupling the three points
    # ----------------------------------------------------------------------

    def _couple_points(self):
        for key, entries in self.pdict.items():
            for (threshold, pvars, j) in entries:
                if self.glitch_feat is None or key == self.glitch_feat:
                    for k in range(N_POINTS - 1):
                        self.m.addConstr(pvars[k] <= pvars[k + 1],
                                         name=f"glitch_mono_f{key}_{j}_{k}")
                else:
                    for k in range(N_POINTS - 1):
                        self.m.addConstr(pvars[k] == pvars[k + 1],
                                         name=f"glitch_fix_f{key}_{j}_{k}")
        if self.glitch_feat is None:
            self._add_feature_selector()

    def _eligible(self):
        return [f for f, entries in self.pdict.items()
                if self._min_gap([t for (t, _, _) in entries]) > 0]

    def _add_feature_selector(self):
        eligible = self._eligible()
        if not eligible:
            print("feature has two distinct splits, so no glitch")
            exit(0)
            # utils.print_error("arguments",
            #                   "no feature has two distinct splits, so no glitch "
            #                   "can exist in this model")
        self.q_feat = {f: self.m.addVar(vtype=GRB.BINARY, name=f"q_f{f}")
                       for f in eligible}
        self.m.addConstr(quicksum(self.q_feat.values()) == 1,
                         name="pick_one_feature")

        for f, entries in self.pdict.items():
            moves = [pvars[N_POINTS - 1] - pvars[0] for (_, pvars, _) in entries]
            S = self.m.addVar(lb=0.0, ub=float(len(entries)), name=f"S_f{f}")
            self.m.addConstr(S == quicksum(moves), name=f"S_def_f{f}")
            if f in self.q_feat:
                self.m.addGenConstrIndicator(self.q_feat[f], 1, S,
                                             GRB.GREATER_EQUAL, 1.0,
                                             name=f"vary_f{f}")
                self.m.addGenConstrIndicator(self.q_feat[f], 0, S,
                                             GRB.EQUAL, 0.0, name=f"fix_f{f}")
            else:
                self.m.addConstr(S == 0.0, name=f"fix_f{f}_ineligible")

    def _chosen_feature(self):
        picked = [f for f, v in self.q_feat.items() if v.x > 0.5]
        return picked[0] if picked else None

    # ----------------------------------------------------------------------
    #  Output side
    # ----------------------------------------------------------------------

    def _tree_extremes(self):
        lo = hi = 0.0
        for t in range(len(self.leaf_count) - 1):
            vals = self.leaf_v_list[self.leaf_count[t]:self.leaf_count[t + 1]]
            lo += float(min(vals))
            hi += float(max(vals))
        return lo, hi

    def _add_margins(self):
        lo, hi = self._tree_extremes()
        lo += self.base_val
        hi += self.base_val
        margins = []
        for k in range(N_POINTS):
            mk = self.m.addVar(lb=lo, ub=hi, vtype=GRB.CONTINUOUS, name=f"margin_{k}")
            self.m.addConstr(
                mk == LinExpr(self.leaf_v_list, self.L[k]) + self.base_val,
                name=f"margin_def_{k}",
            )
            margins.append(mk)
        return margins, lo, hi

    def _add_output_shape(self):
        
        m1, m2, m3 = self.margins

        s = [self.m.addVar(vtype=GRB.BINARY, name=f"side_{k}") for k in range(N_POINTS)]
        for k in range(N_POINTS):
            self.m.addGenConstrIndicator(s[k], True, self.margins[k],
                                         GRB.GREATER_EQUAL, SIGN_EPS, name=f"s{k}_pos")
            self.m.addGenConstrIndicator(s[k], False, self.margins[k],
                                         GRB.LESS_EQUAL, -SIGN_EPS, name=f"s{k}_neg")
        self.m.addConstr(s[0] == s[2], name="ends_same_side")
        self.m.addConstr(s[0] + s[1] == 1, name="middle_other_side")
        self.side = s

        vals = [m1, m2, m3]
        span = self.margin_ub - self.margin_lb

        steps = []
        for a, b, tag in [(0, 1, "12"), (1, 2, "23")]:
            diff = self.m.addVar(lb=-span, ub=span, vtype=GRB.CONTINUOUS,
                                 name=f"out_diff_{tag}")
            absd = self.m.addVar(lb=0.0, ub=span, vtype=GRB.CONTINUOUS,
                                 name=f"out_abs_{tag}")
            self.m.addConstr(diff == vals[b] - vals[a], name=f"out_diff_def_{tag}")
            self.m.addGenConstrAbs(absd, diff, name=f"abs_d{tag}")
            steps.append(absd)
            

        out_diff = self.m.addVar(lb=0.0, ub=span, vtype=GRB.CONTINUOUS, name="out_move")
        self.m.addGenConstrMin(out_diff, steps, name="out_move_def")
        self.out_span = span
        return out_diff, vals

    # ----------------------------------------------------------------------
    #  Input side
    # ----------------------------------------------------------------------

    @staticmethod
    def _min_gap(thresholds):
        if len(thresholds) < 2:
            return 0.0
        return min(thresholds[i + 1] - thresholds[i] for i in range(len(thresholds) - 1))

    def _add_input_distance(self):
        if self.glitch_feat is None:
            return self._add_input_distance_any()
        f = self.glitch_feat
        entries = self.pdict.get(f, [])
        thresholds = [t for (t, _, _) in entries]
        min_gap = self._min_gap(thresholds)
        if min_gap <= 0:
            utils.print_error("arguments",
                              f"feature {f} does not have two distinct splits, "
                              "so no glitch can exist on it")

        terms = [float(thresholds[i] - thresholds[i - 1])
                 * (entries[i][1][N_POINTS - 1] - entries[i][1][0])
                 for i in range(1, len(entries))]
        D = self.m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="D")
        self.m.addConstr(D == quicksum(terms), name=f"D_def_f{f}")
        # the two ends are distinct, so they are at least one split apart
        self.m.addConstr(D >= min_gap, name="D_lower")
        return D, min_gap

    def _feature_distance(self, f):
        """Crossed-gap distance from x0 to x2 along one feature."""
        entries = self.pdict[f]
        thresholds = [t for (t, _, _) in entries]
        terms = [float(thresholds[i] - thresholds[i - 1])
                 * (entries[i][1][N_POINTS - 1] - entries[i][1][0])
                 for i in range(1, len(entries))]
        Df = self.m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                           name=f"D_f{f}")
        self.m.addConstr(Df == quicksum(terms), name=f"D_def_f{f}")
        return Df

    def _add_input_distance_any(self):
        eligible = self._eligible()
        if not eligible:
            utils.print_error("arguments",
                              "no feature has two distinct splits, so no glitch "
                              "can exist in this model")
        per_feature = [self._feature_distance(f) for f in eligible]
        min_gap = min(self._min_gap([t for (t, _, _) in self.pdict[f]])
                      for f in eligible)
        D = self.m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="D")
        self.m.addGenConstrMax(D, per_feature, name="D_max")
        # whichever feature is picked, its two ends are a split apart
        self.m.addConstr(D >= min_gap, name="D_lower")
        return D, min_gap

    def _add_magnitude(self):
        if self.options.alpha == 0:
            alpha = self.m.addVar(lb=0.0, ub=self.out_span / self.min_gap, name="alpha")
            self.m.setParam('NonConvex', 2)
            self.m.addConstr(self.out_diff >= alpha * self.D, name="normalized_gap")
            self.m.setObjective(alpha, GRB.MAXIMIZE)
            return alpha
        self.m.addConstr(self.out_diff >= self.options.alpha * self.D,
                         name="normalized_gap")
        return None

    # ----------------------------------------------------------------------
    #  Local search
    # ----------------------------------------------------------------------

    def _pin_to_local_box(self):
        
        local_range = milp.milpSolver.local_check_update_range(
            self, self.local_sample, self.model.op_range_list)
        self.local_range = local_range
        for key, entries in self.pdict.items():
            lo, hi = local_range[key]
            for (threshold, pvars, j) in entries:
                if (hi < threshold) if self.model.split_kind == '<' else (hi <= threshold):
                    forced = 1
                elif (lo >= threshold) if self.model.split_kind == '<' else (lo > threshold):
                    forced = 0
                else:
                    continue
                for k in range(N_POINTS):
                    self.m.addConstr(pvars[k] == forced,
                                     name=f"box_f{key}_{j}_{k}")

    # ----------------------------------------------------------------------
    #  Solving and reporting
    # ----------------------------------------------------------------------

    def _report_size(self):
        print_verbose(self.options, 4, "Encoding",
                      f"{len(self.node_list)} tests, {len(self.leaf_v_list)} leaves, "
                      f"{len(self.leaf_count)-1} trees")

    def _regions(self, k):
        """Decode copy k into one interval per feature, as milp.py's attack does."""
        default_open = '[' if self.model.split_kind == '<' else '('
        default_close = ')' if self.model.split_kind == '<' else ']'
        region = []
        for f in range(self.model.n_features):
            low, high = self.model.op_range_list[f]
            if self.local_range:
                blo, bhi = self.local_range[f]
                low, high = max(low, blo), min(high, bhi)
            region.append(Interval('[', low, high, ']'))

        for key, entries in self.pdict.items():
            low, high = self.model.op_range_list[key]
            if self.local_range:
                blo, bhi = self.local_range[key]
                low, high = max(low, blo), min(high, bhi)
            thresholds = [t for (t, _, _) in entries]
            above = [t for (t, pvars, _) in entries if pvars[k].x > 0.5] + [high]
            reg = [([low] + thresholds)[-len(above)], above[0]]
            reg = [max(reg[0], low), min(reg[1], high)]
            region[key] = Interval('[' if reg[0] == low else default_open,
                                   reg[0],
                                   reg[1],
                                   ']' if reg[1] == high else default_close)
        return region

    def _verify(self, points, feature):
        """Check that the decoded points really are a glitch of the real model.
        """
        preds = self.model.predict(points)
        vals = [float(p) for p in np.ravel(preds)[:N_POINTS]]
        utils.print_verbose(self.options, 0, 'Output values', vals, log=self.log_file)

        if self.model.model_library not in ("xgboost", "lgbm"):
            return None
        eps = 1e-12
        got = [utils.sigmoid_inv(min(max(v, eps), 1 - eps)) for v in vals]
        want = [mk.x for mk in self.margins]
        drift = max(abs(a - b) for a, b in zip(got, want))
        crossed = ((got[0] >= 0) == (got[2] >= 0)) and ((got[1] >= 0) != (got[0] >= 0))
        moved = min(abs(got[1] - got[0]), abs(got[2] - got[1]))
        ok = crossed
        print_verbose(self.options, 2, "Witness margins",
                      f"model {[round(g,6) for g in got]} vs milp "
                      f"{[round(w,6) for w in want]} (drift {drift:.3g})")
        if not ok:
            print_verbose(self.options, -1, "Warning",
                          f"decoded witness does not reproduce the glitch on "
                          f"f{feature}: the model's outputs do not cross and cross back (move={moved:.6g})"
                          , log=self.log_file)
        return ok

    def search(self, options):
        print_verbose(self.options, 5, "", "Starting to solve")
        self.m.setParam("TimeLimit", self.options.timeout)
        if self.options.alpha != 0:
            self.m.setParam("SolutionLimit", 1)

        if self.local_sample:
            self._pin_to_local_box()

        if self.options.verbosity > 8:
            constraint_file = "/tmp/glitch_constraints.lp"
            self.m.write(constraint_file)
            print_verbose(self.options, 8, "MILP constraints saved", f" {constraint_file}")

        tic = time.perf_counter()
        self.m.optimize()
        toc = time.perf_counter()
        timetaken = toc - tic
        print_verbose(self.options, 3, 'Time', f" {timetaken} seconds")

        if self.m.status == GRB.Status.INFEASIBLE:
            utils.print_verbose(options, 0, "No glitch", self.label, log=self.log_file)
            utils.print_verbose(options, 0, "Time", f"{timetaken} seconds", log=self.log_file)
            return False

        if self.m.SolCount == 0:
            print_verbose(self.options, 0, 'Time',
                          f"Timeout on {self.label} after {timetaken} seconds")
            return None

        if self.m.status == GRB.Status.TIME_LIMIT:
            print_verbose(self.options, 0, "Note",
                          "time limit reached; reporting the best glitch found so far",
                          log=self.log_file)

        feature = self._chosen_feature() if self.glitch_feat is None else self.glitch_feat
        name = self.model.feature_names.get(feature, feature)
        alpha = self.out_diff.x / self.D.x if self.D.x > 0 else float('inf')

        utils.print_verbose(options, 0, 'Glitch feature', f"{feature} ({name})", log=self.log_file)
        utils.print_verbose(options, 0, 'Magnitude alpha', f"{alpha}", log=self.log_file)
        utils.print_verbose(options, 0, 'Time', timetaken, log=self.log_file)
        print_verbose(self.options, 2, 'Input distance', self.D.x)
        print_verbose(self.options, 2, 'Output movement', self.out_diff.x)

        points = []
        for k in range(N_POINTS):
            region = self._regions(k)
            utils.print_verbose(options, 3, f"region{k}", self.model.print_reg(region))
            points.append(self.model.region2point(region))

        for k, x in enumerate(points):
            utils.print_array_verbose(options, 0, f'Glitch point {k}:', x, log=self.log_file)
        utils.print_verbose(options, 0, f'Values of f{feature}',
                            [x[feature] for x in points], log=self.log_file)

        self._verify(points, feature)
        return True


def runner(idx, n, task, options, model):
    f, sample = task
    # --all_single builds feature-carrying tasks without --features ever being
    # passed, so the task -- not the command line -- decides the mode
    options.features = f if f is not None else []
    if options.log_file:
        log_path = os.path.join(options.log_folder, f"Query_{idx+1}.txt")
        log_file = open(log_path, "w")
    else:
        log_file = None
    utils.print_verbose(options, 0, "--==> Query", f"{idx+1}/{n}", log=log_file)
    solver = glitchSolver(model, options=options)
    solver.local_sample = sample
    solver.log_file = log_file
    if sample is not None:
        utils.print_verbose(options, 5, "Sample", sample, log=log_file)
    result = solver.search(options)
    if log_file:
        log_file.close()
    return idx, result


def eligible_features(model, feats, options):
    """A feature can only host a glitch if it has two distinct splits."""
    keep, skip = [], []
    for f in feats:
        if len(model.get_guard_list(f)) >= 2:
            keep.append(f)
        else:
            skip.append(f)
    if skip:
        utils.print_verbose(options, 0, "Skipped (fewer than two splits)", skip)
    return keep


def feature_sets(model, options):
    """ --all_single turns into one fixed-feature problem per feature.

    With no --features at all there is nothing to enumerate: one query, with the
    feature left to the solver.
    """
    if not options.all_single and not options.features:
        if options.all_features:
            utils.print_error("arguments",
                              "--all_features is not supported by --solver glitch")
        if not eligible_features(model, list(range(model.n_features)), options):
            utils.print_error("arguments",
                              "no feature in this model has two distinct splits")
        return [None]

    if not options.all_single:
        if options.all_features:
            utils.print_error("arguments",
                              "--all_features is not supported by --solver glitch: "
                              "a glitch is along one feature, so name it with ")
        feats = sorted(set(options.features))
        if len(feats) > 1:
            utils.print_error("arguments",
                              "a glitch is along a single feature: give one "
                              "--features, or --all_single for one query per "
                              "feature")
        feats = eligible_features(model, feats, options)
        if not feats:
            print("feature has two distinct splits, so no glitch")
            exit(0)
            # utils.print_error("arguments",
            #                   "none of the requested features has two distinct splits")
        return [feats]

    types = model.feature_types or []
    is_cat = lambda i: i < len(types) and types[i] in ['categorical', 'bool']
    feat = []
    if options.numeric:
        feat.extend(i for i in range(model.n_features) if not is_cat(i))
    if options.categorical:
        feat.extend(i for i in range(model.n_features) if is_cat(i))
    if not feat:
        feat = list(range(model.n_features))
    feat = eligible_features(model, feat, options)

    if options.limit_sens > 0:
        k = min(options.limit_sens, len(feat))
        rng = np.random.default_rng(options.seed)
        chosen = sorted(int(f) for f in rng.choice(feat, size=k, replace=False))
        utils.print_verbose(options, -1, "Sampled single features",
                            f"{k}/{len(feat)} of {model.n_features} "
                            f"(seed {options.seed}): {chosen}")
        feat = chosen
    return [[f] for f in feat]


def glitch_solver(options):

    random.seed(options.seed)
    np.random.seed(options.seed)

    # ---------------------------------------
    # Load model
    # ---------------------------------------
    e = ensemble.Ensemble(options)
    e.load(print_vitals=True)
    model = e

    sens_sets = feature_sets(model, options)

    if options.local_check_samples:
        tasks = [(f, sample) for sample in options.local_check_samples for f in sens_sets]
    else:
        tasks = [(f, None) for f in sens_sets]

    if len(tasks) < 5:
        idx_results = [runner(i, len(tasks), t, options, model) for i, t in enumerate(tasks)]
        results = [r for _, r in idx_results]
        utils.sequential_logging(options, idx_results)
        print()
    else:
        local_verbosity = options.verbosity
        options.verbosity = -1
        results = []
        batch = []
        with tqdm.tqdm(total=len(tasks)) as pbar:
            for idx, r in Parallel(n_jobs=options.n_jobs, return_as="generator")(
                        delayed(runner)(i, len(tasks), t, options, model)
                        for i, t in enumerate(tasks)
            ):
                results.append(r)
                batch.append((idx, r))
                n_gl = sum(r is True for r in results)
                n_to = sum(r is None for r in results)
                pbar.set_postfix(glitch=n_gl, timeout=n_to, refresh=False)
                pbar.update(1)
                if batch and len(batch) >= options.n_jobs:
                    utils.merge_batch(batch, options)
                    batch = []
        if batch:
            utils.merge_batch(batch, options)
        print()
        options.verbosity = local_verbosity

    if options.log_file and not options.debug and not options.no_remove:
        os.rmdir(options.log_folder)

    n_gl = sum(r is True for r in results)
    n_to = sum(r is None for r in results)
    utils.print_verbose(options, -1, "Fraction of queries with a glitch:",
                        f"{n_gl}/{len(results)}" + (f" ({n_to} timed out)" if n_to else ""))
    if options.log_file:
        with open(options.log_file, 'a') as out:
            utils.print_info("alpha", f"{options.alpha}", outs=out)
            utils.print_info("Glitch", f"{n_gl}/{len(results)}", outs=out)
            utils.print_info("Timeout", f"{n_to}/{len(results)}", outs=out)
