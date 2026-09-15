import utils
import ensemble
import math
import z3
from options import *
import ast
import numpy as np
from rangedbooster import ExtendedBooster
import time
from converttoopb import roundingSolve
import data_distance
import copy
import xgboost as xgb
from ensemble import Interval

from joblib import Parallel, delayed
import tqdm



def dump_solver(solver, filename):
    smt2 = solver.sexpr()
    with open(filename, mode="w", encoding="ascii") as f:  # overwrite
        f.write(smt2)
        f.close()


def solve(options,phi):
    tic = time.perf_counter()
    s = z3.Solver()
    s.set("random_seed", options.seed)
    s.add(phi)
    r = s.check()
    toc = time.perf_counter()
    if r == z3.sat:
        m = s.model()
        return m
    return None


def search_anomaly_for_features(
    ensemble,
    features,
    precision,
    # truelabel,
    n_classes,
    model,
    trees,
    n_trees,
    op_range_list,
    base_val,
    feature_names,
    options: Options
):
    testing = False

    if options.verbosity > 3:
        utils.print_info( "center value", base_val)
        utils.print_info( "lower gap"   , options.lgap)
        utils.print_info( "upper gap"   , options.ugap)
    
    lgap = int(options.lgap * precision)
    ugap = int(options.ugap * precision)
    
    if testing:
        n_trees = 3

    trees = trees[trees["Tree"] < n_trees]
    
    truelabel  = options.truelabel
    otherlabel = options.otherlabel

    vars1 = {}
    vars2 = {}

    # ------------------------------------------------------
    # For deugging only
    # ------------------------------------------------------
    interested_bits = {}
    
    # -------------------------------------------------------
    # Make variable for each node
    # -------------------------------------------------------
    if options.encoding == "allsum":
        for idx, row in trees.iterrows():
            vars1[row["ID"]] = z3.Real("v1-" + "i-" + row["ID"])
            vars2[row["ID"]] = z3.Real("v2-" + "i-" + row["ID"])
    else:
        for idx, row in trees.iterrows():
            vars1[row["ID"]] = z3.Bool("v1-" + "b-" + row["ID"])
            vars2[row["ID"]] = z3.Bool("v2-" + "b-" + row["ID"])

    # -----------------------------------------------------------
    # Make bits for each feature and constrains on the feature bits
    # -----------------------------------------------------------
    def make_bits_for_features(i, prefix, sliced, vmap, cons):
        prev = False
        for r, row in sliced.iterrows():
            split = row["Split"]
            split = str(split)
            fname = f"f{i}_{split}"
            v = z3.Bool(f"{prefix}_b_" + fname)
            cons.append(z3.Implies(prev, v))
            prev = v
            vmap[fname] = v

        
    split_bit_map = {}
    split_sat_value_map = {}
    split_guard_map = {}
    ord_bits_cons = []
    n_features = ensemble.n_features

    for i in range(n_features):
        sliced = trees[(trees["Feature"] == f"f{i}")][["Feature", "Split"]].copy()
        sliced.sort_values(["Split"], inplace=True)
        sliced.drop_duplicates(inplace=True)
        sliced = sliced[
            (op_range_list[i][0] < sliced["Split"])
            & (sliced["Split"] <= op_range_list[i][1])
        ]
        split_bit_map[i] = []
        prev = op_range_list[i][0]
        for r, row in sliced.iterrows():
            var_name = f"f{i}" + "_" + str(row["Split"])
            split_bit_map[i].append(var_name)
            split_sat_value_map[var_name] = float(prev)
            # if ensemble.split_kind == "<":
            #     split_sat_value_map[var_name] = float(prev)
            # else:
            #     split_sat_value_map[var_name] = float(row["Split"])
            split_guard_map[var_name] = float(row["Split"]) 
            prev = float(row["Split"])
        make_bits_for_features(i, "v1", sliced, vars1, ord_bits_cons)
        make_bits_for_features(i, "v2", sliced, vars2, ord_bits_cons)
        split_sat_value_map[f"f{i}" + "_" + str("Last")] = prev
        # if ensemble.split_kind == "<":
        #     split_sat_value_map[f"f{i}" + "_" + str("Last")] = prev
        # else:
        #     split_sat_value_map[f"f{i}" + "_" + str("Last")] = op_range_list[i][1]

    def not_too_far(d_idx, vars1, vars2, cons):  # TODO
        num_splits = len(split_bit_map[d_idx])
        allowed_diff = max(5, int(num_splits / 10))
        for r in range(-1, num_splits):
            if r + allowed_diff >= num_splits:
                continue
            if r != -1:
                b_r0 = vars1[split_bit_map[d_idx][r]]
            else:
                b_r0 = False
            edge_cond = z3.And(z3.Not(b_r0), vars1[split_bit_map[d_idx][r + 1]])
            cons.append(
                z3.Implies(edge_cond, vars2[split_bit_map[d_idx][r + allowed_diff]])
            )
            if r != -1:
                b_r0 = vars2[split_bit_map[d_idx][r]]
            else:
                b_r0 = False
            edge_cond = z3.And(z3.Not(b_r0), vars2[split_bit_map[d_idx][r + 1]])
            cons.append(
                z3.Implies(edge_cond, vars1[split_bit_map[d_idx][r + allowed_diff]])
            )

    # def limit_range(d_idx, vars_list, cons):
    #     for i, b_name in enumerate(split_bit_map[d_idx]):
    #         value = split_sat_value_map[b_name]
    #         try:
    #             if value < limit_range_list[d_idx][0]:
    #                 for vars in vars_list:
    #                     cons.append(z3.Not(vars[b_name]))
    #             if limit_range_list[d_idx][1] <= value:
    #                 for vars in vars_list:
    #                     cons.append(vars[b_name])
    #         except:
    #             pass

    rev_feature_names = {}
    for i in feature_names:
        rev_feature_names[feature_names[i]] = i 
        
    def add_clause_restriction(clause, sensitive_features):
        # give count on guards
        cons = []
        for (is_pos,v,g) in clause:
            v_idx = rev_feature_names[v]
            if v_idx in sensitive_features: return True # We do not restrict
            lit_cons = True
            for i, b_name in enumerate(split_bit_map[v_idx]):
                value = split_guard_map[b_name]
                if g <= value:
                    lit_cons= vars1[b_name]
                    break
            if not is_pos: lit_cons = z3.Not( lit_cons )
            cons.append(lit_cons)

        return z3.Or(cons)
    
    # ---------------------------
    # Helpers: subtree ranges & sensitivity
    # ---------------------------
    def compute_subtree_ranges(trees, precision):
        """
        Build rangemap: node ID → (min_leaf_value, max_leaf_value) for its
        subtree.  Single pass over the DataFrame in reverse order (children
        before parents) — O(n), no recursion, no repeated scans.
        """
        # Pre-extract columns as numpy arrays for fast access
        ids      = trees["ID"].values
        features = trees["Feature"].values
        gains    = trees["Gain"].values
        yes_ids  = trees["Yes"].values
        no_ids   = trees["No"].values
        n        = len(ids)

        rangemap = {}
        # Reverse iteration: tree dumps are BFS/pre-order so children
        # always appear after their parent → reverse guarantees every
        # child is processed before its parent.
        for i in range(n - 1, -1, -1):
            nid = ids[i]
            if features[i] == "Leaf":
                g = gains[i]
                rangemap[nid] = (
                    int(math.floor(g * precision)),
                    int(math.ceil(g * precision)),
                )
            else:
                y = yes_ids[i]
                n_ = no_ids[i]
                ymin, ymax = rangemap.get(y, (0, 0))
                nmin, nmax = rangemap.get(n_, (0, 0))
                rangemap[nid] = (
                    ymin if ymin < nmin else nmin,
                    ymax if ymax > nmax else nmax,
                )
        return rangemap

    def in_distro_clause_cons( sensitive_features, in_distro_clause_file ):
        if in_distro_clause_file:
            ofile = open(in_distro_clause_file, "r")
            clauses = ofile.readlines()
            clauses = [ ast.literal_eval(clause) for clause in clauses]
            cons = [add_clause_restriction(clause, sensitive_features) for clause in clauses]
            return cons
        return []
        
    def all_equal_but_a_few(d_idxs, vars_list, num_features):
        cons = []
        vars1, vars2 = vars_list[0], vars_list[1]
        for idx in range(0, num_features):
            if idx in d_idxs:
                continue
                # if not close:
                #     continue
                exactly1neq = [
                    (z3.Not(vars1[fname] == vars2[fname]), 1)
                    for fname in split_bit_map[idx]
                ]
                if len(exactly1neq) == 0:
                    continue
                cons.append(z3.PbLe(exactly1neq, 100000000))
            for fname in split_bit_map[idx]:
                cons.append(vars1[fname] == vars2[fname])
        if options.small_change:
            for d_idx in d_idxs:
                not_too_far(d_idx, vars1, vars2, cons)
        return cons

    def get_feature_bit(feature, split, vars):
        f = int(feature[1:])
        try:
            if split <= op_range_list[f][0]:
                return False
            elif split > op_range_list[f][1]:
                return True
        except:
            if split <= 0:
                return False
        return vars[feature + "_" + str(split)]

    # ---------------------------------------
    #
    # ---------------------------------------

    def gen_cons_tree(trees, vars, up):
        cons = []
        for idx, row in trees.iterrows():
            v = vars[row["ID"]]
            if row["Feature"] == "Leaf":
                if up:
                    expr = int(np.ceil(row["Gain"] * precision))
                else:
                    expr = int(np.floor(row["Gain"] * precision))
            else:
                split = row["Split"]
                f = int(row["Feature"][1:])
                if split <= op_range_list[f][0]:
                    cond = False
                elif split > op_range_list[f][1]:
                    cond = True
                else:
                    cond = vars[row["Feature"] + "_" + str(split)]
                yes = vars[row["Yes"]]
                no = vars[row["No"]]
                expr = z3.If(cond, yes, no)
            cons.append(v == expr)
        return cons

    pows = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]

    def abstract_prob(p, precision, up):
        if up:
            if options.sureofcounter:
                val = int(np.floor(p * precision))
            else:
                val = int(np.ceil(p * precision))
            # for i,abst in enumerate(pows):
            #     if val <= abst: break
        else:
            if options.sureofcounter:
                val = int(np.ceil(p * precision))
            else:
                val = int(np.floor(p * precision))
            # for i,abst in enumerate(pows):
            #     if val < abst: break
        return val

    def gen_ancestor_constraints(row, parent, v, cons):
        # ----------------------------------------------------
        # Add constraints saying that if a leaf is visited
        # then all ancestor are visited.
        # The following adds binary clauses in the constraints,
        # therefore the faster propagation in unit propagation.
        # -----------------------------------------------------
        ancestor = row["ID"]
        while ancestor in parent and parent[ancestor]:
            ancestor, cond, _ = parent[ancestor]
            cons.append(z3.Implies(v, cond))

    def is_affected_by_change(row, parent):
        ancestor = row["ID"]
        while ancestor in parent and parent[ancestor]:
            ancestor, _, f = parent[ancestor]
            if f in features:
                return True
        return False
    

    def gen_pb_cons_tree(trees, vars, up, rangemap={}, stop=lambda x, y: False):
        cons = []
        values = {}
        ignore = []
        parent = {}
        unaffected = set()

        if n_classes == 1:
            affected = set()
        else:
            affected = {}
            for i in range(n_classes):
                affected[i] = set()
        if up:
            up_name = "u-"
        else:
            up_name = "d-"
        for tid in range(n_trees * n_classes):
            values[tid] = {}
            parent[f"{tid}-{ensemble.get_root_name()}"] = None
        for idx, row in trees.iterrows():
            v = vars[row["ID"]]
            tid = row["Tree"] * n_classes + row["class"]
            if row["ID"] in ignore:
                ignore.append(row["Yes"])
                ignore.append(row["No"])
                continue
            if row["Feature"] == "Leaf":
                tid = row["Tree"] * n_classes + row["class"]
                val = abstract_prob(row["Gain"], precision, up)  # What is this? Arhaan
                bit = z3.Bool(f"{up_name}{tid}-{val}")
                if val in values[tid]:
                    values[tid][val][1].append(v)
                else:
                    values[tid][val] = (bit, [v])
                if options.ancestor_cons:
                    gen_ancestor_constraints(row, parent, v, cons)
                if not is_affected_by_change(row, parent):
                    unaffected.add(row["ID"])
                else:
                    if n_classes == 1:
                        if up:
                            affected.add((val, bit))
                        else:
                            # print("not reaching")
                            affected.add((-val, bit))
                    else:
                        affected[row["class"]].add((val, bit))
            elif stop(row, rangemap):
                # ----------------------------------------------------------
                # DEPRECATED
                # ----------------------------------------------------------
                # Do not explore the subtree that have similar output leaves
                # ----------------------------------------------------------
                if up:
                    val = int(np.ceil(precision * rangemap[row["ID"]][1]))
                else:
                    val = int(np.floor(precision * rangemap[row["ID"]][0]))
                bit = z3.Bool(f"{up_name}{tid}-{val}")
                if val in values[tid]:
                    values[tid][val][1].append(v)
                else:
                    values[tid][val] = (bit, [v])
                if options.ancestor_cons:
                    gen_ancestor_constraints(row, parent, v, cons)
                if not is_affected_by_change(row, parent):
                    unaffected.add(row["ID"])
                # Don't traverse this tree further
                ignore.append(row["Yes"])
                ignore.append(row["No"])
            else:
                cond = get_feature_bit(row["Feature"], row["Split"], vars)
                if cond is True:
                    if options.debug:
                        print("NODE", row["ID"], "forced YES ->", row["Yes"])
                    cons.append(vars[row["Yes"]] == v)
                    parent[row["Yes"]] = (row["ID"], True, int(row["Feature"][1:]))
                    ignore.append(row["No"])

                # -------- forced FALSE split --------
                elif cond is False:
                    if options.debug:
                        print("NODE", row["ID"], "forced NO ->", row["No"])
                    cons.append(vars[row["No"]] == v)
                    parent[row["No"]] = (row["ID"], True, int(row["Feature"][1:]))
                    ignore.append(row["Yes"])

                # -------- normal symbolic split --------
                else:
                    if options.debug:
                        print("NODE", row["ID"], "symbolic split")
                    cons.append(z3.And(v, cond) == vars[row["Yes"]])
                    cons.append(z3.And(v, z3.Not(cond)) == vars[row["No"]])
                    parent[row["Yes"]] = (row["ID"], cond, int(row["Feature"][1:]))
                    parent[row["No"]] = (row["ID"], z3.Not(cond), int(row["Feature"][1:]))

        cons += [vars[f"{tid}-{ensemble.get_root_name()}"] for tid in range(n_trees)]  # Root nodes are true
        all_leaves = []
        if n_classes > 2:
            for i in range(n_classes):
                all_leaves.append([])

        for tid in range(n_trees * n_classes):
            bits_map = values[tid]
            tree_leaves = []
            if n_classes > 2:
                curlabel = tid % n_classes
            for val, (bit, leaves) in bits_map.items():
                if n_classes > 2:
                    all_leaves[curlabel].append((val, bit))
                else:
                    all_leaves.append((val, bit))
                tree_leaves.append((1, bit))
                cons.append(z3.Or(leaves) == bit)
            cons.append(z3.PbEq(tree_leaves, 1))
            # for pair in bits: all_leaves.append(pair)
        return cons, all_leaves, affected, unaffected

    def gowal_distance_constraint(
        sample,
        features,
        feature_types,
        op_range_list,
        split_bit_map,
        split_sat_value_map,
        vars1,
        perturb,
        binary_perturb,
        ):

        cont_lhs_terms = []
        binary_lhs_term = []
        # constant  = 0.0
        cont_constant = 0.0
        bin_constant  = 0.0

        for feat_idx in range(len(op_range_list)):
            if feat_idx in features:
                continue
            fnames = split_bit_map[feat_idx]
            if len(fnames) == 0:
                continue

            s_f   = float(sample[feat_idx])
            ftype = feature_types[feat_idx]
            low, high = op_range_list[feat_idx]
            thresholds  = [split_guard_map[fn] for fn in fnames]
            breakpoints = [low] + thresholds + [high]
            K = len(thresholds)

            if ftype == 'linear':
                span = high - low if high != low else 1.0
                def interval_dist(k):
                    lb_k, ub_k = breakpoints[k], breakpoints[k+1]
                    if s_f < lb_k:   return (lb_k - s_f) / span
                    if s_f >= ub_k:  return (s_f - ub_k) / span
                    return 0.0

            elif ftype == 'log':
                if low <= 0 or s_f <= 0:
                    continue
                log_span = math.log(high) - math.log(low)
                if log_span == 0:
                    continue
                log_s = math.log(s_f)
                def interval_dist(k):
                    lb_k, ub_k = breakpoints[k], breakpoints[k+1]
                    log_lb = math.log(lb_k) if lb_k > 0 else -float('inf')
                    log_ub = math.log(ub_k) if ub_k > 0 else  float('inf')
                    if log_s < log_lb:   return (log_lb - log_s) / log_span
                    if log_s >= log_ub:  return (log_s - log_ub) / log_span
                    return 0.0

            elif ftype in ('categorical', 'bool'):
                def interval_dist(k):
                    lb_k, ub_k = breakpoints[k], breakpoints[k+1]
                    return 0.0 if lb_k <= s_f < ub_k else 1.0

            else:
                continue
            

            # constant += interval_dist(K)
            if ftype in ('categorical', 'bool'):
                bin_constant += interval_dist(K)
            else:
                cont_constant += interval_dist(K)

            for k in range(K):
                
                dist = interval_dist(k) - interval_dist(k + 1)
                if dist != 0.0:
                    if ftype in ('categorical', 'bool'): 
                        binary_lhs_term.append((dist, vars1[fnames[k]]))
                    else:
                        cont_lhs_terms.append((dist, vars1[fnames[k]]))
                        # lhs_terms.append((dist, vars1[fnames[k]]))
        if not cont_lhs_terms and not binary_lhs_term:
            return None
        if binary_perturb >0:
            return [z3.Sum([c * p for c, p in cont_lhs_terms]) <= perturb - cont_constant, 
                    z3.Sum([c * p for c, p in binary_lhs_term]) <= binary_perturb - bin_constant]
        else:
            lhs_terms = cont_lhs_terms + binary_lhs_term  
            return [z3.Sum([c * p for c, p in lhs_terms]) <= perturb - cont_constant - bin_constant]

        
    model = ExtendedBooster(model)
    rangemap = compute_subtree_ranges(trees, precision)
    stop = lambda row, ran: (ran.get(row["ID"], (0, 0))[1] - ran.get(row["ID"], (0, 0))[0]) < (ugap - lgap) / precision
    model = model.booster
    if options.encoding == "allsum":
        cs1 = gen_cons_tree(trees, vars1, up=True)
        cs2 = gen_cons_tree(trees, vars2, up=False)
        expr1 = sum([vars1[f"{tid}-{ensemble.get_root_name()}"] for tid in range(n_trees)])
        expr2 = sum([vars2[f"{tid}-{ensemble.get_root_name()}"] for tid in range(n_trees)])
        prop = [(expr1 > ugap), (expr2 < lgap)]
    else:
        cs1, up_leaves, up_affected, unaffected = gen_pb_cons_tree(
            trees, vars1, up=True , rangemap=rangemap, stop=stop
        )
        cs2, down_leaves, down_affected, _ = gen_pb_cons_tree(
            trees, vars2, up=False , rangemap=rangemap, stop=stop
        )
        unchanged = []
        affected_diff = []

        def merge_and_negate(list1, list2):
            return list1 + [(-w, var) for w, var in list2]

        if options.unaffected_cons:
            if options.verbosity > 5: print(f"# {len(unaffected)} leaves are marked as unaffected")
            for leaf in unaffected:
                unchanged.append(vars1[leaf] == vars2[leaf])
        if options.affected_cons:
            if ensemble.multiclass:
                if truelabel == -1:
                    zero = True
                    oraffected = []

                    for i in range(n_classes):
                        for j in range(n_classes):
                            if j == i: continue
                            if len(list(up_affected[i]) + list(down_affected[j])) != 0:
                                oraffected.append( z3.PbGe( list(up_affected[i]) + list(down_affected[j]), ugap-lgap) )
                                zero = False
                    if zero:
                        a = z3.Bool("triv")
                        affected_diff = [z3.PbEq([(1, a), (1, z3.Not(a))], 0)]
                    else:
                        affected_diff = [z3.Or(*oraffected)]
                else:
                    if otherlabel == -1:
                        zero = True
                        oraffected = []
                        for i in range(n_classes):
                            if i == truelabel: continue
                            if (len(list(up_affected[truelabel])+ list(down_affected[i])) != 0):
                                oraffected.append(z3.PbGe(list(up_affected[truelabel]) + list(down_affected[i]),ugap-lgap,))
                                zero = False
                        if zero:
                            a = z3.Bool("triv")
                            affected_diff = [z3.PbEq([(1, a), (1, z3.Not(a))], 0)]
                        else:
                            affected_diff = [z3.Or(*oraffected)]
                    else:
                        if (len(list(up_affected[truelabel])+ list(down_affected[otherlabel]))!= 0):
                            affected_diff = [
                                z3.PbGe(                                    
                                    merge_and_negate( list(up_affected[truelabel]), list(up_affected[otherlabel]) ) +
                                    merge_and_negate( list(down_affected[otherlabel]), list(down_affected[truelabel]) ),
                                    ugap - lgap, # 2 * gap,
                                )
                            ]
                        else:
                            a = z3.Bool("triv")
                            affected_diff = [z3.PbEq([(1, a), (1, z3.Not(a))], 0)]
            else:
                if len(list(up_affected) + list(down_affected)) != 0:
                    affected_diff = [z3.PbGe(list(up_affected) + list(down_affected), ugap - lgap)]
                else:
                    a = z3.Bool("triv")
                    affected_diff = [z3.PbEq([(1, a), (1, z3.Not(a))], 0)]
            # print(list(up_affected)+list(down_affected))
        prop = unchanged + affected_diff
        if n_classes == 1:
            ugap = ugap - int(np.ceil(base_val*precision))
            lgap = lgap - int(np.floor(base_val*precision))
            prop += [z3.PbGe(up_leaves, ugap ), z3.PbLe(down_leaves, lgap)]
        else:
            if truelabel != -1:
                if otherlabel == -1:
                    orcond = []
                    for i in range(n_classes):
                        if i == truelabel:
                            continue
                        prop += [z3.PbGe(merge_and_negate(up_leaves[truelabel], up_leaves[i]),ugap-lgap,)]  # , z3.PbLe(down_leaves[truelabel] + down_leaves[i], lgap)]
                        temp = []
                        for j in range(n_classes):
                            if j == i: continue
                            if j == truelabel:
                                temp.append(z3.PbGe(merge_and_negate(down_leaves[i], down_leaves[j]),ugap-lgap,))
                            else:
                                if options.strong_multi:
                                    temp.append(z3.PbGe(merge_and_negate(down_leaves[i], down_leaves[j]),ugap-lgap,))
                                else:
                                    temp.append(z3.PbGe(merge_and_negate(down_leaves[i], down_leaves[j]),0,))
                        orcond.append(z3.And(*temp))
                    prop += [z3.Or(*orcond)]
                else:
                    # assert(False) # Why this combination exists?
                    for i in range(n_classes):
                        if i == truelabel: continue
                        prop += [z3.PbGe(merge_and_negate(up_leaves[truelabel], up_leaves[i]),ugap-lgap,)]  # , z3.PbLe(down_leaves[truelabel] + down_leaves[i], lgap)]
                    for i in range(n_classes):
                        if i == otherlabel: continue
                        prop += [z3.PbGe(merge_and_negate(down_leaves[otherlabel], down_leaves[i]),ugap-lgap,)]
                        # else:
                        #     if options.strong_multi:
                        #         prop += [
                        #             z3.PbGe(
                        #                 merge_and_negate(
                        #                     down_leaves[i], down_leaves[otherlabel]
                        #                 ),
                        #                 gap,
                        #             )
                        #         ]
                        #     else:
                        #         prop += [
                        #             z3.PbGe(
                        #                 merge_and_negate(
                        #                     down_leaves[otherlabel], down_leaves[i]
                        #                 ),
                        #                 0,
                        #             )
                        #         ]
            else:
                mainorcond = []
                for j in range(n_classes):
                    temp = []
                    orcond = []
                    for i in range(n_classes):
                        if i == j: continue
                        # TDDO: why gap?
                        temp += [z3.PbGe(merge_and_negate(up_leaves[j], up_leaves[i]), gap)]  # , z3.PbLe(down_leaves[truelabel] + down_leaves[i], lgap)]
                        temp2 = []
                        for k in range(n_classes):
                            if k == i: continue
                            if k == j:
                                temp2.append(z3.PbGe(merge_and_negate(down_leaves[i], down_leaves[k]),ugap-lgap,))
                            else:
                                if options.strong_multi:
                                    temp2.append(z3.PbGe(merge_and_negate(down_leaves[i], down_leaves[k]),ugap-lgap,))
                                else:
                                    temp2.append(z3.PbGe(merge_and_negate(down_leaves[i], down_leaves[k]),0,))
                        orcond.append(z3.And(*temp2))
                    temp += [z3.Or(*orcond)]
                    mainorcond.append(z3.And(*temp))
                prop += [z3.Or(*mainorcond)]

    
    # ---------------------------------------
    # Collect all constraints
    # ---------------------------------------
    aone = all_equal_but_a_few(features, [vars1, vars2], n_features)

    clauses = in_distro_clause_cons( features, options.in_distro_clauses_file )
    
    all_cons = ord_bits_cons + cs1 + cs2 + aone + prop + clauses

    if options.gowal:
        gowal_cons = gowal_distance_constraint(
                sample          = options.current_sample,   
                features        = set(features),
                feature_types   = ensemble.feature_types,
                op_range_list   = op_range_list,
                split_bit_map   = split_bit_map,
                split_sat_value_map = split_sat_value_map,
                vars1           = vars1,
                perturb  = options.perturb,
                binary_perturb  = options.binary_perturb
            )
        if gowal_cons is not None:
            all_cons = all_cons + gowal_cons
    
    if options.verbosity > 6:
        print(prop)
        # print(all_cons)

    tic = time.perf_counter()
    if options.solver == "pb" or options.solver == "naive_smt":
        m = solve(options,all_cons)
    elif options.solver == "rounding":
        m = roundingSolve(all_cons)
    elif options.solver == "roundingsoplex":
        m = roundingSolve(all_cons, soplex=True)
    elif options.solver == "z3withSoftConstr":
        opt = z3.Optimize()
        for c in all_cons:
            opt.add(c)
        all_cons = ord_bits_cons + cs1 + cs2 + aone + prop 
        from z3 import is_expr
        if clauses !=[]:
            for clause in clauses:
                if is_expr(clause):
                    opt.add_soft(clause)
        if opt.check() == z3.sat:
            m = opt.model()
    else:
        utils.print_error('arguments', 'Solving method is not selected!')
    toc = time.perf_counter()
    solvingtime = toc - tic
    if m:
        d1 = []
        d2 = []
        region1 = []
        region2 = []
        lbr = '[' if ensemble.split_kind == '<' else '('
        rbr = ')' if ensemble.split_kind == '<' else ']'
        for idx in range(0, ensemble.n_features):
            temp = [split_sat_value_map[fname] for fname in split_bit_map[idx]]
            if len(split_bit_map[idx]) == 0:
                v1 = split_sat_value_map[f"f{idx}_Last"]
                v2 = split_sat_value_map[f"f{idx}_Last"]
                region1.append(Interval('[',op_range_list[idx][0],op_range_list[idx][1],']'))
                region2.append(Interval('[',op_range_list[idx][0],op_range_list[idx][1],']'))
            else:
                v1 = f"f{idx}_Last"
                next_v1 = f"f{idx}_Last"
                breaknext = False
                lbr = '['
                for fname in split_bit_map[idx]:
                    if breaknext:
                        next_v1 = fname
                        
                        break
                    if options.solver == "pb" or options.solver == "naive_smt":
                        cond = z3.is_true(m[vars1[fname]])
                    else:
                        cond = m[vars1[fname]]
                    if cond:
                        v1 = fname
                        breaknext = True
                    else:
                        lbr = '[' if ensemble.split_kind == '<' else '('
                        

                if v1 == f"f{idx}_Last": 
                    interval = Interval(lbr,split_sat_value_map[v1],op_range_list[idx][1],']')
                else:
                    interval = Interval(lbr,split_sat_value_map[v1],split_sat_value_map[next_v1],rbr)
                region1.append(interval)
                
                v2 = f"f{idx}_Last"
                next_v2 = f"f{idx}_Last"
                breaknext = False
                lbr = '['
                for fname in split_bit_map[idx]:
                    if breaknext:
                        next_v2 = fname
                        
                        break
                    if options.solver == "pb" or options.solver == "naive_smt":
                        cond = z3.is_true(m[vars2[fname]])
                    else:
                        cond = m[vars2[fname]]
                    if cond:
                        v2 = fname
                        breaknext = True
                    else:
                        lbr = '[' if ensemble.split_kind == '<' else '('
                if v2 == f"f{idx}_Last":
                    region2.append(Interval(lbr, split_sat_value_map[v2], op_range_list[idx][1],rbr)) 
                else:
                    region2.append(Interval(lbr,split_sat_value_map[v2],split_sat_value_map[next_v2],rbr)) 
                v1 = split_sat_value_map[v1]
                v2 = split_sat_value_map[v2]
            d1.append(v1)
            d2.append(v2)
        
        return [d1, d2], solvingtime, [region1,region2]
        # return [d1, d2], solvingtime
    else:
        return None, solvingtime, None


def pb_solver( options ):

    # ----------------------
    # Accessing arguments
    # ----------------------
    # close = options.close

        
    #---------------------------------------
    # Load model
    #---------------------------------------
    e = ensemble.Ensemble(options)
    e.load(print_vitals=True)
    base_val      = e.get_base_value()    
    feature_names = e.feature_names
    op_range_list = e.op_range_list
    debug               = options.debug
    local_check_samples = options.local_check_samples
    
    utils.print_verbose(options,0,f'Running the solver with precision level', options.precision)

    # --------------------------------------
    # Configure sensitive features
    # --------------------------------------
    features = options.features
    # if options.all_features: features = [i for i in range(e.n_features)]
    # if features is None: features = [0]    

    
    op_range_list2=[]
    def local_check_update_range(sample, op_range_list):
        op_range_list2=[]
        for i in range(0, e.n_features):
            if  (i in features):
                op_range_list2.append(op_range_list[i])
                continue
            perturb = e.get_perturb(i,sample[i])
            # print(op_range_list[i],f"{sample[i]}+{perturb}")
            list_item=(sample[i]-perturb,sample[i]+perturb)
            if math.isnan(op_range_list[i][0]) or math.isnan(op_range_list[i][1]):
                op_range_list2.append(list_item)
            elif max(list_item[0],op_range_list[i][0])<=min(list_item[1],op_range_list[i][1]):
                list_item2=(max(list_item[0],op_range_list[i][0]),min(list_item[1],op_range_list[i][1]))
                op_range_list2.append(list_item2)
            else:
                op_range_list2.append(op_range_list[i])
        return op_range_list2
    
    def runner(idx,n,tupl):
        # print(f"\rQuery {i+1}/{n}  sensitive={state[0]}", end="", flush=True)
        # if  options.n_jobs == 1:
        #     bar_width = 30
        #     filled = int(bar_width * (i + 1) / n)
        #     bar = '█' * filled + ' ' * (bar_width - filled)
        #     print(f"\r[{bar}] {(i+1)/n*100:.1f}%  Query {i+1}/{n} sensitive={state[0]}", end="", flush=True)
            
        if options.log_file:
            log_path = os.path.join(options.log_folder, f"Query_{idx+1}.txt")
            log_file = open(log_path, "w")
        else:
            log_file = None
        utils.print_verbose(options, 0, "--==> Query", f"{idx+1}/{n}",log=log_file)
            
        f = tupl[0]
        precision = tupl[1]
        op_range_list = tupl[2]
        options.current_sample = tupl[3] if len(tupl) > 3 else None
        start_time = time.time()
        def handler(signum, frame):
            raise Exception("end of time")
        # signal.signal(signal.SIGALRM, handler)
        # signal.alarm(options.timeout)
        # if True:
        if options.current_sample is not None:
            utils.print_verbose(options, 5, "Sample", options.current_sample, log=log_file)

        try:
            pair_point, solvingtime, region_pair = search_anomaly_for_features(
                e,
                f,
                precision,
                e.n_classes,
                e.model,
                e.trees,
                e.n_trees,
                op_range_list,
                base_val,
                e.feature_names,
                options
            )
            # utils.print_info('Time:', solvingtime)
        except (RuntimeError, ValueError) as err:
            utils.print_verbose(options,0,"Error:",err)
            # utils.print_verbose(options, 0, f"Insensitive", f)
            # utils.print_verbose(options, 0, "Time", f"{(time.time() - start_time)} seconds")
            return False
        timetaken = time.time() - start_time
        if region_pair != None:
            utils.print_verbose(options,2,"region1",e.print_reg(region_pair[0]))
            utils.print_verbose(options,2,"region2",e.print_reg(region_pair[1]))
            point1 = e.region2point(region_pair[0])
            point2 = e.region2point(region_pair[1])
            result = [point1, point2]
            vals = e.predict(result)
                
            # print(f"********************************")
            result_copy = result[0].copy()
            # result_copy_2=copy.deepcopy(result)
            
            utils.print_verbose(options,0,'Sensitive', f, log=log_file)
            utils.print_verbose(options,0,'Time', timetaken, log=log_file)
            # print(f"Time {(time.time() - start_time)} seconds")
            if False:
                for x in f:
                    result[0][x] = (result[0][x], result[1][x])
                print('Sensitive samples:',result[0])
            else:
                if options.compute_data_distance:
                    data_distance.compute_data_distance(result[0], f,
                                                        e.feature_names,
                                                        e.n_features,
                                                        e.trees, options,
                                                        weights=e.feature_weights)
                # for x in f:
                #     # result_copy_2[0][x] = (result_copy_2[0][x], result_copy_2[1][x])
                #     result[0][x] = f"{result[0][x]}"  
                #     result[1][x] = f"{result[1][x]}" 
                colored_example =  utils.colour_example(result)
                
                utils.print_array_verbose( options, 0, 'Sensitive sample 1:', colored_example[0])
                utils.print_array_verbose( options, 0, 'Sensitive sample 2:', colored_example[1])

                # ----------------------
                # Print log
                # ----------------------
                utils.print_array_verbose( options, 50, 'Sensitive sample 1:', result[0], log=log_file)
                utils.print_array_verbose( options, 50, 'Sensitive sample 2:', result[1], log=log_file)

            utils.print_verbose(options,0,'Output values:',vals,log=log_file)
            if (vals[0] < 0.5) == (vals[1] < 0.5):
                utils.print_error("Wrong result", "both counter examples belong to the same class")
                
            if options.plot:
                e.plot_variations( result_copy, f, op_range_list)
            if log_file:
                log_file.close()
            return idx, True
        else:
            utils.print_verbose(options,0,"Insensitive", f, log=log_file)
            utils.print_verbose(options,0,'Time', timetaken, log=log_file)
            if log_file:
                log_file.close()
            return idx, False
        
    sense_sets = [features]
    if options.all_single: sense_sets = [ [f] for f in range(0, e.n_features) ]

    tasks = [ (fs, options.precision, op_range_list) for fs in sense_sets]

    
    
    if options.local_check_samples:
        # if len(options.local_check_samples) == 0 or len(op_range_list) != len(options.local_check_samples[0]):
        #     utils.print_error( "Input", "#features in models does not match the #features in sample" )
        tasks = []
        for sample in options.local_check_samples:
            op_range_list2 = local_check_update_range( sample, op_range_list )
            for fs in sense_sets:
                tasks.append((fs, options.precision, op_range_list2,sample))
        utils.print_verbose(options,0,"Number of queries", f" {len(sense_sets)} x {len(options.local_check_samples)} = {len(tasks)}")
        
            
    if len(tasks) < 5:
        idx_results = [runner(i,len(tasks),params) for i,params in enumerate(tasks)]
        results = [r for _, r in idx_results]
        utils.sequential_logging(options,idx_results)
        print()   
    else:
        local_verbosity = options.verbosity
        local_plot = options.plot
        options.verbosity = -1
        options.plot = False
        results = []
        batch   = []

        with tqdm.tqdm(total=len(tasks)) as pbar:
            for idx,r in Parallel(n_jobs=options.n_jobs, return_as="generator")(delayed(runner)(i,len(tasks),p) for i,p in enumerate(tasks)):
                results.append(r)
                batch.append((idx, r))
                pbar.set_postfix(sensitive=sum(results)); pbar.update(1)
                if batch and len(batch) >= options.n_jobs:
                        utils.merge_batch(batch,options)
                        batch = []
        if batch:
            utils.merge_batch(batch, options)
            batch = []
        print()
        
        options.verbosity = local_verbosity
        options.plot = local_plot
    if options.log_file and not options.debug and not options.no_remove:
        os.rmdir(options.log_folder) 
    utils.print_verbose( options, -1, "Fraction of sensitive queries", f"{sum(results)}/{len(results)}")
    # except multiprocessing.context.TimeoutError as e:
    #     print(f"Insensitive: {e}")
