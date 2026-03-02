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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xgboost as xgb

def dump_solver(solver, filename):
    smt2 = solver.sexpr()
    with open(filename, mode="w", encoding="ascii") as f:  # overwrite
        f.write(smt2)
        f.close()


def solve(phi):
    tic = time.perf_counter()
    s = z3.Solver()
    s.add(phi)
    r = s.check()
    toc = time.perf_counter()
    if r == z3.sat:
        m = s.model()
        return m
    return None


def plot_variations(model, data, features, trees, feature_names, op_range_list):
    fvalues = []
    for feature in features:
        sliced = trees[(trees["Feature"] == f"f{feature}")][["Feature", "Split"]].copy()
        sliced.sort_values(["Split"], inplace=True)
        sliced.drop_duplicates(inplace=True)
        sliced = sliced[
            (op_range_list[feature][0] < sliced["Split"])
            & (sliced["Split"] <= op_range_list[feature][1])
        ]
        # values = [op_range_list[feature][0]] + sliced['Split'].tolist()
        values = sliced["Split"].tolist()
        fvalues.append(values)

    # print(fvalues[0])
    predictions = []
    if len(fvalues) == 2:
        for v in fvalues[1]:
            data[features[1]] = v
            rows = []
            for f0 in fvalues[0]:
                data[features[0]] = f0
                rows.append(data.copy())
            predict_col = model.predict(xgb.DMatrix(rows))
            predictions.append(predict_col)
    else:
        rows = []
        for f0 in fvalues[0]:
            data[features[0]] = f0
            rows.append(data.copy())
        predict_col = model.predict(xgb.DMatrix(rows))
        # predict_col = [ sigmoid_inv(x) for x in predict_col]
        predictions.append(predict_col)

    plt.style.use("_mpl-gallery")
    if len(fvalues) == 1:
        fig, ax = plt.subplots()
        ax.plot(fvalues[0], predictions[0], linewidth=2.0)
        plt.ylabel("Predict")
        plt.xlabel(feature_names[features[0]])
    else:
        ax = plt.figure().add_subplot(projection="3d")
        X, Y = np.meshgrid(np.array(fvalues[0]), np.array(fvalues[1]))
        Z = np.array(predictions)
        ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cm.Blues,
            # edgecolor='royalblue',
            # lw=0.5, #rstride=8, cstride=8,
            # alpha=0.3
        )
        # ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
        # ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
        # ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
        ax.set(
            xlabel=feature_names[features[0]],
            ylabel=feature_names[features[1]],
            zlabel="Predict",
        )
    plt.show()


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
    # print(truelabel)
    # print(otherlabel)
    # exit()

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
            if ensemble.split_kind == "<":
                split_sat_value_map[var_name] = float(prev)
            else:
                split_sat_value_map[var_name] = float(row["Split"])
            split_guard_map[var_name] = float(row["Split"]) 
            prev = float(row["Split"])
        make_bits_for_features(i, "v1", sliced, vars1, ord_bits_cons)
        make_bits_for_features(i, "v2", sliced, vars2, ord_bits_cons)
        if ensemble.split_kind == "<":
            split_sat_value_map[f"f{i}" + "_" + str("Last")] = prev
        else:
            split_sat_value_map[f"f{i}" + "_" + str("Last")] = op_range_list[i][1]

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
                    unaffected.insert(row["ID"])
                # Don't traverse this tree further
                ignore.append(row["Yes"])
                ignore.append(row["No"])
            else:
                cond = get_feature_bit(row["Feature"], row["Split"], vars)
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

        
    model = ExtendedBooster(model)
    stop = lambda x, y: False
    model = model.booster
    if options.encoding == "allsum":
        cs1 = gen_cons_tree(trees, vars1, up=True)
        cs2 = gen_cons_tree(trees, vars2, up=False)
        expr1 = sum([vars1[f"{tid}-{ensemble.get_root_name()}"] for tid in range(n_trees)])
        expr2 = sum([vars2[f"{tid}-{ensemble.get_root_name()}"] for tid in range(n_trees)])
        prop = [(expr1 > ugap), (expr2 < lgap)]
    else:
        cs1, up_leaves, up_affected, unaffected = gen_pb_cons_tree(
            trees, vars1, up=True #, rangemap=rangemap, stop=stop
        )
        cs2, down_leaves, down_affected, _ = gen_pb_cons_tree(
            trees, vars2, up=False #, rangemap=rangemap, stop=stop
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

    if options.verbosity > 6:
        print(prop)
        # print(all_cons)

    tic = time.perf_counter()
    if options.solver == "pb" or options.solver == "naive_smt":
        m = solve(all_cons)
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
        print('Solving method is not selected!')
        exit()
    toc = time.perf_counter()
    solvingtime = toc - tic
    if m:
        d1 = []
        d2 = []
        for idx in range(0, ensemble.n_features):
            if len(split_bit_map[idx]) == 0:
                v1 = split_sat_value_map[f"f{idx}_Last"]
                v2 = split_sat_value_map[f"f{idx}_Last"]
            else:
                v1 = f"f{idx}_Last"
                next_v1 = f"f{idx}_Last"
                breaknext = False
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
                v2 = f"f{idx}_Last"
                next_v2 = f"f{idx}_Last"
                breaknext = False
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
                v1 = split_sat_value_map[v1]
                v2 = split_sat_value_map[v2]
            d1.append(v1)
            d2.append(v2)
        
        return [d1, d2], solvingtime
    else:
        return None, solvingtime


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
    
                
    print('# Running the solver with precision level:', options.precision)

    # --------------------------------------
    # Configure sensitive features
    # --------------------------------------
    features = options.features
    if options.all_features: features = [i for i in range(e.n_features)]
    if features is None: features = [0]    

    
    op_range_list2=[]
    def local_check_update_range(sample, op_range_list):
        op_range_list2=[]
        for i in range(0, e.n_features):
            if  (i in features):
                op_range_list2.append(op_range_list[i])
                continue
            list_item=(sample[i]-options.perturb,sample[i]+options.perturb)
            if math.isnan(op_range_list[i][0]) or math.isnan(op_range_list[i][1]):
                op_range_list2.append(list_item)
            elif max(list_item[0],op_range_list[i][0])<=min(list_item[1],op_range_list[i][1]):
                list_item2=(max(list_item[0],op_range_list[i][0]),min(list_item[1],op_range_list[i][1]))
                op_range_list2.append(list_item2)
            else:
                op_range_list2.append(op_range_list[i])
        return op_range_list2

    def runner(i,tupl):
        print(f"Query {i}:")
        f = tupl[0]
        precision = tupl[1]
        op_range_list = tupl[2]
        
        start_time = time.time()
        def handler(signum, frame):
            raise Exception("end of time")
        # signal.signal(signal.SIGALRM, handler)
        # signal.alarm(options.timeout)
        # if True:
        try:
            result, solvingtime = search_anomaly_for_features(
                e,
                f,
                precision,
                # truelabel,
                e.n_classes,
                e.model,
                e.trees,
                e.n_trees,
                op_range_list,
                base_val,
                e.feature_names,
                options
            )
            utils.print_info('Time:', solvingtime)
        except Exception as err:
            print(err)
            print(f, "Insensitive")
            print(f"Time {(time.time() - start_time)} seconds")
            return
        
        if result != None:
            vals = e.predict(result)
            result_copy = result[0].copy()
            result_copy_2=copy.deepcopy(result)
            
            print('Sensitive:', f)
            if False:
                for x in f:
                    result[0][x] = (result[0][x], result[1][x])
                print('Sensitive samples:',result[0])
            else:
                if options.compute_data_distance:
                    data_distance.compute_data_distance(result[0], f,
                                                        e.feature_names,
                                                        e.n_features,
                                                        e.trees, options)
                for x in f:
                    result_copy_2[0][x] = (result_copy_2[0][x], result_copy_2[1][x])
                    # result[0][x] = f"\033[91m{result[0][x]}\033[0m"  
                    # result[1][x] = f"\033[91m{result[1][x]}\033[0m" 
                    result[0][x] = f"{result[0][x]}"  
                    result[1][x] = f"{result[1][x]}" 
                utils.print_array( 'Sensitive sample 1:', result[0])
                utils.print_array( 'Sensitive sample 2:', result[1])
                
            print('Output values:',vals)

            output2=[]

            # if options.local_check:
            #     utils.print_array( 'Input sample:', local_check)
            #     for i in range(0,len(result_copy_2[0])):
            #         if isinstance(result_copy_2[0][i],tuple):
            #             output2.append('NA')
            #         else:
            #             output2.append(result_copy_2[0][i]-local_check[i])
            #     print('Difference matrix :',output2)            

            if options.plot:
                plot_variations(model, result_copy, f, trees, feature_names, op_range_list)
        else:
            print("Insensitive", f)

    if options.all_single:
        tasks = [([f], options.precision, op_range_list) for f in range(0, n_features)]
    else:
        tasks = [(features, options.precision, op_range_list)]


    if options.local_check_samples:
        if len(options.local_check_samples) == 0 or len(op_range_list) != len(options.local_check_samples[0]):
            print('Errror number of input does not match with the number of the inputs of the model!')
            exit()
        tasks = []
        for sample in options.local_check_samples:
            op_range_list2 = local_check_update_range( sample, op_range_list )
            tasks.append((features, options.precision, op_range_list2))

            
    # if debug:
    results = [runner(i,params) for i,params in enumerate(tasks)]
    # else:
    # results = Parallel(n_jobs=-1)( delayed(runner)(params) for params in tqdm(tasks) )
    # except multiprocessing.context.TimeoutError as e:
    #     print(f"Insensitive: {e}")
