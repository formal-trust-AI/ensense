#!/usr/bin/python3
debug = False
apply_op_ranges = False

# debug=True
encoding = ""


from utils import model_files
import json
import sys
import argparse
import joblib
import xgboost as xgb
import os
import numpy as np
import pandas as pd
import itertools
import ast
import z3
import math
import random
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from rangedbooster import ExtendedBooster
import pdb
from converttoopb import roundingSolve

from subprocess import check_output
from xyplot import Curve
from milp import main as milp_solver

import matplotlib.pyplot as plt
import matplotlib.cm as cm

sureofcounter = False
# solver = "z3"

leafmapping = {}
action = ""
if len(sys.argv) > 1:
    action = sys.argv[1]


pd.set_option("display.max_rows", 500)

# --------------------------------------
# Utilities
# --------------------------------------


def find_leaf_node_id(tree_data, leaf_index):
    """
    This function takes a tree's JSON data and a leaf index,
    and returns the corresponding leaf node ID and its value.
    """

    def traverse(node):
        if "leaf" in node:
            return [f"{node['nodeid']}"]  # Return the leaf node
        else:
            return traverse(node["children"][0]) + traverse(node["children"][1])

    # Get all the leaf nodes by traversing the tree
    leaf_nodes = traverse(tree_data)

    # Find the corresponding leaf node

    if leaf_index < len(leaf_nodes):
        return leaf_nodes[leaf_index]
    else:
        raise ValueError(f"Leaf index {leaf_index} out of bounds for tree")


def sigmoid_inv(x):
    return -np.log(1 / x - 1)


def find_least_node(tid, tree):
    rmap = {}
    for _, row in tree.iterrows():
        if row["Feature"] != "Leaf":
            rmap[row["ID"]] = row["Yes"]
    v = f"{tid}-0"
    while True:
        if v in rmap:
            v = rmap[v]
        else:
            break
    return v


def find_most_node(tid, tree):
    rmap = {}
    for _, row in tree.iterrows():
        if row["Feature"] != "Leaf":
            rmap[row["ID"]] = row["No"]
    v = f"{tid}-0"
    while True:
        if v in rmap:
            v = rmap[v]
        else:
            break
    return v


def eval_tree_rec(data, nid, tree):
    rows = tree[tree["Node"] == nid]
    for idx, row in rows.iterrows():
        f = row["Feature"]
        if f == "Leaf":
            return row["Gain"]
        else:
            f = int(f[1:])
            diff = data[f] - row["Split"]
            # if diff != 0 and abs(diff) < 0.001: print(row['Feature'],row['Split'],diff)
            if data[f] < row["Split"]:
                child = row["Yes"]
            else:
                child = row["No"]
            child = int(child.split("-")[1])
            return eval_tree_rec(data, child, tree)


def eval_tree(data, tree):
    return eval_tree_rec(data, 0, tree)


def eval_trees(data, num_tree, trees):
    vals = []
    for i in range(0, num_tree):
        tree = trees[(trees["Tree"] == i)]
        v = eval_tree(data, tree)
        vals.append(v)
    return round(1.0 / (1.0 + math.exp(-sum(vals))), 7)
    # return sum(vals)


def eval_trees_compare(data1, data2, num_tree, trees):
    vals = []
    for i in range(0, num_tree):
        tree = trees[(trees["Tree"] == i)]
        v1 = eval_tree(data1, tree)
        v2 = eval_tree(data2, tree)
        if v1 != v2:
            print(i, v1, v2)


def z3_val(v):
    if z3.is_int_value(v):
        return v.numerator_as_long()
    if z3.is_rational_value(v):
        return float(v.numerator_as_long()) / float(v.denominator_as_long())
    assert False


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

    # dump_solver(s,'solver.smt2')
    if r == z3.sat:
        m = s.model()
        return m
        # print( m )
    else:
        pass
        # print(r)
    return None


# --------------------------------------
# Model handel
# --------------------------------------

# model_file = '../models/xgb_sbi-2.0.3.json'

# model_file = '../models/tree_verification_models/diabetes_robust/0020.model'


import sys


# names =[
#     "CREDIT_SUM_4M",
#     "DEBIT_SUM_6M",
#     "CREDIT_CNT_2M",
#     "CREDIT_SUM_5M",
#     "L9M_DR_CR_AMT_RATIO",
#     "CREDIT_CNT_10M",
#     "L2M_DR_CR_AMT_RATIO",
#     "L3M_DR_CR_AMT_RATIO",
#     "ATM_DR_TXN_AMT_7M",
#     "L5M_DR_CR_AMT_RATIO",
#     "AGE",
#     "ATM_DR_TXN_AMT_4M",
#     "DEBIT_CNT_3M",
#     "L6M_DR_CR_AMT_RATIO",
#     "UPI_DR_CNT_11M",
#     "L8M_DR_CR_AMT_RATIO",
#     "LM_DR_CR_AMT_RATIO",
#     "L7M_DR_CR_AMT_RATIO",
#     "L11M_DR_CR_AMT_RATIO"
# ]

# names = names+[""]*100

# operating_range ={
#     "CREDIT_SUM_4M" :(50000,400000), # Two: But unrelated
#     "CREDIT_SUM_5M" :(50000,400000), # Two:
#     "DEBIT_SUM_6M"  :(50000,400000), # Two?

#     "CREDIT_CNT_2M" :(5,50), # Two
#     "DEBIT_CNT_3M"  :(5,50), # two
#     "CREDIT_CNT_10M":(5,50), # One

#     "UPI_DR_CNT_11M":(0,70),

#     "ATM_DR_TXN_AMT_4M":(10,70000), # Two: But unrelated
#     "ATM_DR_TXN_AMT_7M":(10,70000), # Two: Related

#     "LM_DR_CR_AMT_RATIO"  :(.9,11),
#     "L2M_DR_CR_AMT_RATIO" :(.9,11),
#     "L3M_DR_CR_AMT_RATIO" :(.9,11),
#     "L5M_DR_CR_AMT_RATIO" :(.9,11),
#     "L6M_DR_CR_AMT_RATIO" :(.9,11),
#     "L8M_DR_CR_AMT_RATIO" :(.9,11),
#     "L7M_DR_CR_AMT_RATIO" :(.9,11),
#     "L9M_DR_CR_AMT_RATIO" :(.9,11),
#     "L11M_DR_CR_AMT_RATIO":(.9,11),

#     "AGE":(25,50),
# }


# op_range_list = []
# for i,f in enumerate(names):
#     if f in operating_range:
#         op_range_list.append( operating_range[f] )
#     else:
#         op_range_list.append((-10**8,10**8))
# op_range_list.append((0,10**8))


import pickle


def open_model(model_file, max_trees=None):
    model = xgb.Booster({"nthread": 4})  # init model
    try:
        model = pickle.load(open(model_file, "rb"))
    except:
        model.load_model(model_file)  # load data
    # dump_dotty(model)
    trees = model.trees_to_dataframe()
    n_trees = model.num_boosted_rounds()
    n_features = model.num_features()
    dump = model.get_dump(with_stats=True)
    num_classes = len(dump) // (n_trees)

    # index of each tree grows as round*num_classes + class_num starting with round 0
    trees["class"] = trees["Tree"] % num_classes
    trees["Tree"] = trees["Tree"] // num_classes

    tree_depths = []
    for tree in dump:
        lines = tree.split("\n")
        # The depth of the tree is the maximum number of tabs (representing levels) in any line
        max_depth = max(line.count("\t") for line in lines if line.strip() != "")
        tree_depths.append(max_depth)
    depth = max(tree_depths)
    if max_trees is not None and max_trees != -1 and n_trees > max_trees:
        trees = trees[trees["Tree"] < max_trees]
        n_trees = max_trees  # TODO: This does not edit the model, so final solving might not show any unfairness
    data = [
        ("model name", model_file),
        ("total trees", n_trees),
        ("number of classes", num_classes),
        ("number of features", n_features),
        ("max depth", depth),
        ("all trees", len(dump)),
    ]

    for label, value in data:
        print(f"{label}: {value}")
    return model, trees, n_trees, n_features, num_classes


def resave_model(model_file):
    # outfile = model_file[:-5]+'resaved.model'
    # os.system(f"rm {outfile}")
    outfile = model_file[:-5] + "resaved.json"
    model = xgb.Booster({"nthread": 4})  # init model
    model.load_model(model_file)  # load data
    model.save_model(outfile)  # load data
    print(model_file, outfile)


# exit()

# ---------------------------------------------------


def contrib_eval(data):
    data = xgb.DMatrix([data])
    pred_new = model.predict(
        data,
        # iteration_range=(0, 100),
        pred_contribs=True,
        # validate_features=False
    )
    return pred_new


# # shap_values = explainer(data)
# print(prediction)
# # print(shap_values.values)
# # print(shap_values.base_values)
# # print(shap_values.data)
# # prob = round(1.0/(1.0 + math.exp(-sum(pred_new[0]))),7)
# print(pred_new)
# exit()


# for i,f in enumerate(names):
#     if 'RATIO' in f:
#         sliced = trees[ (trees['Feature'] == f'f{i}') ]
#         print( i,f, sliced['Split'].min(), sliced['Split'].max() )

# exit()

# print(sliced)
# exit()


def dump_dotty(model):
    # os.makedirs('tree_dots', exist_ok=True)
    for i in range(model.num_boosted_rounds()):
        tree_dot = xgb.to_graphviz(model, num_trees=i)
        tree_dot.format = "dot"
        tree_dot.render(f"/tmp/tree_{i}")


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
        values = [op_range_list[feature][0]] + sliced["Split"].tolist()
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
        print(Z.shape)
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


# working here
def search_anomaly_for_features(
    features,
    gap,
    precision,
    truelabel,
    n_classes,
    model,
    trees,
    n_trees,
    args,
    op_range_list,
):
    assert n_classes > 2

    gap = int(gap * precision)
    testing = True

    if testing:
        n_trees = 3

    trees = trees[trees["Tree"] < n_trees]

    # have to start here now
    vars1 = {}
    vars2 = {}

    # -------------------------------------------------------
    # Make value for each node
    # -------------------------------------------------------
    if encoding == "allsum":
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
    split_value_map = {}
    ord_bits_cons = []
    for i in range(model.num_features()):
        sliced = trees[(trees["Feature"] == f"f{i}")][["Feature", "Split"]].copy()
        sliced.sort_values(["Split"], inplace=True)
        sliced.drop_duplicates(inplace=True)
        sliced = sliced[
            (op_range_list[i][0] < sliced["Split"])
            & (sliced["Split"] <= op_range_list[i][1])
        ]
        split_bit_map[i] = []
        if op_range_list[i][0]:
            prev = op_range_list[i][0]
        else:
            prev = -(10**8)
        for r, row in sliced.iterrows():
            var_name = f"f{i}" + "_" + str(row["Split"])
            split_bit_map[i].append(var_name)
            split_value_map[var_name] = prev
            prev = float(row["Split"])
        make_bits_for_features(i, "v1", sliced, vars1, ord_bits_cons)
        make_bits_for_features(i, "v2", sliced, vars2, ord_bits_cons)
        split_value_map[f"f{i}" + "_" + str("Last")] = prev
    # print(split_value_map)

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

    def limit_range(d_idx, vars_list, cons):
        for i, b_name in enumerate(split_bit_map[d_idx]):
            value = split_value_map[b_name]
            try:
                if value < limit_range_list[d_idx][0]:
                    for vars in vars_list:
                        cons.append(z3.Not(vars[b_name]))
                if limit_range_list[d_idx][1] <= value:
                    for vars in vars_list:
                        cons.append(vars[b_name])
            except:
                pass

    def all_equal_but_a_few(d_idxs, vars_list, num_features):
        cons = []
        vars1, vars2 = vars_list[0], vars_list[1]
        for idx in range(0, model.num_features()):
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
        if args.small_change:
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
            if sureofcounter:
                val = int(np.floor(p * precision))
            else:
                val = int(np.ceil(p * precision))
            # for i,abst in enumerate(pows):
            #     if val <= abst: break
        else:
            if sureofcounter:
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
        while parent[ancestor]:
            ancestor, cond, _ = parent[ancestor]
            cons.append(z3.Implies(v, cond))

    def is_affected_by_change(row, parent):
        ancestor = row["ID"]
        while parent[ancestor]:
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
        affected = set()
        if up:
            up_name = "u-"
        else:
            up_name = "d-"
        for tid in range(n_trees * n_classes):
            values[tid] = {}
            parent[f"{tid}-0"] = None
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
                leafmapping[bit] = (row["Gain"], row["ID"])
                if val in values[tid]:
                    values[tid][val][1].append(v)
                else:
                    values[tid][val] = (bit, [v])
                if args.ancestor_cons:
                    gen_ancestor_constraints(row, parent, v, cons)
                if not is_affected_by_change(row, parent):
                    unaffected.add(row["ID"])
                else:
                    if up:
                        affected.add((val, bit))
                    else:
                        affected.add((-val, bit))
            elif stop(row, rangemap):
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
                if args.ancestor_cons:
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

        cons += [vars[f"{tid}-0"] for tid in range(n_trees)]  # Root nodes are true
        all_leaves = []
        for i in range(n_classes):
            all_leaves.append([])
        # print(values)
        # exit()
        for tid in range(n_trees):
            curlabel = tid % n_classes
            bits_map = values[tid]
            tree_leaves = []
            for val, (bit, leaves) in bits_map.items():
                if curlabel == truelabel:
                    all_leaves[curlabel].append((val, bit))
                else:
                    all_leaves[curlabel].append((-val, bit))
                tree_leaves.append((1, bit))
                cons.append(z3.Or(leaves) == bit)
            cons.append(z3.PbEq(tree_leaves, 1))
            # for pair in bits: all_leaves.append(pair)
        return cons, all_leaves, affected, unaffected

    # ---------------------------------------
    # Output constrains
    # ---------------------------------------
    ugap = gap
    lgap = -gap

    model = ExtendedBooster(model)
    if args.stop:
        model.compute_node_ranges()
    rangemap = model.node_ranges
    if args.stop:
        stop = (
            lambda row, ran: ran[row["ID"]][1] - ran[row["ID"]][0]
            < args.stop_param * gap / precision
        )
        print("Will be stopping")
    else:
        stop = lambda x, y: False
    model = model.booster
    if encoding == "allsum":
        cs1 = gen_cons_tree(trees, vars1, up=True)
        cs2 = gen_cons_tree(trees, vars2, up=False)
        expr1 = sum([vars1[f"{tid}-0"] for tid in range(n_trees)])
        expr2 = sum([vars2[f"{tid}-0"] for tid in range(n_trees)])
        prop = [(expr1 > ugap), (expr2 < lgap)]
    else:
        cs1, up_leaves, up_affected, unaffected = gen_pb_cons_tree(
            trees, vars1, up=True, rangemap=rangemap, stop=stop
        )
        cs2, down_leaves, down_affected, _ = gen_pb_cons_tree(
            trees, vars2, up=False, rangemap=rangemap, stop=stop
        )
        unchanged = []
        affected_diff = []
        if args.unaffected_cons:
            print(f"{len(unaffected)} leaves are marked as unaffected")
            for leaf in unaffected:
                unchanged.append(vars1[leaf] == vars2[leaf])
        if args.affected_cons:
            if len(list(up_affected) + list(down_affected)) != 0:
                affected_diff = [
                    z3.PbGe(list(up_affected) + list(down_affected), ugap - lgap)
                ]
            else:
                a = z3.Bool("triv")
                affected_diff = [z3.PbEq([(1, a), (1, z3.Not(a))], 0)]
            # print(list(up_affected)+list(down_affected))
        prop = (
            unchanged + affected_diff
        )  # +[z3.PbGe(up_leaves,ugap), z3.PbLe(down_leaves,lgap)]
        # small number, can be changed.
        ugap = 1
        lgap = -1
        orcond = []
        for i in range(n_classes):
            if i == truelabel:
                continue
            prop += [
                z3.PbGe(up_leaves[truelabel] + up_leaves[i], ugap)
            ]  # , z3.PbLe(down_leaves[truelabel] + down_leaves[i], lgap)]
            orcond.append(z3.PbLe(down_leaves[truelabel] + down_leaves[i], lgap))

        prop += [z3.Or(*orcond)]

        # print(len(up_leaves))

    # ---------------------------------------
    # Collect all constraints
    # ---------------------------------------
    aone = all_equal_but_a_few(features, [vars1, vars2], model.num_features())

    # ord_bits_cons = constraints such that if gaurd g is true then g' > g is also true
    # csi is tree encoding
    # aone is equal features constraint
    # prop
    all_cons = ord_bits_cons + cs1 + cs2 + aone + prop

    # prop = [ (expr1 - expr2 > gap)]
    # print(cs1)
    # print(cs2)
    # for c in aone: print(c)
    # if testing:
    #     for c in all_cons: print(c)
    #     exit()

    print("Started solving")
    tic = time.perf_counter()
    if args.solver == "z3":
        m = solve(all_cons)
    elif args.solver == "rounding":
        m = roundingSolve(all_cons)
    elif args.solver == "roundingsoplex":
        m = roundingSolve(all_cons, soplex=True)
    toc = time.perf_counter()
    solvingtime = toc - tic

    if m:
        # prods = list(map(lambda x: bool(m[x[1]])*x[0], up_leaves))
        # print(sum(prods))
        # prods = list(map(lambda x: bool(m[x[1]])*x[0], down_leaves))
        # print(sum(prods))
        # prods = list(map(lambda x: bool(m[x[1]])*leafmapping[x[1]][0], up_leaves))
        # print(sum(prods))
        # prods = list(map(lambda x: bool(m[x[1]])*leafmapping[x[1]][0], down_leaves))
        # print(sum(prods))
        #
        # for i in up_leaves:
        #     if m[i[1]]:
        #         print(leafmapping[i[1]][1], end=", ")
        # print()
        # for i in down_leaves:
        #     if m[i[1]]:
        #         print(leafmapping[i[1]][1], end=", ")
        # print()
        d1 = []
        d2 = []
        for idx in range(0, model.num_features()):
            # if idx == 3:
            #     pdb.set_trace()
            if len(split_bit_map[idx]) == 0:
                v1 = 0
                v2 = 0
            else:
                v1 = f"f{idx}_Last"
                next_v1 = f""
                breaknext = False
                for fname in split_bit_map[idx]:
                    if breaknext:
                        next_v1 = fname
                        break
                    if args.solver == "z3":
                        cond = z3.is_true(m[vars1[fname]])
                    else:
                        cond = m[vars1[fname]]
                    if cond:
                        v1 = fname
                        breaknext = True
                v2 = f"f{idx}_Last"
                next_v2 = ""
                breaknext = False
                for fname in split_bit_map[idx]:
                    if breaknext:
                        next_v2 = fname
                        break
                    if args.solver == "z3":
                        cond = z3.is_true(m[vars2[fname]])
                    else:
                        cond = m[vars2[fname]]
                    if cond:
                        v2 = fname
                        breaknext = True
                # if next_v1 == '':
                #     v1 = split_value_map[v1]+1
                # else:
                #     v1 = (split_value_map[v1]+split_value_map[next_v1])/2
                # if next_v2 == '':
                #     v2 = split_value_map[v2]+1
                # else:
                #     v2 = (split_value_map[v2]+split_value_map[next_v2])/2
                v1 = split_value_map[v1]
                v2 = split_value_map[v2]
            d1.append(v1)
            d2.append(v2)
            # print( f'f{idx}', v1, v2 )
        # print(d2)
        # my1 = eval_trees(d1, n_trees, trees )
        # my2 = eval_trees(d2, n_trees, trees )
        # print(my1,my2)

        return [d1, d2], solvingtime
    else:
        return None, solvingtime

        # print(f'{tid}-0', m[vars1[f'{tid}-0']],m[vars2[f'{tid}-0']])


def main(args):
    encoding = args.encoding

    if args.sure_counterexamples:
        sureofcounter = True

    # Accessing arguments
    solver = args.solver
    close = args.close
    gap = args.gap
    precision = args.precision
    max_trees = args.max_trees
    features = args.features
    debug = args.debug
    truelabel = args.truelabel

    if args.filenum.isdigit():
        model_file = model_files[int(args.filenum)]
    else:
        model_file = args.filenum
    mfile = model_file

    model, trees, n_trees, n_features, n_classes = open_model(mfile, max_trees)

    feature_names = [""] * n_features
    op_range_list = [(-(10**8), 10**8)] * n_features

    # doubt starts
    if args.details:
        details_fname = args.details
        if not os.path.exists(details_fname):
            print(f"Missing : {details_fname}")
            exit()
        details = pd.read_csv(details_fname)
        for index, row in details.iterrows():
            i = row["feature"]
            feature_names[i] = row["name"]
            op_range_list[i] = (row["lb"], row["ub"])

    # doubt ends
    def runner(tupl):
        f = tupl[0]
        precision = tupl[1]
        gap = tupl[2]  # work with this
        truelabel = tupl[3]
        # limit_range_list[10] = (age,age)
        # print(f'Searching for prediction change from {gap} to -{gap} by changing feature {names[f]}:')
        start_time = time.time()
        # working on below thing
        result, solvingtime = search_anomaly_for_features(
            f,
            gap,
            precision,
            truelabel,
            n_classes,
            model,
            trees,
            n_trees,
            args,
            op_range_list,
        )
        print(f"Solving Time = {solvingtime}")
        # pdb.set_trace()
        if result != None:
            if max_trees is not None and max_trees > 0:
                vals = model.predict(xgb.DMatrix(result), ntree_limit=max_trees)
            else:
                vals = model.predict(xgb.DMatrix(result))
            # leaf_indices = model.predict(xgb.DMatrix(result), pred_leaf=True)
            # print(leaf_indices)
            result_copy = result[0].copy()
            for x in f:
                result[0][x] = (result[0][x], result[1][x])
            # print( f, precision, int(time.time() - start_time),sigmoid_inv(vals), result[0] )
            print(f, precision, int(time.time() - start_time), vals, result[0])
            # [ result[1][x] for x in f ]

            if args.plot:
                plot_variations(
                    model, result_copy, f, trees, feature_names, op_range_list
                )

        else:
            print(f, precision, int(time.time() - start_time), "Unsat")

    if features is None:
        features = [0]
    tasks = [(features, precision, gap, truelabel)]
    if args.all_single:
        tasks = [([f], precision, gap, truelabel) for f in range(0, n_features)]
    else:
        tasks = [(features, precision, gap, truelabel)]
    # if debug:
    results = [runner(params) for params in tasks]
    # else:
    #     results = Parallel(n_jobs=-1, timeout=60*60)( delayed(runner)(params) for params in tqdm(tasks) )


if action != "print":
    # for mfile in model_files:
    #     resave_model(mfile)
    # exit()
    parser = argparse.ArgumentParser(
        description="Find sensitivity on any single feature"
    )
    parser.add_argument(
        "filenum",
        help="An integer file number. (Look in utils.py for list of files) or a filename",
    )
    # Truelabel
    parser.add_argument(
        "--truelabel", help="Label of true class, required", type=int, default=-1
    )

    # Add the 'solver' argument with choices
    parser.add_argument(
        "--solver",
        choices=["z3", "rounding", "roundingsoplex", "milp"],
        help="The solver to use. Choose either 'z3' or 'rounding'.",
    )
    parser.add_argument(
        "--encoding",
        choices=["pb", "allsum"],
        help="Encoding to use, choose allsum for naive, pb for psuedoboolean",
        default="pb",
    )

    # Add the 'close' argument which is a boolean (true/false)
    parser.add_argument(
        "--close",
        type=lambda x: x.lower() in ("true", "1"),
        default=False,
        help="Close option, either 'true' or 'false'. Default is 'false'.",
    )
    parser.add_argument(
        "--max_trees",
        type=int,
        default=None,
        help="Maximum number of trees to consider",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="whether to stop when the range of a node becomes less than a threshold",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run serially and stop on pdb statements"
    )
    parser.add_argument(
        "--stop_param",
        type=float,
        default=0.1,
        help="Tunes how aggresssively we fold nodes",
    )
    parser.add_argument(
        "--all_single", action="store_true", help="run on all singular feature sets"
    )

    # parser.add_argument(
    #     '--small_change',
    #     action = 'store_true',
    #     help='restrict small changes in the input, i.e., upto 5 guard flips'
    # )

    parser.add_argument(
        "--small_change",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="restrict small changes in the input, i.e., upto 5 guard flips",
    )
    parser.add_argument(
        "--ancestor_cons",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="constraints encoding leaf implies ancestor visited",
    )
    parser.add_argument(
        "--affected_cons",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="constraints encoding that affected leafs must create enouch change",
    )
    parser.add_argument(
        "--unaffected_cons",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="constraints encoding that unaffected leafs do not change",
    )
    parser.add_argument(
        "--precise",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Do precise analysis, only works for milp",
    )

    parser.add_argument("--plot", action="store_true", help="plot the results")

    parser.add_argument(
        "--sure_counterexamples",
        action="store_true",
        help="Be sure about counterexamples and unsure about fairness",
    )
    parser.add_argument(
        "--gap", type=float, default=1, help="Gap for checking sensitivity"
    )
    parser.add_argument(
        "--precision", type=float, default=100, help="Scale for checking sensitivity"
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=None,
        help="Indexes of the features for which to do sensitivity analysis",
    )

    parser.add_argument(
        "--details",
        type=str,
        default=None,
        help="File containing names of features and their bounds",
    )

    # Parse the arguments
    args = parser.parse_args()
    print(args)
    if args.solver == "milp":
        # milp is left
        milp_solver(args)
    else:
        # working on this
        print("reaching?")
        main(args)
