## Code built on top of the work done in https://github.com/chenhongge/RobustTrees/blob/master/xgbKantchelianAttack.py

import ensemble
import pprint
from gurobipy import *
from sklearn.datasets import load_svmlight_file
from scipy import sparse
import numpy as np
import json
import random
import os
import time
from utils import print_info,print_verbose
from prob import *
import utils
import ast
import data_distance
import numpy as np
from ensemble import Interval
import importlib.util
from pathlib import Path
from pca_constraint import compute_pca_params, evaluate_pca_row_values
# _PCA_PATH = Path(__file__).resolve().parents[1] / "utils" / "pca_constraint.py"
# _PCA_SPEC = importlib.util.spec_from_file_location("pca_constraint", _PCA_PATH)
# _PCA_MOD = importlib.util.module_from_spec(_PCA_SPEC)
# _PCA_SPEC.loader.exec_module(_PCA_MOD)
# compute_pca_params = _PCA_MOD.compute_pca_params
# evaluate_pca_row_values = _PCA_MOD.evaluate_pca_row_values

ROUND_DIGITS = 6

class node_wrapper(object):

    def __init__(
        self,
        treeid,
        nodeid,
        attribute,
        threshold,
        left_leaves,
        right_leaves,
        root=False,
    ):
        # left_leaves and right_leaves are the lists of leaf indices in self.leaf_v_list
        self.attribute = attribute
        self.threshold = threshold
        self.node_pos = []
        self.leaves_lists = []
        self.add_leaves(treeid, nodeid, left_leaves, right_leaves, root)

    def print(self):
        print(
            "node_pos{}, attr:{}, th:{}, leaves:{}".format(
                self.node_pos, self.attribute, self.threshold, self.leaves_lists
            )
        )

    def add_leaves(self, treeid, nodeid, left_leaves, right_leaves, root=False):
        self.node_pos.append({"treeid": treeid, "nodeid": nodeid})
        if root:
            self.leaves_lists.append((left_leaves, right_leaves, "root"))
        else:
            self.leaves_lists.append((left_leaves, right_leaves))

    def add_grb_var(self, node_grb_var, leaf_grb_var_list):
        self.p_grb_var = node_grb_var
        self.l_grb_var_list = []
        for item in self.leaves_lists:
            left_leaf_grb_var = [leaf_grb_var_list[i] for i in item[0]]
            right_leaf_grb_var = [leaf_grb_var_list[i] for i in item[1]]
            if len(item) == 3:
                self.l_grb_var_list.append(
                    (left_leaf_grb_var, right_leaf_grb_var, "root")
                )
            else:
                self.l_grb_var_list.append((left_leaf_grb_var, right_leaf_grb_var))


class milpSolver(object):

    def __init__(
        self,
        model,
        order=np.inf,
        round_digits=ROUND_DIGITS,
        LP=False,
        binary=True,
        pos_json_input=None,
        neg_json_input=None,
        options = None,
    ):
        self.LP = LP
        self.binary = binary or (pos_json_input == None) or (neg_json_input == None)
        self.options = options
        self.n_classes = model.n_classes
        self.multiclass = self.options.multiclass
        #self.strongmulti = self.options.strong_multi
        self.guard_val = (options.ugap-options.lgap)/2 #guard_val
        self.round_digits = round_digits
        self.model = model
        self.base_val = model.get_base_value()
        self.lgap = self.options.lgap
        self.ugap = self.options.ugap

        #Dataset
        if self.options.prob:
            self.X, self.y = getdatafile(self.options.data_file)
            self.pos_mean, self.neg_mean = get_mean(self.X, self.y)
            self.probs, self.guards, self.leaf_data_list = createprobs(model, self.X, self.y,self.round_digits)
        #over

        if self.binary:
            tree_json_str = model.model.get_dump(dump_format="json")
            self.json_file = [json.loads(tree) for tree in tree_json_str]
            if self.options.max_trees is not None:
                self.json_file = self.json_file[: self.options.max_trees * self.n_classes]
            # temp = "temporary{}.json".format(str(round(time.time() * 1000))[-4:])
            # model.model.dump_model(temp, dump_format="json")
            # with open(temp) as f:
            #     if self.options.max_trees is not None:
            #         self.json_file = json.load(f)[: self.options.max_trees * self.n_classes]
            #     else:
            #         self.json_file = json.load(f)
            # if type(self.json_file) is not list:
            #     raise ValueError("model input should be a list of dict loaded by json")
            # else:
            #     os.remove(temp)
        else:
            self.pos_json_file = pos_json_input
            self.neg_json_file = neg_json_input

        self.order = order
        # two nodes with identical decision are merged in this list, their left and right leaves and in the list, third element of the tuple
        self.node_list = []
        self.leaf_v_list = []  # list of all leaf values
        self.leaf_pos_list = []  # list of leaves' position in xgboost model
        self.leaf_class_list = []
        self.leaf_count = [0]  # total number of leaves in the first i trees
        node_check = (
            {}
        )  # track identical decision nodes. {(attr, th):<index in node_list>}
        self.unaffected_leaves = []
        self.affected_leaves = []

        self.varyingFeat = self.options.features
        if self.varyingFeat is None:
            self.varyingFeat = [0]

        def dfs(tree, treeid, root=False, neg=False, unaffected=False):
            if "leaf" in tree.keys():
                if neg:
                    self.leaf_v_list.append(-tree["leaf"])
                else:
                    self.leaf_v_list.append(tree["leaf"])
                self.leaf_class_list.append(treeid % self.n_classes)
                self.leaf_pos_list.append({"treeid": treeid, "nodeid": tree["nodeid"]})
                if unaffected:
                    self.unaffected_leaves.append(len(self.leaf_v_list) - 1)
                else:
                    self.affected_leaves.append(len(self.leaf_v_list) - 1)
                return [len(self.leaf_v_list) - 1]
            else:
                
                attribute, threshold, nodeid = (
                    tree["split"],
                    tree["split_condition"],
                    tree["nodeid"],
                )
                if type(attribute) == str:
                    attribute = int(attribute[1:])

                # threshold = round(threshold, self.round_digits)
                # XGBoost can only offer precision up to 8 digits, however, minimum difference between two splits can be smaller than 1e-8
                # here rounding may be an option, but its hard to choose guard value after rounding
                # for example, if round to 1e-6, then guard value should be 5e-7, or otherwise may cause mistake
                # xgboost prediction has a precision of 1e-8, so when min_diff<1e-8, there is a precision problem
                # if we do not round, xgboost.predict may give wrong results due to precision, but manual predict on json file should always work
                left_subtree = None
                right_subtree = None
                for subtree in tree["children"]:
                    if subtree["nodeid"] == tree["yes"]:
                        left_subtree = subtree
                    if subtree["nodeid"] == tree["no"]:
                        right_subtree = subtree
                if left_subtree == None or right_subtree == None:
                    pprint.pprint(tree)
                    raise ValueError("should be a tree but one child is missing")
                if root:
                    unaffected = True
                if int(tree["split"][1:]) in self.varyingFeat:
                    unaffected = False
                left_leaves = dfs(left_subtree, treeid, False, neg, unaffected)
                right_leaves = dfs(right_subtree, treeid, False, neg, unaffected)
                if (attribute, threshold) not in node_check:
                    self.node_list.append(
                        node_wrapper(
                            treeid,
                            nodeid,
                            attribute,
                            threshold,
                            left_leaves,
                            right_leaves,
                            root,
                        )
                    )
                    node_check[(attribute, threshold)] = len(self.node_list) - 1
                else:
                    node_index = node_check[(attribute, threshold)]
                    self.node_list[node_index].add_leaves(
                        treeid, nodeid, left_leaves, right_leaves, root
                    )
                return left_leaves + right_leaves

        up_cons_all = []
        down_cons_all = []
        self.is_precise = False
        if self.options.precision == 0:
            self.is_precise = True
        if self.binary:
            for i, tree in enumerate(self.json_file):

                dfs(tree, i, root=True)
                new_leaves = self.leaf_v_list[self.leaf_count[-1] :]
                up_cons = {}
                down_cons = {}
                if not self.is_precise:
                    for idx, leaf in enumerate(new_leaves):
                        up_val = int(np.ceil(leaf * self.options.precision))
                        down_val = int(np.floor(leaf * self.options.precision))
                        if up_val not in up_cons.keys():
                            up_cons[up_val] = [self.leaf_count[-1] + idx]
                        else:
                            up_cons[up_val] += [self.leaf_count[-1] + idx]
                        if down_val not in down_cons.keys():
                            down_cons[down_val] = [self.leaf_count[-1] + idx]
                        else:
                            down_cons[down_val] += [self.leaf_count[-1] + idx]
                
                up_cons_all.append(up_cons)
                down_cons_all.append(down_cons)
                

                self.leaf_count.append(len(self.leaf_v_list))
            if len(self.json_file) + 1 != len(self.leaf_count):
                raise ValueError("leaf count error")
        else:
            for i, tree in enumerate(self.pos_json_file):
                dfs(tree, i, root=True)
                self.leaf_count.append(len(self.leaf_v_list))
            for i, tree in enumerate(self.neg_json_file):
                dfs(tree, i + len(self.pos_json_file), root=True, neg=True)
                self.leaf_count.append(len(self.leaf_v_list))
            if len(self.pos_json_file) + len(self.neg_json_file) + 1 != len(
                self.leaf_count
            ):
                raise ValueError("leaf count error")

        self.env = Env(empty=True)
        self.env.setParam('OutputFlag', 0)
        self.env.start()
        self.m = Model("attack",env=self.env)
        self.m.setParam("OutputFlag", 0)
        self.m.setParam("Threads", 1)  #! Number of threads
        self.P = self.m.addVars(len(self.node_list), vtype=GRB.BINARY, name="p")
        self.P2 = self.m.addVars(len(self.node_list), vtype=GRB.BINARY, name="ps")
        self.L = self.m.addVars(len(self.leaf_v_list), lb=0, ub=1, name="l")
        self.L2 = self.m.addVars(len(self.leaf_v_list), lb=0, ub=1, name="ls")
        self.up_vars = []
        self.down_vars = []
        for idx, up_cons in enumerate(up_cons_all):
            self.up_vars.append(
                self.m.addVars((up_cons.keys()), lb=0, ub=1, name=f"up-{idx}")
            )
        for idx, down_cons in enumerate(down_cons_all):
            self.down_vars.append(
                self.m.addVars((down_cons.keys()), lb=0, ub=1, name=f"down-{idx}")
            )
        self.llist = [self.L[key] for key in range(len(self.L))]
        self.llist2 = [self.L2[key] for key in range(len(self.L2))]
        self.plist = [self.P[key] for key in range(len(self.P))]
        self.plist2 = [self.P2[key] for key in range(len(self.P2))]

        # p dictionary by attributes, {attr1:[(threshold1, gurobiVar1),(threshold2, gurobiVar2),...],attr2:[...]}
        self.pdict = {}
            
        for i, node in enumerate(self.node_list):
            node.add_grb_var(self.plist[i], self.llist)
            node.add_grb_var(self.plist2[i], self.llist2)
            if node.attribute not in self.pdict:
                self.pdict[node.attribute] = [
                    (node.threshold, self.plist[i], self.plist2[i])
                ]
            else:
                self.pdict[node.attribute].append(
                    (node.threshold, self.plist[i], self.plist2[i])
                )

        # all but a few features can vary
        for key in self.pdict.keys():
            self.pdict[key].sort(key=lambda tup: tup[0])
            if len(self.pdict[key]) > 1:
                for i in range(len(self.pdict[key]) - 1):
                    self.m.addConstr(
                        self.pdict[key][i][1] <= self.pdict[key][i + 1][1],
                        name="p_consis_attr{}_{}th".format(key, i),
                    )
                    if key in self.varyingFeat:
                        self.m.addConstr(
                            self.pdict[key][i][2] <= self.pdict[key][i + 1][2],
                            name="p_consis_attr{}_{}th_2".format(key, i),
                        )
                    else:
                        self.m.addConstr(
                            self.pdict[key][i][2] == self.pdict[key][i][1],
                            name="p_consis_attr{}_{}th_2".format(key, i),
                        )
            if key not in self.varyingFeat:
                self.m.addConstr(
                    self.pdict[key][-1][2] == self.pdict[key][-1][1],
                    name="p_consis_attr{}_{}th_2".format(key, -1),
                )

        self.x = {}
        self.x2 = {}
        self._add_feature_value_variables()
        self._link_feature_values_to_threshold_bits()
        if self.options.pca_data:
            self._add_pca_constraints()

        # all leaves sum up to 1
        for i in range(len(self.leaf_count) - 1):
            if not self.is_precise:
                t = [self.up_vars[i][j] for j in self.up_vars[i]]
                self.m.addConstr(
                    LinExpr(
                        [1] * (len(self.up_vars[i])),
                        [self.up_vars[i][j] for j in self.up_vars[i]],
                    )
                    == 1,
                    name="leaf_sum_one_for_tree{}".format(i),
                )
                self.m.addConstr(
                    LinExpr(
                        [1] * (len(self.down_vars[i])),
                        [self.down_vars[i][j] for j in self.down_vars[i]],
                    )
                    == 1,
                    name="leaf_sum_one_for_tree{}".format(i),
                )
                up_cons = up_cons_all[i]
                for up_c in up_cons:
                    leaf_vars = [self.llist[j] for j in up_cons[up_c]]
                    for leaf_var in leaf_vars:
                        self.m.addConstr(self.up_vars[i][up_c] >= leaf_var)
                    self.m.addConstr(
                        LinExpr([1] * len(leaf_vars), leaf_vars)
                        >= self.up_vars[i][up_c]
                    )
                down_cons = down_cons_all[i]
                for down_c in down_cons:
                    leaf_vars = [self.llist2[j] for j in down_cons[down_c]]
                    for leaf_var in leaf_vars:
                        self.m.addConstr(self.down_vars[i][down_c] >= leaf_var)
                    self.m.addConstr(
                        LinExpr([1] * len(leaf_vars), leaf_vars)
                        >= self.down_vars[i][down_c]
                    )

            else:
                leaf_vars = [
                    self.llist[j]
                    for j in range(self.leaf_count[i], self.leaf_count[i + 1])
                ]
                self.m.addConstr(
                    LinExpr([1] * (len(leaf_vars)), leaf_vars) == 1,
                    name="leaf_sum_one_for_tree{}".format(i),
                )

                leaf_vars = [
                    self.llist2[j]
                    for j in range(self.leaf_count[i], self.leaf_count[i + 1])
                ]
                self.m.addConstr(
                    LinExpr([1] * (len(leaf_vars)), leaf_vars) == 1,
                    name="leaf_sum_one_for_tree_2{}".format(i),
                )

        if self.options.unaffected_cons:
            print_verbose(self.options, 8, "", f"{len(self.unaffected_leaves)} leaves marked unaffected")
            print_verbose(self.options, 8, "", f"{len(self.llist)} total leaves")
            for i in self.unaffected_leaves:
                self.m.addConstr(self.llist[i] == self.llist2[i])

        # node leaves constraints
        for j in range(len(self.node_list)):
            p = self.plist[j]
            p2 = self.plist2[j]
            for k in range(len(self.node_list[j].leaves_lists)):
                left_l = [self.llist[i] for i in self.node_list[j].leaves_lists[k][0]]
                right_l = [self.llist[i] for i in self.node_list[j].leaves_lists[k][1]]
                left_l2 = [self.llist2[i] for i in self.node_list[j].leaves_lists[k][0]]
                right_l2 = [
                    self.llist2[i] for i in self.node_list[j].leaves_lists[k][1]
                ]
                if len(self.node_list[j].leaves_lists[k]) == 3:
                    self.m.addConstr(
                        LinExpr([1] * len(left_l), left_l) - p == 0,
                        name="p{}_root_left_{}".format(j, k),
                    )
                    self.m.addConstr(
                        LinExpr([1] * len(right_l), right_l) + p == 1,
                        name="p_{}_root_right_{}".format(j, k),
                    )
                    self.m.addConstr(
                        LinExpr([1] * len(left_l2), left_l2) - p2 == 0,
                        name="p{}_root_left_{}_2".format(j, k),
                    )
                    self.m.addConstr(
                        LinExpr([1] * len(right_l2), right_l2) + p2 == 1,
                        name="p_{}_root_right_{}_2".format(j, k),
                    )
                else:
                    self.m.addConstr(
                        LinExpr([1] * len(left_l), left_l) - p <= 0,
                        name="p{}_left_{}".format(j, k),
                    )
                    self.m.addConstr(
                        LinExpr([1] * len(right_l), right_l) + p <= 1,
                        name="p{}_right_{}".format(j, k),
                    )
                    self.m.addConstr(
                        LinExpr([1] * len(left_l2), left_l2) - p2 <= 0,
                        name="p{}_left_{}_2".format(j, k),
                    )
                    self.m.addConstr(
                        LinExpr([1] * len(right_l2), right_l2) + p2 <= 1,
                        name="p{}_right_{}_2".format(j, k),
                    )
        if self.options.in_distro_clauses_file:
            featureDict = self.model.feature_names
            print("featureDict", featureDict)
            revFeatureDict = {v: k for k, v in featureDict.items()}
            ofile = open(self.options.in_distro_clauses_file, "r")
            clauses = ofile.readlines()
  
                
            clauses = [ ast.literal_eval(clause) for clause in clauses]
            for clause in clauses:
                self.add_clause_cons(clause,revFeatureDict)
        self.m.update()

    def _feature_bounds(self, feature_idx):
        low, high = self.model.op_range_list[feature_idx]
        if np.isnan(low) or np.isinf(low):
            low = 0.0
        if np.isnan(high) or np.isinf(high):
            high = 1.0
        return float(low), float(high)

    def _add_feature_value_variables(self):
        for feature_idx in range(self.model.n_features):
            low, high = self._feature_bounds(feature_idx)
            self.x[feature_idx] = self.m.addVar(
                lb=low, ub=high, vtype=GRB.CONTINUOUS, name=f"x_{feature_idx}"
            )
            self.x2[feature_idx] = self.m.addVar(
                lb=low, ub=high, vtype=GRB.CONTINUOUS, name=f"x2_{feature_idx}"
            )

    def _link_one_feature_to_threshold_bits(self, feature_idx, x_var, p_index):
        if feature_idx not in self.pdict:
            return

        low, high = self._feature_bounds(feature_idx)
        span = high - low
        if span <= 0:
            span = 1.0

        for threshold, p_var, p_var2 in self.pdict[feature_idx]:
            bit_var = p_var if p_index == 1 else p_var2
            self.m.addConstr(
                x_var <= threshold + span * (1 - bit_var),
                name=f"x_le_th_f{feature_idx}_{threshold}_{p_index}",
            )
            self.m.addConstr(
                x_var >= threshold - span * bit_var,
                name=f"x_ge_th_f{feature_idx}_{threshold}_{p_index}",
            )

    def _link_feature_values_to_threshold_bits(self):
        for feature_idx in range(self.model.n_features):
            self._link_one_feature_to_threshold_bits(feature_idx, self.x[feature_idx], 1)
            self._link_one_feature_to_threshold_bits(feature_idx, self.x2[feature_idx], 2)

    def _add_pca_constraints(self):
        pca = compute_pca_params(
            csv_path=self.options.pca_data,
            center=True,
            verbose=(self.options.verbosity > 0),
        )

        ImP = pca["ImP"]
        eps = float(pca["epsilon"])
        print("eps for PCA constraints:", eps)
        mean = pca["mean"]
        feature_names = pca["feature_names"]

        feature_indices = []
        for fname in feature_names:
            if isinstance(fname, str) and fname.startswith("f"):
                feature_idx = int(fname[1:])
            elif isinstance(fname, str) and fname.isdigit():
                feature_idx = int(fname)
            else:
                raise ValueError(f"Unsupported PCA feature name: {fname}")
            if feature_idx < 0 or feature_idx >= self.model.n_features:
                raise ValueError(f"PCA feature {fname} is outside model range")
            feature_indices.append(feature_idx)

        for row in range(len(feature_indices)):
            expr1 = LinExpr()
            expr2 = LinExpr()
            for col in range(len(feature_indices)):
                coeff = float(ImP[row][col])
                feature_idx = feature_indices[col]
                expr1 += coeff * (self.x[feature_idx] - float(mean[col]))
                expr2 += coeff * (self.x2[feature_idx] - float(mean[col]))

            self.m.addConstr(
                expr1 <= eps,
                name=f"pca_up_row_{row}",
            )
            self.m.addConstr(
                expr1 >= -eps,
                name=f"pca_low_row_{row}",
            )
            self.m.addConstr(
                expr2 <= eps,
                name=f"pca_up_row2_{row}",
            )
            self.m.addConstr(
                expr2 >= -eps,
                name=f"pca_low_row2_{row}",
            )

        print_verbose(
            self.options,
            0,
            "[PCA-MILP]",
            f"Added {4 * len(feature_indices)} PCA constraints over mapped features {feature_indices}",
        )
    
    
    def add_clause_cons(self, clause,revFeatureDict):
        cons = []
        for (is_pos,v,g) in clause:
            feat = revFeatureDict[v]
            if feat in self.varyingFeat:
                return
            if feat not in self.pdict:
                continue
            thresholds = self.pdict[feat]
            litscon = 1
            temp = []
            for thres in thresholds:
                if is_pos:
                    if g<=thres[0]:
                        litscon = thres[1]
                        break
                else:
                    if thres[0]<g:
                        temp.append(thres[1])
                    else:
                        break
            if is_pos:
                cons.append(litscon)
            else:
                if len(temp) == 0:
                    litscon = 0
                else:
                    litscon = temp[-1]
                cons.append((1-litscon))
            
        if cons:
            self.m.addConstr(
                quicksum(cons) >= 1,
                name=f"clause_cons_{hash(str(clause))}",
            )

    def local_check_update_range(self,sample, op_range_list):
        op_range_list2=[]
        for i in range(0, self.model.n_features):
            if  (i in self.varyingFeat):
                op_range_list2.append(op_range_list[i])
                continue
            list_item=(sample[i]-self.options.perturb,sample[i]+self.options.perturb)
            if math.isnan(op_range_list[i][0]) or math.isnan(op_range_list[i][1]):
                op_range_list2.append(list_item)
            elif max(list_item[0],op_range_list[i][0])<=min(list_item[1],op_range_list[i][1]):
                list_item2=(max(list_item[0],op_range_list[i][0]),min(list_item[1],op_range_list[i][1]))
                op_range_list2.append(list_item2)
            else:
                op_range_list2.append(op_range_list[i])
        return op_range_list2
    
                    
    def attack(self,options):
        print_verbose( self.options, 5, "", "Starting to solve" )
        # print("\n==================================")


        if self.options.affected_cons:
            if self.multiclass:
                class_lists = []
                for i in range(self.n_classes):
                    class_lists.append(np.array(self.leaf_class_list) == i)
                if self.options.truelabel == -1:
                    pass
                else:
                    if self.options.otherlabel == -1:
                        pass
                    else:
                        if len(self.affected_leaves) > 0: 
                            print_verbose( self.options, 7, "", "Applying Affected cons" )
                            valid_true = np.array(self.affected_leaves)[
                                np.array(class_lists[self.options.truelabel])[
                                    self.affected_leaves
                                ]
                            ]
                            valid_other = np.array(self.affected_leaves)[
                                np.array(class_lists[self.options.otherlabel])[
                                    self.affected_leaves
                                ]
                            ]
                            self.m.addConstr(
                                LinExpr(
                                    np.array(self.leaf_v_list)[valid_true],
                                    np.array(self.llist)[valid_true],
                                )
                                - LinExpr(
                                    np.array(self.leaf_v_list)[valid_other],
                                    np.array(self.llist)[valid_other],
                                )
                                - LinExpr(
                                    np.array(self.leaf_v_list)[valid_true],
                                    np.array(self.llist2)[valid_true],
                                )
                                + LinExpr(
                                    np.array(self.leaf_v_list)[valid_other],
                                    np.array(self.llist2)[valid_other],
                                )
                                >= self.ugap-self.lgap, #2 * self.guard_val,
                                name="affected",
                            )

            else:
                print_verbose( self.options, 7, "", "Applying Affected cons" )
                if len(self.affected_leaves) > 0:
                    self.m.addConstr(
                        LinExpr(
                            np.array(self.leaf_v_list)[self.affected_leaves],
                            np.array(self.llist)[self.affected_leaves],
                        )
                        - LinExpr(
                            np.array(self.leaf_v_list)[self.affected_leaves],
                            np.array(self.llist2)[self.affected_leaves],
                        )
                        >= self.ugap-self.lgap, #2 * self.guard_val,
                        name="affected",
                    )
        up_weights = self.leaf_v_list
        down_weights = self.leaf_v_list
        if not self.is_precise:
            all_up_vals = []
            all_up_variables = []
            for d in self.up_vars:
                for key, val in d.items():
                    all_up_vals.append(key)
                    all_up_variables.append(val)
            all_down_vals = []
            all_down_variables = []
            for d in self.down_vars:
                for key, val in d.items():
                    all_down_vals.append(key)
                    all_down_variables.append(val)
            if self.multiclass:
                #--------------------------
                # Multiclass model
                #--------------------------
                all_up_vals = np.array(all_up_vals)
                all_down_vals = np.array(all_down_vals)
                all_up_variables = np.array(all_up_variables)
                all_down_variables = np.array(all_down_variables)
                class_lists = []
                for i in range(self.n_classes):
                    class_lists.append(np.array(self.leaf_class_list) == i)
                if self.options.truelabel == -1:
                    pass
                else:
                    if self.options.otherlabel == -1:
                        pass
                    else:
                        for i in range(self.n_classes):
                            if i == self.options.truelabel:
                                continue
                            self.m.addConstr(
                                LinExpr(
                                    all_up_vals[class_lists[self.options.truelabel]],
                                    all_up_variables[
                                        class_lists[self.options.truelabel]
                                    ],
                                )
                                - LinExpr(
                                    all_up_vals[class_lists[i]],
                                    all_up_variables[class_lists[i]],
                                )
                                >= (self.ugap-self.lgap) * self.options.precision,
                                name=f"mislabel_{i}",
                            )
                        for i in range(self.n_classes):
                            if i == self.options.otherlabel:
                                continue
                            if self.options.strong_multi or i == self.options.truelabel:
                                self.m.addConstr(
                                    LinExpr(
                                        all_down_vals[
                                            class_lists[self.options.otherlabel]
                                        ],
                                        all_down_variables[
                                            class_lists[self.options.otherlabel]
                                        ],
                                    )
                                    - LinExpr(
                                        all_down_vals[class_lists[i]],
                                        all_down_variables[class_lists[i]],
                                    )
                                    >= (self.ugap-self.lgap), # self.guard_val, # Why there is no multiplication with precision?
                                    name=f"mislabel2_{i}",
                                )
                            else:
                                self.m.addConstr(
                                    LinExpr(
                                        all_down_vals[
                                            class_lists[self.options.otherlabel]
                                        ],
                                        all_down_variables[
                                            class_lists[self.options.otherlabel]
                                        ],
                                    )
                                    - LinExpr(
                                        all_down_vals[class_lists[i]],
                                        all_down_variables[class_lists[i]],
                                    )
                                    >= 0,
                                    name=f"mislabel2_{i}",
                                )

                        if self.options.objective:
                            self.m.setObjective(
                                LinExpr(
                                    all_up_vals[class_lists[self.options.truelabel]],
                                    all_up_variables[
                                        class_lists[self.options.truelabel]
                                    ],
                                )
                                - LinExpr(
                                    all[class_lists[self.options.otherlabel]],
                                    self.llist2[class_lists[self.options.otherlabel]],
                                ),
                                GRB.MAXIMIZE,
                            )
            else:
                #--------------------------
                # Binary model
                #--------------------------
                self.m.addConstr(
                    LinExpr(all_up_vals, all_up_variables) + self.base_val
                    >= self.ugap * self.options.precision, # self.guard_val
                    name="mislabel",
                )
                self.m.addConstr(
                    LinExpr(all_down_vals, all_down_variables) + self.base_val
                    <=  self.lgap * self.options.precision, #-self.guard_val
                    name="mislabel-2",
                )
                if self.options.objective:
                    self.m.setObjective(
                        LinExpr(all_up_vals, all_up_variables)
                        - LinExpr(all_down_vals, all_down_variables),
                        GRB.MAXIMIZE,
                    )
        else:
            if self.multiclass:
                #--------------------------
                # Multiclass model (without precision)
                #--------------------------
                up_weights = np.array(up_weights)
                down_weights = np.array(down_weights)
                self.llist = np.array(self.llist)
                self.llist2 = np.array(self.llist2)
                class_lists = []
                for i in range(self.n_classes):
                    class_lists.append(np.array(self.leaf_class_list) == i)

                if self.options.truelabel == -1:
                    pass
                else:
                    if self.options.otherlabel == -1:
                        pass
                    else:
                        for i in range(self.n_classes):
                            if i == self.options.truelabel:
                                continue
                            self.m.addConstr(
                                LinExpr(
                                    up_weights[class_lists[self.options.truelabel]],
                                    self.llist[class_lists[self.options.truelabel]],
                                )
                                - LinExpr(
                                    up_weights[class_lists[i]],
                                    self.llist[class_lists[i]],
                                )
                                >= (self.ugap-self.lgap), #self.guard_val,
                                name=f"mislabel_{i}",
                            )
                        for i in range(self.n_classes):
                            if i == self.options.otherlabel:
                                continue
                            self.m.addConstr(
                                LinExpr(
                                    down_weights[class_lists[self.options.otherlabel]],
                                    self.llist2[class_lists[self.options.otherlabel]],
                                )
                                - LinExpr(
                                    down_weights[class_lists[i]],
                                    self.llist2[class_lists[i]],
                                )
                                >= (self.ugap-self.lgap), #self.guard_val,
                                name=f"mislabel_{i}",
                            )
                            # if self.options.strong_multi or i == self.options.truelabel:
                            #     self.m.addConstr(
                            #         LinExpr(
                            #             down_weights[
                            #                 class_lists[self.options.otherlabel]
                            #             ],
                            #             self.llist2[
                            #                 class_lists[self.options.otherlabel]
                            #             ],
                            #         )
                            #         - LinExpr(
                            #             down_weights[class_lists[i]],
                            #             self.llist2[class_lists[i]],
                            #         )
                            #         >= self.guard_val,
                            #         name=f"mislabel2_{i}",
                            #     )
                            #     print(i)
                            # else:
                            #     self.m.addConstr(
                            #         LinExpr(
                            #             down_weights[
                            #                 class_lists[self.options.otherlabel]
                            #             ],
                            #             self.llist2[
                            #                 class_lists[self.options.otherlabel]
                            #             ],
                            #         )
                            #         - LinExpr(
                            #             down_weights[class_lists[i]],
                            #             self.llist2[class_lists[i]],
                            #         )
                            #         >= 0,
                            #         name=f"mislabel2_{i}",
                            #     )

                        if self.options.objective:
                            self.m.setObjective(
                                LinExpr(
                                    up_weights[class_lists[self.options.truelabel]],
                                    self.llist[class_lists[self.options.truelabel]],
                                )
                                - LinExpr(
                                    down_weights[class_lists[self.options.otherlabel]],
                                    self.llist2[class_lists[self.options.otherlabel]],
                                )
                                - LinExpr(
                                    down_weights[class_lists[self.options.truelabel]],
                                    self.llist2[class_lists[self.options.truelabel]],
                                )
                                + LinExpr(
                                    up_weights[class_lists[self.options.otherlabel]],
                                    self.llist[class_lists[self.options.otherlabel]],
                                ),
                                GRB.MAXIMIZE,
                            )
            else:
                #--------------------------
                # Binary model
                #--------------------------
                self.m.addConstr(
                    LinExpr(up_weights, self.llist) + self.base_val >= self.ugap, name="mislabel" # self.guard_val
                )
                self.m.addConstr(
                    LinExpr(down_weights, self.llist2) +self.base_val <= self.lgap, #-self.guard_val,
                    name="mislabel-2",
                )
                print_verbose( self.options, 5, "", f"base_val : {self.base_val} guard :{self.guard_val}" )
                if self.options.prob:
                    diffs = []
                    upvarobj = []
                    downvarobj = []
                    eps = 1e-30
                    
                    for key in self.probs.keys():
                        keys = list(self.probs[key].keys())
                        if key not in self.pdict:
                            continue
                        max_terms = min(len(keys) - 1, len(self.pdict[key]))
                        for i in range(max_terms):
                            diffs.append(np.log(self.probs[key][keys[i]] + eps)-np.log(self.probs[key][keys[i+1]]+eps) )
                            upvarobj.append(self.pdict[key][i][1])
                            downvarobj.append(self.pdict[key][i][2])
                    if self.options.objective:
                        self.m.setObjective(
                            LinExpr(np.array(diffs), np.array(upvarobj))
                            + LinExpr(np.array(diffs), np.array(downvarobj)) ,
                            GRB.MAXIMIZE,
                        )
                    else:
                        self.m.setObjective(LinExpr(np.array(diffs), np.array(upvarobj))
                            +LinExpr(np.array(diffs), np.array(downvarobj)) ,
                            GRB.MAXIMIZE,
                        )
                else:
                    if self.options.objective:
                        self.m.setObjective(
                            LinExpr(up_weights, self.llist)
                            - LinExpr(down_weights, self.llist2),
                            GRB.MAXIMIZE,
                        )

        self.m.update()
        self.m.setParam("TimeLimit", 60 * 60)
        # self.m.setParam("SolutionLimit", 1)
        if self.options.prob:
            self.m.setParam(GRB.Param.PoolSolutions, 1)  # Get up to 10 solutions
        else:
            self.m.setParam("SolutionLimit", 1)
        

        tic = time.perf_counter()
        if self.options.verbosity > 8:
            # Save the MILP constraints to a file
            constraint_file = "/tmp/milp_constraints.lp"
            self.m.write(constraint_file)
            print_verbose( self.options, 8, "MILP constraints saved", f" {constraint_file}" )
        
        if self.local_sample:
            local_range = self.local_check_update_range(self.local_sample, self.model.op_range_list)
            for key in self.pdict.keys():
                lo, hi = local_range[key]
                for (threshold, p_var, p2_var) in self.pdict[key]:
                    # p=1 means x <= threshold, p=0 means x > threshold
                    if hi <= threshold:       
                        self.m.addConstr(p_var == 1)
                        self.m.addConstr(p2_var == 1)
                    elif lo > threshold:      
                        self.m.addConstr(p_var == 0)
                        self.m.addConstr(p2_var == 0)
                    else:
                        pass
        
        
        self.m.optimize()
        toc = time.perf_counter()
        timetaken = toc - tic
        print_verbose( self.options, 3, 'Time', f" {(toc - tic)} seconds" )
        if self.m.status == GRB.Status.INFEASIBLE:
            print("Insensitive",self.varyingFeat)
            # print(self.varyingFeat,(toc-tic),"Insensitive")
            print_info('Time', timetaken)

            return False

        if self.m.status == GRB.Status.TIME_LIMIT:
            print_verbose( self.options, 3, 'Time', f"Timeout" )
            return False
        
        print_info('Sensitive features', f"{self.varyingFeat}" )
        print_info('Time', timetaken)
        
        x =[0] * self.model.n_features
        x2 =[0] * self.model.n_features
        # -- intializing region pair --
        region1 = []
        region2 = []

        for f in range(self.model.n_features):
            if hasattr(self.model,"op_range_list"):
                oprange = self.model.op_range_list[f]
                low,high = (oprange[0],oprange[1])
            else:
                low, high = -np.inf,np.inf
            region1.append(Interval('(',low,high,')'))
            region2.append(Interval('(',low,high,')'))


        for i in range(1):
            self.m.setParam(GRB.Param.SolutionNumber, i)
            print_verbose(self.options, 5, "", f"\nSolution {i+1}")
            print_verbose(self.options, 5, "", f"Objective Value: {self.m.PoolObjVal}")
            #-------------------------------------------------------------
            #  DO NOT REMOVE THIS COMMENTED CODE
            #-----------------------------------------------------------
            # for key in self.pdict.keys():
            #     trees = self.model.trees
            #     splits = trees[trees['Feature'] == f'f{key}']['Split']
            #     vals1 = [node[0] for node in self.pdict[key] if node[1].x > 0.5] + [splits.max()+1]
            #     x[key] = (
            #         vals1[0] + ([splits.min()-1] + [node[0] for node in self.pdict[key]])[-len(vals1)]
            #     ) / 2
            #     vals2 = [node[0] for node in self.pdict[key] if node[2].x > 0.5] + [splits.max()+1]
            #     x2[key] = (
            #         vals2[0] + ([splits.min()-1] + [node[0] for node in self.pdict[key]])[-len(vals2)]
            #     ) / 2

            # x = np.array(x)
            # x2 = np.array(x2)
            # pred1 = self.model.predict([x])#,pred_leaf=True)#[0][0]
            # pred2 = self.model.predict([x2])#,pred_leaf=True) #[0][0]
            
            #-----------region pair-------------------------------------------
            default_open = '[' if self.model.split_kind == '<' else '('
            defaul_close = ')' if self.model.split_kind == '<' else ']'
            for key in self.pdict.keys():
                trees = self.model.trees
                splits = trees[trees['Feature'] == f'f{key}']['Split']
                low, high = self.model.op_range_list[key]
                vals1 = [node[0] for node in self.pdict[key] if node[1].x > 0.5] + [high]
                reg1 = [([low] + [node[0] for node in self.pdict[key]])[-len(vals1)] ,vals1[0]]

                vals2 = [node[0] for node in self.pdict[key] if node[2].x > 0.5] + [high]
                reg2 = [([low] + [node[0] for node in self.pdict[key]])[-len(vals2)] ,vals2[0]]
                
                region1[key]=Interval('(' if reg1[0] == low else default_open,
                                      reg1[0],
                                      reg1[1],
                                      ')' if reg1[1] == high else defaul_close)
                region2[key]=Interval('(' if reg2[0] == low else default_open,
                                      reg2[0],
                                      reg2[1],
                                      ')' if reg2[1] == high else defaul_close)
            #--------------------------------------------------------------------------------------
        utils.print_verbose(options,0,"region1",self.model.print_reg(region1))
        utils.print_verbose(options,0,"region2",self.model.print_reg(region2))
        xbound = {}
        x2bound = {}
        features = list(self.model.feature_names.keys())
        op_range_list = self.model.op_range_list
        # print(op_range_list)
        
        active_leaf1 = []
        active_leaf2 = []
        sum1 =0
        sum2 =0

        if self.options.verbosity > 9:
            for i in range(len(self.llist)):
                if self.llist[i].x > 0.5:
                    active_leaf1.append((i,self.leaf_v_list[i]))
                    sum1 += self.leaf_v_list[i]
                if self.llist2[i].x > 0.5:
                    active_leaf2.append((i,self.leaf_v_list[i]))
                    sum2 += self.leaf_v_list[i]
            print_verbose(self.options, 9, f"diff_sum",f"{sum1-sum2}")
            print_verbose(self.options, 9, "", f"sigmoid(x+base_val): {1/(1+np.exp(-(sum1+self.base_val) ))}, sigmoid(x2+base_val): {1/(1+np.exp(-(sum2 + self.base_val)))}")
        
        op_range_list = []
        for j in range(self.model.n_features):
            splits = trees[trees['Feature'] == f'f{j}']['Split']
            min_val, max_val = splits.min(), splits.max()
            if pd.isna(min_val) or pd.isna(max_val):
                op_range_list.append((0, 1))
            else:
                op_range_list.append((float(min_val - 1), float(max_val + 1)))
        
        # print("----------",op_range_list)
        
        for i,key in enumerate(features):
            xbound[key] = tuple(op_range_list[i])
            x2bound[key] = tuple(op_range_list[i])
            # print(f"Feature {key} bounds: {xbound[key]} {x2bound[key]} op_range: {op_range_list[i]}")

        active1_set = {act[0] for act in active_leaf1}
        active2_set = {act[0] for act in active_leaf2}

        for node in self.node_list:
            attr, thres = node.attribute, node.threshold
            for left, right, *_ in node.leaves_lists:
                # print(f"left:{left}\n right{right}")
                left_set = set(left)
                right_set = set(right)
                if left_set & active1_set:
                    lb, ub = xbound[attr]
                    xbound[attr] = (lb, min(ub, thres))
                    # xbound[attr][1] = min(xbound[attr][1], thres)
                elif right_set & active1_set:
                    lb, ub = xbound[attr]
                    xbound[attr] = (max(lb, thres), ub)
                    # xbound[attr][0] = max(xbound[attr][0], thres)
                else: pass

                if left_set & active2_set:
                    lb2, ub2 = x2bound[attr]
                    x2bound[attr] = (lb2, min(ub2, thres))
                    # x2bound[attr][1] = min(x2bound[attr][1], thres)
                elif right_set & active2_set:
                    lb2, ub2 = x2bound[attr]
                    x2bound[attr] = (max(lb2, thres), ub2)
                    # x2bound[attr][0] = max(x2bound[attr][0], thres)
                else: pass

        if self.options.compute_data_distance:
            data_distance.compute_data_distance(xbound,self.varyingFeat,self.model.feature_names,self.model.n_features,self.model.trees,options)
            # data_distance.compute_data_distance(xbound,self.varyingFeat,self.model.feature_names,self.model.n_features,self.model.trees,options,dist_type='L0')
            # data_distance.compute_data_distance(xbound,self.varyingFeat,self.model.feature_names,self.model.n_features,self.model.trees,options,dist_type='L1')
            # data_distance.compute_data_distance(xbound,self.varyingFeat,self.model.feature_names,self.model.n_features,self.model.trees,options,dist_type='L2')
            # data_distance.compute_data_distance(xbound,self.varyingFeat,self.model.feature_names,self.model.n_features,self.model.trees,options,dist_type='Linf')
        
        x = self.model.region2point(region1)
        x2 = self.model.region2point(region2)
        if self.options.pca_data:
            pca = compute_pca_params(
                csv_path=self.options.pca_data,
                center=True,
                verbose=(self.options.verbosity > 0),
            )
            feature_indices = []
            for fname in pca["feature_names"]:
                if isinstance(fname, str) and fname.startswith("f"):
                    feature_indices.append(int(fname[1:]))
                elif isinstance(fname, str) and fname.isdigit():
                    feature_indices.append(int(fname))
            point1_pca = []
            point2_pca = []
            for idx in feature_indices:
                point1_pca.append(x[idx])
                point2_pca.append(x2[idx])
            milp_x = []
            milp_x2 = []
            for idx in feature_indices:
                milp_x.append(self.x[idx].X)
                milp_x2.append(self.x2[idx].X)
            milp_expr1_vals = evaluate_pca_row_values(milp_x, pca["ImP"], pca["mean"])
            milp_expr2_vals = evaluate_pca_row_values(milp_x2, pca["ImP"], pca["mean"])
            print("[PCA] milp x (dataset dims) =", milp_x)
            print("[PCA] milp x2 (dataset dims) =", milp_x2)
            print("[PCA] expr1(milp x) =", milp_expr1_vals)
            print("[PCA] expr2(milp x2) =", milp_expr2_vals)
            expr1_vals = evaluate_pca_row_values(point1_pca, pca["ImP"], pca["mean"])
            expr2_vals = evaluate_pca_row_values(point2_pca, pca["ImP"], pca["mean"])
            print("[PCA] expr1(region1 point) =", expr1_vals)
            print("[PCA] expr2(region2 point) =", expr2_vals)
        pred1 = self.model.predict([x])
        pred2 = self.model.predict([x2])
        # print(x)
        # print(x2)
        res = []
        x_colored = []
        x2_colored = []
        differentfeature = []
        for i in range(len(x)):
            if x[i] != x2[i]:
                res.append((x[i], x2[i]))
                x_colored.append(f"\033[91m{x[i]}\033[0m")
                x2_colored.append(f"\033[91m{x2[i]}\033[0m")
                differentfeature.append(i)
            else:
                res.append(x[i])
                x_colored.append(str(x[i]))
                x2_colored.append(str(x2[i]))
        
        # print("Inputs ", res)
        utils.print_array( 'Sensitive sample 1:', x_colored)
        utils.print_array( 'Sensitive sample 2:', x2_colored)
        # with open("res.pkl", "wb") as f:
        #     pickle.dump((x, x2), f)
        
        for i in range(0,len(x)):
            if(x[i] == 0): x[i]   = 0.0000000001
            if(x2[i] == 0): x2[i] = 0.0000000001
        # pred1 = self.model.predict([x])#,pred_leaf=True)#[0][0]
        # pred2 = self.model.predict([x2])#,pred_leaf=True) #[0][0]

        print(f"Output Values: {pred1} {pred2}")
        return True

    def check(self, x, json_file):
        # Due to XGBoost precision issues, some attacks may not succeed if tested using model.predict.
        # We manually run the tree on the json file here to make sure those attacks are actually successful.
        print("-------------------------------------\nstart checking")
        print("manually run trees")
        leaf_values = []
        for item in json_file:
            tree = item.copy()
            while "leaf" not in tree.keys():
                attribute, threshold, nodeid = (
                    tree["split"],
                    tree["split_condition"],
                    tree["nodeid"],
                )
                if type(attribute) == str:
                    attribute = int(attribute[1:])
                if x[attribute] < threshold:
                    if tree["children"][0]["nodeid"] == tree["yes"]:
                        tree = tree["children"][0].copy()
                    elif tree["children"][1]["nodeid"] == tree["yes"]:
                        tree = tree["children"][1].copy()
                    else:
                        pprint.pprint(tree)
                        print("x[attribute]:", x[attribute])
                        raise ValueError("child not found")
                else:
                    if tree["children"][0]["nodeid"] == tree["no"]:
                        tree = tree["children"][0].copy()
                    elif tree["children"][1]["nodeid"] == tree["no"]:
                        tree = tree["children"][1].copy()
                    else:
                        pprint.pprint(tree)
                        print("x[attribute]:", x[attribute])
                        raise ValueError("child not found")
            leaf_values.append(tree["leaf"])
        manual_res = np.sum(leaf_values)
        print("leaf values:{}, \nsum:{}".format(leaf_values, manual_res))
        return manual_res


def main(options):
    
    random.seed(8)
    np.random.seed(8)

    #---------------------------------------
    # Load model
    #---------------------------------------
    e = ensemble.Ensemble(options)
    e.load(print_vitals = True)
    model = e
    sens_sets = [options.features]
    if options.all_single: sens_sets = [ [f] for f in range(0, model.n_features) ]
    if options.local_check_samples:
        op_range_list = model.op_range_list
        if len(op_range_list) != len(options.local_check_samples[0]):
            print('Error: number of inputs does not match model!')
            exit()
        # for f in sens_sets:
        results = []
        idx = 0
        for sample in options.local_check_samples:
            for f in sens_sets:
                options.features = f
                utils.print_verbose(options, -1, "--==> Query", f"{idx}")
                solver = milpSolver(model, options=options)
                solver.local_sample = sample          # <-- pass sample in
                results.append(solver.attack(options))
                idx += 1
        utils.print_verbose( options, -1, "Fraction of sensitive queries:", f"{sum(results)}/{len(results)}")
    else:
        utils.print_verbose(options, -1, "--==> Query", f"0")
        solver = milpSolver(
            model,
            options=options,
        )
        solver.local_sample = None
        solver.attack(options)
        
    # solver = milpSolver(
    #     model,
    #     options=options,
    # )

    # global_start = time.time()
    # solver.attack(options)
