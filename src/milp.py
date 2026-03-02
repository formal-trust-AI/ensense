## Code built on top of the work done in https://github.com/chenhongge/RobustTrees/blob/master/xgbKantchelianAttack.py

import ensemble

import pprint
from joblib import Parallel, delayed
import pdb
from gurobipy import *
from sklearn.datasets import load_svmlight_file
from scipy import sparse
import numpy as np
import json
import math
import random
import os
import xgboost as xgb
import time
import argparse
from utils import model_files, open_model, sigmoid, sigmoid_inv
import pickle
from prob import *
import utils
import ast
import data_distance

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
        guard_val=0.67,
        round_digits=ROUND_DIGITS,
        LP=False,
        binary=True,
        pos_json_input=None,
        neg_json_input=None,
        varyingFeat=[],
        debug=False,
        args={},
        options = None,
    ):
        self.LP = LP
        self.binary = binary or (pos_json_input == None) or (neg_json_input == None)
        self.debug = debug
        self.args = args
        self.options = options
        self.n_classes = model.n_classes
        self.multiclass = self.options.multiclass
        self.strongmulti = self.options.strong_multi
        self.guard_val = (options.ugap-options.lgap)/2 #guard_val
        self.round_digits = round_digits
        self.model = model
        self.base_val = model.get_base_value()
        self.lgap = self.options.lgap
        self.ugap = self.options.ugap
        

        #Dataset
        if args["prob"]:
            self.X, self.y = getdatafile(args["data_file"])
            self.pos_mean, self.neg_mean = get_mean(self.X, self.y)
            self.probs, self.guards, self.leaf_data_list = createprobs(model, self.X, self.y,self.round_digits)
        #over

        if self.binary:
            temp = "temporary{}.json".format(str(round(time.time() * 1000))[-4:])
            model.model.dump_model(temp, dump_format="json")
            with open(temp) as f:
                if args["max_trees"] is not None:
                    self.json_file = json.load(f)[: args["max_trees"] * self.n_classes]
                else:
                    self.json_file = json.load(f)
            if type(self.json_file) is not list:
                raise ValueError("model input should be a list of dict loaded by json")
            else:
                os.remove(temp)
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

        self.varyingFeat = varyingFeat
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

                threshold = round(threshold, self.round_digits)
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
        if self.args["precision"] == 0:
            self.args["precise"] = True
        if self.binary:
            for i, tree in enumerate(self.json_file):

                dfs(tree, i, root=True)
                new_leaves = self.leaf_v_list[self.leaf_count[-1] :]
                up_cons = {}
                down_cons = {}
                if not self.args["precise"]:
                    for idx, leaf in enumerate(new_leaves):
                        up_val = int(np.ceil(leaf * self.args["precision"]))
                        down_val = int(np.floor(leaf * self.args["precision"]))
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

        self.m = Model("attack")
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

        # all leaves sum up to 1
        for i in range(len(self.leaf_count) - 1):
            if not self.args["precise"]:
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
            print(f"{len(self.unaffected_leaves)} leaves marked unaffected")
            print(f"{len(self.llist)} total leaves")
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
        if self.args["in_distro_clauses"]:
            featureDict = self.model.feature_names
            print("featureDict", featureDict)
            revFeatureDict = {v: k for k, v in featureDict.items()}
            in_distro_clause_file = self.args["in_distro_clauses"]
            ofile = open(in_distro_clause_file, "r")
            clauses = ofile.readlines()
  
                
            clauses = [ ast.literal_eval(clause) for clause in clauses]
            for clause in clauses:
                self.add_clause_cons(clause,revFeatureDict)
        self.m.update()
    
    
    
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

                    
    # def compute_distance(self,result, data_file):  
    #     if not os.path.exists(data_file):
    #         print(f"Data file {data_file} does not exist.")
    #         return 
    #     data = pd.read_csv(data_file,nrows=10000)
    #     n_features = self.model[3]
    #     trees = self.model[1]
    #     sensitive_features = self.varyingFeat
    #     feature_list = [""]*n_features
    #     featurenames = self.model[6]
    #     for k in range(len(featurenames)):
    #         feature_list[k] = featurenames[k]
    #     missing_cols = [col for col in feature_list if col not in data.columns]
    #     for col in missing_cols: 
    #         missing_indices = feature_list.index(col)
    #         data[missing_cols] = result[0][missing_indices]
    #     data = data[feature_list]
    #     dist_type = "SegmentL1"
    #     # dist_type = "L2"
    #     if dist_type == "SegmentL1":
    #         segments = utils.feature_segments(trees,n_features)
    #         utils.data_distance( data, result[0], sensitive_features, segments=segments, dist_type="SegmentL1" )
    #     else:
    #         utils.data_distance( data, result[0], sensitive_features, dist_type=dist_type) 
            
    

    def attack(self,options):

        print("\n==================================")
        # if self.args["in_distro_clauses"]:
        #     featureDict = self.model[6]
        #     revFeatureDict = {v: k for k, v in featureDict.items()}
        #     in_distro_clause_file = self.args["in_distro_clauses"]
        #     ofile = open(in_distro_clause_file, "r")
        #     clauses = ofile.readlines()
        #     clauses = [ ast.literal_eval(clause) for clause in clauses]
        #     for clause in clauses:
        #         self.add_clause_cons(clause,revFeatureDict)


        if self.options.affected_cons:
            if self.multiclass:
                class_lists = []
                for i in range(self.n_classes):
                    class_lists.append(np.array(self.leaf_class_list) == i)
                if self.args["truelabel"] == -1:
                    pass
                else:
                    if self.args["otherlabel"] == -1:
                        pass
                    else:
                        if len(self.affected_leaves) > 0: 
                            print("Applying Affected cons")
                            valid_true = np.array(self.affected_leaves)[
                                np.array(class_lists[self.args["truelabel"]])[
                                    self.affected_leaves
                                ]
                            ]
                            valid_other = np.array(self.affected_leaves)[
                                np.array(class_lists[self.args["otherlabel"]])[
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
                print("Applying Affected cons")
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
        if not args["precise"]:
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
                if self.args["truelabel"] == -1:
                    pass
                else:
                    if self.args["otherlabel"] == -1:
                        pass
                    else:
                        for i in range(self.n_classes):
                            if i == self.args["truelabel"]:
                                continue
                            self.m.addConstr(
                                LinExpr(
                                    all_up_vals[class_lists[self.args["truelabel"]]],
                                    all_up_variables[
                                        class_lists[self.args["truelabel"]]
                                    ],
                                )
                                - LinExpr(
                                    all_up_vals[class_lists[i]],
                                    all_up_variables[class_lists[i]],
                                )
                                >= (self.ugap-self.lgap) * self.args["precision"],
                                name=f"mislabel_{i}",
                            )
                        for i in range(self.n_classes):
                            if i == self.args["otherlabel"]:
                                continue
                            if self.args["strong_multi"] or i == self.args["truelabel"]:
                                self.m.addConstr(
                                    LinExpr(
                                        all_down_vals[
                                            class_lists[self.args["otherlabel"]]
                                        ],
                                        all_down_variables[
                                            class_lists[self.args["otherlabel"]]
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
                                            class_lists[self.args["otherlabel"]]
                                        ],
                                        all_down_variables[
                                            class_lists[self.args["otherlabel"]]
                                        ],
                                    )
                                    - LinExpr(
                                        all_down_vals[class_lists[i]],
                                        all_down_variables[class_lists[i]],
                                    )
                                    >= 0,
                                    name=f"mislabel2_{i}",
                                )

                        if args["objective"]:
                            self.m.setObjective(
                                LinExpr(
                                    all_up_vals[class_lists[self.args["truelabel"]]],
                                    all_up_variables[
                                        class_lists[self.args["truelabel"]]
                                    ],
                                )
                                - LinExpr(
                                    all[class_lists[self.args["otherlabel"]]],
                                    self.llist2[class_lists[self.args["otherlabel"]]],
                                ),
                                GRB.MAXIMIZE,
                            )
            else:
                #--------------------------
                # Binary model
                #--------------------------
                self.m.addConstr(
                    LinExpr(all_up_vals, all_up_variables) + self.base_val
                    >= self.ugap * self.args["precision"], # self.guard_val
                    name="mislabel",
                )
                self.m.addConstr(
                    LinExpr(all_down_vals, all_down_variables) + self.base_val
                    <=  self.lgap * self.args["precision"], #-self.guard_val
                    name="mislabel-2",
                )
                if args["objective"]:
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

                if self.args["truelabel"] == -1:
                    pass
                else:
                    if self.args["otherlabel"] == -1:
                        pass
                    else:
                        for i in range(self.n_classes):
                            if i == self.args["truelabel"]:
                                continue
                            self.m.addConstr(
                                LinExpr(
                                    up_weights[class_lists[self.args["truelabel"]]],
                                    self.llist[class_lists[self.args["truelabel"]]],
                                )
                                - LinExpr(
                                    up_weights[class_lists[i]],
                                    self.llist[class_lists[i]],
                                )
                                >= (self.ugap-self.lgap), #self.guard_val,
                                name=f"mislabel_{i}",
                            )
                        for i in range(self.n_classes):
                            if i == self.args["otherlabel"]:
                                continue
                            self.m.addConstr(
                                LinExpr(
                                    down_weights[class_lists[self.args["otherlabel"]]],
                                    self.llist2[class_lists[self.args["otherlabel"]]],
                                )
                                - LinExpr(
                                    down_weights[class_lists[i]],
                                    self.llist2[class_lists[i]],
                                )
                                >= (self.ugap-self.lgap), #self.guard_val,
                                name=f"mislabel_{i}",
                            )
                            print(f"selfguard val {self.guard_val}")
                            # if self.args["strong_multi"] or i == self.args["truelabel"]:
                            #     self.m.addConstr(
                            #         LinExpr(
                            #             down_weights[
                            #                 class_lists[self.args["otherlabel"]]
                            #             ],
                            #             self.llist2[
                            #                 class_lists[self.args["otherlabel"]]
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
                            #                 class_lists[self.args["otherlabel"]]
                            #             ],
                            #             self.llist2[
                            #                 class_lists[self.args["otherlabel"]]
                            #             ],
                            #         )
                            #         - LinExpr(
                            #             down_weights[class_lists[i]],
                            #             self.llist2[class_lists[i]],
                            #         )
                            #         >= 0,
                            #         name=f"mislabel2_{i}",
                            #     )

                        if args["objective"]:
                            self.m.setObjective(
                                LinExpr(
                                    up_weights[class_lists[self.args["truelabel"]]],
                                    self.llist[class_lists[self.args["truelabel"]]],
                                )
                                - LinExpr(
                                    down_weights[class_lists[self.args["otherlabel"]]],
                                    self.llist2[class_lists[self.args["otherlabel"]]],
                                )
                                - LinExpr(
                                    down_weights[class_lists[self.args["truelabel"]]],
                                    self.llist2[class_lists[self.args["truelabel"]]],
                                )
                                + LinExpr(
                                    up_weights[class_lists[self.args["otherlabel"]]],
                                    self.llist[class_lists[self.args["otherlabel"]]],
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
                print(f"base_val : {self.base_val} guard :{self.guard_val}")
                if args["prob"]:
                    diffs = []
                    upvarobj = []
                    downvarobj = []
                    lamba = args["lambda"]
                    eps = 1e-30
                    
                    for key in self.probs.keys():
                        keys = list(self.probs[key].keys())
                        for i in range(len(keys)-1):
                            diffs.append(np.log(self.probs[key][keys[i]] + eps)-np.log(self.probs[key][keys[i+1]]+eps) )
                            upvarobj.append(self.pdict[key][i][1])
                            downvarobj.append(self.pdict[key][i][2])
                    if args["objective"]:
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
                    if args["objective"]:
                        self.m.setObjective(
                            LinExpr(up_weights, self.llist)
                            - LinExpr(down_weights, self.llist2),
                            GRB.MAXIMIZE,
                        )

        self.m.update()
        self.m.setParam("TimeLimit", 60 * 60)
        # self.m.setParam("SolutionLimit", 1)
        if self.args["prob"]:
            self.m.setParam(GRB.Param.PoolSolutions, 1)  # Get up to 10 solutions
        else:
            self.m.setParam("SolutionLimit", 1)
        

        tic = time.perf_counter()
        # Save the MILP constraints to a file
        constraint_file = "milp_constraints.lp"
        self.m.write(constraint_file)
        print(f"MILP constraints saved to {constraint_file}")
        self.m.optimize()
        toc = time.perf_counter()
        print('Time:', (toc - tic), 'seconds')
        if self.m.status == GRB.Status.INFEASIBLE:
            print(self.varyingFeat,(toc-tic),"Insensitive")
            return

        if self.m.status == GRB.Status.TIME_LIMIT:
            print("Timeout")
            return
        
        print(f"Sensitive features: {self.varyingFeat}")
        
        x =[0] * self.model.n_features
        x2 =[0] * self.model.n_features
        for i in range(1):
            self.m.setParam(GRB.Param.SolutionNumber, i)
            print(f"\nSolution {i+1}")
            print(f"Objective Value: {self.m.PoolObjVal}")
            
            

            
            for key in self.pdict.keys():
                trees = self.model.trees
                splits = trees[trees['Feature'] == f'f{key}']['Split']
                vals1 = [node[0] for node in self.pdict[key] if node[1].x > 0.5] + [splits.max()+1]
                x[key] = (
                    vals1[0] + ([splits.min()-1] + [node[0] for node in self.pdict[key]])[-len(vals1)]
                ) / 2
                vals2 = [node[0] for node in self.pdict[key] if node[2].x > 0.5] + [splits.max()+1]
                x2[key] = (
                    vals2[0] + ([splits.min()-1] + [node[0] for node in self.pdict[key]])[-len(vals2)]
                ) / 2

            x = np.array(x)
            x2 = np.array(x2)
            k = 5
            # print(mulprob(getprob(x,self.probs,self.guards)))
            # print(mulprob(getprob(x2,self.probs,self.guards)))
            # print(addprob(getprob(x,self.probs,self.guards)))
            # print(addprob(getprob(x2,self.probs,self.guards)))
            # print("Closest 5-1:", get_dist(x,self.X,self.y,k))
            # print("Closest 5-2:",get_dist(x2,self.X,self.y,k))
            # print("Mean 1:", get_dist(x,self.X,self.y,0))
            # print("Mean 2:", get_dist(x2,self.X,self.y,0))
            # print("-----------------------------------\n")
            res = []
            
            for i in range(len(x)):
                if x[i] != x2[i]:
                    res.append((x[i], x2[i]))
                else:
                    res.append(x[i])

                
        # xcopy = x.copy()
        # x2copy = x2.copy()
        # # print("xcopy",xcopy)
        xbound = {}
        x2bound = {}
        features = list(self.model.feature_names.keys())
        op_range_list = self.model.op_range_list
        # print(op_range_list)
        
        active_leaf1 = []
        active_leaf2 = []
        sum1 =0
        sum2 =0
        
        for i in range(len(self.llist)):
            if self.llist[i].x > 0.5:
                active_leaf1.append((i,self.leaf_v_list[i]))
                sum1 += self.leaf_v_list[i]
            if self.llist2[i].x > 0.5:
                active_leaf2.append((i,self.leaf_v_list[i]))
                sum2 += self.leaf_v_list[i]
        print(f"sigmoid(x+base_val): {1/(1+np.exp(-(sum1+self.base_val) ))}, sigmoid(x2+base_val): {1/(1+np.exp(-(sum2 + self.base_val)))}")
        print(f"sum1: {sum1-sum2}")

        # print(active_leaf1[0][1], self.llist[active_leaf1[0][0]])
        # print(active_leaf2[0][1], self.llist[active_leaf2[0][0]])
        # print(trees[(trees['Tree']==0) & (trees['Feature'] == 'Leaf') ]) #
        # exit()
        
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

        # print("Final bounds after MILP:")
        # for key in xbound:
        #     print(f"Feature {key} bounds: {xbound[key]} {x2bound[key]}") 
        # print(f"varying features: {self.varyingFeat}")
        # for key in xbound:
        #     lb1, ub1 = xbound[key]
        #     lb2, ub2 = x2bound[key]
        #     shared_lb = max(lb1, lb2)
        #     shared_ub = min(ub1, ub2)

        #     if key not in self.varyingFeat and shared_lb < shared_ub:
                
        #         xbound[key] = (shared_lb , shared_ub)

            # else:
            #     print(f"not varying feature {key}, setting bounds to shared bounds")
                # x2bound[key] = (shared_lb , shared_ub)
            # else:
            #     if shared_lb > shared_ub:
            #         x[key] = (lb1 + ub1) / 2
            #         x2[key] = (lb2 + ub2) / 2
            #     else:
            #         x_range = (lb1, shared_lb) if lb1 < shared_lb else (shared_ub, ub1)
            #         x2_range = (lb2, shared_lb) if lb2 < shared_lb else (shared_ub, ub2)

            #         x[key] = (x_range[0] + x_range[1]) / 2 if x_range[0] < x_range[1] else (shared_lb + shared_ub) / 2
            #         x2[key] = (x2_range[0] + x2_range[1]) / 2 if x2_range[0] < x2_range[1] else (shared_lb + shared_ub) / 2

        

        
        # print("sensitive bounds",xbound)
        # print(f"distance flag {args["compute_data_distance"]}")
        # input()
        if args["compute_data_distance"]:
            # print(f"=========================================")
            print(f"First region = {xbound}")
            # print(f"Second region = {x2bound}")
            data_distance.compute_data_distance(xbound,self.varyingFeat,self.model.feature_names,self.model.n_features,self.model.trees,options,dist_type='L0')
            data_distance.compute_data_distance(xbound,self.varyingFeat,self.model.feature_names,self.model.n_features,self.model.trees,options,dist_type='L1')
            data_distance.compute_data_distance(xbound,self.varyingFeat,self.model.feature_names,self.model.n_features,self.model.trees,options,dist_type='L2')
            data_distance.compute_data_distance(xbound,self.varyingFeat,self.model.feature_names,self.model.n_features,self.model.trees,options,dist_type='Linf')
            
            
        # x = np.array(x)
        # x2 = np.array(x2)
        
        res = []
        x_colored = []
        x2_colored = []
        differentfeature = []
        for i in range(len(x)):
            if x[i] != x2[i]:
                res.append((x[i], x2[i]))
                #x_colored.append(f"\033[91m{x[i]}\033[0m")
                #x2_colored.append(f"\033[91m{x2[i]}\033[0m")
                differentfeature.append(i)
            else:
                res.append(x[i])
                x_colored.append(str(x[i]))
                x2_colored.append(str(x2[i]))
        # print("Inputs ", res)
        print(f"feature varying: {differentfeature}")
        utils.print_array( 'Sensitive sample 1:', x_colored)
        utils.print_array( 'Sensitive sample 2:', x2_colored)
        # with open("res.pkl", "wb") as f:
        #     pickle.dump((x, x2), f)
        for i in range(0,len(x)):
            if(x[i] == 0): x[i]   = 0.0000000001
            if(x2[i] == 0): x2[i] = 0.0000000001
        pred1 = self.model.predict([x])#,pred_leaf=True)#[0][0]
        pred2 = self.model.predict([x2])#,pred_leaf=True) #[0][0]

        # utils.dump_dotty(self.model[0])

        # pred1 = self.model.pred_leaf_contribs(x)[0]#, pred_leaf=True)
        # pred2 = self.model.pred_leaf_contribs(x2)[0]#, pred_leaf=True)

        # p1,nodes = utils.eval_trees(x, 500, trees,base_val=0.3)
        # for i in range(0,500):
        #     if nodes[i] != pred1[0][i]:
        #         print(i)
        # print(p1)
        
        # val1 = np.dot(np.array([i.X for i in self.llist]), up_weights)
        # val2 = np.dot(np.array([i.X for i in self.llist2]), down_weights)

        print(f"Output Values: {pred1} {pred2}")
        
        # val1 = np.dot(np.array([i.X for i in self.llist]), up_weights)
        # class_lists = []
        # for i in range(self.n_classes):
        #     class_lists.append(np.array(self.leaf_class_list) == i)

        # for i in range(self.n_classes):
        #     val2 = np.dot(np.array([i.X for i in self.llist])[class_lists[i]], up_weights[class_lists[i]])
        #     print(val2)

        # print("down")
        # for i in range(self.n_classes):
        #     val2 = np.dot(np.array([i.X for i in self.llist2])[class_lists[i]], down_weights[class_lists[i]])
        #     print(val2)
        if args["prob"]:
            k = 5
            # print("5-nearest distance1: ",get_dist(x,self.X,k))
            # print("5-nearest distance2: ",get_dist(x2,self.X,k))
            # print("Distance from mean1: ",get_dist(x,self.X,0))
            # print("Distance from mean2: ",get_dist(x2,self.X,0))


        # print(f"Predictions inv: {sigmoid_inv(pred1)} {sigmoid_inv(pred2)}")
        # print(f"MILP: {val1} {val2}")

        # if(self.args["multiclass"]):
        #     class_lists = []
        #     for i in range(self.n_classes):
        #         class_lists.append(np.array(self.leaf_class_list) == i)
        #     print('mislabel constraint1:', np.sum((np.array(self.leaf_v_list)*np.array([item.x for item in self.llist]))[class_lists[self.args["truelabel"]]]) - np.sum((np.array(self.leaf_v_list)*np.array([item.x for item in self.llist]))[class_lists[self.args["otherlabel"]]]))
        #     print('mislabel constrain2:',np.sum((np.array(self.leaf_v_list)*np.array([item.x for item in self.llist2]))[class_lists[self.args["truelabel"]]]) - np.sum((np.array(self.leaf_v_list)*np.array([item.x for item in self.llist2]))[class_lists[self.args["otherlabel"]]]))
        #
        # else:
        #     print('mislabel constraint1:', np.sum((np.array(self.leaf_v_list)*np.array([item.x for item in self.llist]))))
        #     print('mislabel constrain2:', np.sum(np.array(self.leaf_v_list)*np.array([item.x for item in self.llist2])))
        # print([self.pdict[x][0] for x in self.pdict])
        # print([self.pdict[x][1].x for x in self.pdict])
        # print([self.pdict[x][2].x for x in self.pdict])
        # if (not suc):
        #     if self.binary:
        #         manual_res = self.check(x, self.json_file)
        #     else:
        #         manual_res = self.check(x, self.pos_json_file) - self.check(x, self.neg_json_file)
        #     print('manual prediction result:', manual_res)
        #     if (not self.binary and manual_res>=0)  or (self.binary and int(manual_res>0) == label):
        #         print('** manual prediction shows attack failed!! **')
        #     else:
        #         print('** manual prediction shows attack succeeded!! **')
        # return x

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


args = {}


def main(args_inp,options):
    global args
    args = vars(args_inp)
    # print(args)
    # print(options)
    
    random.seed(8)
    np.random.seed(8)

    #---------------------------------------
    # Load model
    #---------------------------------------
    e = ensemble.Ensemble(options)
    e.load(print_vitals = True)
    # base_val = e.get_base_value()
    
    # # ----------------------------------------------------
    # # Read model file
    # # ----------------------------------------------------
    # if options.model_file.isdigit():
    #     bst = open_model(model_files[int(options.model_file)], args["max_trees"])
    # else:
    #     bst = open_model(options.model_file, args["max_trees"],details_file = args['details']) #NV
    # model = xgboost_wrapper( e, binary=True, max_trees=args["max_trees"] )
    model = e
    if not args["all_features"]:
        varyingFeat = args["features"]
    else:
        varyingFeat = [i for i in range(0, e.n_features)]
        
    if args["precision"] < 0:
        args["precision"] = 10 * bst[-3]
        print(f'Precision={args["precision"]}')

    attack = milpSolver(
        model,
        guard_val=args["gap"],
        varyingFeat=varyingFeat,
        debug=args["debug"],
        args=args,
        options=options,
    )

    global_start = time.time()
    attack.attack(options)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("filenum", type=str, help="model path")
#     parser.add_argument(
#         "-g", "--gap", type=float, default=GUARD_VAL, help="guard value"
#     )
#     parser.add_argument(
#         "-r",
#         "--round_digits",
#         type=int,
#         default=ROUND_DIGITS,
#         help="number of digits to round",
#     )
#     parser.add_argument(
#         "--max_trees",
#         type=int,
#         default=None,
#         help="Maximum number of trees to consider",
#     )
#     parser.add_argument(
#         "--features",
#         type=int,
#         nargs="+",
#         default=None,
#         help="Indexes of the features for which to do sensitivity analysis",
#     )
#     parser.add_argument(
#         "--debug", action="store_true", help="Run serially and stop on pdb statements"
#     )
#     main(parser.parse_args())
