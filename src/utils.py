import xgboost as xgb
import numpy as np
import json
import os
import pickle
import pandas as pd
import math
import sys
import io

def print_info( what, data ):
    print(f"# {what} : {data}")
   
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def sigmoid_inv(y):
    return np.log(y/(1-y))
def sigmoid_inv_diff( y, base_value ):
    return sigmoid_inv(y) - sigmoid_inv(base_value)

def dump_info( options, level, msg):
    if options.verbosity > level:
        print('#', msg)

def clean_data(data,feature_details):
    for idx, row in feature_details.iterrows():
        if row['type'] == 'log':
            data[row['name']] = data[row['name']].apply(lambda x: math.log(x) if x > 0 else 0)
    return data

def dump_dotty(model):
    # os.makedirs('tree_dots', exist_ok=True)
    for i in range(model.num_boosted_rounds()):
        tree_dot = xgb.to_graphviz(model, num_trees=i)
        tree_dot.format = "dot"
        tree_dot.render(f"/tmp/tree_{i}")


model_files = [
    #  '../models/tree_verification_models/binary_mnist_unrobust/1000.resaved.json',
    # '../models/tree_verification_models/ori_mnist_robust_new/0200.resaved.json',
    # '../models/tree_verification_models/ori_mnist_unrobust_new/0200.resaved.json',
    # '../models/tree_verification_models/covtype_robust/0080.resaved.json',
    # '../models/tree_verification_models/covtype_unrobust/0080.resaved.json',
    # '../models/tree_verification_models/fashion_robust_new/0200.resaved.json',
    # '../models/tree_verification_models/fashion_unrobust_new/0200.resaved.json',
    # '../models/tree_verification_models/webspam_robust_new/0100.resaved.json',
    # '../models/tree_verification_models/webspam_unrobust_new/0100.resaved.json',
    "../models/tree_verification_models/breast_cancer_robust/0004.resaved.json",
    "../models/tree_verification_models/breast_cancer_unrobust/0004.resaved.json",
    "../models/tree_verification_models/diabetes_robust/0020.resaved.json",
    "../models/tree_verification_models/diabetes_unrobust/0020.resaved.json",
    "../models/tree_verification_models/cod-rna_unrobust/0080.resaved.json",
    "../models/tree_verification_models/binary_mnist_robust/1000.resaved.json",
    # '../models/tree_verification_models/higgs_robust/0300.resaved.json',
    "../models/tree_verification_models/higgs_unrobust/0300.resaved.json",
    "../models/tree_verification_models/ijcnn_robust_new/0060.resaved.json",
    # '../models/tree_verification_models/ijcnn_unrobust_new/0060.resaved.json',
    # 'smallmodel.pkl',
    # 'selftrained_model1.pkl',
    # 'self_model2.pkl',
    # '../models/rf_mnist_100_6.pkl',
    # '../models/new_model_200_6-0.pkl',
    # '../models/rf_db_50_6.pkl',
    # '../models/rf_db_75_6.pkl',
    # '../models/rf_db_100_6.pkl',
    "../models/new_model_5_6.pkl",
    "../models/new_model_10_6.pkl",
    "../models/new_model_20_6.pkl",
    "../models/new_model_25_6.pkl",
    "../models/new_model_35_6.pkl",
    "../models/new_model_50_6.pkl",
    "../models/new_model_65_6.pkl",
    "../models/new_model_75_6.pkl",
    "../models/new_model_80_6.pkl",
    "../models/new_model_100_6.pkl",
    "../models/new_model_125_6.pkl",
    "../models/new_model_150_6.pkl",
    "../models/new_model_175_6.pkl",
    "../models/new_model_200_6.pkl",
    # '../models/mult_feat_100.pkl'
    # '../models/mult_feat_75.pkl',
    # '../models/mult_feat_50.pkl',
    # '../models/mult_feat_20.pkl',
    # '../models/mult_feat_15.pkl',
    # '../models/mult_feat_10.pkl',
    "../models/xgb_sbi.json",
]

def model_details_file(n_features, details_fname):
    if details_fname == None:
        exit()
    # details_fname = args.details
    if not os.path.exists(details_fname):
        print(f"Missing : {details_fname}")
        exit()
    details = pd.read_csv(details_fname)
    feature_names = {} #[""] * n_features
    op_range_list = [(-1000000,10000000)] * n_features
    for index, row in details.iterrows():
        i = row["feature"]
        feature_names[i] = str(row["name"])
        # print(f"feature_name :{feature_names[i]}")
        op_range_list[i] = (row["lb"], row["ub"])
    return feature_names, op_range_list



def print_array( leading_text, a ):
    print(leading_text,end="[ ")
    if(len(a) > 0 ): print(a[0],end=" ")
    for v in a[1:] : print(",",v,end=" ")
    print("]")
        
    

# def sigmoid_inv(x):
#     return -np.log(1 / x - 1)


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

#---------------------------------------
# Evaluate trees
#---------------------------------------

def eval_tree_rec(data, nid, tree,verbose=False):
    rows = tree[tree["Node"] == nid]
    for idx, row in rows.iterrows():
        f = row["Feature"]
        if f == "Leaf":
            return row["Node"],row["Gain"]
        else:
            f = int(f[1:])
            diff = data[f] - row["Split"]
            if verbose: print('Tree path:', row['Feature'],row['Split'],data[f],diff)
            # if diff != 0 and abs(diff) < 0.001: print(row['Feature'],row['Split'],diff)
            if data[f] < row["Split"]:
                child = row["Yes"]
            else:
                child = row["No"]
            child = int(child.split("-")[1])
            return eval_tree_rec(data, child, tree,verbose)


def eval_tree(data, tree,verbose=False):
    return eval_tree_rec(data, 0, tree,verbose)


def eval_trees(data, num_tree, trees,base_val=0.5,verbose=False):
    vals = []
    nodes = []
    for i in range(0, num_tree):
        tree = trees[(trees["Tree"] == i)]
        nid,v = eval_tree(data, tree,verbose)
        vals.append(v)
        nodes.append(nid)
    print(nodes)
    s = sum(vals)+sigmoid_inv(base_val)
    return round(1.0 / (1.0 + math.exp(-s)), 7),nodes


def eval_trees_compare(data1, data2, num_tree, trees):
    vals = []
    for i in range(0, num_tree):
        tree = trees[(trees["Tree"] == i)]
        v1 = eval_tree(data1, tree)
        v2 = eval_tree(data2, tree)
        if v1 != v2:
            print(i, v1, v2)

#------------------------------------------------

def get_bench_info(benchidx: int):
    """
    Returns ntrees, benchname, nfeat, depth
    """
    benchname = model_files[benchidx].split("/")[-2]
    if benchname == "models":
        benchname = model_files[benchidx].split("/")[-1].split(".")[0]
    model = xgb.Booster({"nthread": 4})  # init model
    try:
        model = pickle.load(open(model_files[benchidx], "rb"))
    except:
        model.load_model(model_files[benchidx])  # load data
    # dump_dotty(model)
    ntrees = model.num_boosted_rounds()
    nfeat = model.num_features()
    dump = model.get_dump(with_stats=True)
    tree_depths = []
    for tree in dump:
        lines = tree.split("\n")
        # The depth of the tree is the maximum number of tabs (representing levels) in any line
        max_depth = max(line.count("\t") for line in lines if line.strip() != "")
        tree_depths.append(max_depth)
    depth = max(tree_depths)
    return ntrees, benchname, nfeat, depth


basepath = os.path.dirname(__file__)
model_files = list(map(lambda x: basepath + "/" + x, model_files))


def open_model(
        model_file,
        max_trees=None,
        model_library="xgboost",
        veritas=False,
        multiclass=False,
        max_classes=None,
        details_file=None ):
    if not os.path.exists(model_file):
        print(f"{model_file} does not exists!")
        exit()    
    if model_library == "xgboost":
        return open_model_xgb(model_file, max_trees=max_trees, max_classes=max_classes, veritas=veritas, details_file = details_file)
    elif model_library == "randomforest":
        return open_model_sklearn(model_file, max_trees, multiclass=multiclass, details_file = details_file)
    else:
        return None


def open_model_xgb(model_file, max_trees=None, max_classes=None, veritas=False, details_file=None):
    if model_file.isdigit():
        model_file = model_files[int(model_file)]
    model = xgb.Booster({"nthread": 4})  # init model
    try:
        model = pickle.load(open(model_file, "rb"))
    except:
        model.load_model(model_file)  # load data
    
    # dump_dotty(model)
    # model = model.get_booster()
    if veritas:
        return model, 0, 0, 0, 0
    try:
        import json
        with open(model_file, 'r') as f:
            model_json = json.load(f)
        feature_names = model_json['learner']["feature_names"]
        
        m_names = {}
        i = 0
        for f in feature_names:
            m_names[i] = f
            i = i +1
        feature_names = m_names
    except:
        feature_names  = None
    
    model.orig_f_names = model.feature_names
    if model.feature_names is not None:
        model.feature_names = [f"f{i}" for i in range(len(model.feature_names))]
    model.feature_names = None
    
    # --------------------------------
    # Get the base score
    # --------------------------------
    try:
        import json
        with open(model_file, 'r') as f:
            model_json = json.load(f)
        base_score = float(model_json['learner']['learner_model_param']['base_score']) # model.base_score
    except:
        base_score = 0.5

    # try:
    #     base_score = model.base_score
    #     # model = model.get_booster()
    # except:
    #     pass

    trees = model.trees_to_dataframe()
    # sys.stdout = old_stdout
    n_trees = model.num_boosted_rounds()
    n_features = model.num_features()

    dump = model.get_dump(with_stats=True)
    num_classes = len(dump) // (n_trees)
    if num_classes != 1:
        trees["class"] = trees["Tree"] % num_classes
        trees["Tree"] = trees["Tree"] // num_classes
    else:
        trees["class"] = 0

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
    # Comment out before running
    if max_classes is not None and max_classes != -1 and num_classes > max_classes:
        trees = trees[trees["class"] < max_classes]
        num_classes = max_classes
    
    if( feature_names == None): feature_names = [""] * n_features
    op_range_list = [(float(i.min()-1), float(i.max())) for i in (trees[trees['Feature'] == f'f{j}']['Split'] for j in list(range(n_features)))]
    if(details_file):feature_names, op_range_list= model_details_file(n_features, details_file)
    else: feature_names = {i: f"{i}" for i in range(n_features)}
    data = [
        ("model name", model_file),
        ("trees per class", n_trees),
        ("number of class es", num_classes),
        ("number of features", n_features),
        ("base score", base_score),
        ("max depth", depth),
        # ("Feature names", feature_names),
        ("all trees", len(dump)),
        ("Feature names", feature_names),
        # ("operating range of features", op_range_list),
    ]
    # for label, value in data:
    #     print(f"{label}: {value}")
    return model, trees, n_trees, n_features, num_classes, base_score, feature_names, op_range_list


def resave_model(model_file):
    # outfile = model_file[:-5]+'resaved.model'
    # os.system(f"rm {outfile}")
    outfile = model_file[:-5] + "resaved.json"
    model = xgb.Booster({"nthread": 4})  # init model
    model.load_model(model_file)  # load data
    model.save_model(outfile)  # load data
    print(model_file, outfile)


def open_model_sklearn(model_file, max_trees=None, max_classes=None, multiclass=False, details_file=None):
    if model_file.isdigit():
        model_file = model_files[int(model_file)]

    num_classes = 1

    def extract_tree_data(rf_model):

        def feat_name(feat):
            if feat >= 0:
                return "f" + str(feat)
            else:
                return "Leaf"

        tree_data_list = []
        for tree_index, tree in enumerate(rf_model.estimators_):
            tree_structure = tree.tree_
            for node_index in range(tree_structure.node_count):

                if multiclass:
                    gain = tree_structure.value[node_index].tolist()[0]
                else:
                    gain = tree_structure.value[node_index].tolist()[0][0]
                tree_data_list.append(
                    {
                        "Tree": tree_index,
                        "Node": node_index,
                        "ID": str(tree_index) + "-" + str(node_index),
                        "Feature": feat_name(tree_structure.feature[node_index]),
                        "Split": tree_structure.threshold[node_index],
                        "Yes": str(tree_index)
                        + "-"
                        + str(tree_structure.children_left[node_index]),
                        "No": str(tree_index)
                        + "-"
                        + str(tree_structure.children_right[node_index]),
                        "Missing": str(tree_index)
                        + "-"
                        + str(tree_structure.children_left[node_index]),
                        "Impurity": tree_structure.impurity[node_index],
                        "Samples": tree_structure.n_node_samples[node_index],
                        "Gain": gain,
                    }
                )
        return pd.DataFrame(tree_data_list)

    with open(model_file, "rb") as file:
        model = pickle.load(file)  # dump_dotty(model)
    trees = extract_tree_data(model)
    n_trees = model.n_estimators
    n_features = model.n_features_in_
    # dump = model.get_dump(with_stats=True)
    if multiclass:
        num_classes = len(trees["Gain"][0])

    tree_depths = [tree.get_depth() for tree in model.estimators_]

    depth = max(tree_depths)
    if max_trees is not None and max_trees != -1 and n_trees > max_trees:
        trees = trees[trees["Tree"] < max_trees]
        n_trees = max_trees  # TODO: This does not edit the model, so final solving might not show any unfairness
    if max_classes is not None and max_classes != -1 and num_classes > max_classes:
        trees = trees[trees["class"] < max_classes]
        num_classes = max_classes
    data = [
        ("model name", model_file),
        ("trees per class", n_trees),
        ("number of classes", num_classes),
        ("number of features", n_features),
        ("max depth", depth),
    ]

    for label, value in data:
        print(f"# {label}: {value}")

    return model, trees, n_trees, n_features, num_classes, 0 # TODO: Do random forests have a base score


def feature_segments(trees,n_features):
    segments = {}
    for i in range(n_features):
        sliced = trees[(trees["Feature"] == f"f{i}")][["Feature", "Split"]].copy()
        sliced.sort_values(["Split"], inplace=True)
        sliced.drop_duplicates(inplace=True)
        segments[i] = sliced['Split'].values.tolist()
    return segments
        
