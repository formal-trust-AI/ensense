import os
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import lightgbm
import pickle
import joblib
import xgboost as xgb
import sklearn
import sklearn.ensemble
import options
import utils
import math
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

pd.set_option('display.max_rows', 200)

class Interval:
    def __init__(self,bracklb,lb,ub,brackub):
        
        self.bracklb = bracklb
        self.lb = lb
        self.ub = ub
        self.brackub = brackub

    def printintv(self):
        return f"{self.bracklb} {self.lb} {self.ub} {self.brackub}"
    
class Ensemble:
    def __init__(self, options):
        self.options = options
        self.model_file   = options.model_file
        self.details_file = options.details_file
        self.model = None
        self.trees = None # 
        self.model_library = options.model_library 
        self.max_trees = options.max_trees
        self.max_classes = None # tobe deprecated
        self.n_trees = None
        self.n_features = None
        self.n_classes =None
        self.base_score = None
        self.feature_names = None
        self.feature_types = None
        self.op_range_list = None
        self.depth = None
        self.split_kind = "<"
        self.multiclass = False
        self.guard_lists = {}

    
    def load_file(self):
        # --------------------------------------------------
        # Load model, filename decides the type of the model
        # --------------------------------------------------
        fname = self.model_file
        if fname[-4:] == ".pkl":
            self.model = pickle.load(open(self.model_file, "rb"))
        elif fname[-4:] == ".sav":
            self.model = pickle.load(open(self.model_file, "rb"))
        elif fname[-7:] == ".joblib":
            self.model = joblib.load(self.model_file)
        elif fname[-4:] == ".txt": # lgbm
            # todo: lgbm has extention .txt. Can we modify it
            self.model = lightgbm.Booster(model_file = self.model_file)
        elif fname[-5:] == ".json":
            # self.model = pickle.load(open(self.model_file, "rb"))
            self.model = xgb.Booster({"nthread": 4})  # init model
            self.model.load_model(self.model_file)  # load data
        else:
            utils.print_error("input", "Failed to indentify the model type!")

    def print_vitals(self):
        # utils.print_verbose(self.options, 1, "Model name", self.model_file)
        data = [
            ("Model name", self.model_file),
            ("Trees per class", self.n_trees),
            ("Number of classes", self.n_classes),
            ("Number of features", self.n_features),
            ("Base score", self.base_score),
            ("Max depth", self.depth),
            ("Feature names", self.feature_names),
        ]
        for label, value in data:
            utils.print_verbose(self.options, 0, label, value)
        for f in range(self.n_features):
            utils.print_verbose(self.options, 4, f"Guards of f{f}", self.get_guard_list(f,self.op_range_list))

    def unsupported_model(self):
        utils.print_error( "internal", f'Cannot load model for library {self.model_library} yet!')

    def load( self, model_file=None, print_vitals = False, *arg ):
        if model_file  != None: self.model_file = model_file
        # ---------------------------------------------------------
        # Load file. Try opening with pickle first then use xgb lib
        # ---------------------------------------------------------
        self.load_file()
        # ---------------------------------------------------------
        # Identify the model type!
        # ---------------------------------------------------------
        if( isinstance(self.model, sklearn.ensemble._forest.RandomForestClassifier) ):
            self.model_library = "rf"
        elif( isinstance(self.model, xgb.sklearn.XGBClassifier) ):
            self.model = self.model.get_booster()
            self.model_library = "xgboost"
        elif( isinstance(self.model, xgb.core.Booster) ):
            self.model_library = "xgboost"
        elif( isinstance(self.model, lightgbm.basic.Booster) ):
            self.model_library = "lgbm"
        elif( isinstance(self.model, lightgbm.sklearn.LGBMClassifier) ):
            self.model.booster_.save_model( '/tmp/model.txt' )
            self.model = lightgbm.Booster( model_file= '/tmp/model.txt' )
            self.model_library = "lgbm"
        else:
            utils.print_error( "input", f"Tool does not support model of type {type(self.model)}")
            
            
        if self.model_library == "xgboost":
                                    
            #--------------------------------------
            # Read feature names from the model
            #--------------------------------------
            try:
                with open(self.model_file, 'r') as f: model_json = json.load(f)
                feature_names = model_json['learner']["feature_names"]
                self.feature_names = {}
                for idx,f in enumerate(feature_names): self.feature_names[idx] = f                
            except:
                self.feature_names  = None

            
            if self.feature_names == None and self.model.feature_names != None:
                self.feature_names = {}
                for idx,f in enumerate(self.model.feature_names): 
                    self.feature_names[idx] = f                
            self.model.feature_names = None

            # --------------------------------
            # trees, n_trees, n_features
            # --------------------------------
            self.trees = self.model.trees_to_dataframe()
            self.n_trees = self.model.num_boosted_rounds()
            self.n_features = self.model.num_features()
            
            # --------------------------------
            # Get the base score
            # --------------------------------
            try:
                model_json = json.loads(self.model.save_config())
                # a list: one entry for binary, one per class for multiclass
                self.base_score = utils.parse_base_score(
                    model_json['learner']['learner_model_param']['base_score'])
            except KeyError:
                self.base_score = [0.5]   # model family carries no base_score
            except (ValueError, TypeError) as e:
                utils.print_error("Model", f"could not parse base_score: {e}")

            # ---------------------------------------------------------
            # Compute the number of classes and assign trees to classes
            # ---------------------------------------------------------
            dump = self.model.get_dump(with_stats=True)
            self.n_classes = len(dump) // (self.n_trees)
            if self.n_classes != 1:
                self.trees["class"] = self.trees["Tree"] % self.n_classes
                self.trees["Tree"] = self.trees["Tree"] // self.n_classes
            else:
                self.trees["class"] = 0

            # ---------------------------------------------------------
            # Compute maximum depth of the trees
            # ---------------------------------------------------------
            tree_depths = []
            for tree in dump:
                lines = tree.split("\n")
                # The depth of the tree is the maximum number of tabs (representing levels) in any line
                max_depth = max(line.count("\t") for line in lines if line.strip() != "")
                tree_depths.append(max_depth)
            self.depth = max(tree_depths)

            self.objective = json.loads(self.model.save_config())['learner']['objective']['name']
            
        elif self.model_library == "lgbm":
            self.feature_names = {}
            replace_map = {None: 'Leaf'}
            for idx,f in enumerate(self.model.feature_name()): 
                self.feature_names[idx] = f
                replace_map[f] = f"f{idx}"
                
            self.trees = self.model.trees_to_dataframe()
            self.trees['split_feature'] = self.trees['split_feature'].replace(replace_map)
            self.trees.rename(columns = { 'tree_index'   :'Tree',
                                          'node_index'   :'ID',
                                          'split_feature':'Feature',
                                          'left_child'   : 'Yes',
                                          'right_child'  : 'No',                                          
                                          'threshold'    : 'Split',
                                          'value'        : 'Gain',
                                         },inplace=True)
            self.trees["Node"] = self.trees['ID'].str.split('-').str[1]            
            self.n_trees = self.model.num_trees()
            self.n_features = self.model.num_feature()
            self.base_score = 0.5 # dummy value; has no meaning; as far as I know!
            self.trees["class"] = 0            
            config = self.model.dump_model()
            
            self.n_classes = config.get('num_class', 1)
            self.split_kind = "<="
            
            def get_tree_depth(tree):
                if 'left_child' not in tree and 'right_child' not in tree: return 1
                return 1 + max(get_tree_depth(tree['left_child']), get_tree_depth(tree['right_child']))
            self.depth = max([get_tree_depth(tree['tree_structure']) for tree in config['tree_info']])
        elif self.model_library == "rf":
            self.n_trees = self.model.n_estimators
            self.n_features = self.model.n_features_in_
            self.depth =max([tree.get_depth() for tree in self.model.estimators_])            
            self.trees = self.extract_tree_data()
            self.base_score = self.n_trees/2
            self.split_kind = "<="
        else:
            utils.print_error( "internal", f'Cannot load model for library {self.model_library} yet!')

        # ---------------------------------------------------------
        # Adjust the number of trees to consider
        # ---------------------------------------------------------
        if self.max_trees is not None and self.n_trees > self.max_trees:
            self.trees = self.trees[self.trees["Tree"] < self.max_trees]
            self.n_trees = self.max_trees

        # -----------------------------------------------------------
        # Adjust the number of classes to consider: why this feature?
        # -----------------------------------------------------------
        if self.max_classes is not None and self.n_classes > self.max_classes:
            self.trees = self.trees[self.trees["class"] < self.max_classes]
            self.n_classes = self.max_classes

        # -----------------------------------------------------------
        # Auto op range computation
        # -----------------------------------------------------------            
        self.op_range_list = [(float(i.min()-1), float(i.max()+1))
                              for i in (self.trees[self.trees['Feature'] == f'f{j}']['Split']
                                        for j in list(range(self.n_features)))]
        # self.op_range_list = []
        # for j in range(self.n_features):
        #     splits = self.trees[self.trees['Feature'] == f'f{j}']['Split']
        #     min_val, max_val = splits.min(), splits.max()
        #     if pd.isna(min_val) or pd.isna(max_val):
        #         self.op_range_list.append((-np.inf, np.inf))
        #     else:
        #         self.op_range_list.append((float(min_val - 1), float(max_val + 1)))

        # -----------------------------------------------------------
        # More information about the model from another file
        # -----------------------------------------------------------   
        self.feature_weights = [1.0] * self.n_features         
        if self.details_file:
            feature_names, self.op_range_list, self.feature_types,self.feature_weights = utils.model_details_file(self.n_features,
                                                                                            self.details_file)
            assert(len(feature_names) == self.n_features)
        else:
            feature_names = {i: f"{i}" for i in range(self.n_features)}
            
        # --------------------------------------------------------------
        # Update feature names only if model does not provide feature names
        # --------------------------------------------------------------
        if self.feature_names == None or self.feature_names == {}:
            self.feature_names = feature_names
        
        # -----------------------------------------------------------
        # Check multiclass
        # -----------------------------------------------------------
        self.multiclass = self.options.multiclass
        if self.options.multiclass:
            if self.n_classes < 3:
                utils.print_error( "input", "Less than three output classes for multiclass!")
        else:
            if self.n_classes != 1:
                utils.print_error( "arguments", "Model is not binary classifier use multiclass options!")

        # --------------------------------------
        # Modify options after loading the model
        # --------------------------------------
        self.options.lgap,self.options.ugap = self.get_interpret_gap( self.options.lgap, self.options.ugap )
        # ------------------------------------------------------------------------------
        # milp solver does not necessarily need precision, but pb solver needs precision   
        # ------------------------------------------------------------------------------
        if (self.options.precision == None or self.options.precision == 0) and self.options.solver != 'milp':
            self.options.precision = max(5*self.n_trees,100)
            
        # ------------------------------------------------------------------------------
        # Modify sensitive features
        # ------------------------------------------------------------------------------
        if self.options.all_features:
            self.options.features = [i for i in range(self.n_features)]
        
        # --------------------------------------
        # Print vitals
        # --------------------------------------
        if print_vitals == True:
            self.print_vitals()

        # --------------------------------------------
        # Check if sensitive features are in the range
        # --------------------------------------------
        for f in self.options.features:
            if f < 0 or f >= self.n_features:
                utils.print_error( "arguments", "Sensitive features are out of range!" )
                
        # --------------------------------------
        # Generate random samples if needed
        # --------------------------------------
        if self.options.random_samples > 0 and self.options.local_check_samples == None:
            np.random.seed(self.options.seed)
            self.options.local_check_samples = [
                [ self.get_sample(i) for i in range(self.n_features)]
                for _ in range(self.options.random_samples)
            ]
            utils.print_verbose( self.options,5, f'Random samples:', self.options.local_check_samples )

        # ------------------------------------------
        # Check if read samples match the model!!
        # ------------------------------------------
        if self.options.local_check_samples != None and self.n_features != len(self.options.local_check_samples[0]):
            utils.print_error( "Input", "#features in models does not match the #features in sample" )

        utils.print_verbose( self.options,7,f'Gap values:', f"{self.options.lgap} {self.options.ugap}")
        if self.options.verbosity > 7:
            self.dump_to_dot()


    def get_root_name(self):
        if self.model_library == "lgbm":
            return "S0"
        return 0

    def get_base_value(self):
        if self.model_library in ("xgboost", "lgbm"):
            base = np.ravel(self.base_score).astype(float)
            if self.n_classes and self.n_classes > 1:
                # softmax uses the identity link: class k's margin gets + base[k]
                # (verified against xgboost: margin - sum(leaves) == base_score)
                if base.size == 1:
                    return [float(base[0])] * self.n_classes
                if base.size != self.n_classes:
                    utils.print_error("Model", f"base_score has {base.size} entries "
                                               f"for {self.n_classes} classes")
                return [float(v) for v in base]
            # binary uses the logistic link, so base_score is a probability
            return utils.sigmoid_inv(float(base[0]))
        elif self.model_library == "rf":
            return -self.n_trees/2
        else:
            self.unsupported_model()

    def get_interpret_gap(self, lgap, ugap ):
        if self.model_library == "xgboost":
            if lgap <= 0 or ugap >= 1:
                utils.print_error( "arguments", f"Gap [{lgap},{ugap}] is out of range!")
            return utils.sigmoid_inv( lgap ), utils.sigmoid_inv( ugap )
        elif self.model_library == "rf":
            return ( (lgap-0.5)*self.n_trees, (ugap-0.5)*self.n_trees)
        elif self.model_library == "lgbm":
            if lgap <= 0 or ugap >= 1:
                utils.print_error( "arguments", f"Gap [{lgap},{ugap}] is out of range!")
            return utils.sigmoid_inv( lgap ), utils.sigmoid_inv( ugap )
        else:
            self.unsupported_model()

    
    # TODO: make this work
    def add_feature_names(self, fnames ):
        if fnames == None:
            self.feature_names = [f"f{i}" for i in range(self.n_features)]
        self.feature_names = fnames
    
    def predict( self, data, *arg ):

        if self.model_library == "xgboost":
            if self.max_trees != None:
                result = self.model.predict(xgb.DMatrix(data), *arg, iteration_range=(0, self.options.max_trees))
            else:
                result = self.model.predict(xgb.DMatrix(data), *arg)        
                      
        elif self.model_library == "rf":
            result = self.model.predict(data)
        elif self.model_library == "lgbm":
            result = self.model.predict(data)
        else:
            print(f'prediction is Not supported for {self.model_library} yet!')

        #-------------------------------------
        # Should always happen
        #-------------------------------------
        if self.options.verbosity > 5:
            for d in data:
                our_result = self.eval_trees(d,verbose=self.options.verbosity)
                print('Library evaluation',result)
                print('Our evaluation',our_result)        
            # assert( our_result == result.tolist() )
        
        return result

    #     def pred_leaf_contribs(self, input_data):
    #         input_data, _ = self.maybe_flat(input_data)
    #         ori_input = np.copy(input_data)
    #         input_data = xgb.DMatrix(sparse.csr_matrix(input_data))
    #         ori_input = xgb.DMatrix(sparse.csr_matrix(ori_input))
    #         return np.array([self.model[0].predict(input_data, output_margin=True, iteration_range=(i,i+1))[0] for i in range(self.model[0].num_boosted_rounds())])


    #     def predict_logits(self, input_data):
    #         input_data, _ = self.maybe_flat(input_data)
    #         input_back = np.copy(input_data)
    #         input_data = sparse.csr_matrix(input_data)
    #         input_data = xgb.DMatrix(input_data)
    #         test_predict = np.array(self.model[0].predict(input_data))
    #         return test_predict

    #     def predict_label(self, input_data):
    #         return self.predict(input_data)

    def dump_xgb_to_dot(self, out_dir="/tmp/"):
        # os.makedirs('tree_dots', exist_ok=True)
        for i in range(self.model.num_boosted_rounds()):
            tree_dot = xgb.to_graphviz(self.model, num_trees=i)
            tree_dot.format = "dot"
            tree_dot.render(f"{out_dir}/tree_{i}")
            
    def dump_lightgbm_to_dot(self, out_dir="/tmp/"):
        for i in range(self.n_trees):
            dot = lightgbm.create_tree_digraph( self.model, tree_index=i, name=f"tree_{i}" )
            dot.save(f"{out_dir}/tree_{i}.dot")
            

    def dump_forest_to_dot( self, out_dir="/tmp/" ):
        from sklearn.tree import export_graphviz
        import os
        os.makedirs(out_dir, exist_ok=True)
        feature_names = [f"f{i}" for i in range(self.n_features)]
        for i, est in enumerate(self.model.estimators_):
            export_graphviz(
                est,
                out_file=f"{out_dir}/tree_{i:03d}.dot",
                feature_names=feature_names,
                class_names=[str(c) for c in self.model.classes_],  # omit for regressors
                rounded=True,
                filled=True,
                impurity=True,      # show Gini/entropy
                proportion=True,    # normalize sample counts
                special_characters=True
                # max_depth=3,      # uncomment to keep files small
                # precision=2       # control numeric precision in labels
            )
            
    def dump_to_dot( self, out_dir="/tmp/" ):
        if self.model_library   == "xgboost": self.dump_xgb_to_dot(out_dir)
        elif self.model_library == "rf"     : self.dump_forest_to_dot(out_dir)
        elif self.model_library == "lgbm"   : self.dump_lightgbm_to_dot(out_dir)
        else:
            print(f'Library {self.model_library} not supported!')

    def extract_tree_data(self):
        assert( self.model_library == "rf" )

        n_classes = 1
        for tree_index, tree in enumerate(self.model.estimators_):
            tree_structure = tree.tree_
            for node_index in range(tree_structure.node_count):
                n_classes = len(tree_structure.value[node_index].tolist()[0])
                break
            break
        if n_classes == 2: n_classes = 1
        self.n_classes = n_classes
        # self.n_classes = n_classes
        
        def feat_name(feat):
            if feat >= 0:
                return "f" + str(feat)
            else:
                return "Leaf"

        tree_data_list = []
        for tree_index, tree in enumerate(self.model.estimators_):
            tree_structure = tree.tree_
            for node_index in range(tree_structure.node_count):
                for class_index in range(self.n_classes):
                    gain = tree_structure.value[node_index].tolist()[0][class_index]
                    # if tree_structure.feature[node_index] < 0:
                    #     output_class = np.argmax(tree_structure.value[node_index])
                    #     if class_index == output_class:
                    #         gain = 1
                    #     else:
                    #         gain = 0
                    tree_data_list.append(
                        {
                            "Tree": tree_index,
                            "class": class_index,
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

    # --------------------------
    # Code for evaluating trees
    # --------------------------

    def eval_tree_rec(self,data, nid, tree,verbose=0):
        rows = tree[tree["Node"] == nid]
        for idx, row in rows.iterrows():
            f = row["Feature"]
            if f == "Leaf":
                return row["Node"],row["Gain"]
            else:
                f = int(f[1:])
                diff = data[f] - row["Split"]
                if verbose > 8: print('Tree path:', row['Feature'],row['Split'],data[f],diff)
                # if diff != 0 and abs(diff) < 0.001: print(row['Feature'],row['Split'],diff)
                if self.split_kind == "<": c = data[f] <  row["Split"]
                elif self.split_kind == "<="   : c = data[f] <= row["Split"]
                else: assert(False)
                if c:
                    child = row["Yes"]
                else:
                    child = row["No"]
                child = child.split("-")[1]
                if child.isdigit(): child = int(child)
                return self.eval_tree_rec(data, child, tree,verbose)


    def eval_tree(self,data, tree,verbose=False):
        return self.eval_tree_rec(data, self.get_root_name(), tree,verbose)


    def eval_trees(self, inputs, verbose=False):
        if not isinstance(inputs[0], list): inputs = [inputs]
        outputs = []
        for data in inputs:
            class_vals = []
            class_nodes = []
            for c in range(0, self.n_classes ):
                vals = []
                nodes = []
                for i in range(0, self.n_trees):
                    tree = self.trees[(self.trees["Tree"] == i)&(self.trees["class"] == c)]
                    nid,v = self.eval_tree(data, tree, verbose)
                    vals.append(v)
                    nodes.append(nid)
                class_vals.append( sum(vals) )
                class_nodes.append((vals,nodes))
            if self.n_classes == 1:
                if self.model_library == "xgboost":
                    s = class_vals[0]+utils.sigmoid_inv(self.base_score[0])
                    output = round(1.0 / (1.0 + math.exp(-s)), 7)
                elif self.model_library == "rf":
                    if class_vals[0] < (self.n_trees/2):
                        output = 1
                    else:
                        output = 0
                elif self.model_library == "lgbm":
                    s = class_vals[0]+utils.sigmoid_inv(self.base_score[0])
                    output = round(1.0 / (1.0 + math.exp(-s)), 7)
                else:
                    self.unsupported_model()
            else:
                # a per-class base shifts each margin by a different amount, so it
                # only drops out of the argmax when every entry is identical
                base = np.asarray(self.get_base_value(), dtype=float)
                output = np.argmax(np.asarray(class_vals, dtype=float) + base)
            if verbose: output = (output,class_vals,class_nodes)                
            outputs.append( output )
        return outputs

    def eval_trees_compare(self,data1, data2, num_tree, trees):
        vals = []
        for i in range(0, self.n_trees):
            tree = self.trees[(self.trees["Tree"] == i)]
            v1 = self.eval_tree(data1, tree)
            v2 = self.eval_tree(data2, tree)
            if v1 != v2:
                print(i, v1, v2)
     
    def region2point(self, region):
        eps = 1e-6
        point = [0.0] * len(region)

        for idx, reg in enumerate(region):
            lb, ub = reg.lb, reg. ub
            lb_open = (reg.bracklb == '(')
            ub_open = (reg.brackub == ')')

            if pd.isna(lb) or pd.isna(ub):
                assert(pd.isna(lb) and pd.isna(ub))
                point[idx] = 0.0
                continue
            if lb == -np.inf and ub == np.inf:
                point[idx] = 0.0
                continue
            
            if not lb_open: 
                if math.ceil(lb) < ub: lb = math.ceil(lb)
                point[idx] = lb
                continue
            
            if not ub_open:
                if math.floor(ub) > lb: ub = math.floor(ub)
                point[idx] = ub
                continue
            
            if lb != -np.inf and ub != np.inf:
                lo = lb + eps if lb_open else lb
                hi = ub - eps if ub_open else ub
                if lo > hi:
                    point[idx] = (lb + ub) / 2.0
                else:
                    point[idx] = (lo + hi) / 2.0
                continue

            if lb == -np.inf:
                point[idx] = math.floor(ub - (eps if ub_open else 0.0))
                continue
            # ub = inf
            point[idx] = math.ceil(lb + (eps if lb_open else 0.0))

        return point

    
    def print_reg(self,region):
        data  = "[ "
        for idx,reg in enumerate(region):
            data += f"{idx}:{reg.printintv()} "
        data +=" ]"
        return data
                    
    def get_sample(self, f):
        typ     = self.feature_types[f]
        lb,ub   = self.op_range_list[f]
        if typ == 'linear':
            return np.random.uniform(lb, ub)
        elif typ == 'log':
            return np.exp( np.random.uniform( np.log(lb), np.log(ub) ) ) 
        elif typ in ('bool', 'categorical'):
            return np.random.choice([0, 1])
        else:
            print('Urecognized feature type')
            assert(False)
            
    def get_perturb(self, f, val):
        typ     = self.feature_types[f]
        lb,ub   = self.op_range_list[f]
        perturb = self.options.perturb
        if typ == 'linear':
            return perturb*(ub-lb)
        elif typ == 'log':
            return perturb*(np.log(ub)-np.log(lb))*val
        elif typ in ('bool', 'categorical'):
            return 2 # Ensures 0 1 range in the final
        else:
            print('Urecognized feature type')
            assert(False)
            
    def get_guard_list( self, f, op_range_list = None):
        update_list = False
        if op_range_list == None:
            op_range_list = self.op_range_list
            if f in self.guard_lists:
                return self.guard_lists[f]
            else:
                update_list = True
        trees = self.trees
        sliced = trees[(trees["Feature"] == f"f{f}")][["Feature", "Split"]].copy()
        sliced.sort_values(["Split"], inplace=True)
        sliced.drop_duplicates(inplace=True)
        sliced = sliced[(op_range_list[f][0] <= sliced["Split"]) & (sliced["Split"] <= op_range_list[f][1])]
        glist = sliced["Split"].tolist()
        if update_list == True:
            self.guard_lists[f] = glist
        return glist
        
    def plot_variations(self, data, features, op_range_list):
        if len(features) > 2:
            features = features[0:2]
        fvalues = []
        for feature in features:
            values = self.get_guard_list(feature, op_range_list)
            if self.split_kind == '<':
                values = [op_range_list[feature][0]] + values
            else:
                values = values + [op_range_list[feature][1]]
            fvalues.append(values)

        predictions = []
        if len(fvalues) == 2:
            for v in fvalues[1]:
                data[features[1]] = v
                rows = []
                for f0 in fvalues[0]:
                    data[features[0]] = f0
                    rows.append(data.copy())
                predict_col = self.predict(rows)
                predictions.append(predict_col)
        else:
            rows = []
            for f0 in fvalues[0]:
                data[features[0]] = f0
                rows.append(data.copy())
            predict_col = self.predict(rows)
            predictions.append(predict_col)

        plt.style.use("_mpl-gallery")
        if len(fvalues) == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(fvalues[0], predictions[0], linewidth=2.0)
            plt.ylabel("Predict")
            plt.xlabel(self.feature_names[features[0]])
        else:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(projection="3d")
            # ax = plt.figure().add_subplot(projection="3d",figsize=(10, 6))
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
                xlabel=self.feature_names[features[0]],
                ylabel=self.feature_names[features[1]],
                zlabel="Predict",
            )
        plt.tight_layout()
        if self.options.plot_file:
            utils.print_verbose(self.options, 1, "Plot is saved in file", self.options.plot_file)
            plt.savefig( self.options.plot_file, bbox_inches='tight', pad_inches=1.0, dpi=800 )
        else:
            plt.show()

    # def output(self,timetaken,sesnfeat, sensreg,outputvalues):
    #     print(f"# Time: {timetaken}")
    #     print(f"sensitive feature(s): {sesnfeat}")
    #     if sensreg == [] and outputvalues == []:
    #         print(f"Insensitive")
    #     elif len(sensreg)>0 and len(outputvalues)>0:
    #         region1, region2 = self.frmt_region(sensreg[0]), self.frmt_region(sensreg[1])
    #         print( 'Sensitive sample 1:', "[ "," , ".join(region1)," ]")
    #         print( 'Sensitive sample 2:', "[ "," , ".join(region2)," ]")

    #         print(f"Output Values: {np.array(outputvalues)}")
    #     else:
    #         print(f"something went wrong")
            

# ------------------------------------------------
#  To be removed
# -----------------------------------------------

    
