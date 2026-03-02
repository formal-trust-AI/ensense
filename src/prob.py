from sklearn.datasets import fetch_openml, load_iris, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import bisect
import math 
import pandas as pd
import xgboost as xgb
import json

def load_breast_cancer_data():
    print("loading breast cancer dataset")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

def load_pima_diabetes_data():
    print("loading pima diabetes dataset")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    data = pd.read_csv(url, header=None, names=columns)
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    # Handle missing values (0s in some columns)
    imputer = SimpleImputer(missing_values=0, strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X, y

def load_diabetes_data():
    data = pd.read_csv("./models/dataset/diabetes/diabetes_train.csv")
    # Rename columns from f1, f2, ... to 1, 2, ...
    new_columns = {}
    for col in data.columns:
        if col.startswith("f"):
            new_columns[col] = str(int(col[1:]))
    data = data.rename(columns=new_columns)
    data['0'] = 0
    # input()
    X = data.drop("label", axis=1)
    y = data["label"]
    return X, y

def load_spambase_data():
    print("loading spambase dataset")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    data = pd.read_csv(url, header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

def load_credit_card_fraud_data():
    print("loading credit card fraud dataset")
    # Ensure the file 'creditcard.csv' is in the same directory as your script
    try:
        import kagglehub
        # Download latest version
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

        print("Path to dataset files:", path)
        data = pd.read_csv(path + '/creditcard.csv')  # Download the dataset from Kaggle first

    except FileNotFoundError:
        print("Error: 'creditcard.csv' not found. Please download it from Kaggle and place it in the same directory.")
        return None, None
    X = data.drop("Class", axis=1)
    y = data["Class"]
    return X, y

def load_adult_income_data():
    print("loading adult income dataset")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", 
               "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    data = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
    data = data.dropna()
    X = data.drop("income", axis=1)
    y = data["income"].apply(lambda x: 1 if x == ">50K" else 0)
    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    return X, y

def load_higgs_data():
    print("loading higgs dataset")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    data = pd.read_csv(url, header=None, compression="gzip")
    
    X = data.iloc[:, 1:]  # Features (columns 1 to 28)
    y = data.iloc[:, 0]   # Target (column 0)
    return X, y

def load_3v8_mnist_data():
    print("loading mnist")
    # Load MNIST data
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]

    # Filter for digits 8 and 3 only
    mask = (y == '8') | (y == '3')
    X, y = X[mask], y[mask]

    # Convert labels to binary: 8 -> 1, 3 -> 0
    y_binary = (y == '3').astype(int)
    return X, y_binary


def getdatafile(data_file):
    # print(f"loading data from {data_file}")
    data = pd.read_csv(data_file)
    # Rename columns from f1, f2, ... to 1, 2, ...
    new_columns = {}
    for col in data.columns:
        if col.startswith("f") and col!='fnlwgt' and col != 'fixed acidity' and col!='free sulfur dioxide':
            new_columns[col] = str(int(col[1:]))
        else:
            new_columns[col] = col
    data = data.rename(columns=new_columns)
    data['0'] = 0
    X = data.drop("label", axis=1)
    y = data["label"]
    return X, y
    


def get_data(dataset):
    
    if(dataset == 0):
        return load_adult_income_data()
    elif dataset == 1:
        return load_breast_cancer_data()
    elif dataset == 2:
        return load_credit_card_fraud_data()
    elif dataset == 3:
        return load_pima_diabetes_data()
    elif dataset == 4:
        return load_spambase_data()
    elif dataset == 5:
        return load_higgs_data()
    elif dataset == 6:
        return load_diabetes_data()
    else:
        return load_3v8_mnist_data()

def get_mean(X, y):
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    mean_pos = np.mean(X_pos, axis=0)
    mean_neg = np.mean(X_neg, axis=0)

    return mean_pos, mean_neg

def smallest_greater_than_k(l, k):
    index = bisect.bisect_right(l, k)  # Find insertion point for k
    return l[index] if index < len(l) else None  # Return element if exists

def feat_name(n):
    return f"f{n}"

def sort_filter(lis):
    if(len(lis) == 0):
        return lis
    lis = sorted(lis)
    nl = []
    nl.append(lis[0])
    for i in range(1, len(lis)):
        if(lis[i]!=lis[i-1]):
            nl.append(lis[i])
    return nl

def getprob(inp,probs, guards):
    n_features = len(guards.keys())
    curprob = []
    for i in range(n_features):
        val = inp[i]
        repguard = smallest_greater_than_k(guards[feat_name(i)],val)
        curprob.append(probs[i][repguard])

    return curprob

def createprobs(model, X, y,round_digit):
    X = X[y == 1]
    n_features = model.n_features
    trees = model.trees
    guard = {}
    for i in range(n_features):
        guard[feat_name(i)] = [np.inf]
    for idx, row in trees.iterrows():
        if row["Feature"] != "Leaf":
            guard[row["Feature"]].append(round(row["Split"],round_digit))
    for i in range(n_features):
        guard[feat_name(i)] = sort_filter(guard[feat_name(i)])

    probs = {}

    for i in range(n_features):
        probs[i] = {}
        for g in guard[feat_name(i)]:
            probs[i][g] = 0

    if not isinstance(X, np.ndarray):
        X = X.to_numpy()
    for i in range(min(len(X), 1000)):
        for j in range(n_features-1):
            # print(f"i: {i}, j: {j}, X[i][j]: {X[i][j]}")
            repguard = smallest_greater_than_k(guard[feat_name(j)],X[i][j])
            probs[j][repguard] += 1/min(len(X), 1000)
    return probs, guard, None
#     dmat = xgb.DMatrix(X[:1000]) 



# # Predict leaf indices
#     leaf_indices = model.model[0].predict(dmat, pred_leaf=True)
#     df = pd.DataFrame(leaf_indices)

#     # Get all tree dump as JSON
#     trees = model.model[0].get_dump(dump_format="json")

#     # Collect all leaf indices per tree
#     all_leaf_indices_per_tree = []
#     for tree_json in trees:
#         tree_obj = json.loads(tree_json)
#         leaves = []

#         # Recursively walk the tree to find all leaf node IDs
#         def collect_leaves(node):
#             if "leaf" in node:
#                 leaves.append(node['nodeid'])
#             else:
#                 collect_leaves(node['children'][0])
#                 collect_leaves(node['children'][1])

#         collect_leaves(tree_obj)
#         all_leaf_indices_per_tree.append(sorted(leaves))

#     # Now count how many samples hit each leaf
#     count_list = []
#     for tree_idx, leaves in enumerate(all_leaf_indices_per_tree):
#         sample_leaves = df[tree_idx].value_counts().to_dict()
#         for leaf in leaves:
#             count = sample_leaves.get(leaf, 0)
#             count_list.append(count)

    return probs, guard, None

def addprob(probs):
    return np.sum(probs)

def mulprob(probs):
    return math.prod(probs)

def get_dist(x, X, k = 0):
    """
    0 = mean
    k = k-nearest sum
    """

    if(k == 0):
        mean_X = np.mean(X, axis=0)
        distance = np.linalg.norm(x - mean_X)

        return distance
    else:
        distances = np.linalg.norm(X - x, axis=1)    
        k_nearest_distances = np.sort(distances)[:k]
        return np.sum(k_nearest_distances)
