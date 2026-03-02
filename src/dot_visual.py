import xgboost as xgb
import joblib

final_raw_output = 0
model_file = "../models/xgb_sbi.pkl"


def parse_tree(tree_str):
    tree = {}
    for line in tree_str.split("\n"):
        if line.strip() == "":
            continue
        splits = line.split(":")
        node_id = int(splits[0])
        if "leaf" in splits[1]:
            value = float(splits[1].split("=")[1])
            tree[node_id] = ("leaf", value)
        else:
            splits2 = splits[1].split("<")
            feature = splits2[0][splits2[0].index("[") + 1 :]
            threshold = float(splits2[1][: splits2[1].index("]")])
            yes_node = int(splits[1].split("yes=")[1].split(",")[0])
            no_node = int(splits[1].split("no=")[1].split(",")[0])
            tree[node_id] = (feature, threshold, yes_node, no_node)
    return tree


def predict(tree, input_data):
    node = 0
    while True:
        info = tree[node]
        if info[0] == "leaf":
            return info[1]
        else:
            feature, threshold, yes_node, no_node = info
            if input_data[feature] < threshold:
                node = yes_node
            else:
                node = no_node


input_data = {
    "ATM_DR_TXN_AMT_7M": 50000.0,
    "AGE": 30,
    "CREDIT_CNT_10M": 6,
    "ATM_DR_TXN_AMT_4M": 25000.0,
    "DEBIT_CNT_3M": 3,
    "L5M_DR_CR_AMT_RATIO": 0.8,
    "DEBIT_SUM_6M": 50000.0,
    "CREDIT_SUM_5M": 15000.0,
    "L8M_DR_CR_AMT_RATIO": 0.7,
    "CREDIT_SUM_4M": 20000.0,
    "UPI_DR_CNT_11M": 4,
    "L2M_DR_CR_AMT_RATIO": 0.9,
    "L6M_DR_CR_AMT_RATIO": 0.6,
    "L3M_DR_CR_AMT_RATIO": 0.85,
    "L11M_DR_CR_AMT_RATIO": 0.75,
    "L9M_DR_CR_AMT_RATIO": 0.65,
    "L7M_DR_CR_AMT_RATIO": 0.55,
    "LM_DR_CR_AMT_RATIO": 0.95,
    "CREDIT_CNT_2M": 4,
}

# input_data = {'ATM_DR_TXN_AMT_7M': 100000.00394702, 'AGE': 30.0, 'CREDIT_CNT_10M': 5.99605298, 'ATM_DR_TXN_AMT_4M': 500.99605298, 'DEBIT_CNT_3M': 2.99605298, 'L5M_DR_CR_AMT_RATIO': 1.45936203, 'DEBIT_SUM_6M': 20000.0, 'CREDIT_SUM_5M': 1499.00394702, 'L8M_DR_CR_AMT_RATIO': 0.169890001, 'CREDIT_SUM_4M': 1200.99605298, 'UPI_DR_CNT_11M': 4.00394702, 'L2M_DR_CR_AMT_RATIO': -0.0039240199, 'L6M_DR_CR_AMT_RATIO': 1.03115296, 'L3M_DR_CR_AMT_RATIO': 0.197712004, 'L11M_DR_CR_AMT_RATIO': 0.972496986, 'L9M_DR_CR_AMT_RATIO': 0.0025269799, 'L7M_DR_CR_AMT_RATIO': 0.977612972, 'LM_DR_CR_AMT_RATIO': 0.0050029797, 'CREDIT_CNT_2M': 1.99605298}

# prediction = predict(tree, input_data)
# print(prediction)

# exit()


model = joblib.load(model_file)

booster = model.get_booster()

num_boosters = len(booster.get_dump())

for i in range(num_boosters):
    tree_str = booster.get_dump()[i]
    tree = parse_tree(tree_str)
    prediction = predict(tree, input_data)
    final_raw_output += prediction

print(final_raw_output)
