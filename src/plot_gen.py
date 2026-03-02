import joblib
import pandas as pd
import matplotlib.pyplot as plt
import math


def parse_tree_thresholds(tree_str):
    tree = {}
    thresholds = {}
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

            if feature not in thresholds:
                thresholds[feature] = []
            thresholds[feature].append(threshold)
    return tree, thresholds


def get_threshold_vals(differing_feature):
    model = joblib.load(model_file)
    booster = model.get_booster()
    trees = []
    thresholds = {}
    for tree_str in booster.get_dump():
        tree, tree_thresholds = parse_tree_thresholds(tree_str)
        trees.append(tree)

        for feature, feature_thresholds in tree_thresholds.items():
            if feature not in thresholds:
                thresholds[feature] = []
            thresholds[feature].extend(feature_thresholds)
    return thresholds[differing_feature]


def xgb_predict(model, input_data):
    example_input = {key: [value] for key, value in input_data.items()}

    dummy_df = pd.DataFrame(example_input)
    dummy_df.columns = [
        "ATM_DR_TXN_AMT_7M",
        "AGE",
        "CREDIT_CNT_10M",
        "ATM_DR_TXN_AMT_4M",
        "DEBIT_CNT_3M",
        "L5M_DR_CR_AMT_RATIO",
        "DEBIT_SUM_6M",
        "CREDIT_SUM_5M",
        "L8M_DR_CR_AMT_RATIO",
        "CREDIT_SUM_4M",
        "UPI_DR_CNT_11M",
        "L2M_DR_CR_AMT_RATIO",
        "L6M_DR_CR_AMT_RATIO",
        "L3M_DR_CR_AMT_RATIO",
        "L11M_DR_CR_AMT_RATIO",
        "L9M_DR_CR_AMT_RATIO",
        "L7M_DR_CR_AMT_RATIO",
        "LM_DR_CR_AMT_RATIO",
        "CREDIT_CNT_2M",
    ]
    specified_order = model.get_booster().feature_names
    dummy_df = dummy_df[specified_order]
    class_probabilities = model.predict(
        dummy_df, output_margin=True, validate_features=False
    )
    return class_probabilities


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


model_file = "../models/xgb_sbi.pkl"
model = joblib.load(model_file)
differing_feature = "CREDIT_SUM_4M"
input_data = {
    "ATM_DR_TXN_AMT_7M": 50023.6016,
    "AGE": 26.0,
    "CREDIT_CNT_10M": 4.9990799745,
    "ATM_DR_TXN_AMT_4M": 8009.43994,
    "DEBIT_CNT_3M": 5.9990799745,
    "L5M_DR_CR_AMT_RATIO": 1.00307298,
    "DEBIT_SUM_6M": 3695.0,
    "CREDIT_SUM_5M": 138840.75,
    "L8M_DR_CR_AMT_RATIO": 0.3522819595,
    "CREDIT_SUM_4M": 10526.0,
    "UPI_DR_CNT_11M": 23.9990799745,
    "L2M_DR_CR_AMT_RATIO": 4.10344791,
    "L6M_DR_CR_AMT_RATIO": 0.6136479695,
    "L3M_DR_CR_AMT_RATIO": 0.1372329765,
    "L11M_DR_CR_AMT_RATIO": 0.363191009,
    "L9M_DR_CR_AMT_RATIO": 1.05759001,
    "L7M_DR_CR_AMT_RATIO": 0.571090996,
    "LM_DR_CR_AMT_RATIO": 0.978754997,
    "CREDIT_CNT_2M": 4.9990799745,
}
x_axis = (
    [0] + list(set(get_threshold_vals(differing_feature))) + [10526, 10008.1289799745]
)
x_axis.sort()
print(x_axis)
# print(xgb_predict(model,input_data))
y_axis = []
for i in range(len(x_axis)):
    temp = input_data
    temp[differing_feature] = x_axis[i]
    y_axis.append(sigmoid(xgb_predict(model, temp)[0]))
# print(y_axis)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, y_axis, marker="o")
plt.title(differing_feature)
plt.grid(True)
# plt.show()
plt.savefig(f"../plots/{differing_feature}.png")

plt.close()
