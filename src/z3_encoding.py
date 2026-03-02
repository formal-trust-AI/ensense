from z3 import *
import joblib
import time

start_time = time.time()

model_file = "../models/xgb_sbi.pkl"

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


def encode_tree(tree, variables, node_id=0):
    node = tree[node_id]
    if node[0] == "leaf":
        return node[1]
    else:
        feature, threshold, yes_node, no_node = node
        feature_var = variables[feature]
        return If(
            feature_var < threshold,
            encode_tree(tree, variables, yes_node),
            encode_tree(tree, variables, no_node),
        )


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


def get_threshold_vals():
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
    for feature in input_data.keys():
        thresholds[feature] = list(set(thresholds[feature]))
        thresholds[feature].sort()
    return thresholds


unk_input_data = {feature: Real(feature) for feature in input_data.keys()}
# unk_input_data = input_data
x_prime = {feature: Real(f"{feature}_prime") for feature in input_data.keys()}

model = joblib.load(model_file)
booster = model.get_booster()

trees = [parse_tree(tree_str) for tree_str in booster.get_dump()]

tree_expressions = [encode_tree(tree, unk_input_data) for tree in trees]
output = sum(tree_expressions)

tree_expressions_prime = [encode_tree(tree, x_prime) for tree in trees]
output_prime = sum(tree_expressions_prime)

delta = 1

# constraint1 = output - output_prime > delta
# constraint2 = output_prime - output > delta

epsilon_in_percent = 10

epsilon_features = {
    feature: epsilon_in_percent * input_data[feature] / 100
    for feature in input_data.keys()
}

# print(epsilon_features)

feature_constraints = [
    If(
        unk_input_data[feature] - x_prime[feature] >= 0,
        unk_input_data[feature] - x_prime[feature],
        x_prime[feature] - unk_input_data[feature],
    )
    < epsilon_features[feature]
    for feature in input_data.keys()
]

# print(feature_constraints)
# exit()
equalities = [
    unk_input_data[feature] == x_prime[feature] for feature in input_data.keys()
]
# differing_feature = 'CREDIT_CNT_2M'
# equalities = [unk_input_data[feature] == x_prime[feature] for feature in input_data.keys() if feature != differing_feature]
int_equalities = [If(equality, 1, 0) for equality in equalities]

thresholds = get_threshold_vals()

s = Solver()
s.add(Sum(int_equalities) == len(input_data) - 1)
s.add(z3.Abs(output) > delta)
s.add(z3.Abs(output_prime) > delta)
# s.add(Or(z3.Abs(output)>10*z3.Abs(output_prime),z3.Abs(output_prime)>10*z3.Abs(output)))
s.add(*feature_constraints)
s.add(output * output_prime < 0)
for feature in input_data.keys():
    s.add(unk_input_data[feature] > 0.1)
    s.add(unk_input_data[feature] <= input_data[feature] * 10)
    s.add(x_prime[feature] > 0.1)
    s.add(x_prime[feature] <= input_data[feature] * 10)
    s.add(unk_input_data[feature] > thresholds[feature][10])
    s.add(unk_input_data[feature] < thresholds[feature][-10])
    # print(feature,thresholds[feature])
    s.add(x_prime[feature] > thresholds[feature][10])
    s.add(x_prime[feature] < thresholds[feature][-10])
# s.add(unk_input_data[differing_feature] > thresholds[10])
# s.add(unk_input_data[differing_feature] < thresholds[-1])
# s.add(x_prime[differing_feature] > thresholds[10])
# s.add(x_prime[differing_feature] < thresholds[-1])
# print(thresholds)
if s.check() == sat:
    print("Counterexample found")
    m = s.model()
    x_values = {
        feature: float(m.evaluate(var).as_decimal(10).rstrip("?"))
        for feature, var in unk_input_data.items()
    }
    print(x_values)
    x_prime_values = {
        feature: float(m.evaluate(var).as_decimal(10).rstrip("?"))
        for feature, var in x_prime.items()
    }
    print(x_prime_values)
    differing_feature = None
    for feature, equality in zip(input_data.keys(), equalities):
        if m.evaluate(equality) == False:
            differing_feature = feature
            break
    print(differing_feature, thresholds[differing_feature])
    print(f"Differing feature: {differing_feature}")
    print(
        f"Value in unk_input_data: {m.evaluate(unk_input_data[differing_feature]).as_decimal(10).rstrip('?')}"
    )
    print(
        f"Value in x_prime: {m.evaluate(x_prime[differing_feature]).as_decimal(10).rstrip('?')}"
    )
    print("Output:", m.evaluate(output).as_decimal(10).rstrip("?"))
    print("Output Prime:", m.evaluate(output_prime).as_decimal(10).rstrip("?"))
else:
    print("No counterexample found")

end_time = time.time()

time_taken = end_time - start_time

print(f"Time taken to run the script: {time_taken} seconds")
