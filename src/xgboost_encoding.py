#!/usr/bin/python3

import joblib
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import z3
import math
import random

# model_file = '../models/xgb_sbi.json'
model_file = "../models/xgb_sbi.pkl"


pd.set_option("display.max_rows", 500)


# pd.set_option('display.max_columns', None)
def create_plots():
    xgb_model = joblib.load(model_file)
    num_boosters = len(xgb_model.get_booster().get_dump())
    # os.makedirs('tree_plots', exist_ok=True)
    # for i in range(num_boosters):
    #     plt.figure(figsize=(100, 20))
    #     xgb.plot_tree(xgb_model.get_booster(), num_trees=i, ax=plt.gca())
    #     plt.savefig(f'tree_plots/tree_{i}.png')
    #     plt.close()
    #     # break
    # xgb_model = xgb.Booster()  # init model
    # xgb_model.load_model('xgb_sbi.pkl')  # load data

    os.makedirs("tree_dots", exist_ok=True)
    for i in range(num_boosters):
        tree_dot = xgb.to_graphviz(xgb_model, num_trees=i)
        tree_dot.format = "dot"
        tree_dot.render(f"tree_dots/tree_{i}")


model = joblib.load(model_file)
num_boosters = len(model.get_booster().get_dump())
max_depth = model.get_params()["max_depth"]
num_features = model.n_features_in_
learning_rate = model.get_params()["learning_rate"]

print(f"Number of boosters: {num_boosters}")
print(f"Maximum depth: {max_depth}")
print(f"Number of features: {num_features}")
print(f"Learning rate: {learning_rate}")

# example_input = np.random.rand(19)*100
# print(example_input)


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
input_data = {
    "ATM_DR_TXN_AMT_7M": 72500.0,
    "AGE": 35.9999935,
    "CREDIT_CNT_10M": 0.9999935,
    "ATM_DR_TXN_AMT_4M": 9.43999958,
    "DEBIT_CNT_3M": 57.0,
    "L5M_DR_CR_AMT_RATIO": 1.36038995,
    "DEBIT_SUM_6M": 63638.0,
    "CREDIT_SUM_5M": 96000.5937935,
    "L8M_DR_CR_AMT_RATIO": 0.0905574976,
    "CREDIT_SUM_4M": 24804.0,
    "UPI_DR_CNT_11M": 10.9999935,
    "L2M_DR_CR_AMT_RATIO": 0.0242100004,
    "L6M_DR_CR_AMT_RATIO": 10.0,
    "L3M_DR_CR_AMT_RATIO": 6.82201165,
    "L11M_DR_CR_AMT_RATIO": 6.4999e-06,
    "L9M_DR_CR_AMT_RATIO": 10.0,
    "L7M_DR_CR_AMT_RATIO": 10.0,
    "LM_DR_CR_AMT_RATIO": 0.0089499997,
    "CREDIT_CNT_2M": 0.9999935,
}

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
# print(dummy_df)
booster = model.get_booster()

predictions = model.predict(dummy_df, validate_features=False)
print(predictions)

class_probabilities = model.predict(
    dummy_df, output_margin=True, validate_features=False
)

# Print class probabilities and predicted classes
print("Class raw output", class_probabilities)

# dtest = xgb.DMatrix(dummy_df)

# # Now you can get the leaf indices
# leaf_indices = booster.predict(dtest, pred_leaf=True, validate_features=False)
# print(leaf_indices)


# prediction = predict_example(model, list(example_input.values()), expected_feature_names)

# print(f"Prediction for the example input: {prediction}")

# create_plots()
