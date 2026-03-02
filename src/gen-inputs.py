#!/usr/bin/python3

import joblib
import xgboost as xgb
import shap
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import z3
import math
import random
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import pdb

from xyplot import Curve

pd.set_option("display.max_rows", 500)

model_file = "../models/xgb_sbi-2.0.3.json"

model = xgb.Booster({"nthread": 4})  # init model
model.load_model(model_file)  # load data
trees = model.trees_to_dataframe()
n_trees = model.num_boosted_rounds()
n_features = model.num_features()

names = [
    "CREDIT_SUM_4M",
    "DEBIT_SUM_6M",
    "CREDIT_CNT_2M",
    "CREDIT_SUM_5M",
    "L9M_DR_CR_AMT_RATIO",
    "CREDIT_CNT_10M",
    "L2M_DR_CR_AMT_RATIO",
    "L3M_DR_CR_AMT_RATIO",
    "ATM_DR_TXN_AMT_7M",
    "L5M_DR_CR_AMT_RATIO",
    "AGE",
    "ATM_DR_TXN_AMT_4M",
    "DEBIT_CNT_3M",
    "L6M_DR_CR_AMT_RATIO",
    "UPI_DR_CNT_11M",
    "L8M_DR_CR_AMT_RATIO",
    "LM_DR_CR_AMT_RATIO",
    "L7M_DR_CR_AMT_RATIO",
    "L11M_DR_CR_AMT_RATIO",
]

# 8804.340134089078, 18412.985864318813, 48.84088807517018, 334.7164864686726, 3.7759183108626635,
# 29.13720362946213, 29.684706918504993, 0.9145364364484933, 13.013191941121672, 0.8832918563726034, 18.760796656586034, 2.2306596984807303, 37.061669880754856, 1.0867956264766891, 10.49011668768002, 2.1004332527545957, 0.7996945391976434, 0.21436059909348723, 0.21783668968949738, 0


def transform(i, value):
    if "RATIO" in names[i]:
        return value / (1 + value)
    if "SUM" in names[i]:
        return np.log10(value) / 5.5
    if "TXN" in names[i]:
        return np.log10(value) / 4.9
    if "CNT" in names[i]:
        return value / 65
    if "AGE" in names[i]:
        return value / 68
    return value


def invert(i, value):
    if "RATIO" in names[i]:
        return value / (1 - value)
    if "SUM" in names[i]:
        return 10 ** (5.5 * value)  # np.log10(value)/5.5
    if "TXN" in names[i]:
        return 10 ** (4.9 * value)  # np.log10(value)/4.9
    if "CNT" in names[i]:
        return value * 65
    if "AGE" in names[i]:
        return value * 68
    return value


def gen_input():
    x = []
    rs = []
    for i in range(model.num_features()):
        v = random.uniform(0, 1)
        x.append(invert(i, v))
        rs.append(v)
    return x, rs


def run_model():
    data, rands = gen_input()
    dataM = xgb.DMatrix([data])
    prediction = model.predict(dataM)
    return data, rands, prediction[0]


for i in range(50000):
    data, rands, prediction = run_model()
    # print(*data, sep=", ", end=", ")
    print(*rands, sep=", ", end=", ")
    if prediction > 0.5:
        print(1)
    else:
        print(0)


# print(data,prediction)
exit()

for i in range(model.num_features()):
    # if 'RATIO' not in names[i]: continue
    sliced = trees[(trees["Feature"] == f"f{i}")][["Feature", "Split"]].copy()
    sliced.sort_values(["Split"], inplace=True)
    sliced.drop_duplicates(inplace=True)
    r = sliced["Split"].apply(lambda x: transform(i, x))
    print(r.max())
    # exit()
    # sliced = sliced[ (op_range_list[i][0] < sliced['Split']) & (sliced['Split'] <= op_range_list[i][1]) ]
    # print(np.log10(sliced['Split']))
    # print(sliced['Split']/(sliced['Split']+1))
    # print(sliced['Split'])

# f(+infty) = 1
# f(1) = 0.5
# f(0) = 0
# ln(x) [0,infty] -> [-\inty,infty]
# 1/1+e^{-ln(x)}
# 1/1+(1/x)
# y = x/x+1
# yx + y = x
# y = x(1-y)
# x = y/(1-y)
