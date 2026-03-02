import joblib
import os
import pandas as pd
import re
import sys
from typing import List, Tuple
import numpy as np
import json, math
import xgboost as xgb

def remap_point(actual_vals, names, encoding_map):
    out = {}
    for i, (v, name) in enumerate(zip(actual_vals, names)):
        if name in encoding_map:
            cats = encoding_map[name]  # e.g., ["A11","A12",...]
            # assume ordinals 0..K-1 -> categories; round & clamp
            idx = int(round(v))
            idx = max(0, min(len(cats) - 1, idx))
            out[name] = cats[idx]
        else:
            out[name] = float(v)
    return out

model_name = 'german_credit'
datsetpath = f'./models/dataset/{model_name}/train.csv'
data2 = f'./models/dataset/{model_name}/test.csv'
# combined_data = f'./models/dataset/{model_name}/full.csv'
df = pd.read_csv(datsetpath)
# df2 = pd.read_csv(data2)
# df = pd.concat([df1,df2])
# df.to_csv(combined_data)
# print(df.columns)
# df = df.drop(columns=['label'])
featurenames = [f for f in df.columns if f !='label']
modelpath = f'./models/{model_name}/{model_name}_t800_d6.json'
scalefile = f'./models/dataset/{model_name}/scaler.pkl'
featuremap = f'./models/dataset/{model_name}/encoding_map.json'
clausefile = f'./output/learned-clauses_{model_name}_t800_d6.txt'
detailfile = f'./models/dataset/{model_name}/details.csv'
with open(scalefile, "rb") as f:
    scaler = joblib.load(f)
    
with open(featuremap, "r", encoding="utf-8") as f:
    enc_map = json.load(f)

os.system(f"python3 ./src/sensitive.py --prob --solver milp --gap 3 {modelpath}  --data_file {datsetpath} --all_opt  --features 0 --compute_data_distance --details {detailfile} --in_distro_clauses {clausefile} > result.txt")


def encode_value(name, val):
    if name in enc_map: 
        cats = enc_map[name]
        if isinstance(val, str):
            try:
                return cats.index(val)  
            except ValueError:
                raise ValueError(f"{name}: unknown category '{val}'. Expected one of {cats}.")
        idx = int(math.floor(float(val)))
        return max(0, min(len(cats)-1, idx))
    else:
        return float(val)

text = open('result.txt', encoding="utf-8").read()
text = re.sub(r"\x1b\[[0-9;]*m", "", text)  

m = re.search(r"Sensitive\s*sample\s*1:\s*\[(.*?)\].*?Sensitive\s*sample\s*2:\s*\[(.*?)\]",
              text, flags=re.I|re.S)
if not m: raise SystemExit("Pairs not found.")

to_floats = lambda s: [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s)]
s1, s2 = to_floats(m.group(1)), to_floats(m.group(2))
print(s1,s2)

X1_raw_df = pd.DataFrame([s1], columns=featurenames)
X2_raw_df = pd.DataFrame([s2], columns=featurenames)

X1_scaled = scaler.transform(X1_raw_df.values)                # returns ndarray
X1_scaled_df = pd.DataFrame(X1_scaled, columns=featurenames) 
X2_scaled = scaler.transform(X2_raw_df.values)                # returns ndarray
X2_scaled_df = pd.DataFrame(X2_scaled, columns=featurenames) 

bst = xgb.Booster()
bst.load_model(modelpath)
dmat1 = xgb.DMatrix(X2_scaled_df)
dmat2 = xgb.DMatrix(X2_scaled_df)
pred1 = bst.predict(dmat1)
pred2 = bst.predict(dmat2)
print(f"********pred; : {pred1}  pred2 {pred2}")

X_scaled = np.vstack([s1, s2])               
X_actual = scaler.inverse_transform(X_scaled)
s1_actual = X_actual[0].tolist()
s2_actual = X_actual[1].tolist()

print("s1 (actual) =", s1_actual)
print("s2 (actual) =", s2_actual)

s1_human = remap_point(s1_actual, featurenames, enc_map)
s2_human = remap_point(s2_actual, featurenames, enc_map)
print(f"datset 1")
for k in featurenames:
    print(f"{k}: {s1_human[k]}")
print(f"datset2")
for k in featurenames:
    print(f"{k}: {s2_human[k]}")
    
x1_raw = np.array([encode_value(n, s1_human[n]) for n in featurenames], dtype=float).reshape(1, -1)
x2_raw = np.array([encode_value(n, s2_human[n]) for n in featurenames], dtype=float).reshape(1, -1)
print(x1_raw)
x1_scaled = scaler.transform(x1_raw)
x2_scaled = scaler.transform(x2_raw)
x1_raw_list = [encode_value(n, s1_human[n]) for n in featurenames]
X1_raw_df = pd.DataFrame([x1_raw_list], columns=featurenames)
x2_raw_list = [encode_value(n, s2_human[n]) for n in featurenames]
X2_raw_df = pd.DataFrame([x2_raw_list], columns=featurenames)
# x1 = [-0.16666682849999997, 0.279412 , 0.625 , -0.388888944 , 0.059425500000000006 , -0.25 , 0.625 , 0.5 , 0.5 , 0.0 , 0.5 , 0.5 , 0.3125 , 0.75 , 0.75 , -0.16666682849999997 , 0.1666668435 , 0.5 , 0.5 , 0.5 ]
# x2 = [ 1.5 , 0.279412 , 0.625 , -0.388888944 , 0.059425500000000006 , -0.25 , 0.625 , 0.5 , 0.5 , 0.0 , 0.5 , 0.5 , 0.3125 , 0.75 , 0.75 , -0.16666682849999997 , 0.1666668435 , 0.5 , 0.5 , 0.5 ]
# X1_raw_df = pd.DataFrame([x1], columns=featurenames)
# X2_raw_df = pd.DataFrame([x2], columns=featurenames)

X1_scaled = scaler.transform(X1_raw_df.values)                # returns ndarray
X1_scaled_df = pd.DataFrame(X1_scaled, columns=featurenames) 
X2_scaled = scaler.transform(X2_raw_df.values)                # returns ndarray
X2_scaled_df = pd.DataFrame(X2_scaled, columns=featurenames) 

bst = xgb.Booster()
bst.load_model(modelpath)
dmat1 = xgb.DMatrix(X2_scaled_df)
dmat2 = xgb.DMatrix(X2_scaled_df)
pred1 = bst.predict(dmat1)
pred2 = bst.predict(dmat2)
print(f"pred; : {pred1}  pred2 {pred2}")