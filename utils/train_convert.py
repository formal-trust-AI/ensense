import xgboost as xgb
import pickle
name = "Spambase_maxdepth_9"

# Step 1: Load the model from .pkl
with open(f"{name}.pkl", "rb") as f:
    model = pickle.load(f)

# Step 2: Save to XGBoost JSON format
# This only works if `model` is an instance of xgboost.Booster or xgboost.XGBModel
model.save_model(f"{name}.json")

m = xgb.Booster()
m.load_model(f"{name}.json")
print(f"number of trees: {len(m.get_dump())}")
# Print the depth of each tree
maxdepth = 0
avgdepth = 0
for i, tree in enumerate(m.get_dump()):
    tree_depth = max(int(line.split('[')[0].count('\t')) for line in tree.splitlines() if ':' in line)
    maxdepth = max(maxdepth, tree_depth)
    avgdepth += tree_depth
avgdepth /= len(m.get_dump())
print(f"max depth: {maxdepth}")
print(f"avg depth: {avgdepth}")