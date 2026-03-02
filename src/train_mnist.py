import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import pickle
import os

# ========== Config ==========
save_pickle_path = "model_3vs8.pkl"
save_json_path = "model_3vs8.json"
# ============================

# 1. Load MNIST
print("📥 Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(str)

# 2. Filter for digits 3 and 8
mask = (y == '3') | (y == '8')
X_bin = X[mask]
y_bin = y[mask]
y_bin = np.where(y_bin == '3', 0, 1)  # 0 = '3', 1 = '8'

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

# 4. Train XGBoost Booster model
print("🧠 Training model...")
dtrain = xgb.DMatrix(X_train, label=y_train)
param = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.1,
    "nthread": 4,
    "n_estimators": 100,
    "depth": 6,
    
}
num_round = 100
model = xgb.train(param, dtrain, num_round)

# 5. Save model in both formats
print("💾 Saving model...")
with open(save_pickle_path, "wb") as f:
    pickle.dump(model, f)
model.save_model(save_json_path)

# 6. Load model with dual-format support
print("🔁 Testing model loading...")
model_loaded = xgb.Booster({"nthread": 4})
try:
    with open(save_pickle_path, "rb") as f:
        model_loaded = pickle.load(f)
    print("✅ Loaded model from pickle")
except Exception as e:
    print("⚠️ Pickle load failed, trying JSON...")
    model_loaded.load_model(save_json_path)
    print("✅ Loaded model from JSON")

# 7. Predict
dtest = xgb.DMatrix(X_test)
y_pred = model_loaded.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)

# 8. Evaluate
acc = accuracy_score(y_test, y_pred_binary)
print(f"🎯 Accuracy: {acc:.4f}")
print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))
