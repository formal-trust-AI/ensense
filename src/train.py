import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load MNIST data
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

# Filter for digits 8 and 3 only
mask = (y == '8') | (y == '3')
X, y = X[mask], y[mask]

# Convert labels to binary: 8 -> 1, 3 -> 0
y_binary = (y == '8').astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save the entire model (scikit-learn API)
joblib.dump(model, "xgb_mnist_8_vs_3_model.pkl")

# Optional: Save the internal XGBoost booster model
model.get_booster().save_model("xgb_mnist_8_vs_3_booster.json")
