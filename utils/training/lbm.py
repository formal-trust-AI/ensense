import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- Load your data ---
# Assuming your data is in CSV format and includes a target column
# Replace with your actual file paths
train_df = pd.read_csv("../dataset/churn/train.csv")
test_df = pd.read_csv("../dataset/churn/test.csv")

# --- Specify features and target ---
target_col = 'label'  # change this to your actual target column name
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# --- Create LightGBM datasets ---
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# --- Set model parameters ---
params = {
    'objective': 'binary',  # or 'multiclass' for multi-class
    'metric': 'binary_logloss',  # or 'multi_logloss'
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'num_threads': 4
}

# --- Train the model ---
print("Training model...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=100,
    # early_stopping_rounds=10
)

# --- Make predictions ---
y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_prob > 0.5).astype(int)

# --- Evaluate ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Save model ---
model.save_model('lightgbm_model.joblib')

