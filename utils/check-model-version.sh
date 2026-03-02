#!/bin/bash

# List of versions to check
versions=("0.71" "0.72.1" "0.80" "0.81" "0.82" "0.90" "1.0.0rc2" "1.0.0" "1.0.1" "1.0.2" "1.1.0rc1" "1.1.0rc2" "1.1.0" "1.1.1" "1.2.0rc2" "1.2.0" "1.2.1rc1" "1.2.1" "1.3.0rc1" "1.3.0.post0" "1.3.1" "1.3.2" "1.3.3" "1.4.0rc1" "1.4.0" "1.4.1" "1.4.2" "1.5.0rc1" "1.5.0" "1.5.1" "1.5.2" "1.6.0rc1" "1.6.0" "1.6.1" "1.6.2" "1.7.0rc1" "1.7.0.post0" "1.7.1" "1.7.2" "1.7.3" "1.7.4" "1.7.5" "1.7.6" "2.0.0rc1" "2.0.0" "2.0.1" "2.0.2" "2.0.3")
versions=("1.7.1" "1.7.2" "1.7.3" "1.7.4" "1.7.5" "1.7.6" "2.0.0rc1" "2.0.0" "2.0.1" "2.0.2" "2.0.3")

# versions=("1.0.0")



model_path="../models/xgb_sbi.pkl"

python_script="
import pickle
import xgboost
print('XGBoost version:', xgboost.__version__)
with open('$model_path', 'rb') as f:
    model = pickle.load(f)
print('Model loaded successfully')
"

for version in ${versions[@]}; do
    echo "Trying XGBoost version $version"
    pip3 uninstall -y xgboost
    pip3 install xgboost==$version
    python3 -c "$python_script"
    if [ $? -eq 0 ]; then
        echo "Success with XGBoost version $version"
        break
    else
        echo "Failure with XGBoost version $version"
    fi
done
