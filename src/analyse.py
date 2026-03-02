import pickle
import numpy as np
import xgboost as xgb
import sys
import matplotlib.pyplot as plt






#  with open("result.pkl", "wb") as f:
#     pickle.dump((x, x2), f)




if len(sys.argv) > 1:
    tag = sys.argv[1]
else:
    tag = 'default_tag'

# Path to your pickle file
# pickle_file_path = f'./counterexample_new/result_{tag}.pkl'
pickle_file_path = f'./resultww.pkl'

# Load the model and inputs
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# model_file = "xgb_mnist_8_vs_3_booster.json"
model_file = '../models/mnist_100_6.pkl'

# Extract the model and inputs
model = xgb.Booster({"nthread": 4})  # init model
try:
    model = pickle.load(open(model_file, "rb"))
except:
    model.load_model(model_file)  # load data

print(f"feature names: {model.num_features()}")

input1 = data[0]
input2 = data[1]

# Ensure inputs are in correct shape (2D array)
input1 = np.array(input1).reshape(1, -1)
input2 = np.array(input2).reshape(1, -1)

print(input1.shape)

# Predict the classes
pred1 = model.predict(xgb.DMatrix(input1))[0]
pred2 = model.predict(xgb.DMatrix(input2))[0]

# Output predictions
print(f"Prediction for input1: {pred1}")
print(f"Prediction for input2: {pred2}")

plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.imshow(input1.reshape(28, 28), cmap='gray')
plt.title(f'Prediction: {pred1}')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(input2.reshape(28, 28), cmap='gray')
plt.title(f'Prediction: {pred2}')
plt.axis('off')

plt.tight_layout()
plt.savefig(f'./resultww.png')
# plt.show()
