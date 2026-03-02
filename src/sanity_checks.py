import numpy as np
import pickle
import joblib
import lightgbm
import xgboost as xgb
import sys

def load_file(model_file):
        # --------------------------------------------------
        # Load model, filename decides the type of the model
        # --------------------------------------------------
        fname = model_file
        if fname[-4:] == ".pkl":
            return "pickle", pickle.load(open(model_file, "rb"))
        elif fname[-4:] == ".sav":
            return "pickle", pickle.load(open(model_file, "rb"))
        elif fname[-7:] == ".joblib":
            return "joblib", joblib.load(model_file)
        elif fname[-4:] == ".txt": # lgbm
            # todo: lgbm has extention .txt. Can we modify it
            return "lightgbm", lightgbm.Booster(model_file = model_file)
        elif fname[-5:] == ".json":
            # self.model = pickle.load(open(self.model_file, "rb"))
            model = xgb.Booster({"nthread": 4})
            model.load_model(model_file)
            return "xgb", model   # init model
        else:
            print("Failed to indentify the file type!")
            sys.exit(1)
    
            
def sanity(modelfile, x1, x2, out1, out2,tol=1e-3):
    modeltype, model = load_file(modelfile)
    
    X1 = np.asarray(x1, dtype=float).reshape(1,-1)
    X2 = np.asarray(x2, dtype=float).reshape(1,-1)
    
    if modeltype == 'xgb':
        p1 = model.predict(xgb.DMatrix(X1),validate_features=False)
        p2 = model.predict(xgb.DMatrix(X2),validate_features=False)
    else: 
        p1 = model.predict(X1)
        p2 = model.predict(X2)
    
    p1 = np.asarray(p1,dtype=float).reshape(-1)
    p2 = np.asarray(p2,dtype=float).reshape(-1)
    out1 = np.asarray(out1,dtype=float).reshape(-1)
    out2 = np.asarray(out2,dtype=float).reshape(-1)
    if abs(p1-out1) <= tol and abs(p2-out2) <= tol:
        return True
    return False

#call sanity(options.model_file,x,x2,pred1,pred2)