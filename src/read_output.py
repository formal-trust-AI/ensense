import json
import pandas as pd
import utils
import numpy as np
import xgboost as xgb
from scipy import sparse
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler

# filename = '../models/adult/adult_t200_d6.json'
# scalefile = '../models/dataset/adult/encoding_map.json'
# datafile = '../models/dataset/adult/adult.csv'
# featuremap = '../models/dataset/adult/feature_map.json'

filename = '../models/pimadiabetes/pimadiabetes_t200_d5.json'
scalefile = '../models/dataset/pimadiabetes/encoding_map.json'
datafile = '../models/dataset/pimadiabetes/pimadiabetes.csv'
featuremap = '../models/dataset/pimadiabetes/feature_map.json'


# FEATURE_COLS = [
#     'age', 'workclass', 'fnlwgt', 'education', 'education-num',
#     'marital-status', 'occupation', 'relationship', 'race', 'sex',
#     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
# ]

FEATURE_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


def compute_scaling_func():
    with open(scalefile, "r") as f:
        encoding_map = json.load(f)

    adult_df = pd.read_csv(datafile)
    encoded_df = adult_df.copy()

    for col, categories in encoding_map.items():
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        encoded_df[col] = encoded_df[col].map(mapping).fillna(-1).astype(int)
    # print(encoded_df[FEATURE_COLS])
    scaler = MinMaxScaler()
    encoded_df[FEATURE_COLS] = scaler.fit_transform(encoded_df[FEATURE_COLS])

    # print("Scaled DataFrame (first 5 rows):")
    # print(encoded_df[FEATURE_COLS].head()) 
    # scaler.fit(encoded_df[FEATURE_COLS])
    # print(encoded_df[FEATURE_COLS].head())  # note the ()
    return scaler

def decode_point(original_point):

    with open(scalefile, "r") as f:
        encoding_map = json.load(f)
    categorical_cols = encoding_map.keys()
    decoded = {}
    for col, val in zip(FEATURE_COLS, original_point):
        if col in categorical_cols:
            idx = int(round(val))
            cats = encoding_map[col]
            if 0 <= idx < len(cats): decoded[col] = cats[idx]
            else: decoded[col] = f"<unknown_{idx}>"
        else: decoded[col] = val
    return decoded

def correct_point(point):
    return [i if 0 <= i <= 1 else (0.0000001 if i < 0 else 1) for i in point]

def retrace_original(scaler,point):
    point = np.array(point)
    unscaled_point = scaler.inverse_transform([point])
    unscaled_point = [ round(i,6) for i in unscaled_point[0]]
    unscaled_pointtt = unscaled_point.copy()
    return unscaled_pointtt,decode_point(unscaled_point)
    
def correctwithfeaturename(point):
    return {i: point[i] if 0 <= point[i] <= 1 else (0.0000001 if point[i] < 0 else 1) for i in point.keys()}

def clamp_point(point):
    out = {}
    for k, v in point.items():
        try:
            if v < 0:
                out[k] = 1e-7
            elif v > 1:
                out[k] = 1
            else:
                out[k] = v
        except TypeError:
            # Non-numeric => decide how to handle; here we set to None
            out[k] = None
    return out

if __name__ == '__main__':
    bst = utils.open_model(filename,None ,details_file = None) #NV
    model = bst[0]
    print(bst[3])
    # points = ['__', ((0.25, 0.375), 0.5), ((0.359579, 1.404479027), 0.1458075820757664), ((1, 2.0), 1.0), ((0.933333, 2.0), 0.6), ((1, 2.0), 0.6666666666666666), ((1, 2.0), 0.8571428571428571), ((0.6, 0.8), 0.6000000000000001), ((0.75, 1), 1.0), ((1, 2.0), 1.0), ((0.105661, 0.106051), 0.0), ((0.520432, 0.539945), 0.4306703397612488), ((0.867347, 1), 0.6020408163265305), ((0.731707, 0.756098), 0.951219512195122)]
    # points =  ['__', ((0.5, 0.625), 0.5), ((0.109728, 0.116297), 0.135473574116081), ((0.6, 0.666667), 0.6), ((0.8, 0.866667), 0.8), ((0.333333, 0.5), 0.3333333333333333), ((0.285714, 0.357143), 0.2857142857142857), ((-0.799999997, 0.2), 0.0), ((0.5, 1), 0.75), ((0.0, 1), 1.0), ((-0.99885998864, 0.00114), 0.0), ((-0.9511019289, 0.18595), 0.0), ((0.387755, 0.408163), 0.3979591836734693), ((0.804878, 0.878049), 0.8048780487804879)]
    # points = [((0.890411, 1), 0.3150684931506849), ((1, 2.0), 0.5), ((0.310796, 0.359579), 0.1872414120970918), ((0.733333, 0.8), 0.6), ((0.933333, 2.0), 0.8), ((0.666667, 1), 0.6666666666666666), ((1, 2.0), 0.9285714285714284), ((0.4, 0.8), 0.2), ((-0.75, 0.25), 0.25), '__', ((0.105661, 0.106051), 0.0), ((0.597567, 1.845500469), 0.5183654729109275), ((1, 2.0), 0.4795918367346938), ((0.439024, 0.609756), 0.7317073170731708)]
    # points = [((0.39726, 0.438356), 0.3835616438356165), ((0.5, 0.625), 0.5), ((0.176192, 0.187245), 0.1536803357737602), ((0.6, 0.666667), 0.6), ((0.8, 0.866667), 0.8), ((0.333333, 0.5), 0.3333333333333333), ((0.071429, 0.285714), 0.2857142857142857), ((-0.799999997, 0.2), 0.0), ((0.5, 1), 1.0), '__', ((-0.99885998864, 0.00114), 0.0), ((-0.9511019289, 0.18595), 0.0), ((0.5, 0.540816), 0.3979591836734693), ((0.243902, 0.292683), 0.2195121951219512)]
    # points = [((0.588235, 1.7058823699999999), 0.5882352941176471), ((0.909548, 1.949748755), 0.7437185929648241), '__', ((0.333333, 0.363636), 0.4848484848484848), ((0.385343, 1.6382978559999999), 0.2801418439716312), ((0.678092, 1.67809242), 0.5603576751117736), ((0.581981, 1.581981242), 0.3941076003415883), ((0.65, 1.6499999760000001), 0.5)]
    points = [((-0.9411764704, 0.058824), 0.0), ((0.693467, 0.723618), 0.6633165829145728), '__', ((-0.9292929322, 0.070707), 0.0), ((-0.9810874704, 0.043735), 0.0), ((0.482861, 0.493294), 0.4828614008941878), ((0.113151, 0.129804), 0.1345004269854824), ((-0.9833333325, 0.016667), 0.0)]
    lb = [-1]*int(bst[3])
    datapoint = [-1]*int(bst[3])
    ub = [-1]*int(bst[3])
    print(lb)
    for i in range(len(points)):
        if points[i] == '__':
            lb[i] = 0#1#0#0.3698630136986301 #0.2054794520547945
            ub[i] = 1#1#0#0.3698630136986301#0.2054794520547945
            datapoint[i] = 0.639344262295082#1#0#0.3698630136986301#0.2054794520547945
        else:
            lb[i] = points[i][0][0]
            ub[i] = points[i][0][1]
            datapoint[i] = points[i][1]
    print(lb,"\n",ub,"\n",datapoint)
    scaler = compute_scaling_func()
    unscalespointlb,originalpointlb = retrace_original(scaler,lb)
    unscalespointub,originalpointub = retrace_original(scaler,ub)
    
    unscaleddatapoint,datapoint = retrace_original(scaler,datapoint)
    
    
    print(unscalespointlb,unscalespointub,unscaleddatapoint)
    print(originalpointlb,"\n",originalpointub,"\n",datapoint)
    with open(scalefile,'r') as f:
        featsenc = json.load(f)
    print("\n",featsenc)
    poss_data = {}
    print(type(unscalespointub))
    for idx,key in enumerate(datapoint.keys()):
        lb = min(unscalespointlb[idx],unscaleddatapoint[idx])
        ub = max(unscalespointub[idx],unscaleddatapoint[idx])
        if key in featsenc:
            temp = [featsenc[key][i] for i in range(int(np.ceil(lb)), min(int(np.floor(ub)) + 1,len(featsenc[key])))]
            poss_data[key] = temp
        else:
            poss_data[key] = [int(np.ceil(lb)), int(np.floor(ub))]
            
    print(poss_data)
    print(f"----------------------------------------------------------------------------")
    train_data = pd.read_csv(datafile)
    for idx,row in train_data.iterrows():
        flag = True
        for key in poss_data.keys():
            if key in featsenc.keys():
                if row[key] not in poss_data[key]:
                    flag = False
            else:
                if poss_data[key][0]<=row[key]<=poss_data[key][1]:
                    pass
                else:
                    flag = False
        if flag == True:
            print(idx,row.to_dict())
        
    # for i in range(len(points)):
    #     if points[i] == '__':
    #         lb[f"f{i}"] = '__'
    #         ub[f"f{i}"] = '__'
    #         datapoint[f"f{i}"] = '__'
    #     else:
    #         lb[f"f{i}"] = points[i][0][0]
    #         ub[f"f{i}"] = points[i][0][1]
    #         datapoint[f"f{i}"] = points[i][1]\
    
    
    # lb = clamp_point(lb)
    # ub = clamp_point(ub)
    # datapoint = clamp_point(datapoint)
    # print(lb,"\n",ub,"\n",datapoint)
    # point1 = [0.5882352941176471,0.7437185929648241,0.6885245901639344,0.48484848484848486,0.2801418439716312,0.5603576751117736,0.3941076003415883,0.5]
    # point1 = [ 0.4041095 , 0.5625 , 0.09353 , 0.6333335 , 0.8333335 , 0.4166665 , 0.3214285 , -0.29999999850000003 , 0.625 , 0.5 , -0.49885999432 , -0.45110196445 , 0.403061 , 0.0609755 ]
    # point2 = [ 0.2260275 , 0.5625 , 0.09353 , 0.6333335 , 0.8333335 , 0.4166665 , 0.3214285 , -0.29999999850000003 , 0.625 , 0.5 , -0.49885999432 , -0.45110196445 , 0.403061 , 0.0609755 ]
    # point1  = [0.2054794520547945,0.5,0.14580758207576644,1.0,0.6,0.6666666666666666,0.8571428571428571,0.6000000000000001,1.0,1.0,0.0,0.43067033976124885,0.6020408163265305,0.951219512195122]
    point1 = [0.0,0.6633165829145728,0.639344262295082,0.0,0.0,0.48286140089418783,0.13450042698548248,0.0]
    # print(len(point1))
    # point1 = correct_point(point1)
    # # point2 = correct_point(point2)
    # print("point1",point1) #,"\npoint2",point2)

    # x = xgb.DMatrix(sparse.csr_matrix(point1)) 
    # # x2 = xgb.DMatrix(sparse.csr_matrix(point2))
    
    # pred1 = model.predict(x)
    # # pred2 = model.predict(x2)
    
    # print(pred1) #,pred2)
    
    # scaler = compute_scaling_func()
    unscaledpoint1,originalpoint1 = retrace_original(scaler,point1)
    # originalpoint2 = retrace_original(scaler,point2)
    print(originalpoint1,"\n")
    # # print(originalpoint2)
