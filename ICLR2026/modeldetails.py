# modelFeature = {
#     'adult' : 15 ,
#     'churn' : 21 ,
#     # 'breast_cancer': 31,
#     'pimadiabetes': 9,
#     'spambase':58,
#     'winequality_red':11,
#     'iris':4,
#     'german_credit':20
# }
#@@ This is the formate of the test model detail to be added in MODELDETAILS
# "<modelname>":{ "model" : "<pathofmodel>",
#         "details" : "<detailfilepath>",
#         "data" : "<trainingdatafilepath>",
#         "test":'<testdatafilepath>',
#         "feature":<numberoffeat>,
#         "clause":'<pathofclauselearned>',
#         }

MODELDETAILS = {
    'brcR': 
        { "model" : "models/tree_verification_models/breast_cancer_robust/0004.resaved.json",
        "details" : "models/dataset/breast_cancer/breast_cancer_details.csv",
        "data" : "models/dataset/breast_cancer/breast_cancer_train.csv",
        "test":'models/dataset/breast_cancer/breast_cancer_test.csv',
        "feature":10,
        "clause":'outputs/output/learned-clauses_breast_cancer_robust.txt',
        },
    "adult":{ "model" : "models/adult/adult_t300_d6.json",
        "details" : "models/dataset/adult/details.csv",
        "data" : "models/dataset/adult/train.csv",
        "test":'models/dataset/adult/test.csv',
        "feature":15,
        "clause":'outputs/outputs_old/learned-clauses_adult_t300_d6.txt',
        }
    
}

GAP_LB = [0.1,0.2,0.3,0.4]
GAP_UB = [0.6,0.7,0.8,0.9]

