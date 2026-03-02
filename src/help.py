import pandas as pd
df = pd.read_csv('./mnist3v8.csv')

col = df.columns.to_list()
    
    
non_zero_cols = [c.replace('f','') for c in col if df[c].nunique() > 1]
print("Columns with more than one unique value:", non_zero_cols)