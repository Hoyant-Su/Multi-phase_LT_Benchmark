import pandas as pd
import numpy as np
from sklearn import preprocessing

# Meta info header columns
n_header_cols = 38

def radiomics_data_parse(*csv_files):
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df.drop(columns=df.columns[:n_header_cols], inplace=True)
        suffix = csv_file.split('features')[-1].split('.csv')[0]
        print(suffix)
        df = df.add_suffix(suffix)
        dfs.append(df)
    
    df_features_merge = pd.concat(dfs, axis=1)
    return df_features_merge
