import pandas as pd
import numpy as np
from os.path import join
import os
import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.getcwd()


train_df = pd.read_pickle(join("..", "..", "data", "kcat_data", "splits", "train_df_kcat_rx.pkl"))
test_df = pd.read_pickle(join("..", "..", "data", "kcat_data", "splits", "test_df_kcat_rx.pkl"))
random_dataset = pd.read_pickle(join("..", "..", "data", "kcat_data", "random_final_kcat_dataset.pkl"))
# print(train_df.columns)

### Adding 'uniref50_dim1024'
uniref50_dim1024 = pd.read_pickle(join("..", "..", "data", "kcat_data", "Turnup_Protein_vector_dim1024.pkl"))
uniref50_dim1024_dataframe = pd.DataFrame({'Sequence ID': random_dataset['Sequence ID']})
uniref50_dim1024_dataframe['uniref50_dim1024'] = uniref50_dim1024
print(uniref50_dim1024_dataframe.columns)
train_df["uniref50_dim1024"] = [list(uniref50_dim1024_dataframe["uniref50_dim1024"].loc[uniref50_dim1024_dataframe["Sequence ID"] == R_ID])[0] for R_ID in train_df["Sequence ID"]]
test_df["uniref50_dim1024"] = [list(uniref50_dim1024_dataframe["uniref50_dim1024"].loc[uniref50_dim1024_dataframe["Sequence ID"] == R_ID])[0] for R_ID in test_df["Sequence ID"]]
# print(train_df.columns)

### Adding 'rxnfp'
rxnfp = pd.read_pickle(join("..", "..", "data", "kcat_data", "rxnfp.pkl"))
print(rxnfp.columns)
train_df["rxnfp"] = [list(rxnfp["rxnfp"].loc[rxnfp["Reaction ID"] == R_ID])[0] for R_ID in train_df["Reaction ID"]]
test_df["rxnfp"] = [list(rxnfp["rxnfp"].loc[rxnfp["Reaction ID"] == R_ID])[0] for R_ID in test_df["Reaction ID"]]


train_df.to_pickle(join("..", "..", "data", "kcat_data", "warm-splits", "train_data.pkl"))
test_df.to_pickle(join("..", "..", "data", "kcat_data", "warm-splits", "test_data.pkl"))
