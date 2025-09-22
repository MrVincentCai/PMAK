from os.path import join

import numpy as np
import pandas as pd

# -----------------------------------------------Reaction Cold Split------------------------------------------------------

def split_dataframe_reaction(frac, df):
    df1 = pd.DataFrame(columns = list(df.columns))
    df2 = pd.DataFrame(columns = list(df.columns))

    # n_training_samples = int(cutoff * len(df))

    df.reset_index(inplace = True, drop = True)

    # frac = int(1/(1- cutoff))

    train_indices = []
    test_indices = []
    ind = 0
    while len(train_indices) + len(test_indices) < len(df):
        if ind not in train_indices and ind not in test_indices:
            if ind % frac != 0:
                n_old = len(train_indices)
                train_indices.append(ind)
                train_indices = list(set(train_indices))

                while n_old != len(train_indices):
                    n_old = len(train_indices)

                    training_seqs = list(set(df["Reaction ID"].loc[train_indices]))

                    train_indices = train_indices + (list(df.loc[df["Reaction ID"].isin(training_seqs)].index))
                    train_indices = list(set(train_indices))

            else:
                n_old = len(test_indices)
                test_indices.append(ind)
                test_indices = list(set(test_indices))

                while n_old != len(test_indices):
                    n_old = len(test_indices)

                    testing_seqs= list(set(df["Reaction ID"].loc[test_indices]))

                    test_indices = test_indices + (list(df.loc[df["Reaction ID"].isin(testing_seqs)].index))
                    test_indices = list(set(test_indices))

        ind += 1


    df1 = df.loc[train_indices]
    df2 = df.loc[test_indices]

    return df1, df2


data = pd.read_pickle(join("..", "..", "data", "kcat_data", "random_final_kcat_dataset.pkl"))

enzyme_num = len(data["Sequence ID"].unique())  # 2800
reaction_num = len(data["Reaction ID"].unique())  # 2974

train_df, test_df = split_dataframe_reaction(frac = 5, df = data.copy())
print("Test set size: %s" % len(test_df))  # 826
print("Training set size: %s" % len(train_df)) # 3445
print("Size of test set in percent: %s" % np.round(100*len(test_df)/ (len(test_df) + len(train_df))))


train_df.reset_index(inplace = True, drop = True)
test_df.reset_index(inplace = True, drop = True)

train_df.to_pickle(join("..", "..", "data", "kcat_data", "splits", "train_df_kcat_rx.pkl"))
test_df.to_pickle(join("..", "..", "data", "kcat_data", "splits", "test_df_kcat_rx.pkl"))


data_train2 = train_df.copy()
data_train2["index"] = list(data_train2.index)

data_train2, df_fold = split_dataframe_reaction(df=data_train2, frac=5)
indices_fold1 = list(df_fold["index"])
print(len(data_train2), len(indices_fold1))  #  2697 748

data_train2, df_fold = split_dataframe_reaction(df=data_train2, frac=4)
indices_fold2 = list(df_fold["index"])
print(len(data_train2), len(indices_fold2))  # 2024 673

data_train2, df_fold = split_dataframe_reaction(df=data_train2, frac=3)
indices_fold3 = list(df_fold["index"])
print(len(data_train2), len(indices_fold3))  # 1343 681

data_train2, df_fold = split_dataframe_reaction(df=data_train2, frac=2)
indices_fold4 = list(df_fold["index"])
indices_fold5 = list(data_train2["index"])
print(len(data_train2), len(indices_fold4))  # 676 667

fold_indices = [indices_fold1, indices_fold2, indices_fold3, indices_fold4, indices_fold5]

train_indices = [[], [], [], [], []]
test_indices = [[], [], [], [], []]

for i in range(5):
    for j in range(5):
        if i != j:
            train_indices[i] = train_indices[i] + fold_indices[j]
    test_indices[i] = fold_indices[i]

np.save(join("..", "..", "data", "kcat_data", "splits", "CV_train_indices_rx"), train_indices)
np.save(join("..", "..", "data", "kcat_data", "splits", "CV_test_indices_rx"), test_indices)





# -----------------------------------------------Enzyme Cold Split------------------------------------------------------

def split_dataframe_enzyme(frac, df):
    df1 = pd.DataFrame(columns=list(df.columns))
    df2 = pd.DataFrame(columns=list(df.columns))

    # n_training_samples = int(cutoff * len(df))

    df.reset_index(inplace=True, drop=True)

    # frac = int(1/(1- cutoff))

    train_indices = []
    test_indices = []
    ind = 0
    while len(train_indices) + len(test_indices) < len(df):
        if ind not in train_indices and ind not in test_indices:
            if ind % frac != 0:
                n_old = len(train_indices)
                train_indices.append(ind)
                train_indices = list(set(train_indices))

                while n_old != len(train_indices):
                    n_old = len(train_indices)

                    training_seqs = list(set(df["Sequence"].loc[train_indices]))

                    train_indices = train_indices + (list(df.loc[df["Sequence"].isin(training_seqs)].index))
                    train_indices = list(set(train_indices))

            else:
                n_old = len(test_indices)
                test_indices.append(ind)
                test_indices = list(set(test_indices))

                while n_old != len(test_indices):
                    n_old = len(test_indices)

                    testing_seqs = list(set(df["Sequence"].loc[test_indices]))

                    test_indices = test_indices + (list(df.loc[df["Sequence"].isin(testing_seqs)].index))
                    test_indices = list(set(test_indices))

        ind += 1

    df1 = df.loc[train_indices]
    df2 = df.loc[test_indices]

    return df1, df2, train_indices, test_indices


train_df, test_df, train_indices, test_indices = split_dataframe_enzyme(frac = 5, df = data.copy())

print("len of train_indices:", len(train_indices))
print("len of test_indices:", len(test_indices))
print("Test set size: %s" % len(test_df))
print("Training set size: %s" % len(train_df))
print("Size of test set in percent: %s" % np.round(100*len(test_df)/ (len(test_df) + len(train_df))))


train_df.reset_index(inplace = True, drop = True)
test_df.reset_index(inplace = True, drop = True)

train_df.to_pickle(join("..", "..", "data", "kcat_data", "splits", "train_df_kcat.pkl"))
test_df.to_pickle(join("..", "..", "data", "kcat_data", "splits", "test_df_kcat.pkl"))

data_train2 = train_df.copy()
data_train2["index"] = list(data_train2.index)

data_train2, df_fold, _, _ = split_dataframe_enzyme(df=data_train2, frac=5)
indices_fold1 = list(df_fold["index"])
print(len(data_train2), len(indices_fold1))  #

data_train2, df_fold, _, _ = split_dataframe_enzyme(df=data_train2, frac=4)
indices_fold2 = list(df_fold["index"])
print(len(data_train2), len(indices_fold2))

data_train2, df_fold, _, _ = split_dataframe_enzyme(df=data_train2, frac=3)
indices_fold3 = list(df_fold["index"])
print(len(data_train2), len(indices_fold3))

data_train2, df_fold, _, _ = split_dataframe_enzyme(df=data_train2, frac=2)
indices_fold4 = list(df_fold["index"])
indices_fold5 = list(data_train2["index"])
print(len(data_train2), len(indices_fold4))

fold_indices = [indices_fold1, indices_fold2, indices_fold3, indices_fold4, indices_fold5]

train_indices = [[], [], [], [], []]
test_indices = [[], [], [], [], []]

for i in range(5):
    for j in range(5):
        if i != j:
            train_indices[i] = train_indices[i] + fold_indices[j]
    test_indices[i] = fold_indices[i]

np.save(join("..", "..", "data", "kcat_data", "splits", "CV_train_indices"), train_indices)
np.save(join("..", "..", "data", "kcat_data", "splits", "CV_test_indices"), test_indices)