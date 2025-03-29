import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt

from Extract_feature import Get_features

data = pd.read_pickle(join("..", "..", "data", "kcat_data", "random_final_kcat_dataset.pkl"))

enzyme_num = len(data["Sequence ID"].unique())  # 2800
reaction_num = len(data["Reaction ID"].unique())  # 2974
print("酶的数量：", enzyme_num)
print("反应的数量：", reaction_num)

# 获得features_crafted
data_train = pd.read_pickle(join("..", "..", "data", "kcat_data", "splits", "train_df_kcat.pkl"))
# data_train = data_train.head(1)
data_test = pd.read_pickle(join("..", "..", "data", "kcat_data", "splits", "test_df_kcat.pkl"))
sequence_all = data["Sequence"].tolist()
sequence_train = data_train["Sequence"].tolist()
sequence_test = data_test["Sequence"].tolist()
print(len(sequence_all))
print(len(sequence_train))
print(len(sequence_test))
shortest_seq = min(sequence_all, key=len)
longest_seq = max(sequence_all, key=len)
# 获取最短字符串的长度
shortest_length = len(shortest_seq)  # 12
longest_length = len(longest_seq)  # 2324
print(shortest_length)
print(longest_length)


features_crafted_train = Get_features(sequence_train, shortest_length)
features_crafted_test = Get_features(sequence_test, shortest_length)

np.save('features_crafted_train', features_crafted_train)
np.save('features_crafted_test', features_crafted_test)

print("features_crafted_train shape:", features_crafted_train.shape)  # (3421, 1944)
print("features_crafted_test shape:", features_crafted_test.shape)  # (850, 1944)

