import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import LinearRegression
import scipy
import os
from os.path import join
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

import xgboost as xgb

# 反应冷启动
with open(join("..", "..", "data", "kcat_data", "splits", "train_df_kcat_rx.pkl"), 'rb') as f:
    data_train = pickle.load(f)
with open(join("..", "..", "data", "kcat_data", "splits", "test_df_kcat_rx.pkl"), 'rb') as f:
    data_test = pickle.load(f)

data_train.rename(columns = {"geomean_kcat" :"log10_kcat"}, inplace = True)
data_test.rename(columns = {"geomean_kcat" :"log10_kcat"}, inplace = True)

train_indices = list(
    np.load('../data/kcat_data/splits/CV_train_indices_rx.npy', allow_pickle=True))
test_indices = list(np.load('../data/kcat_data/splits/CV_test_indices_rx.npy', allow_pickle=True))


# -----------------------------------------------------------Only ESM1b_ts---------------------------------------------------
print("-------------------Only ESM1b_ts start training--------------------")
train_ESM1b = np.array(list(data_train["ESM1b_ts"]))
train_X = train_ESM1b
train_Y = np.array(list(data_train["log10_kcat"]))

test_ESM1b = np.array(list(data_test["ESM1b_ts"]))
test_X = test_ESM1b
test_Y = np.array(list(data_test["log10_kcat"]))
print(train_X[0].shape)

param = {'learning_rate': 0.2831145406836757,
         'max_delta_step': 0.07686715986169101,
         'max_depth': 4.96836783761305,
          'min_child_weight': 6.905400087083855,
           'num_rounds': 313.1498988074061,
            'reg_alpha': 1.717314107718892,
             'reg_lambda': 2.470354543039016}

num_round = param["num_rounds"]
param["max_depth"] = int(np.round(param["max_depth"]))

del param["num_rounds"]

R2 = []
MSE = []
Pearson = []
y_valid_pred_esm1b_ts = []

for i in range(5):
    train_index, test_index = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(train_X[train_index], label=train_Y[train_index])
    dvalid = xgb.DMatrix(train_X[test_index])

    bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)

    y_valid_pred = bst.predict(dvalid)
    y_valid_pred_esm1b_ts.append(y_valid_pred)
    MSE.append(np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred) ** 2))
    R2.append(r2_score(np.reshape(train_Y[test_index], (-1)), y_valid_pred))
    Pearson.append(stats.pearsonr(np.reshape(train_Y[test_index], (-1)), y_valid_pred)[0])

print(Pearson)
print(MSE)
print(R2)

np.save(join("..", "..", "data", "training_results", "rx_Pearson_CV_xgboost_ESM1b_ts.npy"), np.array(Pearson))
np.save(join("..", "..", "data", "training_results", "rx_MSE_CV_xgboost_ESM1b_ts.npy"), np.array(MSE))
np.save(join("..", "..", "data", "training_results", "rx_R2_CV_xgboost_ESM1b_ts.npy"), np.array(R2))

dtrain = xgb.DMatrix(train_X, label = train_Y)
dtest = xgb.DMatrix(test_X)

bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)

y_test_pred = bst.predict(dtest)
MSE_dif_fp_test = np.mean(abs(np.reshape(test_Y, (-1)) - y_test_pred)**2)
R2_dif_fp_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)

print(np.round(Pearson[0],3) ,np.round(MSE_dif_fp_test,3), np.round(R2_dif_fp_test,3))

np.save(join("..", "..", "data", "training_results", "rx_y_test_pred_xgboost_ESM1b_ts.npy"), bst.predict(dtest))
np.save(join("..", "..", "data", "training_results", "rx_y_test_true_xgboost_ESM1b_ts.npy"), test_Y)

y_test_pred_esm1b_ts = y_test_pred

# [0.7147699897788168, 0.7184143385713685, 0.702318697390397, 0.6357003673019901, 0.7439460258892003]
# [0.6470072899071924, 0.6689176513897966, 0.7634310151145902, 0.8555898939901829, 0.6030071563112823]
# [0.5044709312528666, 0.511075435327202, 0.48919953395869475, 0.40044618293652345, 0.5531274873732697]
# 0.739 0.67 0.544


# -----------------------------------------------------------Only DRFP---------------------------------------------------
print("-------------------Only DRFP start training---------------------------")
train_X = np.array(list(data_train["DRFP"]))
train_Y = np.array(list(data_train["log10_kcat"]))

test_X = np.array(list(data_test["DRFP"]))
test_Y = np.array(list(data_test["log10_kcat"]))
print(train_X[0].shape)

# DRFP的最佳参数
param = {'learning_rate': 0.08987247189322463,
         'max_delta_step': 1.1939737318908727,
         'max_depth': 11.268531225242574,
         'min_child_weight': 2.8172720953826302,
         'num_rounds': 109.03643430746544,
         'reg_alpha': 1.9412226989868904,
         'reg_lambda': 4.950543905603358}


num_round = param["num_rounds"]
param["max_depth"] = int(np.round(param["max_depth"]))

del param["num_rounds"]

R2 = []
MSE = []
Pearson = []
y_valid_pred_DRFP = []

for i in range(5):
    train_index, test_index = train_indices[i], test_indices[i]
    dtrain = xgb.DMatrix(train_X[train_index], label=train_Y[train_index])
    dvalid = xgb.DMatrix(train_X[test_index])

    bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)

    y_valid_pred = bst.predict(dvalid)
    y_valid_pred_DRFP.append(y_valid_pred)
    MSE.append(np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred) ** 2))
    R2.append(r2_score(np.reshape(train_Y[test_index], (-1)), y_valid_pred))
    Pearson.append(stats.pearsonr(np.reshape(train_Y[test_index], (-1)), y_valid_pred)[0])

print(Pearson)
print(MSE)
print(R2)

np.save(join("..", "..", "data", "training_results", "rx_Pearson_CV_xgboost_DRFP.npy"), np.array(Pearson))
np.save(join("..", "..", "data", "training_results", "rx_MSE_CV_xgboost_DRFP.npy"), np.array(MSE))
np.save(join("..", "..", "data", "training_results", "rx_R2_CV_xgboost_DRFP.npy"), np.array(R2))

dtrain = xgb.DMatrix(train_X, label = train_Y)
dtest = xgb.DMatrix(test_X)

bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)

y_test_pred = bst.predict(dtest)
MSE_dif_fp_test = np.mean(abs(np.reshape(test_Y, (-1)) - y_test_pred)**2)
R2_dif_fp_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)

print(np.round(Pearson[0],3) ,np.round(MSE_dif_fp_test,3), np.round(R2_dif_fp_test,3))

np.save(join("..", "..", "data", "training_results", "rx_y_test_pred_xgboost_DRFP.npy"), bst.predict(dtest))
np.save(join("..", "..", "data", "training_results", "rx_y_test_true_xgboost_DRFP.npy"), test_Y)

y_test_pred_drfp = y_test_pred


# -----------------------------------------------------------ESM1b_ts & DRFP (MEAN)---------------------------------------------------
print("-------------------ESM1b_ts & DRFP (MEAN) start training--------------------------")
R2 = []
MSE = []
Pearson = []

for i in range(5):
    train_index, test_index  = train_indices[i], test_indices[i]
    y_valid_pred = np.mean([y_valid_pred_DRFP[i], y_valid_pred_esm1b_ts[i]], axis =0)
    MSE.append(np.mean(abs(np.reshape(train_Y[test_index], (-1)) - y_valid_pred)**2))
    R2.append(r2_score(np.reshape(train_Y[test_index], (-1)), y_valid_pred))
    Pearson.append(stats.pearsonr(np.reshape(train_Y[test_index], (-1)), y_valid_pred)[0])

print(Pearson)
print(MSE)
print(R2)

np.save(join("..", "..", "data", "training_results", "rx_Pearson_CV_xgboost_ESM1b_ts_DRFP_mean.npy"), np.array(Pearson))
np.save(join("..", "..", "data", "training_results", "rx_MSE_CV_xgboost_ESM1b_ts_DRFP_mean.npy"), np.array(MSE))
np.save(join("..", "..", "data", "training_results", "rx_R2_CV_xgboost_ESM1b_ts_DRFP_mean.npy"), np.array(R2))

y_test_pred = np.mean([y_test_pred_drfp, y_test_pred_esm1b_ts], axis =0)

MSE_dif_fp_test = np.mean(abs(np.reshape(test_Y, (-1)) - y_test_pred)**2)
R2_dif_fp_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)
print(np.round(Pearson[0],3) ,np.round(MSE_dif_fp_test,3), np.round(R2_dif_fp_test,3))

np.save(join("..", "..", "data", "training_results", "rx_y_test_pred_xgboost_ESM1b_ts_DRFP_mean.npy"), y_test_pred)
np.save(join("..", "..", "data", "training_results", "rx_y_test_true_xgboost_ESM1b_ts_DRFP_mean.npy"), test_Y)


