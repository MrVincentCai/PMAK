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

for i in range(5):

    data_train = pd.read_pickle(join("..", "..", "data", "kcat_data", "warm-splits", f"{i}_train_data.pkl"))
    data_test = pd.read_pickle(join("..", "..", "data", "kcat_data", "warm-splits", f"{i}_test_data.pkl"))
    data_train.head()

    data_train.rename(columns={"geomean_kcat": "log10_kcat"}, inplace=True)
    data_test.rename(columns={"geomean_kcat": "log10_kcat"}, inplace=True)
    len(data_train), len(data_test)

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

    train_ESM1b = np.array(list(data_train["ESM1b_ts"]))
    train_X = train_ESM1b
    train_Y = np.array(list(data_train["log10_kcat"]))
    test_ESM1b = np.array(list(data_test["ESM1b_ts"]))
    test_X = test_ESM1b
    test_Y = np.array(list(data_test["log10_kcat"]))

    dtrain = xgb.DMatrix(train_X, label=train_Y)
    bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)

    dtest = xgb.DMatrix(test_X)
    y_test_pred = bst.predict(dtest)
    MSE_dif_fp_test = np.mean(abs(np.reshape(test_Y, (-1)) - y_test_pred) ** 2)
    R2_dif_fp_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
    Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)

    print(np.round(Pearson[0], 3), np.round(MSE_dif_fp_test, 3), np.round(R2_dif_fp_test, 3))


    y_test_pred_esm1b_ts = y_test_pred


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

    train_X = np.array(list(data_train["DRFP"]))
    train_Y = np.array(list(data_train["log10_kcat"]))
    test_X = np.array(list(data_test["DRFP"]))
    test_Y = np.array(list(data_test["log10_kcat"]))

    dtrain = xgb.DMatrix(train_X, label=train_Y)
    bst = xgb.train(param, dtrain, int(num_round), verbose_eval=False)

    dtest = xgb.DMatrix(test_X)
    y_test_pred = bst.predict(dtest)

    MSE_dif_fp_test = np.mean(abs(np.reshape(test_Y, (-1)) - y_test_pred) ** 2)
    R2_dif_fp_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
    Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)

    print(np.round(Pearson[0], 3), np.round(MSE_dif_fp_test, 3), np.round(R2_dif_fp_test, 3))


    y_test_pred_drfp = y_test_pred

    y_test_pred = np.mean([y_test_pred_drfp, y_test_pred_esm1b_ts], axis=0)

    MSE_dif_fp_test = np.mean(abs(np.reshape(test_Y, (-1)) - y_test_pred) ** 2)
    R2_dif_fp_test = r2_score(np.reshape(test_Y, (-1)), y_test_pred)
    Pearson = stats.pearsonr(np.reshape(test_Y, (-1)), y_test_pred)

    print(np.round(Pearson[0], 3), np.round(MSE_dif_fp_test, 3), np.round(R2_dif_fp_test, 3))

    np.save(join("..", "..", "data", "training_results", f"Warm{i}_y_test_pred_xgboost_DRFP_ESM1bts_mean.npy"),
            y_test_pred)
    np.save(join("..", "..", "data", "training_results", f"Warm{i}_y_test_true_xgboost_DRFP_ESM1bts_mean.npy"), test_Y)

