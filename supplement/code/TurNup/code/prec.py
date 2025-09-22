import numpy as np
import pandas as pd
import os
from os.path import join
from sklearn.metrics import r2_score
from scipy import stats
import xgboost as xgb

# 数据和模型路径
data_dir = "/mnt/usb3/code/gfy/warm0-fasta_files"
model_save_dir = "/mnt/usb3/code/gfy/data/dulishujuji/model/2W/turnup"
result_save_dir = "/mnt/usb3/code/gfy/data/dulishujuji/results/2W/turnup"  # 修改结果保存路径，避免覆盖模型

# 创建结果保存目录（如果不存在）
os.makedirs(result_save_dir, exist_ok=True)

def load_data(data_dir,file):
    """加载测试数据"""
    print("正在加载测试数据...")
    data_test = pd.read_pickle(join(data_dir, file))
    # 重命名列
    data_test.rename(columns={"log10kcat_max": "log10_kcat"}, inplace=True)
    
    # 检查是否存在反应类型列，如果没有则创建一个虚拟列
    if 'reaction_smiles' not in data_test.columns:
        print("警告: 数据中未找到'reaction_smiles'列，将为所有样本分配相同的反应类型")
        data_test['reaction_smiles'] = 'reaction_1'
    
    return data_test

def load_model(model_path):
    """加载XGBoost模型"""
    print(f"正在加载模型: {model_path}")
    return xgb.Booster(model_file=model_path)

def calculate_scc_by_reaction(y_true, y_pred, reaction_smiless):
    """
    按反应类型计算SCC(Spearman相关系数)并返回平均值
    
    参数:
    y_true (array-like): 真实值
    y_pred (array-like): 预测值
    reaction_smiless (array-like): 反应类型
    
    返回:
    float: 所有反应类型的平均SCC
    dict: 每种反应类型的SCC
    """
    unique_reactions = np.unique(reaction_smiless)
    scc_values = {}
    valid_reactions = 0
    
    for reaction in unique_reactions:
        # 筛选当前反应类型的样本
        mask = reaction_smiless == reaction
        if np.sum(mask) < 3:  # 至少需要3个样本才能计算SCC
            print(f"警告: 反应类型 '{reaction}' 的样本数太少({np.sum(mask)})，无法计算SCC")
            continue
            
        y_true_reaction = y_true[mask]
        y_pred_reaction = y_pred[mask]
        
        # 计算Spearman相关系数
        scc, _ = stats.spearmanr(y_true_reaction, y_pred_reaction)
        scc_values[reaction] = scc
        valid_reactions += 1
    
    # 计算平均SCC
    if valid_reactions > 0:
        mean_scc = np.mean(list(scc_values.values()))
        print(f"计算了 {valid_reactions} 种反应类型的SCC，平均SCC: {mean_scc:.4f}")
        return mean_scc, scc_values
    else:
        print("警告: 没有足够的反应类型来计算SCC")
        return np.nan, {}

def predict_with_esm1b_model(data_test, model_path):
    """使用ESM-1b模型进行预测"""
    print("_______________________使用ESM-1b模型预测______________________")
    # 准备特征和目标变量
    test_ESM1b = np.array(list(data_test["ESM1b"]))
    test_X = test_ESM1b
    test_Y = np.array(list(data_test["log10_kcat"]))
    reaction_smiless = np.array(list(data_test["reaction_smiles"]))
    
    # 加载模型
    bst = load_model(model_path)
    
    # 进行预测
    dtest = xgb.DMatrix(test_X)
    y_pred = bst.predict(dtest)
    
    # 计算评估指标
    mse = np.mean(abs(np.reshape(test_Y, (-1)) - y_pred) ** 2)
    r2 = r2_score(np.reshape(test_Y, (-1)), y_pred)
    pearson, _ = stats.pearsonr(np.reshape(test_Y, (-1)), y_pred)
    mean_scc, scc_by_reaction = calculate_scc_by_reaction(test_Y, y_pred, reaction_smiless)
    
    print(f"ESM-1b模型预测结果 - Pearson相关系数: {pearson:.4f}")
    print(f"ESM-1b模型预测结果 - MSE: {mse:.4f}")
    print(f"ESM-1b模型预测结果 - R2: {r2:.4f}")
    print(f"ESM-1b模型预测结果 - 平均SCC: {mean_scc:.4f}")
    
    # 保存预测结果
    result_df = pd.DataFrame({
        "真实值": test_Y,
        "预测值": y_pred,
        "反应类型": reaction_smiless,
        "样本ID": range(len(test_Y))
    })
    result_path = join(result_save_dir, "xgb_prediction_esm1b.csv")
    result_df.to_csv(result_path, index=False)
    print(f"ESM-1b模型预测结果已保存至: {result_path}")
    
    # 保存SCC结果
    scc_df = pd.DataFrame(list(scc_by_reaction.items()), columns=['反应类型', 'SCC'])
    scc_path = join(result_save_dir, "xgb_scc_esm1b.csv")
    scc_df.to_csv(scc_path, index=False)
    print(f"ESM-1b模型SCC结果已保存至: {scc_path}")
    
    return y_pred, test_Y, reaction_smiless

def predict_with_drfp_model(data_test, model_path):
    """使用DRFP模型进行预测"""
    print("_______________________使用DRFP模型预测______________________")
    # 准备特征和目标变量
    test_drfp = np.array(list(data_test["drfp"]))
    test_X = test_drfp
    test_Y = np.array(list(data_test["log10_kcat"]))
    reaction_smiless = np.array(list(data_test["reaction_smiles"]))
    
    # 加载模型
    bst = load_model(model_path)
    
    # 进行预测
    dtest = xgb.DMatrix(test_X)
    y_pred = bst.predict(dtest)
    
    # 计算评估指标
    mse = np.mean(abs(np.reshape(test_Y, (-1)) - y_pred) ** 2)
    r2 = r2_score(np.reshape(test_Y, (-1)), y_pred)
    pearson, _ = stats.pearsonr(np.reshape(test_Y, (-1)), y_pred)
    mean_scc, scc_by_reaction = calculate_scc_by_reaction(test_Y, y_pred, reaction_smiless)
    
    print(f"DRFP模型预测结果 - Pearson相关系数: {pearson:.4f}")
    print(f"DRFP模型预测结果 - MSE: {mse:.4f}")
    print(f"DRFP模型预测结果 - R2: {r2:.4f}")
    print(f"DRFP模型预测结果 - 平均SCC: {mean_scc:.4f}")
    
    # 保存预测结果
    result_df = pd.DataFrame({
        "真实值": test_Y,
        "预测值": y_pred,
        "反应类型": reaction_smiless,
        "样本ID": range(len(test_Y))
    })
    result_path = join(result_save_dir, "xgb_prediction_drfp.csv")
    result_df.to_csv(result_path, index=False)
    print(f"DRFP模型预测结果已保存至: {result_path}")
    
    # 保存SCC结果
    scc_df = pd.DataFrame(list(scc_by_reaction.items()), columns=['反应类型', 'SCC'])
    scc_path = join(result_save_dir, "xgb_scc_drfp.csv")
    scc_df.to_csv(scc_path, index=False)
    print(f"DRFP模型SCC结果已保存至: {scc_path}")
    
    return y_pred, test_Y, reaction_smiless

def predict_with_combined_model(data_test, model_path):
    """使用组合模型(ESM1b+DRFP)进行预测"""
    print("_______________________使用组合模型预测______________________")
    # 准备特征和目标变量
    test_drfp = np.array(list(data_test["drfp"]))
    test_ESM1b = np.array(list(data_test["ESM1b"]))
    test_X = np.concatenate([test_drfp, test_ESM1b], axis=1)
    test_Y = np.array(list(data_test["log10_kcat"]))
    reaction_smiless = np.array(list(data_test["reaction_smiles"]))
    
    # 加载模型
    bst = load_model(model_path)
    
    # 进行预测
    dtest = xgb.DMatrix(test_X)
    y_pred = bst.predict(dtest)
    
    # 计算评估指标
    mse = np.mean(abs(np.reshape(test_Y, (-1)) - y_pred) ** 2)
    r2 = r2_score(np.reshape(test_Y, (-1)), y_pred)
    pearson, _ = stats.pearsonr(np.reshape(test_Y, (-1)), y_pred)
    mean_scc, scc_by_reaction = calculate_scc_by_reaction(test_Y, y_pred, reaction_smiless)
    
    print(f"组合模型预测结果 - Pearson相关系数: {pearson:.4f}")
    print(f"组合模型预测结果 - MSE: {mse:.4f}")
    print(f"组合模型预测结果 - R2: {r2:.4f}")
    print(f"组合模型预测结果 - 平均SCC: {mean_scc:.4f}")
    
    # 保存预测结果
    result_df = pd.DataFrame({
        "真实值": test_Y,
        "预测值": y_pred,
        "反应类型": reaction_smiless,
        "样本ID": range(len(test_Y))
    })
    result_path = join(result_save_dir, "xgb_prediction_combined.csv")
    result_df.to_csv(result_path, index=False)
    print(f"组合模型预测结果已保存至: {result_path}")
    
    # 保存SCC结果
    scc_df = pd.DataFrame(list(scc_by_reaction.items()), columns=['反应类型', 'SCC'])
    scc_path = join(result_save_dir, "xgb_scc_combined.csv")
    scc_df.to_csv(scc_path, index=False)
    print(f"组合模型SCC结果已保存至: {scc_path}")
    
    return y_pred, test_Y, reaction_smiless

def predict_with_ensemble_model(y_pred_esm1b, y_pred_drfp, test_Y, reaction_smiless):
    """使用集成模型(平均预测结果)进行预测"""
    print("_______________________使用集成模型预测______________________")
    # 对两个模型的预测结果取平均
    y_pred_ensemble = np.mean([y_pred_esm1b, y_pred_drfp], axis=0)
    
    # 计算评估指标
    mse = np.mean(abs(np.reshape(test_Y, (-1)) - y_pred_ensemble) ** 2)
    r2 = r2_score(np.reshape(test_Y, (-1)), y_pred_ensemble)
    pearson, _ = stats.pearsonr(np.reshape(test_Y, (-1)), y_pred_ensemble)
    mean_scc, scc_by_reaction = calculate_scc_by_reaction(test_Y, y_pred_ensemble, reaction_smiless)
    
    print(f"集成模型预测结果 - Pearson相关系数: {pearson:.4f}")
    print(f"集成模型预测结果 - MSE: {mse:.4f}")
    print(f"集成模型预测结果 - R2: {r2:.4f}")
    print(f"集成模型预测结果 - 平均SCC: {mean_scc:.4f}")
    
    # 保存预测结果
    result_df = pd.DataFrame({
        "真实值": test_Y,
        "ESM1b预测值": y_pred_esm1b,
        "DRFP预测值": y_pred_drfp,
        "集成预测值": y_pred_ensemble,
        "反应类型": reaction_smiless,
        "样本ID": range(len(test_Y))
    })
    result_path = join(result_save_dir, "xgb_prediction_HIS7_turnup.csv")
    result_df.to_csv(result_path, index=False)
    print(f"集成模型预测结果已保存至: {result_path}")
    
    # 保存SCC结果
    scc_df = pd.DataFrame(list(scc_by_reaction.items()), columns=['反应类型', 'SCC'])
    scc_path = join(result_save_dir, "xgb_scc_ensemble.csv")
    scc_df.to_csv(scc_path, index=False)
    print(f"集成模型SCC结果已保存至: {scc_path}")
    
    return y_pred_ensemble

def main():
    """主函数"""
    print("开始模型预测...")
    numbers = [1]
    model_save_dir = '.'  # 这里需要根据实际情况设置 model_save_dir 的值
    for i in numbers:
    
        data_dir = f"/mnt/usb/code/gfy/MyModel/"
        model_save_dir = f"/mnt/usb3/code/gfy/data/dulishujuji/model/2W/turnup"
        result_save_dir = f"/mnt/usb3/code/gfy/data/dulishujuji/model/2W/turnup"  # 修改结果保存路径，避免覆盖模型

        k = [0]
        for j in k:
            file = f"HIS7_esm_with_drfp.pkl"
            # 加载数据
            data_test = load_data(data_dir, file)

            # 模型路径
            model_path_esm1b = os.path.join(model_save_dir, "xgb_model_esm1b_ts.model")
            model_path_drfp = os.path.join(model_save_dir, "xgb_model_drfp.model")
            model_path_combined = os.path.join(model_save_dir, "xgb_model_esm1b_ts_drfp_combined.model")

            # 1. 使用ESM-1b模型预测
            y_pred_esm1b, test_Y, reaction_smiless = predict_with_esm1b_model(data_test, model_path_esm1b)

            # 2. 使用DRFP模型预测
            y_pred_drfp, _, _ = predict_with_drfp_model(data_test, model_path_drfp)

            # 3. 使用组合模型预测
            predict_with_combined_model(data_test, model_path_combined)

            # 4. 使用集成模型预测(平均ESM-1b和DRFP的预测结果)
            predict_with_ensemble_model(y_pred_esm1b, y_pred_drfp, test_Y, reaction_smiless)

        print("模型预测完成！")


if __name__ == "__main__":
    main()


#
#import numpy as np
#import pandas as pd
#import os
#from os.path import join
#from sklearn.metrics import r2_score
#from scipy import stats
#import xgboost as xgb
#
## 数据和模型路径
#data_dir = "/mnt/usb3/code/gfy/warm0-fasta_files"
#model_save_dir = "/mnt/usb3/code/gfy/data/dulishujuji/model/2W/turnup"
#result_save_dir = "/mnt/usb3/code/gfy/data/dulishujuji/results/2W/turnup"  # 修改结果保存路径，避免覆盖模型
#
## 创建结果保存目录（如果不存在）
#os.makedirs(result_save_dir, exist_ok=True)
#def load_data():
#    """加载测试数据"""
#    print("正在加载测试数据...")
#    file_path = os.path.join(data_dir, "catpred_test_esm1_with_drfp.pkl")
#    try:
#        # 打开并读取 .pkl 文件
#        data_test = pd.read_pickle(file_path)
#        # 检查数据是否为 DataFrame 类型
#        if isinstance(data_test, pd.DataFrame):
#            # 打印列名
#            print("文件的列名如下：")
#            for col in data_test.columns:
#                print(col)
#        else:
#            print("读取的数据并非 Pandas DataFrame 类型，无法打印列名。")
#    except FileNotFoundError:
#        print(f"未找到 '{file_path}' 文件，请检查文件路径。")
#        return None
#    except Exception as e:
#        print(f"读取文件时出现错误：{e}")
#        return None
#
#    # 重命名列
#    data_test.rename(columns={"value": "log10_kcat"}, inplace=True)
#
#    # 检查是否存在反应类型列，如果没有则创建一个虚拟列
#    if 'reaction_smiles' not in data_test.columns:
#        print("警告: 数据中未找到'reaction_smiles'列，将为所有样本分配相同的反应类型")
#        data_test['reaction_smiles'] = 'reaction_1'
#
#    return data_test
#
#
#def load_model(model_path):
#    """加载XGBoost模型"""
#    print(f"正在加载模型: {model_path}")
#    return xgb.Booster(model_file=model_path)
#
#def calculate_scc_by_reaction(y_true, y_pred, reaction_smiless):
#    """
#    按反应类型计算SCC(Spearman相关系数)并返回平均值
#    
#    参数:
#    y_true (array-like): 真实值
#    y_pred (array-like): 预测值
#    reaction_smiless (array-like): 反应类型
#    
#    返回:
#    float: 所有反应类型的平均SCC
#    dict: 每种反应类型的SCC
#    """
#    unique_reactions = np.unique(reaction_smiless)
#    scc_values = {}
#    valid_reactions = 0
#    
#    for reaction in unique_reactions:
#        # 筛选当前反应类型的样本
#        mask = reaction_smiless == reaction
#        sample_count = np.sum(mask)
#        
#        if sample_count < 3:  # 至少需要3个样本才能计算SCC
#            print(f"警告: 反应类型 '{reaction}' 的样本数太少({sample_count})，无法计算SCC")
#            continue
#            
#        y_true_reaction = y_true[mask]
#        y_pred_reaction = y_pred[mask]
#        
#        # 计算Spearman相关系数
#        scc, _ = stats.spearmanr(y_true_reaction, y_pred_reaction)
#        scc_values[reaction] = scc
#        valid_reactions += 1
#    
#    # 计算平均SCC
#    if valid_reactions > 0:
#        mean_scc = np.mean(list(scc_values.values()))
#        print(f"计算了 {valid_reactions} 种反应类型的SCC，平均SCC: {mean_scc:.4f}")
#        return mean_scc, scc_values
#    else:
#        print("警告: 没有足够的反应类型来计算SCC")
#        return np.nan, {}
#
#def predict_with_esm1b_model(data_test, model_path):
#    """使用ESM-1b模型进行预测"""
#    print("_______________________使用ESM-1b模型预测______________________")
#    # 准备特征和目标变量
#    test_ESM1b = np.array(list(data_test["ESM1b"]))
#    test_X = test_ESM1b
#    test_Y = np.array(list(data_test["log10_kcat"]))
#    reaction_smiless = np.array(list(data_test["reaction_smiles"]))
#    
#    # 加载模型
#    bst = load_model(model_path)
#    
#    # 进行预测
#    dtest = xgb.DMatrix(test_X)
#    y_pred = bst.predict(dtest)
#    
#    # 计算评估指标
#    mse = np.mean(abs(np.reshape(test_Y, (-1)) - y_pred) ** 2)
#    r2 = r2_score(np.reshape(test_Y, (-1)), y_pred)
#    pearson, _ = stats.pearsonr(np.reshape(test_Y, (-1)), y_pred)
#    mean_scc, scc_by_reaction = calculate_scc_by_reaction(test_Y, y_pred, reaction_smiless)
#    
#    print(f"ESM-1b模型预测结果 - Pearson相关系数: {pearson:.4f}")
#    print(f"ESM-1b模型预测结果 - MSE: {mse:.4f}")
#    print(f"ESM-1b模型预测结果 - R2: {r2:.4f}")
#    print(f"ESM-1b模型预测结果 - 平均SCC: {mean_scc:.4f}")
#    
#    # 保存预测结果
#    result_df = pd.DataFrame({
#        "真实值": test_Y,
#        "预测值": y_pred,
#        "反应类型": reaction_smiless,
#        "样本ID": range(len(test_Y))
#    })
#    result_path = join(result_save_dir, "xgb_prediction_esm1b.csv")
#    result_df.to_csv(result_path, index=False)
#    print(f"ESM-1b模型预测结果已保存至: {result_path}")
#    
#    # 保存SCC结果
#    scc_df = pd.DataFrame(list(scc_by_reaction.items()), columns=['反应类型', 'SCC'])
#    scc_path = join(result_save_dir, "xgb_scc_esm1b.csv")
#    scc_df.to_csv(scc_path, index=False)
#    print(f"ESM-1b模型SCC结果已保存至: {scc_path}")
#    
#    return y_pred, test_Y, reaction_smiless
#
#def predict_with_drfp_model(data_test, model_path):
#    """使用DRFP模型进行预测"""
#    print("_______________________使用DRFP模型预测______________________")
#    # 准备特征和目标变量
#    test_drfp = np.array(list(data_test["drfp"]))
#    test_X = test_drfp
#    test_Y = np.array(list(data_test["log10_kcat"]))
#    reaction_smiless = np.array(list(data_test["reaction_smiles"]))
#    
#    # 加载模型
#    bst = load_model(model_path)
#    
#    # 进行预测
#    dtest = xgb.DMatrix(test_X)
#    y_pred = bst.predict(dtest)
#    
#    # 计算评估指标
#    mse = np.mean(abs(np.reshape(test_Y, (-1)) - y_pred) ** 2)
#    r2 = r2_score(np.reshape(test_Y, (-1)), y_pred)
#    pearson, _ = stats.pearsonr(np.reshape(test_Y, (-1)), y_pred)
#    mean_scc, scc_by_reaction = calculate_scc_by_reaction(test_Y, y_pred, reaction_smiless)
#    
#    print(f"DRFP模型预测结果 - Pearson相关系数: {pearson:.4f}")
#    print(f"DRFP模型预测结果 - MSE: {mse:.4f}")
#    print(f"DRFP模型预测结果 - R2: {r2:.4f}")
#    print(f"DRFP模型预测结果 - 平均SCC: {mean_scc:.4f}")
#    
#    # 保存预测结果
#    result_df = pd.DataFrame({
#        "真实值": test_Y,
#        "预测值": y_pred,
#        "反应类型": reaction_smiless,
#        "样本ID": range(len(test_Y))
#    })
#    result_path = join(result_save_dir, "xgb_prediction_drfp.csv")
#    result_df.to_csv(result_path, index=False)
#    print(f"DRFP模型预测结果已保存至: {result_path}")
#    
#    # 保存SCC结果
#    scc_df = pd.DataFrame(list(scc_by_reaction.items()), columns=['反应类型', 'SCC'])
#    scc_path = join(result_save_dir, "xgb_scc_drfp.csv")
#    scc_df.to_csv(scc_path, index=False)
#    print(f"DRFP模型SCC结果已保存至: {scc_path}")
#    
#    return y_pred, test_Y, reaction_smiless
#
#def predict_with_combined_model(data_test, model_path):
#    """使用组合模型(ESM1b+DRFP)进行预测"""
#    print("_______________________使用组合模型预测______________________")
#    # 准备特征和目标变量
#    test_drfp = np.array(list(data_test["drfp"]))
#    test_ESM1b = np.array(list(data_test["ESM1b"]))
#    test_X = np.concatenate([test_drfp, test_ESM1b], axis=1)
#    test_Y = np.array(list(data_test["log10_kcat"]))
#    reaction_smiless = np.array(list(data_test["reaction_smiles"]))
#    
#    # 加载模型
#    bst = load_model(model_path)
#    
#    # 进行预测
#    dtest = xgb.DMatrix(test_X)
#    y_pred = bst.predict(dtest)
#    
#    # 计算评估指标
#    mse = np.mean(abs(np.reshape(test_Y, (-1)) - y_pred) ** 2)
#    r2 = r2_score(np.reshape(test_Y, (-1)), y_pred)
#    pearson, _ = stats.pearsonr(np.reshape(test_Y, (-1)), y_pred)
#    mean_scc, scc_by_reaction = calculate_scc_by_reaction(test_Y, y_pred, reaction_smiless)
#    
#    print(f"组合模型预测结果 - Pearson相关系数: {pearson:.4f}")
#    print(f"组合模型预测结果 - MSE: {mse:.4f}")
#    print(f"组合模型预测结果 - R2: {r2:.4f}")
#    print(f"组合模型预测结果 - 平均SCC: {mean_scc:.4f}")
#    
#    # 保存预测结果
#    result_df = pd.DataFrame({
#        "真实值": test_Y,
#        "预测值": y_pred,
#        "反应类型": reaction_smiless,
#        "样本ID": range(len(test_Y))
#    })
#    result_path = join(result_save_dir, "xgb_prediction_combined.csv")
#    result_df.to_csv(result_path, index=False)
#    print(f"组合模型预测结果已保存至: {result_path}")
#    
#    # 保存SCC结果
#    scc_df = pd.DataFrame(list(scc_by_reaction.items()), columns=['反应类型', 'SCC'])
#    scc_path = join(result_save_dir, "xgb_scc_combined.csv")
#    scc_df.to_csv(scc_path, index=False)
#    print(f"组合模型SCC结果已保存至: {scc_path}")
#    
#    return y_pred, test_Y, reaction_smiless
#
#def predict_with_ensemble_model(y_pred_esm1b, y_pred_drfp, test_Y, reaction_smiless):
#    """使用集成模型(平均预测结果)进行预测"""
#    print("_______________________使用集成模型预测______________________")
#    # 对两个模型的预测结果取平均
#    y_pred_ensemble = np.mean([y_pred_esm1b, y_pred_drfp], axis=0)
#    
#    # 计算评估指标
#    mse = np.mean(abs(np.reshape(test_Y, (-1)) - y_pred_ensemble) ** 2)
#    r2 = r2_score(np.reshape(test_Y, (-1)), y_pred_ensemble)
#    pearson, _ = stats.pearsonr(np.reshape(test_Y, (-1)), y_pred_ensemble)
#    mean_scc, scc_by_reaction = calculate_scc_by_reaction(test_Y, y_pred_ensemble, reaction_smiless)
#    
#    print(f"集成模型预测结果 - Pearson相关系数: {pearson:.4f}")
#    print(f"集成模型预测结果 - MSE: {mse:.4f}")
#    print(f"集成模型预测结果 - R2: {r2:.4f}")
#    print(f"集成模型预测结果 - 平均SCC: {mean_scc:.4f}")
#    
#    # 保存预测结果
#    result_df = pd.DataFrame({
#        "真实值": test_Y,
#        "ESM1b预测值": y_pred_esm1b,
#        "DRFP预测值": y_pred_drfp,
#        "集成预测值": y_pred_ensemble,
#        "反应类型": reaction_smiless,
#        "样本ID": range(len(test_Y))
#    })
#    result_path = join(result_save_dir, "xgb_prediction_ensemble.csv")
#    result_df.to_csv(result_path, index=False)
#    print(f"集成模型预测结果已保存至: {result_path}")
#    
#    # 保存SCC结果
#    scc_df = pd.DataFrame(list(scc_by_reaction.items()), columns=['反应类型', 'SCC'])
#    scc_path = join(result_save_dir, "xgb_scc_ensemble.csv")
#    scc_df.to_csv(scc_path, index=False)
#    print(f"集成模型SCC结果已保存至: {scc_path}")
#    
#    return y_pred_ensemble
#
#def main():
#    """主函数"""
#    print("开始模型预测...")
#    
#    # 加载数据
#    data_test = load_data()
#    
#    # 模型路径
#    model_path_esm1b = join(model_save_dir, "xgb_model_esm1b_ts.model")
#    model_path_drfp = join(model_save_dir, "xgb_model_drfp.model")
#    model_path_combined = join(model_save_dir, "xgb_model_esm1b_ts_drfp_combined.model")
#    
#    # 1. 使用ESM-1b模型预测
#    y_pred_esm1b, test_Y, reaction_smiless = predict_with_esm1b_model(data_test, model_path_esm1b)
#    
#    # 2. 使用DRFP模型预测
#    y_pred_drfp, _, _ = predict_with_drfp_model(data_test, model_path_drfp)
#    
#    # 3. 使用组合模型预测
#    predict_with_combined_model(data_test, model_path_combined)
#    
#    # 4. 使用集成模型预测(平均ESM-1b和DRFP的预测结果)
#    predict_with_ensemble_model(y_pred_esm1b, y_pred_drfp, test_Y, reaction_smiless)
#    
#    print("模型预测完成！")
#
#if __name__ == "__main__":
#    main()