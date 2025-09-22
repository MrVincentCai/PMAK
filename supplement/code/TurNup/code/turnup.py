import numpy as np
import pickle
import pandas as pd
import os
from os.path import join
import warnings
import torch
import esm
from drfp import DrfpEncoder
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy import stats
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib as mpl

# 忽略警告
warnings.filterwarnings("ignore")

# -------------------------- 全局配置 --------------------------
# 数据根路径（根据实际情况修改）
DATA_ROOT = "PMAK_"
# 特征与模型保存根路径
FEATURE_ROOT = join(DATA_ROOT, "features")
MODEL_ROOT = join(DATA_ROOT, "models")
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ESM模型配置
ESM_MODEL_NAME = "esm1b_t33_650M_UR50S"
MAX_SEQ_LENGTH = 1022  # ESM模型最大序列长度
BATCH_SIZE = 16  # ESM特征提取批次大小（根据GPU显存调整）
# 目标列名
TARGET_COL = "log10_kcat"
SOURCE_COL = "geomean_kcat"  # 原始数据中的目标列名

# 创建必要目录
os.makedirs(FEATURE_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)


# -------------------------- 1. CSV转PKL工具函数 --------------------------
def csv_to_pkl(csv_path, pkl_path):
    """将CSV文件转换为PKL格式并保存"""
    if not os.path.exists(pkl_path):
        df = pd.read_csv(csv_path)
        # 重命名目标列（如果需要）
        if SOURCE_COL in df.columns and TARGET_COL not in df.columns:
            df.rename(columns={SOURCE_COL: TARGET_COL}, inplace=True)
        df.to_pickle(pkl_path)
        print(f"✅ 已将CSV转换为PKL: {csv_path} → {pkl_path}")
    else:
        print(f"ℹ️ PKL文件已存在，跳过转换: {pkl_path}")
    return pd.read_pickle(pkl_path)


# -------------------------- 2. ESM特征提取工具函数 --------------------------
def load_esm_model():
    """加载ESM模型"""
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
    model = model.eval().to(DEVICE)
    return model, alphabet


def extract_esm_features(df, seq_col="enzyme_sequence"):
    """为DataFrame中的序列提取ESM特征"""
    # 检查序列列是否存在
    if seq_col not in df.columns:
        raise ValueError(f"数据中缺少序列列: {seq_col}")
    
    # 加载模型
    model, alphabet = load_esm_model()
    batch_converter = alphabet.get_batch_converter()
    
    # 处理过长序列
    df[seq_col] = df[seq_col].apply(
        lambda x: x[:MAX_SEQ_LENGTH] if isinstance(x, str) and len(x) > MAX_SEQ_LENGTH else x
    )
    
    # 提取特征
    sequences = df[seq_col].tolist()
    features = []
    
    for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="提取ESM特征"):
        batch = sequences[i:i+BATCH_SIZE]
        # 准备批次数据
        batch_labels = [f"seq_{j}" for j in range(i, min(i+BATCH_SIZE, len(sequences)))]
        batch_data = list(zip(batch_labels, batch))
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(DEVICE)
        
        # 模型推理
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_reps = results["representations"][33]  # (batch_size, seq_len, 1280)
        
        # 计算序列级特征（平均池化）
        for rep in token_reps:
            # 排除起始和终止token
            seq_rep = rep[1:-1].mean(dim=0).cpu().numpy()
            features.append(seq_rep)
    
    return np.array(features)


# -------------------------- 3. DRFP特征提取工具函数 --------------------------
def extract_drfp_features(df, smiles_col="reaction_smiles"):
    """为DataFrame中的SMILES提取DRFP特征"""
    if smiles_col not in df.columns:
        raise ValueError(f"数据中缺少SMILES列: {smiles_col}")
    
    smiles_list = df[smiles_col].fillna("").tolist()
    # 生成DRFP指纹（2048维）
    fps, _ = DrfpEncoder.encode(smiles_list, nBits=2048)
    return fps


# -------------------------- 4. 数据预处理完整流程 --------------------------
def preprocess_data(csv_train_path, csv_test_path, dataset_name, fold=None):
    """
    完整数据预处理流程：CSV→PKL→ESM特征→DRFP特征
    返回带特征的训练集和测试集PKL路径
    """
    # 1. 定义路径
    fold_suffix = f"_fold_{fold}" if fold is not None else ""
    pkl_train_path = join(FEATURE_ROOT, dataset_name, f"train{fold_suffix}.pkl")
    pkl_test_path = join(FEATURE_ROOT, dataset_name, f"test{fold_suffix}.pkl")
    final_train_path = join(FEATURE_ROOT, dataset_name, f"train{fold_suffix}_with_features.pkl")
    final_test_path = join(FEATURE_ROOT, dataset_name, f"test{fold_suffix}_with_features.pkl")
    
    # 创建数据集目录
    os.makedirs(join(FEATURE_ROOT, dataset_name), exist_ok=True)
    
    # 2. CSV转PKL
    df_train = csv_to_pkl(csv_train_path, pkl_train_path)
    df_test = csv_to_pkl(csv_test_path, pkl_test_path)
    
    # 3. 提取并添加特征（如果尚未处理）
    if not os.path.exists(final_train_path) or not os.path.exists(final_test_path):
        print(f"ℹ️ 开始为{dataset_name}{fold_suffix}提取特征...")
        
        # 提取ESM特征
        print("提取训练集ESM特征...")
        esm_train = extract_esm_features(df_train)
        print("提取测试集ESM特征...")
        esm_test = extract_esm_features(df_test)
        
        # 提取DRFP特征
        print("提取训练集DRFP特征...")
        drfp_train = extract_drfp_features(df_train)
        print("提取测试集DRFP特征...")
        drfp_test = extract_drfp_features(df_test)
        
        # 添加特征到DataFrame
        df_train["ESM1b"] = esm_train.tolist()
        df_test["ESM1b"] = esm_test.tolist()
        df_train["drfp"] = drfp_train.tolist()
        df_test["drfp"] = drfp_test.tolist()
        
        # 保存最终带特征的PKL
        df_train.to_pickle(final_train_path)
        df_test.to_pickle(final_test_path)
        print(f"✅ 特征提取完成，保存至: {final_train_path} 和 {final_test_path}")
    else:
        print(f"ℹ️ 特征已存在，直接加载: {final_train_path} 和 {final_test_path}")
    
    return final_train_path, final_test_path


# -------------------------- 5. 模型训练函数 --------------------------
def train_models(train_path, test_path, dataset_name, fold=None):
    """使用带特征的PKL数据训练4种XGBoost模型"""
    # 1. 加载数据
    data_train = pd.read_pickle(train_path)
    data_test = pd.read_pickle(test_path)
    
    # 确保目标列存在
    if TARGET_COL not in data_train.columns or TARGET_COL not in data_test.columns:
        raise ValueError(f"数据中缺少目标列: {TARGET_COL}")
    
    # 2. 准备模型保存目录
    fold_suffix = f"_fold_{fold}" if fold is not None else ""
    model_dir = join(MODEL_ROOT, dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"📂 模型将保存至: {model_dir}")
    
    # 3. 提取特征和目标值
    # 训练集
    train_ESM1b = np.array(list(data_train["ESM1b"]))
    train_drfp = np.array(list(data_train["drfp"]))
    train_Y = np.array(list(data_train[TARGET_COL].dropna()))
    # 测试集
    test_ESM1b = np.array(list(data_test["ESM1b"]))
    test_drfp = np.array(list(data_test["drfp"]))
    test_Y = np.array(list(data_test[TARGET_COL].dropna()))
    
    # 过滤空值
    valid_train = ~np.isnan(train_Y)
    valid_test = ~np.isnan(test_Y)
    train_ESM1b = train_ESM1b[valid_train]
    train_drfp = train_drfp[valid_train]
    train_Y = train_Y[valid_train]
    test_ESM1b = test_ESM1b[valid_test]
    test_drfp = test_drfp[valid_test]
    test_Y = test_Y[valid_test]


    # -------------------------- 模型1: ESM1b特征 --------------------------
    print("\n----- 训练ESM1b特征模型 -----")
    param = {
        'learning_rate': 0.2831145406836757,
        'max_delta_step': 0.07686715986169101,
        'max_depth': int(np.round(4.96836783761305)),
        'min_child_weight': 6.905400087083855,
        'reg_alpha': 1.717314107718892,
        'reg_lambda': 2.470354543039016,
        'objective': 'reg:squarederror'
    }
    num_round = 313  # 取整处理
    
    dtrain = xgb.DMatrix(train_ESM1b, label=train_Y)
    dtest = xgb.DMatrix(test_ESM1b, label=test_Y)
    
    bst = xgb.train(param, dtrain, num_round, verbose_eval=False)
    esm_model_path = join(model_dir, f"xgb_esm1b{fold_suffix}.model")
    bst.save_model(esm_model_path)
    
    # 评估
    y_pred_esm = bst.predict(dtest)
    mse_esm = np.mean(np.square(test_Y - y_pred_esm))
    r2_esm = r2_score(test_Y, y_pred_esm)
    pearson_esm = stats.pearsonr(test_Y, y_pred_esm)[0]
    
    print(f"ESM1b模型保存至: {esm_model_path}")
    print(f"性能: Pearson={pearson_esm:.4f}, MSE={mse_esm:.4f}, R²={r2_esm:.4f}")


    # -------------------------- 模型2: DRFP特征 --------------------------
    print("\n----- 训练DRFP特征模型 -----")
    param = {
        'learning_rate': 0.08987247189322463,
        'max_delta_step': 1.1939737318908727,
        'max_depth': int(np.round(11.268531225242574)),
        'min_child_weight': 2.8172720953826302,
        'reg_alpha': 1.9412226989868904,
        'reg_lambda': 4.950543905603358,
        'objective': 'reg:squarederror'
    }
    num_round = 109  # 取整处理
    
    dtrain = xgb.DMatrix(train_drfp, label=train_Y)
    dtest = xgb.DMatrix(test_drfp, label=test_Y)
    
    bst = xgb.train(param, dtrain, num_round, verbose_eval=False)
    drfp_model_path = join(model_dir, f"xgb_drfp{fold_suffix}.model")
    bst.save_model(drfp_model_path)
    
    # 评估
    y_pred_drfp = bst.predict(dtest)
    mse_drfp = np.mean(np.square(test_Y - y_pred_drfp))
    r2_drfp = r2_score(test_Y, y_pred_drfp)
    pearson_drfp = stats.pearsonr(test_Y, y_pred_drfp)[0]
    
    print(f"DRFP模型保存至: {drfp_model_path}")
    print(f"性能: Pearson={pearson_drfp:.4f}, MSE={mse_drfp:.4f}, R²={r2_drfp:.4f}")


    # -------------------------- 模型3: ESM1b+DRFP组合特征 --------------------------
    print("\n----- 训练组合特征模型 -----")
    # 拼接特征
    train_combined = np.concatenate([train_ESM1b, train_drfp], axis=1)
    test_combined = np.concatenate([test_ESM1b, test_drfp], axis=1)
    
    param = {
        'learning_rate': 0.05221672412884108,
        'max_delta_step': 1.0767235463496743,
        'max_depth': int(np.round(11.329014411591299)),
        'min_child_weight': 14.724796449973605,
        'reg_alpha': 2.8295816318634452,
        'reg_lambda': 0.6528469146574993,
        'objective': 'reg:squarederror'
    }
    num_round = 299  # 取整处理
    
    dtrain = xgb.DMatrix(train_combined, label=train_Y)
    dtest = xgb.DMatrix(test_combined, label=test_Y)
    
    bst = xgb.train(param, dtrain, num_round, verbose_eval=False)
    combined_model_path = join(model_dir, f"xgb_combined{fold_suffix}.model")
    bst.save_model(combined_model_path)
    
    # 评估
    y_pred_combined = bst.predict(dtest)
    mse_combined = np.mean(np.square(test_Y - y_pred_combined))
    r2_combined = r2_score(test_Y, y_pred_combined)
    pearson_combined = stats.pearsonr(test_Y, y_pred_combined)[0]
    
    print(f"组合模型保存至: {combined_model_path}")
    print(f"性能: Pearson={pearson_combined:.4f}, MSE={mse_combined:.4f}, R²={r2_combined:.4f}")


    # -------------------------- 模型4: 均值融合 --------------------------
    print("\n----- 计算均值融合结果 -----")
    y_pred_mean = (y_pred_esm + y_pred_drfp) / 2
    
    # 评估
    mse_mean = np.mean(np.square(test_Y - y_pred_mean))
    r2_mean = r2_score(test_Y, y_pred_mean)
    pearson_mean = stats.pearsonr(test_Y, y_pred_mean)[0]
    
    # 保存融合结果
    mean_result_path = join(model_dir, f"mean_fusion{fold_suffix}.pkl")
    with open(mean_result_path, "wb") as f:
        pickle.dump({
            "y_true": test_Y,
            "y_pred_esm": y_pred_esm,
            "y_pred_drfp": y_pred_drfp,
            "y_pred_mean": y_pred_mean
        }, f)
    
    print(f"均值融合结果保存至: {mean_result_path}")
    print(f"性能: Pearson={pearson_mean:.4f}, MSE={mse_mean:.4f}, R²={r2_mean:.4f}")


# -------------------------- 6. 数据集训练流程 --------------------------
def train_catpred():
    """训练CatPred基础数据集"""
    print("\n" + "="*50)
    print("开始训练 CatPred 基础数据集")
    print("="*50)
    
    # 数据路径
    train_csv = join(DATA_ROOT, "data", "catpred", "train")
    test_csv = join(DATA_ROOT, "data", "catpred", "test")
    
    # 预处理数据
    train_path, test_path = preprocess_data(
        train_csv, test_csv, 
        dataset_name="catpred"
    )
    
    # 训练模型
    train_models(train_path, test_path, dataset_name="catpred")


def train_cold_enzyme():
    """训练Cold-Enzyme数据集（1-5折）"""
    print("\n" + "="*50)
    print("开始训练 Cold-Enzyme 数据集（1-5折）")
    print("="*50)
    
    for fold in range(1, 6):  # 1-5折
        print(f"\n" + "-"*40)
        print(f"处理 Cold-Enzyme 第 {fold} 折")
        print("-"*40)
        
        # 数据路径
        train_csv = join(DATA_ROOT, "data", "turnup", "cold_enzyme", 
                        f"kcat_train_fold_{fold}_en.csv")
        test_csv = join(DATA_ROOT, "data", "turnup", "cold_enzyme", 
                       f"kcat_val_fold_{fold}_en.csv")
        
        # 检查文件是否存在
        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"❌ 第 {fold} 折文件缺失，跳过")
            continue
        
        # 预处理数据
        train_path, test_path = preprocess_data(
            train_csv, test_csv, 
            dataset_name="cold_enzyme",
            fold=fold
        )
        
        # 训练模型
        train_models(train_path, test_path, dataset_name="cold_enzyme", fold=fold)


def train_cold_reaction():
    """训练Cold-Reaction数据集（1-5折）"""
    print("\n" + "="*50)
    print("开始训练 Cold-Reaction 数据集（1-5折）")
    print("="*50)
    
    for fold in range(1, 6):  # 1-5折
        print(f"\n" + "-"*40)
        print(f"处理 Cold-Reaction 第 {fold} 折")
        print("-"*40)
        
        # 数据路径
        train_csv = join(DATA_ROOT, "data", "turnup", "cold_reaction", 
                        f"kcat_train_fold_{fold}.csv")
        test_csv = join(DATA_ROOT, "data", "turnup", "cold_reaction", 
                       f"kcat_val_fold_{fold}.csv")
        
        # 检查文件是否存在
        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"❌ 第 {fold} 折文件缺失，跳过")
            continue
        
        # 预处理数据
        train_path, test_path = preprocess_data(
            train_csv, test_csv, 
            dataset_name="cold_reaction",
            fold=fold
        )
        
        # 训练模型
        train_models(train_path, test_path, dataset_name="cold_reaction", fold=fold)


def train_warm():
    """训练Warm数据集（0-4折）"""
    print("\n" + "="*50)
    print("开始训练 Warm 数据集（0-4折）")
    print("="*50)
    
    for fold in range(0, 5):  # 0-4折
        print(f"\n" + "-"*40)
        print(f"处理 Warm 第 {fold} 折")
        print("-"*40)
        
        # 数据路径
        train_csv = join(DATA_ROOT, "data", "turnup", "warm", 
                        f"kcat_train_data_{fold}.csv")
        test_csv = join(DATA_ROOT, "data", "turnup", "warm", 
                       f"kcat_test_data_{fold}.csv")
        
        # 检查文件是否存在
        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"❌ 第 {fold} 折文件缺失，跳过")
            continue
        
        # 预处理数据
        train_path, test_path = preprocess_data(
            train_csv, test_csv, 
            dataset_name="warm",
            fold=fold
        )
        
        # 训练模型
        train_models(train_path, test_path, dataset_name="warm", fold=fold)


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    train_catpred()          # CatPred基础数据集
    train_cold_enzyme()      # Cold-Enzyme（1-5折）
    train_cold_reaction()    # Cold-Reaction（1-5折）
    train_warm()             # Warm（0-4折）
    
    print("\n" + "="*50)
    print("🎉 所有数据集训练完成！")
    print(f"模型保存根目录: {MODEL_ROOT}")
    print(f"特征保存根目录: {FEATURE_ROOT}")
    print("="*50)
