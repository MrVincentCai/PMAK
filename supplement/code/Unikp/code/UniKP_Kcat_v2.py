import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import json
from transformers import T5EncoderModel, T5Tokenizer
import re
import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
from scipy import stats
import multiprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 配置参数
ncpu = multiprocessing.cpu_count()
DATA_ROOT = "PMAK_/data"
FEATURE_CACHE_ROOT = "PMAK_/feature_cache"
MODEL_SAVE_ROOT = "PMAK_/saved_models"
TARGET_COL = "log10_kcat"  # 目标列名
SOURCE_COL = "log10kcat_max"  # 原始数据中的目标列名（如需转换）

# 创建必要目录
os.makedirs(FEATURE_CACHE_ROOT, exist_ok=True)
os.makedirs(MODEL_SAVE_ROOT, exist_ok=True)

# 禁用SSL验证（根据需要保留）
import requests
from huggingface_hub import configure_http_backend
session = requests.Session()
session.verify = False
configure_http_backend(backend_factory=lambda: session)
requests.packages.urllib3.disable_warnings()


# -------------------------- 特征提取函数 --------------------------
def smiles_to_vec(Smiles):
    """使用UniKP模型将SMILES转换为向量"""
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    
    # 加载词汇表
    vocab = WordVocab.load_vocab('./external/UniKP/vocab.pkl')
    
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm) > 218:
            print(f'SMILES过长，截断处理: {len(sm)} → 218')
            sm = sm[:109] + sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        seg.extend(padding)
        return ids, seg
    
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    
    # 加载模型
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('./external/UniKP/trfm_12_23000.pkl'))
    trfm.eval()
    
    # 处理输入
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    
    # 生成特征
    with torch.no_grad():
        X = trfm.encode(torch.t(xid))
    
    return X.numpy()


def Seq_to_vec(Sequence):
    """使用ProtT5模型将蛋白质序列转换为向量"""
    # 处理长序列
    seq_series = Sequence.copy()
    for i in range(len(seq_series)):
        if len(seq_series[i]) > 1000:
            seq_series[i] = seq_series[i][:500] + seq_series[i][-500:]  # 截断为1000
    
    # 序列预处理（添加空格分隔）
    sequences_Example = []
    for seq in seq_series:
        if isinstance(seq, str):
            zj = ' '.join(list(seq))  # 每个氨基酸间加空格
            sequences_Example.append(zj)
        else:
            sequences_Example.append('')  # 处理空值
    
    # 加载模型和分词器
    tokenizer = T5Tokenizer.from_pretrained(
        "/mnt/usb/code/gfy/wangye/prot_t5_xl_uniref50", 
        do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained(
        "/mnt/usb/code/gfy/wangye/prot_t5_xl_uniref50"
    )
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    
    # 提取特征
    features = []
    for seq in tqdm(sequences_Example, desc="提取蛋白质特征"):
        # 替换稀有氨基酸
        seq_clean = re.sub(r"[UZOB]", "X", seq)
        # 编码
        ids = tokenizer.batch_encode_plus([seq_clean], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        
        # 生成嵌入
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 处理嵌入（取平均）
        embedding = embedding.last_hidden_state.cpu().numpy()
        seq_len = (attention_mask[0] == 1).sum() - 2  # 排除特殊标记
        if seq_len > 0:
            seq_emd = embedding[0][1:seq_len+1].mean(axis=0)  # 平均池化
        else:
            seq_emd = np.zeros(embedding.shape[-1])  # 空序列用零向量
        
        features.append(seq_emd)
    
    # 清理内存
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return np.array(features)


# -------------------------- 模型训练与评估函数 --------------------------
def train_evaluate(feature_train, label_train, feature_test, label_test, model_save_dir, n_runs=5):
    """训练并评估模型，多次运行取平均"""
    results_list = []
    os.makedirs(model_save_dir, exist_ok=True)
    
    for run in range(n_runs):
        # 训练模型
        model = ExtraTreesRegressor(n_estimators=1000, n_jobs=ncpu, random_state=run)
        model.fit(feature_train, label_train)
        
        # 保存模型
        model_path = os.path.join(model_save_dir, f"extra_trees_run_{run}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 预测与评估
        pred = model.predict(feature_test)
        
        # 计算指标
        results = {
            'R2': r2_score(label_test, pred),
            'MAE': mean_absolute_error(label_test, pred),
            'RMSE': mean_squared_error(label_test, pred, squared=False),
            'Pearson': stats.pearsonr(label_test, pred)[0],
            'p1mag': np.mean(np.abs(label_test - pred) < 1)
        }
        
        results_list.append(results)
        print(f"Run {run+1}/{n_runs} - R2: {results['R2']:.4f}, Pearson: {results['Pearson']:.4f}")
    
    # 计算统计量（均值±标准误）
    stats_dict = {}
    for metric in results_list[0].keys():
        values = [r[metric] for r in results_list]
        stats_dict[metric] = {
            'mean': np.mean(values),
            'stderr': stats.sem(values)
        }
    
    return stats_dict


# -------------------------- 数据集处理函数 --------------------------
def process_dataset(train_df, test_df, dataset_name, fold=None):
    """处理单个数据集：提取特征→训练模型→保存结果"""
    # 1. 准备路径
    fold_suffix = f"_fold_{fold}" if fold is not None else ""
    cache_dir = os.path.join(FEATURE_CACHE_ROOT, dataset_name)
    model_save_dir = os.path.join(MODEL_SAVE_ROOT, dataset_name, f"model{fold_suffix}")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 2. 特征缓存路径
    train_feat_path = os.path.join(cache_dir, f"train_features{fold_suffix}.pkl")
    test_feat_path = os.path.join(cache_dir, f"test_features{fold_suffix}.pkl")
    
    # 3. 提取或加载特征
    if not os.path.exists(train_feat_path) or not os.path.exists(test_feat_path):
        print(f"提取{dataset_name}{fold_suffix}特征...")
        
        # 提取SMILES特征
        print("处理SMILES特征...")
        smiles_train = train_df['reactant_smiles'].fillna('').tolist()
        smiles_test = test_df['reactant_smiles'].fillna('').tolist()
        smiles_feat_train = smiles_to_vec(smiles_train)
        smiles_feat_test = smiles_to_vec(smiles_test)
        
        # 提取序列特征
        print("处理蛋白质序列特征...")
        seq_train = train_df['sequence']
        seq_test = test_df['sequence']
        seq_feat_train = Seq_to_vec(seq_train)
        seq_feat_test = Seq_to_vec(seq_test)
        
        # 组合特征
        feature_train = np.concatenate([smiles_feat_train, seq_feat_train], axis=1)
        feature_test = np.concatenate([smiles_feat_test, seq_feat_test], axis=1)
        
        # 保存特征
        with open(train_feat_path, 'wb') as f:
            pickle.dump(feature_train, f)
        with open(test_feat_path, 'wb') as f:
            pickle.dump(feature_test, f)
        print(f"特征保存至: {cache_dir}")
    else:
        print(f"加载缓存特征{dataset_name}{fold_suffix}...")
        with open(train_feat_path, 'rb') as f:
            feature_train = pickle.load(f)
        with open(test_feat_path, 'rb') as f:
            feature_test = pickle.load(f)
    
    # 4. 提取标签
    label_train = train_df[TARGET_COL].dropna().values
    label_test = test_df[TARGET_COL].dropna().values
    
    # 5. 过滤空标签样本
    valid_train = ~np.isnan(label_train)
    valid_test = ~np.isnan(label_test)
    feature_train = feature_train[valid_train]
    feature_test = feature_test[valid_test]
    label_train = label_train[valid_train]
    label_test = label_test[valid_test]
    
    # 6. 训练模型
    print(f"开始训练{dataset_name}{fold_suffix}...")
    stats = train_evaluate(
        feature_train, label_train,
        feature_test, label_test,
        model_save_dir=model_save_dir,
        n_runs=5
    )
    
    # 7. 打印结果
    print(f"\n{dataset_name}{fold_suffix} 结果汇总:")
    for metric, vals in stats.items():
        print(f"{metric}: {vals['mean']:.4f} ± {vals['stderr']:.4f}")
    print("-" * 60)
    
    return stats


# -------------------------- 各数据集训练入口 --------------------------
def train_catpred():
    """训练catpred数据集"""
    print("\n" + "="*60)
    print("开始训练 catpred 数据集")
    print("="*60)
    
    # 数据路径
    train_path = os.path.join(DATA_ROOT, "catpred", "train")
    test_path = os.path.join(DATA_ROOT, "catpred", "test")
    
    # 读取数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 重命名目标列（如果需要）
    if SOURCE_COL in train_df.columns and TARGET_COL not in train_df.columns:
        train_df.rename(columns={SOURCE_COL: TARGET_COL}, inplace=True)
        test_df.rename(columns={SOURCE_COL: TARGET_COL}, inplace=True)
    
    # 处理数据集
    process_dataset(train_df, test_df, "catpred")


def train_cold_enzyme():
    """训练cold_enzyme数据集（1-5折）"""
    print("\n" + "="*60)
    print("开始训练 cold_enzyme 数据集（1-5折）")
    print("="*60)
    
    for fold in range(1, 6):
        # 数据路径
        train_path = os.path.join(
            DATA_ROOT, "turnup", "cold_enzyme", 
            f"kcat_train_fold_{fold}_en.csv"
        )
        test_path = os.path.join(
            DATA_ROOT, "turnup", "cold_enzyme", 
            f"kcat_val_fold_{fold}_en.csv"
        )
        
        # 检查文件是否存在
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"警告：第{fold}折文件不存在，跳过")
            continue
        
        # 读取数据
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # 重命名目标列
        if SOURCE_COL in train_df.columns and TARGET_COL not in train_df.columns:
            train_df.rename(columns={SOURCE_COL: TARGET_COL}, inplace=True)
            test_df.rename(columns={SOURCE_COL: TARGET_COL}, inplace=True)
        
        # 处理数据集
        process_dataset(train_df, test_df, "cold_enzyme", fold=fold)


def train_cold_reaction():
    """训练cold_reaction数据集（1-5折）"""
    print("\n" + "="*60)
    print("开始训练 cold_reaction 数据集（1-5折）")
    print("="*60)
    
    for fold in range(1, 6):
        # 数据路径
        train_path = os.path.join(
            DATA_ROOT, "turnup", "cold_reaction", 
            f"kcat_train_fold_{fold}.csv"
        )
        test_path = os.path.join(
            DATA_ROOT, "turnup", "cold_reaction", 
            f"kcat_val_fold_{fold}.csv"
        )
        
        # 检查文件是否存在
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"警告：第{fold}折文件不存在，跳过")
            continue
        
        # 读取数据
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # 重命名目标列
        if SOURCE_COL in train_df.columns and TARGET_COL not in train_df.columns:
            train_df.rename(columns={SOURCE_COL: TARGET_COL}, inplace=True)
            test_df.rename(columns={SOURCE_COL: TARGET_COL}, inplace=True)
        
        # 处理数据集
        process_dataset(train_df, test_df, "cold_reaction", fold=fold)


def train_warm():
    """训练warm数据集（0-4折）"""
    print("\n" + "="*60)
    print("开始训练 warm 数据集（0-4折）")
    print("="*60)
    
    for fold in range(0, 5):
        # 数据路径
        train_path = os.path.join(
            DATA_ROOT, "turnup", "warm", 
            f"kcat_train_data_{fold}.csv"
        )
        test_path = os.path.join(
            DATA_ROOT, "turnup", "warm", 
            f"kcat_test_data_{fold}.csv"
        )
        
        # 检查文件是否存在
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"警告：第{fold}折文件不存在，跳过")
            continue
        
        # 读取数据
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # 重命名目标列
        if SOURCE_COL in train_df.columns and TARGET_COL not in train_df.columns:
            train_df.rename(columns={SOURCE_COL: TARGET_COL}, inplace=True)
            test_df.rename(columns={SOURCE_COL: TARGET_COL}, inplace=True)
        
        # 处理数据集
        process_dataset(train_df, test_df, "warm", fold=fold)


# -------------------------- 主函数 --------------------------
if __name__ == '__main__':
    # 依次训练所有数据集
    train_catpred()          # catpred数据集
    train_cold_enzyme()      # cold_enzyme（1-5折）
    train_cold_reaction()    # cold_reaction（1-5折）
    train_warm()             # warm（0-4折）
    
    print("\n" + "="*60)
    print("所有数据集训练完成！")
    print(f"模型保存根目录: {MODEL_SAVE_ROOT}")
    print(f"特征缓存目录: {FEATURE_CACHE_ROOT}")
    print("="*60)
