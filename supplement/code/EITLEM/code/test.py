
import torch
import pandas as pd
import numpy as np
import os
import argparse
from rdkit import Chem
from rdkit import RDLogger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# 配置日志（禁用RDKit警告）
RDLogger.DisableLog('rdApp.*')

# 导入训练代码中的必要组件
from train import KCATDataset, generate_mol_feature
from KCM import EitlemKcatPredictor

# 定义默认的SMILES字符串，当处理失败时使用
DEFAULT_SMILES = "O"


def load_model(model_path, device, mol_feature_dim=167):
    """加载训练好的模型"""
    model = EitlemKcatPredictor(
        mol_in_dim=mol_feature_dim,
        hidden_dim=512,
        protein_dim=1280,
        layer=10,
        dropout=0.5,
        att_layer=10
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, data_loader, device):
    """使用模型进行预测"""
    model.eval()
    pred_list = []
    target_list = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred = model(batch)
            pred_list.append(pred.cpu().numpy())
            target_list.append(batch.y.cpu().numpy())

    # 拼接所有批次的结果
    y_pred = np.concatenate(pred_list).flatten()
    y_true = np.concatenate(target_list).flatten()
    return y_pred, y_true


def calculate_metrics(y_pred, y_true):
    """计算评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pcc, p_value = pearsonr(y_true, y_pred)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "PCC": pcc,
        "P-value": p_value
    }


def process_complex_smiles(smiles):
    """
    处理SMILES：
    1. 保留完整的底物SMILES（不拆分复合物）
    2. 仅在完整SMILES无效时使用默认SMILES
    """
    # 处理空值情况
    if pd.isna(smiles):
        print("检测到空SMILES，使用默认值")
        return DEFAULT_SMILES
    
    # 确保输入是字符串
    smiles = str(smiles)
    
    # 直接验证完整的SMILES（不拆分复合物）
    if is_valid_smiles(smiles):
        return smiles
    else:
        print(f"完整SMILES无效，使用默认值: {DEFAULT_SMILES[:50]}...")
        return DEFAULT_SMILES


def is_valid_smiles(smiles):
    """检查SMILES是否有效（可被RDKit解析）"""
    try:
        # 处理空字符串情况
        if not smiles or pd.isna(smiles):
            return False
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except Exception as e:
        print(f"SMILES验证错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test KCAT prediction model")
    parser.add_argument("--test_pkl", type=str, required=True, help="Path to test pkl file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pt file)")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Output directory for results")
    parser.add_argument("--mol_type", type=str, default="MACCSKeys", choices=["ECFP", "MACCSKeys", "RDKIT"],
                        help="Molecular feature type (must match training setting)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--log10", type=bool, default=True,
                        help="Whether KCAT is log10 transformed (must match training setting)")

    args = parser.parse_args()

    # 验证默认SMILES是否有效
    if not is_valid_smiles(DEFAULT_SMILES):
        print(f"警告: 默认SMILES {DEFAULT_SMILES[:50]}... 无效，可能导致后续错误")
    else:
        print(f"默认SMILES验证有效: {DEFAULT_SMILES[:50]}...")

    # 设备配置
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 确定分子特征维度（与训练时保持一致）
    mol_feature_dim = 167

    # 加载数据
    print(f"Loading test data from {args.test_pkl}")
    test_df = pd.read_pickle(args.test_pkl)
    original_size = len(test_df)
    print(f"Original test data size: {original_size}")
    
    # 检查是否存在'reactant_smiles'列
    if 'reactant_smiles' not in test_df.columns:
        raise ValueError("测试数据中没有找到'reactant_smiles'列，请检查数据格式")
    
    # 显示原始数据中的第一个SMILES，用于调试
    if original_size > 0:
        print(f"原始第一个SMILES: {test_df['reactant_smiles'].iloc[0][:100]}...")

    # -------------------------- SMILES处理逻辑 --------------------------
    # 处理SMILES（保留完整底物SMILES）
    print("Processing SMILES (keeping full substrate SMILES)...")
    # 应用SMILES处理函数
    test_df['processed_substrate'] = test_df['reactant_smiles'].apply(process_complex_smiles)
    
    # 显示处理后的第一个SMILES，用于调试
    if len(test_df) > 0:
        print(f"处理后的第一个SMILES: {test_df['processed_substrate'].iloc[0][:100]}...")
    
    # 统计使用默认SMILES的数量
    default_count = sum(1 for s in test_df['processed_substrate'] if s == DEFAULT_SMILES)
    print(f"使用默认SMILES的样本数量: {default_count}/{original_size}")

    # 统计复合物SMILES的数量（含'.'的SMILES）
    complex_count = sum(1 for s in test_df['reactant_smiles'] if '.' in str(s))
    print(f"复合物SMILES的数量: {complex_count}/{original_size}")

    # 将处理后的SMILES替换原列
    test_df['reactant_smiles'] = test_df['processed_substrate']
    test_df = test_df.drop(columns=['processed_substrate'])  # 清理临时列
    # -------------------------------------------------------------------------

    # 创建测试数据集和数据加载器
    test_dataset = KCATDataset(
        test_df,  # 使用处理后的所有数据，无效的已替换为默认值
        protein_feature_col='esm2_features',
        smile_col='reactant_smiles',
        kcat_col='log10kcat_max',
        mol_type=args.mol_type,
        log10=args.log10
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn
    )

    # 加载模型
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device, mol_feature_dim)

    # 进行预测
    print("Starting prediction...")
    y_pred, y_true = predict(model, test_loader, device)

    # 计算评估指标
    metrics = calculate_metrics(y_pred, y_true)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 保存预测结果，包含使用默认SMILES的标记和是否为复合物SMILES的标记
    result_df = pd.DataFrame({
        "true_kcat": y_true,
        "predicted_kcat": y_pred,
        "used_default_smiles": [s == DEFAULT_SMILES for s in test_df['reactant_smiles']],
        "is_complex_smiles": ['.' in str(s) for s in test_df['reactant_smiles']]
    })
    result_path = os.path.join(args.output_dir, "prediction_results.csv")
    result_df.to_csv(result_path, index=False)
    print(f"Prediction results saved to {result_path}")

    # 保存指标结果
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"复合物SMILES的比例: {complex_count/original_size:.2%}\n")
        f.write(f"使用默认SMILES的样本比例: {default_count/original_size:.2%}\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")


if __name__ == "__main__":
    main()
    