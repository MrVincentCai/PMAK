import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import os
import gc
from os.path import join


from Kcat_model import InteractPre as Model_Regression
import data_processor  

class CFG:
    """配置类，与训练时保持一致"""
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_number = torch.cuda.device_count()
    batch_size = 1  

class CustomDataset(Dataset):
    """数据加载类，与训练时保持一致"""
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        reaction = torch.tensor(self.dataframe.iloc[idx]['rxnfp'], dtype=torch.float32).unsqueeze(0)
        
        
        label_col = find_column(self.dataframe, 'log10kcat_max') or find_column(self.dataframe, 'geomean_kcat') or find_column(self.dataframe, 'log10_value')
        if label_col is None:
            raise ValueError("未找到有效的标签列")
        label = torch.tensor(self.dataframe.iloc[idx][label_col], dtype=torch.float32)
        
        
        protein_col = find_column(self.dataframe, 'uniref50_dim1024') or find_column(self.dataframe, 'Prot5')
        if protein_col is None:
            raise ValueError("未找到有效的蛋白质特征列")
        protein = torch.tensor(self.dataframe.iloc[idx][protein_col], dtype=torch.float32)
        
        return reaction, protein, label

def find_column(df, target_name):
    """查找DataFrame中匹配的列名（忽略大小写），与训练代码保持一致"""
    target_lower = target_name.lower()
    for col in df.columns:
        if col.lower() == target_lower:
            return col
    return None

def load_model(model_path, cfg):
    """加载训练好的模型"""
    model = Model_Regression().to(cfg.DEVICE)
    try:
        
        model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE), strict=False)
        model.eval()  
        print(f"模型加载成功: {model_path}")
        return model
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise

def predict(model, data_loader, cfg):
    """使用模型进行预测"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  
        for reaction, protein, label in data_loader:
            
            reaction = reaction.to(cfg.DEVICE)
            protein = protein.to(cfg.DEVICE)
            label = label.to(cfg.DEVICE)
            
            
            outputs = model(reaction, protein)
            outputs = outputs.squeeze()  
            
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    return all_preds, all_labels

def evaluate_predictions(preds, labels):
    """评估预测结果"""
    mse = mean_squared_error(labels, preds)
    r2 = r2_score(labels, preds)
    pearson_corr, _ = pearsonr(labels, preds)
    mae = mean_absolute_error(labels, preds)
    
    print("\n预测评估指标:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Pearson 相关系数: {pearson_corr:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return {
        'mse': mse,
        'r2': r2,
        'pearson': pearson_corr,
        'mae': mae,
        'preds': preds,
        'labels': labels
    }

def test_dataset_group(group_name, model_dir, test_data_path, results_dir, num_seeds=10):
    """测试特定数据集组的所有种子模型"""
    print(f"\n===== 开始测试数据集组: {group_name} =====")
    
    
    os.makedirs(results_dir, exist_ok=True)
    
    
    print(f"加载测试数据: {test_data_path}")
    test_df = data_processor.load_processed_data(test_data_path)
    if test_df is None or len(test_df) == 0:
        print(f"错误: 无法加载有效的测试数据 {test_data_path}")
        return
    
    
    test_dataset = CustomDataset(test_df)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CFG.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    
    all_results = []
    
    
    for seed in range(num_seeds):
        print(f"\n----- 测试种子 {seed} 的模型 -----")
        
        
        model_path = os.path.join(model_dir, f'model_seed_{seed}.pth')
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在 {model_path}，跳过该种子")
            continue
        
        
        cfg = CFG()
        model = load_model(model_path, cfg)
        
        
        print("开始预测...")
        preds, labels = predict(model, test_loader, cfg)
        
        
        results = evaluate_predictions(preds, labels)
        results['seed'] = seed
        all_results.append(results)
        
        
        result_df = pd.DataFrame({
            '真实值': labels,
            '预测值': preds
        })
        seed_output_csv = os.path.join(results_dir, f'prediction_results_seed_{seed}.csv')
        result_df.to_csv(seed_output_csv, index=False)
        print(f"种子 {seed} 的预测结果已保存至: {seed_output_csv}")
        
        
        del model, preds, labels, result_df
        gc.collect()
        torch.cuda.empty_cache()
    
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv = os.path.join(results_dir, 'prediction_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n所有种子的汇总结果已保存至: {summary_csv}")
        
        
        avg_metrics = {
            '平均MSE': summary_df['mse'].mean(),
            '平均R²': summary_df['r2'].mean(),
            '平均Pearson': summary_df['pearson'].mean(),
            '平均MAE': summary_df['mae'].mean()
        }
        
        print("\n所有种子模型的平均性能:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return avg_metrics
    else:
        print("\n没有有效的模型结果可汇总")
        return None

def main():
    
    print("初始化数据处理所需的模型和分词器...")
    data_processor.init_models()
    
    
    base_paths = {
        "catpred": "E:/PMAK_/data/catpred",
        "cold_enzyme": "E:/PMAK_/data/turnup/cold_enzyme",
        "cold_reaction": "E:/PMAK_/data/turnup/cold_reaction",
        "warm": "E:/PMAK_/data/turnup/warm"
    }
    
    processing_dirs = {
        "csv_dir": "/mnt/usb3/code/gfy/code/catpred_pipeline/data/CatPred-DB/splits_revision",
        "processed_data_dir": "/mnt/usb3/code/gfy/data/processed_data"
    }
    
    
    model_root = "/mnt/usb3/code/gfy/data/models_cyx"
    
    results_root = "/mnt/usb3/code/gfy/data/test_results"
    os.makedirs(results_root, exist_ok=True)
    
    
    catpred_model_dir = os.path.join(model_root, "catpred")
    catpred_test_data = os.path.join(processing_dirs["processed_data_dir"], "catpred", "test", "test_with.pkl")
    catpred_results_dir = os.path.join(results_root, "catpred")
    
    test_dataset_group(
        "catpred",
        catpred_model_dir,
        catpred_test_data,
        catpred_results_dir,
        num_seeds=10
    )
    
    print("\n所有数据集测试完成！")

if __name__ == "__main__":
    main()
