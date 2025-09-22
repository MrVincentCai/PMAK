import sys
from os.path import join
from torch.utils.data import DataLoader, Dataset
import pickle
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import random
import os
import gc
from sklearn.model_selection import train_test_split
import torch.nn as nn


import data_processor


assert torch.cuda.is_available(), "Must have available gpu"

import tempfile
tempfile.tempdir = "/mnt/usb3/tmp"  
os.makedirs(tempfile.tempdir, exist_ok=True)  

def seed_everything(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CFG:
    """配置参数类"""
    gpu_number = torch.cuda.device_count()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    EPOCHES = 40
    lr = 5e-5
    weight_decay = 1e-5
    batch_size = 1  


class CustomDataset(Dataset):
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
    """查找DataFrame中匹配的列名（忽略大小写）"""
    target_lower = target_name.lower()
    for col in df.columns:
        if col.lower() == target_lower:
            return col
    return None


def train_model(model, train_data, test_data, CFG, seed, save_dir):
    """训练模型并保存最佳模型"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    criterion = nn.MSELoss()

    
    train_dataloader = DataLoader(
        train_data,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    history_df = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss', 'mse', 'r2', 'pearson_corr', 'mae'])
    best_r2 = -float('inf')
    model_save_path = os.path.join(save_dir, f'model_seed_{seed}.pth')

    for epoch in range(CFG.EPOCHES):
        
        loss_total = 0.0
        count = 0
        for reaction, protein, label in train_dataloader:
            reaction = reaction.to(CFG.DEVICE)
            protein = protein.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)

            output = model(reaction, protein)
            output = output.squeeze()  
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            count += 1

        train_loss = loss_total / count
        print(f'Epoch {epoch + 1}/{CFG.EPOCHES} - Train Loss: {train_loss:.4f}')

        
        test_loss, mse, r2, pearson_corr, mae = evaluate_model(model, test_dataloader, criterion, CFG)

        
        if r2 > best_r2:
            print(f'R² improved from {best_r2:.4f} to {r2:.4f}. Saving model...')
            best_r2 = r2
            torch.save(model.state_dict(), model_save_path)

        
        history_df = pd.concat([history_df, pd.DataFrame([{
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'mse': mse,
            'r2': r2,
            'pearson_corr': pearson_corr,
            'mae': mae
        }])], ignore_index=True)

    
    history_df.to_csv(os.path.join(save_dir, f'training_history_seed_{seed}.csv'), index=False)
    print(f'Training completed for seed {seed}. Best R²: {best_r2:.4f}')
    return model_save_path


def evaluate_model(model, test_dataloader, criterion, CFG):
    """评估模型并返回性能指标"""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for reaction, protein, label in test_dataloader:
            reaction = reaction.to(CFG.DEVICE)
            protein = protein.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)

            outputs = model(reaction, protein)
            outputs = outputs.squeeze()  
            
            loss = criterion(outputs, label)
            test_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    pearson_corr, _ = pearsonr(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)

    print(f'Test Loss: {test_loss / len(test_dataloader):.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'Pearson Correlation: {pearson_corr:.4f}')
    print(f'MAE: {mae:.4f}')

    return test_loss / len(test_dataloader), mse, r2, pearson_corr, mae


def process_and_train_dataset_group(name, train_csv_files, test_csv_files, 
                                   csv_dir, pkl_dir, processed_data_dir,
                                   save_root, num_seeds=10, batch_size=500):
    """处理数据集组并进行训练"""
    print(f"\n===== 开始处理数据集组: {name} =====")
    
    
    save_dir = os.path.join(save_root, name)
    os.makedirs(save_dir, exist_ok=True)
    
    
    print(f"开始处理训练数据: {name}")
    processed_train_files = data_processor.process_csv_files(
        train_csv_files, 
        csv_dir, 
        os.path.join(pkl_dir, "train"), 
        os.path.join(processed_data_dir, "train"),
        batch_size=batch_size
    )
    
    if not processed_train_files:
        print(f"错误: 没有生成有效的训练数据，跳过数据集组 {name}")
        return
    
    
    train_dfs = []
    for file_path in processed_train_files:
        df = data_processor.load_processed_data(file_path)
        if df is not None and len(df) > 0:
            train_dfs.append(df)
            print(f"已加载训练数据: {file_path}, 样本数: {len(df)}")
    
    if not train_dfs:
        print(f"错误: 没有找到有效的训练数据，跳过数据集组 {name}")
        return
    
    combined_train_data = pd.concat(train_dfs, ignore_index=True)
    print(f"合并后训练数据总量: {len(combined_train_data)}")
    
    
    print(f"开始处理测试数据: {name}")
    processed_test_files = data_processor.process_csv_files(
        test_csv_files, 
        csv_dir, 
        os.path.join(pkl_dir, "test"), 
        os.path.join(processed_data_dir, "test"),
        batch_size=batch_size
    )
    
    if not processed_test_files:
        print(f"错误: 没有生成有效的测试数据，跳过数据集组 {name}")
        return
    
    
    test_dfs = []
    for file_path in processed_test_files:
        df = data_processor.load_processed_data(file_path)
        if df is not None and len(df) > 0:
            test_dfs.append(df)
            print(f"已加载测试数据: {file_path}, 样本数: {len(df)}")
    
    if not test_dfs:
        print(f"错误: 没有找到有效的测试数据，跳过数据集组 {name}")
        return
    
    combined_test_data = pd.concat(test_dfs, ignore_index=True)
    print(f"合并后测试数据总量: {len(combined_test_data)}")
    
    
    train_dataset = CustomDataset(combined_train_data)
    test_dataset = CustomDataset(combined_test_data)
    
    
    trained_models = []
    for seed in range(num_seeds):
        print(f"\n===== 开始训练种子 {seed} 的模型 =====")
        seed_everything(37 + seed)  
        model = Model_Regression().to(CFG.DEVICE)
        model_path = train_model(model, train_dataset, test_dataset, CFG, seed, save_dir)
        trained_models.append(model_path)
        print(f"种子 {seed} 训练完成，模型保存至: {model_path}")
    
    
    del train_dataset, test_dataset, combined_train_data, combined_test_data, train_dfs, test_dfs
    gc.collect()
    return trained_models


if __name__ == "__main__":
    
    print("初始化数据处理所需的模型和分词器...")
    data_processor.init_models()
    
    
    from Kcat_model import InteractPre as Model_Regression

    
    base_paths = {
        "catpred": "E:/PMAK_/data/catpred",
        "cold_enzyme": "E:/PMAK_/data/turnup/cold_enzyme",
        "cold_reaction": "E:/PMAK_/data/turnup/cold_reaction",
        "warm": "E:/PMAK_/data/turnup/warm"
    }
    
    
    processing_dirs = {
        "csv_dir": "/mnt/usb3/code/gfy/code/catpred_pipeline/data/CatPred-DB/splits_revision",
        "pkl_dir": "/mnt/usb3/code/gfy/data/pkl_files",
        "processed_data_dir": "/mnt/usb3/code/gfy/data/processed_data"
    }
    
    
    save_root = "/mnt/usb3/code/gfy/data/models_cyx"
    os.makedirs(save_root, exist_ok=True)
    
    
    catpred_train_files = ["train"]  
    catpred_test_files = ["test"]
    process_and_train_dataset_group(
        "catpred",
        catpred_train_files,
        catpred_test_files,
        join(processing_dirs["csv_dir"], "catpred"),
        join(processing_dirs["pkl_dir"], "catpred"),
        join(processing_dirs["processed_data_dir"], "catpred"),
        save_root
    )
