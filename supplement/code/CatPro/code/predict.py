import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
from utils import *  
from model import *  
from act_model import KcatModel as _KcatModel
from torch.utils.data import DataLoader, Dataset
from argparse import RawDescriptionHelpFormatter
import argparse
from tqdm import tqdm
from sklearn.metrics import r2_score
import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
import pickle
import os
from utils import *  
from model import *  
from act_model import KcatModel as _KcatModel
from act_model import KmModel as _KmModel
from act_model import ActivityModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from argparse import RawDescriptionHelpFormatter
import argparse
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
from utils import *  
from model import *  
from act_model import KcatModel as _KcatModel
from act_model import KmModel as _KmModel
from act_model import ActivityModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from argparse import RawDescriptionHelpFormatter
import argparse
from tqdm import tqdm


import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr  

from rdkit import Chem
from rdkit.Chem import MACCSkeys


import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def save_features(file_path, features_dict):
    """保存特征字典到pickle文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"特征字典已保存到 {file_path}")


def GetMACCSKeys(smiles_list):
    fingerprints = []
    for smiles in smiles_list:
        
        if '.' in smiles:
            sub_smiles = smiles.split('.')
            sub_mols = [Chem.MolFromSmiles(s) for s in sub_smiles]
            valid_sub_mols = [mol for mol in sub_mols if mol is not None]
            
            if not valid_sub_mols:
                fingerprints.append(None)  
                continue
                
            
            sub_fps = [MACCSkeys.GenMACCSKeys(mol) for mol in valid_sub_mols]
            combined_fp = sum(sub_fps) / len(sub_fps)
            fingerprints.append(combined_fp)
            
        else:
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fingerprints.append(None)  
            else:
                fp = MACCSkeys.GenMACCSKeys(mol)
                fingerprints.append(fp)
                
    return fingerprints


def load_features(file_path):
    """从pickle文件加载特征字典，如果文件不存在则返回空字典"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                features_dict = pickle.load(f)
            print(f"成功从 {file_path} 加载特征字典")
            return features_dict
        except Exception as e:
            print(f"加载特征字典时出错: {e}")
            return {}
    return {}


class EnzymeDatasets(Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)


def get_datasets(inp_fpath, ProtT5_model, MolT5_model):
    base_dir = "/mnt/usb3/code/gfy/code/catpred_pipeline/data/CatPred-DB/data/kcat/splits_enzyme/2"
    seq_prot5_file = os.path.join(base_dir, f"seq_ProtT5.pkl")
    smi_molT5_file = os.path.join(base_dir, f"smi_MolT5.pkl")

    inp_df = pd.read_csv(inp_fpath, index_col=0)
    sequences = inp_df["sequence"].values
    smiles_lists = inp_df["substrate"].str.split('.').tolist()

    kcat_labels = inp_df["log10kcat_max"].values

    
    seq_prot5_dict = load_features(seq_prot5_file)
    missing_sequences = [seq for seq in sequences if seq not in seq_prot5_dict]

    if missing_sequences:
        print(f"发现 {len(missing_sequences)} 个缺失的序列特征，正在计算...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            new_features = Seq_to_vec(missing_sequences, ProtT5_model)  
        for seq, feat in zip(missing_sequences, new_features):
            seq_prot5_dict[seq] = feat
        save_features(seq_prot5_file, seq_prot5_dict)
    
    seq_ProtT5 = []
    for seq in sequences:
        if seq in seq_prot5_dict:
            seq_ProtT5.append(seq_prot5_dict[seq])
        else:
            print(f"警告: 序列 {seq[:30]}... 缺失特征，正在重新计算...")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    feat = Seq_to_vec([seq], ProtT5_model)
                    if isinstance(feat, np.ndarray) and feat.size > 0:
                        seq_prot5_dict[seq] = feat
                        seq_ProtT5.append(feat)
                        save_features(seq_prot5_file, seq_prot5_dict)
                        print(f"已计算并保存序列 {seq[:30]}... 的特征")
                    else:
                        raise ValueError("特征为空或无效")
                except Exception as e:
                    print(f"计算失败: {e}，使用零向量替代")
                    if seq_prot5_dict:
                        zero_feat = np.zeros_like(list(seq_prot5_dict.values())[0])
                    else:
                        raise ValueError("特征字典为空，无法生成零向量")
                    seq_ProtT5.append(zero_feat)
    seq_ProtT5 = np.vstack(seq_ProtT5)

    
    smi_molT5_dict = load_features(smi_molT5_file)
    all_smi_molT5 = []
    for smiles_list in tqdm(smiles_lists, desc='Processing MolT5'):
        molT5_feats = []
        for smile in smiles_list:
            if smile.strip() and smile not in smi_molT5_dict:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    feat = get_molT5_embed([smile], MolT5_model)  
                smi_molT5_dict[smile] = feat
            if smile.strip() and smile in smi_molT5_dict:
                molT5_feats.append(smi_molT5_dict[smile])
        if molT5_feats:
            avg_molT5 = np.mean(np.array(molT5_feats), axis=0)
            all_smi_molT5.append(avg_molT5)
        else:
            all_smi_molT5.append(np.zeros(1024))

    save_features(smi_molT5_file, smi_molT5_dict)
    smi_molT5 = np.vstack(all_smi_molT5)

    
    all_smi_macc = []
    for smiles_list in tqdm(smiles_lists, desc='Processing MACCS'):
        macc_feats = []
        for smile in smiles_list:
            if smile.strip():
                feat = GetMACCSKeys([smile])[0]
                if feat is not None:
                    macc_feats.append(feat.ToBitString())

        if macc_feats:
            macc_arrays = [np.array([int(c) for c in fp]) for fp in macc_feats]
            avg_macc = np.mean(macc_arrays, axis=0)
            all_smi_macc.append(avg_macc)
        else:
            all_smi_macc.append(np.zeros(167))

    smi_macc = np.vstack(all_smi_macc)

    
    print(f"seq_ProtT5 shape: {np.array(seq_ProtT5).shape}")
    print(f"smi_molT5 shape: {smi_molT5.shape}")
    print(f"smi_macc shape: {smi_macc.shape}")

    
    feats = th.from_numpy(np.concatenate([seq_ProtT5, smi_molT5, smi_macc], axis=1)).to(th.float32)
    datasets = EnzymeDatasets(feats)
    dataloader = DataLoader(datasets, drop_last=True)
    return smiles_lists, dataloader, kcat_labels


def inference(kcat_model, dataloader, device="cuda:0"):
    kcat_model.eval()
    with th.no_grad():
        pred_list = []
        for step, data in enumerate(dataloader):
            data = data.to(device)
            ezy_feats = data[:, :1024]
            sbt_feats = data[:, 1024:]
            pred_kcat = kcat_model(ezy_feats, sbt_feats).cpu().numpy()
            pred_list.append(pred_kcat)

        return np.concatenate(pred_list, axis=0)


if __name__ == "__main__":
    d = "RUN CATAPRO - Predict kcat only..."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp_fpath", type=str, default="enzyme.fasta",
                        help="Input (.fasta). The path of enzyme file.")
    parser.add_argument("-model_dpath", type=str, default="../models",
                        help="Input. The path of saved models.")
    parser.add_argument("-batch_size", type=int, default=64,
                        help="Input. Batch size")
    parser.add_argument("-device", type=str, default="cuda",
                        help="Input. The device: cuda or cpu.")
    parser.add_argument("-out_fpath", type=str, default="catapro_predict_kcat.csv",
                        help="Output. Store the predicted kcat values in this file.")
    args = parser.parse_args()

    inp_fpath = args.inp_fpath
    model_dpath = args.model_dpath
    batch_size = args.batch_size
    device = args.device
    out_fpath = args.out_fpath

    kcat_model_dpath = f"{model_dpath}"
    ProtT5_model = f"/mnt/usb/code/gfy/wangye/prot_t5_xl_uniref50"
    MolT5_model = f"/mnt/usb3/code/gfy/code/CataPro-master/models/molt5-base-smiles2caption"

    smiles_list, dataloader, true_kcat = get_datasets(inp_fpath, ProtT5_model, MolT5_model)

    pred_kcat_list = []
    for fold in range(10):
        kcat_model = KcatModel(device=device)  
        kcat_model.load_state_dict(th.load(f"{kcat_model_dpath}/{fold}_bestmodel.pth", map_location=device))

        pred_kcat = inference(kcat_model, dataloader, device)
        pred_kcat_list.append(pred_kcat)

    
    pred_kcat = np.mean(np.concatenate(pred_kcat_list, axis=1), axis=1, keepdims=True)

    
    pred_flat = pred_kcat.flatten()
    true_flat = true_kcat.flatten()
    from sklearn.metrics import mean_squared_error
    r2 = r2_score(true_flat, pred_flat)
    mse = mean_squared_error(true_flat, pred_flat)
    mae = mean_absolute_error(true_flat, pred_flat)
    pcc, p_value = pearsonr(true_flat, pred_flat)  

    
    print(f"评估指标:")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE (均方误差): {mse:.4f}")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"PCC (皮尔逊相关系数): {pcc:.4f} (p值: {p_value:.4e})")

    
    joined_smiles = ['.'.join(smiles) if isinstance(smiles, list) else smiles for smiles in smiles_list]

    final_score = np.concatenate([
        np.array(joined_smiles).reshape(-1, 1), 
        pred_kcat, 
        true_kcat.reshape(-1, 1)
    ], axis=1)
    
    
    final_df = pd.DataFrame(
        final_score, 
        columns=["smiles", "pred_log10[kcat(s^-1)]", "true_log10[kcat(s^-1)]"]
    )
    
    
    final_df.to_csv(out_fpath, index=False)
    print(f"预测结果已保存至: {out_fpath}")
    from sklearn.metrics import mean_squared_error
    
    metrics_path = os.path.splitext(out_fpath)[0] + "_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"评估指标:\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"MSE (均方误差): {mse:.4f}\n")
        f.write(f"MAE (平均绝对误差): {mae:.4f}\n")
        f.write(f"PCC (皮尔逊相关系数): {pcc:.4f} (p值: {p_value:.4e})\n")
    print(f"评估指标已保存至: {metrics_path}")