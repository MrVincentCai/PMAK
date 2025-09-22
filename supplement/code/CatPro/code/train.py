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
from rdkit import Chem
from rdkit.Chem import MACCSkeys


def save_features(file_path, features_dict):
    """保存特征字典到pickle文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"特征字典已保存到 {file_path}")


def save_features(file_path, features_dict):
    """保存特征字典到pickle文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"特征字典已保存到 {file_path}")


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
    def __init__(self, values, labels):
        self.values = values
        self.labels = labels

    def __getitem__(self, idx):
        return self.values[idx], self.labels[idx]

    def __len__(self):
        return len(self.values)


import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


def get_datasets(inp_fpath, ProtT5_model, MolT5_model, kf):
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
    seq_ProtT5=np.vstack(seq_ProtT5)
    
    
    smi_molT5_dict = load_features(smi_molT5_file)
    all_smiles = [smile for sublist in smiles_lists for smile in sublist if smile.strip()]
    missing_smiles = [smile for smile in all_smiles if smile not in smi_molT5_dict]

    if missing_smiles:
        print(f"发现 {len(missing_smiles)} 个缺失的SMILES特征，正在计算...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            new_features = Smi_to_vec(missing_smiles, MolT5_model)
        for smi, feat in zip(missing_smiles, new_features):
            smi_molT5_dict[smi] = feat
        save_features(smi_molT5_file, smi_molT5_dict)

    
    smi_molT5 = []
    for smiles_list in smiles_lists:
        mol_feats = []
        for smile in smiles_list:
            if smile.strip():
                if smile in smi_molT5_dict:
                    mol_feats.append(smi_molT5_dict[smile])
                else:
                    print(f"警告: SMILES {smile} 缺失特征，正在重新计算...")
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        try:
                            feat = Smi_to_vec([smile], MolT5_model)
                            if isinstance(feat, np.ndarray) and feat.size > 0:
                                smi_molT5_dict[smile] = feat
                                mol_feats.append(feat)
                                save_features(smi_molT5_file, smi_molT5_dict)
                            else:
                                raise ValueError("特征为空或无效")
                        except Exception as e:
                            print(f"计算失败: {e}，使用零向量替代")
                            if smi_molT5_dict:
                                zero_feat = np.zeros_like(list(smi_molT5_dict.values())[0])
                            else:
                                raise ValueError("特征字典为空，无法生成零向量")
                            mol_feats.append(zero_feat)
        if mol_feats:
            avg_mol = np.mean(np.array(mol_feats), axis=0)
            smi_molT5.append(avg_mol)
        else:
            if smi_molT5_dict:
                zero_feat = np.zeros_like(list(smi_molT5_dict.values())[0])
            else:
                zero_feat = np.zeros(768)  
            smi_molT5.append(zero_feat)
    smi_molT5 = np.vstack(smi_molT5)

    
    all_smi_macc = []
    for smiles_list in tqdm(smiles_lists, desc='Processing MACCS'):
        macc_feats = []
        for smile in smiles_list:
            if smile.strip():  
                try:
                    
                    mol = Chem.MolFromSmiles(smile.strip())
                    if mol is None:
                        raise ValueError(f"无效的SMILES: {smile}")
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        feat = MACCSkeys.GenMACCSKeys(mol)  
                        feat = np.array(feat)
                    macc_feats.append(feat)
                except Exception as e:
                    print(f"处理SMILES '{smile}' 时出错: {str(e)}")
                    
                    macc_feats.append(np.zeros(167))
        if macc_feats:
            avg_macc = np.mean(np.array(macc_feats), axis=0)
            all_smi_macc.append(avg_macc)
        else:
            all_smi_macc.append(np.zeros(167))  
    
    smi_macc = np.vstack(all_smi_macc)

    
    print(f"seq_ProtT5 shape: {np.array(seq_ProtT5).shape}")
    print(f"smi_molT5 shape: {smi_molT5.shape}")
    print(f"smi_macc shape: {smi_macc.shape}")

    
    feats = np.concatenate([seq_ProtT5, smi_molT5, smi_macc], axis=1)
    labels = kcat_labels.reshape(-1, 1)

    
    datasets = []
    for train_index, test_index in kf.split(feats):
        train_feats = th.from_numpy(feats[train_index]).to(th.float32)
        train_labels = th.from_numpy(labels[train_index]).to(th.float32)
        test_feats = th.from_numpy(feats[test_index]).to(th.float32)
        test_labels = th.from_numpy(labels[test_index]).to(th.float32)
        datasets.append((
            DataLoader(EnzymeDatasets(train_feats, train_labels), batch_size=8, shuffle=True, drop_last=True),
            DataLoader(EnzymeDatasets(test_feats, test_labels), batch_size=8, shuffle=False, drop_last=True)
        ))
    return datasets


def train(kcat_model, dataloader, optimizer_kcat, criterion, device="cuda:0", epochs=100):
    kcat_model.train()
    for epoch in range(epochs):
        total_loss_kcat = 0
        
        with tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}') as pbar:
            for step, (data, labels) in enumerate(pbar):
                data = data.to(device)
                labels = labels.to(device)

                ezy_feats = data[:, :1024]
                sbt_feats = data[:, 1024:]

                optimizer_kcat.zero_grad()
                pred_kcat = kcat_model(ezy_feats, sbt_feats)
                loss_kcat = criterion(pred_kcat, labels)
                loss_kcat.backward()
                optimizer_kcat.step()
                total_loss_kcat += loss_kcat.item()

                
                pbar.set_postfix({'Kcat Loss': total_loss_kcat / (step + 1)})

        print(f'Epoch {epoch + 1}/{epochs}: Kcat Loss = {total_loss_kcat / len(dataloader)}')


def evaluate(kcat_model, dataloader, criterion, device="cuda:0"):
    kcat_model.eval()
    total_loss_kcat = 0
    
    with tqdm(dataloader, desc='Evaluating') as pbar:
        with th.no_grad():
            for step, (data, labels) in enumerate(pbar):
                data = data.to(device)
                labels = labels.to(device)

                ezy_feats = data[:, :1024]
                sbt_feats = data[:, 1024:]

                pred_kcat = kcat_model(ezy_feats, sbt_feats)
                loss_kcat = criterion(pred_kcat, labels)
                total_loss_kcat += loss_kcat.item()

                
                pbar.set_postfix({'Kcat Loss': total_loss_kcat / (step + 1)})

    return total_loss_kcat / len(dataloader)

if __name__ == "__main__":
    d = "RUN CATAPRO ..."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp_fpath", type=str, default="enzyme.fasta",
                        help="Input (.fasta). The path of enzyme file.")
    parser.add_argument("-model_dpath", type=str, default="model_dpah",
                        help="Input. The path of saved models.")
    parser.add_argument("-batch_size", type=int, default=8,
                        help="Input. Batch size")
    parser.add_argument("-device", type=str, default="cuda",
                        help="Input. The device: cuda or cpu.")
    parser.add_argument("-epochs", type=int, default=80,
                        help="Input. Number of training epochs.")
    args = parser.parse_args()

    inp_fpath = args.inp_fpath
    model_dpath = args.model_dpath
    batch_size = args.batch_size
    device = args.device
    epochs = args.epochs

    kcat_model_dpath = f"{model_dpath}"
    ProtT5_model = f"/mnt/usb/code/gfy/wangye/prot_t5_xl_uniref50"
    MolT5_model = f"/mnt/usb3/code/gfy/code/CataPro-master/models/molt5-base-smiles2caption"

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    datasets = get_datasets(inp_fpath, ProtT5_model, MolT5_model, kf)

    for fold, (train_dataloader, test_dataloader) in enumerate(datasets):
        kcat_model = KcatModel(device=device)

        criterion = nn.MSELoss()
        optimizer_kcat = th.optim.Adam(kcat_model.parameters(), lr=0.00001)  

        print(f"Training fold {fold + 1}...")
        train(kcat_model, train_dataloader, optimizer_kcat, criterion, device, epochs)

        print(f"Evaluating fold {fold + 1}...")
        test_loss = evaluate(kcat_model, test_dataloader, criterion, device)
        print(f"Fold {fold + 1} Test Loss: {test_loss}")

        th.save(kcat_model.state_dict(), f"{kcat_model_dpath}/{fold}_bestmodel.pth")
