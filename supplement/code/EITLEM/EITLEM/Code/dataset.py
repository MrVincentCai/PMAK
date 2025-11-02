import torch
import os
import math
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np

# 注意：不要在全局作用域初始化RDKit对象！


class EitlemDataSet(Dataset):
    def __init__(self, Pairinfo, ProteinsPath, Smiles, nbits, radius, log10, Type='ECFP'):
        super(EitlemDataSet, self).__init__()
        self.pairinfo = Pairinfo
        self.smiles = torch.load(Smiles) if isinstance(Smiles, str) else Smiles
        self.seq_path = os.path.join(ProteinsPath, '{}.pt')
        self.nbits = nbits
        self.radius = radius
        self.log10 = log10
        self.Type = Type
        # 关键：不在__init__中初始化RDKit对象，推迟到__getitem__（子进程中）
        
    def __getitem__(self, idx):
        # 仅在子进程中导入并初始化RDKit（避免主进程序列化）
        from rdkit.Chem import AllChem, MACCSkeys
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        
        # 子进程内初始化fpgen（不跨进程传递）
        fpgen = AllChem.GetRDKitFPGenerator(fpSize=self.nbits) if self.Type in ["RDKIT", "MACCSKeys_RDKIT"] else None
        
        # 原有逻辑
        pro_id = self.pairinfo[idx][0]
        smi_id = self.pairinfo[idx][1]
        value = self.pairinfo[idx][2]
        protein_emb = torch.load(self.seq_path.format(pro_id))
        
        # 生成Mol对象（子进程内创建，不序列化）
        mol = Chem.MolFromSmiles(self.smiles[smi_id].strip())
        if mol is None:
            # 无效SMILES用0向量替代
            if self.Type == "MACCSKeys":
                fp = [0]*167
            elif self.Type == "MACCSKeys_RDKIT":
                fp1, fp2 = [0]*167, [0]*self.nbits
            else:
                fp = [0]*self.nbits
        else:
            # 生成指纹（子进程内处理，不传递Mol对象）
            if self.Type == "ECFP":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.nbits).ToList()
            elif self.Type == "MACCSKeys":
                fp = MACCSkeys.GenMACCSKeys(mol).ToList()
            elif self.Type == "RDKIT":
                fp = fpgen.GetFingerprint(mol).ToList()
            elif self.Type == "MACCSKeys_RDKIT":
                fp1 = MACCSkeys.GenMACCSKeys(mol).ToList()
                fp2 = fpgen.GetFingerprint(mol).ToList()
        
        # 构建Data对象（仅含Tensor，无RDKit对象）
        if self.Type == "MACCSKeys_RDKIT":
            return Data(
                x=torch.FloatTensor(fp1).unsqueeze(0),
                y=torch.FloatTensor(fp2).unsqueeze(0),
                pro_emb=protein_emb,
                value=torch.FloatTensor([value])
            )
        else:
            return Data(
                x=torch.FloatTensor(fp).unsqueeze(0),
                pro_emb=protein_emb,
                value=torch.FloatTensor([value])
            )
    
    def collate_fn(self, batch):
        return Batch.from_data_list(batch, follow_batch=['pro_emb'])
    
    def __len__(self):
        return len(self.pairinfo)


class EitlemDataLoader(DataLoader):
    def __init__(self, data,** kwargs):
        # 强制使用spawn模式，避免fork导致的继承问题
        super().__init__(data, collate_fn=data.collate_fn, multiprocessing_context='spawn', **kwargs)


# 其他函数保持不变
def shuffle_dataset(dataset):
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]