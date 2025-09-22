import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys



import torch
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np
import re
import gc

def Seq_to_vec( Sequence, tokenizer_path="/mnt/usb/code/gfy/wangye/prot_t5_xl_uniref50",
                            model_path="/mnt/usb/code/gfy/wangye/prot_t5_xl_uniref50"):
    """
    根据蛋白质的 Sequence 序列，计算特征向量。

    :param Sequence: 蛋白质序列列表
    :param tokenizer_path: 分词器的路径
    :param model_path: 模型的路径
    :return: 特征向量列表
    """
    
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    try:
        
        model = T5EncoderModel.from_pretrained(model_path)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败，错误信息：{e}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()

    
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]

    
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    
    gc.collect()
    print(torch.cuda.is_available())

    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i + 1))
        sequences_Example_i = sequences_Example[i]
        
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)

    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize

def GetMACCSKeys(smiles_list):
    """
    Output: np.array, size is 167.
    """
    N_smiles = len(smiles_list)
    if len(set(smiles_list)) == 1:
        mol = Chem.MolFromSmiles(smiles_list[0])
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_str = fp.ToBitString()
        fp_array = np.array([int(i) for i in fp_str])
        final_values = np.concatenate([fp_array.reshape(1, -1)] * N_smiles, axis=0)
    else:
        final_values = []
        for smile in smiles_list:
            mol = Chem.MolFromSmiles(smile)
            fp = MACCSkeys.GenMACCSKeys(mol)
            fp_str = fp.ToBitString()
            fp_array = np.array([int(i) for i in fp_str])
            final_values.append(fp_array.reshape(1, -1))
        final_values = np.concatenate(final_values, axis=0)
    return final_values
    
def get_molT5_embed(smiles_list, Molt5_model):
    tokenizer = T5Tokenizer.from_pretrained(Molt5_model, local_files_only=True)
    
    model = T5EncoderModel.from_pretrained(Molt5_model)
    N_smiles = len(smiles_list)
    if len(set(smiles_list)) == 1:
        input_ids = tokenizer(smiles_list[0], return_tensors="pt").input_ids
        outputs = model(input_ids=input_ids)
        last_hidden_states = outputs.last_hidden_state
        embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()
        final_values = np.concatenate([embed.reshape(1, -1)] * N_smiles, axis=0)
    else:
        final_values = []
        for smile in smiles_list:
            input_ids = tokenizer(smile, return_tensors="pt").input_ids
            outputs = model(input_ids=input_ids)
            last_hidden_states = outputs.last_hidden_state
            embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()
            final_values.append(embed.reshape(1, -1))
        final_values = np.concatenate(final_values, axis=0)
    return final_values

