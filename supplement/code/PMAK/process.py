import pandas as pd
import numpy as np
import pickle
import os
import re
import gc
from tqdm import tqdm
from rdkit import Chem
from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator, get_default_model_and_tokenizer
from transformers import T5Tokenizer, T5EncoderModel
import torch


_model_t5 = None
_tokenizer = None
_rxnfp_generator = None
_device = None

def init_models(tokenizer_path="/mnt/usb/code/gfy/wangye/prot_t5_xl_uniref50", 
                model_path="/mnt/usb/code/gfy/wangye/prot_t5_xl_uniref50"):
    """初始化模型和分词器，应在开始时调用一次"""
    global _model_t5, _tokenizer, _rxnfp_generator, _device
    
    
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {_device}")
    
    
    try:
        _tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
        _model_t5 = T5EncoderModel.from_pretrained(model_path)
        _model_t5 = _model_t5.to(_device)
        _model_t5 = _model_t5.eval()
        print("蛋白质特征模型加载成功！")
    except Exception as e:
        print(f"蛋白质特征模型加载失败：{e}")
        raise
    
    
    try:
        model_rxn, tokenizer_rxn = get_default_model_and_tokenizer()
        _rxnfp_generator = RXNBERTFingerprintGenerator(model_rxn, tokenizer_rxn)
        print("反应指纹模型加载成功！")
    except Exception as e:
        print(f"反应指纹模型加载失败：{e}")
        raise


def convert_to_smiles(input_str):
    """将输入转换为SMILES格式"""
    try:
        if isinstance(input_str, str):
            if input_str.startswith('InChI='):
                mol = Chem.inchi.MolFromInchi(input_str)
            else:
                mol = Chem.MolFromSmiles(input_str)
            if mol is not None:
                return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"转换为SMILES时出错: {e}")
    return None


def seq_to_uniref50_dim1024(sequence):
    """根据蛋白质序列计算uniref50_dim1024特征向量"""
    global _model_t5, _tokenizer, _device
    
    if not isinstance(sequence, str):
        return None
        
    
    if len(sequence) > 1000:
        sequence = sequence[:500] + sequence[-500:]
    
    
    formatted_seq = ' '.join(list(sequence))
    
    try:
        
        formatted_seq = re.sub(r"[UZOB]", "X", formatted_seq)
        
        
        ids = _tokenizer.batch_encode_plus([formatted_seq], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(_device)
        attention_mask = torch.tensor(ids['attention_mask']).to(_device)
        
        with torch.no_grad():
            embedding = _model_t5(input_ids=input_ids, attention_mask=attention_mask)
        
        
        embedding = embedding.last_hidden_state.cpu().numpy()
        seq_len = (attention_mask[0] == 1).sum()
        seq_emd = embedding[0][:seq_len - 1]  
        
        return seq_emd
    
    except Exception as e:
        print(f"计算蛋白质特征时出错: {e}")
        return None


def calculate_rxnfp(reaction_smiles):
    """计算反应指纹"""
    global _rxnfp_generator
    
    if not isinstance(reaction_smiles, str) or '>>' not in reaction_smiles:
        return None
        
    try:
        return _rxnfp_generator.convert(reaction_smiles)
    except Exception as e:
        print(f"计算反应指纹时出错: {e}")
        return None


def process_dataframe_chunk(chunk):
    """处理数据块，添加特征列"""
    
    tqdm.pandas(desc="处理蛋白质序列")
    chunk['uniref50_dim1024'] = chunk['sequence'].progress_apply(seq_to_uniref50_dim1024)
    
    
    valid_protein_mask = chunk['uniref50_dim1024'].notna()
    chunk = chunk[valid_protein_mask].copy()
    print(f"过滤后保留的蛋白质特征: {len(chunk)}/{len(valid_protein_mask)}")
    
    
    chunk['uniref50_dim1024'] = chunk['uniref50_dim1024'].apply(
        lambda x: np.array(x, dtype=np.float32) if x is not None else None
    )
    
    
    tqdm.pandas(desc="处理反应SMILES")
    chunk['rxnfp'] = chunk['reaction_smiles'].progress_apply(calculate_rxnfp)
    
    
    valid_rxn_mask = chunk['rxnfp'].notna()
    chunk = chunk[valid_rxn_mask].copy()
    print(f"过滤后保留的反应指纹: {len(chunk)}/{len(valid_rxn_mask)}")
    
    
    if 'log10_value' in chunk.columns:
        chunk.rename(columns={'log10_value': 'geomean_kcat'}, inplace=True)
    if 'log10kcat_max' in chunk.columns:
        chunk.rename(columns={'log10kcat_max': 'geomean_kcat'}, inplace=True)
    
    return chunk


def process_csv_files(csv_files, csv_dir, pkl_dir, output_dir, batch_size=500, start_index=0):
    """
    处理多个CSV文件，转换为PKL并添加特征
    
    参数:
        csv_files: 要处理的CSV文件名列表（不包含扩展名）
        csv_dir: CSV文件所在目录
        pkl_dir: 转换后的PKL文件保存目录
        output_dir: 处理后的输出文件目录
        batch_size: 批次大小
        start_index: 开始处理的批次索引，用于断点续传
        
    返回:
        处理后的文件路径列表
    """
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)
    
    processed_files = []
    
    
    for base_name in csv_files:
        
        csv_path = os.path.join(csv_dir, f"{base_name}.csv")
        pkl_path = os.path.join(pkl_dir, f"{base_name}.pkl")
        output_file_path = os.path.join(output_dir, f"{base_name}_with.pkl")
        
        
        if not os.path.exists(pkl_path) or os.path.getmtime(csv_path) > os.path.getmtime(pkl_path):
            try:
                df = pd.read_csv(csv_path)
                print(f"读取CSV文件，共 {len(df)} 行: {csv_path}")
                df.to_pickle(pkl_path)
                print(f"已将CSV转换为PKL: {pkl_path}")
            except FileNotFoundError:
                print(f"错误：CSV文件不存在 - {csv_path}")
                continue  
            except Exception as e:
                print(f"转换CSV到PKL失败 {csv_path}：{str(e)}")
                continue
        else:
            try:
                df = pd.read_pickle(pkl_path)
                print(f"读取已存在的PKL文件，共 {len(df)} 行: {pkl_path}")
            except Exception as e:
                print(f"读取PKL文件失败 {pkl_path}：{str(e)}")
                continue
        
        
        required_columns = ['sequence', 'reaction_smiles']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"错误：文件缺少必要的列: {missing_cols}")
            continue
        
        
        num_chunks = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
        print(f"将数据分为 {num_chunks} 个批次处理: {base_name}")
        
        
        for batch_i in range(num_chunks):
            if batch_i < start_index:
                continue  
                
            
            start = batch_i * batch_size
            end = start + batch_size
            chunk = df[start:end]
            
            
            processed_chunk = process_dataframe_chunk(chunk)
            
            
            mode = 'wb' if batch_i == start_index else 'ab'
            
            
            with open(output_file_path, mode) as f:
                pickle.dump(processed_chunk, f)
            
            
            del chunk, processed_chunk
            print(f"已完成批次 {batch_i + 1}/{num_chunks} 的处理")
        
        print(f"文件 {base_name} 处理完成，结果保存至：{output_file_path}\n")
        processed_files.append(output_file_path)
    
    return processed_files


def load_processed_data(processed_file_path):
    """加载处理后的数据文件"""
    data_chunks = []
    try:
        with open(processed_file_path, 'rb') as f:
            while True:
                try:
                    chunk = pickle.load(f)
                    data_chunks.append(chunk)
                except EOFError:
                    break
        
        if not data_chunks:
            print(f"警告：处理后的文件为空: {processed_file_path}")
            return None
            
        
        combined_df = pd.concat(data_chunks, ignore_index=True)
        print(f"成功加载处理后的数据，共 {len(combined_df)} 行: {processed_file_path}")
        return combined_df
        
    except Exception as e:
        print(f"加载处理后的数据失败 {processed_file_path}：{str(e)}")
        return None

