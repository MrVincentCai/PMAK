
import torch
import esm
import pandas as pd
import numpy as np
from tqdm import tqdm  


BATCH_SIZE = 1  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LENGTH = 1022  



model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model = model.eval().to(DEVICE)  
batch_converter = alphabet.get_batch_converter()



def batch_calculate_esm_features(sequences):
    """
    批量计算ESM-1b特征（支持GPU加速）
    参数：
        sequences: 字符串列表（蛋白质序列）
    返回：
        features: numpy数组，形状为 (len(sequences), 1280)
    """
    
    batch_labels = [f"seq_{i}" for i in range(len(sequences))]
    _, _, batch_tokens = batch_converter(list(zip(batch_labels, sequences)))
    batch_tokens = batch_tokens.to(DEVICE)  

    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]  

    
    features = []
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        
        token_rep = token_representations[i, 1: seq_len + 1]
        seq_rep = token_rep.mean(dim=0).cpu().numpy()  
        features.append(seq_rep)

    return np.array(features)




def main():
    
    for i in range(1, 2):
        for j in range(6, 7):
            
            SAVE_PATH = f"/mnt/usb/code/gfy/MyModel/HIS7_esm.pkl"  
            df = pd.read_pickle(f"/mnt/usb/code/gfy/MyModel/HIS7.pkl")
            if "sequence" not in df.columns:
                raise ValueError("缺少'sequence'列")

            
            seq_lengths = df["sequence"].str.len()
            too_long = seq_lengths > MAX_SEQ_LENGTH
            num_too_long = too_long.sum()

            if num_too_long > 0:
                print(f"警告：{num_too_long} 条序列超过ESM-1b最大长度 ({MAX_SEQ_LENGTH}氨基酸)")
                print(f"最长序列长度为 {seq_lengths.max()}")
                print("这些序列将被截断处理")

            
            df["sequence"] = df["sequence"].apply(lambda x: x[:MAX_SEQ_LENGTH] if len(x) > MAX_SEQ_LENGTH else x)

            
            esm_features = []
            total_sequences = len(df["sequence"])
            print(f"开始处理 {total_sequences} 条序列，批次大小={BATCH_SIZE}，设备={DEVICE}")

            
            for k in tqdm(range(0, total_sequences, BATCH_SIZE), desc="处理进度"):
                batch_sequences = df["sequence"].iloc[k: k + BATCH_SIZE].tolist()
                batch_features = batch_calculate_esm_features(batch_sequences)
                esm_features.extend(batch_features)

            
            esm_features = np.array(esm_features)
            print(f"特征提取完成，形状: {esm_features.shape}")

            
            df["ESM1b"] = esm_features.tolist()  

            
            df.to_pickle(SAVE_PATH)
            print(f"处理完成！结果保存至 {SAVE_PATH}")

    


if __name__ == "__main__":
    main()