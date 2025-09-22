

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from drfp import DrfpEncoder
import dill
import pandas as pd
from os.path import join

# 定义一个函数来处理数据
def process_data(file_path, save_path):
    # 读取数据
    all_data = pd.read_pickle(file_path)

    print(f"data_len: {len(all_data)}")
    print(f"文件已加载到data: {file_path}")

    drfp_list = list()
    encoder = DrfpEncoder()

    # 计算 DRFP 指纹
    for reaction_smiles in all_data['reaction_smiles']:
        # 计算 DRFP 指纹
        drfp = encoder.encode([reaction_smiles])[0]
        drfp_list.append(drfp)

    # 将计算得到的列添加到原数据框
    all_data['drfp'] = drfp_list

    # 保存带有 DRFP 的数据
    all_data.to_pickle(save_path)



numbers = [1, 4, 6, 16, 9]
for i in range(1, 2):
    for j in range(1, 2):
        test_file_path = f"/mnt/usb/code/gfy/MyModel/HIS7_esm.pkl"
        test_save_path = f"/mnt/usb/code/gfy/MyModel/HIS7_esm_with_drfp.pkl"
        process_data(test_file_path, test_save_path)
    