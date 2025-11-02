#import os
#import torch
#from tqdm import tqdm
#cmd = 'python ./extract.py esm2_t33_650M_UR50D /mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/catpred/Feature/seq_str.fasta /mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/catpred/Feature/esm2_t33_650M_UR50D --repr_layers 33 --include per_tok'
#os.system(cmd)
#base = "/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/catpred/Feature/esm2_t33_650M_UR50D/"
#def change(index, layer):
#    data = torch.load(base+f'{index}.pt')
#    data = data['representations'][layer]
#    torch.save(data, base+f'{index}.pt')
#file_list = os.listdir(base)
#length = len(file_list)
#for index in tqdm(range(length)):
#    change(index, 33)

import os
import torch
from tqdm import tqdm

# -------------------------- 核心循环配置 --------------------------
# 要循环的编号（1到5）
num_list = [1]
# 固定酶类型（cold_enzyme）
enzyme_type = "cold_enzyme_test"
# 基础路径模板（用{num}占位编号）
base_path_template = "/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/turnup/{enzyme_type}/Feature/{num}/"


# -------------------------- 1. 定义文件处理函数 --------------------------
def change_esm_file(index, layer, esm_dir):
    """修改esm输出的pt文件，只保留指定层的representations"""
    # 拼接完整文件路径
    file_path = os.path.join(esm_dir, f'{index}.pt')
    # 加载并处理数据
    data = torch.load(file_path)
    data = data['representations'][layer]  # 提取指定层特征
    torch.save(data, file_path)  # 覆盖保存


# -------------------------- 2. 外层循环：遍历1到5的编号 --------------------------
for num in num_list:
    print(f"\n=====================================")
    print(f"开始处理编号 {num} 的 {enzyme_type} 数据")
    print(f"=====================================")
    
    # 1. 动态生成当前编号的路径
    current_base_path = base_path_template.format(enzyme_type=enzyme_type, num=num)
    fasta_path = os.path.join(current_base_path, "seq_str.fasta")  # 输入fasta文件
    esm_output_dir = os.path.join(current_base_path, "esm2_t33_650M_UR50D")  # esm输出目录
    
    # 2. 检查输入fasta文件是否存在（避免报错）
    if not os.path.exists(fasta_path):
        print(f"警告：编号{num}的fasta文件不存在 → {fasta_path}，跳过此编号")
        continue
    
    # 3. 创建esm输出目录（若不存在）
    if not os.path.exists(esm_output_dir):
        os.makedirs(esm_output_dir)
        print(f"创建esm输出目录 → {esm_output_dir}")
    
    # 4. 执行extract.py提取esm特征（动态路径）
    print(f"正在运行extract.py提取编号{num}的esm特征...")
    cmd = (
        f'python ./extract.py '
        f'esm2_t33_650M_UR50D '
        f'{fasta_path} '
        f'{esm_output_dir} '
        f'--repr_layers 33 --include per_tok'
    )
    os.system(cmd)  # 执行命令
    print(f"编号{num}的esm特征提取完成 → 输出目录：{esm_output_dir}")
    
    # 5. 处理esm输出的pt文件（只保留33层特征）
    print(f"正在处理编号{num}的pt文件（保留33层特征）...")
    # 获取esm输出目录下的pt文件数量（假设文件名为0.pt、1.pt...）
    file_list = os.listdir(esm_output_dir)
    pt_count = len([f for f in file_list if f.endswith('.pt')])
    
    if pt_count == 0:
        print(f"警告：编号{num}的esm输出目录无pt文件 → {esm_output_dir}，跳过处理")
        continue
    
    # 循环处理每个pt文件（索引0到pt_count-1）
    for index in tqdm(range(pt_count), desc=f"处理编号{num}的pt文件"):
        change_esm_file(index, layer=33, esm_dir=esm_output_dir)
    
    print(f"编号{num}的所有pt文件处理完成！\n")

print("="*50)
print("1-5编号的所有数据处理完成！")
print("="*50)