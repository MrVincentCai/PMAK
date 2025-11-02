import pandas as pd
import torch
import os
from pathlib import Path
from collections import defaultdict

def process_two_csv(train_csv, test_csv, output_dir):
    """
    处理训练和测试两个CSV文件，生成：
    - 共享：index_seq（蛋白质ID→序列）、index_smiles（分子ID→底物SMILES）、seq_str.fasta
    - 分别生成：KCATTrainPairInfo、KCATTestPairInfo
    """
    # 1. 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{output_dir}")

    # 2. 读取两个CSV文件（处理编码问题）
    def read_csv_safe(path):
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")
    
    df_train = read_csv_safe(train_csv)
    df_test = read_csv_safe(test_csv)
    print(f"读取训练集：{len(df_train)} 条，测试集：{len(df_test)} 条")

    # 3. 检查必要列是否存在
    required_cols = ["sequence", "uniprot", "log10kcat_max", "substrate"]
    missing_train = [col for col in required_cols if col not in df_train.columns]
    missing_test = [col for col in required_cols if col not in df_test.columns]
    if missing_train or missing_test:
        print(f"❌ 训练集缺少列：{missing_train}")
        print(f"❌ 测试集缺少列：{missing_test}")
        return

    # 4. 合并训练和测试集，统一生成蛋白质和分子的ID映射
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # 4.1 蛋白质去重（关键：uniprot为空时用空格代替，确保映射一致）
    # 处理合并表中的uniprot空值：nan→空格
    df_combined["uniprot_filled"] = df_combined["uniprot"].fillna(" ")
    # 生成唯一标识列（用处理后的uniprot_filled）
    df_combined["pro_unique_key"] = df_combined["sequence"] + "_" + df_combined["uniprot_filled"]
    # 去重并保留关键列
    unique_proteins = df_combined.drop_duplicates("pro_unique_key")[
        ["sequence", "uniprot_filled", "pro_unique_key"]
    ].reset_index(drop=True)
    # 生成蛋白质ID映射
    pro_id_map = {row["pro_unique_key"]: idx for idx, row in unique_proteins.iterrows()}
    print(f"合并去重后蛋白质数量：{len(unique_proteins)}")

    # 4.2 分子（底物）去重
    unique_substrates = df_combined["substrate"].drop_duplicates().reset_index(drop=True)
    smi_id_map = {smiles: idx for idx, smiles in unique_substrates.items()}
    print(f"合并去重后底物数量：{len(unique_substrates)}")

    # 5. 生成训练集配对信息 KCATTrainPairInfo
    train_pair_info = []
    for _, row in df_train.iterrows():
        # 处理当前行uniprot空值：nan→空格
        uniprot_val = row["uniprot"] if pd.notna(row["uniprot"]) else " "
        pro_key = str(row["sequence"]) + "_" + str(uniprot_val)
        pro_id = pro_id_map[pro_key]
        smi_id = smi_id_map[row["substrate"]]
        kcat_value = row["log10kcat_max"]
        train_pair_info.append([pro_id, smi_id, kcat_value, [0]])
    train_path = output_dir / "KCATTrainPairInfo"
    torch.save(train_pair_info, train_path)
    print(f"✅ 训练集配对信息：{train_path}（{len(train_pair_info)} 条）")

    # 6. 生成测试集配对信息 KCATTestPairInfo
    test_pair_info = []
    for _, row in df_test.iterrows():
        # 处理当前行uniprot空值：nan→空格（与合并表逻辑一致）
        uniprot_val = row["uniprot"] if pd.notna(row["uniprot"]) else " "
        pro_key = str(row["sequence"]) + "_" + str(uniprot_val)
        pro_id = pro_id_map[pro_key]
        smi_id = smi_id_map[row["substrate"]]
        kcat_value = row["log10kcat_max"]
        test_pair_info.append([pro_id, smi_id, kcat_value, [0]])
    test_path = output_dir / "KCATTestPairInfo"
    torch.save(test_pair_info, test_path)
    print(f"✅ 测试集配对信息：{test_path}（{len(test_pair_info)} 条）")

    # 7. 生成共享的 index_seq（蛋白质ID→序列）
    index_seq = {idx: row["sequence"] for idx, row in unique_proteins.iterrows()}
    seq_path = output_dir / "index_seq"
    torch.save(index_seq, seq_path)
    print(f"✅ 共享蛋白质映射：{seq_path}（{len(index_seq)} 个）")

    # 8. 生成共享的 index_smiles（分子ID→底物SMILES）
    index_smiles = {idx: smiles for idx, smiles in unique_substrates.items()}
    smiles_path = output_dir / "index_smiles"
    torch.save(index_smiles, smiles_path)
    print(f"✅ 共享底物映射：{smiles_path}（{len(index_smiles)} 个）")

    # 9. 生成共享的 seq_str.fasta
    fasta_path = output_dir / "seq_str.fasta"
    with open(fasta_path, "w", encoding="utf-8") as f:
        for idx, row in unique_proteins.iterrows():
            f.write(f">{idx}\n{row['sequence']}\n")
    print(f"✅ 蛋白质FASTA：{fasta_path}（{len(unique_proteins)} 条）")

    # 10. 转换总结
    print("\n=== 转换完成 ===")
    print(f"训练集配对：{len(train_pair_info)} 条 | 测试集配对：{len(test_pair_info)} 条")
    print(f"共享蛋白质：{len(index_seq)} 个 | 共享底物：{len(index_smiles)} 个")

if __name__ == "__main__":
    for i in range(1, 2):
        TRAIN_CSV = f"/mnt/usb3/code/gfy/code/catpred_pipeline/data/CatPred-DB/data/kcat/x_mei/kcat-random_train_en.csv"
        TEST_CSV = f"/mnt/usb3/code/gfy/code/catpred_pipeline/data/CatPred-DB/data/kcat/x_mei/kcat-random_test.csv"  # 确认测试集路径正确
        OUTPUT_DIR = f"/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/turnup/cold_enzyme_test/Feature/{i}"
        
        # 检查文件是否存在
        for path in [TRAIN_CSV, TEST_CSV]:
            if not os.path.exists(path):
                print(f"❌ 错误：文件不存在 → {path}")
                exit(1)
        process_two_csv(TRAIN_CSV, TEST_CSV, OUTPUT_DIR)