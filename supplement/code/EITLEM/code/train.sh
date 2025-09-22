#!/bin/bash
# 功能：CatPred基础数据集（train/test）的CSV转PKL、ESM特征计算、模型训练全流程
# 数据集路径
CSV_TRAIN="PMAK_/data/catpred/train"
CSV_TEST="/PMAK_/data/catpred/test"
# 中间文件路径（PKL存储）
PKL_DIR="PMAK_/data/catpred/pkl_files"
PKL_TRAIN="${PKL_DIR}/catpred_train.pkl"
PKL_TEST="${PKL_DIR}/catpred_test.pkl"
# 带ESM特征的输出PKL路径
ESM_PKL_TRAIN="${PKL_DIR}/catpred_train_with_esm.pkl"
ESM_PKL_TEST="${PKL_DIR}/catpred_test_with_esm.pkl"
# 训练参数
TRAIN_SCRIPT="train.py"
ESM_SCRIPT="esm2.py"  # 假设计算ESM特征的脚本为esm2.py
ESM_MODEL="esm2_t33_650M_UR50D"  # ESM模型版本（与文档Section 2.2一致）
OUTPUT_DIR="PMAK_/models/catpred_basic"
BATCH_SIZE=64
EPOCHS=100
LR=0.001
DEVICE=0
MOL_TYPE="MACCSKeys"  # 分子特征类型（与示例一致）

# 创建目录（避免路径不存在）
mkdir -p "${PKL_DIR}"
mkdir -p "${OUTPUT_DIR}"

# -------------------------- 第一步：CSV转PKL --------------------------
echo "1. 开始CSV转PKL（CatPred基础数据集）..."
# 处理训练集CSV
if [ ! -f "${PKL_TRAIN}" ]; then
    python -c "
import pandas as pd
csv_path = '${CSV_TRAIN}'
pkl_path = '${PKL_TRAIN}'
# 读取CSV（根据实际分隔符调整，默认逗号）
df = pd.read_csv(csv_path, sep=',')
# 保存为PKL
df.to_pickle(pkl_path)
print(f'✅ 训练集CSV转PKL完成：{csv_path} → {pkl_path}')
    "
else
    echo "✅ 训练集PKL已存在：${PKL_TRAIN}，跳过转换"
fi

# 处理测试集CSV
if [ ! -f "${PKL_TEST}" ]; then
    python -c "
import pandas as pd
csv_path = '${CSV_TEST}'
pkl_path = '${PKL_TEST}'
df = pd.read_csv(csv_path, sep=',')
df.to_pickle(pkl_path)
print(f'✅ 测试集CSV转PKL完成：{csv_path} → {pkl_path}')
    "
else
    echo "✅ 测试集PKL已存在：${PKL_TEST}，跳过转换"
fi

# -------------------------- 第二步：计算ESM特征 --------------------------
echo -e "\n2. 开始计算ESM特征（使用模型：${ESM_MODEL}）..."
# 处理训练集PKL（添加ESM特征）
if [ ! -f "${ESM_PKL_TRAIN}" ]; then
    python "${ESM_SCRIPT}" "${ESM_MODEL}" "${PKL_TRAIN}" "${ESM_PKL_TRAIN}"
    echo "✅ 训练集ESM特征计算完成：${PKL_TRAIN} → ${ESM_PKL_TRAIN}"
else
    echo "✅ 训练集带ESM特征PKL已存在：${ESM_PKL_TRAIN}，跳过计算"
fi

# 处理测试集PKL（添加ESM特征）
if [ ! -f "${ESM_PKL_TEST}" ]; then
    python "${ESM_SCRIPT}" "${ESM_MODEL}" "${PKL_TEST}" "${ESM_PKL_TEST}"
    echo "✅ 测试集ESM特征计算完成：${PKL_TEST} → ${ESM_PKL_TEST}"
else
    echo "✅ 测试集带ESM特征PKL已存在：${ESM_PKL_TEST}，跳过计算"
fi

# -------------------------- 第三步：模型训练 --------------------------
echo -e "\n3. 开始模型训练（输出目录：${OUTPUT_DIR}）..."
python "${TRAIN_SCRIPT}" \
    --train_pkl "${ESM_PKL_TRAIN}" \
    --test_pkl "${ESM_PKL_TEST}" \
    --output_dir "${OUTPUT_DIR}" \
    --mol_type "${MOL_TYPE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --device "${DEVICE}"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo -e "\n🎉 CatPred基础数据集全流程完成！模型保存于：${OUTPUT_DIR}"
else
    echo -e "\n❌ CatPred基础数据集训练失败，请检查日志"
fi




#!/bin/bash
# 功能：Cold-Enzyme数据集（1-5折）的CSV转PKL、ESM特征计算、模型训练全流程
# 数据集路径前缀
CSV_DIR="E:/PMAK_/data/turnup/cold_enzyme"
# 中间文件路径（PKL存储）
PKL_DIR="E:/PMAK_/data/turnup/cold_enzyme/pkl_files"
# 训练参数
TRAIN_SCRIPT="train.py"
ESM_SCRIPT="esm2.py"
ESM_MODEL="esm2_t33_650M_UR50D"
OUTPUT_ROOT="E:/PMAK_/models/cold_enzyme"
BATCH_SIZE=64
EPOCHS=100
LR=0.001
DEVICE=0
MOL_TYPE="MACCSKeys"
# 折数范围（1-5）
START_FOLD=1
END_FOLD=5

# 创建目录
mkdir -p "${PKL_DIR}"
mkdir -p "${OUTPUT_ROOT}"

# 循环处理每折数据
for ((fold=${START_FOLD}; fold<=${END_FOLD}; fold++)); do
    echo -e "\n=================================================="
    echo "开始处理 Cold-Enzyme 第 ${fold} 折"
    echo "=================================================="
    
    # 1. 定义当前折的文件路径
    CSV_TRAIN="${CSV_DIR}/kcat_train_fold_${fold}_en.csv"
    CSV_TEST="${CSV_DIR}/kcat_val_fold_${fold}_en.csv"
    PKL_TRAIN="${PKL_DIR}/kcat_train_fold_${fold}_en.pkl"
    PKL_TEST="${PKL_DIR}/kcat_val_fold_${fold}_en.pkl"
    ESM_PKL_TRAIN="${PKL_DIR}/kcat_train_fold_${fold}_en_with_esm.pkl"
    ESM_PKL_TEST="${PKL_DIR}/kcat_val_fold_${fold}_en_with_esm.pkl"
    OUTPUT_DIR="${OUTPUT_ROOT}/fold_${fold}"
    
    # 检查原始CSV是否存在
    if [ ! -f "${CSV_TRAIN}" ] || [ ! -f "${CSV_TEST}" ]; then
        echo "❌ 第 ${fold} 折CSV文件缺失（${CSV_TRAIN} 或 ${CSV_TEST}），跳过该折"
        continue
    fi
    mkdir -p "${OUTPUT_DIR}"

    # 2. CSV转PKL
    echo -e "\n1. 第 ${fold} 折：CSV转PKL..."
    # 训练集
    if [ ! -f "${PKL_TRAIN}" ]; then
        python -c "
import pandas as pd
df = pd.read_csv('${CSV_TRAIN}', sep=',')
df.to_pickle('${PKL_TRAIN}')
print(f'✅ 训练集转PKL完成：fold_{fold}')
        "
    else
        echo "✅ 训练集PKL已存在：fold_${fold}"
    fi
    # 测试集
    if [ ! -f "${PKL_TEST}" ]; then
        python -c "
import pandas as pd
df = pd.read_csv('${CSV_TEST}', sep=',')
df.to_pickle('${PKL_TEST}')
print(f'✅ 测试集转PKL完成：fold_{fold}')
        "
    else
        echo "✅ 测试集PKL已存在：fold_${fold}"
    fi

    # 3. 计算ESM特征
    echo -e "\n2. 第 ${fold} 折：计算ESM特征..."
    # 训练集
    if [ ! -f "${ESM_PKL_TRAIN}" ]; then
        python "${ESM_SCRIPT}" "${ESM_MODEL}" "${PKL_TRAIN}" "${ESM_PKL_TRAIN}"
        echo "✅ 训练集ESM特征完成：fold_${fold}"
    else
        echo "✅ 训练集ESM PKL已存在：fold_${fold}"
    fi
    # 测试集
    if [ ! -f "${ESM_PKL_TEST}" ]; then
        python "${ESM_SCRIPT}" "${ESM_MODEL}" "${PKL_TEST}" "${ESM_PKL_TEST}"
        echo "✅ 测试集ESM特征完成：fold_${fold}"
    else
        echo "✅ 测试集ESM PKL已存在：fold_${fold}"
    fi

    # 4. 模型训练
    echo -e "\n3. 第 ${fold} 折：模型训练..."
    python "${TRAIN_SCRIPT}" \
        --train_pkl "${ESM_PKL_TRAIN}" \
        --test_pkl "${ESM_PKL_TEST}" \
        --output_dir "${OUTPUT_DIR}" \
        --mol_type "${MOL_TYPE}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --device "${DEVICE}"

    # 检查训练结果
    if [ $? -eq 0 ]; then
        echo -e "\n🎉 第 ${fold} 折训练完成！模型保存于：${OUTPUT_DIR}"
    else
        echo -e "\n❌ 第 ${fold} 折训练失败，请检查日志"
    fi
done

echo -e "\n=================================================="
echo "Cold-Enzyme 1-5折全流程处理完成！"
echo "=================================================="




#!/bin/bash
# 功能：Cold-Reaction数据集（1-5折）的CSV转PKL、ESM特征计算、模型训练全流程
# 数据集路径前缀
CSV_DIR="E:/PMAK_/data/turnup/cold_reaction"
# 中间文件路径（PKL存储）
PKL_DIR="E:/PMAK_/data/turnup/cold_reaction/pkl_files"
# 训练参数
TRAIN_SCRIPT="train.py"
ESM_SCRIPT="esm2.py"
ESM_MODEL="esm2_t33_650M_UR50D"
OUTPUT_ROOT="E:/PMAK_/models/cold_reaction"
BATCH_SIZE=64
EPOCHS=100
LR=0.001
DEVICE=0
MOL_TYPE="MACCSKeys"
# 折数范围（1-5）
START_FOLD=1
END_FOLD=5

# 创建目录
mkdir -p "${PKL_DIR}"
mkdir -p "${OUTPUT_ROOT}"

# 循环处理每折数据
for ((fold=${START_FOLD}; fold<=${END_FOLD}; fold++)); do
    echo -e "\n=================================================="
    echo "开始处理 Cold-Reaction 第 ${fold} 折"
    echo "=================================================="
    
    # 1. 定义当前折的文件路径
    CSV_TRAIN="${CSV_DIR}/kcat_train_fold_${fold}.csv"
    CSV_TEST="${CSV_DIR}/kcat_val_fold_${fold}.csv"
    PKL_TRAIN="${PKL_DIR}/kcat_train_fold_${fold}.pkl"
    PKL_TEST="${PKL_DIR}/kcat_val_fold_${fold}.pkl"
    ESM_PKL_TRAIN="${PKL_DIR}/kcat_train_fold_${fold}_with_esm.pkl"
    ESM_PKL_TEST="${PKL_DIR}/kcat_val_fold_${fold}_with_esm.pkl"
    OUTPUT_DIR="${OUTPUT_ROOT}/fold_${fold}"
    
    # 检查原始CSV是否存在
    if [ ! -f "${CSV_TRAIN}" ] || [ ! -f "${CSV_TEST}" ]; then
        echo "❌ 第 ${fold} 折CSV文件缺失（${CSV_TRAIN} 或 ${CSV_TEST}），跳过该折"
        continue
    fi
    mkdir -p "${OUTPUT_DIR}"

    # 2. CSV转PKL
    echo -e "\n1. 第 ${fold} 折：CSV转PKL..."
    # 训练集
    if [ ! -f "${PKL_TRAIN}" ]; then
        python -c "
import pandas as pd
df = pd.read_csv('${CSV_TRAIN}', sep=',')
df.to_pickle('${PKL_TRAIN}')
print(f'✅ 训练集转PKL完成：fold_{fold}')
        "
    else
        echo "✅ 训练集PKL已存在：fold_${fold}"
    fi
    # 测试集
    if [ ! -f "${PKL_TEST}" ]; then
        python -c "
import pandas as pd
df = pd.read_csv('${CSV_TEST}', sep=',')
df.to_pickle('${PKL_TEST}')
print(f'✅ 测试集转PKL完成：fold_{fold}')
        "
    else
        echo "✅ 测试集PKL已存在：fold_${fold}"
    fi

    # 3. 计算ESM特征
    echo -e "\n2. 第 ${fold} 折：计算ESM特征..."
    # 训练集
    if [ ! -f "${ESM_PKL_TRAIN}" ]; then
        python "${ESM_SCRIPT}" "${ESM_MODEL}" "${PKL_TRAIN}" "${ESM_PKL_TRAIN}"
        echo "✅ 训练集ESM特征完成：fold_${fold}"
    else
        echo "✅ 训练集ESM PKL已存在：fold_${fold}"
    fi
    # 测试集
    if [ ! -f "${ESM_PKL_TEST}" ]; then
        python "${ESM_SCRIPT}" "${ESM_MODEL}" "${PKL_TEST}" "${ESM_PKL_TEST}"
        echo "✅ 测试集ESM特征完成：fold_${fold}"
    else
        echo "✅ 测试集ESM PKL已存在：fold_${fold}"
    fi

    # 4. 模型训练
    echo -e "\n3. 第 ${fold} 折：模型训练..."
    python "${TRAIN_SCRIPT}" \
        --train_pkl "${ESM_PKL_TRAIN}" \
        --test_pkl "${ESM_PKL_TEST}" \
        --output_dir "${OUTPUT_DIR}" \
        --mol_type "${MOL_TYPE}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --device "${DEVICE}"

    # 检查训练结果
    if [ $? -eq 0 ]; then
        echo -e "\n🎉 第 ${fold} 折训练完成！模型保存于：${OUTPUT_DIR}"
    else
        echo -e "\n❌ 第 ${fold} 折训练失败，请检查日志"
    fi
done

echo -e "\n=================================================="
echo "Cold-Reaction 1-5折全流程处理完成！"
echo "=================================================="




#!/bin/bash
# 功能：Warm数据集（0-4折）的CSV转PKL、ESM特征计算、模型训练全流程
# 数据集路径前缀
CSV_DIR="E:/PMAK_/data/turnup/warm"
# 中间文件路径（PKL存储）
PKL_DIR="E:/PMAK_/data/turnup/warm/pkl_files"
# 训练参数
TRAIN_SCRIPT="train.py"
ESM_SCRIPT="esm2.py"
ESM_MODEL="esm2_t33_650M_UR50D"
OUTPUT_ROOT="E:/PMAK_/models/warm"
BATCH_SIZE=64
EPOCHS=100
LR=0.001
DEVICE=0
MOL_TYPE="MACCSKeys"
# 折数范围（0-4）
START_FOLD=0
END_FOLD=4

# 创建目录
mkdir -p "${PKL_DIR}"
mkdir -p "${OUTPUT_ROOT}"

# 循环处理每折数据
for ((fold=${START_FOLD}; fold<=${END_FOLD}; fold++)); do
    echo -e "\n=================================================="
    echo "开始处理 Warm 第 ${fold} 折"
    echo "=================================================="
    
    # 1. 定义当前折的文件路径
    CSV_TRAIN="${CSV_DIR}/kcat_train_data_${fold}.csv"
    CSV_TEST="${CSV_DIR}/kcat_test_data_${fold}.csv"
    PKL_TRAIN="${PKL_DIR}/kcat_train_data_${fold}.pkl"
    PKL_TEST="${PKL_DIR}/kcat_test_data_${fold}.pkl"
    ESM_PKL_TRAIN="${PKL_DIR}/kcat_train_data_${fold}_with_esm.pkl"
    ESM_PKL_TEST="${PKL_DIR}/kcat_test_data_${fold}_with_esm.pkl"
    OUTPUT_DIR="${OUTPUT_ROOT}/fold_${fold}"
    
    # 检查原始CSV是否存在
    if [ ! -f "${CSV_TRAIN}" ] || [ ! -f "${CSV_TEST}" ]; then
        echo "❌ 第 ${fold} 折CSV文件缺失（${CSV_TRAIN} 或 ${CSV_TEST}），跳过该折"
        continue
    fi
    mkdir -p "${OUTPUT_DIR}"

    # 2. CSV转PKL
    echo -e "\n1. 第 ${fold} 折：CSV转PKL..."
    # 训练集
    if [ ! -f "${PKL_TRAIN}" ]; then
        python -c "
import pandas as pd
df = pd.read_csv('${CSV_TRAIN}', sep=',')
df.to_pickle('${PKL_TRAIN}')
print(f'✅ 训练集转PKL完成：fold_{fold}')
        "
    else
        echo "✅ 训练集PKL已存在：fold_${fold}"
    fi
    # 测试集
    if [ ! -f "${PKL_TEST}" ]; then
        python -c "
import pandas as pd
df = pd.read_csv('${CSV_TEST}', sep=',')
df.to_pickle('${PKL_TEST}')
print(f'✅ 测试集转PKL完成：fold_{fold}')
        "
    else
        echo "✅ 测试集PKL已存在：fold_${fold}"
    fi

    # 3. 计算ESM特征
    echo -e "\n2. 第 ${fold} 折：计算ESM特征..."
    # 训练集
    if [ ! -f "${ESM_PKL_TRAIN}" ]; then
        python "${ESM_SCRIPT}" "${ESM_MODEL}" "${PKL_TRAIN}" "${ESM_PKL_TRAIN}"
        echo "✅ 训练集ESM特征完成：fold_${fold}"
    else
        echo "✅ 训练集ESM PKL已存在：fold_${fold}"
    fi
    # 测试集
    if [ ! -f "${ESM_PKL_TEST}" ]; then
        python "${ESM_SCRIPT}" "${ESM_MODEL}" "${PKL_TEST}" "${ESM_PKL_TEST}"
        echo "✅ 测试集ESM特征完成：fold_${fold}"
    else
        echo "✅ 测试集ESM PKL已存在：fold_${fold}"
    fi

    # 4. 模型训练
    echo -e "\n3. 第 ${fold} 折：模型训练..."
    python "${TRAIN_SCRIPT}" \
        --train_pkl "${ESM_PKL_TRAIN}" \
        --test_pkl "${ESM_PKL_TEST}" \
        --output_dir "${OUTPUT_DIR}" \
        --mol_type "${MOL_TYPE}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --device "${DEVICE}"

    # 检查训练结果
    if [ $? -eq 0 ]; then
        echo -e "\n🎉 第 ${fold} 折训练完成！模型保存于：${OUTPUT_DIR}"
    else
        echo -e "\n❌ 第 ${fold} 折训练失败，请检查日志"
    fi
done

echo -e "\n=================================================="
echo "Warm 0-4折全流程处理完成！"
echo "=================================================="