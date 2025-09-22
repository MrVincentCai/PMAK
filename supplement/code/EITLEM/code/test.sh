#!/bin/bash
# 功能：CatPred基础数据集（train/test）的模型测试，输出结果到结构化目录
# 测试参数配置
TEST_SCRIPT="test.py"
DEVICE=0
MOL_TYPE="MACCSKeys"  # 与训练时的分子特征类型一致
# 数据与模型路径（需与训练脚本的输出路径对应）
ESM_PKL_TEST="PMAK_/data/catpred/pkl_files/catpred_test_with_esm.pkl"  # 训练时生成的带ESM特征的测试集PKL
MODEL_PATH="PMAK_/models/catpred_basic/kcat_model_best_best.pt"  # 训练时保存的最优模型
# 结果输出目录
TEST_RESULT_ROOT="./test_results"
OUTPUT_DIR="${TEST_RESULT_ROOT}/catpred_basic"

# -------------------------- 前置检查 --------------------------
echo "=================================================="
echo "开始 CatPred 基础数据集测试"
echo "=================================================="

# 检查测试集PKL是否存在
if [ ! -f "${ESM_PKL_TEST}" ]; then
    echo "❌ 带ESM特征的测试集PKL缺失：${ESM_PKL_TEST}"
    echo "请先运行训练脚本生成该文件！"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "${MODEL_PATH}" ]; then
    echo "❌ 最优模型文件缺失：${MODEL_PATH}"
    echo "请先完成CatPred基础数据集的训练！"
    exit 1
fi

# 创建结果输出目录
mkdir -p "${OUTPUT_DIR}"
echo "✅ 结果输出目录已创建：${OUTPUT_DIR}"

# -------------------------- 执行测试 --------------------------
echo -e "\n开始执行测试（设备：cuda:${DEVICE}）..."
python "${TEST_SCRIPT}" \
    --test_pkl "${ESM_PKL_TEST}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --mol_type "${MOL_TYPE}" \
    --device "${DEVICE}"

# 检查测试结果
if [ $? -eq 0 ]; then
    echo -e "\n🎉 CatPred基础数据集测试完成！"
    echo "测试结果保存于：${OUTPUT_DIR}"
else
    echo -e "\n❌ CatPred基础数据集测试失败，请检查日志"
fi



#!/bin/bash
# 功能：Cold-Enzyme数据集（1-5折）的模型测试，逐折匹配训练时的模型与数据
# 测试参数配置
TEST_SCRIPT="test.py"
DEVICE=0
MOL_TYPE="MACCSKeys"
# 路径前缀（需与训练脚本的路径对应）
ESM_PKL_DIR="E:/PMAK_/data/turnup/cold_enzyme/pkl_files"  # 带ESM特征的PKL存储目录
MODEL_ROOT="E:/PMAK_/models/cold_enzyme"  # 训练时的模型保存根目录
# 结果输出目录
TEST_RESULT_ROOT="./test_results"
OUTPUT_ROOT="${TEST_RESULT_ROOT}/cold_enzyme"
# 折数范围（1-5，与训练一致）
START_FOLD=1
END_FOLD=5

# -------------------------- 前置检查与测试执行 --------------------------
echo "=================================================="
echo "开始 Cold-Enzyme 数据集 1-5折测试"
echo "=================================================="

# 循环处理每折测试
for ((fold=${START_FOLD}; fold<=${END_FOLD}; fold++)); do
    echo -e "\n----------------------------------------"
    echo "正在处理第 ${fold} 折测试"
    echo "----------------------------------------"
    
    # 1. 定义当前折的文件路径
    ESM_PKL_TEST="${ESM_PKL_DIR}/kcat_val_fold_${fold}_en_with_esm.pkl"  # 测试集=训练时的验证集
    MODEL_PATH="${MODEL_ROOT}/fold_${fold}/kcat_model_best_best.pt"  # 第fold折的最优模型
    OUTPUT_DIR="${OUTPUT_ROOT}/fold_${fold}"
    
    # 2. 检查文件是否存在
    if [ ! -f "${ESM_PKL_TEST}" ]; then
        echo "❌ 第 ${fold} 折测试集PKL缺失：${ESM_PKL_TEST}，跳过该折"
        continue
    fi
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "❌ 第 ${fold} 折模型文件缺失：${MODEL_PATH}，跳过该折"
        continue
    fi
    
    # 3. 创建结果目录
    mkdir -p "${OUTPUT_DIR}"
    echo "✅ 第 ${fold} 折结果目录已创建：${OUTPUT_DIR}"
    
    # 4. 执行测试
    echo "开始第 ${fold} 折测试（设备：cuda:${DEVICE}）..."
    python "${TEST_SCRIPT}" \
        --test_pkl "${ESM_PKL_TEST}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --mol_type "${MOL_TYPE}" \
        --device "${DEVICE}"
    
    # 5. 检查测试结果
    if [ $? -eq 0 ]; then
        echo "✅ 第 ${fold} 折测试完成！结果保存于：${OUTPUT_DIR}"
    else
        echo "❌ 第 ${fold} 折测试失败，请检查日志"
    fi
done

echo -e "\n=================================================="
echo "Cold-Enzyme 1-5折测试全部完成！"
echo "所有结果汇总于：${OUTPUT_ROOT}"
echo "=================================================="


#!/bin/bash
# 功能：Cold-Reaction数据集（1-5折）的模型测试，逐折匹配训练时的模型与数据
# 测试参数配置
TEST_SCRIPT="test.py"
DEVICE=0
MOL_TYPE="MACCSKeys"
# 路径前缀（需与训练脚本的路径对应）
ESM_PKL_DIR="E:/PMAK_/data/turnup/cold_reaction/pkl_files"  # 带ESM特征的PKL存储目录
MODEL_ROOT="E:/PMAK_/models/cold_reaction"  # 训练时的模型保存根目录
# 结果输出目录
TEST_RESULT_ROOT="./test_results"
OUTPUT_ROOT="${TEST_RESULT_ROOT}/cold_reaction"
# 折数范围（1-5，与训练一致）
START_FOLD=1
END_FOLD=5

# -------------------------- 前置检查与测试执行 --------------------------
echo "=================================================="
echo "开始 Cold-Reaction 数据集 1-5折测试"
echo "=================================================="

# 循环处理每折测试
for ((fold=${START_FOLD}; fold<=${END_FOLD}; fold++)); do
    echo -e "\n----------------------------------------"
    echo "正在处理第 ${fold} 折测试"
    echo "----------------------------------------"
    
    # 1. 定义当前折的文件路径
    ESM_PKL_TEST="${ESM_PKL_DIR}/kcat_val_fold_${fold}_with_esm.pkl"  # 测试集=训练时的验证集
    MODEL_PATH="${MODEL_ROOT}/fold_${fold}/kcat_model_best_best.pt"  # 第fold折的最优模型
    OUTPUT_DIR="${OUTPUT_ROOT}/fold_${fold}"
    
    # 2. 检查文件是否存在
    if [ ! -f "${ESM_PKL_TEST}" ]; then
        echo "❌ 第 ${fold} 折测试集PKL缺失：${ESM_PKL_TEST}，跳过该折"
        continue
    fi
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "❌ 第 ${fold} 折模型文件缺失：${MODEL_PATH}，跳过该折"
        continue
    fi
    
    # 3. 创建结果目录
    mkdir -p "${OUTPUT_DIR}"
    echo "✅ 第 ${fold} 折结果目录已创建：${OUTPUT_DIR}"
    
    # 4. 执行测试
    echo "开始第 ${fold} 折测试（设备：cuda:${DEVICE}）..."
    python "${TEST_SCRIPT}" \
        --test_pkl "${ESM_PKL_TEST}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --mol_type "${MOL_TYPE}" \
        --device "${DEVICE}"
    
    # 5. 检查测试结果
    if [ $? -eq 0 ]; then
        echo "✅ 第 ${fold} 折测试完成！结果保存于：${OUTPUT_DIR}"
    else
        echo "❌ 第 ${fold} 折测试失败，请检查日志"
    fi
done

echo -e "\n=================================================="
echo "Cold-Reaction 1-5折测试全部完成！"
echo "所有结果汇总于：${OUTPUT_ROOT}"
echo "=================================================="


#!/bin/bash
# 功能：Warm数据集（0-4折）的模型测试，逐折匹配训练时的模型与数据
# 测试参数配置
TEST_SCRIPT="test.py"
DEVICE=0
MOL_TYPE="MACCSKeys"
# 路径前缀（需与训练脚本的路径对应）
ESM_PKL_DIR="E:/PMAK_/data/turnup/warm/pkl_files"  # 带ESM特征的PKL存储目录
MODEL_ROOT="E:/PMAK_/models/warm"  # 训练时的模型保存根目录
# 结果输出目录
TEST_RESULT_ROOT="./test_results"
OUTPUT_ROOT="${TEST_RESULT_ROOT}/warm"
# 折数范围（0-4，与训练一致）
START_FOLD=0
END_FOLD=4

# -------------------------- 前置检查与测试执行 --------------------------
echo "=================================================="
echo "开始 Warm 数据集 0-4折测试"
echo "=================================================="

# 循环处理每折测试
for ((fold=${START_FOLD}; fold<=${END_FOLD}; fold++)); do
    echo -e "\n----------------------------------------"
    echo "正在处理第 ${fold} 折测试（对应训练折数）"
    echo "----------------------------------------"
    
    # 1. 定义当前折的文件路径
    ESM_PKL_TEST="${ESM_PKL_DIR}/kcat_test_data_${fold}_with_esm.pkl"  # 训练时的测试集=当前测试集
    MODEL_PATH="${MODEL_ROOT}/fold_${fold}/kcat_model_best_best.pt"  # 第fold折的最优模型
    OUTPUT_DIR="${OUTPUT_ROOT}/fold_${fold}"
    
    # 2. 检查文件是否存在
    if [ ! -f "${ESM_PKL_TEST}" ]; then
        echo "❌ 第 ${fold} 折测试集PKL缺失：${ESM_PKL_TEST}，跳过该折"
        continue
    fi
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "❌ 第 ${fold} 折模型文件缺失：${MODEL_PATH}，跳过该折"
        continue
    fi
    
    # 3. 创建结果目录
    mkdir -p "${OUTPUT_DIR}"
    echo "✅ 第 ${fold} 折结果目录已创建：${OUTPUT_DIR}"
    
    # 4. 执行测试
    echo "开始第 ${fold} 折测试（设备：cuda:${DEVICE}）..."
    python "${TEST_SCRIPT}" \
        --test_pkl "${ESM_PKL_TEST}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --mol_type "${MOL_TYPE}" \
        --device "${DEVICE}"
    
    # 5. 检查测试结果
    if [ $? -eq 0 ]; then
        echo "✅ 第 ${fold} 折测试完成！结果保存于：${OUTPUT_DIR}"
    else
        echo "❌ 第 ${fold} 折测试失败，请检查日志"
    fi
done

echo -e "\n=================================================="
echo "Warm 0-4折测试全部完成！"
echo "所有结果汇总于：${OUTPUT_ROOT}"
echo "=================================================="