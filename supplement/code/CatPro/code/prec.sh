#!/bin/bash
# 多数据集批量预测脚本

# 配置基础路径（根据实际环境修改）
BASE_DATA_DIR="PMAK_/data"
BASE_MODEL_DIR="PMAK_/saved_models"
OUTPUT_ROOT="./predictions"  # 所有预测结果的根目录
BATCH_SIZE=64
DEVICE="cuda:0"
PREDICT_SCRIPT="predict.py"  # 预测用的Python脚本

# 创建输出根目录
mkdir -p "$OUTPUT_ROOT"

# -------------------------- CatPred基础数据集预测 --------------------------
echo "===== 开始处理CatPred基础数据集 ====="
dataset="catpred"
data_dir="${BASE_DATA_DIR}/${dataset}"
model_dir="${BASE_MODEL_DIR}/${dataset}"
output_dir="${OUTPUT_ROOT}/${dataset}"
mkdir -p "$output_dir"

# 定义测试文件
test_files=(
    "test"  # CatPred测试数据文件
)

# 处理每个测试文件
for test_file in "${test_files[@]}"; do
    # 构建文件路径
    input_path="${data_dir}/${test_file}"
    output_file="${output_dir}/pred_${test_file}.csv"
    
    # 检查输入文件和模型目录
    if [ ! -f "$input_path" ] && [ ! -d "$input_path" ]; then
        echo "警告: 输入文件 $input_path 不存在，跳过"
        continue
    fi
    if [ ! -d "$model_dir" ]; then
        echo "警告: 模型目录 $model_dir 不存在，跳过"
        continue
    fi
    
    echo "处理 ${dataset}: ${test_file}"
    python "$PREDICT_SCRIPT" \
        -inp_fpath "$input_path" \
        -model_dpath "$model_dir" \
        -batch_size "$BATCH_SIZE" \
        -device "$DEVICE" \
        -out_fpath "$output_file"
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✓ 成功生成预测: $output_file"
    else
        echo "✗ 预测失败: $test_file"
    fi
done


# -------------------------- Cold Enzyme数据集预测（1-5折） --------------------------
echo -e "\n===== 开始处理Cold Enzyme数据集 ====="
dataset="cold_enzyme"
data_prefix="${BASE_DATA_DIR}/turnup/${dataset}"
output_root="${OUTPUT_ROOT}/${dataset}"
mkdir -p "$output_root"

# 循环处理1-5折
for fold in {1..5}; do
    model_dir="${BASE_MODEL_DIR}/${dataset}/model_fold_${fold}"
    test_file="${data_prefix}/kcat_val_fold_${fold}_en.csv"
    output_dir="${output_root}/fold_${fold}"
    mkdir -p "$output_dir"
    output_file="${output_dir}/pred_fold_${fold}.csv"
    
    # 检查文件和模型
    if [ ! -f "$test_file" ]; then
        echo "警告: 第${fold}折测试文件不存在，跳过"
        continue
    fi
    if [ ! -d "$model_dir" ]; then
        echo "警告: 第${fold}折模型目录不存在，跳过"
        continue
    fi
    
    echo "处理 ${dataset} 第${fold}折"
    python "$PREDICT_SCRIPT" \
        -inp_fpath "$test_file" \
        -model_dpath "$model_dir" \
        -batch_size "$BATCH_SIZE" \
        -device "$DEVICE" \
        -out_fpath "$output_file"
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✓ 成功生成预测: $output_file"
    else
        echo "✗ 预测失败: 第${fold}折"
    fi
done


# -------------------------- Cold Reaction数据集预测（1-5折） --------------------------
echo -e "\n===== 开始处理Cold Reaction数据集 ====="
dataset="cold_reaction"
data_prefix="${BASE_DATA_DIR}/turnup/${dataset}"
output_root="${OUTPUT_ROOT}/${dataset}"
mkdir -p "$output_root"

# 循环处理1-5折
for fold in {1..5}; do
    model_dir="${BASE_MODEL_DIR}/${dataset}/model_fold_${fold}"
    test_file="${data_prefix}/kcat_val_fold_${fold}.csv"
    output_dir="${output_root}/fold_${fold}"
    mkdir -p "$output_dir"
    output_file="${output_dir}/pred_fold_${fold}.csv"
    
    # 检查文件和模型
    if [ ! -f "$test_file" ]; then
        echo "警告: 第${fold}折测试文件不存在，跳过"
        continue
    fi
    if [ ! -d "$model_dir" ]; then
        echo "警告: 第${fold}折模型目录不存在，跳过"
        continue
    fi
    
    echo "处理 ${dataset} 第${fold}折"
    python "$PREDICT_SCRIPT" \
        -inp_fpath "$test_file" \
        -model_dpath "$model_dir" \
        -batch_size "$BATCH_SIZE" \
        -device "$DEVICE" \
        -out_fpath "$output_file"
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✓ 成功生成预测: $output_file"
    else
        echo "✗ 预测失败: 第${fold}折"
    fi
done


# -------------------------- Warm数据集预测（0-4折） --------------------------
echo -e "\n===== 开始处理Warm数据集 ====="
dataset="warm"
data_prefix="${BASE_DATA_DIR}/turnup/${dataset}"
output_root="${OUTPUT_ROOT}/${dataset}"
mkdir -p "$output_root"

# 循环处理0-4折
for fold in {0..4}; do
    model_dir="${BASE_MODEL_DIR}/${dataset}/model_fold_${fold}"
    test_file="${data_prefix}/kcat_test_data_${fold}.csv"
    output_dir="${output_root}/fold_${fold}"
    mkdir -p "$output_dir"
    output_file="${output_dir}/pred_fold_${fold}.csv"
    
    # 检查文件和模型
    if [ ! -f "$test_file" ]; then
        echo "警告: 第${fold}折测试文件不存在，跳过"
        continue
    fi
    if [ ! -d "$model_dir" ]; then
        echo "警告: 第${fold}折模型目录不存在，跳过"
        continue
    fi
    
    echo "处理 ${dataset} 第${fold}折"
    python "$PREDICT_SCRIPT" \
        -inp_fpath "$test_file" \
        -model_dpath "$model_dir" \
        -batch_size "$BATCH_SIZE" \
        -device "$DEVICE" \
        -out_fpath "$output_file"
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✓ 成功生成预测: $output_file"
    else
        echo "✗ 预测失败: 第${fold}折"
    fi
done


# -------------------------- 突变数据集预测 --------------------------
echo -e "\n===== 开始处理突变数据集 ====="
dataset="mutation"
data_dir="${BASE_DATA_DIR}/${dataset}"
output_root="${OUTPUT_ROOT}/${dataset}"
mkdir -p "$output_root"

# 突变数据集列表
mutation_datasets=(
    "Tt"    # Ttdata.csv
    "Ss"    # Ssdata.csv
    "HIS3"  # HIS3.csv
    "Tm"    # Tmdata.csv
)

# 处理每个突变数据集
for subdataset in "${mutation_datasets[@]}"; do
    model_dir="${BASE_MODEL_DIR}/mutation_${subdataset}"
    test_file="${data_dir}/${subdataset}data.csv"  # 假设测试数据使用原文件
    output_dir="${output_root}/${subdataset}"
    mkdir -p "$output_dir"
    output_file="${output_dir}/pred_${subdataset}.csv"
    
    # 检查文件和模型
    if [ ! -f "$test_file" ]; then
        echo "警告: ${subdataset}测试文件不存在，跳过"
        continue
    fi
    if [ ! -d "$model_dir" ]; then
        echo "警告: ${subdataset}模型目录不存在，跳过"
        continue
    fi
    
    echo "处理 ${dataset}: ${subdataset}"
    python "$PREDICT_SCRIPT" \
        -inp_fpath "$test_file" \
        -model_dpath "$model_dir" \
        -batch_size "$BATCH_SIZE" \
        -device "$DEVICE" \
        -out_fpath "$output_file"
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✓ 成功生成预测: $output_file"
    else
        echo "✗ 预测失败: ${subdataset}"
    fi
done

echo -e "\n所有预测任务处理完成！"
echo "预测结果保存目录: $OUTPUT_ROOT"
