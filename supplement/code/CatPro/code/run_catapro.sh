



BATCH_SIZE=8
DEVICE="cuda:0"
TRAIN_SCRIPT="train.py"


TRAIN_DATA="PMAK_/data/catpred/train"
TEST_DATA="PMAK_/data/catpred/test"
MODEL_DIR="models/catpred/basic"


mkdir -p "$MODEL_DIR"


echo "开始训练CatPred基础数据集..."
python "$TRAIN_SCRIPT" \
    -inp_fpath "$TRAIN_DATA" \
    -test_fpath "$TEST_DATA" \
    -model_dpath "$MODEL_DIR" \
    -batch_size "$BATCH_SIZE" \
    -device "$DEVICE"


if [ $? -eq 0 ]; then
    echo "CatPred基础数据集训练完成，模型保存于: $MODEL_DIR"
else
    echo "CatPred基础数据集训练失败"
fi







BATCH_SIZE=8
DEVICE="cuda:0"
TRAIN_SCRIPT="train.py"
DATA_PREFIX="PMAK_/data/turnup/cold_enzyme"


for i in {1..5}; do
    
    TRAIN_DATA="${DATA_PREFIX}/kcat_train_fold_${i}_en.csv"
    TEST_DATA="${DATA_PREFIX}/kcat_val_fold_${i}_en.csv"
    MODEL_DIR="models/cold_enzyme/fold_${i}"

    
    if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$TEST_DATA" ]; then
        echo "警告: 第${i}折数据文件不存在，跳过"
        continue
    fi

    
    mkdir -p "$MODEL_DIR"

    
    echo "开始训练Cold Enzyme第${i}折..."
    python "$TRAIN_SCRIPT" \
        -inp_fpath "$TRAIN_DATA" \
        -test_fpath "$TEST_DATA" \
        -model_dpath "$MODEL_DIR" \
        -batch_size "$BATCH_SIZE" \
        -device "$DEVICE"

    
    if [ $? -eq 0 ]; then
        echo "Cold Enzyme第${i}折训练完成，模型保存于: $MODEL_DIR"
    else
        echo "Cold Enzyme第${i}折训练失败"
    fi
done

echo "Cold Enzyme所有折训练完成"







BATCH_SIZE=8
DEVICE="cuda:0"
TRAIN_SCRIPT="train.py"
DATA_PREFIX="PMAK_/data/turnup/cold_reaction"


for i in {1..5}; do
    
    TRAIN_DATA="${DATA_PREFIX}/kcat_train_fold_${i}.csv"
    TEST_DATA="${DATA_PREFIX}/kcat_val_fold_${i}.csv"
    MODEL_DIR="models/cold_reaction/fold_${i}"

    
    if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$TEST_DATA" ]; then
        echo "警告: 第${i}折数据文件不存在，跳过"
        continue
    fi

    
    mkdir -p "$MODEL_DIR"

    
    echo "开始训练Cold Reaction第${i}折..."
    python "$TRAIN_SCRIPT" \
        -inp_fpath "$TRAIN_DATA" \
        -test_fpath "$TEST_DATA" \
        -model_dpath "$MODEL_DIR" \
        -batch_size "$BATCH_SIZE" \
        -device "$DEVICE"

    
    if [ $? -eq 0 ]; then
        echo "Cold Reaction第${i}折训练完成，模型保存于: $MODEL_DIR"
    else
        echo "Cold Reaction第${i}折训练失败"
    fi
done

echo "Cold Reaction所有折训练完成"








BATCH_SIZE=8
DEVICE="cuda:0"
TRAIN_SCRIPT="train.py"
DATA_DIR="PMAK_/data/mutation"


TRAIN_DATA="${DATA_DIR}/Ttdata.csv"
MODEL_DIR="models/mutation/Tt"
mkdir -p "$MODEL_DIR"
echo "开始训练Tt突变数据集..."
python "$TRAIN_SCRIPT" \
    -inp_fpath "$TRAIN_DATA" \
    -model_dpath "$MODEL_DIR" \
    -batch_size "$BATCH_SIZE" \
    -device "$DEVICE"


TRAIN_DATA="${DATA_DIR}/Ssdata.csv"
MODEL_DIR="models/mutation/Ss"
mkdir -p "$MODEL_DIR"
echo "开始训练Ss突变数据集..."
python "$TRAIN_SCRIPT" \
    -inp_fpath "$TRAIN_DATA" \
    -model_dpath "$MODEL_DIR" \
    -batch_size "$BATCH_SIZE" \
    -device "$DEVICE"


TRAIN_DATA="${DATA_DIR}/HIS3.csv"
MODEL_DIR="models/mutation/HIS3"
mkdir -p "$MODEL_DIR"
echo "开始训练HIS3突变数据集..."
python "$TRAIN_SCRIPT" \
    -inp_fpath "$TRAIN_DATA" \
    -model_dpath "$MODEL_DIR" \
    -batch_size "$BATCH_SIZE" \
    -device "$DEVICE"


TRAIN_DATA="${DATA_DIR}/Tmdata.csv"
MODEL_DIR="models/mutation/Tm"
mkdir -p "$MODEL_DIR"
echo "开始训练Tm突变数据集..."
python "$TRAIN_SCRIPT" \
    -inp_fpath "$TRAIN_DATA" \
    -model_dpath "$MODEL_DIR" \
    -batch_size "$BATCH_SIZE" \
    -device "$DEVICE"

echo "所有突变数据集训练完成"








BATCH_SIZE=8
DEVICE="cuda:0"
TRAIN_SCRIPT="train.py"
DATA_PREFIX="PMAK_/data/turnup/warm"


for i in {0..4}; do
    
    TRAIN_DATA="${DATA_PREFIX}/kcat_train_data_${i}.csv"
    TEST_DATA="${DATA_PREFIX}/kcat_test_data_${i}.csv"
    MODEL_DIR="models/warm/fold_${i}"

    
    if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$TEST_DATA" ]; then
        echo "警告: 第${i}折数据文件不存在，跳过"
        continue
    fi

    
    mkdir -p "$MODEL_DIR"

    
    echo "开始训练Warm第${i}折..."
    python "$TRAIN_SCRIPT" \
        -inp_fpath "$TRAIN_DATA" \
        -test_fpath "$TEST_DATA" \
        -model_dpath "$MODEL_DIR" \
        -batch_size "$BATCH_SIZE" \
        -device "$DEVICE"

    
    if [ $? -eq 0 ]; then
        echo "Warm第${i}折训练完成，模型保存于: $MODEL_DIR"
    else
        echo "Warm第${i}折训练失败"
    fi
done

echo "Warm所有折训练完成"
