import numpy as np
import pandas as pd
import pickle
import os
import gc
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# 导入必要的库用于特征提取（与训练时保持一致）
import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import re
from transformers import T5EncoderModel, T5Tokenizer
import requests
from huggingface_hub import configure_http_backend

# 创建自定义HTTP会话并禁用SSL验证（与训练时保持一致）
session = requests.Session()
session.verify = False
configure_http_backend(backend_factory=lambda: session)
requests.packages.urllib3.disable_warnings()

# 确保结果保存目录存在
RESULTS_DIR = '/mnt/usb3/code/gfy/code/catpred_pipeline3/evaluation_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 与训练时相同的特征提取函数
def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('./external/UniKP/vocab.pkl')
    
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm) > 218:
            print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109] + sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('./external/UniKP/trfm_12_23000.pkl'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X

def Seq_to_vec(Sequence):
    # 先创建副本，避免视图/副本问题
    Sequence = Sequence.copy()
    
    for i in range(len(Sequence)):
        if len(Sequence.loc[i]) > 1000:
            Sequence.loc[i] = Sequence.loc[i][:500] + Sequence.loc[i][-500:]
    
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    
    tokenizer = T5Tokenizer.from_pretrained("/mnt/usb/code/gfy/wangye/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("/mnt/usb/code/gfy/wangye/prot_t5_xl_uniref50")
    gc.collect()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    
    features = []
    for i in tqdm(range(len(sequences_Example))):
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
    for i in tqdm(range(len(features))):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    
    return features_normalize

def get_ood_indices(train_clusters, test_clusters):
    return [i for i, cluster in enumerate(test_clusters) if cluster not in train_clusters]

def evaluate_model(model, feature_test, label_test, test_dataset, train_dataset):
    """评估模型在测试集上的性能，返回预测结果和评估指标"""
    # 预测
    pre_label = model.predict(feature_test)
    
    results_dict = {'heldout': {}, 'ood': {}, 'predictions': pre_label}
    
    # 计算heldout性能
    mae = mean_absolute_error(label_test, pre_label)
    r2 = r2_score(label_test, pre_label)
    errors = np.abs(label_test - pre_label)
    p1mag = len(errors[errors < 1]) / len(errors)
    rmse = mean_squared_error(label_test, pre_label, squared=False)
    r = stats.pearsonr(label_test, pre_label)[0]
    
    results_dict['heldout'] = {
        'R2': r2,
        'MAE': mae,
        'p1mag': p1mag,
        'rmse': rmse,
        'r': r
    }
    
    # 计算不同聚类级别的OOD性能
    for N in [99, 80, 60, 40]:
        train_clusters = set(train_dataset[f'sequence_{N}cluster'])
        test_clusters = test_dataset[f'sequence_{N}cluster']
        
        ood_indices = get_ood_indices(train_clusters, test_clusters)
        
        if len(ood_indices) == 0:
            print(f"No OOD samples found for cluster {N}")
            continue
            
        feature_test_ood = feature_test[ood_indices]
        label_test_ood = label_test[ood_indices]
        pre_label_ood = model.predict(feature_test_ood)
        
        mae_ood = mean_absolute_error(label_test_ood, pre_label_ood)
        r2_ood = r2_score(label_test_ood, pre_label_ood)
        errors_ood = np.abs(label_test_ood - pre_label_ood)
        p1mag_ood = len(errors_ood[errors_ood < 1]) / len(errors_ood)
        rmse_ood = mean_squared_error(label_test_ood, pre_label_ood, squared=False)
        r_ood = stats.pearsonr(label_test_ood, pre_label_ood)[0]
        
        results_dict['ood'][f'cluster_{N}'] = {
            'R2': r2_ood,
            'MAE': mae_ood,
            'p1mag': p1mag_ood,
            'rmse': rmse_ood,
            'r': r_ood,
            'indices': ood_indices,
            'predictions': pre_label_ood
        }
    
    return results_dict

def print_evaluation_results(results_dict):
    """打印评估结果"""
    print("模型评估结果:")
    print("==============")
    
    print("\nHeldout Dataset:")
    print("-----------------")
    for metric, value in results_dict['heldout'].items():
        print(f"{metric:<5}: {value:.4f}")
    
    print("\nOOD Datasets:")
    print("--------------")
    for cluster, metrics in results_dict['ood'].items():
        print(f"\n{cluster.upper()}:")
        for metric, value in metrics.items():
            if metric not in ['indices', 'predictions']:  # 不打印索引和预测值
                print(f"{metric:<5}: {value:.4f}")

def save_evaluation_results(model_name, results_dict, test_dataset, label_test):
    """保存评估结果到文件"""
    # 保存评估指标
    metrics_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        # 只保存指标，不保存预测值以减小文件大小
        metrics_only = {
            'heldout': results_dict['heldout'],
            'ood': {k: {m: v for m, v in v.items() if m not in ['indices', 'predictions']} 
                   for k, v in results_dict['ood'].items()}
        }
        pickle.dump(metrics_only, f)
    
    # 保存预测结果到CSV
    predictions_df = test_dataset.copy()
    predictions_df['true_label'] = label_test
    predictions_df['predicted_label'] = results_dict['predictions']
    
    # 添加OOD预测结果
    for cluster, data in results_dict['ood'].items():
        col_name = f'predicted_{cluster}'
        predictions_df[col_name] = None
        if 'indices' in data and 'predictions' in data:
            predictions_df.loc[data['indices'], col_name] = data['predictions']
    
    predictions_path = os.path.join(RESULTS_DIR, f"{model_name}_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    
    print(f"评估结果已保存到: {metrics_path} 和 {predictions_path}")

def load_and_evaluate_model(model_path, feature_test, label_test, test_dataset, train_dataset):
    """加载模型并评估，同时保存结果"""
    print(f"加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 提取模型名称（不含路径和扩展名）
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    print("开始评估模型...")
    results = evaluate_model(model, feature_test, label_test, test_dataset, train_dataset)
    print_evaluation_results(results)
    
    # 保存结果
    save_evaluation_results(model_name, results, test_dataset, label_test)
    
    return model, results

def load_test_data(use_saved_features=True):
    """加载测试数据，可选择使用已保存的特征或重新计算"""
    # 加载测试集数据
    test = pd.read_csv('/mnt/usb3/code/gfy/code/catpred_pipeline3/data/CatPred-DB/data/splits_revision/iteration_4/kcat_test_split_0.19.csv')[:]
    train = pd.read_csv('/mnt/usb3/code/gfy/code/catpred_pipeline3/data/CatPred-DB/data/splits_revision/iteration_4/kcat_train_split_0.81.csv')[:]
    
    sequence_test = test['sequence']
    smiles_test = test['reactant_smiles']
    label_test = np.array(test['log10kcat_max'])
    
    if use_saved_features and os.path.exists('/mnt/usb3/code/gfy/code/catpred_pipeline3/data/external/UniKP/datasets/catpred_kcat/kcat-random_test.pkl'):
        print("使用已保存的测试特征...")
        with open('/mnt/usb3/code/gfy/code/catpred_pipeline3/data/external/UniKP/datasets/catpred_kcat/kcat-random_test.pkl', "rb") as f:
            feature_test = pickle.load(f)
    else:
        print("重新计算测试特征...")
        # 计算SMILES特征
        print("计算SMILES特征...")
        smiles_input_test = smiles_to_vec(smiles_test)
        
        # 计算序列特征
        print("计算序列特征...")
        sequence_input_test = Seq_to_vec(sequence_test)
        
        # 合并特征
        feature_test = np.concatenate((smiles_input_test, sequence_input_test), axis=1)
        
        # 保存特征以备将来使用
        print("保存测试特征...")
        os.makedirs('/mnt/usb3/code/gfy/code/catpred_pipeline3/data/external/UniKP/datasets/catpred_kcat/', exist_ok=True)
        with open('/mnt/usb3/code/gfy/code/catpred_pipeline3/data/external/UniKP/datasets/catpred_kcat/kcat-random_test.pkl', "wb") as f:
            pickle.dump(feature_test, f)
    
    return feature_test, label_test, test, train

def predict_single_sample(model, smiles, sequence, sample_id=None, save=True):
    """使用模型预测单个样本，并可选择保存结果"""
    # 转换为合适的格式
    smiles_input = smiles_to_vec([smiles])
    sequence_series = pd.Series([sequence])
    sequence_input = Seq_to_vec(sequence_series)
    
    # 合并特征
    feature = np.concatenate((smiles_input, sequence_input), axis=1)
    
    # 预测
    prediction = model.predict(feature)
    result = {
        'smiles': smiles,
        'sequence': sequence,
        'prediction': prediction[0],
        'timestamp': pd.Timestamp.now()
    }
    
    if sample_id:
        result['sample_id'] = sample_id
    
    # 保存单个样本的预测结果
    if save:
        single_pred_path = os.path.join(RESULTS_DIR, 'single_predictions.csv')
        # 检查文件是否存在，如果不存在则创建并添加表头
        if not os.path.exists(single_pred_path):
            pd.DataFrame([result]).to_csv(single_pred_path, index=False)
        else:
            pd.DataFrame([result]).to_csv(single_pred_path, mode='a', header=False, index=False)
        print(f"单个样本预测结果已保存到: {single_pred_path}")
    
    return result

def save_average_results(all_results, model_files):
    """保存所有模型的平均性能结果"""
    if len(all_results) <= 1:
        return
    
    # 计算Heldout的平均性能
    heldout_avg = {}
    for metric in all_results[0]['heldout'].keys():
        heldout_avg[metric] = np.mean([res['heldout'][metric] for res in all_results])
    
    # 计算OOD的平均性能
    ood_avg = {}
    for cluster in all_results[0]['ood'].keys():
        cluster_avg = {}
        for metric in all_results[0]['ood'][cluster].keys():
            if metric not in ['indices', 'predictions']:  # 只计算指标的平均值
                cluster_avg[metric] = np.mean([
                    res['ood'][cluster][metric] for res in all_results 
                    if cluster in res['ood']
                ])
        ood_avg[cluster] = cluster_avg
    
    # 保存平均结果
    avg_results = {
        'models_evaluated': model_files,
        'heldout_average': heldout_avg,
        'ood_average': ood_avg,
        'evaluation_time': pd.Timestamp.now()
    }
    
    avg_results_path = os.path.join(RESULTS_DIR, 'average_results.pkl')
    with open(avg_results_path, 'wb') as f:
        pickle.dump(avg_results, f)
    
    # 同时保存为CSV格式以便查看
    avg_df = pd.DataFrame()
    
    # 添加heldout平均结果
    for metric, value in heldout_avg.items():
        avg_df = pd.concat([avg_df, pd.DataFrame({
            'type': 'heldout',
            'cluster': 'average',
            'metric': metric,
            'value': value
        }, index=[0])], ignore_index=True)
    
    # 添加OOD平均结果
    for cluster, metrics in ood_avg.items():
        for metric, value in metrics.items():
            avg_df = pd.concat([avg_df, pd.DataFrame({
                'type': 'ood',
                'cluster': cluster,
                'metric': metric,
                'value': value
            }, index=[0])], ignore_index=True)
    
    avg_csv_path = os.path.join(RESULTS_DIR, 'average_results.csv')
    avg_df.to_csv(avg_csv_path, index=False)
    
    print(f"所有模型的平均性能结果已保存到: {avg_results_path} 和 {avg_csv_path}")

if __name__ == '__main__':
    # 模型保存目录（与训练时相同）
    MODEL_SAVE_DIR = '/mnt/usb3/code/gfy/code/catpred_pipeline3/maranasgroup-CatPred-fabdf38/saved_models'
    
    # 选择要加载的模型（可以是特定模型或所有模型）
    model_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.startswith('extra_trees_model_') and f.endswith('.pkl')]
    model_files.sort()
    
    if not model_files:
        print(f"在目录 {MODEL_SAVE_DIR} 中未找到任何模型文件")
        exit(1)
    
    # 加载测试数据
    feature_test, label_test, test_dataset, train_dataset = load_test_data(use_saved_features=True)
    
    # 评估所有模型
    all_results = []
    for model_file in model_files:
        model_path = os.path.join(MODEL_SAVE_DIR, model_file)
        model, results = load_and_evaluate_model(model_path, feature_test, label_test, test_dataset, train_dataset)
        all_results.append(results)
        print("\n" + "="*50 + "\n")
    
    # 保存平均性能结果
    save_average_results(all_results, model_files)
    
    # 示例：使用第一个模型进行单个样本预测
    if model_files:
        first_model_path = os.path.join(MODEL_SAVE_DIR, model_files[0])
        with open(first_model_path, 'rb') as f:
            first_model = pickle.load(f)
        
        # 替换为实际的SMILES和序列
        sample_smiles = "你的SMILES字符串"
        sample_sequence = "你的蛋白质序列"
        
        # 预测并保存结果
        prediction = predict_single_sample(
            first_model, 
            sample_smiles, 
            sample_sequence, 
            sample_id="example_1", 
            save=True
        )
        print(f"\n单个样本预测结果: {prediction['prediction']}")
