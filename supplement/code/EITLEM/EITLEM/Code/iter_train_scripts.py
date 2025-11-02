#from torch import nn
#import sys
#import re
#import torch
#import pandas as pd  # 新增：用于处理和保存预测结果
#import numpy as np   # 确保已导入（用于数值计算）
## -------------------------- 新增：导入sklearn.metrics中的所需函数 --------------------------
#from sklearn.metrics import mean_squared_error, r2_score
#from eitlem_utils import Tester, Trainer, get_pair_info
#from KCM import EitlemKcatPredictor
#from KMP import EitlemKmPredictor
#from ensemble import ensemble
#from torch.utils.tensorboard import SummaryWriter
#from tqdm import tqdm
#from dataset import EitlemDataSet, EitlemDataLoader
#import os
#import shutil
#import argparse
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=所有日志，1=屏蔽INFO，2=屏蔽WARNING，3=屏蔽ERROR以下
#import tensorflow as tf
#
## -------------------------- 1. 重写ModifiedTester类（已修复函数依赖） --------------------------
#class ModifiedTester(object):
#    def __init__(self, device, loss_fn, log10=False):
#        self.device = device
#        self.loss_fn = loss_fn
#        self.log10 = log10
#        self.saved_file_path = None
#
#    def test(self, model, loader, N, desc):
#        testY = []  # 存储测试集真实值
#        testPredict = []  # 存储测试集预测值
#        loss_total = 0
#        model.eval()
#        with torch.no_grad():
#            for data in tqdm(loader, desc=desc, leave=False):
#                # 模型预测
#                pre_value = model(data.to(self.device))
#                # 累加损失
#                if self.loss_fn is not None:
#                    loss_total += self.loss_fn(pre_value, data.value).item()
#                # 收集真实值和预测值（转为CPU列表，避免显存占用）
#                testY.extend(data.value.cpu().tolist())
#                testPredict.extend(pre_value.cpu().tolist())
#        
#        # 标签反归一化（与原代码逻辑一致，确保数值还原为原始尺度）
#        if not self.log10:
#            testY = np.log10(np.power(2, testY))
#            testPredict = np.log10(np.power(2, testPredict))
#        else:
#            testY = np.array(testY)
#            testPredict = np.array(testPredict)
#        
#        # 计算评估指标（此时mean_squared_error和r2_score已导入，可正常调用）
#        MAE = np.abs(testY - testPredict).sum() / N
#        rmse = np.sqrt(mean_squared_error(testY, testPredict))
#        r2 = r2_score(testY, testPredict)
#        
#        # 新增：返回真实值和预测值（供后续保存）
#        return MAE, rmse, r2, loss_total/N, testY, testPredict
#
#    # 保留原save_model方法
#    def save_model(self, model, file_path):
#        torch.save(model.state_dict(), file_path)
#        if self.saved_file_path is not None:
#            os.remove(self.saved_file_path)
#        self.saved_file_path = file_path
#
#
## -------------------------- 2. 其余函数（kineticsTrainer、getPath等）保持不变 --------------------------
#def kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device):
#    print(2)
#    Epoch = 100
#    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
#    Type = "KCAT"  # 注意：原代码硬编码为KCAT，若需支持KM需删除此句
#    
#    # 1. 定义模型和预测结果的保存路径
#    file_model_dir = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/2/Results/KCAT/Weight/'
#    file_result_dir = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/2/Results/KCAT/Predictions/'  # 预测结果目录
#    kkmPath = None  # 原代码硬编码，若需加载预训练模型需删除此句
#
#    # 创建目录（模型目录+预测结果目录）
#    if not os.path.exists(file_model_dir):
#        os.makedirs(file_model_dir)
#    if not os.path.exists(file_result_dir):
#        os.makedirs(file_result_dir)  # 确保预测结果目录存在
#    file_model_prefix = file_model_dir + 'Eitlem_'
#
#    # 2. 模型初始化（保留原逻辑）
#    if kkmPath is not None:
#        print(Type)
#        trained_weights = torch.load(kkmPath)
#        if Type == 'KCAT':
#            model = EitlemKcatPredictor(167 if molType == 'MACCSKeys' else 1024, 512, 1280, 10, 0.5, 10)
#            pretrained_para = {k[5:]: v for k, v in trained_weights.items() if 'kcat' in k and k[5:] in model.state_dict()}
#        else:
#            model = EitlemKmPredictor(167 if molType == 'MACCSKeys' else 1024, 512, 1280, 10, 0.5, 10)
#            pretrained_para = {k[3:]: v for k, v in trained_weights.items() if 'km' in k and k[3:] in model.state_dict()}
#        model.load_state_dict({**model.state_dict(), **pretrained_para})
#    else:
#        print(Type)
#        if Type == 'KCAT':
#            if molType == 'MACCSKeys':
#                model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
#                print(2)
#            else:
#                model = EitlemKcatPredictor(1024, 512, 1280, 10, 0.5, 10)
#        else:
#            if molType == 'MACCSKeys':
#                model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
#            else:
#                model = EitlemKmPredictor(1024, 512, 1280, 10, 0.5, 10)
#    model = model.to(device)
#
#    # 3. 数据加载（保留原逻辑）
#    train_pair_info, test_pair_info = get_pair_info("/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/Feature/2", Type, False)
#    train_set = EitlemDataSet(train_pair_info, f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/Feature/2/esm2_t33_650M_UR50D2/', f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/Feature/2/index_smiles', 1024, 4, log10, molType)
#    test_set = EitlemDataSet(test_pair_info, f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/Feature/2/esm2_t33_650M_UR50D2/', f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/Feature/2/index_smiles', 1024, 4, log10, molType)
#    train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
#    valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
#
#    # 4. 优化器与损失函数（保留原逻辑）
#    if kkmPath is not None:
#        out_param = list(map(id, model.out.parameters()))
#        rest_param = filter(lambda x: id(x) not in out_param, model.parameters())
#        optimizer = torch.optim.AdamW([
#            {'params': rest_param, 'lr': 1e-4},
#            {'params': model.out.parameters(), 'lr': 1e-3}, 
#        ])
#        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
#    else:
#        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)
#    loss_fn = nn.MSELoss()
#
#    # 5. 初始化修改后的Tester（替换原Tester，关键！）
#    tester = ModifiedTester(device, loss_fn, log10=log10)  # 使用新增返回真实值/预测值的Tester
#    trainer = Trainer(device, loss_fn, log10=log10)  # 保留原Trainer
#
#    print("start to train...")
#    writer = SummaryWriter(f'../Results/{Type}/{train_info}/logs/')
#
#    # 6. 训练循环：新增预测结果保存
#    for epoch in range(1, Epoch + 1):
#        # 训练阶段（保留原逻辑）
#        train_MAE, train_rmse, train_r2, loss_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"{Iteration}iter epoch {epoch} train:")
#        
#        # 测试阶段：新增接收真实值（testY）和预测值（testPredict）
#        MAE_dev, RMSE_dev, R2_dev, loss_dev, testY, testPredict = tester.test(
#            model, valid_loader, len(test_pair_info), desc=f"{Iteration}iter epoch {epoch} valid:"
#        )
#
#        # -------------------------- 新增：保存预测结果到CSV --------------------------
#        # 构建结果DataFrame（包含样本索引、真实值、预测值、当前epoch的评估指标）
#        result_df = pd.DataFrame({
#            "sample_index": range(len(testY)),  # 样本索引（对应测试集顺序）
#            "true_value": testY,                # 反归一化后的真实值
#            "predicted_value": testPredict,     # 反归一化后的预测值
#            "epoch": epoch,                     # 训练轮次
#            "MAE": MAE_dev,                     # 该轮次测试集MAE
#            "RMSE": RMSE_dev,                   # 该轮次测试集RMSE
#            "R2": R2_dev                        # 该轮次测试集R2
#        })
#        # 保存路径（包含epoch和molType，便于区分）
#        result_save_path = f"{file_result_dir}epoch_{epoch}_{molType}_test_predictions.csv"
#        result_df.to_csv(result_save_path, index=False, encoding="utf-8")
#        print(f"Epoch {epoch} 测试集预测结果已保存至：{result_save_path}")
#
#        # 7. 原逻辑：学习率更新、日志写入、模型保存
#        scheduler.step()
#        writer.add_scalars("loss", {'train_loss': loss_train, 'dev_loss': loss_dev}, epoch)
#        writer.add_scalars("RMSE", {'train_RMSE': train_rmse, 'dev_RMSE': RMSE_dev}, epoch)
#        writer.add_scalars("MAE", {'train_MAE': train_MAE, 'dev_MAE': MAE_dev}, epoch)
#        writer.add_scalars("R2", {'train_R2': train_r2, 'dev_R2': R2_dev}, epoch)
#        
#        saved_path = file_model_prefix + f'{molType}_trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}'
#        tester.save_model(model, saved_path)
#        print(f"Epoch {epoch} 模型已保存至：{saved_path}\n")
#
#
## -------------------------- 3. 其他函数（getPath、TransferLearing、parse_args、main）保持原逻辑 --------------------------
#def getPath(Type, TrainType, Iteration):
#    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
#    file_model = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/2/Results/{Type}/{train_info}/Weight/'
#    fileList = os.listdir(file_model)
#    return os.path.join(file_model, fileList[0])
#
#def TransferLearing(Iterations, TrainType, log10, molType='MACCSKeys', device=None):
#    for iteration in range(1, Iterations + 1):
#        print(iteration)
#        if iteration == 1:
#            kineticsTrainer(None, TrainType, 'KCAT', iteration, log10, molType, device)
#            # kineticsTrainer(None, TrainType, 'KM', iteration, log10, molType, device)  # 若需支持KM，取消注释
#        else:
#            kkmPath = getPath('KKM', TrainType, iteration-1)
#            kineticsTrainer(kkmPath, TrainType, 'KCAT', iteration, log10, molType, device)
#            # kineticsTrainer(kkmPath, TrainType, 'KM', iteration, log10, molType, device)  # 若需支持KM，取消注释
#        
#        kcatPath = getPath('KCAT', TrainType, iteration)
#        # kmPath = getPath('KM', TrainType, iteration)  # 若需支持KM，取消注释
#        # KKMTrainer(kcatPath, kmPath, TrainType, iteration, log10, molType, device)  # 若需训练KKM，取消注释
#
#def parse_args():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-i', '--Iteration', type=int, required=True)
#    parser.add_argument('-t', '--TrainType', type=str, required=True)
#    parser.add_argument('-l', '--log10', type=bool, required=False, default=True)
#    parser.add_argument('-m', '--molType', type=str, required=False, default='MACCSKeys')
#    parser.add_argument('-d', '--device', type=int, required=True)
#    return parser.parse_args()
#
#if __name__ == '__main__':
#    torch.backends.cudnn.allow_tf32 = True
#    torch.backends.cuda.matmul.allow_tf32 = True
#    args = parse_args()
#    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
#    print(f"use device {device}")
#    TransferLearing(args.Iteration, args.TrainType, args.log10, args.molType, device)

from torch import nn
import sys
import re
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from eitlem_utils import Tester, Trainer, get_pair_info
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor
from ensemble import ensemble
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import EitlemDataSet, EitlemDataLoader
import os
import shutil
import argparse

# 屏蔽TensorFlow日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


# -------------------------- 1. 重写ModifiedTester类 --------------------------
class ModifiedTester(object):
    def __init__(self, device, loss_fn, log10=False):
        self.device = device
        self.loss_fn = loss_fn
        self.log10 = log10
        self.saved_file_path = None

    def test(self, model, loader, N, desc):
        testY = []
        testPredict = []
        loss_total = 0
        model.eval()
        with torch.no_grad():
            for data in tqdm(loader, desc=desc, leave=False):
                pre_value = model(data.to(self.device))
                if self.loss_fn is not None:
                    loss_total += self.loss_fn(pre_value, data.value).item()
                testY.extend(data.value.cpu().tolist())
                testPredict.extend(pre_value.cpu().tolist())

        # 标签反归一化
        if not self.log10:
            testY = np.log10(np.power(2, testY))
            testPredict = np.log10(np.power(2, testPredict))
        else:
            testY = np.array(testY)
            testPredict = np.array(testPredict)

        # 计算评估指标
        MAE = np.abs(testY - testPredict).sum() / N
        rmse = np.sqrt(mean_squared_error(testY, testPredict))
        r2 = r2_score(testY, testPredict)

        return MAE, rmse, r2, loss_total / N, testY, testPredict

    def save_model(self, model, file_path):
        torch.save(model.state_dict(), file_path)
        if self.saved_file_path is not None and os.path.exists(self.saved_file_path):
            os.remove(self.saved_file_path)
        self.saved_file_path = file_path


# -------------------------- 2. kineticsTrainer（新增动态路径参数） --------------------------
def kineticsTrainer(
    kkmPath, TrainType, Type, Iteration, log10, molType, device,
    enzyme_type, num_id  # 新增：循环参数（如cold_enzyme、2）
):
    print(f"\n=== 开始训练：enzyme_type={enzyme_type}, num_id={num_id}, 迭代={Iteration} ===")
    Epoch = 100
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
    Type = "KCAT"  # 固定为KCAT，若需KM可删除此句

    # -------------------------- 动态路径生成（核心修改） --------------------------
    # 基础路径（包含循环参数enzyme_type和num_id）
    base_data_dir = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/turnup/{enzyme_type}/{num_id}/'
    # 模型保存路径
    file_model_dir = f'{base_data_dir}Results/Weight/'
    # 预测结果保存路径
    file_result_dir = f'{base_data_dir}Results/Predictions/'
    # 日志路径
    log_dir = f'{base_data_dir}Results/KCAT/logs/'
    # 数据特征路径
    feature_dir = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/turnup/{enzyme_type}/Feature/{num_id}/'
    esm_dir = f'{feature_dir}esm2_t33_650M_UR50D/'
    index_smiles_path = f'{feature_dir}index_smiles'

    # 创建目录
    for dir_path in [file_model_dir, file_result_dir, log_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    file_model_prefix = file_model_dir + 'Eitlem_'

    # 模型初始化
    if kkmPath is not None and os.path.exists(kkmPath):
        print(f"加载预训练模型：{kkmPath}")
        trained_weights = torch.load(kkmPath)
        model = EitlemKcatPredictor(167 if molType == 'MACCSKeys' else 1024, 512, 1280, 10, 0.5, 10)
        pretrained_para = {k[5:]: v for k, v in trained_weights.items() if 'kcat' in k and k[5:] in model.state_dict()}
        model.load_state_dict({**model.state_dict(), **pretrained_para})
    else:
        print("初始化新模型")
        model = EitlemKcatPredictor(167 if molType == 'MACCSKeys' else 1024, 512, 1280, 10, 0.5, 10)
    model = model.to(device)

    # 数据加载（路径使用动态参数）
    train_pair_info, test_pair_info = get_pair_info(feature_dir, Type, False)
    train_set = EitlemDataSet(
        train_pair_info, esm_dir, index_smiles_path, 1024, 4, log10, molType
    )
    test_set = EitlemDataSet(
        test_pair_info, esm_dir, index_smiles_path, 1024, 4, log10, molType
    )
    train_loader = EitlemDataLoader(
        data=train_set, batch_size=200, shuffle=True, drop_last=False,
        num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True
    )
    valid_loader = EitlemDataLoader(
        data=test_set, batch_size=200, drop_last=False,
        num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True
    )

    # 优化器与损失函数
    if kkmPath is not None:
        out_param = list(map(id, model.out.parameters()))
        rest_param = filter(lambda x: id(x) not in out_param, model.parameters())
        optimizer = torch.optim.AdamW([
            {'params': rest_param, 'lr': 1e-4},
            {'params': model.out.parameters(), 'lr': 1e-3},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)
    loss_fn = nn.MSELoss()

    # 初始化工具类
    tester = ModifiedTester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)
    writer = SummaryWriter(log_dir)

    # 新增：用于跟踪最高的测试集 R2 和对应的模型路径
    best_r2 = -float('inf')
    best_model_path = None

    # 训练循环（每个epoch）
    for epoch in range(1, Epoch + 1):
        # 训练
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(
            model, train_loader, optimizer, len(train_pair_info),
            f"enzyme={enzyme_type}, id={num_id}, 迭代{Iteration}-Epoch{epoch} 训练"
        )
        # 测试
        MAE_dev, RMSE_dev, R2_dev, loss_dev, testY, testPredict = tester.test(
            model, valid_loader, len(test_pair_info),
            f"enzyme={enzyme_type}, id={num_id}, 迭代{Iteration}-Epoch{epoch} 测试"
        )
        scheduler.step()

        # 保存预测结果
        result_df = pd.DataFrame({
            "sample_index": range(len(testY)),
            "true_value": testY.round(4),
            "predicted_value": testPredict.round(4),
            "epoch": epoch,
            "MAE": round(MAE_dev, 4),
            "RMSE": round(RMSE_dev, 4),
            "R2": round(R2_dev, 4)
        })
        result_save_path = f"{file_result_dir}epoch_{epoch}_{molType}_test_predictions.csv"
        result_df.to_csv(result_save_path, index=False, encoding="utf-8")

        # 写入日志
        writer.add_scalars("loss", {'train_loss': loss_train, 'dev_loss': loss_dev}, epoch)
        writer.add_scalars("RMSE", {'train_RMSE': train_rmse, 'dev_RMSE': RMSE_dev}, epoch)
        writer.add_scalars("MAE", {'train_MAE': train_MAE, 'dev_MAE': MAE_dev}, epoch)
        writer.add_scalars("R2", {'train_R2': train_r2, 'dev_R2': R2_dev}, epoch)

        # 新增：比较当前 R2 与最高 R2，若更高则保存当前模型
        if R2_dev > best_r2:
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_r2 = R2_dev
            best_model_path = file_model_prefix + f'{molType}_best_devR2_{R2_dev:.4f}_epoch{epoch}.pth'
            tester.save_model(model, best_model_path)

        print(f"Epoch {epoch} | 测试R2: {R2_dev:.4f} | 结果保存至：{result_save_path}")

    writer.close()
    print(f"=== enzyme_type={enzyme_type}, num_id={num_id}, 迭代={Iteration} 训练结束 ===\n")


# -------------------------- 3. getPath（动态路径适配） --------------------------
def getPath(Type, TrainType, Iteration, enzyme_type, num_id):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
    file_model_dir = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/turnup/{enzyme_type}/{num_id}/Results/Weight/'
    if not os.path.exists(file_model_dir):
        print(f"警告：路径不存在 {file_model_dir}")
        return None
    fileList = os.listdir(file_model_dir)
    if not fileList:
        print(f"警告：目录为空 {file_model_dir}")
        return None
    return os.path.join(file_model_dir, fileList[0])


# -------------------------- 4. 外层循环控制（新增enzyme_type和num_id循环） --------------------------
def run_multi_enzyme(
    Iterations, TrainType, log10, molType='MACCSKeys', device=None,
    enzyme_types=["cold_enzyme"],  # 要循环的enzyme类型列表
    num_ids=[1, 2, 3, 4, 5]  # 要循环的数字编号列表
):
    # 第一层循环：遍历enzyme类型（如cold_enzyme、hot_enzyme等）
    for enzyme_type in enzyme_types:
        # 第二层循环：遍历数字编号（如1、2、3等）
        for num_id in num_ids:
            print(f"\n=====================================")
            print(f"开始处理：enzyme_type={enzyme_type}, num_id={num_id}")
            print(f"=====================================\n")

            # 第三层循环：遍历迭代次数（原TransferLearing逻辑）
            for iteration in range(1, Iterations + 1):
                print(f"---------- 迭代 {iteration}/{Iterations} ----------")
                if iteration == 1:
                    # 第一次迭代：无预训练模型
                    kineticsTrainer(
                        kkmPath=None,
                        TrainType=TrainType,
                        Type='KCAT',
                        Iteration=iteration,
                        log10=log10,
                        molType=molType,
                        device=device,
                        enzyme_type=enzyme_type,  # 传入循环参数
                        num_id=num_id  # 传入循环参数
                    )
                else:
                    # 后续迭代：加载前一轮KKM模型
                    kkmPath = getPath(
                        Type='KKM',
                        TrainType=TrainType,
                        Iteration=iteration - 1,
                        enzyme_type=enzyme_type,  # 传入循环参数
                        num_id=num_id  # 传入循环参数
                    )
                    if kkmPath is None:
                        print(f"迭代{iteration}：未找到前一轮模型，跳过")
                        continue
                    kineticsTrainer(
                        kkmPath=kkmPath,
                        TrainType=TrainType,
                        Type='KCAT',
                        Iteration=iteration,
                        log10=log10,
                        molType=molType,
                        device=device,
                        enzyme_type=enzyme_type,
                        num_id=num_id
                    )

                # 获取当前迭代的模型路径（如需训练KKM可取消注释）
                # kcatPath = getPath('KCAT', TrainType, iteration, enzyme_type, num_id)


# -------------------------- 5. 命令行参数与主函数 --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--Iteration', type=int, required=True, help="总迭代次数")
    parser.add_argument('-t', '--TrainType', type=str, required=True, help="训练类型标识")
    parser.add_argument('-l', '--log10', type=bool, default=True, help="是否log10处理标签")
    parser.add_argument('-m', '--molType', type=str, default='MACCSKeys', help="分子特征类型")
    parser.add_argument('-d', '--device', type=int, required=True, help="GPU设备编号")
    # 新增：循环参数（默认处理cold_enzyme和2，可通过命令行修改）
    parser.add_argument('-e', '--enzyme_types', type=str, nargs='+', default=["cold_enzyme"], help="要循环的enzyme类型列表，如cold_enzyme hot_enzyme")
    parser.add_argument('-n', '--num_ids', type=int, nargs='+', default=[2], help="要循环的数字编号列表，如1 2 3")
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备：{device}")

    # 启动多层循环训练
    run_multi_enzyme(
        Iterations=args.Iteration,
        TrainType=args.TrainType,
        log10=args.log10,
        molType=args.molType,
        device=device,
        enzyme_types=args.enzyme_types,  # 传入enzyme类型列表
        num_ids=args.num_ids  # 传入数字编号列表
    )