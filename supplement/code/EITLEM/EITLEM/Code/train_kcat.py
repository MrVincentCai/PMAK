from torch import nn
import sys
import re
import torch
from eitlem_utils import Tester, Trainer, get_pair_info
from KCM import EitlemKcatPredictor  # 仅保留kcat模型导入
# 注释Km/KKm相关导入（无需用到）
# from KMP import EitlemKmPredictor
# from ensemble import ensemble
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import EitlemDataSet, EitlemDataLoader
import os
import shutil
import argparse


def kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device):
    """保留原kcat训练逻辑，但后续调用时固定Type='KCAT'、kkmPath=None（无迁移）"""
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"

    if os.path.exists(f'../Results/{Type}/{train_info}'):
        return None
    
    # 固定训练轮次：因无迭代，按从scratch训练设置为100轮
    Epoch = 100  # 直接固定为100轮，无需根据迭代次数调整
    
    file_model = f'../Results/{Type}/{train_info}/Weight/'
    
    # 仅保留kcat模型初始化逻辑（删除Km相关代码）
    if Type == 'KCAT':
        if molType == 'MACCSKeys':
            model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
        else:
            model = EitlemKcatPredictor(1024, 512, 1280, 10, 0.5, 10)
    # 注释Km模型初始化（无需用到）
    # else:
    #     if molType == 'MACCSKeys':
    #         model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
    #     else:
    #         model = EitlemKmPredictor(1024, 512, 1280, 10, 0.5, 10)
    
    # 无迁移（kkmPath固定为None），无需加载KKm权重，直接跳过权重迁移逻辑
    # （原迁移相关代码可保留，但因kkmPath=None，实际不会执行）
    if kkmPath is not None:
        trained_weights = torch.load(kkmPath)
        weights = model.state_dict()
        pretrained_para = {k[5:]: v for k, v in trained_weights.items() if 'kcat' in k and k[5:] in weights}
        weights.update(pretrained_para)
        model.load_state_dict(weights)
    
    if not os.path.exists(file_model):
        os.makedirs(file_model)
    file_model += 'Eitlem_'
    """Train setting."""
    # 仅加载kcat的数据集（Type='KCAT'）
    train_pair_info, test_pair_info = get_pair_info("../Data/", Type, False)
    train_set = EitlemDataSet(train_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    test_set = EitlemDataSet(test_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
    valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
    model = model.to(device)
    
    # 无迁移（kkmPath=None），使用从scratch的优化器配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)
    
    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)
    
    print("start to train kcat model...")
    writer = SummaryWriter(f'../Results/{Type}/{train_info}/logs/')
    for epoch in range(1, Epoch + 1):
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"{Iteration}iter epoch {epoch} train:")
        MAE_dev, RMSE_dev, R2_dev, loss_dev = tester.test(model, valid_loader, len(test_pair_info), desc=f"{Iteration}iter epoch {epoch} valid:")
        scheduler.step()
        # 记录kcat训练日志
        writer.add_scalars("loss",{'train_loss':loss_train, 'dev_loss':loss_dev}, epoch)
        writer.add_scalars("RMSE",{'train_RMSE':train_rmse, 'dev_RMSE':RMSE_dev}, epoch)
        writer.add_scalars("MAE",{'train_MAE':train_MAE, 'dev_MAE':MAE_dev}, epoch)
        writer.add_scalars("R2",{'train_R2':train_r2, 'dev_R2':R2_dev}, epoch)
        # 保存kcat模型权重
        tester.save_model(model, file_model+f'{molType}_trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}')
    pass


# 注释KKMTrainer函数（无需训练KKm）
# def KKMTrainer(kcatPath, kmPath, TrainType, Iteration, log10, molType, device):
#     ...（原代码全部注释或删除）


# 简化getPath函数：仅获取kcat模型权重（无需Km/KKm）
def getPath(Type, TrainType, Iteration):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
    file_model = f'../Results/{Type}/{train_info}/Weight/'
    fileList = os.listdir(file_model)
    return os.path.join(file_model, fileList[0])


# 替换原TransferLearing函数：仅训练1次kcat，无迭代
def train_only_kcat(TrainType, log10=False, molType='MACCSKeys', device=None):
    """仅训练1次kcat模型，无迭代、无迁移"""
    Iteration = 1  # 固定迭代次数为1（仅1轮训练）
    Type = 'KCAT'   # 固定训练类型为kcat
    kkmPath = None  # 无迁移，不加载KKm权重
    
    # 直接调用kcat训练（固定参数：无迁移、kcat类型、1次迭代）
    kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device)
    
    # 训练完成后，可打印权重保存路径
    kcat_weight_path = getPath(Type, TrainType, Iteration)
    print(f"kcat model training finished! Weight saved to: {kcat_weight_path}")


def parse_args():
    """简化命令行参数：删除迭代次数（固定为1），保留关键参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--TrainType', type=str, required=True, help="训练任务名（自定义，如 only_kcat_train）")
    parser.add_argument('-l', '--log10', type=bool, required=False, default=True, help="是否对kcat做log10转换（默认True）")
    parser.add_argument('-m', '--molType', type=str, required=False, default='MACCSKeys', help="底物分子指纹类型（MACCSKeys/ECFP/RDKIT，默认MACCSKeys）")
    parser.add_argument('-d', '--device', type=int, required=True, help="GPU编号（如0，无GPU则设-1）")
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    
    # 初始化设备（GPU/CPU）
    if torch.cuda.is_available() and args.device != -1:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    print(f"use device {device}")
    
    # 调用简化后的函数：仅训练kcat
    train_only_kcat(
        TrainType=args.TrainType,
        log10=args.log10,
        molType=args.molType,
        device=device
    )