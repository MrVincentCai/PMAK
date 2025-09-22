import sys
from os.path import join
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Dataset

from Kcat_model import InteractPre

from torch import nn
import pickle
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import random
import os
import torch.nn.init as init

assert torch.cuda.is_available(), "Must have avaliable gpu"


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)

class CFG:

    # Number CUDA Devices:
    gpu_number = torch.cuda.device_count()
    # DEVICE
    DEVICE = torch.device('cuda:0')
    # ====================================================
    EPOCHES = 300
    # learning rate
    lr = 5e-5
    # weight_decay
    weight_decay = 1e-5



def train_model(model, train_data, test_data, CFG, cv=False, fold=0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    criterion = nn.MSELoss()

    history_df = pd.DataFrame(columns=['epoch', 'train_loss', 'mse', 'r2', 'pearson_corr', 'mae'])

    best_r2 = -float('inf')
    patience = 0.02
    stop_training = False

    for epoch in range(CFG.EPOCHES):
        if stop_training:
            print("Training stopped early due to significant decrease in R².")
            break

        loss_total = 0
        model.train()
        for idx in range(len(train_data)):
            reaction, protein, label = train_data[idx]
            reaction = reaction.to(CFG.DEVICE)
            protein = protein.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)

            output = model(reaction, protein)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        avg_train_loss = loss_total / len(train_data)



        print(f'----------------------------Epoch {epoch + 1}, Loss: {avg_train_loss}-------------------------')

        test_loss, mse, r2, pearson_corr, mae = evaluate_model(model, test_data, criterion, CFG)

        if r2 > best_r2:
            print(f'R² improved from {best_r2} to {r2}. Saving model...')
            best_r2 = r2
            if cv:
                torch.save(model.state_dict(), f'./save_model/CV/Fold_{fold}_reaction_cold.pth')
            else:
                torch.save(model.state_dict(), './save_model/test_reaction_cold.pth')
        # elif best_r2 - r2 > patience:
        #     print(f'R² decreased by more than {patience} (from {best_r2} to {r2}). Stopping training.')
        #     stop_training = True
        history_df = history_df.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            # 'test_loss': test_loss,
            'mse': mse,
            'r2': r2,
            'pearson_corr': pearson_corr,
            'mae': mae
        }, ignore_index=True)

    if cv:
        history_df.to_csv(f'./save_result/CV/Fold_{fold}_reaction_cold.csv', index=False)
    else:
        history_df.to_csv('./save_result/test_rezction_cold.csv', index=False)
    print('Training history has been saved')


def evaluate_model(model, test_data, criterion, CFG):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx in range(len(test_data)):
            reaction, protein, label = test_data[idx]
            reaction = reaction.to(CFG.DEVICE)
            protein = protein.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)

            outputs = model(reaction, protein)
            loss = criterion(outputs, label)

            test_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    pearson_corr, _ = pearsonr(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)

    print(f'MSE: {mse}')
    print(f'R²: {r2}')
    print(f'Pearson Correlation Coefficient: {pearson_corr}')
    print(f'MAE: {mae}')

    return test_loss / len(test_data), mse, r2, pearson_corr, mae

def get_results(model, test_data, CFG):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx in range(len(test_data)):
            reaction, protein, label = test_data[idx]
            reaction = reaction.to(CFG.DEVICE)
            protein = protein.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)

            outputs = model(reaction, protein)  # , crafted_feature

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    pearson_corr, _ = pearsonr(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)

    print(f'MSE: {mse}')
    print(f'R²: {r2}')
    print(f'Pearson Correlation Coefficient: {pearson_corr}')
    print(f'MAE: {mae}')

    return mse, r2, pearson_corr, mae, all_preds, all_labels

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        reaction = torch.tensor(self.dataframe.iloc[idx]['rxnfp'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        protein = torch.tensor(self.dataframe.iloc[idx]['uniref50_dim1024'], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.dataframe.iloc[idx]['geomean_kcat'], dtype=torch.float32).unsqueeze(0)


        return reaction, protein, label


# ------------------------------------------------Cold start---------------------------------------
# enzyme cold
# with open(join("..", "..", "data", "kcat_data", "splits", "train_df_kcat.pkl"), 'rb') as f:
#     train_data = pickle.load(f)
# with open(join("..", "..", "data", "kcat_data", "splits", "test_df_kcat.pkl"), 'rb') as f:
#     test_data = pickle.load(f)

# reaction cold
with open(join("..", "..", "data", "kcat_data", "splits", "train_df_kcat_rx.pkl"), 'rb') as f:
    train_data = pickle.load(f)
with open(join("..", "..", "data", "kcat_data", "splits", "test_df_kcat_rx.pkl"), 'rb') as f:
    test_data = pickle.load(f)

# Cross-Validation

# enzyme
# train_indices = list(
#     np.load('../../data/kcat_data/splits/CV_train_indices.npy', allow_pickle=True))
# test_indices = list(np.load('../../data/kcat_data/splits/CV_test_indices.npy', allow_pickle=True))

# reaction
train_indices = list(
    np.load('../../data/kcat_data/splits/CV_train_indices_rx.npy', allow_pickle=True))
test_indices = list(np.load('../../data/kcat_data/splits/CV_test_indices_rx.npy', allow_pickle=True))
# ------------------------------------------------------------------------------------------------


# ---------------------------------warm start-----------------------------------------------------
# for i in range(5):
#     with open(f'../../data/kcat_data/warm-splits/{i}_train_data.pkl', 'rb') as f:
#         train_data = pickle.load(f)
#     with open(f'../../data/kcat_data/warm-splits/{i}_test_data.pkl', 'rb') as f:
#         test_data = pickle.load(f)
# --------------------------------------------------------------------------------------------------



# ----------------------------------------Cross Validation---------------------------------------------
Pearson_CV = []
MSE_CV = []
R2_CV = []
for i in range(5):
    train_index, test_index = train_indices[i], test_indices[i]

    train_fold = train_data.iloc[train_index]
    train_fold = train_fold.reset_index(drop=True)
    train_dataset = CustomDataset(train_fold)

    test_fold = train_data.iloc[test_index]
    test_fold = test_fold.reset_index(drop=True)
    test_dataset = CustomDataset(test_fold)


    print(len(train_dataset))
    print(len(test_dataset))

    model = InteractPre().to(CFG.DEVICE)
    model.load_state_dict(torch.load(f'./save_model/CV/Fold_{i}_reaction_cold.pth'), strict=False)
    mse, r2, pearson_corr, mae, all_preds, all_labels = get_results(model, test_dataset, CFG)
    Pearson_CV.append(pearson_corr)
    MSE_CV.append(mse)
    R2_CV.append(r2)

    # train_model(model, train_dataset, test_dataset, CFG, cv=True, fold=i)


# train_data = CustomDataset(train_data)
# test_data = CustomDataset(test_data)
#
# print("train_data len:", len(train_data))
# print("test_data len:", len(test_data))
#
# model = InteractPre().to(CFG.DEVICE)
#
# model.load_state_dict(torch.load('./save_model/test_enzyme_cold.pth'), strict=False)
#
# mse, r2, pearson_corr, mae, all_preds, all_labels = get_results(model, test_data, CFG)
# sss
#
# train_model(model, train_data, test_data, CFG)
