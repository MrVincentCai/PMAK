





























import pandas as pd
from sklearn.model_selection import KFold
import random


train_file_path = 'kcat_train_reaction.csv'
train_data = pd.read_csv(train_file_path)


train_grouped = train_data.groupby('Reaction ID')


train_groups = list(train_grouped.groups.keys())


random.seed(42)
random.shuffle(train_groups)


kf = KFold(n_splits=5, shuffle=True, random_state=42)


fold = 1
for train_index, val_index in kf.split(train_groups):
    
    train_sub_groups = [train_groups[i] for i in train_index]
    val_sub_groups = [train_groups[i] for i in val_index]

    
    train_sub_data = pd.concat([train_grouped.get_group(group) for group in train_sub_groups])
    val_sub_data = pd.concat([train_grouped.get_group(group) for group in val_sub_groups])

    
    train_sub_data.to_csv(f'train_fold_{fold}_rea.csv', index=False)
    val_sub_data.to_csv(f'val_fold_{fold}_rea.csv', index=False)

    fold += 1
