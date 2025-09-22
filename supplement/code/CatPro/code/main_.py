import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(file_path):
    
    data = pd.read_csv(file_path)

    

    model_predictions = data['pred_log10[kcat(s^-1)]']

    
    average_predictions = model_predictions

    

    actual_results = data['true_log10[kcat(s^-1)]']

    
    correlation, p_value = pearsonr(actual_results, average_predictions)

    
    mse = mean_squared_error(actual_results, average_predictions)

    
    mae = mean_absolute_error(actual_results, average_predictions)

    
    r_squared = r2_score(actual_results, average_predictions)

    print(f"文件路径: {file_path}")
    print(f"皮尔逊相关系数 (r): {correlation}")
    print(f"p 值: {p_value}")
    print(f"均方误差 (MSE): {mse}")
    print(f"平均绝对误差 (MAE): {mae}")
    print(f"决定系数 (R^2): {r_squared}")
    print("-" * 50)


if __name__ == "__main__":
    
    for i in range(1, 6):
         file_path = f'/mnt/usb3/code/gfy/code/CataPro-master/models/kcat_models/splits/{i}/catapro_turnup.csv'
         calculate_metrics(file_path)
         file_path = f'/mnt/usb3/code/gfy/code/CataPro-master/models/kcat_models/splits_enzyme/{i}/catapro_turnup.csv'
         calculate_metrics(file_path)
         file_path = f'/mnt/usb3/code/gfy/code/CataPro-master/models/kcat_models/splits_kcat/{i}/catapro_turnup.csv'
         calculate_metrics(file_path)












    
    
    
    
    