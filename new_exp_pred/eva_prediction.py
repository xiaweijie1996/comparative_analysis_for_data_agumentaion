import pandas as pd
import numpy as np
import pickle

# Define mse loss
def mse_loss(pred, real):
    return np.mean((pred - real) ** 2)

def rmse_loss(pred, real):
    return np.sqrt(mse_loss(pred, real))    

def mae_loss(pred, real):
    return np.mean(np.abs(pred - real))

if __name__ == "__main__":
    gen_models = ['gmm', 'copula', 'flow', 'real'] # 'DoppelGANger',
    pre_models = ['NN']
    indexes = [ 1.0] #0.05,

    # Load test data
    test_path = f'dsets/test_set_wind.csv'
    real_data_test = pd.read_csv(test_path, index_col=0)
    # drop nan
    real_data_test = real_data_test.dropna()
        
    real_data_test = (real_data_test.iloc[:, :-1].values, real_data_test.iloc[:, -1].values)


    csv = pd.DataFrame(columns=['gen_Model', 'pre_Model' , 'Index', 'MAE', 'RMSE'])
    
    # ------------------- Evaluation of models-------------------
    for gen_model in gen_models:
        for _index in indexes:
            try:
                for pre_model in pre_models: # new_exp_pred/pred_results/pred_results_copula_1.0.csv
                    pre_path = f'new_exp_pred/pred_results/pred_results_{gen_model}_{_index}.csv'
                    print(pre_path)
                    _data = pd.read_csv(pre_path, index_col=0)
                    pre_data = _data.iloc[:, -1].values
                    print(f'Model: {gen_model}, Index: {_index}')
                    print(f'Pre_Model: {pre_model}')    
                    print(f'MAE: {mae_loss(pre_data, real_data_test[1])}')
                    print(f'RMSE: {rmse_loss(pre_data, real_data_test[1])}')
                    
                    csv = csv._append({'gen_Model': gen_model, 'pre_Model':pre_model, 'Index': _index, 'MAE': mae_loss(pre_data, real_data_test[1]), 'RMSE': rmse_loss(pre_data, real_data_test[1])}, ignore_index=True)
            except:
                pass
    
    # ----------Add the results of real data to the csv file----------
    try:
        for pre_model in pre_models:
            path = f'exp_pred/pred_results/{pre_model}_real_pred_results.pickle'
            
            with open(path, 'rb') as f:
                pre_data = pickle.load(f)
            
            print(f'Model: Real Data')
            print(f'Pre_Model: {pre_model}')
            print(f'MAE: {mae_loss(pre_data, real_data_test["output"])}')
            print(f'RMSE: {rmse_loss(pre_data, real_data_test["output"])}')
            
            csv = csv._append({'gen_Model': 'Real Data', 'pre_Model': pre_model, 'Index': 'Real Data', 'MAE': mae_loss(pre_data, real_data_test['output']), 'RMSE': rmse_loss(pre_data, real_data_test['output'])}, ignore_index=True)
    except:
        pass
      
    csv.to_csv('new_exp_pred/eva_results.csv', index=False)
                
            
    
