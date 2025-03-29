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
    gen_models = ['DoppelGANger','gmm', 'copula', 'flow', 'real']
    pre_models = ['NN']
    indexes = [0.05, 0.1, 0.3, 0.5, 0.8, 1.0] #0.05,

    real_data_test_path = 'dsets/test_set_wind_processed.pkl'
    with open(real_data_test_path, 'rb') as f:
        real_data_test = pickle.load(f)

    csv = pd.DataFrame(columns=['gen_Model', 'pre_Model' , 'Index', 'MAE', 'RMSE'])
    
    # ------------------- Evaluation of models-------------------
    for gen_model in gen_models:
        for _index in indexes:
            try:
                for pre_model in pre_models:
                    pre_path = f'exp_pred/pred_results/{pre_model}_{gen_model}_pred_results_{_index}.pickle'
                    
                    with open(pre_path, 'rb') as f:
                        pre_data = pickle.load(f)
                        
                    print(f'Model: {gen_model}, Index: {_index}')
                    print(f'Pre_Model: {pre_model}')    
                    print(f'MAE: {mae_loss(pre_data, real_data_test["output"])}')
                    print(f'RMSE: {rmse_loss(pre_data, real_data_test["output"])}')
                    
                    csv = csv._append({'gen_Model': gen_model, 'pre_Model':pre_model, 'Index': _index, 'MAE': mae_loss(pre_data, real_data_test['output']), 'RMSE': rmse_loss(pre_data, real_data_test['output'])}, ignore_index=True)
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
      
    csv.to_csv('exp_pred/eva_results.csv', index=False)
                
            
    
