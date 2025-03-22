import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
import lightgbm as lgb


if __name__ == '__main__':
    # ---------- Load the config -----------------
    with open('exp_pred/pred_config.yaml', 'r') as f:
        pre_config = yaml.safe_load(f)

    num_round = 10
    results = []
    
    for _m in ['DoppelGANger', 'gmm', 'copula', 'flow' ]: 
        for _index in [0.05, 0.1, 0.3, 0.5, 0.8, 1.0]:
            if _m == 'DoppelGANger':
                _path = f'data_augmentation/augmented_data/{int(_index*100)}percent_dict.pkl'
            elif _m == 'gmm':
                _path = f'data_augmentation/augmented_data/{_index*100}percent_dict_gmm.pkl'
            elif _m == 'flow':
                _path = f'data_augmentation/augmented_data/{_index*100}percent_dict_model.pkl'
            elif _m == 'copula':
                _path = f'data_augmentation/augmented_data/{_index*100}percent_dict_copula.pkl'

            # Load training data
            with open(_path, 'rb') as f:
                aug_data = pickle.load(f)
                
            # Load the data from the path
            with open(_path, 'rb') as f:
                aug_data = pickle.load(f)
                keys = list(aug_data.keys())

            train_input, train_output = aug_data[keys[0]], aug_data[keys[1]]
            train_input = train_input.reshape(-1, train_input.shape[-1])
            train_output = train_output.reshape(-1)
            print(train_input.shape, train_output.shape)
            
            # Load test data
            with open('dsets/test_set_wind_processed.pkl', 'rb') as f:
                real_data_test = pickle.load(f)
                
            real_data_test = (real_data_test['input'], real_data_test['output'])

            # Reshape to: (individual samples, features)
            test_input, test_output = real_data_test
            test_input = test_input.reshape(-1, test_input.shape[-1])
            test_output_flat = test_output.reshape(-1)

            train_data = lgb.Dataset(train_input, label=train_output)
            validation_data = lgb.Dataset(test_input, label=test_output_flat, reference=train_data)

            # fit and predict
            param = {'subsample': 0.9, 'random_state': 22, 'num_leaves': 30,
                     'n_estimators': 200, 'max_depth': 13, 'learning_rate': 0.046, 'verbose': -1,
                     'colsample_bytree': 0.7, 'objective': 'regression', 'metric': "mse"}
            bst_lgb = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
            pred_output = bst_lgb.predict(test_input)

            mse = mean_squared_error(test_output_flat, pred_output)
            mae = mean_absolute_error(test_output_flat, pred_output)
            rmse = root_mean_squared_error(test_output_flat, pred_output)

            print(f"{_m} - {_index}: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

            # Append to results
            results.append({
                'Model': _m,
                'Index': _index,
                'MSE': round(mse, 4),
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4)
            })

            # reshape to (samples, timesteps, features) for plotting
            pred_output_pickle = pred_output.reshape(-1, 48, 1)
            test_output = test_output.reshape(-1, 48, 1)
            pred_output = pred_output.reshape(-1, 48, 1)
            print(pred_output_pickle.shape)
            
            with open(f'exp_pred/pred_results/LGB_{_m}_pred_results_{_index}.pickle', 'wb') as f:
                pickle.dump(pred_output_pickle, f)
            
            # Plot 10 subfigures
            plt.figure(figsize=(15, 10))
            for _ in range(10):
                plt.subplot(5, 2, _+1)
                plt.plot(test_output[_].reshape(-1), label='Real')
                plt.plot(pred_output[_].reshape(-1), label='Pred')
                plt.legend()
                plt.title(f'LGB prediction for {_m} with {_index} augmentation')
            plt.savefig(f'exp_pred/pred_results/plots/LGB_{_m}_pred_results_{_index}.png')

            # reshape back so we don't need to reupload the file
            test_output = test_output.reshape(-1)

        results_df = pd.DataFrame(results)
        print(results_df)

        # Optional: Save to CSV
        results_df.to_csv('exp_pred/metrics_summary.csv', index=False)

    
    # ---------- train on read data -----------------
    with open('dsets/train_set_wind_processed.pkl', 'rb') as f:
        real_data_train = pickle.load(f)
    
    real_data_train = (real_data_train['input'], real_data_train['output'])
    train_input, train_output = real_data_train

    # Reshape to: (individual samples, features)
    train_input = train_input.reshape(-1, train_input.shape[-1])
    train_output = train_output.reshape(-1)
    print(train_input.shape, train_output.shape)

    train_data = lgb.Dataset(train_input, label=train_output)
    validation_data = lgb.Dataset(test_input, label=test_output, reference=train_data)

    param = {'subsample': 0.9, 'random_state': 22, 'num_leaves': 30,
             'n_estimators': 200, 'max_depth': 13, 'learning_rate': 0.046, 'verbose': -1,
             'colsample_bytree': 0.7, 'objective': 'regression', 'metric': "mse"}

    bst_lgb = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
    pred_output = bst_lgb.predict(test_input)

    # Save the prediction
    test_output = test_output.reshape(-1, 48, 1)
    pred_output = pred_output.reshape(-1, 48, 1)
    print(pred_output_pickle.shape)

    with open(f'exp_pred/pred_results/LGB_read_data_pred_results.pickle', 'wb') as f:
        pickle.dump(pred_output_pickle, f)
        
    # Plot 10 subfigures
    plt.figure(figsize=(15, 10))
    for _ in range(10):
        plt.subplot(5, 2, _+1)
        plt.plot(test_output[_].reshape(-1), label='Real')
        plt.plot(pred_output[_].reshape(-1), label='Pred')
        plt.legend()
        plt.title(f'LGB prediction for read data')
    plt.savefig(f'exp_pred/pred_results/plots/LGB_read_data.png')