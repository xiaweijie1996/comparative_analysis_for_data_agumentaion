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


from new_exp_pred.eva_prediction import mse_loss, rmse_loss, mae_loss


if __name__ == '__main__':
    # ---------- Load the config -----------------
    with open('new_exp_pred/pred_config.yaml', 'r') as f:
        pre_config = yaml.safe_load(f)

    num_round = 10
    results = []
    
    for _m in ['flow', 'gmm', 'copula','real']:  # 'flow', 'DoppelGANger', 'gmm', 'copula',
        _index = '1.0'
        if _m == 'gmm':
            _path = 'new_data_aug/augmented_data/gmm_generated_data_1.0.csv'
        elif _m == 'flow':
            _path = 'new_data_aug/augmented_data/fcpflow_generated_data_new_1.0.csv'
        elif _m == 'copula':
            _path = 'new_data_aug/augmented_data/copula_generated_data_1.0.csv'
        elif _m == 'real':
            _path = f'dsets/train_set_wind.csv'
        
        aug_data = pd.read_csv(_path, index_col=0)
        aug_data = aug_data.dropna()
        train_input = aug_data.iloc[:, :-1].values
        train_output = aug_data.iloc[:, -1].values
        
        # Load test data
        test_path = f'dsets/test_set_wind.csv'
        real_data_test = pd.read_csv(test_path, index_col=0)
        # drop nan
        real_data_test = real_data_test.dropna()
            
        real_data_test = (real_data_test.iloc[:, :-1].values, real_data_test.iloc[:, -1].values)


        # Reshape to: (individual samples, features)
        test_input, test_output = real_data_test
        # test_input = test_input.reshape(-1, test_input.shape[-1])
        # test_output_flat = test_output.reshape(-1)

        train_data = lgb.Dataset(train_input, label=train_output)
        validation_data = lgb.Dataset(test_input, label=test_output, reference=train_data)

        # fit and predict
        param = {'subsample': 0.9, 'random_state': 22, 'num_leaves': 30,
                    'n_estimators': 10, 'max_depth': 5, 'learning_rate': 0.046, 'verbose': -1,
                    'colsample_bytree': 0.7, 'objective': 'regression', 'metric': "mse"}
        bst_lgb = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
        pred_output = bst_lgb.predict(test_input)

        mse = mse_loss(test_output, pred_output)
        mae = mae_loss(test_output, pred_output)
        rmse = rmse_loss(test_output, pred_output)

        print(f"{_m} - {_index}: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

        # Append to results
        results.append({
            'Model': _m,
            'Index': _index,
            'MSE': round(mse, 4),
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4)
        })
        
        results_df = pd.DataFrame(results)

        # Optional: Save to CSV
        results_df.to_csv('new_exp_pred/metrics_summary_gbm.csv', index=False)


    #     # reshape to (samples, timesteps, features) for plotting
    #     pred_output_pickle = pred_output
    #     test_output = test_output.reshape(-1, 48, 1)
    #     pred_output = pred_output.reshape(-1, 48, 1)
    #     print(pred_output_pickle.shape)
        
    #     with open(f'exp_pred/pred_results/LGB_{_m}_pred_results_{_index}.pickle', 'wb') as f:
    #         pickle.dump(pred_output_pickle, f)
        
    #     # Plot 10 subfigures
    #     plt.figure(figsize=(15, 10))
    #     for _ in range(10):
    #         plt.subplot(5, 2, _+1)
    #         plt.plot(test_output[_].reshape(-1), label='Real')
    #         plt.plot(pred_output[_].reshape(-1), label='Pred')
    #         plt.legend()
    #         plt.title(f'LGB prediction for {_m} with {_index} augmentation')
    #     plt.savefig(f'exp_pred/pred_results/plots/LGB_{_m}_pred_results_{_index}.png')

    #     # reshape back so we don't need to reupload the file
    #     test_output = test_output.reshape(-1)

    # results_df = pd.DataFrame(results)
    # print(results_df)

    # # Optional: Save to CSV
    # results_df.to_csv('exp_pred/metrics_summary.csv', index=False)

    
    # # # ---------- train on read data -----------------
    # # with open('dsets/train_set_wind_processed.pkl', 'rb') as f:
    # #     real_data_train = pickle.load(f)
    
    # # real_data_train = (real_data_train['input'], real_data_train['output'])
    # # train_input, train_output = real_data_train

    # # # Reshape to: (individual samples, features)
    # # train_input = train_input.reshape(-1, train_input.shape[-1])
    # # train_output = train_output.reshape(-1)
    # # print(train_input.shape, train_output.shape)

    # # train_data = lgb.Dataset(train_input, label=train_output)
    # # validation_data = lgb.Dataset(test_input, label=test_output, reference=train_data)

    # # param = {'subsample': 0.9, 'random_state': 22, 'num_leaves': 30,
    # #          'n_estimators': 200, 'max_depth': 13, 'learning_rate': 0.046, 'verbose': -1,
    # #          'colsample_bytree': 0.7, 'objective': 'regression', 'metric': "mse"}

    # # bst_lgb = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
    # # pred_output = bst_lgb.predict(test_input)

    # # # Save the prediction
    # # test_output = test_output.reshape(-1, 48, 1)
    # # pred_output = pred_output.reshape(-1, 48, 1)
    # # print(pred_output_pickle.shape)

    # # with open(f'exp_pred/pred_results/LGB_read_data_pred_results.pickle', 'wb') as f:
    # #     pickle.dump(pred_output_pickle, f)
        
    # # # Plot 10 subfigures
    # # plt.figure(figsize=(15, 10))
    # # for _ in range(10):
    # #     plt.subplot(5, 2, _+1)
    # #     plt.plot(test_output[_].reshape(-1), label='Real')
    # #     plt.plot(pred_output[_].reshape(-1), label='Pred')
    # #     plt.legend()
    # #     plt.title(f'LGB prediction for read data')
    # # plt.savefig(f'exp_pred/pred_results/plots/LGB_read_data.png')