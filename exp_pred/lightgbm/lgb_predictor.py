import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

import lightgbm as lgb

import exp_pred.pred_tool as pt

if __name__ == '__main__':
    # ---------- Load the config -----------------
    with open('exp_pred/pred_config.yaml', 'r') as f:
        pre_config = yaml.safe_load(f)

    num_round = 10
    
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
            train_input = train_input.reshape(train_input.shape[0], -1)
            train_output = train_output.reshape(train_output.shape[0], -1)
            print(train_input.shape, train_output.shape)
            
            # Load test data
            with open('dsets/test_set_wind_processed.pkl', 'rb') as f:
                real_data_test = pickle.load(f)
                
            real_data_test = (real_data_test['input'], real_data_test['output'])

            test_input, test_output = real_data_test
            test_input = test_input.reshape(test_input.shape[0], -1)
            test_output = test_output.reshape(test_output.shape[0], -1)
            
            
            # Fit the model
            bst_list = []
            for _ in tqdm(range(train_output.shape[1])):
                _output = train_output[:, _]
                train_data = lgb.Dataset(train_input, label=_output)
                validation_data = lgb.Dataset(test_input, label=test_output[:,_], reference=train_data)

                param = {'num_leaves': 31, 'objective': 'binary'}
                param['metric'] = 'auc'
                
                bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
                bst_list.append(bst)
                
            # Make prediction
            pred_output = np.zeros((test_output.shape[0], train_output.shape[1]))
            for _ in tqdm(range(train_output.shape[1])):
                pred_output[:, _] = bst_list[_].predict(test_input)
            
            # Save the prediction
            pred_output_pickle = pred_output.reshape(pred_output.shape[0], 48, 1)
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

            break
        break
    
    # ---------- train on read data -----------------
    with open('dsets/train_set_wind_processed.pkl', 'rb') as f:
        real_data_train = pickle.load(f)
    
    real_data_train = (real_data_train['input'], real_data_train['output'])
    train_input, train_output = real_data_train
    train_input = train_input.reshape(train_input.shape[0], -1)
    train_output = train_output.reshape(train_output.shape[0], -1)
    print(train_input.shape, train_output.shape)
    
    # Fit the model
    bst_list = []
    for _ in tqdm(range(train_output.shape[1])):
        _output = train_output[:, _]
        train_data = lgb.Dataset(train_input, label=_output)
        validation_data = lgb.Dataset(test_input, label=test_output[:,_], reference=train_data)

        param = {'num_leaves': 31, 'objective': 'binary'}
        param['metric'] = 'auc'
        
        bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
        bst_list.append(bst)
        
    # Make prediction
    pred_output = np.zeros((test_output.shape[0], train_output.shape[1]))
    for _ in tqdm(range(train_output.shape[1])):
        pred_output[:, _] = bst_list[_].predict(test_input)
        
    # Save the prediction
    pred_output_pickle = pred_output.reshape(pred_output.shape[0], 48, 1)
    
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