import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import yaml
from tqdm import tqdm

import exp_pred.pred_tool as pt

if __name__ == '__main__':
    # ---------- Load the config -----------------
    with open('exp_pred/pred_config.yaml', 'r') as f:
        pre_config = yaml.safe_load(f)

    for _m in ['DoppelGANger', 'gmm', 'flow', 'copula']:
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
                
            # Fit the model
            svr_list = []
            for _ in tqdm(range(train_output.shape[1])):
                svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
                svr.fit(train_input, train_output[:, _])
                svr_list.append(svr)
                
            # Load test data
            with open('dsets/test_set_wind_processed.pkl', 'rb') as f:
                real_data_test = pickle.load(f)
                
            real_data_test = (real_data_test['input'], real_data_test['output'])

            # Make prediction
            test_input, test_output = real_data_test
            test_input = test_input.reshape(test_input.shape[0], -1)
            test_output = test_output.reshape(test_output.shape[0], -1)
            
            pred_output = np.zeros((test_output.shape[0], train_output.shape[1]))
            for _ in tqdm(range(train_output.shape[1])):
                pred_output[:, _] = svr_list[_].predict(test_input)
                
            # Reshape the output
            pred_output_pickle = pred_output.reshape(pred_output.shape[0], 48, 1)
            print(pred_output_pickle.shape)
            
            # Save the prediction
            with open(f'exp_pred/pred_results/SVR_{_m}_pred_results_{_index}.pickle', 'wb') as f:
                pickle.dump(pred_output_pickle, f)
                
            # Plot the prediction
            plt.figure(figsize=(10, 5))
            plt.subplot(test_output[0].T, label='Real')
            plt.subplot(pred_output[0].T, label='Pred')
            plt.legend()
            plt.title(f'SVR prediction for {_m} with {_index} augmentation')
            plt.savefig(f'exp_pred/pred_results/SVR_{_m}_pred_results_{_index}.png')
            plt.close()    
            
            break
        break