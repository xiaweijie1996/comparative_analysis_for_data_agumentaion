import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import alg.gmm_model as gmm_piplie
import tools.tools_gmm as tg

if __name__ == '__main__':
    # Import the configuration file
    with open("new_data_aug/augmentation_config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    # ----------------- Define the GMM and Scaler -----------------
    gmm = gmm_piplie.GMmodel(n_iter=config["GMM"]["n_iter"], 
                             tol=config["GMM"]["tol"], 
                             covariance_type=config["GMM"]["covariance_type"],
                             n_component= 5
                            )
    
    # --------------------- Data Process -----------------
    for _index in [1.0]:

        # ---------------Data Process-----------------
        _data_path = config["Path"][f"input_path_{_index}"]  
        _data = pd.read_csv(_data_path, index_col=0)
        _data = _data.values
        
        # Drop the nan
        _data = _data[~np.isnan(_data).any(axis=1)]
        
        _test_path = 'dsets/test_set_wind.csv'
        _test_data = pd.read_csv(_test_path, index_col=0)
        _test_data = _test_data.values
        _test_data = _test_data[~np.isnan(_test_data).any(axis=1)]
        
        # ----------------- Fit the GMM -----------------
        _data_scaled = gmm._scaler(_data)
        fitted_gmm = gmm.gmm(_data_scaled)
        
        # Output the optimal number of components
        print(f"Optimal number of components: {gmm.n_component}")
        
        # Save the model
        save_path = os.path.join('new_data_aug/GMM/saved_model', f'GMM_model_{_index}.pkl')
        with open(save_path, 'wb') as file:
            pickle.dump(fitted_gmm, file)
        
        # ----------------- Sample and Plot -----------------
        num_sample = 46993
        _samples, _ = fitted_gmm.sample(num_sample)
        _samples = gmm.scaler.inverse_transform(_samples)
        print(_samples.shape)
        
        # Save sampled data as csv
        save_path = os.path.join('new_data_aug/augmented_data', f'gmm_generated_data_{_index}.csv')
        _input_column = [f'input_{i}' for i in range(config["FCPflow"]["num_channels"]-2)]
        _output_column = ['output']
        _columns = _input_column + _output_column
        _frame = pd.DataFrame(_samples)
        _frame = pd.concat([_frame], axis=0)
        _frame.columns = _columns
        print(_frame.shape)
        _frame.to_csv(save_path)

        
        # if _index == 1.0:
        #     num_sample = 1000
        #     _samples, _ = fitted_gmm.sample(num_sample)
        #     _samples = gmm.scaler.inverse_transform(_samples)
            
        #     # Save sampled data as csv
        #     save_path = os.path.join('data_augmentation/augmented_data', f'gmm_generated_data_0.csv')
        #     _input_column = [f'input_{i}' for i in range(input_length)]
        #     _output_column = [f'output_{i}' for i in range(output_length)]
        #     _columns = _input_column + _output_column
        #     _frame = pd.DataFrame(_samples)
        #     _frame.columns = _columns
        #     _frame.to_csv(save_path)
        
        # # Plot the original data and the GMM
        # plt.figure(figsize=(10, 20))
        
        # # Plot the first axis
        # plt.subplot(4, 1, 1)
        # plt.plot(_data[:, :48], label='Original Wind Direction & Speed', c='blue', alpha=0.5)
        
        # # Plot the second axis
        # plt.subplot(4, 1, 2)
        # plt.plot(_samples[:, :48], label='Generated Wind Direction & Speed', c='blue', alpha=0.5)
        
        # # Plot the third axis
        # plt.subplot(4, 1, 3)
        # plt.plot(_data[:, 49:], label='Original Wind Output', c='red', alpha=0.5)
        
        # # Plot the fourth axis
        # plt.subplot(4, 1, 4)
        # plt.plot(_samples[:, 49:], label='Generated Wind Output', c='red', alpha=0.5)
        
        # plt.savefig(f'data_augmentation/GMM/saved_model/Generated Data Comparison_{_index}.png')
        # plt.close()