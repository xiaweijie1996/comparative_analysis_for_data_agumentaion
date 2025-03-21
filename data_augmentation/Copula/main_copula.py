import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from alg.multicopula_model import EllipticalCopula

import tools.tools_copula as tc

if __name__ == '__main__':
        # Import the configuration file
    with open("data_augmentation/augmentation_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # ----------------- Define the Copula -------------- ---
    for _index in [0.8, 1.0]: # 0.05, 0.1, 0.3, 0.5, 
        # ---------------Data Process-----------------
        _data_path = config["Path"][f"input_path_{_index}"]  
        data_reshape = tc.Datareshape(_data_path)
        _data = data_reshape.creat_new_frame()
        _data = _data.values
        # Drop the nan
        _data = _data[~np.isnan(_data).any(axis=1)]
        
        # ----------------- Fit the Copula -----------------
        if _data.shape[0] < _data.shape[1]*3:
            _n = int(_data.shape[1] / _data.shape[0]) + 3
            _data_sum = [ _data + np.random.normal(0, 0.01, _data.shape) for _ in range(_n)]
            _data = np.vstack(_data_sum)
            print(_data.shape)
        else:
            print(_data.shape)
        
        copula = EllipticalCopula(_data.T)
        print('fitting copula of index ', _index)
        copula.fit()
        
        # Sample from the model
        samples, _ = copula.sample(1000)
        
        # Drop nan of the row
        samples = samples[~np.isnan(samples).any(axis=1)]
        samples = samples.T
        print(samples.shape)
        
        num_sample = 978
        _samples = samples[:num_sample]
        
        # Save sampled data as csv
        save_path = os.path.join('data_augmentation/augmented_data', f'copula_generated_data_{_index}.csv')
        _input_column = [f'input_{i}' for i in range(config["FCPflow"]["num_channels"]-48)]
        _output_column = [f'output_{i}' for i in range(48)]
        _columns = _input_column + _output_column
        _frame = pd.DataFrame(_samples)
        _frame = pd.concat([_frame], axis=0)
        _frame.columns = _columns
        print(_frame.shape)
        _frame.to_csv(save_path)
        
        # Restor the data into a dictionary
        _data_dict = data_reshape.restor_shape(_frame)
        
        # Save the data into a pickle file
        _paht = f'data_augmentation/augmented_data/{_index*100}percent_dict_copula.pkl'
        with open(_paht, 'wb') as _file:
            pickle.dump(_data_dict, _file)
        