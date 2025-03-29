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
    with open("new_data_aug/augmentation_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # ----------------- Define the Copula -------------- ---
    for _index in [1.0]: # 0.05, 0.1, 0.3, 0.5, 
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
        
        # ----------------- Fit the Copula -----------------
        copula = EllipticalCopula(_data.T)
        print('fitting copula of index ', _index)
        copula.fit()
        
        # Sample from the model
        samples, _ = copula.sample(46993*2)
        
        # Drop nan of the row
        samples = samples[~np.isnan(samples).any(axis=1)]
        samples = samples.T
        print(samples.shape)
        
        num_sample = 46993
        _samples = samples[:num_sample]
        
        # Save sampled data as csv
        save_path = os.path.join('new_data_aug/augmented_data', f'copula_generated_data_{_index}.csv')
        _input_column = [f'input_{i}' for i in range(config["FCPflow"]["num_channels"]-2)]
        _output_column = ['output']
        _columns = _input_column + _output_column
        _frame = pd.DataFrame(_samples)
        _frame = pd.concat([_frame], axis=0)
        _frame.columns = _columns
        print(_frame.shape)
        _frame.to_csv(save_path)
        