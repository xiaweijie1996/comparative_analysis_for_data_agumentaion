import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

import alg.models_fcpflow_lin as FCPflows
import tools.tools_train as tl

# Import the configuration file
with open("new_data_aug/augmentation_config.yaml", "r") as file:
    config = yaml.safe_load(file)
        
# Device configuration
device = 'cpu'
    
# Define the FCPflow 
FCPflow = FCPflows.FCPflow(num_blocks=config["FCPflow"]["num_blocks"],
                        num_channels=config["FCPflow"]["num_channels"],
                        hidden_dim=config["FCPflow"]["hidden_dim"],
                        condition_dim=config["FCPflow"]["condition_dim"],
                        sfactor = config["FCPflow"]["sfactor"])

FCPflow.to(device)
print('Number of parameters: ', sum(p.numel() for p in FCPflow.parameters()))

parameter_path = 'new_data_aug/FCPFlow/saved_model/FCPflow_model_1.0.pth'

# Load the model
FCPflow.load_state_dict(torch.load(parameter_path, map_location=device))

# Data Process
_data_path = config["Path"][f"input_path_{1.0}"]  
_data = pd.read_csv(_data_path, index_col=0)
_data = _data.values
    
# Split the data into train, validation and test sets
_original_data = _data.copy()
# Drop nans
_data = _data[~np.isnan(_data).any(axis=1)]
_data = torch.tensor(_data, dtype=torch.float32).to(device)
_zeros = torch.randn(_data.shape[0], 2).to(device) # torch.zeros(_data.shape[0], 1)
_data = torch.cat((_data, _zeros), dim=1)
print(_data.shape)

# Define the data loader
loader, _scaler = tl.create_data_loader(_data.cpu(), config["FCPflow"]["batch_size"])

# Load the data
_data_test = 'dsets/test_set_wind.csv'
_data_test = pd.read_csv(_data_test, index_col=0)
_data_test = _data_test.values
_data_test = _data_test[~np.isnan(_data_test).any(axis=1)]
        
# Sampling
FCPflow.eval()
num_samples = _data_test.shape[0] 
cond_test = torch.zeros(num_samples, 1).to(device)
noise = torch.randn(cond_test.shape[0], config["FCPflow"]["num_channels"]).to(device)
gen_test = FCPflow.inverse(noise, cond_test)
gen_test = torch.cat((gen_test, cond_test), dim=1)
gen_test = _scaler.inverse_transform(gen_test.detach().cpu().numpy())
gen_test = gen_test[:,:config["FCPflow"]["num_channels"]-1]

save_path = os.path.join('new_data_aug/augmented_data', f'fcpflow_generated_data_new_{1.0}.csv')
_input_column = [f'input_{i}' for i in range(config["FCPflow"]["num_channels"]-2)]
_output_column = ['output']
_columns = _input_column + _output_column

# Concatenate the _data with gen_test
_frame = pd.DataFrame(gen_test)
_frame = pd.concat([_frame], axis=0)
_frame.columns = _columns
print(_frame.shape)
_frame.to_csv(save_path)