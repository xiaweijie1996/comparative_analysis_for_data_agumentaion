import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import gc
import torch
import wandb
import yaml
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

import alg.models_fcpflow_lin as FCPflows
import tools.tools_train as tl

# import data_process.data_loader as dl
# wandb.login(key='e4dfed43f8b9543d822f5c8501b98aef46a010f1')

if __name__ == '__main__':
    
    # Import the configuration file
    with open("new_data_aug/augmentation_config.yaml", "r") as file:
        config = yaml.safe_load(file)
            
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
    # ----------------- Define the FCPflow -----------------
    FCPflow = FCPflows.FCPflow(num_blocks=config["FCPflow"]["num_blocks"],
                            num_channels=config["FCPflow"]["num_channels"],
                            hidden_dim=config["FCPflow"]["hidden_dim"],
                            condition_dim=config["FCPflow"]["condition_dim"],
                            sfactor = config["FCPflow"]["sfactor"])
    
    FCPflow.to(device)
    print('Number of parameters: ', sum(p.numel() for p in FCPflow.parameters()))
        
    optimizer = torch.optim.Adam(FCPflow.parameters(), lr=config["FCPflow"]["lr_max"]) # weight_decay=config["FCPflow"]["w_decay"]
    
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, step_size_up=config["FCPflow"]["lr_step_size"], 
                                            base_lr=config["FCPflow"]["lr_min"], max_lr=config["FCPflow"]["lr_max"],
                                                       cycle_momentum=False)

    for _index in [1.0]: # 0.05, 0.05, 0.1, 
        
        wandb.init(project="fcpflow_new", name=f"FCPflow_{_index*100}percent", reinit=True)
        # log the number of parameters
        wandb.config.update({"num_parameters": sum(p.numel() for p in FCPflow.parameters())})
        
        # ---------------Data Process-----------------
        _data_path = config["Path"][f"input_path_{_index}"]  
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
        
        _data_test = 'dsets/test_set_wind.csv'
        _data_test = pd.read_csv(_data_test, index_col=0)
        _data_test = _data_test.values
        _data_test = _data_test[~np.isnan(_data_test).any(axis=1)]
        _data_test = torch.tensor(_data_test, dtype=torch.float32)
        print(_data_test.shape)

        # ----------------- Train Model -----------------
        tl.train(FCPflow, loader, optimizer, config["FCPflow"]["num_epochs"],
                config["FCPflow"]["condition_dim"], device, _scaler, _data_test, scheduler, 
                _index, _wandb=True, _save=True, _plot=True)

        print(f"Training completed successfully for index {_index}!")

        # ----------------- Sample-----------------
        FCPflow.eval()
        num_samples = _data_test.shape[0] # 1000 - config['Data_num'][_index]
        cond_test = torch.zeros(num_samples, 1).to(device)
        noise = torch.randn(cond_test.shape[0], config["FCPflow"]["num_channels"]).to(device)
        gen_test = FCPflow.inverse(noise, cond_test)
        gen_test = torch.cat((gen_test, cond_test), dim=1)
        gen_test = _scaler.inverse_transform(gen_test.detach().cpu().numpy())
        gen_test = gen_test[:,:config["FCPflow"]["num_channels"]-1]
        
        save_path = os.path.join('new_data_aug/augmented_data', f'fcpflow_generated_data_new_{_index}.csv')
        _input_column = [f'input_{i}' for i in range(config["FCPflow"]["num_channels"]-2)]
        _output_column = ['output']
        _columns = _input_column + _output_column
        
        # Concatenate the _data with gen_test
        _frame = pd.DataFrame(gen_test)
        _frame = pd.concat([_frame], axis=0)
        _frame.columns = _columns
        print(_frame.shape)
        _frame.to_csv(save_path)
        
        
        # if _index == 1.0:
        #     # num_samples = 1000
        #     cond_test = torch.zeros(num_samples, 1).to(device)
        #     noise = torch.randn(cond_test.shape[0], config["FCPflow"]["num_channels"]).to(device)
        #     gen_test = FCPflow.inverse(noise, cond_test)
        #     gen_test = torch.cat((gen_test, cond_test), dim=1)
        #     gen_test = _scaler.inverse_transform(gen_test.detach().cpu().numpy())
        #     gen_test = gen_test[:,:config["FCPflow"]["num_channels"]-1]
            
        #     save_path = os.path.join('data_augmentation/augmented_data', f'fcpflow_generated_data_0.csv')
        #     _input_column = [f'input_{i}' for i in range(config["FCPflow"]["num_channels"]-49)]
        #     _output_column = [f'output_{i}' for i in range(48)]
        #     _columns = _input_column + _output_column
        #     # Concatenate the _data with gen_test
        #     _frame = pd.DataFrame(gen_test)
        #     _frame.columns = _columns
        #     print(_frame.shape)
        #     _frame.to_csv(save_path)
            