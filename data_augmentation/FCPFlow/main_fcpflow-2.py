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

import alg.models_fcpflow_lin as FCPflows
import tools.tools_train as tl
# import data_process.data_loader as dl
# wandb.login(key='e4dfed43f8b9543d822f5c8501b98aef46a010f1')

if __name__ == '__main__':
    
    # Import the configuration file
    with open("data_augmentation/augmentation_config.yaml", "r") as file:
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
    # # Load saved model
    # _path = 'data_augmentation/FCPFlow/saved_model/FCPflow_model_0.5.pth'
    # FCPflow.load_state_dict(torch.load(_path))
    
    for _index in [0.05, 0.1, 0.3, 0.5, 0.8, 1.0]: # 0.05, 0.05, 0.1,  0.5, 
        
        wandb.init(project="fcpflow", name=f"FCPflow_{_index*100}percent", reinit=True)
        # log the number of parameters
        wandb.config.update({"num_parameters": sum(p.numel() for p in FCPflow.parameters())})
        
        # ---------------Data Process-----------------
        _data_path = config["Path"][f"input_path_{_index}"]  
        data_reshape = tl.Datareshape(_data_path)
        _data = data_reshape.creat_new_frame()
        _data = _data.values
            
        # Split the data into train, validation and test sets
        _original_data = _data.copy()
        _data = torch.tensor(_data, dtype=torch.float32)
        _zeros = torch.zeros(_data.shape[0], 1)
        _data = torch.cat((_data, _zeros), dim=1)
        # Check if nan exists in the data
        if torch.isnan(_data).any():
            print('Nan exists in the data!')
            # Drop the nan values
            _data = _data[~torch.isnan(_data).any(dim=1)]
        else:
            print('No Nan in the data!')
        
        
        # Define the data loader
        loader, _scaler = tl.create_data_loader(_data, config["FCPflow"]["batch_size"])
        
        # ----------------- Train Model -----------------
        tl.train(FCPflow, loader, optimizer, config["FCPflow"]["num_epochs"],
                config["FCPflow"]["condition_dim"], device, _scaler, loader, scheduler, 
                _index, _wandb=True, _save=True, _plot=True)

        print(f"Training completed successfully for index {_index}!")

        # ----------------- Sample-----------------
        FCPflow.eval()
        num_samples = 978 # 1000 - config['Data_num'][_index]
        cond_test = torch.zeros(num_samples, 1).to(device)
        noise = torch.randn(cond_test.shape[0], config["FCPflow"]["num_channels"]).to(device)
        gen_test = FCPflow.inverse(noise, cond_test)
        gen_test = torch.cat((gen_test, cond_test), dim=1)
        gen_test = _scaler.inverse_transform(gen_test.detach().cpu().numpy())
        gen_test = gen_test[:,:config["FCPflow"]["num_channels"]]
        
        save_path = os.path.join('data_augmentation/augmented_data', f'fcpflow_generated_data_{_index}.csv')
        _input_column = [f'input_{i}' for i in range(config["FCPflow"]["num_channels"]-48)]
        _output_column = [f'output_{i}' for i in range(48)]
        _columns = _input_column + _output_column
        
        # Concatenate the _data with gen_test
        _frame = pd.DataFrame(gen_test)
        _frame = pd.concat([_frame], axis=0)
        _frame.columns = _columns
        print(_frame.shape)
        _frame.to_csv(save_path)
        
        # Restor the data into a dictionary
        _data_dict = data_reshape.restor_shape(_frame)
        
        # Save the data into a pickle file
        _paht = f'data_augmentation/augmented_data/{_index*100}percent_dict_fcpflow.pkl'
        with open(_paht, 'wb') as _file:
            pickle.dump(_data_dict, _file)
        

   