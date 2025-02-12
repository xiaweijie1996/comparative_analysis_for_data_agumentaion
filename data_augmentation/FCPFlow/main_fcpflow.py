import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import wandb
import yaml
import pickle
import numpy as np

import alg.models_fcpflow_lin as FCPflows
import tools.tools_train as tl
# import data_process.data_loader as dl

if __name__ == '__main__':
    # Import the configuration file
    with open("data_augmentation/augmentation_config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    # ----------------- Define the FCPflow -----------------
    FCPflow = FCPflows.FCPflow(num_blocks=config["FCPflow"]["num_blocks"],
                            num_channels=config["FCPflow"]["num_channels"],
                            hidden_dim=config["FCPflow"]["hidden_dim"],
                            condition_dim=config["FCPflow"]["condition_dim"],
                            sfactor = config["FCPflow"]["sfactor"])
    print('Number of parameters: ', sum(p.numel() for p in FCPflow.parameters()))
    
    # ---------------Data Process-----------------
    for _index in [0.1, 0.3, 0.5, 0.8, 1.0]:
        with open(config["Path"][f"input_path_{_index}"], 'rb') as file:
            _data = pickle.load(file)
        
        # Split the data into train, validation and test sets
        _data = np.hstack((_data['train_input'], _data['train_output']))
        _data = torch.tensor(_data, dtype=torch.float32)
        _zeros = torch.zeros(_data.shape[0], 2)
        _data = torch.cat((_data, _zeros), dim=1)
        print(_data.shape)
        
        # Define the data loader
        loader, _scaler = tl.create_data_loader(_data, config["FCPflow"]["batch_size"])

        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ----------------- Train Model -----------------
        optimizer = torch.optim.Adam(FCPflow.parameters(), lr=config["FCPflow"]["lr_max"], weight_decay=config["FCPflow"]["w_decay"])
        
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, step_size_up=config["FCPflow"]["lr_step_size"], 
                                                        base_lr=config["FCPflow"]["lr_min"], max_lr=config["FCPflow"]["lr_max"],
                                                        cycle_momentum=False)
        
        tl.train(FCPflow, loader, optimizer, config["FCPflow"]["num_epochs"],
                config["FCPflow"]["condition_dim"], device, _scaler, loader, scheduler, 
                _index, _wandb=False, _save=True, _plot=True)

        print("Training completed successfully!")
        