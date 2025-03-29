import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import pandas as pd
import yaml
import pickle
import numpy as np
import wandb
import matplotlib.pyplot as plt

import alg as al
import exp_pred.pred_tool as pt

if __name__ == '__main__':
    # ---------- Load the data -----------------
    # Import the configuration file
    with open('new_exp_pred/pred_config.yaml', 'r') as f:
        pre_config = yaml.load(f)
    
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the data
    for _m in ['flow', 'gmm', 'copula','real']:  # 'flow', 'DoppelGANger', 'gmm', 'copula',
        if _m == 'gmm':
            _path = 'new_data_aug/augmented_data/gmm_generated_data_1.0.csv'
        elif _m == 'flow':
            _path = 'new_data_aug/augmented_data/fcpflow_generated_data_new_1.0.csv'
        elif _m == 'copula':
            _path = 'new_data_aug/augmented_data/copula_generated_data_1.0.csv'
        elif _m == 'real':
            _path = f'dsets/train_set_wind.csv'
        
        # Initialize the wandb
        wandb.init(project='wind_prediction_new')
        
        aug_data = pd.read_csv(_path, index_col=0)
        aug_data = aug_data.dropna()

        train_loader = pt.create_data_loader(aug_data,
                                            batch_size=pre_config['NN']['batch_size'], 
                                            default_length=pre_config['NN']['default_length'],
                                            shuffle=True)
        
        # Load test data
        test_path = f'dsets/test_set_wind.csv'
        real_data_test = pd.read_csv(test_path, index_col=0)
        # drop nan
        real_data_test = real_data_test.dropna()
            
        real_data_test = (real_data_test.iloc[:, :-1].values, real_data_test.iloc[:, -1].values)

        # ---------- Load the model -----------------
        predictor = al.NNpredictor(
                pre_config['NN']['input_dim'],
                pre_config['NN']['output_dim'],
                pre_config['NN']['hidden_dim'],
                pre_config['NN']['n_layers'],
                pre_config['NN']['dropout']
            )
        
        print('Number of parameters: {}'.format(sum(p.numel() for p in predictor.model.parameters())))
        
        
        predictor.model.to(device)

        # ---------- Train the model -----------------
        optimizer = torch.optim.Adam(predictor.model.parameters(), lr=pre_config['NN']['lr'])
        
        pt.train(predictor, train_loader, device, optimizer, 
                    epochs=pre_config['NN']['epochs'], 
                    lr=pre_config['NN']['lr'], _model=_m, _index=1.0,
                    test_set=real_data_test)
        
        # Load the best model
        predictor.model.load_state_dict(torch.load(f'new_exp_pred/nn/saved_model/{_m}_model_1.0.pt'))
        
        # ---------- Test the model -----------------
        predictor.model.eval()

        # Make prediction
        input_data = real_data_test[0]
        input_data = torch.tensor(input_data).to(device)
        target_data = real_data_test[1]
        target_data = torch.tensor(target_data).to(device)
        
        target_data = target_data.float()
        input_data = input_data.float()
        output = predictor.model(input_data)
        output = output.cpu().detach().numpy()
        
        # Save the prediction exp_pred//DoppelGANger_model_0.05.pt
        new_dataframe = np.hstack((input_data.cpu().detach().numpy(), output))
        new_dataframe = pd.DataFrame(new_dataframe)
        new_dataframe.to_csv(f'new_exp_pred/pred_results/pred_results_{_m}_{1.0}.csv', index=False)
                    