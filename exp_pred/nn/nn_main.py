import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import pandas as pd
import yaml
import pickle
import numpy as np
import wandb

import alg as al
import exp_pred.pred_tool as pt

if __name__ == '__main__':
    # ---------- Load the data -----------------
    # Import the configuration file
    with open('exp_pred/pred_config.yaml', 'r') as f:
        pre_config = yaml.load(f)
    
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the data
    for _m in ['DoppelGANger', 'gmm', 'copula', 'flow']:
        for _index in [0.05, 0.1, 0.3, 0.5, 0.8, 1.0]:
            if _m == 'DoppelGANger':
                _path = f'data_augmentation/augmented_data/{int(_index*100)}percent_dict.pkl'
            elif _m == 'gmm':
                _path = f'data_augmentation/augmented_data/{_index*100}percent_dict_gmm.pkl'
            elif _m == 'flow':
                _path = f'data_augmentation/augmented_data/{_index*100}percent_dict_model.pkl'
            elif _m == 'copula':
                _path = f'data_augmentation/augmented_data/{_index*100}percent_dict_copula.pkl'
           
            # Initialize the wandb
            wandb.init(project='wind_prediction')
            
            # Load the data from the path
            with open(_path, 'rb') as f:
                aug_data = pickle.load(f)
                keys = list(aug_data.keys())
                print(keys)
            
            train_loader = pt.create_data_loader(aug_data, keys, 
                                                batch_size=pre_config['NN']['batch_size'], 
                                                default_length=pre_config['NN']['default_length'],
                                                shuffle=True)
            
            # Load test data
            with open('dsets/test_set_wind_processed.pkl', 'rb') as f:
                real_data_test = pickle.load(f)
                
            real_data_test = (real_data_test['input'], real_data_test['output'])

            # ---------- Load the model -----------------
            predictor = al.CNNConvpredictor(
                    in_channels=pre_config['NN']['in_channels'],
                    hidden_channels=pre_config['NN']['hidden_channels'],
                    out_channels=pre_config['NN']['out_channels'],
                    kernel_size=pre_config['NN']['kernel_size'],
                    dropout=pre_config['NN']['dropout']
                )
            
            print('Number of parameters: {}'.format(sum(p.numel() for p in predictor.model.parameters())))
            
            
            predictor.model.to(device)

            # ---------- Train the model -----------------
            optimizer = torch.optim.Adam(predictor.model.parameters(), lr=pre_config['NN']['lr'])
            
            pt.train(predictor, train_loader, device, optimizer, 
                     epochs=pre_config['NN']['epochs'], 
                     lr=pre_config['NN']['lr'], _model=_m, _index=_index,
                     test_set=real_data_test)
            
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
            
            # Save the prediction
            save_pred_path = f'exp_pred/pred_results/NN_{_m}_pred_results_{_index}.pickle'
            with open(save_pred_path, 'wb') as f:
                pickle.dump(output, f)
                
            # Plot the prediction and save the figure
            import matplotlib.pyplot as plt
            # plOIT 10 subplots
            fig, axs = plt.subplots(10, 1, figsize=(10, 20))
            for i in range(10):
                axs[i].plot(target_data[i*4].cpu().detach().numpy().flatten(), label='target')
                axs[i].plot(output[i*4].flatten(), label='output')
                axs[i].legend()
                axs[i].set_title(f'Data_augmentation_{_index}')
            plt.savefig(f'exp_pred/pred_results/plots/NN_{_m}_pred_results_{_index}.png')
        