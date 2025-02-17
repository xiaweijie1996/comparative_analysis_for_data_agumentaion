import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import pandas as pd
import yaml
import pickle
import numpy as np

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
    for _m in ['gmm', 'fcpflow']:
        for _index in [0.1, 0.3, 0.5, 0.8, 1.0]:
            # Load augmented data
            datapath = 'data_augmentation/augmented_data/{}_generated_data_{}.csv'.format(_m, _index)
            # Load the data without first column
            aug_data = pd.read_csv(datapath, index_col=0).values
            
            # Load real data
            real_data_path = f'original_data_split/data_dict_{_index}.pickle'
            with open(real_data_path, 'rb') as f:
                real_data = pickle.load(f)
                real_data = np.hstack((real_data['train_input'], real_data['train_output']))
            
            train_loader, scaler = pt.create_data_loader(real_data, aug_data, 
                                                         batch_size=pre_config['NN']['batch_size'], 
                                                         default_length=pre_config['NN']['default_length'],
                                                         shuffle=True)
            
            # ---------- Load the model -----------------
            predictor = al.NNpredictor(input_dim=pre_config['NN']['input_dim'], 
                                       output_dim=pre_config['NN']['output_dim'], 
                                       hidden_dim=pre_config['NN']['hidden_dim'], 
                                       n_layers=pre_config['NN']['n_layers'], 
                                       dropout=pre_config['NN']['dropout'])
            
            print('Number of parameters: {}'.format(sum(p.numel() for p in predictor.model.parameters())))
            
            predictor.model.to(device)

            # ---------- Train the model -----------------
            optimizer = torch.optim.Adam(predictor.model.parameters(), lr=pre_config['NN']['lr'])
            
            pt.train(predictor, train_loader, device, optimizer, 
                     pre_config['NN']['split'], epochs=pre_config['NN']['epochs'], 
                     lr=pre_config['NN']['lr'])
            
            break