import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
import matplotlib.pyplot as plt

def create_data_loader(dict1, batch_size=32, default_length = 765, shuffle=True):
    """
    Create a DataLoader from two NumPy arrays.

    Args:
        numpy_array1 (_type_): real data
        numpy_array2 (_type_): augmented data
        batch_size (int, optional):  Batch size. 
        scaler (_type_, optional): Defaults to StandardScaler().
        default_length (int, optional): The length of the data = Length of the real data + Length of the augmented data. 
        shuffle (bool, optional): Defaults to True.
    
    """
    
    # Check if nan exists in the data, if nan drop
    # if np.isnan(numpy_array1).any():
    #     numpy_array1 = numpy_array1[~np.isnan(numpy_array1).any(axis=1)]
    
    # len_1 = numpy_array1.shape[1]
    # sample_len = default_length - len_1 # Sample length from numpy_array1
    # numpy_array2 = numpy_array2[:, :sample_len] # Get the last sample_len from numpy_array1

    # X 
    input_data = dict1['input']
    target_data = dict1['output']
    
    # Create a DataLoader from the Dataset
    data_loader = DataLoader(TensorDataset(torch.Tensor(input_data), torch.Tensor(target_data)),
                             batch_size=batch_size, shuffle=shuffle)
    
    return data_loader

def train(model, train_loader, device, optimizer, epochs=10, lr=0.001, _model ='gmm', _index='0,1', test_set=None):
    # Define the loss function
    criterion = nn.MSELoss()
    
    # Train the model
    model.model.train()
    initial_loss = 1000
    for epoch in range(epochs):
        for i, (X, y) in enumerate(train_loader):
            input_data = X.to(device)
            target_data = y.to(device)
            
            # Predict
            output = model.model(input_data)
            
            # Calculate the loss
            loss = criterion(output, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute the loss
        model.model.eval()
        with torch.no_grad():
            input_data = test_set[0]
            input_data = torch.tensor(input_data).to(device)
            target_data = test_set[1]
            target_data = torch.tensor(target_data).to(device)
            
            target_data = target_data.float()
            input_data = input_data.float()
            output = model.model(input_data)
            loss_test = criterion(output, target_data)
        
        print('Epoch: {}, Loss in test: {}, Loss in train: {}'.format(epoch, loss_test.item(), loss.item()))
           
        # Save the model if the loss is less than the initial loss
        if loss.item() < initial_loss:
            initial_loss = loss.item()
            torch.save(model.model.state_dict(), 'exp_pred/nn/saved_model/{}_model_{}.pt'.format(_model, _index))
            
            # Plot the prediction
            plt.plot(input_data[0].cpu().detach().numpy(), label='input')
            _len = input_data.size(1)
            plt.plot(np.arange(_len, _len + target_data.size(1)), target_data[0].cpu().detach().numpy(), label='target')
            plt.plot(np.arange(_len, _len + output.size(1)), output[0].cpu().detach().numpy(), label='output')
            plt.legend()
            plt.title('Data_augmentation_{}'.format(_index))
            plt.savefig('exp_pred/nn/saved_model/{}_pred_{}.png'.format(_model, _index))
            plt.close()
            

if __name__ == '__main__':
    # Test the create_data_loader function
    dict1 = {'input_data': np.random.rand(100, 100), 'output_data': np.random.rand(100, 10)}
    train_loader, scaler = create_data_loader(dict1, batch_size=32, default_length=110)
    print(train_loader)
    print(scaler)
    x, y = next(iter(train_loader))
    print(x.size())
    print(y.size())