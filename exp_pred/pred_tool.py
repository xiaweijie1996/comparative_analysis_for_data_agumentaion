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

def create_data_loader(numpy_array2, batch_size=32, scaler = StandardScaler(), default_length = 765, shuffle=True):
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

    # Concatenate the two numpy arrays
    numpy_array = numpy_array2
    
    # Scalr the data
    numpy_array = scaler.fit_transform(numpy_array)
    
    # Convert the NumPy array to a PyTorch Tensor
    tensor_data = torch.Tensor(numpy_array)

    # Create a TensorDataset from the Tensor
    dataset = TensorDataset(tensor_data)
    
    # Create a DataLoader from the Dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader, scaler

def train(model, train_loader, device, optimizer, split, scaler, epochs=10, lr=0.001, _model ='gmm', _index='0,1', test_set=None):
    # Define the loss function
    criterion = nn.MSELoss()
    
    # Train the model
    model.model.train()
    initial_loss = 1000
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            # Get the data
            _data = data[0].to(device)
            
            input_data = _data[:, :-split]
            target_data = _data[:, -split:]
            
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
            _data = torch.Tensor(test_set)
            _data = scaler.transform(_data)
            _data = torch.Tensor(_data).to(device)
            
            input_data = _data[:, :-split]
            target_data = _data[:, -split:]
            
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