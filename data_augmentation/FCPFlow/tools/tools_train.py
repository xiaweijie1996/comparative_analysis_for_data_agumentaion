import os

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import time

import tools.evaluation_m as em

torch.set_default_dtype(torch.float64)

def create_data_loader(numpy_array, batch_size=32, scaler = StandardScaler(), shuffle=True):
    # Check if nan exists in the data, if nan drop
    if np.isnan(numpy_array).any():
        print('There are nan in the data, drop them')
        numpy_array = numpy_array[~np.isnan(numpy_array).any(axis=1)]

    # Scalr the 
    numpy_array = scaler.fit_transform(numpy_array)
    
    # Convert the NumPy array to a PyTorch Tensor
    tensor_data = torch.Tensor(numpy_array)

    # Create a TensorDataset from the Tensor
    dataset = TensorDataset(tensor_data)
    
    # Create a DataLoader from the Dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader, scaler

def log_likelihood(x, type='Gaussian'):
    """Compute log likelihood for x under a uniform distribution in [0,1]^D.
    Args:
    - x (torch.Tensor): Input tensor of shape (batch_size, D)
    Returns:
    - log_likelihood (torch.Tensor): Log likelihood for each sample in the batch. Shape: (batch_size,)
    """
    if type == 'Uniform':
        # Check if all values in x are within the interval [0,1]
        is_inside = ((x >= -1) & (x <= 1)).all(dim=1).float()
        log_likelihood = 0 * is_inside - (x*x).mean(dim=1)*(1-is_inside)
    if type == 'Gaussian':
        log_likelihood = -0.5 * x.pow(2).sum(dim=1)
    return log_likelihood

def plot_figure(pre, re_data, scaler, con_dim, path='Generated Data Comparison.png'):
    # Inverse transform to get the original scale of the data
    orig_data_pre = scaler.inverse_transform(pre.cpu().detach().numpy())
    orig_data_re = scaler.inverse_transform(re_data.cpu().detach().numpy())
    
    cmap = plt.get_cmap('RdBu_r')
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))  # TFour rows for comparison
    
    if con_dim > 0:
        # Original data plot
        _cond_pre = orig_data_pre[:, 49:-con_dim-1].sum(axis=1)
        for i, condition in zip(orig_data_pre[:, 49:-con_dim-1], _cond_pre):
            color = cmap((condition - _cond_pre.min()) / (_cond_pre.max() - _cond_pre.min()))
            axs[0].plot(i, color=color, alpha=0.1)
        axs[0].set_title('Original Wind Output')
        
        # Reconstructed/Generated data plot
        _cond_re = orig_data_re[:, 49:-con_dim-1].sum(axis=1)
        for i, condition in zip(orig_data_re[:, 49:-con_dim-1], _cond_re):
            color = cmap((condition - _cond_re.min()) / (_cond_re.max() - _cond_re.min()))
            axs[1].plot(i, color=color, alpha=0.1)
        axs[1].set_title('Reconstructed/Generated Wind Output')
        
        # Original data plot
        _cond_pre = orig_data_pre[:, :48].sum(axis=1)
        for i, condition in zip(orig_data_pre[:, :48], _cond_pre):
            color = cmap((condition - _cond_pre.min()) / (_cond_pre.max() - _cond_pre.min()))
            axs[2].plot(i, color=color, alpha=0.1)
        axs[2].set_title('Original Wind Information (direction, speed)')
        
        # Reconstructed/Generated data plot
        _cond_re = orig_data_re[:, :48].sum(axis=1)
        for i, condition in zip(orig_data_re[:, :48], _cond_re):
            color = cmap((condition - _cond_re.min()) / (_cond_re.max() - _cond_re.min()))
            axs[3].plot(i, color=color, alpha=0.1)
        axs[3].set_title('Reconstructed/Generated Wind Information (direction, speed)')
        
        
        # Add colorbars to each subplot
        for ax, _cond in zip(axs, [_cond_pre, _cond_re]):
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=_cond.min(), vmax=_cond.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Condition Value scaled', rotation=270, labelpad=20)
        
    else:
        # Original data plot
        for i in orig_data_pre:
            axs[0].plot(i, color='blue', alpha=0.1)
        axs[0].set_title('Original Data')

        # Reconstructed/Generated data plot
        for i in orig_data_re:
            axs[1].plot(i, color='red', alpha=0.1)
        axs[1].set_title('Reconstructed/Generated Data')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



def train(model, train_loader, optimizer, epochs, cond_dim, 
          device, scaler, test_loader, scheduler, pgap=10, 
          index=0.1, _wandb=True, _plot=True, _save=True):
    
    """
    Train the model
    Args:
        model (torch.nn.Module): The generative model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        epochs (int): Number of training epochs.
        cond_dim (int): Dimension of conditional variables.
        device (torch.device): Device to run the model (CPU or GPU).
        scaler (object): Scaler used for normalizing and inverse transforming data.
        test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        pgap (int, optional): Interval for plotting generated samples. Defaults to 100.
        _wandb (bool, optional): Whether to log loss to Weights & Biases (wandb). Defaults to True.
        _plot (bool, optional): Whether to plot generated samples. Defaults to True.
        _save (bool, optional): Whether to save the model when test loss improves. Defaults to True.

    Returns:
        none
    """
    
    model.train()
    loss_mid = 100000000
    for epoch in range(epochs):
        for _, data in enumerate(train_loader):
            model.train()
            pre = data[0].to(device) 
            
            # Split the data into data and conditions
            cond = pre[:,-cond_dim:]
            data = pre[:,:-cond_dim] 
            
            gen, logdet = model(data, cond)
            
            # Compute the log likelihood loss
            llh = log_likelihood(gen, type='Gaussian')
            loss = -llh.mean()-logdet
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
        # ----------------- moniter loss -----------------
        if _wandb:
            wandb.log({'loss': loss.item()})
        # ----------------- moniter loss -----------------
            
        # ----------------- test the model -----------------
        model.eval()
        
        # Test the model
        pre = next(iter(test_loader))[0].to(device)
        cond_test = pre[:,-cond_dim:]
        data_test = pre[:,:-cond_dim]
        noise = torch.randn(data_test.shape[0], data_test.shape[1]).to(device)
        gen_test = model.inverse(noise, cond_test)

        llh_test = em.calculate_energy_distances(gen_test.detach().numpy(), data_test.detach().numpy())
        loss_test = llh_test.mean()
        
        # Save the model
        if _save:
            if loss_test.item() < loss_mid:
                print('save the model')
                save_path = os.path.join('data_augmentation/FCPFlow/saved_model', f'FCPflow_model_{index}.pth')
                torch.save(model.state_dict(), save_path)
                loss_mid = loss_test.item()

                # ----------------- Plot the generated data -----------------
                if _plot:
                    if (epoch % pgap ==0):
                        save_path = os.path.join('data_augmentation/FCPFlow/saved_model',f'FCPflow_generated_{index}.png')
                        plot_figure(pre, re_data, scaler, cond_dim, save_path)
                # ----------------- Plot the generated data -----------------
            
        print(epoch, 'loss LogLikelihood: ', loss.item(), 'loss Energy Distance: ', loss_test.item())
        
        # Plot the generated data
        z = torch.randn(data_test.shape[0], data_test.shape[1]).to(device)
        gen_test = model.inverse(z, cond_test)
        re_data = torch.cat((gen_test, cond_test), dim=1)
        re_data = re_data.detach()
        
        # ----------------- Test the model -----------------
        
        
