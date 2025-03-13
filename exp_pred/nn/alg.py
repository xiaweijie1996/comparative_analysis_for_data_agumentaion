import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn

class NNpredictor:
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.model = self._create_model()

    def _create_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        for _ in range(self.n_layers - 1):
            model.add_module('hidden', nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ))
        model.add_module('output', nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
        ))
        return model
    
class CNNConvpredictor:
    """
    A simple 1D CNN that transforms input (N, 48, 8) into (N, 48, 1).
    """
    def __init__(self, in_channels=8, hidden_channels=48, out_channels=1,
                 kernel_size=3, dropout=0.1):
        """
        Args:
            in_channels:    Number of input channels. For shape (N, 48, 8), this is 8.
            hidden_channels:Number of filters for the hidden convolution layer(s).
            out_channels:   Number of output channels. For shape (N, 48, 1), this is 1.
            kernel_size:    Size of the convolution kernel.
            dropout:        Dropout probability.
        """
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.model = self._create_model()

    def _create_model(self):
        """
        Build a nn.Sequential model that:
          1) Permutes (N, 48, 8) -> (N, 8, 48)
          2) Applies a Conv1d + ReLU + Dropout
          3) Applies another Conv1d to produce out_channels=1
          4) Permutes back to (N, 48, 1)
        """
        # Custom "layer" to permute dimensions
        class Permute(nn.Module):
            def __init__(self, *dims):
                super().__init__()
                self.dims = dims
            def forward(self, x):
                return x.permute(*self.dims)

        model = nn.Sequential(
            Permute(0, 2, 1),
            
            nn.Conv1d(self.in_channels, self.hidden_channels,
                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            nn.BatchNorm1d(self.hidden_channels),
            nn.LeakyReLU(),
            
            nn.Conv1d(self.hidden_channels, self.hidden_channels,
                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            nn.BatchNorm1d(self.hidden_channels),
            nn.LeakyReLU(),
            
            nn.Conv1d(self.hidden_channels, self.out_channels,
                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            Permute(0, 2, 1)
        )

        return model

if __name__ == '__main__':
    # Test the NN predictor
    input_dim = 10
    output_dim = 10
    hidden_dim = 100
    n_layers = 3
    dropout = 0.1
    model = NNpredictor(input_dim, output_dim, hidden_dim, n_layers, dropout)

    x = torch.randn(10, input_dim)
    y = model.model(x)
    print(y.size())
    
    
    # Test the CNN predictor
    x = torch.randn(2, 48, 8)

    # Build the CNN predictor
    cnp = CNNConvpredictor(
        in_channels=8,
        hidden_channels=16*5,
        out_channels=1,
        kernel_size=3,
        dropout=0.1
    )
    # Print model size
    print("Number of parameters:", sum(p.numel() for p in cnp.model.parameters()))
    
    # Forward pass
    y = cnp.model(x)
    # y should be (2, 48, 1)
    print("Output shape:", y.shape)