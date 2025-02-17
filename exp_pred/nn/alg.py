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
    

if __name__ == '__main__':
    input_dim = 10
    output_dim = 10
    hidden_dim = 100
    n_layers = 3
    dropout = 0.1
    model = NNpredictor(input_dim, output_dim, hidden_dim, n_layers, dropout)

    x = torch.randn(10, input_dim)
    y = model.model(x)
    print(y.size())