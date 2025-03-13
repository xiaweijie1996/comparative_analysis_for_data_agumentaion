import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import time
import psutil
import pandas as pd
# import tools.evaluation_m as em

torch.set_default_dtype(torch.float64)

class Datareshape():
    def __init__(self, data_path):
        self.dataframe = pd.read_csv(data_path, index_col=0)
        self.length = self.dataframe.shape[0]
        self.width = self.dataframe.shape[1]
        self.add_month_hour()
        
    def add_month_hour(self):
        # Add month and hour to the dataframe
        # Change the index to datetime
        self.dataframe.index = pd.to_datetime(self.dataframe.index)
        self.dataframe['month'] = self.dataframe.index.month
        self.dataframe['hour'] = self.dataframe.index.hour
        
        # Put month and hour before the last column
        cols = list(self.dataframe.columns)
        cols = cols[:-3] + cols[-2:] + cols[-3:-2]
        self.dataframe = self.dataframe[cols]
        
        
    
    def creat_new_frame(self):
        _num_step = 48
        new_frame = pd.DataFrame()
        for i in range(self.length-_num_step):
            _data = self.dataframe.iloc[i:i+_num_step, :]
            _data = _data.values
            _x = _data[:, :-1].reshape(1, -1)
            _y = _data[:, -1:].reshape(1, -1)
            _data = np.hstack((_x, _y))
            new_frame = pd.concat([new_frame, pd.DataFrame(_data)], axis=0)
        return new_frame
    
    def restor_shape(self, data):
        data = data.values
        length = data.shape[0]
        input_x = data[:, :-48].reshape(length, 48, 8)
        ouput_y = data[:, -48:].reshape(length, 48, 1)
        
        dic = {'input': input_x, 'output': ouput_y}
        return dic

if __name__ == "__main__":
    
    # Test the Datareshape class
    data_path = 'dsets/percentage/continuous/5percent_dataset.csv'
    data_reshape = Datareshape(data_path)
    print(data_reshape.dataframe.head())
    new_frame = data_reshape.creat_new_frame()
    print(new_frame.head())
    
