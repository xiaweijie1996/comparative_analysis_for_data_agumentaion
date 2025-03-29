import pickle
import pandas as pd
import numpy as np


class Datareshape():
    def __init__(self, data_path):
        self.dataframe = pd.read_csv(data_path, index_col=0)
        self.length = int(self.dataframe.shape[0]/48)
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
        new_frame = pd.DataFrame()
        for i in range(self.length):
            _data = self.dataframe.iloc[i*48:(i+1)*48, :]
            _data = _data.values
            _x = _data[:, :-1].reshape(1, -1)
            _y = _data[:, -1:].reshape(1, -1)
            _data = np.hstack((_x, _y))
            new_frame = pd.concat([new_frame, pd.DataFrame(_data)], axis=0)
        return new_frame
    
    def restor_shape(self, frame):
        _data_dict = {}
        data = frame.values
        length = data.shape[0]
        input_x = data[:, :-48].reshape(length, 48, 8)
        ouput_y = data[:, -48:].reshape(length, 48, 1)
        
        _data_dict['input'] = input_x
        _data_dict['output'] = ouput_y
        
        return _data_dict
        
        
if __name__ == '__main__':
    # ---------- Load the data -----------------
    _path = 'data_augmentation/augmented_data/5.0percent_dict_copula.pkl'
    with open(_path, 'rb') as f:
        aug_data = pickle.load(f)
        keys = list(aug_data.keys())
        print(keys)
    print(aug_data['input'].shape)
    print(aug_data['output'].shape)

    # ----------------- Define the Copula -------------- ---
    for _index in [0.05, 0.1, 0.3, 0.5, 0.8, 1.0]: # 0.05, 0.1, 0.3, 0.5, 
        # ---------------Data Process-----------------
        _data_path = f'dsets/percentage/continuous/{int(_index*100)}percent_dataset.csv'
        data_reshape = Datareshape(_data_path)
        _data = data_reshape.creat_new_frame()
        _data = _data.values
        
        # Drop the nan
        _data = _data[~np.isnan(_data).any(axis=1)]
        print(_data.shape)
        
        dict_data = {}
        dict_data['input'] = _data[:, :-48].reshape(-1, 48, 8)
        dict_data['output'] = _data[:, -48:].reshape(-1, 48, 1)
        # Save the data as a pickle file
        with open(f'dsets/percentage/continuous/{int(_index*100)}percent_dataset.pkl', 'wb') as f:
            pickle.dump(dict_data, f)
