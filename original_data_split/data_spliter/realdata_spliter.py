import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pickle
import pandas as pd

import data_process.data_process as dp



if __name__ == '__main__':
    # Catch the wind data
    full_input_data, full_output_data =  dp.catch_the_wind()
    
    # Add lags to the data
    full_input_data, full_output_data = dp.feature_lag(full_input_data, full_output_data)
    
    # Split the data into train, validation and test sets
    data_dict = dp.split_train_test_val(full_input_data, full_output_data)

    # Split the training data into training and reserve sets
    reserve_ratios = [0.1, 0.3, 0.5, 0.8, 1.0]
    for reserve_ratio in reserve_ratios:
        data_dict_reserve = dp.train_data_reserve(data_dict, reserve_ratio)
        
        # print(data_dict_reserve['train_input'].head(),
        #         data_dict_reserve['train_output'].head())
                
        # Save the dictionary to a file
        with open(f'original_data_split/data_dict_{reserve_ratio}.pickle', 'wb') as file:
            pickle.dump(data_dict_reserve, file)
        print('--------------------------------------')

        # Load the dictionary from the file
        with open(f'original_data_split/data_dict_{reserve_ratio}.pickle', 'rb') as file:
            data_dict_reserve = pickle.load(file)
        print(data_dict_reserve['train_input'].shape,
                data_dict_reserve['train_output'].shape)
        print('--------------------------------------')
    
    # Save test data to a file
    with open('original_data_split/data_dict.pickle', 'wb') as file:
        pickle.dump(data_dict, file)

