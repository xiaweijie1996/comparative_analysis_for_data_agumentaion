import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pickle

import data_process.data_loader as dl

# Catch the wind data
full_input_data, full_output_data =  dl.catch_the_wind()

# Split the data into train, validation and test sets
data_dict = dl.split_train_test_val(full_input_data, full_output_data)

# Split the training data into training and reserve sets
reserve_ratios = [0.1, 0.3, 0.5, 0.8, 1.0]
for reserve_ratio in reserve_ratios:
    data_dict_reserve = dl.train_data_reserve(data_dict, reserve_ratio)

    print(data_dict_reserve['train_input'].shape,
            data_dict_reserve['train_output'].shape)

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

