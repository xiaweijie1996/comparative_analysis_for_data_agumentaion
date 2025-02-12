
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
import pickle   
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data_process.data_loader as dl

# Catch the wind data
full_input_data, full_output_data = dl.catch_the_wind()

# Split the data into train, validation and test sets
data_dict = dl.split_train_test_val(full_input_data, full_output_data)

# Load the dictionary from the file
with open(f'exp_data_aug/data_dict_0.5.pickle', 'rb') as file:
    data_dict_reserve = pickle.load(file)
    
train_data = data_dict_reserve['train_input']
train_target = data_dict_reserve['train_output']
train_data, train_target = dl.nan_processing(train_data, train_target)

test_data = data_dict['test_input']
test_target = data_dict['test_output']
test_data, test_target = dl.nan_processing(test_data, test_target)

# Fit the model
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(train_data, train_target[:, 0])

# Predict the output
pred_target = regr.predict(test_data)

# Plot the results
plt.figure()
plt.plot(test_target[:, 0], 'k', label='True')
plt.plot(pred_target, 'r', label='Predicted')
plt.legend()
plt.savefig('exp_pred/svr/svr_predictor.png')
print('--------------------------------------')


