import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load test data
with open('new_exp_pred/pred_results/target.pickle', 'rb') as f:
    real_data_test = pickle.load(f)
    
path_real = 'new_exp_pred/pred_results/LGB_real_pred_results_1.0.pickle'
path_gmm = 'new_exp_pred/pred_results/LGB_gmm_pred_results_1.0.pickle'
path_flow = 'new_exp_pred/pred_results/LGB_flow_pred_results_1.0.pickle'
path_copula = 'new_exp_pred/pred_results/LGB_copula_pred_results_1.0.pickle'

with open(path_real, 'rb') as f:
    real_data = pickle.load(f)

with open(path_gmm, 'rb') as f:
    gmm_data = pickle.load(f)
    
with open(path_flow, 'rb') as f:
    flow_data = pickle.load(f)
    
with open(path_copula, 'rb') as f:
    copula_data = pickle.load(f)

    
print(real_data_test.shape, real_data.shape, gmm_data.shape, flow_data.shape, copula_data.shape)

# Plot the data
index = 4*48
plt.figure(figsize=(10, 4))
plt.plot(real_data_test[index:index+48].T, label='Target')
plt.plot(real_data[index:index+48].T, label='Real Data')
plt.plot(gmm_data[index:index+48].T, label='GMMs')
plt.plot(flow_data[index:index+48].T, label='FCPflow')
plt.plot(copula_data[index:index+48].T, label='t-Copula')
plt.xlabel('Time Step [30 Minutes]') 
plt.ylabel('Energy Output [kWh]')
plt.legend()
plt.grid()
plt.savefig('new_exp_pred/comparison_plot.png')