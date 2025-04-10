import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load test data
with open('dsets/test_set_wind_processed.pkl', 'rb') as f:
    real_data_test = pickle.load(f)
    
path_real = 'exp_pred/pred_results/LGB_read_data_pred_results.pickle'
path_gmm = 'exp_pred/pred_results/LGB_gmm_pred_results_1.0.pickle'
path_flow = 'exp_pred/pred_results/LGB_flow_pred_results_1.0.pickle'
path_copula = 'exp_pred/pred_results/LGB_copula_pred_results_1.0.pickle'
path_doppel = 'exp_pred/pred_results/LGB_DoppelGANger_pred_results_1.0.pickle'

with open(path_real, 'rb') as f:
    real_data = pickle.load(f)

with open(path_gmm, 'rb') as f:
    gmm_data = pickle.load(f)
    
with open(path_flow, 'rb') as f:
    flow_data = pickle.load(f)
    
with open(path_copula, 'rb') as f:
    copula_data = pickle.load(f)
    
with open(path_doppel, 'rb') as f:
    doppel_data = pickle.load(f)
    
print(real_data_test['output'].shape, real_data.shape, gmm_data.shape, flow_data.shape, copula_data.shape, doppel_data.shape)
real_data_test = real_data_test['output'].reshape(real_data_test['output'].shape[0], -1)
real_data = real_data.reshape(real_data.shape[0], -1)
gmm_data = gmm_data.reshape(gmm_data.shape[0], -1)
flow_data = flow_data.reshape(flow_data.shape[0], -1)
copula_data = copula_data.reshape(copula_data.shape[0], -1)
doppel_data = doppel_data.reshape(doppel_data.shape[0], -1)
print(real_data_test.shape, real_data.shape, gmm_data.shape, flow_data.shape, copula_data.shape, doppel_data.shape)

# Plot the data
index = 3
plt.figure(figsize=(10, 4))
plt.plot(real_data_test[index].T, label='Target')
plt.plot(real_data[index].T, label='Real Data')
plt.plot(gmm_data[index].T, label='GMMs')
plt.plot(flow_data[index].T, label='FCPflow')
plt.plot(copula_data[index].T, label='t-Copula')
plt.plot(doppel_data[index].T, label='DoppelGANger')
plt.xlabel('Time Step [30 Minutes]') 
plt.ylabel('Energy Output [kWh]')
plt.legend()
plt.grid()
plt.savefig('exp_pred/comparison_plot.png')