import pandas as pd
import yaml

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

with open('new_data_aug/augmentation_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

for _index in [1.0]: # [0.1, 0.3, 0.5, 0.8, 
    _data_path = config["Path"][f"input_path_{_index}"]  
    _data = pd.read_csv(_data_path, index_col=0)
    print(_data.shape)
    
    # Plot the first the data
    plt.plot(_data.values, alpha = 0.1, c ='blue')
    plt.title(f"Original data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.savefig(f"new_data_aug/train_data.png")
    plt.close()
    
    # Plot the correlation matrix
    corr = _data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Matrix")
    plt.savefig(f"new_data_aug/correlation_matrix.png")
    plt.close()

test_path = 'dsets/test_set_wind.csv'
_data = pd.read_csv(test_path, index_col=0)
print(_data.shape)

# Plot the first the data
plt.plot(_data.values, alpha = 0.1, c ='blue')
plt.title(f"Original data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig(f"new_data_aug/test_data.png")
plt.close()

# Plot the correlation matrix
corr = _data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix")
plt.savefig(f"new_data_aug/test_correlation_matrix.png")
plt.close()