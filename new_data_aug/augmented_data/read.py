import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# select_list
select_list = [1,0]

# Load the training data
path_train = 'dsets/train_set_wind.csv'
train_data = pd.read_csv(path_train, index_col=0)
print('train_data shape:', train_data.shape)

# Load the test data
path_test = 'dsets/test_set_wind.csv'
test_data = pd.read_csv(path_test, index_col=0)
print('test_data shape:', test_data.shape)

# Load the augmented data
path_aug_GMM = 'new_data_aug/augmented_data/gmm_generated_data_1.0.csv'
aug_data_GMM = pd.read_csv(path_aug_GMM, index_col=0)
print('aug_data_GMM shape:', aug_data_GMM.shape)

# Load the augmented data copula
path_aug_copula = 'new_data_aug/augmented_data/copula_generated_data_1.0.csv'
aug_data_copula = pd.read_csv(path_aug_copula, index_col=0)
print('aug_data_copula shape:', aug_data_copula.shape)

# Load the augmented data FCPFlow
path_aug_FCPFlow = 'new_data_aug/augmented_data/fcpflow_generated_data_new_1.0.csv'
aug_data_FCPFlow = pd.read_csv(path_aug_FCPFlow, index_col=0)
print('aug_data_FCPFlow shape:', aug_data_FCPFlow.shape)


# Plot the training data
plt.figure(figsize=(10, 20))
plt.subplot(5, 1, 1)
plt.plot(train_data.iloc[:7291,:].T,  color='blue', alpha=0.01)
plt.title('Training Data')
plt.xlabel('Time')
plt.ylabel('y')

print(1)
plt.subplot(5, 1, 2)
plt.plot(test_data.T,  color='orange', alpha=0.01)
plt.title('Test Data')
plt.xlabel('Time')
plt.ylabel('y')

print(1)
plt.subplot(5, 1, 3)
plot_data = aug_data_GMM.iloc[:7291,:]
plt.plot(plot_data.T,  color='green', alpha=0.01)
plt.title('Augmented Data (GMM)')
plt.xlabel('Time')
plt.ylabel('y')

print(1)
plt.subplot(5, 1, 4)
plot_data = aug_data_copula.iloc[:7291,:]
plt.plot(plot_data.T,  color='red', alpha=0.01)
plt.title('Augmented Data (Copula)')
plt.xlabel('Time')
plt.ylabel('y')
print(1)

plt.subplot(5, 1, 5)
plot_data = aug_data_FCPFlow.iloc[:7291,:]
plt.plot(plot_data.T,  color='red', alpha=0.01)
plt.title('Augmented Data (FCPFlow)')
plt.xlabel('Time')
plt.ylabel('y')

print(1)
plt.savefig('new_data_aug/augmented_data/train_test_data.png')
plt.show()

