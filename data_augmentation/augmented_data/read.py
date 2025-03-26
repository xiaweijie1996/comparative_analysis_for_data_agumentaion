import pickle 
# Plot the data
import matplotlib.pyplot as plt

# select_list
select_list = [0.8]


# Load the data
for i in select_list:
    _paht = f'data_augmentation/augmented_data/{int(i*100)}percent_dict.pkl'
    with open(_paht, 'rb') as _file:
        _data = pickle.load(_file)

    # PLot the data into one plot
    y = _data['train_output']
    y = y.reshape(978, -1)
    for i in y:
        plt.plot(i, c='b', alpha=0.1)
    plt.savefig(f'data_augmentation/augmented_data/percent_dict_gang.png')
    plt.close()
    break


for i in select_list:
    _paht = f'data_augmentation/augmented_data/{i*100}percent_dict_gmm.pkl'
    with open(_paht, 'rb') as _file:
        _data = pickle.load(_file)
        
    # PLot the data into one plot
    y = _data['output']
    y = y.reshape(978, -1)
    for i in y:
        plt.plot(i, c='b', alpha=0.1)
    plt.savefig(f'data_augmentation/augmented_data/percent_dict_gmm.png')
    plt.close()
    break

for i in select_list:
    _paht = f'data_augmentation/augmented_data/{i*100}percent_dict_fcpflow_1.pkl'
    with open(_paht, 'rb') as _file:
        _data = pickle.load(_file)
    
    # PLot the data into one plot
    y = _data['output']
    y = y.reshape(-1, 48)
    
    print(y.shape)
    for i in y:
        plt.plot(i, c='b', alpha=0.1)
    plt.savefig(f'data_augmentation/augmented_data/percent_dict_fcpflow.png')
    plt.close()
    break


for i in select_list:
    _paht = f'data_augmentation/augmented_data/{i*100}percent_dict_copula.pkl'
    with open(_paht, 'rb') as _file:
        _data = pickle.load(_file)

    # PLot the data into one plot
    y = _data['output']
    y = y.reshape(978, -1)
    for i in y:
        plt.plot(i, c='b', alpha=0.1)
    plt.savefig(f'data_augmentation/augmented_data/percent_dict_copula.png')
    plt.close()
    break