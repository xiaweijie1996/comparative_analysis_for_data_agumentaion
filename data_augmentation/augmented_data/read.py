import pickle 

# Load the data
for i in [0.05, 0.1, 0.3, 0.5, 0.8, 1.0]:
    _paht = f'data_augmentation/augmented_data/{int(i*100)}percent_dict.pkl'
    with open(_paht, 'rb') as _file:
        _data = pickle.load(_file)
        
    print(_data.keys())

    print(_data['train_input'].shape)
    print(_data['train_output'].shape)