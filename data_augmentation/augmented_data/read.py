import pickle 

# Load the data
_paht = 'data_augmentation/augmented_data/10percent_dict.pkl'
with open(_paht, 'rb') as _file:
    _data = pickle.load(_file)
    
print(_data.keys())

print(_data['train_input'].shape)
print(_data['train_output'].shape)