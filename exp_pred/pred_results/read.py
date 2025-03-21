import pickle


path = 'exp_pred/pred_results/DoppelGANger_pred_results_0.1.pickle'

with open(path, 'rb') as f:
    data = pickle.load(f)

print(data.shape)