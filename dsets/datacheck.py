import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import data_augmentation.FCPFlow.tools.tools_train as tl

path = 'dsets/test_set_wind_processed.csv'
data = pd.read_csv(path, index_col=0)
print(data.shape)
plt.plot(data.values.T)
plt.savefig('dsets/test_set_wind.png')
plt.show()

paht2 = 'dsets/test_set_wind.csv'
loader = tl.Datareshape(paht2)
data2 = loader.creat_new_frame()
print(data2.shape)
plt.plot(data2.values.T)
plt.savefig('dsets/test_set_wind2.png')
plt.savefig('dsets/test_set_wind2.png')
plt.show()

paht3 = 'dsets/percentage/continuous/5percent_dataset.csv'
loader = tl.Datareshape(paht3)
data3 = loader.creat_new_frame()
print(data3.shape)
plt.plot(data3.values.T)
plt.savefig('dsets/5percent_dataset.png')
plt.show()