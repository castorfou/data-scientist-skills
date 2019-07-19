# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% imports
import numpy as np
import pandas as pd

#%% numpy instantiation
transport = ['train', 'bateau']
np_transport = np.array(transport)
print(np_transport[0])
transport.pop()

#%% test dataframe
d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data = d)
df
df = df+df
df2=pd.DataFrame(data=d)
df2.reindex(index=[2, 3])