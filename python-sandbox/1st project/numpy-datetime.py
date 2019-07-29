# -*- coding: utf-8 -*-
"""
numpy datetime
Created on Mon Jul 29 08:52:14 2019

@author: N561507
"""

import pandas as pd
import numpy as np

data_brute=['un','soleil','d','ete']
date_brutes=['2020-10-12','2018-12-27','2019-01-01','2019-07-29']


n_data=np.array(list(zip(date_brutes,data_brute)))
print(n_data)
p_data=pd.DataFrame(n_data,columns=['date','mot'])
print(p_data)



#%% transformer l'index en timeseries
p2_data = p_data.copy()
p2_data['date2']=pd.to_datetime(p2_data['date'])
del p2_data['date']
p2_data=p2_data.sort_values('date2')
p2_data.set_index('date2',inplace=True)
print(p2_data.info())


print(p2_data['2019'])

print(p2_data.groupby(p2_data.index.year).describe())

