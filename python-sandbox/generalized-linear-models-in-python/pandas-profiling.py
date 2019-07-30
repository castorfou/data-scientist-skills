# -*- coding: utf-8 -*-
"""
test pandas profiling
Created on Mon Jul 29 22:26:08 2019

@author: N561507
"""

#%% test pandas-profiling
import pandas as pd
import pandas_profiling

wells=pd.read_csv('wells.csv', index_col=0)
wells.profile_report(style={'full_width':True})
