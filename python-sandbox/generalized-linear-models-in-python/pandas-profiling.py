# -*- coding: utf-8 -*-
"""
test pandas profiling
Created on Mon Jul 29 22:26:08 2019

@author: N561507
"""

#%% test pandas-profiling
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import pandas_profiling

wells=pd.read_csv('wells.csv', index_col=0)
wells.profile_report(style={'full_width':True})
