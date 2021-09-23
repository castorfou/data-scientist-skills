# -*- coding: utf-8 -*-
"""
Extract from datacamp
Import locally
Created on Wed Jul 24 09:43:58 2019

@author: N561507
"""

#%%get initial content from datacamp
import numpy as np
np.savetxt("default.txt",default)
#!cat default.txt #and copy paste in a local file default.txt
np.savetxt("bill_amounts.txt",bill_amounts)
#!cat bill_amounts.txt #and copy paste in a local file bill_amounts.txt

#%% initial data
import numpy as np
test_features=np.loadtxt("test_features.txt")
test_targets=np.loadtxt("test_targets.txt")
print(test_features.shape, test_targets.shape)

#%% The dangers of local minima
#got it from datacamp:
#import inspect
#print(inspect.getsource(loss_function))


#%% get data from exercise Training with Keras
import numpy as np
print(sign_language_labels.shape)
np.savetxt("sign_language_labels.txt",sign_language_labels) 
!ls
!cat sign_language_labels.txt

print(sign_language_features.shape)
np.savetxt("sign_language_features.txt",sign_language_features) 
!tar -zcvf sign_language_features.txt.tar.gz sign_language_features.txt

!curl --upload-file ./sign_language_features.txt.tar.gz https://transfer.sh/sign_language_features.txt.tar.gz 
#https://transfer.sh/IAaIo/sign_language_features.txt.tar.gz


#%% get data from exercise Preparing to train with Estimators
import pandas as pd
housing = pd.DataFrame({'name': ['Raphael', 'Donatello'],'mask': ['red', 'purple'],'weapon': ['sai', 'bo staff']})
housing.to_csv('housing_df.csv')
!curl --upload-file ./housing_df.csv https://transfer.sh/housing_df.csv 

import pandas as pd
housing=pd.read_csv('housing_df.csv', index_col=0)
