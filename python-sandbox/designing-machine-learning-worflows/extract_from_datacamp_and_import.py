# -*- coding: utf-8 -*-
"""
Extract from datacamp
Import locally
@author: F279814
"""

#%% get data from exercise Bringing it all together
import pandas as pd
X_train.to_csv('X_train.csv')
y_train.to_csv('y_train.csv')
X_test.to_csv('X_test.csv')
y_test.to_csv('y_test.csv')
!tar -zcvf CSV.tar.gz *
!curl --upload-file ./CSV.tar.gz https://transfer.sh/CSV.tar.gz 
#https://transfer.sh/cEb2N/CSV.tar.gz

import pandas as pd
X_train=pd.read_csv('X_train.csv', index_col=0)
y_train=pd.read_csv('y_train.csv', index_col=0)
X_test=pd.read_csv('X_test.csv', index_col=0)
y_test=pd.read_csv('y_test.csv', index_col=0)



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
