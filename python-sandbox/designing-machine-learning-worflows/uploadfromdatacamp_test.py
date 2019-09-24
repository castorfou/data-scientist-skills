# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:48:03 2019

@author: F279814
"""

#%% get function from datacamp (example)
#import inspect
#print(inspect.getsource(my_metric))
   
#%% example of usage for dataframe/series
######################## from datacamp
uploadToFileIO(X_train, X_test, y_train, y_test)

######################### from local

from uploadfromdatacamp import saveFromFileIO
import pandas as pd

############get files
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/5ZlMWu',  'X_train.csv': 'https://file.io/kI8Dvv'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/6jHnZ5',  'y_train.csv': 'https://file.io/Qz1YT8'}}"
prefix='Chap35_'
saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
############load objects
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_test=pd.read_csv(prefix+'y_test.csv', index_col=0, header=None,squeeze=True)
y_train=pd.read_csv(prefix+'y_train.csv', index_col=0, header=None,squeeze=True)


#%% example of usage for pkl files
######################## from datacamp

uploadToFileIO_pushto_fileio('model.pkl')
    
######################### from local
from uploadfromdatacamp import saveFromFileIO

############get files
url='https://file.io/KHpP8R'
tobesaved_as='model.pkl'
prefix='Chap35_'
tobedownloaded="{pipeline:{'"+tobesaved_as+"': '"+url+"'}}"
saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#%% example of usage for pipeline pkl files
######################## from datacamp
import pickle
with open('pipe.pkl', 'wb') as file:
    pickle.dump(pipe, file)
uploadToFileIO_pushto_fileio('pipe.pkl')

######################### from local
############get files
url='https://file.io/KHpP8R'
tobesaved_as='pipe.pkl'
prefix='Chap36_'
tobedownloaded="{pipeline:{'"+tobesaved_as+"': '"+url+"'}}"
saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
    
############load objects
with open(prefix+tobesaved_as, 'rb') as file:
    pipe = pickle.load(file)

#%% import
from uploadfromdatacamp import *
import pandas as pd
import numpy as np
import glob
#string
test_string='je suis un string'
#ndarray
a_numpy = np.random.rand(10 ** 3, 5)
b_numpy = np.array([10, 20, 30, 40, 50])
#dataframe
df1 = pd.DataFrame(a_numpy) 
#series
ds2 = pd.Series(b_numpy)
#liste 
liste=[1., 2., 3.]

 #liste des objets à tester
liste_a_tester=[test_string, df1, a_numpy, ds2,liste]
liste_filenames=glob.glob("_TEST*.*")
    

#%% test_uploadToFileIO_getfilename
#return appropriate filename for a given variable
#def uploadToFileIO_get_filename(variable):
def test_uploadToFileIO_get_filename():
    for arg in liste_a_tester:
        print(uploadToFileIO_get_filename(arg))
    
test_uploadToFileIO_get_filename()

#%% test uploadToFileIO_saveas_filename
#save variable as a file named filename
#no return
#def uploadToFileIO_saveas_filename(variable,filename):
def test_uploadToFileIO_saveas_filename():    #ndarray
    for arg in liste_a_tester:
        filename=uploadToFileIO_get_filename(arg)
        uploadToFileIO_saveas_filename(arg, '_TEST_uploadfromdatacamp'+filename)
   
test_uploadToFileIO_saveas_filename()

#%% test def uploadToFileIO_pushto_fileio(filename,proxy=''):
#upload filename to file.io and return url of this file on file.io
#as an optionnal parameter take a proxy            
def test_uploadToFileIO_pushto_fileio():
    proxy="10.225.92.1:80"
    for filename in liste_filenames:
        uploadToFileIO_pushto_fileio(filename, proxy=proxy)
        
test_uploadToFileIO_pushto_fileio()
    
    
#%% def test_uploadToFileIO
#def uploadToFileIO(*argv, proxy=''):
    # liste des objets à envoyer
    #print un nested dictionnaire avec {type: {filename: url_sur_fileIO}}
    #à donner en entree de saveFromFileIO
def test_uploadToFileIO():
    proxy="10.225.92.1:80"
    _TEST_test_string=test_string
    _TEST_df1=df1
    _TEST_a_numpy=a_numpy
    _TEST_ds2=ds2
    _TEST_liste=liste
    print(uploadToFileIO(test_string, df1, a_numpy, ds2,liste,proxy=proxy))

test_uploadToFileIO()
    
    
    
    
    
    
    
    