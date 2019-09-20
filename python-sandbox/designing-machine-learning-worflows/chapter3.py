# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 08:35:36 2019

@author: F279814
"""

#%% Exercise - Your first pipeline - again! - init
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
#from uploadfromdatacamp import saveFromFileIO
import pandas as pd

#uploadFromDatacamp(X_train, y_train)
#{pandas.core.frame.DataFrame: {'Chap31_X_train.csv': 'https://file.io/Pev7YU'}, pandas.core.series.Series: {'Chap31_y_train.csv': 'https://file.io/npZaEc'}}
#saveFromFileIO("{pandas.core.frame.DataFrame: {'Chap31_X_train.csv': 'https://file.io/Pev7YU'}, pandas.core.series.Series: {'Chap31_y_train.csv': 'https://file.io/npZaEc'}}")
X_train=pd.read_csv('Chap31_X_train.csv',index_col=0)
y_train=pd.read_csv('Chap31_y_train.csv', index_col=0, header=None,squeeze=True)

#%% Exercise 31 - Your first pipeline - again!

# Create pipeline with feature selector and classifier
pipe = Pipeline([
    ('feature_selection', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier(random_state=2))])

# Create a parameter grid
params = {
   'feature_selection__k':[10, 20],
    'clf__n_estimators':[2, 5]}

# Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid=params, cv=3)

# Fit it to the data and print the best value combination
print(grid_search.fit(X_train, y_train).best_params_)

#%%Exercise 32 - Custom scorers in pipelines - init 
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
#import shutil

# Create pipeline with feature selector and classifier
pipe = Pipeline([
    ('feature_selection', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier(random_state=2))])

# Create a parameter grid
params = {
   'feature_selection__k':[10, 20],
    'clf__n_estimators':[2, 5]}

#copy rename Chap31_X_train.csv, Chap31_y_train.csv
#shutil.copyfile('Chap31_X_train.csv', 'Chap32_X_train.csv')
#shutil.copyfile('Chap31_y_train.csv', 'Chap32_y_train.csv')
X_train=pd.read_csv('Chap32_X_train.csv',index_col=0)
y_train=pd.read_csv('Chap32_y_train.csv', index_col=0, header=None,squeeze=True)

#got it from datacamp:
#import inspect
#print(inspect.getsource(my_metric))

def my_metric(y_test, y_est, cost_fp=10.0, cost_fn=1.0):
    tn, fp, fn, tp = confusion_matrix(y_test, y_est).ravel()
    return cost_fp * fp + cost_fn * fn

#%%Exercise 32 - Custom scorers in pipelines

# Create a custom scorer
scorer = make_scorer(roc_auc_score)

# Initialize the CV object
gs = GridSearchCV(pipe, param_grid=params, scoring=scorer)

# Fit it to the data and print the winning combination
print(gs.fit(X_train, y_train).best_params_)

#now for f1_score
# Create a custom scorer
scorer = make_scorer(f1_score)

# Initialise the CV object
gs = GridSearchCV(pipe, param_grid=params, scoring=scorer)

# Fit it to the data and print the winning combination
print(gs.fit(X_train, y_train).best_params_)

#now with a custom metric my_metric()
# Create a custom scorer
scorer = make_scorer(my_metric)

# Initialise the CV object
gs = GridSearchCV(pipe, param_grid=params, scoring=scorer)

# Fit it to the data and print the winning combination
print(gs.fit(X_train, y_train).best_params_)

#%% Exercise 33 - Pickles - init
from sklearn.ensemble import RandomForestClassifier
#from uploadfromdatacamp import saveFromFileIO
import pandas as pd
import pickle

#uploadFromDatacamp(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/2P6tqE',  'X_train.csv': 'https://file.io/h8xukQ'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/AluJpN',  'y_train.csv': 'https://file.io/yAJiGI'}}"
#saveFromFileIO(tobedownloaded, prefix='Chap33_')
X_train=pd.read_csv('Chap33_X_train.csv',index_col=0)
X_test=pd.read_csv('Chap33_X_test.csv',index_col=0)
y_test=pd.read_csv('Chap33_y_test.csv', index_col=0, header=None,squeeze=True)
y_train=pd.read_csv('Chap33_y_train.csv', index_col=0, header=None,squeeze=True)



#%% Exercise 33 - Pickles

# Fit a random forest to the training set
clf = RandomForestClassifier(random_state=42).fit(  X_train, y_train)

# Save it to a file, to be pushed to production
with open('model.pkl', 'wb') as file:
    pickle.dump(clf, file=file)

# Now load the model from file in the production environment
with open('model.pkl', 'rb') as file:
    clf_from_file = pickle.load(file)

# Predict the labels of the test dataset
preds = clf_from_file.predict(X_test)

#%% Exercise 34 - Custom function transformers in pipelines - init
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

#%% Exercise 34 - Custom function transformers in pipelines 
# Define a feature extractor to flag very large values
def more_than_average(X, multiplier=1.0):
  Z = X.copy()
  Z[:,1] = Z[:,1] > multiplier*np.mean(Z[:,1])
  return Z

# Convert your function so that it can be used in a pipeline
pipe = Pipeline([
  ('ft', FunctionTransformer(more_than_average)),
  ('clf', RandomForestClassifier(random_state=2))])

# Optimize the parameter multiplier using GridSearchCV
params = {'ft__multiplier':[1,2,3]}
grid_search = GridSearchCV(pipe, param_grid=params)

print(uploadToFileIO(pipe,proxy="10.225.92.1:80"))

#%%Exercise 35 - Challenge the champion - init
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
#from uploadfromdatacamp import saveFromFileIO
import pandas as pd
import pickle

#uploadFromDatacamp(X_train, X_test, y_train, y_test)
#get model.pkl
#curl_command=" ".join(str(x) for x in ['curl', '-F', "file=@model.pkl", 'https://file.io'])
#sortie_curl = subprocess.getoutput(curl_command)

tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/5ZlMWu',  'X_train.csv': 'https://file.io/kI8Dvv'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/6jHnZ5',  'y_train.csv': 'https://file.io/Qz1YT8'}}"
prefix='Chap35_'
#saveFromFileIO(tobedownloaded, prefix=prefix)
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_test=pd.read_csv(prefix+'y_test.csv', index_col=0, header=None,squeeze=True)
y_train=pd.read_csv(prefix+'y_train.csv', index_col=0, header=None,squeeze=True)

#https://file.io/y1qsln
tobedownloaded="{filename: {'model.pkl': 'https://file.io/y1qsln'}}"
#saveFromFileIO(tobedownloaded)


#%%Exercise 35 - Challenge the champion
# Load the current model from disk
champion = pickle.load(open('model.pkl', 'rb'))

# Fit a Gaussian Naive Bayes to the training data
challenger = GaussianNB().fit(X_train, y_train)

# Print the F1 test scores of both champion and challenger
print(f1_score(y_test, champion.predict(X_test)))
print(f1_score(y_test, challenger.predict(X_test)))

# Write back to disk the best-performing model
with open('model.pkl', 'wb') as file:
    pickle.dump(champion, file=file)

#%% Exercise 36 - Cross-validation statistics - init

#from datacamp
import pickle
from uploadfromdatacamp import saveFromFileIO
import pandas as pd

#with open('pipe.pkl', 'wb') as file:
#    pickle.dump(pipe, file)
#uploadToFileIO_pushto_fileio('pipe.pkl')

# 'https://file.io/fRWOKA'
url='https://file.io/fRWOKA'
tobesaved_as='pipe.pkl'
prefix='Chap36_'
tobedownloaded="{pipeline:{'"+tobesaved_as+"': '"+url+"'}}"
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
with open(prefix+tobesaved_as, 'rb') as file:
    pipe = pickle.load(file)

#uploadToFileIO(X_train,y_train)
#{pandas.core.frame.DataFrame: {'X_train.csv': 'https://file.io/8cNkWJ'}, pandas.core.series.Series: {'y_train.csv': 'https://file.io/6R2JXV'}}
############get files
tobedownloaded="{pandas.core.frame.DataFrame: {'X_train.csv': 'https://file.io/8cNkWJ'}, pandas.core.series.Series: {'y_train.csv': 'https://file.io/6R2JXV'}}"
prefix='Chap36_'
saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
############load objects
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
y_train=pd.read_csv(prefix+'y_train.csv', index_col=0, header=None,squeeze=True)


   
#%% Exercise 36 - Cross-validation statistics
    
