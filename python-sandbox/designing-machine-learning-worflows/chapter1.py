# -*- coding: utf-8 -*-
"""
Chapter 1
Created on Wed Jul 31 15:21:26 2019

@author: N561507
"""

#%%Exercise - Feature engineering - init
non_numeric_columns=['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']
#import pandas as pd
#credit.to_csv('credit.csv')
#!curl -F "file=@credit.csv" https://file.io
#{"success":true,"key":"ilu4i4","link":"https://file.io/ilu4i4","expiry":"14 days"}
import pandas as pd
credit=pd.read_csv('credit.csv', index_col=0)
from sklearn.preprocessing import LabelEncoder

#%%Exercise - Feature engineering

# Inspect the first few lines of your data using head()
credit.head(3)

# Create a label encoder for each column. Encode the values
for column in non_numeric_columns:
    le = LabelEncoder()
    credit[column] = le.fit_transform(credit[column])

# Inspect the data types of the columns of the data frame
print(credit.dtypes)

#%% Exercise - Your first pipeline - init
import pandas as pd
X,y=credit.drop('class',1), credit['class']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
accuracies={'ab': 0.75}

#%% Exercise - Your first pipeline

# Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)

# Create a random forest classifier, fixing the seed to 2
rf_model = RandomForestClassifier(random_state=2).fit(
  X_train, y_train)

# Use it to predict the labels of the test data
rf_predictions = rf_model.predict(X_test)

# Assess the accuracy of both classifiers
accuracies['rf'] = accuracy_score(y_test, rf_predictions)

#%% Exercise - Grid search CV for model complexity
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import KNeighborsClassifier


# Set a range for n_estimators from 10 to 40 in steps of 10
param_grid = {'n_estimators': range(10, 50, 10)}

# Optimize for a RandomForestClassifier() using GridSearchCV
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X, y)
grid.best_params_

# Define a grid for n_estimators ranging from 1 to 10
param_grid = {'n_estimators': range(1, 11)}

# Optimize for a AdaBoostClassifier() using GridSearchCV
grid = GridSearchCV(AdaBoostClassifier(), param_grid, cv=3)
grid.fit(X, y)
grid.best_params_

# Define a grid for n_neighbors with values 10, 50 and 100
param_grid = {'n_neighbors': [10,50,100]}

# Optimize for KNeighborsClassifier() using GridSearchCV
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
grid.fit(X, y)
grid.best_params_

#%% Exercise - Categorical encodings - init
from sklearn.preprocessing import LabelEncoder

#%% Exercise - Categorical encodings
# Create numeric encoding for credit_history
credit_history_num = LabelEncoder().fit_transform(
  credit['credit_history'])

# Create a new feature matrix including the numeric encoding
X_num = pd.concat([X, pd.Series(credit_history_num)], axis=1)

# Create new feature matrix with dummies for credit_history
X_hot = pd.concat(
  [X, pd.get_dummies(credit['credit_history'])], axis=1)

# Compare the number of features of the resulting DataFrames
X_hot.shape[1] > X_num.shape[1]


#%% Exercise - Feature transformations - init
from sklearn.feature_selection import chi2, SelectKBest
import numpy as np

#%% Exercise - Feature transformations
# Function computing absolute difference from column mean
def abs_diff(x):
    return np.abs(x-np.mean(x))

# Apply it to the credit amount and store to new column
credit['diff'] = abs_diff(credit.credit_amount)

# Create a feature selector with chi2 that picks one feature
sk = SelectKBest(chi2, k=1)

# Use the selector to pick between credit_amount and diff
sk.fit(credit[['credit_amount', 'diff']], credit['class'])

# Inspect the results
sk.get_support()

#%% Exercise - Bringing it all together - init
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
import pandas as pd
X_train=pd.read_csv('X_train.csv', index_col=0)
y_train=pd.read_csv('y_train.csv', index_col=0)
X_test=pd.read_csv('X_test.csv', index_col=0)
y_test=pd.read_csv('y_test.csv', index_col=0)



#%% Exercise - Bringing it all together

# Find the best value for max_depth among values 2, 5 and 10
grid_search = GridSearchCV(
  rfc(random_state=1), param_grid={'max_depth':[2,5,10]})
best_value = grid_search.fit(
  X_train, y_train).best_params_['max_depth']

# Using the best value from above, fit a random forest
clf = rfc(
  random_state=1, max_depth=best_value).fit(X_train, y_train)

# Apply SelectKBest with chi2 and pick top 100 features
vt = SelectKBest(chi2, k=100).fit(X_train, y_train)

# Create a new dataset only containing the selected features
X_train_reduced = vt.transform(X_train)
