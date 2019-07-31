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