# -*- coding: utf-8 -*-
"""
Chapter4 - Estimators
Created on Wed Jul 24 10:48:41 2019

@author: N561507
"""


#%% Exercise - Preparing to train with Estimators - initial data
# Import feature_column and tensorflow
import tensorflow as tf
from tensorflow import feature_column
import pandas as pd
# Load data from csv
housing = pd.read_csv('housing_df.csv',index_col=0)
# Convert to numpy array
print(housing.shape)


#%% Exercise - Preparing to train with Estimators
# Define feature columns for bedrooms and bathrooms
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]

def input_fn():
	# Define the labels
	labels = np.array(housing.price)
	# Define the features
	features = {'bedrooms':np.array(housing['bedrooms']), 
                'bathrooms':np.array(housing['bathrooms'])}
	return features, labels

#%% Exercise - Defining Estimators
from tensorflow import estimator

# Define the model and set the number of steps
model = estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2,2])
model.train(input_fn, steps=1)

#Modify the code to use a LinearRegressor(), remove the hidden_units, and set the number of steps to 2.
# Define the model and set the number of steps
model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)

