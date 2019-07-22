# -*- coding: utf-8 -*-
"""
Chapiter2

Created on Mon Jul 22 08:44:23 2019

@author: N561507
"""

#%%	How to import and convert data

# Import numpy and pandas
import numpy as np
import pandas as pd
# Load data from csv
housing = pd.read_csv('kc_housing.csv')
# Convert to numpy array
housing = np.array(housing)
df = pd.DataFrame(housing)


#%% Cannot see numpy datatypes in variable explorer
import numpy as np
hello = [1,2,3,4,5]
hello = np.array(hello)
#(Spyder developer here) Support for object arrays will be added in Spyder 4, to be released in 2019.
#Spyder 4 will be released (most probably) in September 2019. – Carlos Cordoba Jun 20 at 10:58

#%% Setting the data type
# suite de la cell "How to import and convert data"
# Load KC dataset
housing = pd.read_csv('kc_housing.csv')
# Convert price column to float32
price = np.array(housing['price'], np.float32)
# Convert waterfront column to Boolean
waterfront = np.array(housing['waterfront'], np.bool)

#%% Setting the data type, casting with tf
import tensorflow as tf
# Load KC dataset
housing = pd.read_csv('kc_housing.csv')
# Convert price column to float32
price = tf.cast(housing['price'], tf.float32)
# Convert waterfront column to Boolean
waterfront = tf.cast(housing['waterfront'], tf.bool)
print(type(waterfront))
#<class 'tensorflow.python.framework.ops.Tensor'>

#%% Exercise - Load data using pandas
# Import pandas under the alias pd
import pandas as pd

# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'

# Load the dataset as a dataframe named housing 
housing = pd.read_csv(data_path)

# Print the price column of housing
print(housing.price)

#%% Exercice - Setting the data type
# Import numpy and tensorflow with their standard aliases
import numpy as np
import tensorflow as tf

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# Print price and waterfront
print(price)
print(waterfront)

#%% get version of tensorflow
import tensorflow as tf
print(tf.__version__)
