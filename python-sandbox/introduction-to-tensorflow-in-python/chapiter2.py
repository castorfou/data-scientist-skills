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



#%% copy paste variables
import pandas as pd

price_df = pd.DataFrame(price)
price_df.to_clipboard()
#predictions_df = pd.DataFrame(predictions)
#predictions_df.to_excel('predictions.xls')
#predictions_df.to_csv('predictions.csv')
#!cat predictions.csv

#%%read predictions numpy array from predictions.csv (got from datacamp)
import pandas as pd
predictions_df=pd.read_csv('predictions.csv')
print(predictions_df.head())
predictions=predictions_df.to_numpy()[:,1]

#%% Loss functions in TensorFlow mse
# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean squared error (mse)
loss = keras.losses.mse(price, predictions)

# Print the mean squared error (mse)
print(loss.numpy())

#%% Loss functions in TensorFlow mae
# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean squared error (mse)
loss = keras.losses.mae(price, predictions)

# Print the mean squared error (mse)
print(loss.numpy())

#%% Exercise - Modifying the loss function
import tensorflow as tf
from tensorflow import Variable,  keras

# Initialize a variable named scalar
scalar = Variable(1.0, tf.float32)

# Define the model
def model(scalar, features = features):
  	return scalar * features

# Define a loss function
def loss_function(scalar, features = features, targets = targets):
	# Compute the predicted values
	predictions = model(scalar, features)
    
	# Return the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Evaluate the loss function and print the loss
print(loss_function(scalar).numpy())

#%% Linear regression in TensorFlow
# Define the targets and features
price = np.array(housing['price'], np.float32)
size = np.array(housing['sqft_living'], np.float32)
# Define the intercept and slope
intercept = tf.Variable(0.1, np.float32)
slope = tf.Variable(0.1, np.float32)
# Define a linear regression model
def linear_regression(intercept, slope, features = size):
	return intercept + features*slope
# Compute the predicted values and loss
def loss_function(intercept, slope, targets = price, features = size):
	predictions = linear_regression(intercept, slope)
	return tf.keras.losses.mse(targets, predictions)
	
# Define an optimization operation
opt = tf.keras.optimizers.Adam()
# Minimize the loss function and print the loss
for j in range(1000):
	opt.minimize(lambda: loss_function(intercept, slope),\
	var_list=[intercept, slope])
	print(loss_function(intercept, slope))
    # Print the trained parameters
	print(intercept.numpy(), slope.numpy())	

#%% Exercise - load price_log and size_log from datacamp | datacamp part
import pandas as pd
size_log_df=pd.DataFrame(size_log)
size_log_df.to_csv('size_log_df.csv')
!cat size_log_df.csv
#copy/paste to size_log.csv on my pc
price_log_df=pd.DataFrame(price_log)
price_log_df.to_csv('price_log_df.csv')
!cat price_log_df.csv
#copy/paste to price_log.csv on my pc

#%% Exercise - load price_log and size_log from datacamp | local part
import pandas as pd
size_df=pd.read_csv('size_log.csv')
size_log=size_df.to_numpy()[:,1]
price_df=pd.read_csv('price_log.csv')
price_log=price_df.to_numpy()[:,1]

#%% Exercise - Set up a linear regression
# Define a linear regression model
def linear_regression(intercept, slope, features = size_log):
	return intercept + features*slope

# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features = size_log, targets = price_log):
	# Set the predicted values
	predictions = linear_regression(intercept, slope, features)
    
    # Return the mean squared error loss
	return keras.losses.mse(targets, predictions)

# Compute the loss for different slope and intercept values
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())

#%% Exercise - Train a linear model
# Initialize an adam optimizer
opt = keras.optimizers.Adam(0.5)

for j in range(500):
	# Apply minimize, pass the loss function, and supply the variables
	opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])

	# Print every 100th value of the loss    
	if j % 100 == 0:
		print(loss_function(intercept, slope).numpy())

import matplotlib.pyplot as plt

#retrieve source code from datacamp
#import inspect
#print(inspect.getsource(plot_results))
def plot_results(intercept, slope):
	size_range = np.linspace(6,14,100)
	price_pred = [intercept+slope*s for s in size_range]
	plt.scatter(size_log, price_log, color = 'black')
	plt.plot(size_range, price_pred, linewidth=3.0, color='red')
	plt.xlabel('log(size)')
	plt.ylabel('log(price)')
	plt.title('Scatterplot of data and fitted regression line')
	plt.show()
    
# Plot data and regression line
plot_results(intercept, slope)
#I had to change options in spyder to vizualise
#Select from the menu Tools > Preferences, then IPython console in the list of categories on the left, then the tab Graphics at the top, and change the Graphics backend from Inline to e.g. Automatic


#%% Multiple linear regression
bedrooms=housing.bedrooms.to_numpy()
print(max(bedrooms))
params=Variable([0.1 , 0.05, 0.02], dtype=tf.float32)

# Define the linear regression model
def linear_regression(params, feature1 = size_log, feature2 = bedrooms):
	return params[0] + feature1*params[1] + feature2*params[2]

# Define the loss function
def loss_function(params, targets = price_log, feature1 = size_log, feature2 = bedrooms):
	# Set the predicted values
	predictions = linear_regression(params, feature1, feature2)  
  
	# Use the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Define the optimize operation
opt = keras.optimizers.Adam()

def print_results(params):
	return print('loss: {:0.3f}, intercept: {:0.3f}, slope_1: {:0.3f}, slope_2: {:0.3f}'.format(loss_function(params).numpy(), params[0].numpy(), params[1].numpy(), params[2].numpy()))


# Perform minimization and print trainable variables
for j in range(10):
	opt.minimize(lambda: loss_function(params), var_list=[params])
	print_results(params)


#%% Training a linear model in batches
# Import tensorflow, pandas, and numpy
import tensorflow as tf
import pandas as pd
import numpy as np
# Define trainable variables
intercept = tf.Variable(0.1, tf.float32)
slope = tf.Variable(0.1, tf.float32)
# Define the model
def linear_regression(intercept, slope, features):
	return intercept + features*slope
# Compute predicted values and return loss function
def loss_function(intercept, slope, targets, features):
	predictions = linear_regression(intercept, slope, features)
	return tf.keras.losses.mse(targets, predictions)
# Define optimization operation
opt = tf.keras.optimizers.Adam()
# Load the data in batches from pandas
for batch in pd.read_csv('kc_house_data.csv', chunksize=100):
	# Extract the target and feature columns
	price_batch = np.array(batch['price'], np.float32)
	size_batch = np.array(batch['sqft_lot'], np.float32)
	# Minimize the loss function
	opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch), 		var_list=[intercept, slope])
# Print parameter values
print(intercept.numpy(), slope.numpy())

#%% Exercise - Preparing to batch train
from tensorflow import float32
# Define the intercept and slope
intercept = Variable(10.0, float32)
slope = Variable(0.5, float32)

# Define the model
def linear_regression(intercept, slope, features):
	# Define the predicted values
	return intercept+slope*features

# Define the loss function
def loss_function(intercept, slope, targets, features):
	# Define the predicted values
	predictions = linear_regression(intercept, slope, features)
			
	# Define the MSE loss
	return keras.losses.mse(targets, predictions)

#%% Exercise - Training a linear model in batches
# Initialize adam optimizer
opt = keras.optimizers.Adam()

# Load data in batches
for batch in pd.read_csv('kc_house_data.csv', chunksize=100):
	size_batch = np.array(batch['sqft_lot'], np.float32)
    
	# Extract the price values for the current batch
	price_batch = np.array(batch['price'], np.float32)

	# Complete the loss, fill in the variable list, and minimize
	opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch), var_list=[intercept, slope])

# Print trained parameters
print(intercept.numpy(), slope.numpy())



