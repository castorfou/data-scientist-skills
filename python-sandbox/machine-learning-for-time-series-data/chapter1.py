# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:59:12 2019

@author: F279814
"""

#%% Exercise - Plotting a time series (I) - init

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(data, data2)
tobedownloaded="{pandas.core.frame.DataFrame: {'data.csv': 'https://file.io/9CEDtA',  'data2.csv': 'https://file.io/sGJc8B'}}"
prefix='data_from_datacamp/ZZZ_Chap11_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

data=pd.read_csv(prefix+'data.csv',index_col=0)
data2=pd.read_csv(prefix+'data2.csv',index_col=0)


#%% Exercise - Plotting a time series (I)

print(data[:5])

# Print the first 5 rows of data2
print(data2[:5])

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(y='data_values', ax=axs[0])
data2.iloc[:1000].plot(y='data_values', ax=axs[1])
plt.show()


#%% Exercise - Plotting a time series (II) - init

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(data, data2)
tobedownloaded="{pandas.core.frame.DataFrame: {'data.csv': 'https://file.io/GeZCNs',  'data2.csv': 'https://file.io/1pyvuZ'}}"
prefix='data_from_datacamp/ZZZ_Chap12_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

data=pd.read_csv(prefix+'data.csv',index_col=0)
data2=pd.read_csv(prefix+'data2.csv',index_col=0)


#%% Exercise - Plotting a time series (II)

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(x='time', y='data_values', ax=axs[0])
data2.iloc[:1000].plot(x='time', y='data_values', ax=axs[1])
plt.show()

#%% Exercise - Fitting a simple model: classification - init

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(data)
tobedownloaded="{pandas.core.frame.DataFrame: {'data.csv': 'https://file.io/xUbTEA'}}"
prefix='data_from_datacamp/ZZZ_Chap13_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

data=pd.read_csv(prefix+'data.csv',index_col=0)



#%% Exercise - Fitting a simple model: classification - init

# Print the first 5 rows for inspection
print(data.head())

from sklearn.svm import LinearSVC

# Construct data for the model
X = data[['petal length (cm)', 'petal width (cm)']]
y = data[['target']]

# Fit the model
model = LinearSVC()
model.fit(X, y)

#%% Exercise - Predicting using a classification model - init

#uploadToFileIO(targets)
tobedownloaded="{pandas.core.frame.DataFrame: {'targets.csv': 'https://file.io/N2gdyP'}}"
prefix='data_from_datacamp/ZZZ_Chap14_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

targets=pd.read_csv(prefix+'targets.csv',index_col=0)


#%% Exercise - Predicting using a classification model

# Create input array
X_predict = targets[['petal length (cm)', 'petal width (cm)']]

# Predict with the model
predictions = model.predict(X_predict)
print(predictions)

# Visualize predictions and actual values
plt.scatter(X_predict['petal length (cm)'], X_predict['petal width (cm)'],
            c=predictions, cmap=plt.cm.coolwarm)
plt.title("Predicted class values")
plt.show()

#%% Exercise - Fitting a simple model: regression - init

#uploadToFileIO(boston)
tobedownloaded="{pandas.core.frame.DataFrame: {'boston.csv': 'https://file.io/i4TdE8'}}"
prefix='data_from_datacamp/ZZZ_Chap15_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

boston=pd.read_csv(prefix+'boston.csv',index_col=0)


#%% Exercise - Fitting a simple model: regression

from sklearn import linear_model

# Prepare input and output DataFrames
X = boston[['AGE']]
y = boston[['RM']]

# Fit the model
model = linear_model.LinearRegression()
model.fit(X,y)

#%% Exercise - Predicting using a regression model - init

from uploadfromdatacamp import loadNDArrayFromCsv


#uploadToFileIO(new_inputs)
tobedownloaded="{numpy.ndarray: {'new_inputs.csv': 'https://file.io/1EBvfb'}}"
prefix='data_from_datacamp/ZZZ_Chap16_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

new_inputs=loadNDArrayFromCsv(prefix+'new_inputs.csv')



#%% Exercise - Predicting using a regression model

# Generate predictions with the model using those inputs
predictions = model.predict(new_inputs.reshape(-1,1))

# Visualize the inputs and predicted values
plt.scatter(new_inputs, predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()


#%% Exercise - Inspecting the classification data - init



#%% Exercise - Inspecting the classification data 

import librosa as lr
from glob import glob

# List all the wav files in the folder
audio_files = glob(data_dir + '/*.wav')

# Read in the first audio file, create the time array
audio, sfreq = lr.load(audio_files[0])
time = np.arange(0, len(audio)) / sfreq

# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()

#%% Exercise - Inspecting the regression data - init

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from uploadfromdatacamp import saveFromFileIO

#prices=pd.read_csv('prices.csv',index_col=0)
#uploadToFileIO(prices)
tobedownloaded="{pandas.core.frame.DataFrame: {'prices.csv': 'https://file.io/1sQ1rA'}}"
prefix='data_from_datacamp/ZZZ_Chap18_'
saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#%% Exercise - Inspecting the regression data

# Read in the data
data = pd.read_csv(prefix+'prices.csv', index_col=0)

# Convert the index of the DataFrame to datetime
data.index = pd.to_datetime(data.index)
print(data.head())

# Loop through each column, plot its values over time
fig, ax = plt.subplots()
for column in data.columns:
    data[column].plot(ax=ax, label=column)
ax.legend()
plt.show()

