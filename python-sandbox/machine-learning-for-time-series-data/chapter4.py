# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:08:42 2019

@author: F279814
"""

#%% Exercise - Creating time-shifted features - init

import pandas as pd
from uploadfromdatacamp import saveFromFileIO
import numpy as np

#uploadToFileIO(prices_perc)
tobedownloaded="{pandas.core.series.Series: {'prices_perc.csv': 'https://file.io/xgQvl6'}}"
prefix='data_from_datacamp/ZZZ_Chap41_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
prices_perc=pd.read_csv(prefix+'prices_perc.csv',index_col=0, parse_dates=True)

import matplotlib.pyplot as plt


#%% Exercise - Creating time-shifted features

# These are the "time lags"
shifts = np.arange(1, 11).astype(int)

# Use a dictionary comprehension to create name: value pairs, one pair per shift
shifted_data = {"lag_{}_day".format(day_shift): prices_perc.shift(day_shift) for day_shift in shifts}

# Convert into a DataFrame for subsequent use
prices_perc_shifted = pd.DataFrame(shifted_data)

# Plot the first 100 samples of each
ax = prices_perc_shifted.iloc[:100].plot(cmap=plt.cm.viridis)
prices_perc.iloc[:100].plot(color='r', lw=2)
ax.legend(loc='best')
plt.show()

#%% Exercise - Special case: Auto-regressive models - init

import pandas as pd
from uploadfromdatacamp import saveFromFileIO
import numpy as np
from sklearn.linear_model import Ridge

#uploadToFileIO(prices_perc, prices_perc_shifted)
tobedownloaded="{pandas.core.series.Series: {'prices_perc.csv': 'https://file.io/0dRcPR'}, pandas.core.frame.DataFrame: {'prices_perc_shifted.csv': 'https://file.io/vXjBjZ'}}"
prefix='data_from_datacamp/ZZZ_Chap42_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
prices_perc_shifted=pd.read_csv(prefix+'prices_perc_shifted.csv',index_col=0, parse_dates=True)
prices_perc=pd.read_csv(prefix+'prices_perc.csv',index_col=0, header=None,squeeze=True,parse_dates=True)

import matplotlib.pyplot as plt


#%% Exercise - Special case: Auto-regressive models

# Replace missing values with the median for each column
X = prices_perc_shifted.fillna(np.nanmedian(prices_perc_shifted))
y = prices_perc.fillna(np.nanmedian(prices_perc))

# Fit the model
model = Ridge()
model.fit(X, y)

#%% Exercise - Visualize regression coefficients - init



#%% Exercise - Visualize regression coefficients

def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(names, coefs)
    ax.set(xlabel='Coefficient name', ylabel='Coefficient value')
    
    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax

# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, X.columns.to_list() , ax=axs[1])
plt.show()

#%% Exercise - Auto-regression with a smoother time series - init

import pandas as pd
from uploadfromdatacamp import saveFromFileIO
import numpy as np
from sklearn.linear_model import Ridge

#uploadToFileIO(prices_perc_shifted)
tobedownloaded="{pandas.core.frame.DataFrame: {'prices_perc_shifted.csv': 'https://file.io/ksZRv9'}}"
prefix='data_from_datacamp/ZZZ_Chap43_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
prices_perc_shifted=pd.read_csv(prefix+'prices_perc_shifted.csv',index_col=0, parse_dates=True)

import matplotlib.pyplot as plt

#uploadToFileIO(prices_perc)
tobedownloaded="{pandas.core.series.Series: {'prices_perc.csv': 'https://file.io/1YXPui'}}"
prefix='data_from_datacamp/ZZZ_Chap43_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
prices_perc=pd.read_csv(prefix+'prices_perc.csv',index_col=0, header=None,squeeze=True,parse_dates=True)

# Replace missing values with the median for each column
X = prices_perc_shifted.fillna(np.nanmedian(prices_perc_shifted))
y = prices_perc.fillna(np.nanmedian(prices_perc))

# Fit the model
model = Ridge()
model.fit(X, y)

#%% Exercise - Auto-regression with a smoother time series 

# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, prices_perc_shifted.columns, ax=axs[1])
plt.show()

#%% Exercise - Cross-validation with shuffling - init

import pandas as pd
from uploadfromdatacamp import saveFromFileIO, loadNDArrayFromCsv
import numpy as np
from sklearn.linear_model import Ridge

#uploadToFileIO(X, y)
tobedownloaded="{numpy.ndarray: {'X.csv': 'https://file.io/DliEUT',  'y.csv': 'https://file.io/cBhU6F'}}"
prefix='data_from_datacamp/ZZZ_Chap44_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
X=loadNDArrayFromCsv(prefix+'X.csv')
y=loadNDArrayFromCsv(prefix+'y.csv')


import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Fit the model
model = Ridge()
model.fit(X, y)

#print(inspect.getsource(visualize_predictions))
def visualize_predictions(results):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Loop through our model results to visualize them
    for ii, (prediction, score, indices) in enumerate(results):
        # Plot the predictions of the model in the order they were generated
        offset = len(prediction) * ii
        axs[0].scatter(np.arange(len(prediction)) + offset, prediction, label='Iteration {}'.format(ii))
        
        # Plot the predictions of the model according to how time was ordered
        axs[1].scatter(indices, prediction)
    axs[0].legend(loc="best")
    axs[0].set(xlabel="Test prediction number", title="Predictions ordered by test prediction number")
    axs[1].set(xlabel="Time", title="Predictions ordered by time")
    plt.show()

#%% Exercise - Cross-validation with shuffling
# Import ShuffleSplit and create the cross-validation object
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr], y[tr])
    
    # Generate predictions on the test data, score the predictions, and collect
    prediction = model.predict(X[tt])
    score = r2_score(y[tt], prediction)
    results.append((prediction, score, tt))

# Custom function to quickly visualize predictions
visualize_predictions(results)