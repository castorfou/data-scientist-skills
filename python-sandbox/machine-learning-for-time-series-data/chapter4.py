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

#%% Exercise - Cross-validation without shuffling - init



#%% Exercise - Cross-validation without shuffling

# Create KFold cross-validation object
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=False, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr],y[tr])
    
    # Generate predictions on the test data and collect
    prediction = model.predict(X[tt])
    results.append((prediction, tt))
    
# Custom function to quickly visualize predictions
visualize_predictions(results)

#%% Exercise - Time-based cross-validation - init



#%% Exercise - Time-based cross-validation

# Import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

# Create time-series cross-validation object
cv = TimeSeriesSplit(n_splits=10)

# Iterate through CV splits
fig, ax = plt.subplots()
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Plot the training data on each iteration, to see the behavior of the CV
    ax.plot(tr, ii + y[tr])

ax.set(title='Training data on each CV iteration', ylabel='CV iteration')
plt.show()

#%% Exercise - Bootstrapping a confidence interval

from sklearn.utils import resample

def bootstrap_interval(data, percentiles=(2.5, 97.5), n_boots=100):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create our empty array to fill the results
    bootstrap_means = np.zeros([n_boots, data.shape[-1]])
    for ii in range(n_boots):
        # Generate random indices for our data *with* replacement, then take the sample mean
        random_sample = resample(data)
        bootstrap_means[ii] = random_sample.mean(axis=0)
        
    # Compute the percentiles of choice for the bootstrapped means
    percentiles = np.percentile(bootstrap_means, percentiles, axis=0)
    return percentiles

#%% Exercise - Calculating variability in model coefficients - init

import numpy as np

list_feature=['AAPL_lag_1_day', 'YHOO_lag_1_day', 'NVDA_lag_1_day', 'AAPL_lag_2_day',
       'YHOO_lag_2_day', 'NVDA_lag_2_day', 'AAPL_lag_3_day', 'YHOO_lag_3_day',
       'NVDA_lag_3_day', 'AAPL_lag_4_day', 'YHOO_lag_4_day', 'NVDA_lag_4_day']
feature_names=pd.Index(list_feature)
# Fit the model
model = Ridge()

#uploadToFileIO(X, y)
tobedownloaded="{numpy.ndarray: {'X.csv': 'https://file.io/s3rckh',  'y.csv': 'https://file.io/9YfKyx'}}"
prefix='data_from_datacamp/ZZZ_Chap45_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
X=loadNDArrayFromCsv(prefix+'X.csv')
y=loadNDArrayFromCsv(prefix+'y.csv')



#%% Exercise - Calculating variability in model coefficients

# Iterate through CV splits
n_splits = 100
cv = TimeSeriesSplit(n_splits=n_splits)

# Create empty array to collect coefficients
coefficients = np.zeros([n_splits, X.shape[1]])

for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Fit the model on training data and collect the coefficients
    model.fit(X[tr], y[tr])
    coefficients[ii] = model.coef_
    
# Calculate a confidence interval around each coefficient
bootstrapped_interval = bootstrap_interval(coefficients)

# Plot it
fig, ax = plt.subplots()
ax.scatter(feature_names, bootstrapped_interval[0], marker='_', lw=3)
ax.scatter(feature_names, bootstrapped_interval[1], marker='_', lw=3)
ax.set(title='95% confidence interval for model coefficients')
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()    


#%% Exercise - Visualizing model score variability over time - init
import inspect
#print(inspect.getsource(my_pearsonr))
from functools import partial

def my_pearsonr(est, X, y):
    return np.corrcoef(est.predict(X).squeeze(), y.squeeze())[1, 0]
list_index=['2010-04-05', '2010-04-28', '2010-05-21', '2010-06-16',
               '2010-07-12', '2010-08-04', '2010-08-27', '2010-09-22',
               '2010-10-15', '2010-11-09', '2010-12-03', '2010-12-29',
               '2011-01-24', '2011-02-16', '2011-03-14', '2011-04-06',
               '2011-05-02', '2011-05-25', '2011-06-20', '2011-07-14',
               '2011-08-08', '2011-08-31', '2011-09-26', '2011-10-19',
               '2011-11-11', '2011-12-07', '2012-01-03', '2012-01-27',
               '2012-02-22', '2012-03-16', '2012-04-11', '2012-05-04',
               '2012-05-30', '2012-06-22', '2012-07-18', '2012-08-10',
               '2012-09-05', '2012-09-28', '2012-10-23', '2012-11-19',
               '2012-12-13', '2013-01-09', '2013-02-04', '2013-02-28',
               '2013-03-25', '2013-04-18', '2013-05-13', '2013-06-06',
               '2013-07-01', '2013-07-25', '2013-08-19', '2013-09-12',
               '2013-10-07', '2013-10-30', '2013-11-22', '2013-12-18',
               '2014-01-14', '2014-02-07', '2014-03-05', '2014-03-28',
               '2014-04-23', '2014-05-16', '2014-06-11', '2014-07-07',
               '2014-07-30', '2014-08-22', '2014-09-17', '2014-10-10',
               '2014-11-04', '2014-11-28', '2014-12-23', '2015-01-20',
               '2015-02-12', '2015-03-10', '2015-04-02', '2015-04-28',
               '2015-05-21', '2015-06-16', '2015-07-10', '2015-08-04',
               '2015-08-27', '2015-09-22', '2015-10-15', '2015-11-09',
               '2015-12-03', '2015-12-29', '2016-01-25', '2016-02-18',
               '2016-03-14', '2016-04-07', '2016-05-02', '2016-05-25',
               '2016-06-20', '2016-07-14', '2016-08-08', '2016-08-31',
               '2016-09-26', '2016-10-19', '2016-11-11', '2016-12-07']
times_scores=pd.Index(list_index,dtype='datetime64[ns]')

#%% Exercise - Visualizing model score variability over time

from sklearn.model_selection import cross_val_score

# Generate scores for each split to see how the model performs over time
scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)

# Convert to a Pandas Series object
scores_series = pd.Series(scores, index=times_scores, name='score')

# Bootstrap a rolling confidence interval for the mean score
scores_lo = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=2.5))
scores_hi = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=97.5))

# Plot the results
fig, ax = plt.subplots()
scores_lo.plot(ax=ax, label="Lower confidence interval")
scores_hi.plot(ax=ax, label="Upper confidence interval")
ax.legend()
plt.show()

#%% Exercise - Accounting for non-stationarity - init

#%% Exercise - Accounting for non-stationarity

# Pre-initialize window sizes
window_sizes = [25, 50, 75, 100]

# Create an empty DataFrame to collect the stores
all_scores = pd.DataFrame(index=times_scores)

# Generate scores for each split to see how the model performs over time
for window in window_sizes:
    # Create cross-validation object using a limited lookback window
    cv = TimeSeriesSplit(n_splits=100, max_train_size=window)
    
    # Calculate scores across all CV splits and collect them in a DataFrame
    this_scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)
    all_scores['Length {}'.format(window)] = this_scores

# Visualize the scores
ax = all_scores.rolling(10).mean().plot(cmap=plt.cm.coolwarm)
ax.set(title='Scores for multiple windows', ylabel='Correlation (r)')
plt.show()