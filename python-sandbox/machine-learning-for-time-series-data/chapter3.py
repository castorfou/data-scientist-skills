# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:38:19 2019

@author: F279814
"""

#%% Exercise - Introducing the dataset - init

import pandas as pd
import numpy as np
from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(prices)
tobedownloaded="{pandas.core.frame.DataFrame: {'prices.csv': 'https://file.io/eq6I8r'}}"
prefix='data_from_datacamp/ZZZ_Chap31_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
prices=pd.read_csv(prefix+'prices.csv',index_col=0)

import matplotlib.pyplot as plt


#%% Exercise - Introducing the dataset

# Plot the raw values over time
prices.plot()
plt.show()

# Scatterplot with one company per axis
prices.plot.scatter('EBAY','YHOO')
plt.show()

# Scatterplot with color relating to time
prices.plot.scatter('EBAY', 'YHOO', c=np.arange(prices.shape[0]), 
                    cmap=plt.cm.viridis, colorbar=True)
plt.show()

#%% Exercise - Fitting a simple regression model - init

import pandas as pd
from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(all_prices)
tobedownloaded="{pandas.core.frame.DataFrame: {'data.csv': 'https://file.io/8cNv4W'}}"
prefix='data_from_datacamp/ZZZ_Chap32_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
all_prices=pd.read_csv(prefix+'data.csv',index_col=0)


#%% Exercise - Fitting a simple regression model

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Use stock symbols to extract training data
X = all_prices[['NVDA', 'EBAY','YHOO']]
y = all_prices[['AAPL']]

# Fit and score the model with cross-validation
scores = cross_val_score(Ridge(), X, y, cv=3)
print(scores)

#%% Exercise - Visualizing predicted values - init

import numpy as np
from uploadfromdatacamp import saveFromFileIO, loadNDArrayFromCsv

#uploadToFileIO(X,y)
tobedownloaded="{numpy.ndarray: {'X.csv': 'https://file.io/g9WxGI',  'y.csv': 'https://file.io/c6TnaL'}}"
prefix='data_from_datacamp/ZZZ_Chap33_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

X=loadNDArrayFromCsv(prefix+'X.csv')
y=loadNDArrayFromCsv(prefix+'y.csv')


#%% Exercise - Visualizing predicted values

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=.8, shuffle=False, random_state=1)

# Fit our model and generate predictions
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)

# Visualize our predictions along with the "true" values, and print the score
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, color='k', lw=3)
ax.plot(predictions, color='r', lw=2)
plt.show()

#%% Exercise - Visualizing messy data - init
import pandas as pd
from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(prices)
tobedownloaded="{pandas.core.frame.DataFrame: {'prices.csv': 'https://file.io/LVzeR7'}}"
prefix='data_from_datacamp/ZZZ_Chap34_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
prices=pd.read_csv(prefix+'prices.csv',index_col=0)

import matplotlib.pyplot as plt

#%% Exercise - Visualizing messy data

# Visualize the dataset
prices.plot(legend=False)
plt.tight_layout()
plt.show()

# Count the missing values of each time series
missing_values = prices.isna().sum()
print(missing_values)

#%% Exercise - Imputing missing values - init

#uploadToFileIO(prices)
tobedownloaded="{pandas.core.frame.DataFrame: {'prices.csv': 'https://file.io/CWY3H5'}}"
prefix='data_from_datacamp/ZZZ_Chap35_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
prices=pd.read_csv(prefix+'prices.csv',index_col=0,parse_dates=True)

import matplotlib.pyplot as plt


#%% Exercise - Imputing missing values - init

# Create a function we'll use to interpolate and plot
def interpolate_and_plot(prices, interpolation):

    # Create a boolean mask for missing values
    missing_values = prices.isna()

    # Interpolate the missing values
    prices_interp = prices.interpolate(interpolation)

    # Plot the results, highlighting the interpolated values in black
    fig, ax = plt.subplots(figsize=(10, 5))
    prices_interp.plot(color='k', alpha=.6, ax=ax, legend=False)
    
    # Now plot the interpolated values on top in red
    prices_interp[missing_values].plot(ax=ax, color='r', lw=3, legend=False)
    plt.show()


# Interpolate using the latest non-missing value
interpolation_type = 'zero'
interpolate_and_plot(prices, interpolation_type)

# Interpolate linearly
interpolation_type = 'linear'
interpolate_and_plot(prices, interpolation_type)

# Interpolate with a quadratic function
interpolation_type = 'quadratic'
interpolate_and_plot(prices, interpolation_type)    
   

#%% Exercise - Transforming raw data - init

import pandas as pd
from uploadfromdatacamp import saveFromFileIO
import numpy as np

#uploadToFileIO(prices)
tobedownloaded="{pandas.core.frame.DataFrame: {'prices.csv': 'https://file.io/XGOfyF'}}"
prefix='data_from_datacamp/ZZZ_Chap36_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
prices=pd.read_csv(prefix+'prices.csv',index_col=0,parse_dates=True)

import matplotlib.pyplot as plt



#%% Exercise - Transforming raw data

# Your custom function
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot
prices_perc = prices.rolling(20).apply(percent_change)
prices_perc.loc["2014":"2015"].plot()
plt.show()

#%% Exercise - Handling outliers - init


#%% Exercise - Handling outliers

def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))
    
    # Calculate a mask for the differences that are > 3 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std(series) * 3)
    
    # Replace these values with the median accross the data
    series[this_mask] = np.nanmedian(series)
    return series

# Apply your preprocessing function to the timeseries and plot the results
prices_perc = prices_perc.apply(replace_outliers)
prices_perc.loc["2014":"2015"].plot()
plt.show()

#%% Exercise - Engineering multiple rolling features at once - init

import pandas as pd
from uploadfromdatacamp import saveFromFileIO
import numpy as np

#uploadToFileIO(prices_perc)
tobedownloaded="{pandas.core.series.Series: {'prices_perc.csv': 'https://file.io/WjiLvc'}}"
prefix='data_from_datacamp/ZZZ_Chap37_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
prices_perc=pd.read_csv(prefix+'prices_perc.csv',index_col=0, header=None,squeeze=True,parse_dates=True)

import matplotlib.pyplot as plt


#%% Exercise - Engineering multiple rolling features at once

# Define a rolling window with Pandas, excluding the right-most datapoint of the window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')

# Define the features you'll calculate for each window
#Define a list consisting of four features you will calculate: 
#the minimum, maximum, mean, and standard deviation (in that order).
features_to_calculate = [np.min, np.max, np.mean, np.std]

# Calculate these features for your rolling window object
features = prices_perc_rolling.aggregate(features_to_calculate)

# Plot the results
ax = features.loc[:"2011-01"].plot()
prices_perc.loc[:"2011-01"].plot(ax=ax, color='k', alpha=.2, lw=3)
ax.legend(loc=(1.01, .6))
plt.show()

#%% Exercise - Percentiles and partial functions - init



#%% Exercise - Percentiles and partial functions

# Import partial from functools
from functools import partial
percentiles = [1, 10, 25, 50, 75, 90, 99]

# Use a list comprehension to create a partial function for each quantile
percentile_functions = [partial(np.percentile, q=percentile) for percentile in percentiles]

# Calculate each of these quantiles on the data using a rolling window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')
features_percentiles = prices_perc_rolling.aggregate(percentile_functions)

# Plot a subset of the result
ax = features_percentiles.loc[:"2011-01"].plot(cmap=plt.cm.viridis)
ax.legend(percentiles, loc=(1.01, .5))
plt.show()

#%% Exercise - Using "date" information - init

import pandas as pd
from uploadfromdatacamp import saveFromFileIO
import numpy as np

#uploadToFileIO(prices)
tobedownloaded="{pandas.core.frame.DataFrame: {'prices.csv': 'https://file.io/fEMzF1',  'prices_perc.csv': 'https://file.io/rRl62b'}}"
prefix='data_from_datacamp/ZZZ_Chap38_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
prices=pd.read_csv(prefix+'prices.csv',index_col=0, parse_dates=True)
prices_perc=pd.read_csv(prefix+'prices_perc.csv',index_col=0, parse_dates=True)

import matplotlib.pyplot as plt



#%% Exercise - Using "date" information
# Extract date features from the data, add them as columns
prices_perc['day_of_week'] = prices_perc.index.weekday
prices_perc['week_of_year'] = prices_perc.index.week
prices_perc['month_of_year'] = prices_perc.index.month

# Print prices_perc
print(prices_perc)