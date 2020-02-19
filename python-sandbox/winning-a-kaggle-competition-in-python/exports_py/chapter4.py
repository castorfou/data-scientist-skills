#!/usr/bin/env python
# coding: utf-8

# # Baseline model
# 

# ## Replicate validation score
# You've seen both validation and Public Leaderboard scores in the video. However, the code examples are available only for the test data. To get the validation scores you have to repeat the same process on the holdout set.
# 
# Throughout this chapter, you will work with New York City Taxi competition data. The problem is to predict the fare amount for a taxi ride in New York City. The competition metric is the root mean squared error.
# 
# The first goal is to evaluate the Baseline model on the validation data. You will replicate the simplest Baseline based on the mean of "fare_amount". Recall that as a validation strategy we used a 30% holdout split with validation_train as train and validation_test as holdout DataFrames. Both of them are available in your workspace.

# ### init

# In[2]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(validation_train, validation_test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'validation_test.csv': 'https://file.io/hmbPQr',
  'validation_train.csv': 'https://file.io/lEmlvU'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
validation_test = pd.read_csv(prefix+'validation_test.csv',index_col=0)
validation_train = pd.read_csv(prefix+'validation_train.csv',index_col=0)


# In[3]:


validation_train.head()


# ### code

# In[5]:


import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculate the mean fare_amount on the validation_train data
naive_prediction = np.mean(validation_train['fare_amount'])

# Assign naive prediction to all the holdout observations
validation_test['pred'] = naive_prediction

# Measure the local RMSE
rmse = sqrt(mean_squared_error(validation_test['fare_amount'], validation_test['pred']))
print('Validation RMSE for Baseline I model: {:.3f}'.format(rmse))


# ## Baseline based on the date
# We've already built 3 different baseline models. To get more practice, let's build a couple more. The first model is based on the grouping variables. It's clear that the ride fare could depend on the part of the day. For example, prices could be higher during the rush hours.
# 
# Your goal is to build a baseline model that will assign the average "fare_amount" for the corresponding hour. For now, you will create the model for the whole train data and make predictions for the test dataset.
# 
# The train and test DataFrames are available in your workspace. Moreover, the "pickup_datetime" column in both DataFrames is already converted to a datetime object for you.

# ### init

# In[13]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(train, test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'test.csv': 'https://file.io/SdAPiJ',
  'train.csv': 'https://file.io/mU5wVE'}}
"""
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
test = pd.read_csv(prefix+'test.csv',index_col=0, parse_dates=['pickup_datetime'])
train = pd.read_csv(prefix+'train.csv',index_col=0, parse_dates=['pickup_datetime'])


# In[14]:


train.head()


# In[15]:


train.info()


# ### code

# In[16]:


# Get pickup hour from the pickup_datetime column
train['hour'] = train['pickup_datetime'].dt.hour
test['hour'] = test['pickup_datetime'].dt.hour

# Calculate average fare_amount grouped by pickup hour 
hour_groups = train.groupby('hour')['fare_amount'].mean()

# Make predictions on the test set
test['fare_amount'] = test.hour.map(hour_groups)

# Write predictions
test[['id','fare_amount']].to_csv(prefix+'hour_mean_sub.csv', index=False)


# ## Baseline based on the gradient boosting
# Let's build a final baseline based on the Random Forest. You've seen a huge score improvement moving from the grouping baseline to the Gradient Boosting in the video. Now, you will use sklearn's Random Forest to further improve this score.
# 
# The goal of this exercise is to take numeric features and train a Random Forest model without any tuning. After that, you could make test predictions and validate the result on the Public Leaderboard. Note that you've already got an "hour" feature which could also be used as an input to the model.

# ### init

# In[19]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(train, test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'test.csv': 'https://file.io/L97IDp',
  'train.csv': 'https://file.io/NjXqT8'}}
"""
prefixToc='1.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
test = pd.read_csv(prefix+'test.csv',index_col=0, parse_dates=['pickup_datetime'])
train = pd.read_csv(prefix+'train.csv',index_col=0, parse_dates=['pickup_datetime'])


# In[22]:


train.select_dtypes(include='number')


# ### code

# In[24]:


from sklearn.ensemble import RandomForestRegressor

# Select only numeric features
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
            'dropoff_latitude', 'passenger_count', 'hour']

# Train a Random Forest model
rf = RandomForestRegressor()
rf.fit(train[features], train.fare_amount)

# Make predictions on the test data
test['fare_amount'] = rf.predict(test[features])

# Write predictions
test[['id','fare_amount']].to_csv(prefix+'rf_sub.csv', index=False)


# # Hyperparameter tuning
# 

# ## Grid search
# Recall that we've created a baseline Gradient Boosting model in the previous lesson. Your goal now is to find the best max_depth hyperparameter value for this Gradient Boosting model. This hyperparameter limits the number of nodes in each individual tree. You will be using K-fold cross-validation to measure the local performance of the model for each hyperparameter value.
# 
# You're given a function get_cv_score(), which takes the train dataset and dictionary of the model parameters as arguments and returns the overall validation RMSE score over 3-fold cross-validation.

# ### init

# In[32]:


""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(get_cv_score)
"""
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
def get_cv_score(train, params):
    # Create KFold object
    kf = KFold(n_splits=3, shuffle=True, random_state=123)

    rmse_scores = []
    
    # Loop through each split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    
        # Train a Gradient Boosting model
        gb = GradientBoostingRegressor(random_state=123, **params).fit(cv_train[features], cv_train.fare_amount)
    
        # Make predictions on the test data
        pred = gb.predict(cv_test[features])
    
        fold_score = np.sqrt(mean_squared_error(cv_test['fare_amount'], pred))
        rmse_scores.append(fold_score)
    
    return np.round(np.mean(rmse_scores) + np.std(rmse_scores), 5)


# In[33]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(train )
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'train.csv': 'https://file.io/MT19wB'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
train = pd.read_csv(prefix+'train.csv',index_col=0)


# ### code

# In[34]:


# Possible max depth values
max_depth_grid = [3,6,9,12,15]
results = {}

# For each value in the grid
for max_depth_candidate in max_depth_grid:
    # Specify parameters for the model
    params = {'max_depth': max_depth_candidate}

    # Calculate validation score for a particular hyperparameter
    validation_score = get_cv_score(train, params)

    # Save the results for each max depth value
    results[max_depth_candidate] = validation_score   
print(results)


# ## 2D grid search
# The drawback of tuning each hyperparameter independently is a potential dependency between different hyperparameters. The better approach is to try all the possible hyperparameter combinations. However, in such cases, the grid search space is rapidly expanding. For example, if we have 2 parameters with 10 possible values, it will yield 100 experiment runs.
# 
# Your goal is to find the best hyperparameter couple of max_depth and subsample for the Gradient Boosting model. subsample is a fraction of observations to be used for fitting the individual trees.
# 
# You're given a function get_cv_score(), which takes the train dataset and dictionary of the model parameters as arguments and returns the overall validation RMSE score over 3-fold cross-validation.

# ### code

# In[36]:


import itertools

# Hyperparameter grids
max_depth_grid = [3,5,7]
subsample_grid = [0.8,0.9,1.0]
results = {}

# For each couple in the grid
for max_depth_candidate, subsample_candidate in itertools.product(max_depth_grid, subsample_grid):
    params = {'max_depth': max_depth_candidate,
              'subsample': subsample_candidate}
    validation_score = get_cv_score(train, params)
    # Save the results for each couple
    results[(max_depth_candidate, subsample_candidate)] = validation_score   
print(results)


# # Model ensembling
# 

# ## Model blending
# You will start creating model ensembles with a blending technique.
# 
# Your goal is to train 2 different models on the New York City Taxi competition data. Make predictions on the test data and then blend them using a simple arithmetic mean.
# 
# The train and test DataFrames are already available in your workspace. features is a list of columns to be used for training and it is also available in your workspace. The target variable name is "fare_amount".

# ### init

# In[38]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(train, test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'test.csv': 'https://file.io/b2Veyh',
  'train.csv': 'https://file.io/LX4SEh'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
test = pd.read_csv(prefix+'test.csv',index_col=0)
train = pd.read_csv(prefix+'train.csv',index_col=0)


# In[39]:


features = ['pickup_longitude',
 'pickup_latitude',
 'dropoff_longitude',
 'dropoff_latitude',
 'passenger_count',
 'distance_km',
 'hour']


# ### code

# In[40]:


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Train a Gradient Boosting model
gb = GradientBoostingRegressor().fit(train[features], train.fare_amount)

# Train a Random Forest model
rf = RandomForestRegressor().fit(train[features], train.fare_amount)

# Make predictions on the test data
test['gb_pred'] = gb.predict(test[features])
test['rf_pred'] = rf.predict(test[features])

# Find mean of model predictions
test['blend'] = (test['gb_pred'] + test['rf_pred']) / 2
print(test[['gb_pred', 'rf_pred', 'blend']].head(3))


# ## Model stacking I
# Now it's time for stacking. To implement the stacking approach, you will follow the 6 steps we've discussed in the previous video:
# 
# - Split train data into two parts
# - Train multiple models on Part 1
# - Make predictions on Part 2
# - Make predictions on the test data
# - Train a new model on Part 2 using predictions as features
# - Make predictions on the test data using the 2nd level model
# 
# train and test DataFrames are already available in your workspace. features is a list of columns to be used for training on the Part 1 data and it is also available in your workspace. Target variable name is "fare_amount".

# ### code

# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Split train data into two parts
part_1, part_2 = train_test_split(train, test_size=0.5, random_state=123)

# Train a Gradient Boosting model on Part 1
gb = GradientBoostingRegressor().fit(part_1[features], part_1.fare_amount)

# Train a Random Forest model on Part 1
rf = RandomForestRegressor().fit(part_1[features], part_1.fare_amount)


# In[43]:


# Make predictions on the Part 2 data
part_2['gb_pred'] = gb.predict(part_2[features])
part_2['rf_pred'] = rf.predict(part_2[features])

# Make predictions on the test data
test['gb_pred'] = gb.predict(test[features])
test['rf_pred'] = rf.predict(test[features])


# ## Model stacking II
# OK, what you've done so far in the stacking implementation:
# 
# - Split train data into two parts
# - Train multiple models on Part 1
# - Make predictions on Part 2
# - Make predictions on the test data
# 
# Now, your goal is to create a second level model using predictions from steps 3 and 4 as features. So, this model is trained on Part 2 data and then you can make stacking predictions on the test data.
# 
# part_2 and test DataFrames are already available in your workspace. Gradient Boosting and Random Forest predictions are stored in these DataFrames under the names "gb_pred" and "rf_pred", respectively.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(part_2 , test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'part_2.csv': 'https://file.io/OGVcb8',
  'test.csv': 'https://file.io/i7krVF'}}
"""
prefixToc='3.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
part_2 = pd.read_csv(prefix+'part_2.csv',index_col=0)
test = pd.read_csv(prefix+'test.csv',index_col=0)


# In[2]:


part_2.head()


# ### code

# In[3]:


from sklearn.linear_model import LinearRegression

# Create linear regression model without the intercept
lr = LinearRegression(fit_intercept=False)

# Train 2nd level model on the Part 2 data
lr.fit(part_2[['gb_pred', 'rf_pred']], part_2.fare_amount)

# Make stacking predictions on the test data
test['stacking'] = lr.predict(test[['gb_pred', 'rf_pred']])

# Look at the model coefficients
print(lr.coef_)


# # Final tips

# ## Testing Kaggle forum ideas
# Unfortunately, not all the Forum posts and Kernels are necessarily useful for your model. So instead of blindly incorporating ideas into your pipeline, you should test them first.
# 
# You're given a function get_cv_score(), which takes a train dataset as an argument and returns the overall validation root mean squared error over 3-fold cross-validation. The train DataFrame is already available in your workspace.
# 
# You should try different suggestions from the Kaggle Forum and check whether they improve your validation score.

# ### init

# In[4]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(train)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'train.csv': 'https://file.io/GxU6cU'}}
"""
prefixToc='4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
train = pd.read_csv(prefix+'train.csv',index_col=0)


# In[17]:


""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(get_cv_score)
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error
def get_cv_score(train):
    features = ['pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude',
            'passenger_count', 'distance_km', 'hour', 'weird_feature']
    
    features = [x for x in features if x in train.columns]
  
    # Create KFold object
    kf = KFold(n_splits=3, shuffle=True, random_state=123)

    rmse_scores = []
    
    # Loop through each split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    
        # Train a Gradient Boosting model
        gb = GradientBoostingRegressor(random_state=123).fit(cv_train[features], cv_train.fare_amount)
    
        # Make predictions on the test data
        pred = gb.predict(cv_test[features])
    
        fold_score = np.sqrt(mean_squared_error(cv_test['fare_amount'], pred))
        rmse_scores.append(fold_score)
    
    return np.round(np.mean(rmse_scores) + np.std(rmse_scores), 5)


# In[18]:


train.head()


# ### code

# This first suggestion worked. Suggestion 2: Sum of "pickup_latitude" and "distance_km" is a good feature. Let's try it!
# 

# In[19]:


# Delete passenger_count column
new_train_1 = train.drop('passenger_count', axis=1)

# Compare validation scores
initial_score = get_cv_score(train)
new_score = get_cv_score(new_train_1)

print('Initial score is {} and the new score is {}'.format(initial_score, new_score))


# This first suggestion worked. Suggestion 2: Sum of "pickup_latitude" and "distance_km" is a good feature. Let's try it!
# 

# In[21]:


# Create copy of the initial train DataFrame
new_train_2 = train.copy()

# Find sum of pickup latitude and ride distance
new_train_2['weird_feature'] = new_train_2.pickup_latitude + new_train_2.distance_km

# Compare validation scores
initial_score = get_cv_score(train)
new_score = get_cv_score(new_train_2)

print('Initial score is {} and the new score is {}'.format(initial_score, new_score))


# In[ ]:




