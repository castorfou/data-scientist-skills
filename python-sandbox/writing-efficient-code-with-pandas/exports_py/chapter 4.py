#!/usr/bin/env python
# coding: utf-8

# # Data transformation using .groupby().transform
# 

# ## The min-max normalization using .transform()
# A very common operation is the min-max normalization. It consists in rescaling our value of interest by deducting the minimum value and dividing the result by the difference between the maximum and the minimum value. For example, to rescale student's weight data spanning from 160 pounds to 200 pounds, you subtract 160 from each student's weight and divide the result by 40 (200 - 160).
# 
# You're going to define and apply the min-max normalization to all the numerical variables in the restaurant data. You will first group the entries by the time the meal took place (Lunch or Dinner) and then apply the normalization to each group separately.
# 
# Remember you can always explore the dataset and see how it changes in the IPython Shell, and refer to the slides in the Slides tab.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(restaurant_data)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'restaurant_data.csv': 'https://file.io/aQNaDHlV'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
restaurant_data = pd.read_csv(prefix+'restaurant_data.csv',index_col=0)


# ### code

# In[4]:


# Define the min-max transformation
min_max_tr = lambda x: (x - x.min()) / (x.max() - x.min())

# Group the data according to the time
restaurant_grouped = restaurant_data.groupby('time')

# Apply the transformation
restaurant_min_max_group = restaurant_grouped.transform(min_max_tr)
print(restaurant_min_max_group.head())


# ## Transforming values to probabilities
# In this exercise, we will apply a probability distribution function to a pandas DataFrame with group related parameters by transforming the tip variable to probabilities.
# 
# The transformation will be a exponential transformation. The exponential distribution is defined as
# 
# e−λ∗x∗λ
# where λ (lambda) is the mean of the group that the observation x belongs to.
# 
# You're going to apply the exponential distribution transformation to the size of each table in the dataset, after grouping the data according to the time of the day the meal took place. Remember to use each group's mean for the value of λ.
# 
# In Python, you can use the exponential as np.exp() from the NumPy library and the mean value as .mean().

# ### code

# In[5]:


import numpy as np


# In[7]:


# Define the exponential transformation
exp_tr = lambda x: np.exp(-x.mean()*x) * x.mean()

# Group the data according to the time
restaurant_grouped = restaurant_data.groupby('time')

# Apply the transformation
restaurant_exp_group = restaurant_grouped['tip'].transform(exp_tr)
print(restaurant_exp_group.head())


# ## Validation of normalization
# For this exercise, we will perform a z-score normalization and verify that it was performed correctly.
# 
# A distinct characteristic of normalized values is that they have a mean equal to zero and standard deviation equal to one.
# 
# After you apply the normalization transformation, you can group again on the same variable, and then check the mean and the standard deviation of each group.
# 
# You will apply the normalization transformation to every numeric variable in the poker_grouped dataset, which is the poker_hands dataset grouped by Class.

# ### init

# In[8]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(poker_hands)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'poker_hands.csv': 'https://file.io/2ruAxPwk'}}
"""
prefixToc='1.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
poker_hands = pd.read_csv(prefix+'poker_hands.csv',index_col=0)


# In[11]:


poker_grouped = poker_hands.groupby('Class')


# ### code

# In[12]:


zscore = lambda x: (x - x.mean()) / x.std()

# Apply the transformation
poker_trans = poker_grouped.transform(zscore)
print(poker_trans.head())


# In[14]:


# Re-group the grouped object and print each group's means and standard deviation
poker_regrouped = poker_trans.groupby(poker_hands['Class'])

print(np.round(poker_regrouped.mean(), 3))
print(poker_regrouped.std())


# # Missing value imputation using transform()
# 

# ## Identifying missing values
# The first step before missing value imputation is to identify if there are missing values in our data, and if so, from which group they arise.
# 
# For the same restaurant_data data you encountered in the lesson, an employee erased by mistake the tips left in 65 tables. The question at stake is how many missing entries came from tables that smokers where present vs tables with no-smokers present.
# 
# Your task is to group both datasets according to the smoker variable, count the number or present values and then calculate the difference.
# 
# We're imputing tips to get you to practice the concepts taught in the lesson. From an ethical standpoint, you should not impute financial data in real life, as it could be considered fraud.

# ### init

# In[16]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(restaurant_data, restaurant_nan)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'restaurant_data.csv': 'https://file.io/JxreuBrl',
  'restaurant_nan.csv': 'https://file.io/dudKDy2a'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
restaurant_data = pd.read_csv(prefix+'restaurant_data.csv',index_col=0)
restaurant_nan = pd.read_csv(prefix+'restaurant_nan.csv',index_col=0)


# ### code

# In[18]:


# Group both objects according to smoke condition
restaurant_nan_grouped = restaurant_nan.groupby('smoker')

# Store the number of present values
restaurant_nan_nval = restaurant_nan_grouped['tip'].count()

# Print the group-wise missing entries
print(restaurant_nan_grouped['total_bill'].count() - restaurant_nan_nval)


# ## Missing value imputation
# As the majority of the real world data contain missing entries, replacing these entries with sensible values can increase the insight you can get from our data.
# 
# In the restaurant dataset, the "total_bill" column has some missing entries, meaning that you have not recorded how much some tables have paid. Your task in this exercise is to replace the missing entries with the median value of the amount paid, according to whether the entry was recorded on lunch or dinner (time variable).

# ### code

# In[20]:


# Define the lambda function
missing_trans = lambda x: x.fillna(x.median())


# In[21]:


# Group the data according to time
restaurant_grouped = restaurant_data.groupby('time')

# Apply the transformation
restaurant_impute = restaurant_grouped.transform(missing_trans)
print(restaurant_impute.head())


# # Data filtration using the filter() function
# 

# ## Data filtration
# As you noticed in the video lesson, you may need to filter your data for various reasons.
# 
# In this exercise, you will use filtering to select a specific part of our DataFrame:
# 
# by the number of entries recorded in each day of the week
# by the mean amount of money the customers paid to the restaurant each day of the week

# ### code

# In[23]:


# Filter the days where the count of total_bill is greater than $40
total_bill_40 = restaurant_data.groupby('day').filter(lambda x: x['total_bill'].count() > 40)

# Print the number of tables where total_bill is greater than $40
print('Number of tables where total_bill is greater than $40:', total_bill_40.shape[0])


# In[24]:


# Select only the entries that have a mean total_bill greater than $20
total_bill_20 = total_bill_40.groupby('day').filter(lambda x : x['total_bill'].mean() > 20)

# Print days of the week that have a mean total_bill greater than $20
print('Days of the week that have a mean total_bill greater than $20:', total_bill_20.day.unique())


# In[ ]:




