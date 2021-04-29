#!/usr/bin/env python
# coding: utf-8

# # Data type constraints
# 
# ```python
# 
# # String to integers
# # Print sum of all Revenue column
# sales['Revenue'].sum()
# '23153$1457$36865$32474$472$27510$16158$5694$6876$40487$807$6893$9153$6895$4216..
# # Remove $ from Revenue column
# sales['Revenue'] = sales['Revenue'].str.strip('$')
# sales['Revenue'] = sales['Revenue'].astype('int')
# # Verify that Revenue is now an integer
# assert sales['Revenue'].dtype == 'int'
# 
# # Numeric or categorical?
# # Convert to categorical
# df["marriage_status"] = df["marriage_status"].astype('category')
# df.describe()
# marriage_status
# count 241
# unique 4
# top 1
# freq 120
# 
# 
# 
# ```

# # Data range constraints
# 
# ```python
# 
# 
# # drop values
# import pandas as pd
# # Output Movies with rating > 5
# movies[movies['avg_rating'] > 5]
# movie_name
# avg_rating
# 23 A Beautiful Mind 6
# 65 La Vita e Bella 6
# 77 Amelie 6
# # Drop values using filtering
# movies = movies[movies['avg_rating'] <= 5]
# # Drop values using .drop()
# movies.drop(movies[movies['avg_rating'] > 5].index, inplace = True)
# # Assert results
# assert movies['avg_rating'].max() <= 5
# 
# 
# # Change out of range value to upper limit
# # Convert avg_rating > 5 to 5
# movies.loc[movies['avg_rating'] > 5, 'avg_rating'] = 5
# # Assert statement
# assert movies['avg_rating'].max() <= 5
# 
# # Date range example
# today_date = dt.date.today()
# Drop the data
# # Drop values using filtering
# user_signups = user_signups[user_signups['subscription_date'] < today_date]
# # Drop values using .drop()
# user_signups.drop(user_signups[user_signups['subscription_date'] > today_date].index, inplace = True)
# Hardcode dates with upper limit
# # Drop values using filtering
# user_signups.loc[user_signups['subscription_date'] > today_date, 'subscription_date'] = today_date
# # Assert is true
# assert user_signups.subscription_date.max().date() <= today_date
# 
# 
# ```

# ## Tire size constraints
# > 
# > In this lesson, you're going to build on top of the work you've been doing with the `ride_sharing` DataFrame. You'll be working with the `tire_sizes` column which contains data on each bike's tire size.
# > 
# > Bicycle tire sizes could be either 26″, 27″ or 29″ and are here correctly stored as a categorical value. In an effort to cut maintenance costs, the ride sharing provider decided to set the maximum tire size to be 27″.
# > 
# > In this exercise, you will make sure the `tire_sizes` column has the correct range by first converting it to an integer, then setting and testing the new upper limit of 27″ for tire sizes.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(ride_sharing)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'ride_sharing.csv': 'https://file.io/X6hoOa3LYCsS'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
ride_sharing = pd.read_csv(prefix+'ride_sharing.csv',index_col=0)


# In[2]:


ride_sharing.info()


# In[3]:


ride_sharing['tire_sizes']=ride_sharing['tire_sizes'].astype('category')


# In[4]:


ride_sharing['tire_sizes'].describe()


# ### code

# [Tire size constraints | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/common-data-problems-1?ex=6)
# 
# > -   Convert the `tire_sizes` column from `category` to `'int'`.
# > -   Use `.loc[]` to set all values of `tire_sizes` above 27 to 27.
# > -   Reconvert back `tire_sizes` to `'category'` from `int`.
# > -   Print the description of the `tire_sizes`.

# In[7]:


# Convert tire_sizes to integer
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('int')

# Set all values above 27 to 27
ride_sharing.loc[ride_sharing['tire_sizes'] > 27, 'tire_sizes'] = 27

# Reconvert tire_sizes back to categorical
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('category')

# Print tire size description
print(ride_sharing['tire_sizes'].describe())


# ## Back to the future
# > 
# > A new update to the data pipeline feeding into the `ride_sharing` DataFrame has been updated to register each ride's date. This information is stored in the `ride_date` column of the type `object`, which represents strings in `pandas`.
# > 
# > A bug was discovered which was relaying rides taken today as taken next year. To fix this, you will find all instances of the `ride_date` column that occur anytime in the future, and set the maximum possible value of this column to today's date. Before doing so, you would need to convert `ride_date` to a `datetime` object.
# > 
# > The `datetime` package has been imported as `dt`, alongside all the packages you've been using till now

# ### init

# In[8]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(ride_sharing)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'ride_sharing.csv': 'https://file.io/Dkg2rSM37lnn'}}
"""
prefixToc='2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
ride_sharing = pd.read_csv(prefix+'ride_sharing.csv',index_col=0)


# In[9]:


ride_sharing.info()


# In[10]:


ride_sharing['tire_sizes']=ride_sharing['tire_sizes'].astype('category')


# In[11]:


ride_sharing['tire_sizes'].describe()


# ### code

# [Back to the future | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/common-data-problems-1?ex=7)
# 
# > -   Convert `ride_date` to a `datetime` object and store it in `ride_dt` column using `to_datetime()`.
# > -   Create the variable `today`, which stores today's date by using the `dt.date.today()` function.
# > -   For all instances of `ride_dt` in the future, set them to today's date.
# > -   Print the maximum date in the `ride_dt` column.

# In[13]:


import datetime as dt

# Convert ride_date to datetime
ride_sharing['ride_dt'] = pd.to_datetime(ride_sharing['ride_date']).dt.date

# Save today's date
today = dt.date.today()

# Set all in the future to today's date
ride_sharing.loc[ride_sharing['ride_dt'] > today, 'ride_dt'] = today

# Print maximum of ride_dt column
print(ride_sharing['ride_dt'].max())


# # Uniqueness constraints
# 
# ```python
# 
# # How to find duplicate values?
# # Get duplicates across all columns
# duplicates = height_weight.duplicated()
# print(duplicates)
# 
# # How to find duplicate rows?
# # The .duplicated() method
# subset : List of column names to check for duplication.
# keep : Whether to keep first ( 'first' ), last ( 'last' ) or all ( False ) duplicate values.
# # Column names to check for duplication
# column_names = ['first_name','last_name','address']
# duplicates = height_weight.duplicated(subset = column_names, keep = False)
# 
# # How to find duplicate rows?
# # Output duplicate values
# height_weight[duplicates].sort_values(by = 'first_name')
# 
# # How to treat duplicate values?
# # The .drop_duplicates() method
# subset : List of column names to check for duplication.
# keep : Whether to keep first ( 'first' ), last ( 'last' ) or all ( False ) duplicate values.
# inplace : Drop duplicated rows directly inside DataFrame without creating new object ( True).
# # Drop duplicates
# height_weight.drop_duplicates(inplace = True)
# 
# # How to treat duplicate values?
# The .groupby() and .agg() methods
# # Group by column names and produce statistical summaries
# column_names = ['first_name','last_name','address']
# summaries = {'height': 'max', 'weight': 'mean'}
# height_weight = height_weight.groupby(by = column_names).agg(summaries).reset_index()
# # Make sure aggregation is done
# duplicates = height_weight.duplicated(subset = column_names, keep = False)
# height_weight[duplicates].sort_values(by = 'first_name')
# 
# ```

# ## Finding duplicates
# > 
# > A new update to the data pipeline feeding into `ride_sharing` has added the `ride_id` column, which represents a unique identifier for each ride.
# > 
# > The update however coincided with radically shorter average ride duration times and irregular user birth dates set in the future. Most importantly, the number of rides taken has increased by 20% overnight, leading you to think there might be both complete and incomplete duplicates in the `ride_sharing` DataFrame.
# > 
# > In this exercise, you will confirm this suspicion by finding those duplicates. A sample of `ride_sharing` is in your environment, as well as all the packages you've been working with thus far.

# ### init

# In[14]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(ride_sharing)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'ride_sharing.csv': 'https://file.io/fjp81hCjFhFs'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
ride_sharing = pd.read_csv(prefix+'ride_sharing.csv',index_col=0)


ride_sharing.info()

ride_sharing['tire_sizes']=ride_sharing['tire_sizes'].astype('category')

ride_sharing['tire_sizes'].describe()


# ### code
# 

# [Finding duplicates | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/common-data-problems-1?ex=10)
# 
# > -   Find duplicated rows of `ride_id` in the `ride_sharing` DataFrame while setting `keep` to `False`.
# > -   Subset `ride_sharing` on `duplicates` and sort by `ride_id` and assign the results to `duplicated_rides`.
# > -   Print the `ride_id`, `duration` and `user_birth_year` columns of `duplicated_rides` in that order.

# In[15]:


# Find duplicates
duplicates = ride_sharing.duplicated('ride_id', keep=False)

# Sort your duplicated rides
duplicated_rides = ride_sharing[duplicates].sort_values('ride_id')

# Print relevant columns of duplicated_rides
print(duplicated_rides[['ride_id','duration','user_birth_year']])


# ## Treating duplicates
# > 
# > In the last exercise, you were able to verify that the new update feeding into `ride_sharing` contains a bug generating both complete and incomplete duplicated rows for some values of the `ride_id` column, with occasional discrepant values for the `user_birth_year` and `duration` columns.
# > 
# > In this exercise, you will be treating those duplicated rows by first dropping complete duplicates, and then merging the incomplete duplicate rows into one while keeping the average `duration`, and the minimum `user_birth_year` for each set of incomplete duplicate rows.

# [Treating duplicates | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/common-data-problems-1?ex=11)
# 
# > -   Drop complete duplicates in `ride_sharing` and store the results in `ride_dup`.
# > -   Create the `statistics` dictionary which holds **min**imum aggregation for `user_birth_year` and **mean** aggregation for `duration`.
# > -   Drop incomplete duplicates by grouping by `ride_id` and applying the aggregation in `statistics`.
# > -   Find duplicates again and run the `assert` statement to verify de-duplication.

# In[17]:


# Drop complete duplicates from ride_sharing
ride_dup = ride_sharing.drop_duplicates()

# Create statistics dictionary for aggregation function
statistics = {'user_birth_year': 'min', 'duration': 'mean'}

# Group by ride_id and compute new statistics
ride_unique = ride_dup.groupby('ride_id').agg(statistics).reset_index()

# Find duplicated values again
duplicates = ride_unique.duplicated(subset = 'ride_id', keep = False)
duplicated_rides = ride_unique[duplicates == True]

# Assert duplicates are processed
assert duplicated_rides.shape[0] == 0


# In[ ]:




