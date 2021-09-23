#!/usr/bin/env python
# coding: utf-8

# # Reading date and time data in Pandas
# 
# ```python
# 
# # Loading datetimes with parse_dates
# import pandas as pd
# # Import W20529's rides in Q4 2017
# rides = pd.read_csv('capital-onebike.csv',
# parse_dates = ['Start date', 'End date'])
# # Or:
# rides['Start date'] = pd.to_datetime(rides['Start date'],
# format = "%Y-%m-%d %H:%M:%S")
# 
# # Timezone-aware arithmetic
# # Create a duration column
# rides['Duration'] = rides['End date'] - rides['Start date']
# # Print the first 5 rows
# print(rides['Duration'].head(5))
# 
# # Loading datetimes with parse_dates
# rides['Duration'].dt.total_seconds().head(5)
# 
# ```

# ## Making timedelta columns
# > 
# > Earlier in this course, you wrote a loop to subtract `datetime` objects and determine how long our sample bike had been out of the docks. Now you'll do the same thing with Pandas.
# > 
# > `rides` has already been loaded for you.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(rides)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'rides.csv': 'https://file.io/5rbtRnQ4fCbW'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
rides = pd.read_csv(prefix+'rides.csv',index_col=0)


# In[2]:


rides.info()


# In[5]:


rides['Start date']=rides['Start date'].astype('datetime64')
rides['End date']=rides['End date'].astype('datetime64')


# ### code

# In[10]:


# Subtract the start date from the end date
ride_durations = rides['End date'] - rides['Start date']

# Convert the results to seconds
rides['Duration'] = ride_durations.dt.total_seconds()

print(rides['Duration'].head())


# # Summarizing datetime data in Pandas
# 
# ```python
# 
# # Witrh pandas > 0.23
# 
# # Summarizing data in Pandas
# # Average time out of the dock
# rides['Duration'].mean()
# Timedelta('0 days 00:19:38.931034')
# # Total time out of the dock
# rides['Duration'].sum()
# Timedelta('3 days 22:58:10')
# 
# # Summarizing data in Pandas
# # Percent of time out of the dock
# rides['Duration'].sum() / timedelta(days=91)
# 0.04348417785917786
# 
# # Summarizing datetime in Pandas
# # Average duration by month
# rides.resample('M', on = 'Start date')['Duration seconds'].mean()
# Start date
# 2017-10-31 1886.453704
# 2017-11-30 854.174757
# 2017-12-31 635.101266
# Freq: M, Name: Duration seconds, dtype: float64
#             
#             
# # Summarizing datetime in Pandas
# rides\
# .resample('M', on = 'Start date')\
# ['Duration seconds']\
# .mean()\
# .plot() 
# 
# # Summarizing datetime in Pandas
# rides\
# .resample('D', on = 'Start date')\
# ['Duration seconds']\
# .mean()\
# .plot()
# ```

# ## How many joyrides?
# > 
# > Suppose you have a theory that some people take long bike rides before putting their bike back in the same dock. Let's call these rides "joyrides".
# > 
# > You only have data on one bike, so while you can't draw any bigger conclusions, it's certainly worth a look.
# > 
# > Are there many joyrides? How long were they in our data set? Use the median instead of the mean, because we know there are some very long trips in our data set that might skew the answer, and the median is less sensitive to outliers.

# In[11]:


# Create joyrides
joyrides = (rides['Start station'] == rides['End station'])

# Total number of joyrides
print("{} rides were joyrides".format(joyrides.sum()))

# Median of all rides
print("The median duration overall was {:.2f} seconds"      .format(rides['Duration'].median()))

# Median of joyrides
print("The median duration for joyrides was {:.2f} seconds"      .format(rides[joyrides]['Duration'].median()))


# ## It's getting cold outside, W20529
# > 
# > Washington, D.C. has mild weather overall, but the average high temperature in October (68ºF / 20ºC) is certainly higher than the average high temperature in December (47ºF / 8ºC). People also travel more in December, and they work fewer days so they commute less.
# > 
# > How might the weather or the season have affected the length of bike trips?

# In[12]:


# Import matplotlib
import matplotlib.pyplot as plt

# Resample rides to daily, take the size, plot the results
rides.resample('D', on = 'Start date')  .size()  .plot(ylim = [0, 15])

# Show the results
plt.show()


# In[13]:


# Import matplotlib
import matplotlib.pyplot as plt

# Resample rides to monthly, take the size, plot the results
rides.resample('M', on = 'Start date')  .size()  .plot(ylim = [0, 150])

# Show the results
plt.show()


# ## Members vs casual riders over time
# > 
# > Riders can either be "Members", meaning they pay yearly for the ability to take a bike at any time, or "Casual", meaning they pay at the kiosk attached to the bike dock.
# > 
# > Do members and casual riders drop off at the same rate over October to December, or does one drop off faster than the other?
# > 
# > As before, `rides` has been loaded for you. You're going to use the Pandas method `.value_counts()`, which returns the number of instances of each value in a Series. In this case, the counts of "Member" or "Casual".

# In[14]:


# Resample rides to be monthly on the basis of Start date
monthly_rides = rides.resample('M', on = 'Start date')['Member type']

# Take the ratio of the .value_counts() over the total number of rides
print(monthly_rides.value_counts() / monthly_rides.size())


# ## Combining groupby() and resample()
# > 
# > A very powerful method in Pandas is `.groupby()`. Whereas `.resample()` groups rows by some time or date information, `.groupby()` groups rows based on the values in one or more columns. For example, `rides.groupby('Member type').size()` would tell us how many rides there were by member type in our entire DataFrame.
# > 
# > `.resample()` can be called after `.groupby()`. For example, how long was the median ride by month, and by Membership type?

# In[15]:


# Group rides by member type, and resample to the month
grouped = rides.groupby('Member type')  .resample('M', on = 'Start date')

# Print the median duration for each group
print(grouped['Duration'].median())


# # Additional datetime methods in Pandas
# 
# ```python
# 
# # Timezones in Pandas
# rides['Start date'].head(3)\
# .dt.tz_localize('America/New_York')
# 0 2017-10-01 15:23:25-04:00
# 1 2017-10-01 15:42:57-04:00
# 2 2017-10-02 06:37:10-04:00
# Name: Start date, dtype: datetime64[ns, America/New_York]
# 
# # Timezones in Pandas
# # Try to set a timezone...
# rides['Start date'] = rides['Start date']\
# .dt.tz_localize('America/New_York')
# AmbiguousTimeError: Cannot infer dst time from '2017-11-05 01:56:50',
# try using the 'ambiguous' argument
# # Handle ambiguous datetimes
# rides['Start date'] = rides['Start date']\
# .dt.tz_localize('America/New_York', ambiguous='NaT')
# rides['End date'] = rides['End date']\
# .dt.tz_localize('America/New_York', ambiguous='NaT')        
# 
# 
# # Shift the indexes
# # Shift the indexes forward one, padding with NaT
# rides['End date'].shift(1).head(3)
# 
# 
# ```

# ## Timezones in Pandas
# > 
# > Earlier in this course, you assigned a timezone to each `datetime` in a list. Now with Pandas you can do that with a single method call.
# > 
# > (Note that, just as before, your data set actually includes some ambiguous datetimes on account of daylight saving; for now, we'll tell Pandas to not even try on those ones. Figuring them out would require more work.)

# In[17]:


# Localize the Start date column to America/New_York
rides['Start date'] = rides['Start date'].dt.tz_localize('America/New_York', ambiguous='NaT')

# Print first value
print(rides['Start date'].iloc[0])


# In[18]:


# Convert the Start date column to Europe/London
rides['Start date'] = rides['Start date'].dt.tz_convert('Europe/London')

# Print the new value
print(rides['Start date'].iloc[0])


# ## How long per weekday?
# > 
# > Pandas has a number of datetime-related attributes within the `.dt` accessor. Many of them are ones you've encountered before, like `.dt.month`. Others are convenient and save time compared to standard Python, like `.dt.weekday_name`.

# In[21]:


# Add a column for the weekday of the start of the ride
rides['Ride start weekday'] = rides['Start date'].dt.day_name()

# Print the median trip time per weekday
print(rides.groupby('Ride start weekday')['Duration'].median())


# ## How long between rides?
# > 
# > For your final exercise, let's take advantage of Pandas indexing to do something interesting. How much time elapsed between rides?

# ### init

# In[25]:


rides['Start date'] = rides['Start date'].dt.tz_convert('America/New_York')
rides['End date'] = rides['End date'].dt.tz_localize('America/New_York', ambiguous='NaT')


# ### code

# In[26]:


# Shift the index of the end date up one; now subract it from the start date
rides['Time since'] = rides['Start date'] - (rides['End date'].shift(1))

# Move from a timedelta to a number of seconds, which is easier to work with
rides['Time since'] = rides['Time since'].dt.total_seconds()

# Resample to the month
monthly = rides.resample('M', on='Start date')

# Print the average hours between rides each month
print(monthly['Time since'].mean()/(60*60))


# In[ ]:




