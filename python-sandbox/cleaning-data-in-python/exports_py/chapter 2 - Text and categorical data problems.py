#!/usr/bin/env python
# coding: utf-8

# # Membership constraints
# 
# ```python
# 
# # Finding inconsistent categories
# inconsistent_categories = set(study_data['blood_type']).difference(categories['blood_type'])
# print(inconsistent_categories)
# {'Z+'}
# # Get and print rows with inconsistent categories
# inconsistent_rows = study_data['blood_type'].isin(inconsistent_categories)
# study_data[inconsistent_rows]
# name
# birthday blood_type
# 5 Jennifer 2019-12-17
# Z+
# 
# # Dropping inconsistent categories
# inconsistent_categories = set(study_data['blood_type']).difference(categories['blood_type'])
# inconsistent_rows = study_data['blood_type'].isin(inconsistent_categories)
# inconsistent_data = study_data[inconsistent_rows]
# # Drop inconsistent categories and get consistent data only
# consistent_data = study_data[~inconsistent_rows]
# ```

# ## Finding consistency
# > 
# > In this exercise and throughout this chapter, you'll be working with the `airlines` DataFrame which contains survey responses on the San Francisco Airport from airline customers.
# > 
# > The DataFrame contains flight metadata such as the airline, the destination, waiting times as well as answers to key questions regarding cleanliness, safety, and satisfaction. Another DataFrame named `categories` was created, containing all correct possible values for the survey columns.
# > 
# > In this exercise, you will use both of these DataFrames to find survey answers with inconsistent values, and drop them, effectively performing an outer and inner join on both these DataFrames as seen in the video exercise. The `pandas` package has been imported as `pd`, and the `airlines` and `categories` DataFrames are in your environment.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(airlines, categories)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'airlines.csv': 'https://file.io/xPfG689sAVWn',
  'categories.csv': 'https://file.io/TVCLlH7rxkpR'}}
  """
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
airlines = pd.read_csv(prefix+'airlines.csv',index_col=0)
categories = pd.read_csv(prefix+'categories.csv',index_col=0)


# ### code

# [Finding consistency | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/text-and-categorical-data-problems?ex=3)
# 
# > -   Print the `categories` DataFrame and take a close look at all possible correct categories of the survey columns.
# > -   Print the unique values of the survey columns in `airlines` using the `.unique()` method.

# In[5]:


# Print categories DataFrame
print(categories)

# Print unique values of survey columns in airlines
print('Cleanliness: ', airlines['cleanliness'].unique(), "\n")
print('Safety: ', airlines['safety'].unique(), "\n")
print('Satisfaction: ', airlines['satisfaction'].unique(), "\n")


# [Finding consistency | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/text-and-categorical-data-problems?ex=3)
# 
# > -   Create a set out of the `cleanliness` column in `airlines` using `set()` and find the inconsistent category by finding the **difference** in the `cleanliness` column of `categories`.
# > -   Find rows of `airlines` with a `cleanliness` value not in `categories` and print the output.

# In[7]:


# Find the cleanliness category in airlines not in categories
cat_clean = set(airlines['cleanliness']).difference(categories['cleanliness'])

# Find rows with that category
cat_clean_rows = airlines['cleanliness'].isin(cat_clean)

# Print rows with inconsistent category
print(airlines[cat_clean_rows])


# [Finding consistency | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/text-and-categorical-data-problems?ex=3)
# 
# > Print the rows with the consistent categories of `cleanliness` only.

# In[8]:


# Print rows with consistent categories only
print(airlines[~cat_clean_rows])


# # Categorical variables
# 
# ```python
# 
# # Capitalization: 'married' , 'Married' , 'UNMARRIED' , 'unmarried' ..
# marriage_status['marriage_status'] = marriage_status['marriage_status'].str.upper()
# 
# # Trailing spaces: 'married ' , 'married' , 'unmarried' , ' unmarried' ..
# demographics['marriage_status'] = demographics['marriage_status'].str.strip()
# 
# # Collapsing data into categories
# # Using qcut()
# import pandas as pd
# group_names = ['0-200K', '200K-500K', '500K+']
# demographics['income_group'] = pd.qcut(demographics['household_income'], q = 3,
# labels = group_names)
# # Print income_group column
# demographics[['income_group', 'household_income']]
# 
# # Using cut() - create category ranges and names
# ranges = [0,200000,500000,np.inf]
# group_names = ['0-200K', '200K-500K', '500K+']
# # Create income group column
# demographics['income_group'] = pd.cut(demographics['household_income'], bins=ranges,
# labels=group_names)
# demographics[['income_group', 'household_income']]
# 
# # Collapsing data into categories
# # Map categories to fewer ones: reducing categories in categorical column.
# # Create mapping dictionary and replace
# mapping = {'Microsoft':'DesktopOS', 'MacOS':'DesktopOS', 'Linux':'DesktopOS',
# 'IOS':'MobileOS', 'Android':'MobileOS'}
# devices['operating_system'] = devices['operating_system'].replace(mapping)
# devices['operating_system'].unique()
# ```

# ## Inconsistent categories
# > 
# > In this exercise, you'll be revisiting the `airlines` DataFrame from the previous lesson.
# > 
# > As a reminder, the DataFrame contains flight metadata such as the airline, the destination, waiting times as well as answers to key questions regarding cleanliness, safety, and satisfaction on the San Francisco Airport.
# > 
# > In this exercise, you will examine two categorical columns from this DataFrame, `dest_region` and `dest_size` respectively, assess how to address them and make sure that they are cleaned and ready for analysis. The `pandas` package has been imported as `pd`, and the `airlines` DataFrame is in your environment.

# In[9]:


# Print unique values of both columns
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())


# In[10]:


# Lower dest_region column and then replace "eur" with "europe"
airlines['dest_region'] = airlines['dest_region'].str.lower()
airlines['dest_region'] = airlines['dest_region'].replace({'eur':'europe'})


# In[11]:


# Remove white spaces from `dest_size`
airlines['dest_size'] = airlines['dest_size'].str.strip()

# Verify changes have been effected
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())


# ## Remapping categories
# > 
# > To better understand survey respondents from `airlines`, you want to find out if there is a relationship between certain responses and the day of the week and wait time at the gate.
# > 
# > The `airlines` DataFrame contains the `day` and `wait_min` columns, which are categorical and numerical respectively. The `day` column contains the exact day a flight took place, and `wait_min` contains the amount of minutes it took travelers to wait at the gate. To make your analysis easier, you want to create two new categorical variables:
# > 
# > -   `wait_type`: `'short'` for 0-60 min, `'medium'` for 60-180 and `long` for 180+
# > -   `day_week`: `'weekday'` if day is in the weekday, `'weekend'` if day is in the weekend.
# > 
# > The `pandas` and `numpy` packages have been imported as `pd` and `np`. Let's create some new categorical data!

# [Remapping categories | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/text-and-categorical-data-problems?ex=7)
# 
# > -   Create the ranges and labels for the `wait_type` column mentioned in the description above.
# > -   Create the `wait_type` column by from `wait_min` by using `pd.cut()`, while inputting `label_ranges` and `label_names` in the correct arguments.
# > -   Create the `mapping` dictionary mapping weekdays to `'weekday'` and weekend days to `'weekend'`.
# > -   Create the `day_week` column by using `.replace()`.

# In[14]:


import numpy as np


# In[17]:


# Create ranges for categories
label_ranges = [0, 60, 180, np.inf]
label_names = ['short', 'mrdium', 'long']

# Create wait_type column
airlines['wait_type'] = pd.cut(airlines['wait_min'], bins = label_ranges, 
                                labels = label_names)

# Create mappings and replace
mappings = {'Monday':'weekday', 'Tuesday':'weekday', 'Wednesday': 'weekday', 
            'Thursday': 'weekday', 'Friday': 'weekday', 
            'Saturday': 'weekend', 'Sunday': 'weekend'}

airlines['day_week'] = airlines['day'].replace(mappings)


# # Cleaning text data
# 
# ```python
# 
# # Fixing the phone number column
# # Replace "+" with "00"
# phones["Phone number"] = phones["Phone number"].str.replace("+", "00")
# # Replace "-" with nothing
# phones["Phone number"] = phones["Phone number"].str.replace("-", "")
# # Replace phone numbers with lower than 10 digits to NaN
# digits = phones['Phone number'].str.len()
# phones.loc[digits < 10, "Phone number"] = np.nan
# 
# # Fixing the phone number column
# # Find length of each row in Phone number column
# sanity_check = phone['Phone number'].str.len()
# # Assert minmum phone number length is 10
# assert sanity_check.min() >= 10
# # Assert all numbers do not have "+" or "-"
# assert phone['Phone number'].str.contains("+|-").any() == False
# 
# # Regular expressions in action
# # Replace letters with nothing
# phones['Phone number'] = phones['Phone number'].str.replace(r'\D+', '')
# phones.head()
# 
# ```

# ## Removing titles and taking names
# > 
# > While collecting survey respondent metadata in the `airlines` DataFrame, the full name of respondents was saved in the `full_name` column. However upon closer inspection, you found that a lot of the different names are prefixed by honorifics such as `"Dr."`, `"Mr."`, `"Ms."` and `"Miss"`.
# > 
# > Your ultimate objective is to create two new columns named `first_name` and `last_name`, containing the first and last names of respondents respectively. Before doing so however, you need to remove honorifics.
# > 
# > The `airlines` DataFrame is in your environment, alongside `pandas` as `pd.`

# [Removing titles and taking names | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/text-and-categorical-data-problems?ex=9)
# 
# > -   Remove `"Dr."`, `"Mr."`, `"Miss"` and `"Ms."` from `full_name` by replacing them with an empty string `""` in that order.
# > -   Run the `assert` statement using `.str.contains()` that tests whether `full_name` still contains any of the honorifics.

# ### init

# In[19]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(airlines)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'airlines.csv': 'https://file.io/qNLxyI9GEyFA'}}
  """
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
airlines = pd.read_csv(prefix+'airlines.csv',index_col=0)


# ### code

# In[22]:


# Replace "Dr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Dr.","")

# Replace "Mr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Mr.","")

# Replace "Miss" with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Miss","")


# Replace "Ms." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Ms.","")


# Assert that full_name has no honorifics
assert airlines['full_name'].str.contains('Ms.|Mr.|Miss|Dr.').any() == False


# ## Keeping it descriptive
# > 
# > To further understand travelers' experiences in the San Francisco Airport, the quality assurance department sent out a qualitative questionnaire to all travelers who gave the airport the worst score on all possible categories. The objective behind this questionnaire is to identify common patterns in what travelers are saying about the airport.
# > 
# > Their response is stored in the `survey_response` column. Upon a closer look, you realized a few of the answers gave the shortest possible character amount without much substance. In this exercise, you will isolate the responses with a character count higher than **_40_** , and make sure your new DataFrame contains responses with **_40_** characters or more using an `assert` statement.
# > 
# > The `airlines` DataFrame is in your environment, and `pandas` is imported as `pd`.

# [Keeping it descriptive | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/text-and-categorical-data-problems?ex=10)
# 
# > -   Using the `airlines` DataFrame, store the length of each instance in the `survey_response` column in `resp_length` by using `.str.len()`.
# > -   Isolate the rows of `airlines` with `resp_length` higher than `40`.
# > -   Assert that the smallest survey response length in `airlines_survey` is now bigger than 40.

# ### init

# In[23]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(airlines)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'airlines.csv': 'https://file.io/wZ9Rdh26pikG'}}
"""
prefixToc='3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
airlines = pd.read_csv(prefix+'airlines.csv',index_col=0)


# In[26]:


# Store length of each row in survey_response column
resp_length = airlines['survey_response'].str.len()

# Find rows in airlines where resp_length > 40
airlines_survey = airlines[resp_length > 40]

# Assert minimum survey_response length is > 40
assert airlines_survey['survey_response'].str.len().min() > 40

# Print new survey_response column
print(airlines_survey['survey_response'])


# In[ ]:




