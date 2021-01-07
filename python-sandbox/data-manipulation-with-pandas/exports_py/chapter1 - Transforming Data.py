#!/usr/bin/env python
# coding: utf-8

# # Introducing DataFrames

# [Inspecting a DataFrame | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/transforming-data?ex=2)
# 
# > ## Inspecting a DataFrame
# > 
# > When you get a new DataFrame to work with, the first thing you need to do is explore it and see what it contains. There are several useful methods and attributes for this.
# > 
# > -   `.head()` returns the first few rows (the “head” of the DataFrame).
# > -   `.info()` shows information on each of the columns, such as the data type and number of missing values.
# > -   `.shape` returns the number of rows and columns of the DataFrame.
# > -   `.describe()` calculates a few summary statistics for each column.
# > 
# > `homelessness` is a DataFrame containing estimates of homelessness in each U.S. state in 2018. The `individual` column is the number of homeless individuals not part of a family with children. The `family_members` column is the number of homeless individuals part of a family with children. The `state_pop` column is the state's total population.
# > 
# > `pandas` is imported for you.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(homelessness)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'homelessness.csv': 'https://file.io/vTM1t2ehXds4'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
homelessness = pd.read_csv(prefix+'homelessness.csv',index_col=0)


# ### code

# > Print the head of the `homelessness` DataFrame.

# In[2]:


# Print the head of the homelessness data
print(homelessness.head())


# > Print information about the column types and missing values in `homelessness`.

# In[3]:


# Print information about homelessness
print(homelessness.info())


# > Print the number of rows and columns in `homelessness`.

# In[4]:


# Print the shape of homelessness
print(homelessness.shape)


# > Print some summary statistics that describe the `homelessness` DataFrame.

# In[5]:


# Print a description of homelessness
print(homelessness.describe())


# In[ ]:




