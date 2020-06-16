#!/usr/bin/env python
# coding: utf-8

# # Replace scalar values using .replace()
# 

# ## Replacing scalar values I
# In this exercise, we will replace a list of values in our dataset by using the .replace() method with another list of desired values.
# 
# We will apply the functions in the poker_hands DataFrame. Remember that in the poker_hands DataFrame, each row of columns R1 to R5 represents the rank of each card from a player's poker hand spanning from 1 (Ace) to 13 (King). The Class feature classifies each hand as a category, and the Explanation feature briefly explains each hand.
# 
# The poker_hands DataFrame is already loaded for you, and you can explore the features Class and Explanation.
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
uploadToFileIO(poker_hands)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'poker_hands.csv': 'https://file.io/F0BFtS9v'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
poker_hands = pd.read_csv(prefix+'poker_hands.csv',index_col=0)


# ### code

# In[2]:


# Replace Class 1 to -2 
poker_hands['Class'].replace(1, -2, inplace=True)
# Replace Class 2 to -3
poker_hands['Class'].replace(2, -3, inplace=True)

print(poker_hands[['Class', 'Explanation']])


# ## Replace scalar values II
# As discussed in the video, in a pandas DataFrame, it is possible to replace values in a very intuitive way: we locate the position (row and column) in the Dataframe and assign in the new value you want to replace with. In a more pandas-ian way, the .replace() function is available that performs the same task.
# 
# You will be using the names DataFrame which includes, among others, the most popular names in the US by year, gender and ethnicity.
# 
# Your task is to replace all the babies that are classified as FEMALE to GIRL using the following methods:
# 
# intuitive scalar replacement
# using the .replace() function

# ### init

# In[3]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(names)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'names.csv': 'https://file.io/8vpXqFV8'}}
"""
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
names = pd.read_csv(prefix+'names.csv',index_col=0)


# In[6]:


import time as time


# ### code

# In[8]:


start_time = time.time()

# Replace all the entries that has 'FEMALE' as a gender with 'GIRL'
names['Gender'].loc[names.Gender == 'FEMALE'] = 'GIRL'

print("Time using .loc[]: {} sec".format(time.time() - start_time))


# In[10]:


start_time = time.time()

# Replace all the entries that has 'FEMALE' as a gender with 'GIRL'
names['Gender'].replace('FEMALE', 'GIRL', inplace=True)

print("Time using .replace(): {} sec".format(time.time() - start_time))


# # Replace values using lists
# 

# ## Replace multiple values I
# In this exercise, you will apply the .replace() function for the task of replacing multiple values with one or more values. You will again use the names dataset which contains, among others, the most popular names in the US by year, gender and Ethnicity.
# 
# Thus you want to replace all ethnicities classified as black or white non-hispanics to non-hispanic. Remember, the ethnicities are stated in the dataset as follows: ['BLACK NON HISP', 'BLACK NON HISPANIC', 'WHITE NON HISP' , 'WHITE NON HISPANIC'] and should be replaced to 'NON HISPANIC'

# ### code

# In[15]:


start_time = time.time()

# Replace all non-Hispanic ethnicities with 'NON HISPANIC'
names['Ethnicity'].loc[(names['Ethnicity'] == 'WHITE NON HISPANIC') | 
                      (names['Ethnicity'] == 'BLACK NON HISPANIC') | 
                      (names['Ethnicity'] == 'WHITE NON HISP') | 
                      (names['Ethnicity'] == 'BLACK NON HISP')] = 'NON HISPANIC'

print("Time using .loc[]: sec".format(time.time() - start_time))


# In[16]:


start_time = time.time()

# Replace all non-Hispanic ethnicities with 'NON HISPANIC'
names['Ethnicity'].replace(['WHITE NON HISPANIC', 'BLACK NON HISPANIC', 'WHITE NON HISP', 'BLACK NON HISP'], 'NON HISPANIC', inplace=True)

print("Time using .replace(): {} sec".format(time.time() - start_time))


# ## Replace multiple values II
# As discussed in the video, instead of using the .replace() function multiple times to replace multiple values, you can use lists to map the elements you want to replace one to one with those you want to replace them with.
# 
# As you have seen in our popular names dataset, there are two names for the same ethnicity. We want to standardize the naming of each ethnicity by replacing
# 
# - 'ASIAN AND PACI' to 'ASIAN AND PACIFIC ISLANDER'
# - 'BLACK NON HISP' to 'BLACK NON HISPANIC'
# - 'WHITE NON HISP' to 'WHITE NON HISPANIC'
# 
# In the DataFrame names, you are going to replace all the values on the left by the values on the right.

# ### code

# In[17]:


start_time = time.time()

# Replace ethnicities as instructed
names['Ethnicity'].replace(['ASIAN AND PACI','BLACK NON HISP', 'WHITE NON HISP'], ['ASIAN AND PACIFIC ISLANDER', 'BLACK NON HISPANIC', 'WHITE NON HISPANIC'], inplace=True)

print("Time using .replace(): {} sec".format(time.time() - start_time))


# # Replace values using dictionaries
# 

# ## Replace single values I
# In this exercise, we will apply the following replacing technique of replacing multiple values using dictionaries on a different dataset.
# 
# We will apply the functions in the data DataFrame. Each row represents the rank of 5 cards from a playing card deck, spanning from 1 (Ace) to 13 (King) (features R1, R2, R3, R4, R5). The feature 'Class' classifies each row to a category (from 0 to 9) and the feature 'Explanation' gives a brief explanation of what each class represents.
# 
# The purpose of this exercise is to categorize the two types of flush in the game ('Royal flush' and 'Straight flush') under the 'Flush' name.

# ### init

# In[18]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(poker_hands)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'poker_hands.csv': 'https://file.io/MwTKzkiE'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
poker_hands = pd.read_csv(prefix+'poker_hands.csv',index_col=0)


# ### code

# In[19]:


# Replace Royal flush or Straight flush to Flush
poker_hands.replace({'Royal flush':'Flush', 'Straight flush':'Flush'}, inplace=True)
print(poker_hands['Explanation'].head())


# ## Replace single values II
# For this exercise, we will be using the names DataFrame. In this dataset, the column 'Rank' shows the ranking of each name by year. For this exercise, you will use dictionaries to replace the first ranked name of every year as 'FIRST', the second name as 'SECOND' and the third name as 'THIRD'.
# 
# You will use dictionaries to replace one single value per key.
# 
# You can already see the first 5 names of the data, which correspond to the 5 most popular names for all the females belonging to the 'ASIAN AND PACIFIC ISLANDER' ethnicity in 2011.

# ### code

# In[21]:


# Replace the number rank by a string
names['Rank'].replace({1:'FIRST', 2:'SECOND', 3:'THIRD'}, inplace=True)
print(names.head())


# ## Replace multiple values III
# As you saw in the video, you can use dictionaries to replace multiple values with just one value, even from multiple columns. To show the usefulness of replacing with dictionaries, you will use the names dataset one more time.
# 
# In this dataset, the column 'Rank' shows which rank each name reached every year. You will change the rank of the first three ranked names of every year to 'MEDAL' and those from 4th and 5th place to 'ALMOST MEDAL'.
# 
# You can already see the first 5 names of the data, which correspond to the 5 most popular names for all the females belonging to the 'ASIAN AND PACIFIC ISLANDER' ethnicity in 2011.

# ### code

# In[23]:


# Replace the rank of the first three ranked names to 'MEDAL'
names.replace({'Rank': {'FIRST':'MEDAL', 'SECOND':'MEDAL', 'THIRD':'MEDAL'}}, inplace=True)

# Replace the rank of the 4th and 5th ranked names to 'ALMOST MEDAL'
names.replace({'Rank': {4:'ALMOST MEDAL', 5:'ALMOST MEDAL'}}, inplace=True)
print(names.head())


# In[ ]:




