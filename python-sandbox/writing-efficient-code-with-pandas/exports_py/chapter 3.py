#!/usr/bin/env python
# coding: utf-8

# # Looping using the .iterrows() function
# 

# ## Create a generator for a pandas DataFrame
# As you've seen in the video, you can easily create a generator out of a pandas DataFrame. Each time you iterate through it, it will yield two elements:
# 
# the index of the respective row
# a pandas Series with all the elements of that row
# You are going to create a generator over the poker dataset, imported as poker_hands. Then, you will print all the elements of the 2nd row, using the generator.
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
{pandas.core.frame.DataFrame: {'poker_hands.csv': 'https://file.io/rXBi5mmL'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
poker_hands = pd.read_csv(prefix+'poker_hands.csv',index_col=0)


# ### code

# In[2]:


# Create a generator over the rows
generator = poker_hands.iterrows()

# Access the elements of the 2nd row
first_element = next(generator)
second_element = next(generator)
print(first_element, second_element)


# ## The iterrows() function for looping
# You just saw how to create a generator out of a pandas DataFrame. You will now use this generator and see how to take advantage of that method of looping through a pandas DataFrame, still using the poker_hands dataset.
# 
# Specifically, we want the sum of the ranks of all the cards, if the index of the hand is an odd number. The ranks of the cards are located in the odd columns of the DataFrame.

# ### code

# In[5]:


data_generator = poker_hands.iterrows()

for index, values in data_generator:
  	# Check if index is odd
    if (index % 2 == 1):
      	# Sum the ranks of all the cards
        hand_sum = sum([values[1], values[3], values[5], values[7], values[9]])


# # Looping using the .apply() function
# 

# ## .apply() function in every cell
# As you saw in the lesson, you can use .apply() to map a function to every cell of the DataFrame, regardless the column or the row.
# 
# You're going to try it out on the poker_hands dataset. You will use .apply() to square every cell of the DataFrame. The native Python way to square a number n is n**2.

# ### code

# In[6]:


# Define the lambda transformation
get_square = lambda x: x**2

# Apply the transformation
data_sum = poker_hands.apply(get_square)
print(data_sum.head())


# ## .apply() for rows iteration
# .apply() is a very useful to iterate through the rows of a DataFrame and apply a specific function.
# 
# You will work on a subset of the poker_hands dataset, which includes only the rank of all the five cards of each hand in each row (this subset is generated for you in the script). You're going to get the variance of every hand for all ranks, and every rank for all hands.

# ### code

# In[7]:


import numpy as np


# In[8]:


# Define the lambda transformation
get_variance = lambda x: np.var(x)

# Apply the transformation
data_tr = poker_hands[['R1', 'R2', 'R3', 'R4', 'R5']].apply(get_variance, axis=1)
print(data_tr.head())


# Modify the script to apply the function on every rank.

# In[9]:


get_variance = lambda x: np.var(x)

# Apply the transformation
data_tr = poker_hands[['R1', 'R2', 'R3', 'R4', 'R5']].apply(get_variance, axis=0)
print(data_tr.head())


# # Vectorization over pandas series
# 

# ## pandas vectorization in action
# In this exercise, you will apply vectorization over pandas series to:
# 
# calculate the mean rank of all the cards in each hand (row)
# calculate the mean rank of each of the 5 cards in each hand (column)
# You will use the poker_hands dataset once again to compare both methods' efficiency.

# ### code

# In[11]:


import time as time


# In[12]:


# Calculate the mean rank in each hand
row_start_time = time.time()
mean_r = poker_hands[['R1', 'R2', 'R3', 'R4', 'R5']].mean(axis=1)
print("Time using pandas vectorization for rows: {} sec".format(time.time() - row_start_time))
print(mean_r.head())

# Calculate the mean rank of each of the 5 card in all hands
col_start_time = time.time()
mean_c = poker_hands[['R1', 'R2', 'R3', 'R4', 'R5']].mean(axis=0)
print("Time using pandas vectorization for columns: {} sec".format(time.time() - col_start_time))
print(mean_c.head())


# # Vectorization with NumPy arrays using .values()
# 

# ## Vectorization methods for looping a DataFrame
# Now that you're familiar with vectorization in pandas and NumPy, you're going to compare their respective performances yourself.
# 
# Your task is to calculate the variance of all the hands in each hand using the vectorization over pandas Series and then modify your code using the vectorization over Numpy ndarrays method.

# ### code

# In[13]:


# Calculate the variance in each hand
start_time = time.time()
poker_var = poker_hands[['R1', 'R2', 'R3', 'R4', 'R5']].var(axis=1)
print("Time using pandas vectorization: {} sec".format(time.time() - start_time))
print(poker_var.head())


# In[14]:


# Calculate the variance in each hand
start_time = time.time()
poker_var = poker_hands[['R1', 'R2', 'R3', 'R4', 'R5']].values.var(axis=1, ddof=1)
print("Time using NumPy vectorization: {} sec".format(time.time() - start_time))
print(poker_var[0:5])


# In[ ]:




