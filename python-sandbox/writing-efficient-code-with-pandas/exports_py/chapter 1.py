#!/usr/bin/env python
# coding: utf-8

# # The need for efficient coding I
# 

# ## Measuring time I
# In the lecture slides, you saw how the time.time() function can be loaded and used to assess the time required to perform a basic mathematical operation.
# 
# Now, you will use the same strategy to assess two different methods for solving a similar problem: calculate the sum of squares of all the positive integers from 1 to 1 million (1,000,000).
# 
# Similar to what you saw in the video, you will compare two methods; one that uses brute force and one more mathematically sophisticated.
# 
# In the function formula, we use the standard formula
# 
# N∗(N+1)(2N+1)6
# where N=1,000,000.
# 
# In the function brute_force we loop over each number from 1 to 1 million and add it to the result.

# ### init

# In[4]:


###################
##### inspect Function
###################

""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(formula)
"""

def formula(N):
    return N*(N+1)*(2*N+1)/6


# In[7]:


###################
##### inspect Function
###################

""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(brute_force)
"""
def brute_force(N):
    res = 0
    UL = N+1
    for i in range(1,UL):
        res+=i^2
    return res


# ### code

# In[8]:


import time as time


# In[9]:


# Calculate the result of the problem using formula() and print the time required
N = 1000000
fm_start_time = time.time()
first_method = formula(N)
print("Time using formula: {} sec".format(time.time() - fm_start_time))

# Calculate the result of the problem using brute_force() and print the time required
sm_start_time = time.time()
second_method = brute_force(N)
print("Time using the brute force: {} sec".format(time.time() - sm_start_time))


# ## Measuring time II
# As we discussed in the lectures, in the majority of cases, a list comprehension is faster than a for loop.
# 
# In this demonstration, you will see a case where a list comprehension and a for loop have so small difference in efficiency that choosing either method will perform this simple task instantly.
# 
# In the list words, there are random words downloaded from the Internet. We are interested to create another list called listlet in which we only keep the words that start with the letter b.
# 
# In case you are not familiar with dealing with strings in Python, each string has the .startswith() attribute, which returns a True/False statement whether the string starts with a specific letter/phrase or not.

# ### init

# In[11]:


###################
##### liste de mots (list)
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(words)
"""

tobedownloaded="""
{list: {'words.txt': 'https://file.io/DPFwjPW9'}}
"""
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
from downloadfromFileIO import loadListFromTxt
words = loadListFromTxt(prefix+'words.txt')


# ### code

# In[12]:


# Store the time before the execution
start_time = time.time()

# Execute the operation
letlist = [wrd for wrd in words if wrd.startswith('b')]

# Store and print the difference between the start and the current time
total_time_lc = time.time() - start_time
print('Time using list comprehension: {} sec'.format(total_time_lc))


# In[13]:


# Store the time before the execution
start_time = time.time()

# Execute the operation
letlist = []
for wrd in words:
    if wrd.startswith('b'):
        letlist.append(wrd)
        
# Print the difference between the start and the current time
total_time_fl = time.time() - start_time
print('Time using for loop: {} sec'.format(total_time_fl))


# # Locate rows: .iloc[] and .loc[]
# 

# ## Row selection: loc[] vs iloc[]
# A big part of working with DataFrames is to locate specific entries in the dataset. You can locate rows in two ways:
# 
# By a specific value of a column (feature).
# By the index of the rows (index). In this exercise, we will focus on the second way.
# If you have previous experience with pandas, you should be familiar with the .loc and .iloc indexers, which stands for 'location' and 'index location' respectively. In most cases, the indices will be the same as the position of each row in the Dataframe (e.g. the row with index 13 will be the 14th entry).
# 
# While we can use both functions to perform the same task, we are interested in which is the most efficient in terms of speed.

# ### init

# In[14]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(poker_hands)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'poker_hands.csv': 'https://file.io/ESnGOlTD'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
poker_hands = pd.read_csv(prefix+'poker_hands.csv',index_col=0)


# ### code

# In[15]:


# Define the range of rows to select: row_nums
row_nums = range(0, 1000)

# Select the rows using .loc[] and row_nums and record the time before and after
loc_start_time = time.time()
rows = poker_hands.loc[row_nums]
loc_end_time = time.time()

# Print the time it took to select the rows using .loc[]
print("Time using .loc[]: {} sec".format(loc_end_time - loc_start_time))


# In[16]:


# Select the rows using .iloc[] and row_nums and record the time before and after
iloc_start_time = time.time()
rows = poker_hands.iloc[row_nums]
iloc_end_time = time.time()

# Print the time it took to select the rows using .iloc
print("Time using .iloc[]: {} sec".format(iloc_end_time-iloc_start_time))


# ## Column selection: .iloc[] vs by name
# In the previous exercise, you saw how the .loc[] and .iloc[] functions can be used to locate specific rows of a DataFrame (based on the index). Turns out, the .iloc[] function performs a lot faster (~ 2 times) for this task!
# 
# Another important task is to find the faster function to select the targeted features (columns) of a DataFrame. In this exercise, we will compare the following:
# 
# using the index locator .iloc()
# using the names of the columns While we can use both functions to perform the same task, we are interested in which is the most efficient in terms of speed.
# In this exercise, you will continue working with the poker data which is stored in poker_hands. Take a second to examine the structure of this DataFrame by calling poker_hands.head() in the console!

# In[17]:


# Use .iloc to select the first 6 columns and record the times before and after
iloc_start_time = time.time()
cols = poker_hands.iloc[:,:6]
iloc_end_time = time.time()

# Print the time it took
print("Time using .iloc[] : {} sec".format(iloc_end_time - iloc_start_time))


# In[18]:


# Use simple column selection to select the first 6 columns 
names_start_time = time.time()
cols = poker_hands[['S1', 'R1', 'S2', 'R2', 'S3', 'R3']]
names_end_time = time.time()

# Print the time it took
print("Time using selection by name : {} sec".format(names_end_time - names_start_time))


# # Select random rows
# 

# ## Random row selection
# In this exercise, you will compare the two methods described for selecting random rows (entries) with replacement in a pandas DataFrame:
# 
# The built-in pandas function .random()
# The NumPy random integer number generator np.random.randint()
# Generally, in the fields of statistics and machine learning, when we need to train an algorithm, we train the algorithm on the 75% of the available data and then test the performance on the remaining 25% of the data.
# 
# For this exercise, we will randomly sample the 75% percent of all the played poker hands available, using each of the above methods, and check which method is more efficient in terms of speed.

# ### code

# In[19]:


# Extract number of rows in dataset
N=poker_hands.shape[0]

# Select and time the selection of the 75% of the dataset's rows
rand_start_time = time.time()
poker_hands.iloc[np.random.randint(low=0, high=N, size=int(0.75 * N))]
print("Time using Numpy: {} sec".format(time.time() - rand_start_time))


# In[20]:


# Select and time the selection of the 75% of the dataset's rows using sample()
samp_start_time = time.time()
poker_hands.sample(int(0.75 * N), axis=0, replace = True)
print("Time using .sample: {} sec".format(time.time() - samp_start_time))


# ## Random column selection
# In the previous exercise, we examined two ways to select random rows from a pandas DataFrame. We can use the same functions to randomly select columns in a pandas DataFrame.
# 
# To randomly select 4 columns out of the poker dataset, you will use the following two functions:
# 
# The built-in pandas function .random()
# The NumPy random integer number generator np.random.randint()

# ### code

# In[21]:


# Extract number of columns in dataset
D=poker_hands.shape[1]

# Select and time the selection of 4 of the dataset's columns using NumPy
np_start_time = time.time()
poker_hands.iloc[:,np.random.randint(low=0, high=D, size=4)]
print("Time using NymPy's random.randint(): {} sec".format(time.time() - np_start_time))


# In[22]:


# Select and time the selection of 4 of the dataset's columns using pandas
pd_start_time = time.time()
poker_hands.sample(4, axis=1)
print("Time using panda's .sample(): {} sec".format(time.time() - pd_start_time))


# In[ ]:




