#!/usr/bin/env python
# coding: utf-8

# # Comparing strings
# 
# ```python
# # Simple string comparison
# # Lets us compare between two strings
# from fuzzywuzzy import fuzz
# # Compare reeding vs reading
# fuzz.WRatio('Reeding', 'Reading') #86
# 
# 
# # Comparison with arrays
# # Import process
# from fuzzywuzzy import process
# # Define string and array of possible matches
# string = "Houston Rockets vs Los Angeles Lakers"
# choices = pd.Series(['Rockets vs Lakers', 'Lakers vs Rockets',
# 'Houson vs Los Angeles', 'Heat vs Bulls'])
# process.extract(string, choices, limit = 2)
# [('Rockets vs Lakers', 86, 0), ('Lakers vs Rockets', 86, 1)]
# 
# # Collapsing all of the state
# # For each correct category
# for state in categories['state']:
#     # Find potential matches in states with typoes
#     matches = process.extract(state, survey['state'], limit = survey.shape[0])
#     # For each potential match match
#     for potential_match in matches:
#         # If high similarity score
#         if potential_match[1] >= 80:
#             # Replace typo with correct category
#             survey.loc[survey['state'] == potential_match[0], 'state'] = state
# 
#             
# ```

# ## The cutoff point
# > 
# > In this exercise, and throughout this chapter, you'll be working with the `restaurants` DataFrame which has data on various restaurants. Your ultimate goal is to create a restaurant recommendation engine, but you need to first clean your data.
# > 
# > This version of `restaurants` has been collected from many sources, where the `cuisine_type` column is riddled with typos, and should contain only `italian`, `american` and `asian` cuisine types. There are so many unique categories that remapping them manually isn't scalable, and it's best to use string similarity instead.
# > 
# > Before doing so, you want to establish the cutoff point for the similarity score using the `fuzzywuzzy`'s `process.extract()` function by finding the similarity score of the most _distant_ typo of each category.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(restaurants)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'restaurants.csv': 'https://file.io/0DYxl1uuHNU5'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
restaurants = pd.read_csv(prefix+'restaurants.csv',index_col=0)


# ### code

# [The cutoff point | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/record-linkage-4?ex=3)
# 
# > -   Import `process` from `fuzzywuzzy`.
# > -   Store the unique `cuisine_type`s into `unique_types`.
# > -   Calculate the similarity of `'asian'`, `'american'`, and `'italian'` to all possible `cuisine_type`s using `process.extract()`, while returning all possible matches.

# In[7]:


# Import process from fuzzywuzzy
from fuzzywuzzy import process

# Store the unique values of cuisine_type in unique_types
unique_types = restaurants['cuisine_type'].unique()

# Calculate similarity of 'asian' to all values of unique_types
print(process.extract('asian', unique_types, limit = len(unique_types)))

# Calculate similarity of 'american' to all values of unique_types
print(process.extract('american', unique_types, limit = len(unique_types)))

# Calculate similarity of 'italian' to all values of unique_types
print(process.extract('italian', unique_types, limit = len(unique_types)))


# ## Remapping categories II
# > 
# > In the last exercise, you determined that the distance cutoff point for remapping typos of `'american'`, `'asian'`, and `'italian'` cuisine types stored in the `cuisine_type` column should be 80.
# > 
# > In this exercise, you're going to put it all together by finding matches with similarity scores equal to or higher than 80 by using `fuzywuzzy.process`'s `extract()` function, for each correct cuisine type, and replacing these matches with it. Remember, when comparing a string with an array of strings using `process.extract()`, the output is a list of tuples where each is formatted like:
# > 
# >     (closest match, similarity score, index of match)
# >     
# > 
# > The `restaurants` DataFrame is in your environment, and you have access to a `categories` list containing the correct cuisine types (`'italian'`, `'asian'`, and `'american'`).

# In[8]:


categories = ['italian', 'asian', 'american']


# [Remapping categories II | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/record-linkage-4?ex=4)
# 
# > Return all of the unique values in the `cuisine_type` column of `restaurants`.

# In[10]:


# Inspect the unique values of the cuisine_type column
print(restaurants['cuisine_type'].unique())


# [Remapping categories II | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/record-linkage-4?ex=4)
# 
# > Okay! Looks like you will need to use some string matching to correct these misspellings!
# > 
# > -   As a first step, create a list of `matches`, comparing `'italian'` with the restaurant types listed in the `cuisine_type` column.

# In[11]:


# Create a list of matches, comparing 'italian' with the cuisine_type column
matches = process.extract('italian', restaurants['cuisine_type'], limit = len(restaurants['cuisine_type']))

# Inspect the first 5 matches
print(matches[0:5])


# [Remapping categories II | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/record-linkage-4?ex=4)
# 
# > Now you're getting somewhere! Now you can iterate through `matches` to reassign similar entries.
# > 
# > -   Within the `for` loop, use an `if` statement to check whether the similarity score in each `match` is greater than or equal to 80.
# > -   If it is, use `.loc` to select rows where `cuisine_type` in `restaurants` is _equal_ to the current match (which is the first element of `match`), and reassign them to be `'italian'`.

# In[15]:


# Create a list of matches, comparing 'italian' with the cuisine_type column
matches = process.extract('italian', restaurants['cuisine_type'], limit=len(restaurants.cuisine_type))

# Iterate through the list of matches to italian
for match in matches:
  # Check whether the similarity score is greater than or equal to 80
  if (match[1] >= 80):
    # Select all rows where the cuisine_type is spelled this way, and set them to the correct cuisine
    restaurants.loc[restaurants['cuisine_type'] == match[0], 'cuisine_type'] = 'italian'


# [Remapping categories II | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/record-linkage-4?ex=4)
# 
# > Finally, you'll adapt your code to work with every restaurant type in `categories`.
# > 
# > -   Using the variable `cuisine` to iterate through `categories`, embed your code from the previous step in an outer `for` loop.
# > -   Inspect the final result. _This has been done for you._

# In[16]:


# Iterate through categories
for cuisine in categories:  
  # Create a list of matches, comparing cuisine with the cuisine_type column
  matches = process.extract(cuisine, restaurants['cuisine_type'], limit=len(restaurants.cuisine_type))

  # Iterate through the list of matches
  for match in matches:
     # Check whether the similarity score is greater than or equal to 80
    if match[1] >= 80:
      # If it is, select all rows where the cuisine_type is spelled this way, and set them to the correct cuisine
      restaurants.loc[restaurants['cuisine_type'] == match[0]] = cuisine
      
# Inspect the final result
print(restaurants['cuisine_type'].unique())


# # Generating pairs
# 
# ```python
# 
# # Generating pairs
# # Import recordlinkage
# import recordlinkage
# # Create indexing object
# indexer = recordlinkage.Index()
# # Generate pairs blocked on state
# indexer.block('state')
# pairs = indexer.index(census_A, census_B)
# 
# # Comparing the DataFrames
# # Generate the pairs
# pairs = indexer.index(census_A, census_B)
# # Create a Compare object
# compare_cl = recordlinkage.Compare()
# # Find exact matches for pairs of date_of_birth and state
# compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
# compare_cl.exact('state', 'state', label='state')
# # Find similar matches for pairs of surname and address_1 using string similarity
# compare_cl.string('surname', 'surname', threshold=0.85, label='surname')
# compare_cl.string('address_1', 'address_1', threshold=0.85, label='address_1')
# # Find matches
# potential_matches = compare_cl.compute(pairs, census_A, census_B)
# 
# 
# # Finding the only pairs we want
# potential_matches[potential_matches.sum(axis = 1) => 2]
# ```

# ## Pairs of restaurants
# > 
# > In the last lesson, you cleaned the `restaurants` dataset to make it ready for building a restaurants recommendation engine. You have a new DataFrame named `restaurants_new` with new restaurants to train your model on, that's been scraped from a new data source.
# > 
# > You've already cleaned the `cuisine_type` and `city` columns using the techniques learned throughout the course. However you saw duplicates with typos in restaurants names that require record linkage instead of joins with `restaurants`.
# > 
# > In this exercise, you will perform the first step in record linkage and generate possible pairs of rows between `restaurants` and `restaurants_new`. Both DataFrames, `pandas` and `recordlinkage` are in your environment.

# ### init

# In[19]:


import recordlinkage


# In[20]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(restaurants, restaurants_new)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'restaurants.csv': 'https://file.io/qQPzMnqha5Pv',
  'restaurants_new.csv': 'https://file.io/5Z6r6lR8n4G6'}}
  """
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
restaurants = pd.read_csv(prefix+'restaurants.csv',index_col=0)
restaurants_new = pd.read_csv(prefix+'restaurants_new.csv',index_col=0)


# ### code

# [Pairs of restaurants | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/record-linkage-4?ex=7)
# 
# > -   Instantiate an indexing object by using the `Index()` function from `recordlinkage`.
# > -   Block your pairing on `cuisine_type` by using `indexer`'s' `.block()` method.
# > -   Generate pairs by indexing `restaurants` and `restaurants_new` in that order.

# In[21]:


# Create an indexer and object and find possible pairs
indexer = recordlinkage.Index()

# Block pairing on cuisine_type
indexer.block('cuisine_type')

# Generate pairs
pairs = indexer.index(restaurants, restaurants_new)


# ## Similar restaurants
# > 
# > In the last exercise, you generated pairs between `restaurants` and `restaurants_new` in an effort to cleanly merge both DataFrames using record linkage.
# > 
# > When performing record linkage, there are different types of matching you can perform between different columns of your DataFrames, including exact matches, string similarities, and more.
# > 
# > Now that your pairs have been generated and stored in `pairs`, you will find exact matches in the `city` and `cuisine_type` columns between each pair, and similar strings for each pair in the `rest_name` column. Both DataFrames, `pandas` and `recordlinkage` are in your environment.

# [Similar restaurants | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/record-linkage-4?ex=8)
# 
# > Instantiate a comparison object using the `recordlinkage.Compare()` function.

# In[22]:


# Create a comparison object
comp_cl = recordlinkage.Compare()


# [Similar restaurants | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/record-linkage-4?ex=8)
# 
# > -   Use the appropriate `comp_cl` method to find exact matches between the `city` and `cuisine_type` columns of both DataFrames.
# > -   Use the appropriate `comp_cl` method to find similar strings with a `0.8` similarity threshold in the `rest_name` column of both DataFrames.

# In[23]:


# Find exact matches on city, cuisine_types 
comp_cl.exact('city', 'city', label='city')
comp_cl.exact('cuisine_type', 'cuisine_type', label = 'cuisine_type')

# Find similar matches of rest_name
comp_cl.string('rest_name', 'rest_name', label='name', threshold = 0.8) 


# [Similar restaurants | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/record-linkage-4?ex=8)
# 
# > Compute the comparison of the pairs by using the `.compute()` method of `comp_cl`.

# In[24]:


# Get potential matches and print
potential_matches = comp_cl.compute(pairs, restaurants, restaurants_new)
print(potential_matches)


# [Similar restaurants | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/record-linkage-4?ex=8)
# 
# > Question
# > 
# > Print out `potential_matches`, the columns are the columns being compared, with values being 1 for a match, and 0 for not a match for each pair of rows in your DataFrames. To find potential matches, you need to find rows with more than matching value in a column. You can find them with
# > 
# >     potential_matches[potential_matches.sum(axis = 1) >= n]
# >     
# > 
# > Where `n` is the minimum number of columns you want matching to ensure a proper duplicate find, what do you think should the value of `n` be?

# In[28]:


potential_matches[potential_matches.sum(axis = 1) >= 3]


# # Linking DataFrames
# 
# ```python
# 
# 
# # Probable matches
# matches = potential_matches[potential_matches.sum(axis = 1) >= 3]
# print(matches)
# 
# # Get the indices
# matches.index
# MultiIndex(levels=[['rec-1007-org', 'rec-1016-org', 'rec-1054-org', 'rec-1066-org',
# 'rec-1070-org', 'rec-1075-org', 'rec-1080-org', 'rec-110-org', ...
# # Get indices from census_B only
# duplicate_rows = matches.index.get_level_values(1)
# print(census_B_index)
#                     
# # Linking DataFrames
# # Finding duplicates in census_B
# census_B_duplicates = census_B[census_B.index.isin(duplicate_rows)]
# # Finding new rows in census_B
# census_B_new = census_B[~census_B.index.isin(duplicate_rows)]
# # Link the DataFrames!
# full_census = census_A.append(census_B_new)   
#                     
# 
# # Recap
# # Import recordlinkage and generate pairs and compare across columns
# ...
# # Generate potential matches
# potential_matches = compare_cl.compute(full_pairs, census_A, census_B)
# # Isolate matches with matching values for 3 or more columns
# matches = potential_matches[potential_matches.sum(axis = 1) >= 3]
# # Get index for matching census_B rows only
# duplicate_rows = matches.index.get_level_values(1)
# # Finding new rows in census_B
# census_B_new = census_B[~census_B.index.isin(duplicate_rows)]
# # Link the DataFrames!
# full_census = census_A.append(census_B_new)
# ```

# ## Linking them together!
# > 
# > In the last lesson, you've finished the bulk of the work on your effort to link `restaurants` and `restaurants_new`. You've generated the different pairs of potentially matching rows, searched for exact matches between the `cuisine_type` and `city` columns, but compared for similar strings in the `rest_name` column. You stored the DataFrame containing the scores in `potential_matches`.
# > 
# > Now it's finally time to link both DataFrames. You will do so by first extracting all row indices of `restaurants_new` that are matching across the columns mentioned above from `potential_matches`. Then you will subset `restaurants_new` on these indices, then append the non-duplicate values to `restaurants`. All DataFrames are in your environment, alongside `pandas` imported as `pd`.

# [Linking them together! | Python](https://campus.datacamp.com/courses/cleaning-data-in-python/record-linkage-4?ex=11)
# 
# > -   Isolate instances of `potential_matches` where the row sum is above or equal to 3 by using the `.sum()` method.
# > -   Extract the second column index from `matches`, which represents row indices of matching record from `restaurants_new` by using the `.get_level_values()` method.
# > -   Subset `restaurants_new` for rows that are not in `matching_indices`.
# > -   Append `non_dup` to `restaurants`.

# In[30]:


# Isolate potential matches with row sum >=3
matches = potential_matches[potential_matches.sum(axis=1) >= 3]

# Get values of second column index of matches
matching_indices = matches.index.get_level_values(1)

# Subset restaurants_new based on non-duplicate values
non_dup = restaurants_new[~restaurants_new.index.isin(matching_indices)]

# Append non_dup to restaurants
full_restaurants = restaurants.append(non_dup)
print(full_restaurants)


# In[ ]:




