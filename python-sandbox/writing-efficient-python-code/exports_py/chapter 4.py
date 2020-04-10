#!/usr/bin/env python
# coding: utf-8

# # Intro to pandas DataFrame iteration
# 

# ## Iterating with .iterrows()
# In the video, we discussed that .iterrows() returns each DataFrame row as a tuple of (index, pandas Series) pairs. But, what does this mean? Let's explore with a few coding exercises.
# 
# A pandas DataFrame has been loaded into your session called pit_df. This DataFrame contains the stats for the Major League Baseball team named the Pittsburgh Pirates (abbreviated as 'PIT') from the year 2008 to the year 2012. It has been printed into your console for convenience.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(pit_df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'pit_df.csv': 'https://file.io/PEQmvZ'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
pit_df = pd.read_csv(prefix+'pit_df.csv',index_col=0)


# ### code

# In[2]:


# Iterate over pit_df and print each row
for i,row in pit_df.iterrows():
    print(row)


# In[3]:


# Iterate over pit_df and print each index variable and then each row
for i,row in pit_df.iterrows():
    print(i)
    print(row)
    print(type(row))


# In[4]:


# Use one variable instead of two to store the result of .iterrows()
for row_tuple in pit_df.iterrows():
    print(row_tuple)


# In[6]:


# Print the row and type of each row
for row_tuple in pit_df.iterrows():
    print(row_tuple)
    print(type(row_tuple))


# ## Run differentials with .iterrows()
# You've been hired by the San Francisco Giants as an analyst—congrats! The team's owner wants you to calculate a metric called the run differential for each season from the year 2008 to 2012. This metric is calculated by subtracting the total number of runs a team allowed in a season from the team's total number of runs scored in a season. 'RS' means runs scored and 'RA' means runs allowed.
# 
# The below function calculates this metric:
# ```
# def calc_run_diff(runs_scored, runs_allowed):
# 
#     run_diff = runs_scored - runs_allowed
# 
#     return run_diff
# ```
# A DataFrame has been loaded into your session as giants_df and printed into the console. Let's practice using .iterrows() to add a run differential column to this DataFrame.

# ### init

# In[7]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(giants_df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'giants_df.csv': 'https://file.io/N3kBks'}}
"""
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
giants_df = pd.read_csv(prefix+'giants_df.csv',index_col=0)


# In[12]:


def calc_run_diff(runs_scored, runs_allowed):

    run_diff = runs_scored - runs_allowed

    return run_diff


# ### code

# In[9]:


# Create an empty list to store run differentials
run_diffs = []


# In[14]:


# Write a for loop and collect runs allowed and runs scored for each row
for i,row in giants_df.iterrows():
    runs_scored = row['RS']
    runs_allowed = row['RA']
    
    # Use the provided function to calculate run_diff for each row
    run_diff = calc_run_diff(runs_scored, runs_allowed)
    
    # Append each run differential to the output list
    run_diffs.append(run_diff)

giants_df['RD'] = run_diffs
print(giants_df)


# # Another iterator method: .itertuples()
# 

# ## Iterating with .itertuples()
# Remember, .itertuples() returns each DataFrame row as a special data type called a namedtuple. You can look up an attribute within a namedtuple with a special syntax. Let's practice working with namedtuples.
# 
# A pandas DataFrame has been loaded into your session called rangers_df. This DataFrame contains the stats ('Team', 'League', 'Year', 'RS', 'RA', 'W', 'G', and 'Playoffs') for the Major League baseball team named the Texas Rangers (abbreviated as 'TEX').

# ### init

# In[15]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(rangers_df)
"""

tobedownloaded="""
 {pandas.core.frame.DataFrame: {'rangers_df.csv': 'https://file.io/6ihmVR'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
rangers_df = pd.read_csv(prefix+'rangers_df.csv',index_col=0)


# ### code

# In[16]:


# Loop over the DataFrame and print each row
for row in rangers_df.itertuples():
  print(row)


# In[19]:


# Loop over the DataFrame and print each row's Index, Year and Wins (W)
for row in rangers_df.itertuples():
  i = row.Index
  year = row.Year
  wins = row.W
  print(i, year, wins)


# In[20]:


# Loop over the DataFrame and print each row's Index, Year and Wins (W)
for row in rangers_df.itertuples():
  i = row.Index
  year = row.Year
  wins = row.W
    
  # Check if rangers made Playoffs (1 means yes; 0 means no)
  if row.Playoffs == 1:
    print(i, year, wins)


# ## Run differentials with .itertuples()
# The New York Yankees have made a trade with the San Francisco Giants for your analyst contract— you're a hot commodity! Your new boss has seen your work with the Giants and now wants you to do something similar with the Yankees data. He'd like you to calculate run differentials for the Yankees from the year 1962 to the year 2012 and find which season they had the best run differential.
# 
# You've remembered the function you used when working with the Giants and quickly write it down:
# ```
# def calc_run_diff(runs_scored, runs_allowed):
# 
#     run_diff = runs_scored - runs_allowed
# 
#     return run_diff
# ```
# 
# Let's use .itertuples() to loop over the yankees_df DataFrame (which has been loaded into your session) and calculate run differentials.

# ### init

# In[21]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(yankees_df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'yankees_df.csv': 'https://file.io/iaesii'}}
"""
prefixToc='2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
yankees_df = pd.read_csv(prefix+'yankees_df.csv',index_col=0)


# In[24]:


def calc_run_diff(runs_scored, runs_allowed):

    run_diff = runs_scored - runs_allowed

    return run_diff


# ### code

# In[25]:


run_diffs = []

# Loop over the DataFrame and calculate each row's run differential
for row in yankees_df.itertuples():
    
    runs_scored = row.RS
    runs_allowed = row.RA
    run_diff = calc_run_diff(runs_scored, runs_allowed)
    
    run_diffs.append(run_diff)


# In[26]:


# Append new column
yankees_df['RD'] = run_diffs
print(yankees_df)


# Question
# 
# In what year within your DataFrame did the New York Yankees have the highest run differential?
# You'll need to rerun the code that creates the 'RD' column if you'd like to analyze the DataFrame with code rather than looking at the console output.

# In[29]:


yankees_df[['Year', 'RD']].sort_values(by='RD', ascending=False).head()


# # pandas alternative to looping
# 

# ![image.png](attachment:image.png)

# ## Analyzing baseball stats with .apply()
# The Tampa Bay Rays want you to analyze their data.
# 
# They'd like the following metrics:
# 
# The sum of each column in the data
# The total amount of runs scored in a year ('RS' + 'RA' for each year)
# The 'Playoffs' column in text format rather than using 1's and 0's
# The below function can be used to convert the 'Playoffs' column to text:
# ```
# def text_playoffs(num_playoffs): 
#     if num_playoffs == 1:
#         return 'Yes'
#     else:
#         return 'No' 
# ```
# Use .apply() to get these metrics. A DataFrame (rays_df) has been loaded and printed to the console. This DataFrame is indexed on the 'Year' column.

# ### init

# In[30]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(rays_df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'rays_df.csv': 'https://file.io/ULWo8Q'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
rays_df = pd.read_csv(prefix+'rays_df.csv',index_col=0)


# In[31]:


def text_playoffs(num_playoffs): 
    if num_playoffs == 1:
        return 'Yes'
    else:
        return 'No' 


# ### code

# In[32]:


# Gather sum of all columns
stat_totals = rays_df.apply(sum, axis=0)
print(stat_totals)


# In[33]:


# Gather total runs scored in all games per year
total_runs_scored = rays_df[['RS', 'RA']].apply(sum, axis=1)
print(total_runs_scored)


# In[35]:


# Convert numeric playoffs to text
textual_playoffs = rays_df.apply(lambda row: text_playoffs(row['Playoffs']), axis=1)
print(textual_playoffs)


# ## Settle a debate with .apply()
# Word has gotten to the Arizona Diamondbacks about your awesome analytics skills. They'd like for you to help settle a debate amongst the managers. One manager claims that the team has made the playoffs every year they have had a win percentage of 0.50 or greater. Another manager says this is not true.
# 
# Let's use the below function and the .apply() method to see which manager is correct.
# ```
# def calc_win_perc(wins, games_played):
#     win_perc = wins / games_played
#     return np.round(win_perc,2)
# ```
# A DataFrame named dbacks_df has been loaded into your session.

# ### init

# In[36]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(dbacks_df)
"""

tobedownloaded="""
 {pandas.core.frame.DataFrame: {'dbacks_df.csv': 'https://file.io/bjqHbg'}}
"""
prefixToc='3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
dbacks_df = pd.read_csv(prefix+'dbacks_df.csv',index_col=0)


# In[37]:


def calc_win_perc(wins, games_played):
    win_perc = wins / games_played
    return np.round(win_perc,2)


# ### code

# In[38]:


# Display the first five rows of the DataFrame
print(dbacks_df.head())


# In[39]:


# Create a win percentage Series 
win_percs = dbacks_df.apply(lambda row: calc_win_perc(row['W'], row['G']), axis=1)
print(win_percs, '\n')


# In[40]:


# Append a new column to dbacks_df
dbacks_df['WP'] = win_percs
print(dbacks_df, '\n')

# Display dbacks_df where WP is greater than 0.50
print(dbacks_df[dbacks_df['WP'] >= 0.50])


# # Optimal pandas iterating
# 

# ## Replacing .iloc with underlying arrays
# Now that you have a better grasp on a DataFrame's internals let's update one of your previous analyses to leverage a DataFrame's underlying arrays. You'll revisit the win percentage calculations you performed row by row with the .iloc method:
# ```
# def calc_win_perc(wins, games_played):
#     win_perc = wins / games_played
#     return np.round(win_perc,2)
# 
# win_percs_list = []
# 
# for i in range(len(baseball_df)):
#     row = baseball_df.iloc[i]
# 
#     wins = row['W']
#     games_played = row['G']
# 
#     win_perc = calc_win_perc(wins, games_played)
# 
#     win_percs_list.append(win_perc)
# 
# baseball_df['WP'] = win_percs_list
# ```
# Let's update this analysis to use arrays instead of the .iloc method. A DataFrame (baseball_df) has been loaded into your session.

# ### init

# In[41]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(baseball_df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'baseball_df.csv': 'https://file.io/eq14qF'}}
"""
prefixToc='4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
baseball_df = pd.read_csv(prefix+'baseball_df.csv',index_col=0)


# In[42]:


def calc_win_perc(wins, games_played):
    win_perc = wins / games_played
    return np.round(win_perc,2)


# ### code

# In[44]:


# Use the W array and G array to calculate win percentages
win_percs_np = calc_win_perc(baseball_df['W'].values, baseball_df['G'].values)


# In[45]:


# Append a new column to baseball_df that stores all win percentages
baseball_df['WP'] = win_percs_np

print(baseball_df.head())


# Question
# 
# Use timeit in cell magic mode within your IPython console to compare the runtimes between the old code block using .iloc and the new code you developed using NumPy arrays.
# 
# Don't include the code that defines the calc_win_perc() function or the print() statements or when timing.
# 
# You should include eight lines of code when timing the old code block and two lines of code when timing the new code you developed. You may need to press SHIFT+ENTER when using timeit in cell magic mode to get to a new line within your IPython console.

# In[46]:


get_ipython().run_cell_magic('timeit', '', "win_percs_np = calc_win_perc(baseball_df['W'].values, baseball_df['G'].values)\nbaseball_df['WP'] = win_percs_np")


# In[47]:


get_ipython().run_cell_magic('timeit', '', "win_percs_list = []\n\nfor i in range(len(baseball_df)):\n    row = baseball_df.iloc[i]\n\n    wins = row['W']\n    games_played = row['G']\n\n    win_perc = calc_win_perc(wins, games_played)\n\n    win_percs_list.append(win_perc)\n\nbaseball_df['WP'] = win_percs_list")


# ## Bringing it all together: Predict win percentage
# A pandas DataFrame (baseball_df) has been loaded into your session. For convenience, a dictionary describing each column within baseball_df has been printed into your console. You can reference these descriptions throughout the exercise.
# 
# You'd like to attempt to predict a team's win percentage for a given season by using the team's total runs scored in a season ('RS') and total runs allowed in a season ('RA') with the following function:
# ```
# def predict_win_perc(RS, RA):
#     prediction = RS ** 2 / (RS ** 2 + RA ** 2)
#     return np.round(prediction, 2)
#     
# ```
# 
# 
# Let's compare the approaches you've learned to calculate a predicted win percentage for each season (or row) in your DataFrame.

# ### init

# In[48]:


def predict_win_perc(RS, RA):
    prediction = RS ** 2 / (RS ** 2 + RA ** 2)
    return np.round(prediction, 2)


# ### code

# In[49]:


win_perc_preds_loop = []

# Use a loop and .itertuples() to collect each row's predicted win percentage
for row in baseball_df.itertuples():
    runs_scored = row.RS
    runs_allowed = row.RA
    win_perc_pred = predict_win_perc(runs_scored, runs_allowed)
    win_perc_preds_loop.append(win_perc_pred)


# In[50]:


# Apply predict_win_perc to each row of the DataFrame
win_perc_preds_apply = baseball_df.apply(lambda row: predict_win_perc(row['RS'], row['RA']), axis=1)


# In[51]:


# Calculate the win percentage predictions using NumPy arrays
win_perc_preds_np = predict_win_perc(baseball_df['RS'].values, baseball_df['RA'].values)
baseball_df['WP_preds'] = win_perc_preds_np
print(baseball_df.head())


# Question
# 
# Compare runtimes within your IPython console between all three approaches used to calculate the predicted win percentages.
# 
# Use %%timeit (cell magic mode) to time the six lines of code (not including comment lines) for the .itertuples() approach. You may need to press SHIFT+ENTER after entering %%timeit to get to a new line within your IPython console.
# 
# Use %timeit (line magic mode) to time the .apply() approach and the NumPy array approach separately. Each has only one line of code (not including comment lines).
# 
# What is the order of approaches from fastest to slowest?

# In[52]:


get_ipython().run_cell_magic('timeit', '', "win_perc_preds_loop = []\n\n# Use a loop and .itertuples() to collect each row's predicted win percentage\nfor row in baseball_df.itertuples():\n    runs_scored = row.RS\n    runs_allowed = row.RA\n    win_perc_pred = predict_win_perc(runs_scored, runs_allowed)\n    win_perc_preds_loop.append(win_perc_pred)")


# In[53]:


get_ipython().run_cell_magic('timeit', '', "win_perc_preds_apply = baseball_df.apply(lambda row: predict_win_perc(row['RS'], row['RA']), axis=1)")


# In[54]:


get_ipython().run_cell_magic('timeit', '', "win_perc_preds_np = predict_win_perc(baseball_df['RS'].values, baseball_df['RA'].values)")


# ![image.png](attachment:image.png)

# In[ ]:




