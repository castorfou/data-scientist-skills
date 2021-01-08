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


# [Parts of a DataFrame | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/transforming-data?ex=3)
# 
# > ## Parts of a DataFrame
# > 
# > To better understand DataFrame objects, it's useful to know that they consist of three components, stored as attributes:
# > 
# > -   `.values`: A two-dimensional NumPy array of values.
# > -   `.columns`: An index of columns: the column names.
# > -   `.index`: An index for the rows: either row numbers or row names.
# > 
# > You can usually think of indexes as a list of strings or numbers, though the pandas `Index` data type allows for more sophisticated options. (These will be covered later in the course.)
# > 
# > `homelessness` is available.

# In[2]:


# Import pandas using the alias pd
import pandas as pd

# Print the values of homelessness
print(homelessness.values)

# Print the column index of homelessness
print(homelessness.columns)

# Print the row index of homelessness
print(homelessness.index)


# # Sorting and subsetting
# 
# ```python
# #Sorting by multiple variables
# dogs.sort_values(["weight_kg", "height_cm"], ascending=[True, False])
# 
# #Subsetting based on dates
# dogs[dogs["date_of_birth"] > "2015-01-01"]
# 
# #Subsetting based on multiple conditions
# is_lab = dogs["breed"] == "Labrador"
# is_brown = dogs["color"] == "Brown"
# dogs[is_lab & is_brown]
# dogs[ (dogs["breed"] == "Labrador") & (dogs["color"] == "Brown") ]
# 
# #Subsetting using .isin()
# is_black_or_brown = dogs["color"].isin(["Black", "Brown"])
# dogs[is_black_or_brown]
# ```

# [Sorting rows | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/transforming-data?ex=5)
# 
# > ## Sorting rows
# > 
# > Finding interesting bits of data in a DataFrame is often easier if you change the order of the rows. You can sort the rows by passing a column name to `.sort_values()`.
# > 
# > In cases where rows have the same value (this is common if you sort on a categorical variable), you may wish to break the ties by sorting on another column. You can sort on multiple columns in this way by passing a list of column names.
# > 
# > ![image.png](attachment:image.png)
# > 
# > By combining `.sort_values()` with `.head()`, you can answer questions in the form, "What are the top cases where…?".
# > 
# > `homelessness` is available and `pandas` is loaded as `pd`.

# > -   Sort `homelessness` by the number of homeless individuals, from smallest to largest, and save this as `homelessness_ind`.
# > -   Print the head of the sorted DataFrame.

# In[4]:


homelessness.head()


# In[5]:


# Sort homelessness by individual
homelessness_ind = homelessness.sort_values(['individuals'], ascending=True)

# Print the top few rows
print(homelessness_ind.head())


# > -   Sort `homelessness` by the number of homeless `family_members` in descending order, and save this as `homelessness_fam`.
# > -   Print the head of the sorted DataFrame.

# In[6]:


# Sort homelessness by descending family members
homelessness_fam = homelessness.sort_values(['family_members'], ascending=False)

# Print the top few rows
print(homelessness_fam.head())


# > -   Sort `homelessness` first by region (ascending), and then by number of family members (descending). Save this as `homelessness_reg_fam`.
# > -   Print the head of the sorted DataFrame.

# In[7]:


# Sort homelessness by region, then descending family members
homelessness_reg_fam = homelessness.sort_values(['region', 'family_members'], ascending=[True, False])


# Print the top few rows
print(homelessness_reg_fam.head())


# [Subsetting columns | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/transforming-data?ex=6)
# 
# > ## Subsetting columns
# > 
# > When working with data, you may not need all of the variables in your dataset. Square brackets (`[]`) can be used to select only the columns that matter to you in an order that makes sense to you. To select only `"col_a"` of the DataFrame `df`, use
# > 
# >     df["col_a"]
# >     
# > 
# > To select `"col_a"` and `"col_b"` of `df`, use
# > 
# >     df[["col_a", "col_b"]]
# >     
# > 
# > `homelessness` is available and `pandas` is loaded as `pd`.

# > -   Create a DataFrame called `individuals` that contains only the `individuals` column of `homelessness`.
# > -   Print the head of the result.

# In[8]:


# Select the individuals column
individuals = homelessness['individuals']

# Print the head of the result
print(individuals.head())


# > -   Create a DataFrame called `state_fam` that contains only the `state` and `family_members` columns of `homelessness`, in that order.
# > -   Print the head of the result.

# In[9]:


# Select the state and family_members columns
state_fam = homelessness[['state', 'family_members']]

# Print the head of the result
print(state_fam.head())


# > -   Create a DataFrame called `ind_state` that contains the `individuals` and `state` columns of `homelessness`, in that order.
# > -   Print the head of the result.

# In[10]:


# Select only the individuals and state columns, in that order
ind_state = homelessness[['individuals', 'state']]

# Print the head of the result
print(ind_state.head())


# [Subsetting rows | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/transforming-data?ex=7)
# 
# > ## Subsetting rows
# > 
# > A large part of data science is about finding which bits of your dataset are interesting. One of the simplest techniques for this is to find a subset of rows that match some criteria. This is sometimes known as _filtering rows_ or _selecting rows_.
# > 
# > There are many ways to subset a DataFrame, perhaps the most common is to use relational operators to return `True` or `False` for each row, then pass that inside square brackets.
# > 
# >     dogs[dogs["height_cm"] > 60]
# >     dogs[dogs["color"] == "tan"]
# >     
# > 
# > You can filter for multiple conditions at once by using the "bitwise and" operator, `&`.
# > 
# >     dogs[(dogs["height_cm"] > 60) & (dogs["color"] == "tan")]
# >     
# > 
# > `homelessness` is available and `pandas` is loaded as `pd`.

# > Filter `homelessness` for cases where the number of individuals is greater than ten thousand, assigning to `ind_gt_10k`. _View the printed result._

# In[11]:


# Filter for rows where individuals is greater than 10000
ind_gt_10k = homelessness[homelessness['individuals']>10000]

# See the result
print(ind_gt_10k)


# > Filter `homelessness` for cases where the USA Census region is `"Mountain"`, assigning to `mountain_reg`. _View the printed result._

# In[12]:


# Filter for rows where region is Mountain
mountain_reg = homelessness[homelessness['region']=='Mountain']

# See the result
print(mountain_reg)


# > Filter `homelessness` for cases where the number of `family_members` is less than one thousand and the `region` is "Pacific", assigning to `fam_lt_1k_pac`. _View the printed result._

# In[13]:


# Filter for rows where family_members is less than 1000 
# and region is Pacific
fam_lt_1k_pac = homelessness[(homelessness['family_members']<10000) & (homelessness[('region')]=='Pacific')]

# See the result
print(fam_lt_1k_pac)


# [Subsetting rows by categorical variables | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/transforming-data?ex=8)
# 
# > ## Subsetting rows by categorical variables
# > 
# > Subsetting data based on a categorical variable often involves using the "or" operator (`|`) to select rows from multiple categories. This can get tedious when you want all states in one of three different regions, for example. Instead, use the `.isin()` method, which will allow you to tackle this problem by writing one condition instead of three separate ones.
# > 
# >     colors = ["brown", "black", "tan"]
# >     condition = dogs["color"].isin(colors)
# >     dogs[condition]
# >     
# > 
# > `homelessness` is available and `pandas` is loaded as `pd`.

# > Filter `homelessness` for cases where the USA census region is "South Atlantic" or it is "Mid-Atlantic", assigning to `south_mid_atlantic`. _View the printed result._

# In[15]:


# Subset for rows in South Atlantic or Mid-Atlantic regions
south_mid_atlantic = homelessness[homelessness['region'].isin(['South Atlantic', 'Mid-Atlantic'])]

# See the result
print(south_mid_atlantic)


# > Filter `homelessness` for cases where the USA census `state` is in the list of Mojave states, `canu`, assigning to `mojave_homelessness`. _View the printed result._

# In[16]:


# The Mojave Desert states
canu = ["California", "Arizona", "Nevada", "Utah"]

# Filter for rows in the Mojave Desert states
mojave_homelessness = homelessness[homelessness['state'].isin(canu)]

# See the result
print(mojave_homelessness)


# # New columns

# [Adding new columns | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/transforming-data?ex=10)
# 
# > ## Adding new columns
# > 
# > You aren't stuck with just the data you are given. Instead, you can add new columns to a DataFrame. This has many names, such as _transforming_, _mutating_, and _feature engineering_.
# > 
# > You can create new columns from scratch, but it is also common to derive them from other columns, for example, by adding columns together or by changing their units.
# > 
# > `homelessness` is available and `pandas` is loaded as `pd`.

# > -   Add a new column to `homelessness`, named `total`, containing the sum of the `individuals` and `family_members` columns.
# > -   Add another column to `homelessness`, named `p_individuals`, containing the proportion of homeless people in each state who are individuals.

# In[17]:


# Add total col as sum of individuals and family_members
homelessness['total']=homelessness['individuals']+homelessness['family_members']

# Add p_individuals col as proportion of individuals
homelessness['p_individuals']=homelessness['individuals']/homelessness['total']

# See the result
print(homelessness)


# [Combo-attack! | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/transforming-data?ex=11)
# 
# > ## Combo-attack!
# > 
# > You've seen the four most common types of data manipulation: sorting rows, subsetting columns, subsetting rows, and adding new columns. In a real-life data analysis, you can mix and match these four manipulations to answer a multitude of questions.
# > 
# > In this exercise, you'll answer the question, "Which state has the highest number of homeless individuals per 10,000 people in the state?" Combine your new `pandas` skills to find out.

# > -   Add a column to `homelessness`, `indiv_per_10k`, containing the number of homeless individuals per ten thousand people in each state.
# > -   Subset rows where `indiv_per_10k` is higher than `20`, assigning to `high_homelessness`.
# > -   Sort `high_homelessness` by descending `indiv_per_10k`, assigning to `high_homelessness_srt`.
# > -   Select only the `state` and `indiv_per_10k` columns of `high_homelessness_srt` and save as `result`. _Look at the `result`._

# In[18]:


# Create indiv_per_10k col as homeless individuals per 10k state pop
homelessness["indiv_per_10k"] = 10000 * homelessness['individuals'] / homelessness['state_pop'] 

# Subset rows for indiv_per_10k greater than 20
high_homelessness = homelessness[homelessness['indiv_per_10k']>20]

# Sort high_homelessness by descending indiv_per_10k
high_homelessness_srt = high_homelessness.sort_values(['indiv_per_10k'], ascending=False)

# From high_homelessness_srt, select the state and indiv_per_10k cols
result = high_homelessness_srt[['state', 'indiv_per_10k']]

# See the result
print(result)


# ![image.png](attachment:image.png)

# In[ ]:




