#!/usr/bin/env python
# coding: utf-8

# # Why do missing values exist?
# 

# ## How sparse is my data?
# Most data sets contain missing values, often represented as NaN (Not a Number). If you are working with Pandas you can easily check how many missing values exist in each column.
# 
# Let's find out how many of the developers taking the survey chose to enter their age (found in the Age column of sub_df) and their gender (Gender column of sub_df).

# ### code

# In[1]:


# Import pandas
import pandas as pd

# Import so_survey_csv into so_survey_df
so_survey_df = pd.read_csv('data_from_datacamp/Combined_DS_v10.csv')


# In[2]:


# Subset the DataFrame
sub_df = so_survey_df[['Age', 'Gender']]

# Print the number of non-missing values
print(sub_df.notnull().sum())


# ## Finding the missing values
# While having a summary of how much of your data is missing can be useful, often you will need to find the exact locations of these missing values. Using the same subset of the StackOverflow data from the last exercise (sub_df), you will show how a value can be flagged as missing.

# ### code

# In[3]:


# Print the top 10 entries of the DataFrame
print(sub_df.head(10))


# In[4]:


# Print the locations of the missing values
print(sub_df.head(10).isnull())


# In[5]:


# Print the locations of the non-missing values
print(sub_df.head(10).notnull())


# # Dealing with missing values (I)
# 

# ## Listwise deletion
# The simplest way to deal with missing values in your dataset when they are occurring entirely at random is to remove those rows, also called 'listwise deletion'.
# 
# Depending on the use case, you will sometimes want to remove all missing values in your data while other times you may want to only remove a particular column if too many values are missing in that column.

# In[6]:


# Print the number of rows and columns
print(so_survey_df.shape)


# In[7]:


# Create a new DataFrame dropping all incomplete rows
no_missing_values_rows = so_survey_df.dropna(how='any')

# Print the shape of the new DataFrame
print(no_missing_values_rows.shape)


# In[8]:


# Create a new DataFrame dropping all columns with incomplete rows
no_missing_values_cols = so_survey_df.dropna(how='any', axis=1)

# Print the shape of the new DataFrame
print(no_missing_values_cols.shape)


# In[9]:


# Drop all rows where Gender is missing
no_gender = so_survey_df.dropna(subset=['Gender'], how='any')

# Print the shape of the new DataFrame
print(no_gender.shape)


# ## Replacing missing values with constants
# While removing missing data entirely maybe a correct approach in many situations, this may result in a lot of information being omitted from your models.
# 
# You may find categorical columns where the missing value is a valid piece of information in itself, such as someone refusing to answer a question in a survey. In these cases, you can fill all missing values with a new category entirely, for example 'No response given'.

# In[10]:


# Print the count of occurrences
print(so_survey_df['Gender'].value_counts())


# In[11]:


# Replace missing values
so_survey_df['Gender'].fillna('Not Given', inplace=True)

# Print the count of each value
print(so_survey_df['Gender'].value_counts())


# # Dealing with missing values (II)
# 

# ## Filling continuous missing values
# In the last lesson, you dealt with different methods of removing data missing values and filling in missing values with a fixed string. These approaches are valid in many cases, particularly when dealing with categorical columns but have limited use when working with continuous values. In these cases, it may be most valid to fill the missing values in the column with a value calculated from the entries present in the column.

# ### code

# In[12]:


# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df.StackOverflowJobsRecommend[:5])


# In[13]:


# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)

# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df['StackOverflowJobsRecommend'].head())


# In[14]:


# Round the StackOverflowJobsRecommend values
so_survey_df['StackOverflowJobsRecommend'] = round(so_survey_df['StackOverflowJobsRecommend'])

# Print the top 5 rows
print(so_survey_df['StackOverflowJobsRecommend'].head())


# # Dealing with other data issues
# 

# ## Dealing with stray characters (I)
# In this exercise, you will work with the RawSalary column which contains the wages of the respondents along with the currency symbols and commas, such as $42,000. When importing data from Microsoft Excel, more often that not you will come across data in this form.

# In[15]:


# Remove the commas in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(',', '')


# In[16]:


# Remove the dollar signs in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('$','')


# ## Dealing with stray characters (II)
# In the last exercise, you could tell quickly based off of the df.head() call which characters were causing an issue. In many cases this will not be so apparent. There will often be values deep within a column that are preventing you from casting a column as a numeric type so that it can be used in a model or further feature engineering.
# 
# One approach to finding these values is to force the column to the data type desired using pd.to_numeric(), coercing any values causing issues to NaN, Then filtering the DataFrame by just the rows containing the NaN values.
# 
# Try to cast the money_df column as a float and it will fail as an additional character can now be found in it. Find the character and remove it so the column can be cast as a float.

# ### code

# In[23]:


# Attempt to convert the column to numeric values
numeric_vals = pd.to_numeric(so_survey_df['RawSalary'], errors='coerce')

# Find the indexes of missing values
idx = numeric_vals.isnull()

# Print the relevant rows
print(so_survey_df['RawSalary'][idx])


# - Did you notice the pound (£) signs in the RawSalary column? Remove these signs like you did in the previous exercise.
# - Convert the RawSalary column to float.
# 

# In[24]:


# Replace the offending characters
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('£', '')

# Convert the column to float
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype('float')

# Print the column
print(so_survey_df['RawSalary'])


# ## Method chaining
# When applying multiple operations on the same column (like in the previous exercises), you made the changes in several steps, assigning the results back in each step. However, when applying multiple successive operations on the same column, you can "chain" these operations together for clarity and ease of management. This can be achieved by calling multiple methods sequentially:
# 
# Method chaining
# 
# df['column'] = df['column'].method1().method2().method3()
# 
# Same as 
# 
# df['column'] = df['column'].method1()
# 
# df['column'] = df['column'].method2()
# 
# df['column'] = df['column'].method3()
# 
# In this exercise you will repeat the steps you performed in the last two exercises, but do so using method chaining.

# ### code

# - Remove the commas (,) from the RawSalary column.
# - Remove the dollar ($) signs from the RawSalary column.
# - Remove the pound (£) signs from the RawSalary column.
# - Convert the RawSalary column to float.

# In[28]:


# Import pandas
import pandas as pd

# Import so_survey_csv into so_survey_df
so_survey_df = pd.read_csv('data_from_datacamp/Combined_DS_v10.csv')


# In[29]:


# Use method chaining
so_survey_df['RawSalary'] = so_survey_df['RawSalary']                              .str.replace(',','')                              .str.replace('$','')                              .str.replace('£','')                              .astype('float')
 
# Print the RawSalary column
print(so_survey_df['RawSalary'])


# In[ ]:




