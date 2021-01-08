#!/usr/bin/env python
# coding: utf-8

# # Summary statistics
# 
# ```python
# #Summarizing numerical data
# dogs["height_cm"].mean()
# 
# .median() , .mode()
# .min() , .max()
# .var() , .std()
# .sum()
# .quantile()
# 
# #The .agg() method
# def pct30(column):
# return column.quantile(0.3)
# dogs["weight_kg"].agg(pct30)
# 
# #Multiple summaries
# def pct40(column):
# return column.quantile(0.4)
# dogs["weight_kg"].agg([pct30, pct40])
# 
# #Cumulative sum
# dogs["weight_kg"].cumsum()
# 
# #Cumulative statistics
# .cummax()
# .cummin()
# .cumprod()
# ``` 

# [Mean and median | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-data-cb562b9d-5e79-4c37-b15f-30fc1567e8f1?ex=2)
# 
# > ## Mean and median
# > 
# > Summary statistics are exactly what they sound like - they summarize many numbers in one statistic. For example, mean, median, minimum, maximum, and standard deviation are summary statistics. Calculating summary statistics allows you to get a better sense of your data, even if there's a lot of it.
# > 
# > `sales` is available and `pandas` is loaded as `pd`.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(sales)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'sales.csv': 'https://file.io/AeA5ZTuOevt0'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
sales = pd.read_csv(prefix+'sales.csv',index_col=0)


# ### code

# > -   Explore your new DataFrame first by printing the first few rows of the `sales` DataFrame.
# > -   Print information about the columns in `sales`.
# > -   Print the mean of the `weekly_sales` column.
# > -   Print the median of the `weekly_sales` column.

# In[3]:


# Print the head of the sales DataFrame
print(sales.head())

# Print the info about the sales DataFrame
print(sales.info())

# Print the mean of weekly_sales
print(sales['weekly_sales'].mean())

# Print the median of weekly_sales
print(sales['weekly_sales'].median())


# [Summarizing dates | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-data-cb562b9d-5e79-4c37-b15f-30fc1567e8f1?ex=3)
# 
# > ## Summarizing dates
# > 
# > Summary statistics can also be calculated on date columns that have values with the data type `datetime64`. Some summary statistics — like mean — don't make a ton of sense on dates, but others are super helpful, for example, minimum and maximum, which allow you to see what time range your data covers.
# > 
# > `sales` is available and `pandas` is loaded as `pd`.

# In[5]:


# Print the maximum of the date column
print(sales['date'].max())

# Print the minimum of the date column
print(sales['date'].min())


# [Efficient summaries | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-data-cb562b9d-5e79-4c37-b15f-30fc1567e8f1?ex=4)
# 
# > ## Efficient summaries
# > 
# > While pandas and NumPy have tons of functions, sometimes, you may need a different function to summarize your data.
# > 
# > The `.agg()` method allows you to apply your own custom functions to a DataFrame, as well as apply functions to more than one column of a DataFrame at once, making your aggregations super-efficient. For example,
# > 
# >     df['column'].agg(function)
# >     
# > 
# > In the custom function for this exercise, "IQR" is short for inter-quartile range, which is the 75th percentile minus the 25th percentile. It's an alternative to standard deviation that is helpful if your data contains outliers.
# > 
# > `sales` is available and `pandas` is loaded as `pd`.

# > Use the custom `iqr` function defined for you along with `.agg()` to print the IQR of the `temperature_c` column of `sales`.

# In[6]:


# A custom IQR function
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)
    
# Print IQR of the temperature_c column
print(sales['temperature_c'].agg(iqr))


# > Update the column selection to use the custom `iqr` function with `.agg()` to print the IQR of `temperature_c`, `fuel_price_usd_per_l`, and `unemployment`, in that order.

# In[7]:


# A custom IQR function
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)

# Update to print IQR of temperature_c, fuel_price_usd_per_l, & unemployment
print(sales[["temperature_c", 'fuel_price_usd_per_l', 'unemployment']].agg(iqr))


# > Update the aggregation functions called by `.agg()`: include `iqr` and `np.median` in that order.

# In[9]:


# Import NumPy and create custom IQR function
import numpy as np
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)

# Update to print IQR and median of temperature_c, fuel_price_usd_per_l, & unemployment
print(sales[["temperature_c", "fuel_price_usd_per_l", "unemployment"]].agg([iqr, np.median]))


# [Cumulative statistics | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-data-cb562b9d-5e79-4c37-b15f-30fc1567e8f1?ex=5)
# 
# > ## Cumulative statistics
# > 
# > Cumulative statistics can also be helpful in tracking summary statistics over time. In this exercise, you'll calculate the cumulative sum and cumulative max of a department's weekly sales, which will allow you to identify what the total sales were so far as well as what the highest weekly sales were so far.
# > 
# > A DataFrame called `sales_1_1` has been created for you, which contains the sales data for department 1 of store 1. `pandas` is loaded as `pd`.

# ### init

# In[10]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(sales_1_1)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'sales_1_1.csv': 'https://file.io/ZcyKUdOj5yil'}}
"""
prefixToc='1.4'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
sales_1_1 = pd.read_csv(prefix+'sales_1_1.csv',index_col=0)


# ### code

# > -   Sort the rows of `sales_1_1` by the `date` column in ascending order.
# > -   Get the cumulative sum of `weekly_sales` and add it as a new column of `sales_1_1` called `cum_weekly_sales`.
# > -   Get the cumulative maximum of `weekly_sales`, and add it as a column called `cum_max_sales`.
# > -   Print the `date`, `weekly_sales`, `cum_weekly_sales`, and `cum_max_sales` columns.

# In[12]:


# Sort sales_1_1 by date
sales_1_1 = sales_1_1.sort_values('date', ascending=True)

# Get the cumulative sum of weekly_sales, add as cum_weekly_sales col
sales_1_1['cum_weekly_sales'] = sales_1_1['weekly_sales'].cumsum()

# Get the cumulative max of weekly_sales, add as cum_max_sales col
sales_1_1['cum_max_sales']=sales_1_1['weekly_sales'].cummax()

# See the columns you calculated
print(sales_1_1[["date", "weekly_sales", "cum_weekly_sales", "cum_max_sales"]])


# # Counting
# 
# ```python
# #Dropping duplicate names
# vet_visits.drop_duplicates(subset="name")
# 
# #Dropping duplicate pairs
# unique_dogs = vet_visits.drop_duplicates(subset=["name", "breed"])
# 
# #Counting
# unique_dogs["breed"].value_counts(sort=True)
# ```

# [Dropping duplicates | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-data-cb562b9d-5e79-4c37-b15f-30fc1567e8f1?ex=7)
# 
# > ## Dropping duplicates
# > 
# > Removing duplicates is an essential skill to get accurate counts because often, you don't want to count the same thing multiple times. In this exercise, you'll create some new DataFrames using unique values from `sales`.
# > 
# > `sales` is available and `pandas` is imported as `pd`.

# > -   Remove rows of `sales` with duplicate pairs of `store` and `type` and save as `store_types` and print the head.
# > -   Remove rows of `sales` with duplicate pairs of `store` and `department` and save as `store_depts` and print the head.
# > -   Subset the rows that are holiday weeks using the `is_holiday` column, and drop the duplicate `date`s, saving as `holiday_dates`.
# > -   Select the `date` column of `holiday_dates`, and print.

# In[14]:


# Drop duplicate store/type combinations
store_types = sales.drop_duplicates(subset=['store', 'type'])
print(store_types.head())

# Drop duplicate store/department combinations
store_depts = sales.drop_duplicates(subset=['store', 'department'])
print(store_depts.head())

# Subset the rows where is_holiday is True and drop duplicate dates
holiday_dates = sales[sales['is_holiday']==True].drop_duplicates(subset='date')

# Print date col of holiday_dates
print(holiday_dates['date'])


# [Counting categorical variables | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-data-cb562b9d-5e79-4c37-b15f-30fc1567e8f1?ex=8)
# 
# > ## Counting categorical variables
# > 
# > Counting is a great way to get an overview of your data and to spot curiosities that you might not notice otherwise. In this exercise, you'll count the number of each type of store and the number of each department number using the DataFrames you created in the previous exercise:
# > 
# >     # Drop duplicate store/type combinations
# >     store_types = sales.drop_duplicates(subset=["store", "type"])
# >     
# >     # Drop duplicate store/department combinations
# >     store_depts = sales.drop_duplicates(subset=["store", "department"])
# >     
# > 
# > The `store_types` and `store_depts` DataFrames you created in the last exercise are available, and `pandas` is imported as `pd`.

# > -   Count the number of stores of each store `type` in `store_types`.
# > -   Count the proportion of stores of each store `type` in `store_types`.
# > -   Count the number of different `department`s in `store_depts`, sorting the counts in descending order.
# > -   Count the proportion of different `department`s in `store_depts`, sorting the proportions in descending order.

# In[19]:


# Count the number of stores of each type
store_counts = store_types['type'].value_counts()
print(store_counts)

# Get the proportion of stores of each type
store_props = store_counts/len(store_types)
print(store_props)

# Count the number of each department number and sort
dept_counts_sorted = store_depts['department'].value_counts(sort=True, ascending=False)
print(dept_counts_sorted)

# Get the proportion of departments of each number and sort
dept_props_sorted = store_depts['department'].value_counts(sort=True, normalize=True)
print(dept_props_sorted)


# # Grouped summary statistics
# 
# ```python
# 
# #Summaries by group
# dogs[dogs["color"] == "Black"]["weight_kg"].mean()
# dogs[dogs["color"] == "Brown"]["weight_kg"].mean()
# 
# #Grouped summaries
# dogs.groupby("color")["weight_kg"].mean()
# 
# #Multiple grouped summaries
# dogs.groupby("color")["weight_kg"].agg([min, max, sum])
# 
# #Grouping by multiple variables
# dogs.groupby(["color", "breed"])["weight_kg"].mean()
# 
# #Many groups, many summaries
# dogs.groupby(["color", "breed"])[["weight_kg", "height_cm"]].mean()
# ```

# [What percent of sales occurred at each store type? | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-data-cb562b9d-5e79-4c37-b15f-30fc1567e8f1?ex=10)
# 
# > ## What percent of sales occurred at each store type?
# > 
# > While `.groupby()` is useful, you can calculate grouped summary statistics without it.
# > 
# > Walmart distinguishes three types of stores: "supercenters," "discount stores," and "neighborhood markets," encoded in this dataset as type "A," "B," and "C." In this exercise, you'll calculate the total sales made at each store type, without using `.groupby()`. You can then use these numbers to see what proportion of Walmart's total sales were made at each type.
# > 
# > `sales` is available and `pandas` is imported as `pd`.

# > -   Calculate the total `weekly_sales` over the whole dataset.
# > -   Subset for `type` `"A"` stores, and calculate their total weekly sales.
# > -   Do the same for `type` `"B"` and `type` `"C"` stores.
# > -   Combine the A/B/C results into a list, and divide by `sales_all` to get the proportion of sales by type.

# In[21]:


# Calc total weekly sales
sales_all = sales["weekly_sales"].sum()

# Subset for type A stores, calc total weekly sales
sales_A = sales[sales["type"] == "A"]["weekly_sales"].sum()

# Subset for type B stores, calc total weekly sales
sales_B = sales[sales["type"] == "B"]["weekly_sales"].sum()

# Subset for type C stores, calc total weekly sales
sales_C = sales[sales["type"] == "C"]["weekly_sales"].sum()

# Get proportion for each type
sales_propn_by_type = [sales_A, sales_B, sales_C] / sales_all
print(sales_propn_by_type)


# [Calculations with .groupby() | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-data-cb562b9d-5e79-4c37-b15f-30fc1567e8f1?ex=11)
# 
# > ## Calculations with .groupby()
# > 
# > The `.groupby()` method makes life much easier. In this exercise, you'll perform the same calculations as last time, except you'll use the `.groupby()` method. You'll also perform calculations on data grouped by two variables to see if sales differ by store type depending on if it's a holiday week or not.
# > 
# > `sales` is available and `pandas` is loaded as `pd`.

# > -   Group `sales` by `"type"`, take the sum of `"weekly_sales"`, and store as `sales_by_type`.
# > -   Calculate the proportion of sales at each store type by dividing by the sum of `sales_by_type`. Assign to `sales_propn_by_type`.

# In[23]:


# Group by type; calc total weekly sales
sales_by_type = sales.groupby("type")["weekly_sales"].sum()

# Get proportion for each type
sales_propn_by_type = sales_by_type / sum(sales_by_type)
print(sales_propn_by_type)


# > Group `sales` by `"type"` and "`is_holiday`", take the sum of `weekly_sales`, and store as `sales_by_type_is_holiday`.

# In[24]:


# Group by type and is_holiday; calc total weekly sales
sales_by_type_is_holiday = sales.groupby(['type', 'is_holiday'])["weekly_sales"].sum()
print(sales_by_type_is_holiday)


# [Multiple grouped summaries | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-data-cb562b9d-5e79-4c37-b15f-30fc1567e8f1?ex=12)
# 
# > ## Multiple grouped summaries
# > 
# > Earlier in this chapter, you saw that the `.agg()` method is useful to compute multiple statistics on multiple variables. It also works with grouped data. NumPy, which is imported as `np`, has many different summary statistics functions, including: `np.min`, `np.max`, `np.mean`, and `np.median`.
# > 
# > `sales` is available and `pandas` is imported as `pd`.

# > -   Import `numpy` with the alias `np`.
# > -   Get the min, max, mean, and median of `weekly_sales` for each store type using `.groupby()` and `.agg()`. Store this as `sales_stats`. Make sure to use `numpy` functions!
# > -   Get the min, max, mean, and median of `unemployment` and `fuel_price_usd_per_l` for each store type. Store this as `unemp_fuel_stats`.

# In[33]:


# Import numpy with the alias np
import numpy as np

# For each store type, aggregate weekly_sales: get min, max, mean, and median
sales_stats = sales.groupby('type')['weekly_sales'].agg([min, max, np.mean, np.median])

# Print sales_stats
print(sales_stats)

# For each store type, aggregate unemployment and fuel_price_usd_per_l: get min, max, mean, and median
unemp_fuel_stats = sales.groupby('type')[['unemployment', 'fuel_price_usd_per_l']].agg([min, max,np.mean, np.median])

# Print unemp_fuel_stats
print(unemp_fuel_stats)


# # Pivot tables
# 
# ```python
# 
# #pivot table
# dogs.pivot_table(values="weight_kg",index="color")
# 
# #Different statistics
# import numpy as np
# dogs.pivot_table(values="weight_kg", index="color", aggfunc=np.median)
# 
# #Multiple statistics
# dogs.pivot_table(values="weight_kg", index="color", aggfunc=[np.mean, np.median])
# 
# #Pivot on two variables
# dogs.groupby(["color", "breed"])["weight_kg"].mean()
# dogs.pivot_table(values="weight_kg", index="color", columns="breed")
# 
# #Filling missing values in pivot tables
# dogs.pivot_table(values="weight_kg", index="color", columns="breed", fill_value=0)
# 
# # Summing with pivot tables
# dogs.pivot_table(values="weight_kg", index="color", columns="breed",
# fill_value=0, margins=True)
# 
# ```

# [Pivoting on one variable | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-data-cb562b9d-5e79-4c37-b15f-30fc1567e8f1?ex=14)
# 
# > ## Pivoting on one variable
# > 
# > Pivot tables are the standard way of aggregating data in spreadsheets. In pandas, pivot tables are essentially just another way of performing grouped calculations. That is, the `.pivot_table()` method is just an alternative to `.groupby()`.
# > 
# > In this exercise, you'll perform calculations using `.pivot_table()` to replicate the calculations you performed in the last lesson using `.groupby()`.
# > 
# > `sales` is available and `pandas` is imported as `pd`.

# > Get the mean `weekly_sales` by `type` using `.pivot_table()` and store as `mean_sales_by_type`.

# In[35]:


# Pivot for mean weekly_sales for each store type
mean_sales_by_type = sales.pivot_table(values='weekly_sales', index='type')

# Print mean_sales_by_type
print(mean_sales_by_type)


# > Get the mean and median (using NumPy functions) of `weekly_sales` by `type` using `.pivot_table()` and store as `mean_med_sales_by_type`.

# In[36]:


# Import NumPy as np
import numpy as np

# Pivot for mean and median weekly_sales for each store type
mean_med_sales_by_type = sales.pivot_table(values='weekly_sales', index='type', aggfunc=[np.mean, np.median])

# Print mean_med_sales_by_type
print(mean_med_sales_by_type)


# > Get the mean of `weekly_sales` by `type` and `is_holiday` using `.pivot_table()` and store as `mean_sales_by_type_holiday`.

# In[37]:


# Pivot for mean weekly_sales by store type and holiday 
mean_sales_by_type_holiday = sales.pivot_table(values="weekly_sales", index='type', columns='is_holiday')

# Print mean_sales_by_type_holiday
print(mean_sales_by_type_holiday)


# [Fill in missing values and sum values with pivot tables | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-data-cb562b9d-5e79-4c37-b15f-30fc1567e8f1?ex=15)
# 
# > ## Fill in missing values and sum values with pivot tables
# > 
# > The `.pivot_table()` method has several useful arguments, including `fill_value` and `margins`.
# > 
# > -   `fill_value` replaces missing values with a real value (known as _imputation_). What to replace missing values with is a topic big enough to have its own course ([Dealing with Missing Data in Python](https://www.datacamp.com/courses/dealing-with-missing-data-in-python)), but the simplest thing to do is to substitute a dummy value.
# > -   `margins` is a shortcut for when you pivoted by two variables, but also wanted to pivot by each of those variables separately: it gives the row and column totals of the pivot table contents.
# > 
# > In this exercise, you'll practice using these arguments to up your pivot table skills, which will help you crunch numbers more efficiently!
# > 
# > `sales` is available and `pandas` is imported as `pd`.

# > Print the mean `weekly_sales` by `department` and `type`, filling in any missing values with `0`.

# In[39]:


# Print mean weekly_sales by department and type; fill missing values with 0
print(sales.pivot_table(values="weekly_sales", index='department', columns='type', fill_value=0))


# > Print the mean `weekly_sales` by `department` and `type`, filling in any missing values with `0` and summing all rows and columns.

# In[41]:


# Print the mean weekly_sales by department and type; fill missing values with 0s; sum all rows and cols
print(sales.pivot_table(values="weekly_sales", index="department", columns="type", fill_value=0, margins=True))


# In[ ]:




