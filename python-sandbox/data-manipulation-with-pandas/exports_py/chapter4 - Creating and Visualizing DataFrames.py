#!/usr/bin/env python
# coding: utf-8

# # Visualizing your data
# 
# ```python
# 
# # Histograms
# import matplotlib.pyplot as plt
# dog_pack["height_cm"].hist(bins=20)
# 
# # Bar plots
# avg_weight_by_breed = dog_pack.groupby("breed")["weight_kg"].mean()
# avg_weight_by_breed.plot(kind="bar", title="Mean Weight by Dog Breed")
# 
# # Line plots
# sully.head()
# sully.plot(x="date", y="weight_kg", kind="line")
# 
# # Rotating axis labels
# sully.plot(x="date", y="weight_kg", kind="line", rot=45)
# 
# # Scatter plots
# dog_pack.plot(x="height_cm", y="weight_kg", kind="scatter")
# 
# # Layering plots
# dog_pack[dog_pack["sex"]=="F"]["height_cm"].hist()
# dog_pack[dog_pack["sex"]=="M"]["height_cm"].hist()
# 
# # Add a legend
# plt.legend(["F", "M"])
# 
# # Transparency
# dog_pack[dog_pack["sex"]=="F"]["height_cm"].hist(alpha=0.7)
# dog_pack[dog_pack["sex"]=="M"]["height_cm"].hist(alpha=0.7)
# plt.legend(["F", "M"])
# ```

# [Which avocado size is most popular? | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=2)
# 
# > ## Which avocado size is most popular?
# > 
# > Avocados are increasingly popular and delicious in guacamole and on toast. The Hass Avocado Board keeps track of avocado supply and demand across the USA, including the sales of three different sizes of avocado. In this exercise, you'll use a bar plot to figure out which size is the most popular.
# > 
# > Bar plots are great for revealing relationships between categorical (size) and numeric (number sold) variables, but you'll often have to manipulate your data first in order to get the numbers you need for plotting.
# > 
# > `pandas` has been imported as `pd`, and `avocados` is available.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(avocados)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'avocados.csv': 'https://file.io/5TNo5ROfaGwW'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
avocados = pd.read_csv(prefix+'avocados.csv',index_col=0)


# ### code

# > -   Print the head of the `avocados` dataset. _What columns are available?_
# > -   For each avocado size group, calculate the total number sold, storing as `nb_sold_by_size`.
# > -   Create a bar plot of the number of avocados sold by size.
# > -   Show the plot.

# In[5]:


# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Look at the first few rows of data
print(avocados.head())

# Get the total number of avocados sold of each size
nb_sold_by_size = avocados.groupby('size')['nb_sold'].sum()

# Create a bar plot of the number of avocados sold by size
nb_sold_by_size.plot(kind='bar')

# Show the plot
plt.show()


# [Changes in sales over time | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=3)
# 
# > ## Changes in sales over time
# > 
# > Line plots are designed to visualize the relationship between two numeric variables, where each data values is connected to the next one. They are especially useful for visualizing the change in a number over time since each time point is naturally connected to the next time point. In this exercise, you'll visualize the change in avocado sales over three years.
# > 
# > `pandas` has been imported as `pd`.

# > -   Get the total number of avocados sold on each date. _The DataFrame has two rows for each date -- one for organic, and one for conventional_. Save this as `nb_sold_by_date`.
# > -   Create a line plot of the number of avocados sold.
# > -   Show the plot.

# In[7]:


# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Get the total number of avocados sold on each date
nb_sold_by_date = avocados.groupby('date')['nb_sold'].sum()

# Create a line plot of the number of avocados sold by date
nb_sold_by_date.plot(kind='line')

# Show the plot
plt.show()


# [Avocado supply and demand | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=4)
# 
# > ## Avocado supply and demand
# > 
# > Scatter plots are ideal for visualizing relationships between numerical variables. In this exercise, you'll compare the number of avocados sold to average price and see if they're at all related. If they're related, you may be able to use one number to predict the other.
# > 
# > `matplotlib.pyplot` has been imported as `plt` and `pandas` has been imported as `pd`.

# > -   Create a scatter plot with `nb_sold` on the x-axis and `avg_price` on the y-axis. Title it `"Number of avocados sold vs. average price"`.
# > -   Show the plot.

# In[17]:


plt.rcParams['figure.figsize'] = [11, 7]
plt.style.use('fivethirtyeight')


# In[18]:


# Scatter plot of nb_sold vs avg_price with title
avocados.plot(kind='scatter', x='nb_sold', y='avg_price', title='Number of avocados sold vs. average price')

# Show the plot
plt.show()


# [Price of conventional vs. organic avocados | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=5)
# 
# > ## Price of conventional vs. organic avocados
# > 
# > Creating multiple plots for different subsets of data allows you to compare groups. In this exercise, you'll create multiple histograms to compare the prices of conventional and organic avocados.
# > 
# > `matplotlib.pyplot` has been imported as `plt` and `pandas` has been imported as `pd`.

# > -   Subset `avocados` for the conventional type, and the average price column. Create a histogram.
# > -   Create a histogram of `avg_price` for organic type avocados.
# > -   Add a legend to your plot, with the names "conventional" and "organic".
# > -   Show your plot.

# In[25]:


# Histogram of conventional avg_price 
avocados[avocados.type=='conventional']['avg_price'].hist()

# Histogram of organic avg_price
avocados[avocados.type=='organic']['avg_price'].hist()

# Add a legend
plt.legend(['conventional', 'organic'])

# Show the plot
plt.show()


# > Modify your code to adjust the transparency of both histograms to `0.5` to see how much overlap there is between the two distributions.

# In[26]:


# Histogram of conventional avg_price 
avocados[avocados.type=='conventional']['avg_price'].hist(alpha=0.5)

# Histogram of organic avg_price
avocados[avocados.type=='organic']['avg_price'].hist(alpha=0.5)

# Add a legend
plt.legend(['conventional', 'organic'])

# Show the plot
plt.show()


# > Modify your code to use 20 bins in both histograms.

# In[27]:


# Histogram of conventional avg_price 
avocados[avocados.type=='conventional']['avg_price'].hist(bins=20, alpha=0.5)

# Histogram of organic avg_price
avocados[avocados.type=='organic']['avg_price'].hist(bins=20, alpha=0.5)

# Add a legend
plt.legend(['conventional', 'organic'])

# Show the plot
plt.show()


# # Missing values
# 
# ```python
# 
# # Detecting missing values
# dogs.isna()
# 
# # Detecting any missing values
# dogs.isna().any()
# 
# # Counting missing values
# dogs.isna().sum()
# 
# # Plotting missing values
# import matplotlib.pyplot as plt
# dogs.isna().sum().plot(kind="bar")
# plt.show()
# 
# # Removing rows containing missing values
# dogs.dropna()
# 
# # Replacing missing values
# dogs.fillna(0)
# 
# ```

# [Finding missing values | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=7)
# 
# > ## Finding missing values
# > 
# > Missing values are everywhere, and you don't want them interfering with your work. Some functions ignore missing data by default, but that's not always the behavior you might want. Some functions can't handle missing values at all, so these values need to be taken care of before you can use them. If you don't know where your missing values are, or if they exist, you could make mistakes in your analysis. In this exercise, you'll determine if there are missing values in the dataset, and if so, how many.
# > 
# > `pandas` has been imported as `pd` and `avocados_2016`, a subset of `avocados` that contains only sales from 2016, is available.

# ### init

# In[37]:


avocados_2016 = avocados[avocados.year == 2016].reset_index(drop=True)


# In[39]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(avocados_2016)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'avocados_2016.csv': 'https://file.io/4PnDNYnECG9L'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
avocados_2016 = pd.read_csv(prefix+'avocados_2016.csv',index_col=0)


# > -   Print a DataFrame that shows whether each value in `avocados_2016` is missing or not.
# > -   Print a summary that shows whether _any_ value in each column is missing or not.
# > -   Create a bar plot of the total number of missing values in each column.

# In[40]:


# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Check individual values for missing values
print(avocados_2016.isna())

# Check each column for missing values
print(avocados_2016.isna().any())

# Bar plot of missing values by variable
avocados_2016.isna().sum().plot(kind="bar")

# Show plot
plt.show()


# [Removing missing values | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=8)
# 
# > ## Removing missing values
# > 
# > Now that you know there are some missing values in your DataFrame, you have a few options to deal with them. One way is to remove them from the dataset completely. In this exercise, you'll remove missing values by removing all rows that contain missing values.
# > 
# > `pandas` has been imported as `pd` and `avocados_2016` is available.

# > -   Remove the rows of `avocados_2016` that contain missing values and store the remaining rows in `avocados_complete`.
# > -   Verify that all missing values have been removed from `avocados_complete`. Calculate each column that has NAs and print.

# In[41]:


# Remove rows with missing values
avocados_complete = avocados_2016.dropna()

# Check if any columns contain missing values
print(avocados_complete.isna().any())


# [Replacing missing values | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=9)
# 
# > ## Replacing missing values
# > 
# > Another way of handling missing values is to replace them all with the same value. For numerical variables, one option is to replace values with 0— you'll do this here. However, when you replace missing values, you make assumptions about what a missing value means. In this case, you will assume that a missing number sold means that no sales for that avocado type were made that week.
# > 
# > In this exercise, you'll see how replacing missing values can affect the distribution of a variable using histograms. You can plot histograms for multiple variables at a time as follows:
# > 
# >     dogs[["height_cm", "weight_kg"]].hist()
# >     
# > 
# > `pandas` has been imported as `pd` and `matplotlib.pyplot` has been imported as `plt`. The `avocados_2016` dataset is available.

# > -   A list has been created, `cols_with_missing`, containing the names of columns with missing values: `"small_sold"`, `"large_sold"`, and `"xl_sold"`.
# > -   Create a histogram of those columns.
# > -   Show the plot.

# In[42]:


# List the columns with missing values
cols_with_missing = ["small_sold", "large_sold", "xl_sold"]

# Create histograms showing the distributions cols_with_missing
avocados_2016[cols_with_missing].hist()

# Show the plot
plt.show()


# > -   Replace the missing values of `avocados_2016` with `0`s and store the result as `avocados_filled`.
# > -   Create a histogram of the `cols_with_missing` columns of `avocados_filled`.

# In[43]:


# Fill in missing values with 0
avocados_filled = avocados_2016.fillna(0)

# Create histograms of the filled columns
avocados_filled[cols_with_missing].hist()

# Show the plot
plt.show()


# # Creating DataFrames

# [List of dictionaries | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=11)
# 
# > ## List of dictionaries
# > 
# > You recently got some new avocado data from 2019 that you'd like to put in a DataFrame using the list of dictionaries method. Remember that with this method, you go through the data row by row.
# > 
# > ![image.png](attachment:image.png)
# > 
# > `pandas` as `pd` is imported.

# > -   Create a list of dictionaries with the new data called `avocados_list`.
# > -   Convert the list into a DataFrame called `avocados_2019`.
# > -   Print your new DataFrame.

# In[44]:


# Create a list of dictionaries with new data
avocados_list = [
    {'date': '2019-11-03', 'small_sold': 10376832, 'large_sold': 7835071},
    {'date': '2019-11-10', 'small_sold': 10717154, 'large_sold': 8561348},
]

# Convert list into DataFrame
avocados_2019 = pd.DataFrame(avocados_list)

# Print the new DataFrame
print(avocados_2019)


# [Dictionary of lists | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=12)
# 
# > ## Dictionary of lists
# > 
# > Some more data just came in! This time, you'll use the dictionary of lists method, parsing the data column by column.
# > 
# > ![image.png](attachment:image.png)
# > 
# > `pandas` as `pd` is imported.

# > -   Create a dictionary of lists with the new data called `avocados_dict`.
# > -   Convert the dictionary to a DataFrame called `avocados_2019`.
# > -   Print your new DataFrame.

# In[45]:


# Create a dictionary of lists with new data
avocados_dict = {
  "date": ['2019-11-17', '2019-12-01'],
  "small_sold": [10859987, 9291631],
  "large_sold": [7674135, 6238096]
}

# Convert dictionary into DataFrame
avocados_2019 = pd.DataFrame(avocados_dict)

# Print the new DataFrame
print(avocados_2019)


# # Reading and writing CSVs
# 
# ```python
# 
# # CSV to DataFrame
# import pandas as pd
# new_dogs = pd.read_csv("new_dogs.csv")
# 
# # DataFrame to CSV
# new_dogs.to_csv("new_dogs_with_bmi.csv")
# ```

# [CSV to DataFrame | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=14)
# 
# > ## CSV to DataFrame
# > 
# > You work for an airline, and your manager has asked you to do a competitive analysis and see how often passengers flying on other airlines are involuntarily bumped from their flights. You got a CSV file (`airline_bumping.csv`) from the Department of Transportation containing data on passengers that were involuntarily denied boarding in 2016 and 2017, but it doesn't have the exact numbers you want. In order to figure this out, you'll need to get the CSV into a pandas DataFrame and do some manipulation!
# > 
# > `pandas` is imported for you as `pd`. `"airline_bumping.csv"` is in your working directory.

# ### init

# In[46]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('airline_bumping.csv')
"""

tobedownloaded="""
{numpy.ndarray: {'airline_bumping.csv': 'https://file.io/t5h50W4stebe'}}
"""
prefixToc = '4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# > -   Read the CSV file `"airline_bumping.csv"` and store it as a DataFrame called `airline_bumping`.
# > -   Print the first few rows of `airline_bumping`.

# In[47]:


# Read CSV as DataFrame called airline_bumping
airline_bumping = pd.read_csv('airline_bumping.csv')

# Take a look at the DataFrame
print(airline_bumping.head())


# > For each airline group, select the `nb_bumped`, and `total_passengers` columns, and calculate the sum (for both years). Store this as `airline_totals`.

# In[49]:


# For each airline, select nb_bumped and total_passengers and sum
airline_totals = airline_bumping.groupby('airline')[['nb_bumped', 'total_passengers']].sum()
airline_totals


# > Create a new column of `airline_totals` called `bumps_per_10k`, which is the number of passengers bumped per 10,000 passengers in 2016 and 2017.

# In[52]:


# Create new col, bumps_per_10k: no. of bumps per 10k passengers for each airline
airline_totals["bumps_per_10k"] = airline_totals['nb_bumped'] / airline_totals['total_passengers'] * 10000
airline_totals


# > Print `airline_totals` to see the results of your manipulations.

# In[53]:


# Print airline_totals
print(airline_totals)


# [DataFrame to CSV | Python](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=15)
# 
# > ## DataFrame to CSV
# > 
# > You're almost there! To make things easier to read, you'll need to sort the data and export it to CSV so that your colleagues can read it.
# > 
# > `pandas` as `pd` has been imported for you.

# > -   Sort `airline_totals` by the values of `bumps_per_10k` from highest to lowest, storing as `airline_totals_sorted`.
# > -   Print your sorted DataFrame.
# > -   Save the sorted DataFrame as a CSV called `"airline_totals_sorted.csv"`.

# In[54]:


# Create airline_totals_sorted
airline_totals_sorted = airline_totals.sort_values('bumps_per_10k', ascending=False)

# Print airline_totals_sorted
print(airline_totals_sorted)

# Save as airline_totals_sorted.csv
airline_totals_sorted.to_csv('airline_totals_sorted.csv')


# In[ ]:




