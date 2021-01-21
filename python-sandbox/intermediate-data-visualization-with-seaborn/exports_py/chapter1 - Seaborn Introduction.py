#!/usr/bin/env python
# coding: utf-8

# # Introduction to Seaborn
# 
# ```python
# # Seaborn distplot 
# import seaborn as sns
# sns.distplot(df['alcohol'])
# ```

# [Reading a csv file | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/seaborn-introduction?ex=3)
# 
# > ## Reading a csv file
# > 
# > Before you analyze data, you will need to read the data into a [pandas](https://pandas.pydata.org/) DataFrame. In this exercise, you will be looking at data from US School Improvement Grants in 2010. This program gave nearly \$4B to schools to help them renovate or improve their programs.
# > 
# > This first step in most data analysis is to import `pandas` and `seaborn` and read a data file in order to analyze it further.
# > 
# > _This course introduces a lot of new concepts, so if you ever need a quick refresher, download the [Seaborn Cheat Sheet](https://datacamp-community-prod.s3.amazonaws.com/f9f06e72-519a-4722-9912-b5de742dbac4) and keep it handy!_

# In[1]:


grant_file='https://s3.amazonaws.com/assets.datacamp.com/production/course_7030/datasets/schoolimprovement2010grants.csv'


# > -   Import `pandas` and `seaborn` using the standard naming conventions.
# > -   The path to the csv file is stored in the `grant_file` variable.
# > -   Use `pandas` to read the file.
# > -   Store the resulting DataFrame in the variable `df`.

# In[2]:


# import all modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the DataFrame
df = pd.read_csv(grant_file)


# [Comparing a histogram and distplot | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/seaborn-introduction?ex=4)
# 
# > ## Comparing a histogram and distplot
# > 
# > The `pandas` library supports simple plotting of data, which is very convenient when data is already likely to be in a `pandas` DataFrame.
# > 
# > Seaborn generally does more statistical analysis on data and can provide more sophisticated insight into the data. In this exercise, we will compare a `pandas` histogram vs the `seaborn` distplot.

# > Use the pandas' `plot.hist()` function to plot a histogram of the `Award_Amount` column.

# In[3]:


# Display pandas histogram
df['Award_Amount'].plot.hist()
plt.show()

# Clear out the pandas histogram
plt.clf()


# > Use Seaborn's `distplot()` function to plot a distribution plot of the same column.

# In[4]:


# Display a Seaborn distplot
sns.distplot(df['Award_Amount'])
plt.show()

# Clear the distplot
plt.clf()


# In[5]:


# Display a Seaborn distplot
sns.displot(df['Award_Amount'])
plt.show()

# Clear the distplot
plt.clf()


# # Using the distribution plot
# 
# ```python
# # Creating a histogram
# sns.distplot(df['alcohol'], kde=False, bins=10)
# 
# # Alternative data distributions
# sns.distplot(df['alcohol'], hist=False, rug=True)
# 
# # Further Customizations
# sns.distplot(df['alcohol'], hist=False,rug=True, kde_kws={'shade':True})
# ```

# [Plot a histogram | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/seaborn-introduction?ex=6)
# 
# > ## Plot a histogram
# > 
# > The `distplot()` function will return a Kernel Density Estimate (KDE) by default. The KDE helps to smooth the distribution and is a useful way to look at the data. However, Seaborn can also support the more standard histogram approach if that is more meaningful for your analysis.

# > -   Create a `distplot` for the data and disable the KDE.
# > -   Explicitly pass in the number 20 for the number of bins in the histogram.
# > -   Display the plot using `plt.show()`.

# In[7]:


# Create a distplot
sns.distplot(df['Award_Amount'],
             kde=False,
             bins=20)

# Display the plot
plt.show()


# [Rug plot and kde shading | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/seaborn-introduction?ex=7)
# 
# > ## Rug plot and kde shading
# > 
# > Now that you understand some function arguments for `distplot()`, we can continue further refining the output. This process of creating a visualization and updating it in an incremental fashion is a useful and common approach to look at data from multiple perspectives.
# > 
# > Seaborn excels at making this process simple.

# > -   Create a `distplot` of the `Award_Amount` column in the `df`.
# > -   Configure it to show a shaded kde (using the `kde_kws` dictionary).
# > -   Add a rug plot above the x axis.
# > -   Display the plot.

# In[8]:


# Create a distplot of the Award Amount
sns.distplot(df['Award_Amount'],
             hist=False,
             rug=True,
             kde_kws={'shade':True})

# Plot the results
plt.show()


# # Regression Plots in Seaborn
# 
# ```python
# # Introduction to regplot
# sns.regplot(x="alcohol", y="pH", data=df)
# 
# # lmplot faceting
# sns.lmplot(x="quality", y="alcohol",data=df, hue="type") 
# sns.lmplot(x="quality", y="alcohol",data=df, col="type")
# ```

# [Create a regression plot | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/seaborn-introduction?ex=10)
# 
# > ## Create a regression plot
# > 
# > For this set of exercises, we will be looking at FiveThirtyEight's data on which US State has the worst drivers. The data set includes summary level information about fatal accidents as well as insurance premiums for each state as of 2010.
# > 
# > In this exercise, we will look at the difference between the regression plotting functions.

# ### init

# In[10]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/gzHHaHaVCeO7'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
df = pd.read_csv(prefix+'df.csv',index_col=0)


# ### code

# > -   The data is available in the dataframe called `df`.
# > -   Create a regression plot using `regplot()` with `"insurance_losses"` on the x axis and `"premiums"` on the y axis.

# In[12]:


# Create a regression plot of premiums vs. insurance_losses
sns.regplot(x='insurance_losses', y='premiums', data=df)



# Display the plot
plt.show()


# > -   Create a regression plot of `"premiums"` versus `"insurance_losses"` using `lmplot()`.
# > -   Display the plot.

# In[13]:


# Create an lmplot of premiums vs. insurance_losses
sns.lmplot(x='insurance_losses', y='premiums', data=df)


# Display the second plot
plt.show()


# [Plotting multiple variables | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/seaborn-introduction?ex=11)
# 
# > ## Plotting multiple variables
# > 
# > Since we are using `lmplot()` now, we can look at the more complex interactions of data. This data set includes geographic information by state and area. It might be interesting to see if there is a difference in relationships based on the `Region` of the country.

# > -   Use `lmplot()` to look at the relationship between `insurance_losses` and `premiums`.
# > -   Plot a regression line for each `Region` of the country.

# In[14]:


# Create a regression plot using hue
sns.lmplot(data=df,
           x="insurance_losses",
           y="premiums",
           hue="Region")

# Show the results
plt.show()


# [Facetting multiple regressions | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/seaborn-introduction?ex=12)
# 
# > ## Facetting multiple regressions
# > 
# > `lmplot()` allows us to facet the data across multiple rows and columns. In the previous plot, the multiple lines were difficult to read in one plot. We can try creating multiple plots by `Region` to see if that is a more useful visualization.

# > -   Use `lmplot()` to look at the relationship between `insurance_losses` and `premiums`.
# > -   Create a plot for each `Region` of the country.
# > -   Display the plots across multiple rows.

# In[16]:


# Create a regression plot with multiple rows
sns.lmplot(data=df,
           x="insurance_losses",
           y="premiums",
           row="Region")

# Show the plot
plt.show()


# In[ ]:




