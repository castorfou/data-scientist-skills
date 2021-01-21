#!/usr/bin/env python
# coding: utf-8

# # Categorical Plot Types
# 
# ```python
# # Plots of each observation - stripplot
# sns.stripplot(data=df, y="DRG Definition",
# x="Average Covered Charges",
# jitter=True)
# 
# # Plots of each observation - swarmplot
# sns.swarmplot(data=df, y="DRG Definition",
# x="Average Covered Charges")
# 
# # Abstract representations - boxplot
# sns.boxplot(data=df, y="DRG Definition",
# x="Average Covered Charges")
# 
# # Abstract representation - violinplot
# sns.violinplot(data=df, y="DRG Definition",
# x="Average Covered Charges")
# 
# # Abstract representation - lvplot
# sns.lvplot(data=df, y="DRG Definition",
# x="Average Covered Charges")
# 
# # Statistical estimates - barplot
# sns.barplot(data=df, y="DRG Definition",
# x="Average Covered Charges",
# hue="Region")
# 
# # Statistical estimates - pointplot
# sns.pointplot(data=df, y="DRG Definition",
# x="Average Covered Charges",
# hue="Region")
# 
# # Statistical estimates - countplot
# sns.countplot(data=df, y="DRG_Code", hue="Region")
# ```

# [stripplot() and swarmplot() | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/additional-plot-types?ex=2)
# 
# > ## stripplot() and swarmplot()
# > 
# > Many datasets have categorical data and Seaborn supports several useful plot types for this data. In this example, we will continue to look at the 2010 School Improvement data and segment the data by the types of school improvement models used.
# > 
# > As a refresher, here is the KDE distribution of the Award Amounts:
# > 
# > ![](https://assets.datacamp.com/production/repositories/2210/datasets/2d65e2dd7875735d1db7f6ff0faa1d4577db50d3/tuition_kde_plot.png)
# > 
# > While this plot is useful, there is a lot more we can learn by looking at the individual Award\_Amounts and how they are distributed among the 4 categories.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/Nxmfd3rD5F0O'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
df = pd.read_csv(prefix+'df.csv',index_col=0)


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### code

# > Create a `stripplot` of the `Award_Amount` with the `Model Selected` on the y axis with `jitter` enabled.

# In[3]:


# Create the stripplot
sns.stripplot(data=df,
         x='Award_Amount',
         y='Model Selected',
         jitter=True)

plt.show()


# > Create a `swarmplot()` of the same data, but also include the `hue` by `Region`.

# In[4]:


# Create and display a swarmplot with hue set to the Region
sns.swarmplot(data=df,
         x='Award_Amount',
         y='Model Selected',
         hue='Region')

plt.show()


# [boxplots, violinplots and lvplots | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/additional-plot-types?ex=3)
# 
# > ## boxplots, violinplots and lvplots
# > 
# > Seaborn's categorical plots also support several abstract representations of data. The API for each of these is the same so it is very convenient to try each plot and see if the data lends itself to one over the other.
# > 
# > In this exercise, we will use the color palette options presented in Chapter 2 to show how colors can easily be included in the plots.

# > -   Create and display a `boxplot` of the data with `Award_Amount` on the x axis and `Model Selected` on the y axis.

# In[5]:


# Create a boxplot
sns.boxplot(data=df,
         x='Award_Amount',
         y='Model Selected')

plt.show()
plt.clf()


# > Create and display a similar `violinplot` of the data, but use the `husl` palette for colors.

# In[7]:


# Create a violinplot with the husl palette
sns.violinplot(data=df,
         x='Award_Amount',
         y='Model Selected',
         palette='husl')

plt.show()
plt.clf()


# > Create and display an `lvplot` using the `Paired` palette and the `Region` column as the `hue`.

# In[10]:


# Create a lvplot with the Paired palette and the Region column as the hue
sns.boxenplot(data=df,
         x='Award_Amount',
         y='Model Selected',
         palette='Paired',
         hue='Region')

plt.show()
plt.clf()


# [barplots, pointplots and countplots | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/additional-plot-types?ex=4)
# 
# > ## barplots, pointplots and countplots
# > 
# > The final group of categorical plots are `barplots`, `pointplots` and `countplot` which create statistical summaries of the data. The plots follow a similar API as the other plots and allow further customization for the specific problem at hand.

# > Create a `countplot` with the `df` dataframe and `Model Selected` on the y axis and the color varying by `Region`.

# In[11]:


# Show a countplot with the number of models used with each region a different color
sns.countplot(data=df,
         y="Model Selected",
         hue="Region")

plt.show()
plt.clf()


# > -   Create a `pointplot` with the `df` dataframe and `Model Selected` on the x-axis and `Award_Amount` on the y-axis.
# > -   Use a `capsize` in the `pointplot` in order to add caps to the error bars.

# In[12]:


# Create a pointplot and include the capsize in order to show caps on the error bars
sns.pointplot(data=df,
         y='Award_Amount',
         x='Model Selected',
         capsize=.1)

plt.show()
plt.clf()


# > Create a `barplot` with the same data on the x and y axis and change the color of each bar based on the `Region` column.

# In[13]:


# Create a barplot with each Region shown as a different color
sns.barplot(data=df,
         y='Award_Amount',
         x='Model Selected',
         hue='Region')

plt.show()
plt.clf()


# # Regression Plots
# 
# ```python
# # Plotting with regplot()
# sns.regplot(data=df, x='temp',
# y='total_rentals', marker='+')
# 
# # Evaluating regression with residplot()
# sns.residplot(data=df, x='temp', y='total_rentals')
# 
# # Polynomial regression
# sns.regplot(data=df, x='temp',
# y='total_rentals', order=2)
# 
# # residplot with polynomial regression
# sns.residplot(data=df, x='temp',
# y='total_rentals', order=2)
# 
# # Categorical values
# sns.regplot(data=df, x='mnth', y='total_rentals',
# x_jitter=.1, order=2)
# 
# # Estimators
# sns.regplot(data=df, x='mnth', y='total_rentals',
# x_estimator=np.mean, order=2)
# 
# # Binning the data
# sns.regplot(data=df,x='temp',y='total_rentals',
# x_bins=4)
# ```

# [Regression and residual plots | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/additional-plot-types?ex=6)
# 
# > ## Regression and residual plots
# > 
# > Linear regression is a useful tool for understanding the relationship between numerical variables. Seaborn has simple but powerful tools for examining these relationships.
# > 
# > For these exercises, we will look at some details from the US Department of Education on 4 year college tuition information and see if there are any interesting insights into which variables might help predict tuition costs.
# > 
# > For these exercises, all data is loaded in the `df` variable.

# ### init

# In[14]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/H5TGsH0ox3nA'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
df = pd.read_csv(prefix+'df.csv',index_col=0)


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### code

# > -   Plot a regression plot comparing `Tuition` and average SAT scores(`SAT_AVG_ALL`).
# > -   Make sure the values are shown as green triangles.

# In[16]:


# Display a regression plot for Tuition
sns.regplot(data=df,
         y='Tuition',
         x="SAT_AVG_ALL",
         marker='^',
         color='g')

plt.show()
plt.clf()


# > Use a residual plot to determine if the relationship looks linear.

# In[17]:


# Display the residual plot
sns.residplot(data=df,
          y='Tuition',
          x="SAT_AVG_ALL",
          color='g')

plt.show()
plt.clf()


# [Regression plot parameters | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/additional-plot-types?ex=7)
# 
# > ## Regression plot parameters
# > 
# > Seaborn's regression plot supports several parameters that can be used to configure the plots and drive more insight into the data.
# > 
# > For the next exercise, we can look at the relationship between tuition and the percent of students that receive Pell grants. A Pell grant is based on student financial need and subsidized by the US Government. In this data set, each University has some percentage of students that receive these grants. Since this data is continuous, using `x_bins` can be useful to break the percentages into categories in order to summarize and understand the data.

# > Plot a regression plot of `Tuition` and `PCTPELL`.

# In[18]:


# Plot a regression plot of Tuition and the Percentage of Pell Grants
sns.regplot(data=df,
            y='Tuition',
            x="PCTPELL")

plt.show()
plt.clf()


# > Create another plot that breaks the `PCTPELL` column into 5 different bins.

# In[20]:


# Create another plot that estimates the tuition by PCTPELL
sns.regplot(data=df,
            y='Tuition',
            x="PCTPELL",
            x_bins=5)

plt.show()
plt.clf()


# > Create a final regression plot that includes a 2nd `order` polynomial regression line.

# In[21]:


# The final plot should include a line using a 2nd order polynomial
sns.regplot(data=df,
            y='Tuition',
            x="PCTPELL",
            x_bins=5,
            order=2)

plt.show()
plt.clf()


# [Binning data | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/additional-plot-types?ex=8)
# 
# > ## Binning data
# > 
# > When the data on the x axis is a continuous value, it can be useful to break it into different bins in order to get a better visualization of the changes in the data.
# > 
# > For this exercise, we will look at the relationship between tuition and the Undergraduate population abbreviated as `UG` in this data. We will start by looking at a scatter plot of the data and examining the impact of different bin sizes on the visualization.

# > Create a `regplot` of `Tuition` and `UG` and set the `fit_reg` parameter to `False` to disable the regression line.

# In[22]:


# Create a scatter plot by disabling the regression line
sns.regplot(data=df,
            y='Tuition',
            x="UG",
            fit_reg=False)

plt.show()
plt.clf()


# > Create another plot with the `UG` data divided into 5 bins.

# In[23]:


# Create a scatter plot and bin the data into 5 bins
sns.regplot(data=df,
            y='Tuition',
            x="UG",
            x_bins=5)

plt.show()
plt.clf()


# > Create a `regplot()` with the data divided into 8 bins.

# In[24]:


# Create a regplot and bin the data into 8 bins
sns.regplot(data=df,
         y='Tuition',
         x="UG",
         x_bins=8)

plt.show()
plt.clf()


# # Matrix plots
# 
# ```python
# 
# # Getting data in the right format
# pd.crosstab(df["mnth"], df["weekday"],
# values=df["total_rentals"],aggfunc='mean').round(0)
# 
# # Build a heatmap
# sns.heatmap(pd.crosstab(df["mnth"], df["weekday"],
# values=df["total_rentals"], aggfunc='mean')
# )
# 
# # Customize a heatmap
# sns.heatmap(df_crosstab, annot=True, fmt="d",
# cmap="YlGnBu", cbar=False, linewidths=.5)
# 
# # Centering a heatmap
# sns.heatmap(df_crosstab, annot=True, fmt="d",
# cmap="YlGnBu", cbar=True,
# center=df_crosstab.loc[9, 6])
# 
# # Plotting a correlation matrix
# sns.heatmap(df.corr())
# ```

# [Creating heatmaps | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/additional-plot-types?ex=10)
# 
# > ## Creating heatmaps
# > 
# > A heatmap is a common matrix plot that can be used to graphically summarize the relationship between two variables. For this exercise, we will start by looking at guests of the Daily Show from 1999 - 2015 and see how the occupations of the guests have changed over time.
# > 
# > The data includes the date of each guest appearance as well as their occupation. For the first exercise, we need to get the data into the right format for Seaborn's `heatmap` function to correctly plot the data. All of the data has already been read into the `df` variable.

# > -   Use pandas' `crosstab()` function to build a table of visits by `Group` and `Year`.
# > -   Print the `pd_crosstab` DataFrame.
# > -   Plot the data using Seaborn's `heatmap()`.

# ### init

# In[26]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/vuR0gvnspt2I'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
df = pd.read_csv(prefix+'df.csv',index_col=0)


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### code

# In[27]:


# Create a crosstab table of the data
pd_crosstab = pd.crosstab(df["Group"], df["YEAR"])
print(pd_crosstab)

# Plot a heatmap of the table
sns.heatmap(pd_crosstab)

# Rotate tick marks for visibility
plt.yticks(rotation=0)
plt.xticks(rotation=90)

plt.show()


# [Customizing heatmaps | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/additional-plot-types?ex=11)
# 
# > ## Customizing heatmaps
# > 
# > Seaborn supports several types of additional customizations to improve the output of a heatmap. For this exercise, we will continue to use the Daily Show data that is stored in the `df` variable but we will customize the output.

# > -   Create a crosstab table of `Group` and `YEAR`
# > -   Create a heatmap of the data using the `BuGn` palette
# > -   Disable the `cbar` and increase the `linewidth` to 0.3

# In[28]:


# Create the crosstab DataFrame
pd_crosstab = pd.crosstab(df["Group"], df["YEAR"])

# Plot a heatmap of the table with no color bar and using the BuGn palette
sns.heatmap(pd_crosstab, cbar=False, cmap="BuGn", linewidths=0.3)

# Rotate tick marks for visibility
plt.yticks(rotation=0)
plt.xticks(rotation=90)

#Show the plot
plt.show()
plt.clf()


# In[ ]:




