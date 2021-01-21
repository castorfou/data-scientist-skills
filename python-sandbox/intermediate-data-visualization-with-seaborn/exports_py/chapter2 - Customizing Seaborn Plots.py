#!/usr/bin/env python
# coding: utf-8

# # Using Seaborn Styles
# 
# ```python
# # Setting Styles
# # Seaborn has default configurations that can be applied with sns.set()
# # These styles can override matplotlib and pandas plots as well
# sns.set()
# 
# # Theme examples with sns.set_style()
# for style in ['white','dark','whitegrid','darkgrid','ticks']:
#     sns.set_style(style)
#     sns.distplot(df['Tuition'])
#     plt.show()
# 
# # Removing axes with despine()
# sns.set_style('white')
# sns.distplot(df['Tuition'])
# sns.despine(left=True)
# ```

# [Setting the default style | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/customizing-seaborn-plots?ex=2)
# 
# > ## Setting the default style
# > 
# > For these exercises, we will be looking at fair market rent values calculated by the US Housing and Urban Development Department. This data is used to calculate guidelines for several federal programs. The actual values for rents vary greatly across the US. We can use this data to get some experience with configuring Seaborn plots.
# > 
# > All of the necessary imports for `seaborn`, `pandas` and `matplotlib` have been completed. The data is stored in the `pandas` DataFrame `df`.
# > 
# > _By the way, if you haven't downloaded it already, check out the [Seaborn Cheat Sheet](https://datacamp-community-prod.s3.amazonaws.com/f9f06e72-519a-4722-9912-b5de742dbac4). It includes an overview of the most important concepts, functions and methods and might come in handy if you ever need a quick refresher!_

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
{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/iZUFDU3bLpHq'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
df = pd.read_csv(prefix+'df.csv',index_col=0)


# ### code

# > -   Plot a `pandas` histogram without adjusting the style.
# > -   Set Seaborn's default style.
# > -   Create another `pandas` histogram of the `fmr_2` column which represents fair market rent for a 2-bedroom apartment.

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Plot the pandas histogram
df['fmr_2'].plot.hist()
plt.show()
plt.clf()

# Set the default seaborn style
sns.set()

# Plot the pandas histogram again
df['fmr_2'].plot.hist()
plt.show()
plt.clf()


# [Comparing styles | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/customizing-seaborn-plots?ex=3)
# 
# > ## Comparing styles
# > 
# > Seaborn supports setting different styles that can control the aesthetics of the final plot. In this exercise, you will plot the same data in two different styles in order to see how the styles change the output.

# > Create a `distplot()` of the `fmr_2` column in `df` using a `dark` style. Use `plt.clf()` to clear the figure.

# In[7]:


plt.clf()
sns.set_style('dark')
sns.distplot(df['fmr_2'])
plt.show()


# > Create the same `distplot()` of `fmr_2` using a `whitegrid` style. Clear the plot after showing it.

# In[8]:


plt.clf()
sns.set_style('whitegrid')
sns.distplot(df['fmr_2'])
plt.show()


# [Removing spines | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/customizing-seaborn-plots?ex=4)
# 
# > ## Removing spines
# > 
# > In general, visualizations should minimize extraneous markings so that the data speaks for itself. Seaborn allows you to remove the lines on the top, bottom, left and right axis, which are often called spines.

# > -   Use a `white` style for the plot.
# > -   Create a `lmplot()` comparing the `pop2010` and the `fmr_2` columns.
# > -   Remove the top and right spines using `despine()`.

# In[10]:


# Set the style to white
sns.set_style('white')

# Create a regression plot
sns.lmplot(data=df,
           x='pop2010',
           y='fmr_2')

# Remove the spines
sns.despine(left=True)

# Show the plot and clear the figure
plt.show()
plt.clf()


# # Colors in Seaborn
# 
# ```python
# # Defining a color for a plot
# sns.set(color_codes=True)
# sns.distplot(df['Tuition'], color='g')
# 
# # Palettes
# for p in sns.palettes.SEABORN_PALETTES:
#     sns.set_palette(p)
#     sns.distplot(df['Tuition'])
#     
# # Displaying Palettes
# for p in sns.palettes.SEABORN_PALETTES:
#     sns.set_palette(p)
#     sns.palplot(sns.color_palette())
#     plt.show()
#     
# # Defining Custom Palettes
# # Circular colors = when the data is not ordered 
# sns.palplot(sns.color_palette("Paired", 12))
# 
# # Sequential colors = when the data has a consistent range from high to low
# sns.palplot(sns.color_palette("Blues", 12))
# 
# # Diverging colors = when both the low and high values are interesting
# sns.palplot(sns.color_palette("BrBG", 12))
# 
# ```

# [Matplotlib color codes | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/customizing-seaborn-plots?ex=6)
# 
# >  ## Matplotlib color codes
# > 
# > Seaborn offers several options for modifying the colors of your visualizations. The simplest approach is to explicitly state the color of the plot. A quick way to change colors is to use the standard `matplotlib` color codes.

# > -   Set the default Seaborn style and enable the `matplotlib` color codes.
# > -   Create a `distplot` for the `fmr_3` column using `matplotlib`'s magenta (`m`) color code.

# In[11]:


# Set style, enable color code, and create a magenta distplot
sns.set(color_codes=True)
sns.distplot(df['fmr_3'], color='m')

# Show the plot
plt.show()


# [Using default palettes | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/customizing-seaborn-plots?ex=7)
# 
# > ## Using default palettes
# > 
# > Seaborn includes several default palettes that can be easily applied to your plots. In this example, we will look at the impact of two different palettes on the same `distplot`.

# > -   Create a `for` loop to show the difference between the `bright` and `colorblind` palette.
# > -   Set the palette using the `set_palette()` function.
# > -   Use a `distplot` of the `fmr_3` column.

# In[12]:


# Loop through differences between bright and colorblind palettes
for p in ['bright', 'colorblind']:
    sns.set_palette(p)
    sns.distplot(df['fmr_3'])
    plt.show()
    
    # Clear the plots    
    plt.clf()


# [Creating Custom Palettes | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/customizing-seaborn-plots?ex=9)
# 
# > ## Creating Custom Palettes
# > 
# > Choosing a cohesive palette that works for your data can be time consuming. Fortunately, Seaborn provides the `color_palette()` function to create your own custom sequential, categorical, or diverging palettes. Seaborn also makes it easy to view your palettes by using the `palplot()` function.
# > 
# > In this exercise, you can experiment with creating different palettes.

# > Create and display a `Purples` sequential palette containing 8 colors.

# In[13]:


sns.palplot(sns.color_palette("Purples", 8))


# > Create and display a palette with 10 colors using the `husl` system.

# In[14]:


sns.palplot(sns.color_palette("husl", 10))


# > Create and display a diverging palette with 6 colors `coolwarm`.

# In[15]:


sns.palplot(sns.color_palette("coolwarm", 6))


# # Customizing with matplotlib
# 
# ```python
# 
# # Matplotlib Axes
# fig, ax = plt.subplots()
# sns.distplot(df['Tuition'], ax=ax)
# ax.set(xlabel="Tuition 2013-14")
# 
# # Further Customizations
# fig, ax = plt.subplots()
# sns.distplot(df['Tuition'], ax=ax)
# ax.set(xlabel="Tuition 2013-14",ylabel="Distribution", xlim=(0, 50000),title="2013-14 Tuition and Fees Distribution")
# 
# # Combining Plots
# fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(7,4))
# sns.distplot(df['Tuition'], ax=ax0)
# sns.distplot(df.query('State == "MN"')['Tuition'], ax=ax1)
# ax1.set(xlabel="Tuition (MN)", xlim=(0, 70000))
# ax1.axvline(x=20000, label='My Budget', linestyle='--')
# ax1.legend()
# ```

# [Using matplotlib axes | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/customizing-seaborn-plots?ex=11)
# 
# > ## Using matplotlib axes
# > 
# > Seaborn uses `matplotlib` as the underlying library for creating plots. Most of the time, you can use the Seaborn API to modify your visualizations but sometimes it is helpful to use `matplotlib`'s functions to customize your plots. The most important object in this case is `matplotlib`'s `axes`.
# > 
# > Once you have an `axes` object, you can perform a lot of customization of your plot.
# > 
# > In these examples, the US HUD data is loaded in the dataframe `df` and all libraries are imported.

# ### init

# In[16]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/ENoLSIl6KeAV'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
df = pd.read_csv(prefix+'df.csv',index_col=0)


# ### code

# > -   Use `plt.subplots()` to create a axes and figure objects.
# > -   Plot a `distplot` of column `fmr_3` on the axes.
# > -   Set a more useful label on the x axis of "3 Bedroom Fair Market Rent".

# In[17]:


# Create a figure and axes
fig, ax = plt.subplots()

# Plot the distribution of data
sns.distplot(df['fmr_3'], ax=ax)

# Create a more descriptive x axis label
ax.set(xlabel="3 Bedroom Fair Market Rent")

# Show the plot
plt.show()


# [Additional plot customizations | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/customizing-seaborn-plots?ex=12)
# 
# > ## Additional plot customizations
# > 
# > The `matplotlib` API supports many common customizations such as labeling axes, adding titles, and setting limits. Let's complete another customization exercise.

# > -   Create a `distplot` of the `fmr_1` column.
# > -   Modify the x axis label to say "1 Bedroom Fair Market Rent".
# > -   Change the x axis limits to be between 100 and 1500.
# > -   Add a descriptive title of "US Rent" to the plot.

# In[18]:


# Create a figure and axes
fig, ax = plt.subplots()

# Plot the distribution of 1 bedroom rents
sns.distplot(df['fmr_1'], ax=ax)

# Modify the properties of the plot
ax.set(xlabel="1 Bedroom Fair Market Rent",
       xlim=(100,1500),
       title="US Rent")

# Display the plot
plt.show()


# [Adding annotations | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/customizing-seaborn-plots?ex=13)
# 
# > ## Adding annotations
# > 
# > Each of the enhancements we have covered can be combined together. In the next exercise, we can annotate our distribution plot to include lines that show the mean and median rent prices.
# > 
# > For this example, the palette has been changed to `bright` using `sns.set_palette()`

# > -   Create a figure and axes.
# > -   Plot the `fmr_1` column distribution.
# > -   Add a vertical line using `axvline` for the `median` and `mean` of the values which are already defined.

# In[20]:


median=634.0
mean=706.3254351016984


# In[22]:


# Create a figure and axes. Then plot the data
fig, ax = plt.subplots()
sns.distplot(df['fmr_1'], ax=ax)

# Customize the labels and limits
ax.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100,1500), title="US Rent")

# Add vertical lines for the median and mean
ax.axvline(x=median, color='m', label='Median', linestyle='--', linewidth=2)
ax.axvline(x=mean, color='b', label='Mean', linestyle='-', linewidth=2)

# Show the legend and plot the data
ax.legend()
plt.show()


# [Multiple plots | Python](https://campus.datacamp.com/courses/intermediate-data-visualization-with-seaborn/customizing-seaborn-plots?ex=14)
# 
# > ## Multiple plots
# > 
# > For the final exercise we will plot a comparison of the fair market rents for 1-bedroom and 2-bedroom apartments.

# > -   Create two axes objects, `ax0` and `ax1`.
# > -   Plot `fmr_1` on `ax0` and `fmr_2` on `ax1`.
# > -   Display the plots side by side.

# In[23]:


# Create a plot with 1 row and 2 columns that share the y axis label
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)

# Plot the distribution of 1 bedroom apartments on ax0
sns.distplot(df['fmr_1'], ax=ax0)
ax0.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100,1500))

# Plot the distribution of 2 bedroom apartments on ax1
sns.distplot(df['fmr_2'], ax=ax1)
ax1.set(xlabel="2 Bedroom Fair Market Rent", xlim=(100,1500))

# Display the plot
plt.show()


# In[ ]:




