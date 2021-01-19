#!/usr/bin/env python
# coding: utf-8

# # Introduction to relational plots and subplots
# 
# ```python
# # Using relplot()
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.relplot(x="total_bill", y="tip", data=tips, kind="scatter")
# plt.show()
# 
# # Subplots in columns
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.relplot(x="total_bill", y="tip", data=tips, kind="scatter", col="smoker")
# plt.show()
# 
# # Subplots in rows
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.relplot(x="total_bill", y="tip", data=tips, kind="scatter", row="smoker")
# plt.show()
# 
# # Subplots in rows and columns
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.relplot(x="total_bill", y="tip", data=tips, kind="scatter", col="smoker", row="time")
# plt.show()
# 
# # Wrapping columns
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.relplot(x="total_bill",y="tip",data=tips,kind="scatter",col="day",col_wrap=2)
# plt.show()
# 
# # Ordering columns
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.relplot(x="total_bill",y="tip",data=tips,kind="scatter",col="day",col_wrap=2,col_order=["Thur","Fri","Sat","Sun"])
# plt.show()
# ```

# [Creating subplots with col and row | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-two-quantitative-variables?ex=2)
# 
# > ## Creating subplots with col and row
# > 
# > We've seen in prior exercises that students with more absences (`"absences"`) tend to have lower final grades (`"G3"`). Does this relationship hold regardless of how much time students study each week?
# > 
# > To answer this, we'll look at the relationship between the number of absences that a student has in school and their final grade in the course, creating separate subplots based on each student's weekly study time (`"study_time"`).
# > 
# > Seaborn has been imported as `sns` and `matplotlib.pyplot` has been imported as `plt`.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(student_data)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'student_data.csv': 'https://file.io/QNyK9Drh0yhA'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
student_data = pd.read_csv(prefix+'student_data.csv',index_col=0)


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### code

# > Modify the code to use `relplot()` instead of `scatterplot()`.

# In[3]:


# Change to use relplot() instead of scatterplot()
sns.relplot(x="absences", y="G3", 
                data=student_data, kind='scatter')

# Show plot
plt.show()


# > Modify the code to create one scatter plot for each level of the variable `"study_time"`, arranged in columns.

# In[4]:


# Change to make subplots based on study time
sns.relplot(x="absences", y="G3", 
            data=student_data,
            kind="scatter", col='study_time')

# Show plot
plt.show()


# > Adapt your code to create one scatter plot for each level of a student's weekly study time, this time arranged in rows.

# In[5]:


# Change this scatter plot to arrange the plots in rows instead of columns
sns.relplot(x="absences", y="G3", 
            data=student_data,
            kind="scatter", 
            row="study_time")

# Show plot
plt.show()


# [Creating two-factor subplots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-two-quantitative-variables?ex=3)
# 
# > ## Creating two-factor subplots
# > 
# > Let's continue looking at the `student_data` dataset of students in secondary school. Here, we want to answer the following question: does a student's first semester grade (`"G1"`) tend to correlate with their final grade (`"G3"`)?
# > 
# > There are many aspects of a student's life that could result in a higher or lower final grade in the class. For example, some students receive extra educational support from their school (`"schoolsup"`) or from their family (`"famsup"`), which could result in higher grades. Let's try to control for these two factors by creating subplots based on whether the student received extra educational support from their school or family.
# > 
# > Seaborn has been imported as `sns` and `matplotlib.pyplot` has been imported as `plt`.

# > Use `relplot()` to create a scatter plot with `"G1"` on the x-axis and `"G3"` on the y-axis, using the `student_data` DataFrame.

# In[7]:


# Create a scatter plot of G1 vs. G3
sns.relplot(x='G1', y='G3', data=student_data, kind='scatter')



# Show plot
plt.show()


# > Create **column** subplots based on whether the student received support from the school (`"schoolsup"`), ordered so that "yes" comes before "no".

# In[9]:


# Adjust to add subplots based on school support
sns.relplot(x="G1", y="G3", 
            data=student_data,
            kind="scatter", col='schoolsup', col_order=['yes', 'no'])

# Show plot
plt.show()


# > Add **row** subplots based on whether the student received support from the family (`"famsup"`), ordered so that "yes" comes before "no". This will result in subplots based on two factors.

# In[10]:


# Adjust further to add subplots based on family support
sns.relplot(x="G1", y="G3", 
            data=student_data,
            kind="scatter", 
            col="schoolsup",
            col_order=["yes", "no"],
           row='famsup', row_order=['yes', 'no'])

# Show plot
plt.show()


# # Customizing scatter plots
# 
# ```python
# 
# # Subgroups with point size
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.relplot(x="total_bill",y="tip",data=tips,kind="scatter",size="size")
# plt.show()
# 
# # Point size and hue
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.relplot(x="total_bill",y="tip",data=tips,kind="scatter",size="size",hue="size")
# plt.show()
# 
# # Subgroups with point style
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.relplot(x="total_bill",y="tip",data=tips,kind="scatter",hue="smoker",style="smoker")
# plt.show()
# 
# # Changing point transparency
# import seaborn as sns
# import matplotlib.pyplot as plt
# # Set alpha to be between 0 and 1
# sns.relplot(x="total_bill",y="tip",data=tips,kind="scatter",alpha=0.4)
# plt.show()
# ```

# [Changing the size of scatter plot points | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-two-quantitative-variables?ex=5)
# 
# > ## Changing the size of scatter plot points
# > 
# > In this exercise, we'll explore Seaborn's `mpg` dataset, which contains one row per car model and includes information such as the year the car was made, the number of miles per gallon ("M.P.G.") it achieves, the power of its engine (measured in "horsepower"), and its country of origin.
# > 
# > What is the relationship between the power of a car's engine (`"horsepower"`) and its fuel efficiency (`"mpg"`)? And how does this relationship vary by the number of cylinders (`"cylinders"`) the car has? Let's find out.
# > 
# > Let's continue to use `relplot()` instead of `scatterplot()` since it offers more flexibility.

# In[11]:


mpg = sns.load_dataset('mpg')


# > -   Use `relplot()` and the `mpg` DataFrame to create a scatter plot with `"horsepower"` on the x-axis and `"mpg"` on the y-axis. Vary the size of the points by the number of cylinders in the car (`"cylinders"`).

# In[13]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create scatter plot of horsepower vs. mpg
sns.relplot(x='horsepower', y='mpg', data=mpg, kind='scatter', size='cylinders')



# Show plot
plt.show()


# > To make this plot easier to read, use `hue` to vary the color of the points by the number of cylinders in the car (`"cylinders"`).

# In[14]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create scatter plot of horsepower vs. mpg
sns.relplot(x="horsepower", y="mpg", 
            data=mpg, kind="scatter", 
            size="cylinders", hue='cylinders')

# Show plot
plt.show()


# [Changing the style of scatter plot points | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-two-quantitative-variables?ex=6)
# 
# > ## Changing the style of scatter plot points
# > 
# > Let's continue exploring Seaborn's `mpg` dataset by looking at the relationship between how fast a car can accelerate (`"acceleration"`) and its fuel efficiency (`"mpg"`). Do these properties vary by country of origin (`"origin"`)?
# > 
# > Note that the `"acceleration"` variable is the time to accelerate from 0 to 60 miles per hour, in seconds. Higher values indicate slower acceleration.

# > Use `relplot()` and the `mpg` DataFrame to create a scatter plot with `"acceleration"` on the x-axis and `"mpg"` on the y-axis. Vary the style and color of the plot points by country of origin (`"origin"`).

# In[17]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create a scatter plot of acceleration vs. mpg
sns.relplot(x='acceleration', y='mpg', data=mpg, kind='scatter', style='origin', hue='origin')



# Show plot
plt.show()


# # Introduction to line plots
# 
# ```python
# # Line plot
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.relplot(x="hour", y="NO_2_mean",data=air_df_mean,kind="line")
# plt.show()
# 
# # Subgroups by location
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.relplot(x="hour", y="NO_2_mean",data=air_df_loc_mean,kind="line",style="location",hue="location")
# plt.show()
# 
# # Adding markers
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.relplot(x="hour", y="NO_2_mean",data=air_df_loc_mean,kind="line",style="location",hue="location",markers=True)
# plt.show()
# 
# # Turning off line style
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.relplot(x="hour", y="NO_2_mean",data=air_df_loc_mean,kind="line",style="location",hue="location",markers=True,dashes=False)
# plt.show()
# 
# # Multiple observations per x-value
# # Line plot
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.relplot(x="hour", y="NO_2",data=air_df,kind="line")
# plt.show()
# 
# # Replacing confidence interval with standard deviation
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.relplot(x="hour", y="NO_2",data=air_df,kind="line",ci="sd")
# plt.show()
# 
# # Turning off confidence interval
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.relplot(x="hour", y="NO_2",data=air_df,kind="line",ci=None)
# plt.show()
# 
# 
# ```

# [Interpreting line plots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-two-quantitative-variables?ex=8)
# 
# > ## Interpreting line plots
# > 
# > In this exercise, we'll continue to explore Seaborn's `mpg` dataset, which contains one row per car model and includes information such as the year the car was made, its fuel efficiency (measured in "miles per gallon" or "M.P.G"), and its country of origin (USA, Europe, or Japan).
# > 
# > How has the average miles per gallon achieved by these cars changed over time? Let's use line plots to find out!

# > Use `relplot()` and the `mpg` DataFrame to create a line plot with `"model_year"` on the x-axis and `"mpg"` on the y-axis.

# In[18]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create line plot
sns.relplot(x='model_year', y='mpg', data=mpg, kind='line')


# Show plot
plt.show()


# [Visualizing standard deviation with line plots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-two-quantitative-variables?ex=9)
# 
# > ## Visualizing standard deviation with line plots
# > 
# > In the last exercise, we looked at how the average miles per gallon achieved by cars has changed over time. Now let's use a line plot to visualize how the _distribution_ of miles per gallon has changed over time.
# > 
# > Seaborn has been imported as `sns` and `matplotlib.pyplot` has been imported as `plt`.

# > Change the plot so the shaded area shows the standard deviation instead of the confidence interval for the mean.

# In[19]:


# Make the shaded area show the standard deviation
sns.relplot(x="model_year", y="mpg",
            data=mpg, kind="line", ci='sd')

# Show plot
plt.show()


# [Plotting subgroups in line plots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-two-quantitative-variables?ex=10)
# 
# > ## Plotting subgroups in line plots
# > 
# > Let's continue to look at the `mpg` dataset. We've seen that the average miles per gallon for cars has increased over time, but how has the average horsepower for cars changed over time? And does this trend differ by country of origin?

# > Use `relplot()` and the `mpg` DataFrame to create a line plot with `"model_year"` on the x-axis and `"horsepower"` on the y-axis. Turn off the confidence intervals on the plot.

# In[21]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create line plot of model year vs. horsepower
sns.relplot(x='model_year', y='horsepower', data=mpg, kind='line', ci=None)



# Show plot
plt.show()


# > -   Create different lines for each country of origin (`"origin"`) that vary in both line style and color.

# In[24]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Change to create subgroups for country of origin
sns.relplot(x="model_year", y="horsepower", 
            data=mpg, kind="line", 
            ci=None, style='origin', hue='origin')

# Show plot
plt.show()


# > -   Add markers for each data point to the lines.
# > -   Use the `dashes` parameter to use solid lines for all countries, while still allowing for different marker styles for each line.

# In[26]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Add markers and make each line have the same style
sns.relplot(x="model_year", y="horsepower", 
            data=mpg, kind="line", 
            ci=None, style="origin", 
            hue="origin", dashes=False, markers=True)

# Show plot
plt.show()


# 
