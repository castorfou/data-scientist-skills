#!/usr/bin/env python
# coding: utf-8

# # Count plots and bar plots
# 
# ```python
# # countplot() vs. catplot()
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.catplot(x="how_masculine",data=masculinity_data,kind="count")
# plt.show()
# 
# # Changing the order
# import matplotlib.pyplot as plt
# import seaborn as sns
# category_order = ["No answer","Not at all","Not very","Somewhat","Very"]
# sns.catplot(x="how_masculine",data=masculinity_data,kind="count",order=category_order)
# plt.show()
# 
# # Bar plots
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.catplot(x="day",y="total_bill",data=tips,kind="bar")
# plt.show()
# 
# # Turning off confidence intervals
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.catplot(x="day",y="total_bill",data=tips,kind="bar",ci=None)
# plt.show()
# 
# # Changing the orientation
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.catplot(x="total_bill",y="day",data=tips,kind="bar")
# plt.show()
# ```

# [Count plots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-a-categorical-and-a-quantitative-variable?ex=2)
# 
# > ## Count plots
# > 
# > In this exercise, we'll return to exploring our dataset that contains the responses to a survey sent out to young people. We might suspect that young people spend a lot of time on the internet, but how much do they report using the internet each day? Let's use a count plot to break down the number of survey responses in each category and then explore whether it changes based on age.
# > 
# > As a reminder, to create a count plot, we'll use the `catplot()` function and specify the name of the categorical variable to count (`x=____`), the Pandas DataFrame to use (`data=____`), and the type of plot (`kind="count"`).
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
uploadToFileIO(survey_data)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'survey_data.csv': 'https://file.io/nDmTzJiXCJrb'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
survey_data = pd.read_csv(prefix+'survey_data.csv',index_col=0)


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### code

# > Use `sns.catplot()` to create a count plot using the `survey_data` DataFrame with `"Internet usage"` on the x-axis.

# In[6]:


# Create count plot of internet usage
sns.catplot(x='Internet usage', data=survey_data, kind='count')


# Show plot
plt.show()


# > Make the bars horizontal instead of vertical.

# In[7]:


# Change the orientation of the plot
sns.catplot(y="Internet usage", data=survey_data,
            kind="count")

# Show plot
plt.show()


# > Create column subplots based on `"Age Category"`, which separates respondents into those that are younger than 21 vs. 21 and older.

# In[8]:


# Create column subplots based on age category
sns.catplot(y="Internet usage", data=survey_data,
            kind="count", col='Age Category')

# Show plot
plt.show()


# [Bar plots with percentages | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-a-categorical-and-a-quantitative-variable?ex=3)
# 
# > ## Bar plots with percentages
# > 
# > Let's continue exploring the responses to a survey sent out to young people. The variable `"Interested in Math"` is `True` if the person reported being interested or very interested in mathematics, and `False` otherwise. What percentage of young people report being interested in math, and does this vary based on gender? Let's use a bar plot to find out.
# > 
# > As a reminder, we'll create a bar plot using the `catplot()` function, providing the name of categorical variable to put on the x-axis (`x=____`), the name of the quantitative variable to summarize on the y-axis (`y=____`), the Pandas DataFrame to use (`data=____`), and the type of categorical plot (`kind="bar"`).
# > 
# > Seaborn has been imported as `sns` and `matplotlib.pyplot` has been imported as `plt`.

# > Use the `survey_data` DataFrame and `sns.catplot()` to create a bar plot with `"Gender"` on the x-axis and `"Interested in Math"` on the y-axis.

# ### init

# In[12]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(survey_data)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'survey_data.csv': 'https://file.io/UYhAQAlhIpb5'}}
"""
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
survey_data = pd.read_csv(prefix+'survey_data.csv',index_col=0)


# ### code

# In[13]:


# Create a bar plot of interest in math, separated by gender
sns.catplot(x='Gender', y='Interested in Math', data=survey_data, kind='bar')


# Show plot
plt.show()


# [Customizing bar plots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-a-categorical-and-a-quantitative-variable?ex=4)
# 
# > ## Customizing bar plots
# > 
# > In this exercise, we'll explore data from students in secondary school. The `"study_time"` variable records each student's reported weekly study time as one of the following categories: `"<2 hours"`, `"2 to 5 hours"`, `"5 to 10 hours"`, or `">10 hours"`. Do students who report higher amounts of studying tend to get better final grades? Let's compare the average final grade among students in each category using a bar plot.
# > 
# > Seaborn has been imported as `sns` and `matplotlib.pyplot` has been imported as `plt`.

# > Use `sns.catplot()` to create a bar plot with `"study_time"` on the x-axis and final grade (`"G3"`) on the y-axis, using the `student_data` DataFrame.

# ### init

# In[14]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(student_data)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'student_data.csv': 'https://file.io/mIpoKJn7nXmT'}}
"""
prefixToc='1.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
student_data = pd.read_csv(prefix+'student_data.csv',index_col=0)


# ### code

# In[15]:


# Create bar plot of average final grade in each study category
sns.catplot(x='study_time', y='G3', data=student_data, kind='bar')



# Show plot
plt.show()


# > Using the `order` parameter, rearrange the categories so that they are in order from lowest study time to highest.

# In[16]:


# Rearrange the categories
sns.catplot(x="study_time", y="G3",
            data=student_data,
            kind="bar", order=['<2 hours', '2 to 5 hours', '5 to 10 hours', '>10 hours'])

# Show plot
plt.show()


# > Update the plot so that it no longer displays confidence intervals.

# In[18]:


# Turn off the confidence intervals
sns.catplot(x="study_time", y="G3",
            data=student_data,
            kind="bar",
            order=["<2 hours", 
                   "2 to 5 hours", 
                   "5 to 10 hours", 
                   ">10 hours"], ci=None)

# Show plot
plt.show()


# # Box plots
# 
# ```python
# # How to create a box plot
# import matplotlib.pyplot as plt
# import seaborn as sns
# g = sns.catplot(x="time",y="total_bill",data=tips,kind="box")
# plt.show()
# 
# # Change the order of categories
# import matplotlib.pyplot as plt
# import seaborn as sns
# g = sns.catplot(x="time",y="total_bill",data=tips,kind="box",order=["Dinner","Lunch"])
# plt.show()
# 
# # Omitting the outliers using `sym`
# import matplotlib.pyplot as plt
# import seaborn as sns
# g = sns.catplot(x="time",y="total_bill",data=tips,kind="box",sym="")
# plt.show()
# 
# # Changing the whiskers using `whis`
# import matplotlib.pyplot as plt
# import seaborn as sns
# g = sns.catplot(x="time",y="total_bill",data=tips,kind="box",whis=[0, 100])
# plt.show()
# ```

# [Create and interpret a box plot | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-a-categorical-and-a-quantitative-variable?ex=6)
# 
# > ## Create and interpret a box plot
# > 
# > Let's continue using the `student_data` dataset. In an earlier exercise, we explored the relationship between studying and final grade by using a bar plot to compare the average final grade (`"G3"`) among students in different categories of `"study_time"`.
# > 
# > In this exercise, we'll try using a box plot look at this relationship instead. As a reminder, to create a box plot you'll need to use the `catplot()` function and specify the name of the categorical variable to put on the x-axis (`x=____`), the name of the quantitative variable to summarize on the y-axis (`y=____`), the Pandas DataFrame to use (`data=____`), and the type of plot (`kind="box"`).
# > 
# > We have already imported `matplotlib.pyplot` as `plt` and `seaborn` as `sns`.

# > Use `sns.catplot()` and the `student_data` DataFrame to create a box plot with `"study_time"` on the x-axis and `"G3"` on the y-axis. Set the ordering of the categories to `study_time_order`.

# In[20]:


# Specify the category ordering
study_time_order = ["<2 hours", "2 to 5 hours", 
                    "5 to 10 hours", ">10 hours"]

# Create a box plot and set the order of the categories

sns.catplot(x='study_time', y='G3', data=student_data, kind='box', order=study_time_order)



# Show plot
plt.show()


# [Omitting outliers | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-a-categorical-and-a-quantitative-variable?ex=7)
# 
# > ## Omitting outliers
# > 
# > Now let's use the `student_data` dataset to compare the distribution of final grades (`"G3"`) between students who have internet access at home and those who don't. To do this, we'll use the `"internet"` variable, which is a binary (yes/no) indicator of whether the student has internet access at home.
# > 
# > Since internet may be less accessible in rural areas, we'll add subgroups based on where the student lives. For this, we can use the `"location"` variable, which is an indicator of whether a student lives in an urban ("Urban") or rural ("Rural") location.
# > 
# > Seaborn has already been imported as `sns` and `matplotlib.pyplot` has been imported as `plt`. As a reminder, you can omit outliers in box plots by setting the `sym` parameter equal to an empty string (`""`).

# > -   Use `sns.catplot()` to create a box plot with the `student_data` DataFrame, putting `"internet"` on the x-axis and `"G3"` on the y-axis.
# > -   Add subgroups so each box plot is colored based on `"location"`.
# > -   Do not display the outliers.

# In[24]:


# Create a box plot with subgroups and omit the outliers

sns.catplot(x='internet', y='G3', data=student_data, kind='box', col='location', hue='location', sym='')




# Show plot
plt.show()


# [Adjusting the whiskers | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-a-categorical-and-a-quantitative-variable?ex=8)
# 
# > ## Adjusting the whiskers
# > 
# > In the lesson we saw that there are multiple ways to define the whiskers in a box plot. In this set of exercises, we'll continue to use the `student_data` dataset to compare the distribution of final grades (`"G3"`) between students who are in a romantic relationship and those that are not. We'll use the `"romantic"` variable, which is a yes/no indicator of whether the student is in a romantic relationship.
# > 
# > Let's create a box plot to look at this relationship and try different ways to define the whiskers.
# > 
# > We've already imported Seaborn as `sns` and `matplotlib.pyplot` as `plt`.

# > Adjust the code to make the box plot whiskers to extend to 0.5 \* IQR. Recall: the IQR is the interquartile range.

# In[25]:


# Set the whiskers to 0.5 * IQR
sns.catplot(x="romantic", y="G3",
            data=student_data,
            kind="box", whis=0.5)

# Show plot
plt.show()


# > Change the code to set the whiskers to extend to the 5th and 95th percentiles.

# In[26]:


# Extend the whiskers to the 5th and 95th percentile
sns.catplot(x="romantic", y="G3",
            data=student_data,
            kind="box",
            whis=[5,95])

# Show plot
plt.show()


# > Change the code to set the whiskers to extend to the min and max values.

# In[27]:


# Set the whiskers at the min and max values
sns.catplot(x="romantic", y="G3",
            data=student_data,
            kind="box",
            whis=[0, 100])

# Show plot
plt.show()


# # Point plots
# 
# ```python
# # Creating a point plot
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.catplot(x="age",y="masculinity_important",data=masculinity_data,hue="feel_masculine",kind="point")
# plt.show()
# 
# # Disconnecting the points
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.catplot(x="age",y="masculinity_important",data=masculinity_data,hue="feel_masculine",kind="point",join=False)
# plt.show()
# 
# # Displaying the median
# import matplotlib.pyplot as plt
# import seaborn as sns
# from numpy import median
# sns.catplot(x="smoker",y="total_bill",data=tips,kind="point",estimator=median)
# plt.show()
# 
# # Customizing the confidence intervals
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.catplot(x="smoker",y="total_bill",data=tips,kind="point",capsize=0.2)
# plt.show()
# 
# # Turning off confidence intervals
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.catplot(x="smoker",y="total_bill",data=tips,kind="point",ci=None)
# plt.show()
# ```

# [Customizing point plots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-a-categorical-and-a-quantitative-variable?ex=10)
# 
# > ## Customizing point plots
# > 
# > Let's continue to look at data from students in secondary school, this time using a point plot to answer the question: does the quality of the student's family relationship influence the number of absences the student has in school? Here, we'll use the `"famrel"` variable, which describes the quality of a student's family relationship from 1 (very bad) to 5 (very good).
# > 
# > As a reminder, to create a point plot, use the `catplot()` function and specify the name of the categorical variable to put on the x-axis (`x=____`), the name of the quantitative variable to summarize on the y-axis (`y=____`), the Pandas DataFrame to use (`data=____`), and the type of categorical plot (`kind="point"`).
# > 
# > We've already imported Seaborn as `sns` and `matplotlib.pyplot` as `plt`.

# > Use `sns.catplot()` and the `student_data` DataFrame to create a point plot with `"famrel"` on the x-axis and number of absences (`"absences"`) on the y-axis.

# In[28]:


# Create a point plot of family relationship vs. absences

sns.catplot(x='famrel', y='absences', data=student_data, kind='point')

            
# Show plot
plt.show()


# > Add "caps" to the end of the confidence intervals with size `0.2`.

# In[30]:


# Add caps to the confidence interval
sns.catplot(x="famrel", y="absences",
			data=student_data,
            kind="point", capsize=0.2)
        
# Show plot
plt.show()


# > Remove the lines joining the points in each category.

# In[31]:


# Remove the lines joining the points
sns.catplot(x="famrel", y="absences",
			data=student_data,
            kind="point",
            capsize=0.2, join=False)
            
# Show plot
plt.show()


# [Point plots with subgroups | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/visualizing-a-categorical-and-a-quantitative-variable?ex=11)
# 
# > ## Point plots with subgroups
# > 
# > Let's continue exploring the dataset of students in secondary school. This time, we'll ask the question: is being in a romantic relationship associated with higher or lower school attendance? And does this association differ by which school the students attend? Let's find out using a point plot.
# > 
# > We've already imported Seaborn as `sns` and `matplotlib.pyplot` as `plt`.

# > Use `sns.catplot()` and the `student_data` DataFrame to create a point plot with relationship status (`"romantic"`) on the x-axis and number of absences (`"absences"`) on the y-axis. Create subgroups based on the school that they attend (`"school"`)

# In[34]:


# Create a point plot with subgroups

sns.catplot(x='romantic', y='absences', data=student_data, kind='point', hue='school')


# Show plot
plt.show()


# > Turn off the confidence intervals for the plot.

# In[35]:


# Turn off the confidence intervals for this plot
sns.catplot(x="romantic", y="absences",
			data=student_data,
            kind="point",
            hue="school", ci=None)

# Show plot
plt.show()


# > Since there may be outliers of students with many absences, import the `median` function from `numpy` and display the median number of absences instead of the average.

# In[36]:


# Import median function from numpy
from numpy import median

# Plot the median number of absences instead of the mean
sns.catplot(x="romantic", y="absences",
			data=student_data,
            kind="point",
            hue="school",
            ci=None, estimator=median)

# Show plot
plt.show()


# 
