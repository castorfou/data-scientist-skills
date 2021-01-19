#!/usr/bin/env python
# coding: utf-8

# # Introduction to Seaborn
# 
# ```python
# # Getting started
# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# # Example 1: Scatter plot
# import seaborn as sns
# import matplotlib.pyplot as plt
# height = [62, 64, 69, 75, 66, 68, 65, 71, 76, 73]
# weight = [120, 136, 148, 175, 137, 165, 154, 172, 200, 187]
# sns.scatterplot(x=height, y=weight)
# plt.show()
# 
# # Example 2: Create a count plot
# import seaborn as sns
# import matplotlib.pyplot as plt
# gender = ["Female", "Female", "Female", "Female", "Male", "Male", "Male", "Male", "Male", "Male"]
# sns.countplot(x=gender)
# plt.show()
# 
# ```

# [Making a scatter plot with lists | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/introduction-to-seaborn?ex=2)
# 
# > ## Making a scatter plot with lists
# > 
# > In this exercise, we'll use a dataset that contains information about 227 countries. This dataset has lots of interesting information on each country, such as the country's birth rates, death rates, and its gross domestic product (GDP). GDP is the value of all the goods and services produced in a year, expressed as dollars per person.
# > 
# > We've created three lists of data from this dataset to get you started. `gdp` is a list that contains the value of GDP per country, expressed as dollars per person. `phones` is a list of the number of mobile phones per 1,000 people in that country. Finally, `percent_literate` is a list that contains the percent of each country's population that can read and write.

# ### init

# In[1]:


###################
##### liste de mots (list)
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(gdp, phones, percent_literate)
"""

tobedownloaded="""
{list: {'gdp.txt': 'https://file.io/dSr6dxIthAnM',
  'percent_literate.txt': 'https://file.io/o8lVIsvegqsq',
  'phones.txt': 'https://file.io/iQiF0hZgzTdg'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
from downloadfromFileIO import loadListFromTxt
gdp = loadListFromTxt(prefix+'gdp.txt')
percent_literate = loadListFromTxt(prefix+'percent_literate.txt')
phones = loadListFromTxt(prefix+'phones.txt')


# ### code

# > Import Matplotlib and Seaborn using the standard naming convention.

# In[2]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns


# > Create a scatter plot of GDP (`gdp`) vs. number of phones per 1000 people (`phones`).

# In[7]:


# Create scatter plot with GDP on the x-axis and number of phones on the y-axis
sns.scatterplot(x=gdp, y=phones)
# Show plot
plt.show()


# > Change the scatter plot so it displays the percent of the population that can read and write (`percent_literate`) on the y-axis.

# In[9]:


sns.scatterplot(x=gdp, y=percent_literate)
# Show plot
plt.show()


# [Making a count plot with a list | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/introduction-to-seaborn?ex=3)
# 
# > ## Making a count plot with a list
# > 
# > In the last exercise, we explored a dataset that contains information about 227 countries. Let's do more exploration of this data - specifically, how many countries are in each region of the world?
# > 
# > To do this, we'll need to use a count plot. Count plots take in a categorical list and return bars that represent the number of list entries per category. You can create one here using a list of regions for each country, which is a variable named `region`.

# ### init

# In[10]:


###################
##### liste de mots (list)
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(region)
"""

tobedownloaded="""
{list: {'region.txt': 'https://file.io/U8LZzdFbgaNL'}}
"""
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
from downloadfromFileIO import loadListFromTxt
region = loadListFromTxt(prefix+'region.txt')


# ### code

# > -   Import Matplotlib and Seaborn using the standard naming conventions.
# > -   Use Seaborn to create a count plot with `region` on the y-axis.
# > -   Display the plot.

# In[11]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create count plot with region on the y-axis
sns.countplot(y=region)

# Show plot
plt.show()


# # Using pandas with Seaborn
# 
# ```python
# 
# # Using DataFrames with countplot()
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# df = pd.read_csv("masculinity.csv")
# sns.countplot(x="how_masculine", data=df)
# plt.show()
# ```

# ["Tidy" vs. "untidy" data | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/introduction-to-seaborn?ex=5)
# 
# > ## "Tidy" vs. "untidy" data
# > 
# > Here, we have a sample dataset from a survey of children about their favorite animals. But can we use this dataset as-is with Seaborn? Let's use Pandas to import the csv file with the data collected from the survey and determine whether it is tidy, which is essential to having it work well with Seaborn.
# > 
# > To get you started, the filepath to the csv file has been assigned to the variable `csv_filepath`.
# > 
# > Note that because `csv_filepath` is a Python variable, you will not need to put quotation marks around it when you read the csv.

# ### init

# In[12]:


csv_filepath = 'https://assets.datacamp.com/production/repositories/3996/datasets/7ac19e11cf7ed61205ffe8da5208794b8e2a5086/1.2.1_example_csv.csv'


# ### code

# > -   Read the csv file located at `csv_filepath` into a DataFrame named `df`.
# > -   Print the head of `df` to show the first five rows.

# In[13]:


# Import Pandas
import pandas as pd

# Create a DataFrame from csv file
df = pd.read_csv(csv_filepath)

# Print the head of df
print(df.head())


# [Making a count plot with a DataFrame | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/introduction-to-seaborn?ex=6)
# 
# > ## Making a count plot with a DataFrame
# > 
# > In this exercise, we'll look at the responses to a survey sent out to young people. Our primary question here is: how many young people surveyed report being scared of spiders? Survey participants were asked to agree or disagree with the statement "I am afraid of spiders". Responses vary from 1 to 5, where 1 is "Strongly disagree" and 5 is "Strongly agree".
# > 
# > To get you started, the filepath to the csv file with the survey data has been assigned to the variable `csv_filepath`.
# > 
# > Note that because `csv_filepath` is a Python variable, you will not need to put quotation marks around it when you read the csv.

# In[14]:


csv_filepath = 'http://assets.datacamp.com/production/repositories/3996/datasets/ab13162732ae9ca1a9a27e2efd3da923ed6a4e7b/young-people-survey-responses.csv'


# > -   Import Matplotlib, Pandas, and Seaborn using the standard names.
# > -   Create a DataFrame named `df` from the csv file located at `csv_filepath`.
# > -   Use the `countplot()` function with the `x=` and `data=` arguments to create a count plot with the `"Spiders"` column values on the x-axis.
# > -   Display the plot.

# In[18]:


# Import Matplotlib, Pandas, and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a DataFrame from csv file
df=pd.read_csv(csv_filepath)

# Create a count plot with "Spiders" on the x-axis
sns.countplot(x='Spiders', data=df)

# Display the plot
plt.show()


# # Adding a third variable with hue
# 
# ```python
# # Tips dataset
# import pandas as pd
# import seaborn as sns
# tips = sns.load_dataset("tips")
# tips.head()
# 
# # A basic scatter plot
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.scatterplot(x="total_bill", y="tip", data=tips)
# plt.show()
# 
# # A scatter plot with hue
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.scatterplot(x="total_bill", y="tip", data=tips, hue="smoker")
# plt.show()
# 
# # Setting hue order
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.scatterplot(x="total_bill", y="tip", data=tips, hue="smoker", hue_order=["Yes","No"])
# plt.show()
# 
# # Specifying hue colors
# import matplotlib.pyplot as plt
# import seaborn as sns
# hue_colors = {"Yes": "black", "No": "red"}
# sns.scatterplot(x="total_bill", y="tip", data=tips, hue="smoker", palette=hue_colors)
# plt.show()
# 
# # Using HTML hex color codes with hue
# import matplotlib.pyplot as plt
# import seaborn as sns
# hue_colors = {"Yes": "#808080", "No": "#00FF00"}
# sns.scatterplot(x="total_bill", y="tip", data=tips, hue="smoker", palette=hue_colors)
# plt.show()
# 
# # Using hue with count plots
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.countplot(x="smoker", data=tips, hue="sex")
# plt.show()
# 
# 
# ```

# [Hue and scatter plots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/introduction-to-seaborn?ex=8)
# 
# > ## Hue and scatter plots
# > 
# > In the prior video, we learned how `hue` allows us to easily make subgroups within Seaborn plots. Let's try it out by exploring data from students in secondary school. We have a lot of information about each student like their age, where they live, their study habits and their extracurricular activities.
# > 
# > For now, we'll look at the relationship between the number of absences they have in school and their final grade in the course, segmented by where the student lives (rural vs. urban area).

# ### init

# In[19]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(student_data)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'student_data.csv': 'https://file.io/Iu8AjiYVealV'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
student_data = pd.read_csv(prefix+'student_data.csv',index_col=0)


# ### code

# > Create a scatter plot with `"absences"` on the x-axis and final grade (`"G3"`) on the y-axis using the DataFrame `student_data`. Color the plot points based on `"location"` (urban vs. rural).

# In[21]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create a scatter plot of absences vs. final grade
sns.scatterplot(x='absences', y='G3', data=student_data, hue='location')



# Show plot
plt.show()


# > Make `"Rural"` appear before `"Urban"` in the plot legend.

# In[22]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Change the legend order in the scatter plot
sns.scatterplot(x="absences", y="G3", 
                data=student_data, 
                hue="location", hue_order=['Rural', 'Urban'])

# Show plot
plt.show()


# [Hue and count plots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-seaborn/introduction-to-seaborn?ex=9)
# 
# > ## Hue and count plots
# > 
# > Let's continue exploring our dataset from students in secondary school by looking at a new variable. The `"school"` column indicates the initials of which school the student attended - either "GP" or "MS".
# > 
# > In the last exercise, we created a scatter plot where the plot points were colored based on whether the student lived in an urban or rural area. How many students live in urban vs. rural areas, and does this vary based on what school the student attends? Let's make a count plot with subgroups to find out.

# > -   Fill in the `palette_colors` dictionary to map the `"Rural"` location value to the color `"green"` and the `"Urban"` location value to the color `"blue"`.
# > -   Create a count plot with `"school"` on the x-axis using the `student_data` DataFrame.
# >     -   Add subgroups to the plot using `"location"` variable and use the `palette_colors` dictionary to make the location subgroups green and blue.

# In[23]:


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create a dictionary mapping subgroup values to colors
palette_colors = {'Rural': "green", 'Urban': "blue"}

# Create a count plot of school with location subgroups
sns.countplot(x='school', data=student_data, hue='location', palette=palette_colors)



# Display plot
plt.show()


# In[ ]:




