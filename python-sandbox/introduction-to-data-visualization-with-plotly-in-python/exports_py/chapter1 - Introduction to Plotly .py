#!/usr/bin/env python
# coding: utf-8

# # Plotly and the Plotly Figure

# In[3]:


#creating our figure

import plotly.graph_objects as go

figure_config = dict({
    "data": [{
        "type":
        "bar",
        "x": [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
            "Sunday"
        ],
        "y": [28, 27, 25, 31, 32, 35, 36]
    }],
    "layout": {
        "title": {
            "text": "Temperatures of the week",
            "x": 0.5,
            "font": {
                'color': 'red',
                'size': 15
            }
        }
    }
})
fig = go.Figure(figure_config)
fig.show()


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ## Fixing a Plotly figure
# > 
# > Your colleague had started a project to create a visualization of sales for the first three months of 2020. However, she then left for some annual leave - but the boss wants the visualization now!
# > 
# > You can see she left behind a dictionary object that has started to define the visualization. It is your task to finish this dictionary with the important key arguments so it can be turned into a Plotly visualization.
# > 
# > In the exercises where it is needed throughout the course, `plotly.graph_objects` has already been loaded as `go`.
# > 
# > There is a `monthly_sales` dictionary that has been partially complete also available.

# ### code
# 
# [Fixing a Plotly figure | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/introduction-to-plotly-1?ex=3)
# 
# > -   Examine the `monthly_sales` dictionary that has been printed in the console to determine what is missing.
# > -   Update the `type` inside the `data` element of `monthly_sales` to be a `bar` chart.
# > -   Update the `text` element of the `title` of the `layout` element to be `'Sales for Jan-Mar 2020'`.
# > -   Create a figure using `go.Figure()` and the `monthly_sales` dictionary.

# In[4]:


monthly_sales = {'data': [{'type': '', 'x': ['Jan', 'Feb', 'March'], 'y': [450, 475, 400]}],
 'layout': {'title': {'text': ''}}}


# In[5]:


# Examine the printed dictionary

# Update the type
monthly_sales['data'][0]['type'] = 'bar'

# Update the title text
monthly_sales['layout']['title']['text'] = 'Sales for Jan-Mar 2020'

# Create a figure
fig = go.Figure(monthly_sales)

# Print it out!
fig.show()


# # Univariate visualizations

# In[8]:


#Bar charts with plotly.express

import plotly.express as px
import pandas as pd

weekly_temps = pd.DataFrame({
    'day': [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
        'Sunday'
    ],
    'temp': [28, 27, 25, 31, 32, 35, 36]
})
fig = px.bar(data_frame=weekly_temps, x='day', y='temp')
fig.show()


# ![image.png](attachment:image.png)

# ```python
# # histogram
# fig = px.histogram(data_frame=penguins, 
#                    x='Body Mass (g)', 
#                    nbins=10)
# fig.show()
# ```

# ![image.png](attachment:image.png)

# ```python
# # boxplot
# fig = px.box(data_frame=penguins,
#              y="Flipper Length (mm)")
# fig.show()
# ```
# ![image.png](attachment:image.png)

# ## Student scores bar graph
# > 
# > The school board has asked you to come and look at some test scores. They want an easy way to visualize the score of different students within a small class. This seems like a simple use case to practice your bar chart skills!
# > 
# > In this exercise, you will help the school board team by creating a bar chart of school test score values.
# > 
# > A DataFrame `student_scores` has been provided. In this and all future exercises where it is needed, `plotly.express` has already been loaded as `px`.

# ### init

# In[9]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(student_scores)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'student_scores.csv': 'https://file.io/KBuOlHzZ098O'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
student_scores = pd.read_csv(prefix+'student_scores.csv',index_col=0)


# In[11]:


import plotly.express as px
import pandas as pd


# ### code

# [Student scores bar graph | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/introduction-to-plotly-1?ex=5)
# 
# > -   Examine the head of the provided `student_scores` DataFrame that has been printed for you to see what it contains.
# > -   Create a simple bar plot, setting the y-axis to be the score and the x-axis to be the student name.
# > -   Add a title to the plot: call it `'Student Scores by Student'`.

# In[10]:


student_scores


# In[12]:


# Create the box plot
fig = px.bar(data_frame=student_scores, 
             x='student_name', 
             y='score', 
             title='Student Scores by Student')

# Show the plot
fig.show()


# ## Box plot of company revenues
# > 
# > You have been contracted by a New York Stock exchange firm who are interested in upping their data visualization capabilities.
# > 
# > They are cautious about this new technology so have tasked you with something simple first. To display the distribution of the revenues of top companies in the USA. They are particularly interested in what kind of revenue puts you in the 'top bracket' of companies.
# > 
# > They also want to know if there are any outliers and how they can explore this in the plot. This sounds like a perfect opportunity for a box plot.
# > 
# > In this exercise, you will help the investment team by creating a box plot of the revenue of top US companies.
# > 
# > There is a `revenues` DataFrame already loaded for your use.

# ### init

# In[13]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(revenues)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'revenues.csv': 'https://file.io/L0c55YORrU53'}}
"""
prefixToc='2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
revenues = pd.read_csv(prefix+'revenues.csv',index_col=0)


# In[11]:


import plotly.express as px
import pandas as pd


# ### code

# [Box plot of company revenues | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/introduction-to-plotly-1?ex=6)
# 
# > -   Examine the head of the provided `revenues` DataFrame that has been printed for you to see what it contains.
# > -   Create a simple box plot, setting the appropriate y-axis for company revenue data.
# > -   Set the `hover_data` (a list of one string value) to show the company name.

# In[15]:


revenues.head()


# In[17]:


# Create the box plot
fig = px.box(
  			# Set the data
  			data_frame=revenues, 
  			# Set the y variable
            y='Revenue', 
            # Add in hover data to see outliers
            hover_data=['Company'])

# Show the plot
fig.show()


# ## Histogram of company revenues
# > 
# > The New York Stock exchange firm loved your previous box plot and want you to do more work for them.
# > 
# > The box plot was a perfect visualization to help them understand the outliers and quartile-related attributes of their company revenue dataset.
# > 
# > However, they want to understand a bit more about the distribution of the data. Are there many companies with smaller revenue, or larger revenue? Is it somewhat bell-shaped or skewed towards higher or lower revenues?
# > 
# > In this exercise, you will help the investment team by creating a histogram of the revenue of top US companies.
# > 
# > There is a `revenues` DataFrame already loaded for your use.

# ### code

# [Histogram of company revenues | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/introduction-to-plotly-1?ex=7)
# 
# > -   Create a histogram with the appropriate DataFrame using `px.histogram()`.
# > -   Set the column `Revenue` to be used along the x-axis.
# > -   Set the number of bins to be 5.

# In[20]:


# Create a simple histogram
fig = px.histogram(
  			data_frame=revenues, 
            # Set up the x-axis
           	x='Revenue',
            # Set the number of bins
            nbins=5)

# Show the plot
fig.show()


#  # Customizing color

# In[23]:


student_scores['city']=['Melbourne', 'Melbourne', 'Sydney', 'Sydney']


# In[27]:


# Our specific colors

fig = px.bar(data_frame=student_scores,
             x='student_name',
             y='score',
             title="Student Scores by Student",
             color_discrete_map={
                 'Melbourne': 'rgb(0,0,128)',
                 'Sydney': 'rgb(235, 207, 52)'
             },
             color='city')
fig.show()


# ![image.png](attachment:image.png)

# In[30]:


# Using built-in color scales

fig = px.bar(data_frame=weekly_temps,
             x='day',
             y='temp',
             color='temp',
             color_continuous_scale='inferno')
fig.show()


# ![image.png](attachment:image.png)

# Many builtin scales available: https://plotly.com/python/builtin-colorscales/

# In[33]:


# Constructing our own color range

my_scale = [('rgb(242, 238, 10)'), ('rgb(242, 95, 10)'), ('rgb(255,0,0)')]
fig = px.bar(data_frame=weekly_temps,
             x='day',
             y='temp',
             color_continuous_scale=my_scale,
             color='temp')
fig.show()


# ![image.png](attachment:image.png)

# ## Coloring student scores bar graph
# > 
# > The previous plot that you created was well received by the school board, but they are wondering if there is a way for you to visually identify good and bad performers.
# > 
# > This would be a great opportunity to utilize color. Specifically, a color scale. You think a scale from red (worst marks) to green (good marks) would be great.
# > 
# > Part of your previous code to create the student scores bar chart has been provided.
# > 
# > The `student_scores` DataFrame is also available. Feel free to print out in the console and inspect it.

# ### init

# In[37]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(student_scores)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'student_scores.csv': 'https://file.io/GWcTyBYAWens'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
student_scores = pd.read_csv(prefix+'student_scores.csv',index_col=0)


# ### code

# [Coloring student scores bar graph | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/introduction-to-plotly-1?ex=10)
# 
# > -   Create a color scale list from red (RGB code `255, 0, 0`) to a nice green (RGB code `3, 252, 40`).
# > -   Create a `plotly.express` bar chart using the `student_scores` DataFrame.
# > -   Set the `color` argument to the name of the appropriate numerical column.
# > -   Use `my_scale` as the color scale for the plot.

# In[39]:


# Create your own continuous color scale
my_scale = ['rgb(255, 0, 0)', 'rgb(3, 252, 40)']

# Create the bar plot
fig = px.bar(data_frame=student_scores, 
             x='student_name', y='score', title='Student Scores by Student',
             # Set the color variable and scale
             color='score',
             color_continuous_scale=my_scale
             )

# Show the plot
fig.show()


# ## Side-by-side revenue box plots with color
# > 
# > The New York Stock Exchange firm you did work for previously has contracted you to extend on your work building the box plot of company revenues.
# > 
# > They want to understand how different industries compare using this same visualization technique from before. They are also particular about what colors are used for what industries. They have prepared a list of industries and the colors as below.
# > 
# > Your task is to create a box plot of company revenues, as before, but include the specified colors based on the list of industries given below.
# > 
# > There is a `revenues` DataFrame already loaded for your use.
# > 
# > **Industry-color RGB definitions**:
# > 
# > -   Tech = 124, 250, 120
# > -   Oil = 112,128,144
# > -   Pharmaceuticals = 137, 109, 247
# > -   Professional Services = 255, 0, 0

# ### init

# In[41]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(revenues)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'revenues.csv': 'https://file.io/xMffz257T0pt'}}
"""
prefixToc='3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
revenues = pd.read_csv(prefix+'revenues.csv',index_col=0)


# ### code

# [Side-by-side revenue box plots with color | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/introduction-to-plotly-1?ex=11)
# 
# > -   Create a dictionary to map the different `Industry` values to the RGB codes noted above.
# > -   Create a box plot of `revenues`, setting the y-axis to be the company revenue.
# > -   Use the industry color map you created in the box plot construction.
# > -   Set the appropriate column for the `color` variable.

# In[60]:


revenues.Revenue = pd.to_numeric(revenues.Revenue, errors='coerce')
# Create the industry-color map
ind_color_map = {'Tech': 'rgb(124, 250, 120)', 'Oil': 'rgb(112, 128, 144)', 
                 'Pharmaceuticals': 'rgb(137, 109, 247)', 'Professional Services': 'rgb(255, 0, 0)'}

# Create the basic box plot
fig = px.box(
  			# Set the data and y variable
  			data_frame=revenues, y='Revenue',
  			# Set the color map and variable
			color_discrete_map=ind_color_map,
			color='Industry')

# Show the plot
fig.show()


# ## Revenue histogram with stacked bars
# > 
# > The New York Stock exchange firm thought your previous histogram provided great insight into how the revenue of the firms they are looking at is distributed.
# > 
# > However, like before, they are interested in learning a bit more about how the industry of the firms could shed more light on what is happening.
# > 
# > Your task is to re-create the histogram of company revenues, as before, but include the specified colors based on the list of industries given below.
# > 
# > There is a `revenues` DataFrame already loaded for your use.
# > 
# > **Industry-color RGB definitions**:
# > 
# > -   Tech = 124, 250, 120
# > -   Oil = 112,128,144
# > -   Pharmaceuticals = 137, 109, 247
# > -   Professional Services = 255, 0, 0

# ### code

# [Revenue histogram with stacked bars | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/introduction-to-plotly-1?ex=12)
# 
# > -   Create a dictionary to map the different `Industry` values to the RGB codes noted above.
# > -   Create a histogram of `revenues`, setting the x-axis to be the company revenue.
# > -   Use the industry color map you created in the histogram plot construction.
# > -   Set the appropriate column for the `color` argument.

# In[61]:


# Create the industry-color map
ind_color_map = {'Tech': 'rgb(124, 250, 120)', 'Oil': 'rgb(112, 128, 144)', 
                 'Pharmaceuticals': 'rgb(137, 109, 247)', 'Professional Services': 'rgb(255, 0, 0)'}

# Create a simple histogram
fig = px.histogram(
  			# Set the data and x variable
  			data_frame=revenues, x='Revenue', nbins=5,
    		# Set the color map and variable
			color_discrete_map=ind_color_map,
			color='Industry')

# Show the plot
fig.show()


# In[ ]:




