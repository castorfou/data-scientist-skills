#!/usr/bin/env python
# coding: utf-8

# # Bivariate visualizations

# ```python
# # scatterplot with plotly.express
# 
# import plotly.express as px
# fig = px.scatter(
#     data_frame=penguins,
#     x="Body Mass (g)",
#     y="Flipper Length (mm)")
# fig.show()
# ```
# 
# ![image.png](attachment:image.png)

# ```python
# # Line charts in plotly.express
# 
# fig = px.line(
# data_frame=msft_stock,
#     x='Date',
#     y='Open',
#     title='MSFT Stock Price (5Y)')
# fig.show()
# ```
# ![image.png](attachment:image.png)

# ```python
# # scatterplots and line plots with graph_objects
# 
# import plotly.graph_objects as go
# 
# fig = go.Figure(go.Scatter(
#     x=penguins['Body Mass (g)'],
#     y=penguins['Flipper Length (mm)'],
#     mode='markers'))
# 
# fig = go.Figure(go.Scatter(
#     x=msft_stock['Date'],
#     y=msft_stock['Opening Stock Price'],
#     mode='lines'))
# 
# ```

# ```python
# # Correlation plot with Plotly
# 
# import plotly.graph_objects as go
# fig = go.Figure(go.Heatmap(
#     x=cr.columns,
#     y=cr.columns,
#     z=cr.values.tolist(),
#     colorscale='rdylgn', zmin=-1, zmax=1))
# fig.show()
# ```
# 
# ![image.png](attachment:image.png)

# ## Building a scatterplot with specific colors
# > 
# > In your work as a data analyst, you have been engaged by a group of Antarctic research scientists to help them explore and report on their work.
# > 
# > They have spent a lot of time collating data on penguin species, but are having difficulty visualizing it to understand what is happening. Specifically, they have asked if you can help them plot their data in relation to statistics on the penguins' body attributes. They also suspect there is some pattern related to species, but are unsure how to plot this extra element.
# > 
# > In this exercise, you will help the scientific team by creating a scatterplot of the 'Culmen' (upper beak) attributes of the scientists' penguin data, ensuring that the species are included as specific colors.
# > 
# > You have been provided a `penguins` DataFrame.

# ### init

# In[10]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(penguins)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'penguins.csv': 'https://file.io/iHacVAT9mc0f'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
penguins = pd.read_csv(prefix+'penguins.csv',index_col=0)


# In[2]:


penguins.head()


# In[3]:


import plotly.express as px


# ### code

# [Building a scatterplot with specific colors | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/customizing-plots-2?ex=2)
# 
# > -   Create a `color_map` dictionary that maps the species (`Adelie`, `Gentoo`, `Chinstrap`) to the RGB codes `(235, 52, 52)`, `(235, 149, 52)`, and `(67, 52, 235)` respectively.
# > -   Create a basic scatterplot using `plotly.express` for the penguins data, visualizing the features `Culmen Length (mm)` on the x-axis and `Culmen Depth (mm)` on the y-axis.
# > -   Set the colors of the scatterplot to be the `Species` and use the `color_map` that you created.

# In[5]:


# Set up the color map
color_map = {'Adelie': 'rgb(235, 52, 52)' , 'Gentoo': 'rgb(235, 149, 52)', 'Chinstrap': 'rgb(67, 52, 235)'}

# Create a scatterplot
fig = px.scatter(data_frame=penguins, title="Penguin Culmen Statistics",
    x='Culmen Length (mm)',
    y='Culmen Depth (mm)',
    # Set the colors to use your color map
    color='Species',
    color_discrete_map=color_map
)

# Show your work
fig.show()


# ## Bird feature correlations
# > 
# > Continuing your work with the Antarctic Research Scientists, they loved the scatterplot you created for them.
# > 
# > Now they are sure there is a relationship between the attributes of the penguins. But how strong is that relationship and in what direction?
# > 
# > They have reached out again for help. Luckily you know just the plot: a correlation plot!
# > 
# > In this exercise, you will help the scientific team by creating a correlation plot between various penguin attributes in the provided `penguins` DataFrame.
# > 
# > `plotly.graph_objects` as been loaded as `go`.

# ### code

# In[9]:


import plotly.graph_objects as go


# [Bird feature correlations | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/customizing-plots-2?ex=3)
# 
# > Create a Pearson correlation table using `pandas` and save it as `penguin_corr`.

# In[6]:


# Create a correlation table with pandas
penguin_corr = penguins.corr(method='pearson')


# [Bird feature correlations | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/customizing-plots-2?ex=3)
# 
# > -   Create the correlation plot using `go.Heatmap()`, with appropriate `x`, `y` and `z` elements.
# > -   Use the red-green `colorscale` (`'rdylgn'`) in the correlation plot.

# In[11]:


# Set up the correlation plot
fig = go.Figure(go.Heatmap(
        # Set the appropriate x, y and z values
        z=penguin_corr.values.tolist(),
        x=penguin_corr.columns,
        y=penguin_corr.columns,
        # Set the color scale
        colorscale='rdylgn'))
# Show the plot
fig.show()


# [Bird feature correlations | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/customizing-plots-2?ex=3)
# 
# > Set the min and max values of the correlation plot to align with the minimum and maximum values a Pearson correlation can take.

# In[16]:


# Set up the correlation plot
fig = go.Figure(go.Heatmap(
  		# Set the appropriate x, y and z values
        z=penguin_corr.values.tolist(),
        x=penguin_corr.columns,
        y=penguin_corr.columns,
  		# Set the color scale,
        colorscale='rdylgn', 
  		# Set min and max values
        zmin=-1, zmax=1))

# Show the plot
fig.show()


# # Customizing hover information and legends

# ```python
# # Variables in hover information
# 
# fig = px.scatter(revenues,
#         x="Revenue",
#         y="employees",
#         hover_data=['age'])
# fig.show()
# 
# 
# # A styled legend
# 
# fig.update_layout({
#         'showlegend': True,
#         'legend': {
#         'title': 'All Companies',
#         'x': 0.5, 'y': 0.8,
#         'bgcolor': 'rgb(246,228,129)'}
# })
# ```
# 
# 

# ## GDP vs. life expectancy legend
# > 
# > You have been contacted by the United Nations to help them understand and visualize their data. They have been playing with various visualization tools, but just can't seem to find the design that they want.
# > 
# > They want to understand the relationship (if it exists) between GDP and life expectancy and have gathered data on over 200 countries to analyze. However, their initial efforts have been confusing to stakeholders and they need a clear legend positioned below the plot to help viewers understand it.
# > 
# > Your task is to create a scatterplot using the provided `life_gdp` DataFrame and style and position the legend as requested.

# ### init

# In[17]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(life_gdp)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'life_gdp.csv': 'https://file.io/ajoaxWzAsrMD'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
life_gdp = pd.read_csv(prefix+'life_gdp.csv',index_col=0)


# In[19]:


life_gdp.head()


# ### code

# [GDP vs. life expectancy legend | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/customizing-plots-2?ex=5)
# 
# > -   Create a scatterplot using the `life_gdp` DataFrame, setting the x-axis to be `Life expectancy`, the y-axis to be `GDP Per Capita` and colors based on `Continent`.
# > -   Create a legend dictionary that is positioned 20% along the x-axis and 95% up the y-axis.
# > -   Format the legend box to have a background color of the legend to be RGB code (60, 240, 201) and border width of 5.
# > -   Update the layout of your scatterplot to show the legend you just created!

# In[20]:


# Create the scatterplot
fig = px.scatter(
        data_frame=life_gdp, 
        x="Life expectancy", 
        y="GDP Per Capita", color='Continent')

# Create the legend
my_legend = {'x': 0.2, 'y': 0.95, 
            'bgcolor': 'rgb(60,240,201)', 'borderwidth': 5}

# Update the figure
fig.update_layout({'showlegend': True, 'legend': my_legend})

# Show the plot
fig.show()


# ## Enhancing our GDP plot
# > 
# > The United Nations loved your previous plot - the legend really stands out and makes it easier to view the plot.
# > 
# > However, there are some interesting data points that are not easy to further analyze due to the limited information in the plot.
# > 
# > Your task is to enhance the plot of `life_gdp` created in the last exercise to include more information in the hover and style it as requested.
# > 
# > Some of the code will be familiar from earlier in the chapter.

# ### code

# [Enhancing our GDP plot | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/customizing-plots-2?ex=6)
# 
# > -   Create a scatterplot using the `life_gdp` DataFrame, setting the x-axis to be `Life expectancy`, the y-axis to be `GDP Per Capita` and colors based on `Continent`.
# > -   Add the columns `Continent`, `Life expectancy`, and `GDP Per Capita` to appear in the hover information.
# > -   Set the `Country` variable to be bold at the top of the hover information.

# In[21]:


# Create the scatterplot
fig = px.scatter(
        data_frame=life_gdp, 
        x="Life expectancy", 
        y="GDP Per Capita", color='Continent',
  # Add columns to the hover information
  hover_data=['Continent', 'Life expectancy', 'GDP Per Capita'],
  # Add bold variable in hover information
  hover_name='Country'
)

# Show the plot
fig.show()


# # Adding annotations

# ```python
# # Data-linked annotations
# 
# my_annotation = {
#         'x': 215111, 'y': 449000,
#         'showarrow': True,'arrowhead': 3,
#         'text': "Our company is doing well",
#         'font' : {'size': 10, 'color': 'black'}}
# fig.update_layout({'annotations': [my_annotation]})
# fig.show()
# 
# # Floating annotation
# float_annotation = {
#         'xref': 'paper', 'yref': 'paper',
#         'x': 0.5, 'y': 0.8,
#         'showarrow': False,
#         'text': "You should <b>BUY</b>",
#         'font' : {'size': 15,'color': 'black'},
#         'bgcolor': 'rgb(255,0,0)'}
# ```

# ## Annotating your savings
# > 
# > You have been working hard over the last 30 weeks to build your savings balance for your first car. However, there is some extra context that needs to be added to explain a drop in savings and, later, a big increase in savings accumulated each fortnight.
# > 
# > Your task is to annotate the bar chart of your savings balance over the weeks and add two key annotations to the plot to explain what happened.
# > 
# > For both annotations:
# > 
# > -   Ensure the arrow is showing and the head of the arrow is size 4
# > -   Make the font black color using the string (`'black'`), not RGB method.
# > 
# > A figure `fig` has already been created using a `savings` DataFrame (with `x` as `Week`) for you.

# ### code

# [Annotating your savings | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/customizing-plots-2?ex=8)
# 
# > -   Create the `loss_annotation` at week 10 (savings 400) with text 'Urgent House Repairs'.
# > -   Create the `gain_annotation` at week 18 (savings 2500) with text 'New Job!'.
# > -   For both annotations, set the arrow to be showing, the arrowhead size to `4`, and the text color to black using the string, not RGB color.
# > -   Add both annotations to the `fig` using `update_layout()`.

# In[ ]:


# Create the first annotation
loss_annotation = {'x': 10, 'y': 400, 'showarrow': True, 'arrowhead': 4,
                    'font': {'color': 'black'}, 'text': 'Urgent House Repairs'}

# Create the second annotation
gain_annotation = {'x': 18, 'y':2500, 'showarrow': True, 'arrowhead': 4,
                    'font': {'color': 'black'}, 'text': 'New Job!'}

# Add annotations to the figure
fig.update_layout({'annotations': [loss_annotation, gain_annotation]})

# Show the plot!
fig.show()


# ## A happier histogram plot
# > 
# > The stock exchange firm you created the histogram for thinks that all the data and plots being created are too impersonal.
# > 
# > They have requested that a positive message be added to the histogram plot of company revenues you recently created.
# > 
# > You have just the right idea - you can wish the viewer a happy day and use the current day of the week for this!
# > 
# > There is a `fig` histogram available for you, feel free to `show()` or `print()` it to remind yourself what it looks like.

# ### code

# [A happier histogram plot | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/customizing-plots-2?ex=9)
# 
# > -   Position the annotation halfway along the x-axis and 95% up the y-axis.
# > -   Set the correct values for `xref` and `yref` to allow for relative positioning.
# > -   Use the provided `today` variable in the text, setting the font size to `20` and text color to `'white'`.
# > -   Use the `update_layout()` method to add the annotation.

# In[ ]:


# Get and format today's date
today = datetime.today().strftime('%A')

# Create the message_annotation
message_annotation = {
  # Set the correct coordinates
  'x': 0.5, 'y': 0.95, 'xref': 'paper', 'yref': 'paper',
  # Set format the text and box
  'text': f'Have a Happy {today} :)',
  'font': {'size': 20, 'color': 'white'},
  'bgcolor': 'rgb(237, 64, 200)', 'showarrow': False}

# Update the figure layout and show
fig.update_layout({'annotations': [message_annotation]})
fig.show()


# # Editing plot axes

# ```python
# 
# # Editing axis titles
# fig.update_xaxes(title_text='Species')
# fig.update_yaxes(title_text='Average Flipper Length')
# 
# # Or with the more general update_layout()
# fig.update_layout('xaxis': {'title': {'text': 'Species'}},
# 'yaxis': {'title':{'text': 'Average Flipper Length'}})
# 
# ```
# 
# See more on the Plotly documentation: https://plotly.com/python/reference/#layout-xaxis
# 
# ```python
# # Editing axes ranges
# 
# fig.update_layout({'yaxis':
#     {'range' : [150,
#     penguin_flippers['av_flip_length'].max() + 30]}
#     })
# 
# # Using log with our data
# fig = px.bar(billionaire_data,
#     x='Country',
#     y='Number Billionaires',
#     log_y=True)
# fig.show()
# 
# ```
# 
# 

# ## Analyzing basketball stats
# > 
# > You have been contracted by a national basketball team to help them visualize and understand key player stats for their top 50 players.
# > 
# > They have requested you to create a plot comparing players' 'Field Goal Percentage' (`FGP`) vs. their 'Points Per Game' (`PPG`). This sounds like a great opportunity to utilize your scatterplot skills!
# > 
# > It is important that this graph is comparable to their other graphs. Therefore, all axes need to start at 0 and the y-axis (`FGP`) needs to have a range of 0-100, since it is a percentage.
# > 
# > You have available a `bball_data` DataFrame with columns `FGP` and `PPG`.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(bball_data)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'bball_data.csv': 'https://file.io/88Ryi9xRP7xl'}}
"""
prefixToc='4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
bball_data = pd.read_csv(prefix+'bball_data.csv',index_col=0)


# In[2]:


bball_data.head()


# In[3]:


import plotly.express as px


# ### code

# [Analyzing basketball stats | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/customizing-plots-2?ex=12)
# 
# > -   Create a `plotly.express` scatterplot with `PPG` on the x-axis and `FGP` on the y-axis and show it to see the default doesn't start at zero.
# > -   Use the `update_layout()` method to change the x-axis range to be from 0 to a buffer of 5 past the maximum of the `PPG` variable.
# > -   Now use the `update_layout()` method to update the range of the y-axis to be 0 to 100.

# In[5]:


# Create and show the plot
fig = px.scatter(
  data_frame=bball_data,
  x='PPG', 
  y='FGP',
  title='Field Goal Percentage vs. Points Per Game')
fig.show()

# Update the x_axis range
fig.update_layout({'xaxis': {'range': [0, bball_data['PPG'].max() + 5]}})
fig.show()

# Update the y_axis range
fig.update_layout({'yaxis': {'range' : [0, 100]}})
fig.show()


# ## Styling scientific research
# > 
# > Now you have mastered customizing your plots it is time to let your creative energy flow!
# > 
# > In this exercise, you are continuing your work with the Antarctic research team to assist them to explore and understand the penguin data they have collected.
# > 
# > They have asked you to help them understand how the flipper length differs between species. Time is short, so you think a quick `plotly.express` visualization would do. However, they also want some specific customizations for the axes' titles and a timestamp annotation when the plot is generated.
# > 
# > Your task is to build a quick bar chart using `plotly.express`, including the specified customizations.
# > 
# > You have available a `penguins` DataFrame and `timestamp` variable which gives the current timestamp.

# ### init

# In[12]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(penguins)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'penguins.csv': 'https://file.io/brYjg63llIcm'}}
"""
prefixToc='4.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
penguins = pd.read_csv(prefix+'penguins.csv',index_col=0)


# In[15]:


from datetime import datetime


# ### code

# [Styling scientific research | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/customizing-plots-2?ex=13)
# 
# > -   Create a `plotly` bar chart from the penguins DataFrame with the title `Flipper Length (mm) by Species`, the x-axis as `spec`, the y-axis as `av_flip_length` and different colors for `spec`.
# > -   Use the `.update_layout()` method to change the `xaxis` and `yaxis` text to `Species` and `Average flipper length (mm)` respectively.
# > -   Add an annotation to the plot without an arrow, placed at an x coordinate of `0.5` and a y coordinate of `1.1` with the `timestamp` when the plot was generated.
# > -   Show your plot

# In[16]:


# Create timestamp
timestamp = datetime.now()

# Create plot
fig = px.bar(penguins, x='spec', y='av_flip_length', color="spec", title='Flipper Length (mm) by Species')

# Change the axis titles
fig.update_layout({'xaxis': {'title': {'text': 'Species'}},
                  'yaxis': {'title': {'text': 'Average Flipper Length (mm)'}}})

# Add an annotation and show
fig.update_layout({'annotations': [{
  "text": f"This graph was generated at {timestamp}", 
  "showarrow": False, "x": 0.5, "y": 1.1, "xref": "paper", "yref": "paper"}]})
fig.show()


# In[ ]:




