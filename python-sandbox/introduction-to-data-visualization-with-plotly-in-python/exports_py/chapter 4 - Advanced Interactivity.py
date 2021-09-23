#!/usr/bin/env python
# coding: utf-8

# # Custom buttons

# ```python
# 
# # Button set up
# 
# my_buttons = [
#     {'label': "Bar plot",
#     'method': "update",
#     'args': [{"type": "bar"}]},
#     {'label': "scatterplot",
#     'method': "update",
#     'args': [{"type": "scatter", 'mode': 'markers'}]}
# ]
# 
# # Let's see what is inside the gure's layout element:
# dir(fig.layout)
# 
# # Let's also what is inside the gure's data element (of the rst trace):
# dir(fig.data[0])
# 
# # Button interactivity
# fig.update_layout({
#     'updatemenus': [{'type': "buttons",
#                 'direction': 'down',
#                 'x': 1.3, 'y': 0.5,
#                 'showactive': True,
#                 'active': 0,
#                 'buttons': my_buttons}]
#     })
# fig.show()
# 
# ```

# ## Rainfall plot type buttons
# > 
# > You have been contacted by the Australian Bureau of Meteorology to assist them in understanding monthly rainfall.
# > 
# > They are not sure what plot would be best to visualize this data, so are wondering if there is a way to make this part of the interactivity. They don't want subplots, as they will only need to view one plot at at time.
# > 
# > Your task is to create a bar plot of rainfall per month with a button that allows the user to easily change from a bar plot to a scatterplot of the same data.
# > 
# > A `rain` DataFrame is loaded for you.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(rain)


"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'rain.csv': 'https://file.io/ebUMiZgb4ln7'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
rain = pd.read_csv(prefix+'rain.csv',index_col=0)


# In[2]:


rain.head()


# In[4]:


import plotly.express as px


# ### code

# [Rainfall plot type buttons | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=2)
# 
# > -   Create a simple bar chart from the provided DataFrame, with the `Month` on the x-axis and `Rainfall` on y-axis.
# > -   Create a button for the bar plot and scatterplot using the plot `type` argument fed to `args`.
# > -   Add the buttons to the plot, setting the `direction` to `down` so the buttons are on top of each other.

# In[5]:


# Create a simple bar chart
fig = px.bar(data_frame=rain, x='Month', y='Rainfall')

# Create the buttons
my_buttons = [{'label': "Bar plot", 'method': "update", 'args': [{"type": 'bar'}]},
  {'label': "scatterplot", 'method': "update", 'args': [{"type": 'scatter', 'mode': 'markers'}]}]

# Add buttons to the plot and show
fig.update_layout({
    'updatemenus': [{
      'type':'buttons','direction': 'down',
      'x': 1.3,'y': 0.5,
      'showactive': True, 'active': 0,
      'buttons': my_buttons}]})
fig.show()


# ## Changing annotations with buttons
# > 
# > A large e-commerce company's sales department has asked you to assist them understanding and visualizing their data.
# > 
# > They have provided monthly sales data but have different metrics to determine monthly performance; the monthly sales value (in dollars) as well as sales volume (number of items sold).
# > 
# > They would like to view all this information on the same plot but easily be able to toggle annotations on and off to facilitate their discussion around this.
# > 
# > In this exercise, you will help the sales department by creating a bar chart of their sales data with buttons to turn key annotations on or off.
# > 
# > You have been provided a `sales` DataFrame.

# ### init

# In[6]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(sales)


"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'sales.csv': 'https://file.io/L5p6f5QVhl1s'}}
"""
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
sales = pd.read_csv(prefix+'sales.csv',index_col=0)


# In[7]:


sales.head()


# ### code

# [Changing annotations with buttons | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=3)
# 
# > -   Create the basic figure object using `plotly.graph_objects` (imported as `go`).
# > -   Add two bar chart type traces, one for the `Sales Volume` and one for `Sales Value` (both with `Month` on the x-axis).

# In[10]:


import plotly.graph_objects as go


# In[11]:


# Create the basic figure
fig = go.Figure()

# Add a trace per metric
fig.add_trace(go.Bar(x=sales["Month"], y=sales['Sales Volume'], name='Sales Volume'))
fig.add_trace(go.Bar(x=sales["Month"], y=sales['Sales Value'], name='Sales Value'))

# Take a look at your plot so far
fig.show()


# [Changing annotations with buttons | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=3)
# 
# > -   Create annotations to identify the best months for sales volume (`October`) and value (`September`).
# > -   Create buttons that will update the `annotations` part of your graph layout, ensuring the relevant annotation object is given to each button.

# In[13]:


# Create annotations
value_annotations=[{'text': 'Sept was the best' ,'showarrow': True, 'x': 'September', 'y': 345397 }]
volume_annotations=[{'text': 'Oct was the best', 'showarrow': True, 'x': 'October', 'y': 71900 }]

# Create buttons
my_buttons = [
{'label': "By Sales Value", 'method': "update", 'args': [{}, {'annotations': value_annotations}]},
{'label': "By Sales Volume", 'method': "update", 'args': [{}, {'annotations': volume_annotations}]}]


# [Changing annotations with buttons | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=3)
# 
# > Update the layout of this figure to insert your new buttons, ensuring they are displayed on top of each other.

# In[14]:


# Add the buttons
fig.update_layout({
    'updatemenus':[{
            'type': "buttons",
            'direction': 'down',
            'x': 1.3,'y': 0.5, 'buttons': my_buttons
            }]})
fig.show()


# # Dropdowns

# ```python
# 
# # Hiding a trace
# args:[{'visible': [True, False, False]}]
#     
# # Create the dropdown
# dropdown_buttons = [
#     {'label': 'Ashfield', 'method': 'update',
#     'args': [{'visible': [True, False, False]},
#     {'title': 'Ashfield'}]},
#     {'label': 'Lidcombe', 'method': 'update',
#     'args': [{'visible': [False, True, False]},
#     {'title': 'Lidcombe'}]},
#     {'label': "Bondi Junction", 'method': "update",
#     'args': [{"visible": [False, False, True]},
#     {'title': 'Bondi Junction'}]}
# ]    
# 
# # Adding the dropdown
# fig.update_layout({
#     'updatemenus':[{
#         'type': "dropdown",
#         'x': 1.3,
#         'y': 0.5,
#         'showactive': True,
#         'active': 0,
#         'buttons': dropdown_buttons}]
# })
# fig.show()
# ```

# ## Growth locations dropdown
# > 
# > The Australian Government is looking to understand which Local Government Areas (LGAs) have had recent strong population growth to assist in planning infrastructure projects.
# > 
# > They have provided you with some data on the top 5 LGAs by the percentage population increase (from 2018 to 2019) and have asked if you can visualize it. However, they want to be able to select a certain state or see everything at once.
# > 
# > In this exercise, you are tasked with creating a bar chart of this data with a drop-down menu to switch between different states and see all states at once.
# > 
# > You have a `pop_growth` DataFrame available.

# ### init

# In[15]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(pop_growth)


"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'pop_growth.csv': 'https://file.io/iS4EVDPMkj4i'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
pop_growth = pd.read_csv(prefix+'pop_growth.csv',index_col=0)


# In[16]:


pop_growth.head()


# ### code

# [Growth locations dropdown | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=5)
# 
# > -   Create the basic figure object using `plotly.graph_objects` (imported as `go`).
# > -   Loop through the DataFrame to subset by state and add a trace (bar chart) for each state.

# In[18]:


# Create the basic figure
fig = go.Figure()

# Loop through the states
for state in ['NSW', 'QLD', 'VIC']:
  	# Subset the DataFrame
    df = pop_growth[pop_growth.State == state]
    # Add a trace for each state subset
    fig.add_trace(go.Bar(x=df['Local Government Area'], y=df['Change %'], name=state))


# [Growth locations dropdown | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=5)
# 
# > -   Create a list of the buttons for your dropdown menu that will update the trace visibility to be either `ALL`, `NSW`, `QLD`, or `VIC`.

# In[19]:


# Create the buttons
dropdown_buttons = [
{'label': "ALL", 'method': 'update', 'args': [{"visible": [True, True, True]}, {"title": "ALL"}]},
{'label': "NSW", 'method': 'update', 'args': [{"visible": [True, False, False]}, {"title": "NSW"}]},
{'label': "QLD", 'method': 'update', 'args': [{"visible": [False, True, False]}, {"title": "QLD"}]},
{'label': "VIC", 'method': 'update', 'args': [{"visible": [False, False, True]}, {"title": "VIC"}]},
]


# [Growth locations dropdown | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=5)
# 
# > Update the layout of this figure to insert your new buttons, ensuring the first button (`0`) is the active one.

# In[20]:


# Update the figure to add dropdown menu
fig.update_layout({
  		'updatemenus': [
        {'active': 0, 'buttons': dropdown_buttons}
        ]})

# Show the plot
fig.show()


# ## Housing prices dropdown
# > 
# > You are working as a data analyst for a real-estate investing firm. The firm has asked you to help them understand property price returns over the last five years for several key Sydney suburbs. Your work will supplement the qualitative analysis on these suburbs. They want to be able to visualize each suburb by itself, but easily switch between them.
# > 
# > They have provided you with some data on the prices of these suburbs in 2015 and 2020.
# > 
# > In this exercise, you are tasked with creating a line chart of this data with a dropdown to select each suburb. Additionally, they have identified that one of the suburbs is having great growth so want to annotate only that trace. You accept the challenge!
# > 
# > You have a `house_prices` DataFrame available.

# ### init

# In[21]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(house_prices)


"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'house_prices.csv': 'https://file.io/yAbVLwNW2Dzq'}}
"""
prefixToc='2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
house_prices = pd.read_csv(prefix+'house_prices.csv',index_col=0)


# In[22]:


house_prices.head()


# ### code

# [Housing prices dropdown | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=6)
# 
# > -   Create a basic figure object using `plotly.graph_objects` (imported as `go`).
# > -   Loop through the DataFrame to subset by state and add a line chart for each suburb of `Year` (x-axis) vs `Median House Price` (y-axis).

# In[24]:


# Create the basic figure
fig = go.Figure()

# Loop through the suburbs
for suburb in ['Greenacre', 'Lakemba']:
  	# Subset the DataFrame
    df = house_prices[house_prices.Suburb == suburb]
    # Add a trace for each suburb subset
    fig.add_trace(go.Scatter(
                   x=df['Year'],
                   y=df['Median House Price'],
                   name=suburb, mode='lines'))


# [Housing prices dropdown | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=6)
# 
# > -   Create a list of the buttons for your dropdown menu that will update the trace visibility to be either `Greenacre`, or `Lakemba`.
# > -   Add the `ga_annotation` to the Greenacre dropdown.

# In[26]:


# Annotation
ga_annotation=[{ 'text': 'Price boom!','showarrow': True, 'x': 'Year: 2018', 'y': 712678}]

# Create the buttons
dropdown_buttons = [
{'label': "Greenacre", 'method': 'update', 'args': [{"visible": [True, False]}, {'title': 'Greenacre', 'annotations': ga_annotation}]},
{'label': "Lakemba", 'method': 'update', 'args': [{"visible": [False, True]}, {'title': 'Lakemba', 'annotations': []}]},
]


# [Housing prices dropdown | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=6)
# 
# > Update the layout of this figure to insert your new dropdown, ensuring the correct `type` is used.

# In[27]:


# Update the figure to add dropdown menu
fig.update_layout({
    'updatemenus':[{
            'type': 'dropdown',
            'x': 1.3, 'y': 0.5,
            'showactive': True, 'active': 0,
            'buttons': dropdown_buttons
            }]})

# Show the plot
fig.show()


# # Sliders

# ```python
# 
# # Revenue vs. Employees with slider
# fig = px.scatter(
#     data_frame=revenues,
#     y='Revenue',
#     x='Employees',
#     color='Industry',
#     animation_frame='Year',
#     animation_group='Company')
# 
# fig.update_layout({
#     'yaxis': {'range': [0, 500000]},
#     'xaxis': {'range': [-100000, 2500000]}
# })
# fig['layout'].pop('updatemenus')
# fig.show()
# 
# 
# # with GO
# # Creating the figure
# fig = go.Figure()
# for island in ['Torgersen', 'Biscoe', 'Dream']:
#     df = penguins[penguins.Island == island]
#     fig.add_trace(go.Scatter(
#         x=df["Culmen Length (mm)"],
#         y=df["Culmen Depth (mm)"], mode='markers', name=island))
# # Creating the slider   
# sliders = [
#     {'steps':[
#         {'method': 'update', 'label': 'Torgersen',
#         'args': [{'visible': [True, False, False]}]},
#         {'method': 'update', 'label': 'Bisco',
#         'args': [{'visible': [False, True , False]}]},
#         {'method': 'update', 'label': 'Dream',
#         'args': [{'visible': [False, False, True]}]}
#     ]}
# ]    
# 
# # Adding the slider
# fig.update_layout({'sliders': sliders})
# fig.show()
# 
# # Fixing the initial display
# # Make traces invisible
# fig.data[1].visible=False
# fig.data[2].visible=False
# fig.update_layout({'sliders': sliders})
# fig.show()
# ```
# 
# 

# ## Rainfall by season slider
# > 
# > The Australian Bureau of Meteorology has contracted you for some follow up work to the visualization they loved on rainfall per month in Sydney.
# > 
# > They would love the ability to cycle through seasons. Not as a change-between as in buttons and dropdowns, but more of a slide-between.
# > 
# > Sounds like the perfect job for a slider!
# > 
# > In this exercise, you are tasked with creating a bar chart of the rainfall data with a slider through the seasons.
# > 
# > You have a `rain_pm` DataFrame available.

# ### init

# In[28]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(rain_pm)


"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'rain_pm.csv': 'https://file.io/pUHq3kzNrcqd'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
rain_pm = pd.read_csv(prefix+'rain_pm.csv',index_col=0)


# In[29]:


rain_pm.head()


# ### code

# [Rainfall by season slider | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=9)
# 
# > -   Create the basic figure object using `plotly.graph_objects` (imported as `go`).
# > -   Loop through the DataFrame to subset season and create a bar chart trace for each, with the `Month` on the x-axis and `Total Rainfall` on the y-axis.

# In[30]:


# Create the basic figure
fig = go.Figure()

# Loop through the states
for season in ['Autumn', 'Winter', 'Spring']:
  	# Subset the DataFrame
    df = rain_pm[rain_pm.Season == season]
    # Add a trace for each season
    fig.add_trace(go.Bar(x=df['Month'], y=df['Total Rainfall'], name=season))


# [Rainfall by season slider | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=9)
# 
# > -   Create a list of the buttons for your dropdown menu that will update the trace visibility based on the season in the slider.
# > -   Set the label to be the season, in the order they were subset.

# In[32]:


# Create the slider elements
sliders = [
    {'steps':[
    {'method': 'update', 'label': 'Autumn', 'args': [{'visible': [True, False, False]}]},
    {'method': 'update', 'label': 'Winter', 'args': [{'visible': [False, True, False]}]},
    {'method': 'update', 'label': 'Spring', 'args': [{'visible': [False, False, True]}]}]}]


# [Rainfall by season slider | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-interactivity?ex=9)
# 
# > Update the layout of this figure to insert your slider.

# In[33]:


# Update the figure to add sliders and show
fig.update_layout({'sliders': sliders})

# Show the plot
fig.show()


# In[ ]:




