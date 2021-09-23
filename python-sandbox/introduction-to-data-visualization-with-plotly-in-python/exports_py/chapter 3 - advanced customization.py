#!/usr/bin/env python
# coding: utf-8

# # Subplots

# ```python
# # Creating a 1x2 subplot
# 
# from plotly.subplots import make_subplots
# fig = make_subplots(rows=2, cols=1)
# fig.add_trace(
#     go.Histogram(x=revenues['Revenue'], nbinsx=5),
#     row=1, col=1)
# fig.add_trace(
#     go.Box(x=revenues['Revenue'],
#     hovertext=revenues['Company']),
#     row=2, col=1)
# fig.show()
# 
# # Subplot titles
# from plotly.subplots import make_subplots
# fig = make_subplots(rows=2, cols=1,
#     subplot_titles=[
#     'Histogram of company revenues',
#     'Box plot of company revenues'])
# ## Add in traces (fig.add_trace())
# fig.update_layout({'title': {'text':
#     'Plots of company revenues',
#     'x': 0.5, 'y': 0.9}})
# fig.show()
# 
# # Subplot legends
# fig.add_trace(
#     go.Histogram(x=revenues.Revenue,
#     nbinsx=5, name='Histogram'),
#     row=1, col=1)
# fig.add_trace(
#     go.Box(x=revenues.Revenue,
#     hovertext=revenues['Company'],
#     name='Box plot'),
#     row=2, col=1)
# 
# # Stacked subplots
# 
# fig = make_subplots(rows=3, cols=1)
# row_num = 1
# for species in ['Adelie', 'Gentoo', 'Chinstrap']:
#     df = penguins[penguins['Species'] == species]
#     fig.add_trace(
#         go.Scatter(x=df['Culmen Length (mm)'],
#         y=df['Culmen Depth (mm)'],
#         name=species, mode='markers'),
#         row=row_num, col=1)
#     row_num +=1
# fig.show()
# 
# # Subplots with shared axes
# fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
# 
# ```

# ## Revenue box subplots
# > 
# > You are now a regular contractor with the New York Stock Exchange, who have asked you to revisit the box plots by industry you created previously.
# > 
# > They are creating some visualizations for a specific presentation and have found that the plot you created before is too wide. They are also only interested in 4 specific industries.
# > 
# > The `make_subplots()` function has been imported for you already.

# ### init

# In[3]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(df_tech)
uploadToFileIO(df_prof_serve)
uploadToFileIO(df_retail)
uploadToFileIO(df_oil)

"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df_tech.csv': 'https://file.io/56DzCqJ8LG1m',
'df_prof_serve.csv': 'https://file.io/ZNRFzaEb6G0T',
'df_retail.csv': 'https://file.io/XDfIBBftF76k',
'df_oil.csv': 'https://file.io/r7HQnrmCLIlw'
}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
df_tech = pd.read_csv(prefix+'df_tech.csv',index_col=0)
df_prof_serve = pd.read_csv(prefix+'df_prof_serve.csv',index_col=0)
df_retail = pd.read_csv(prefix+'df_retail.csv',index_col=0)
df_oil = pd.read_csv(prefix+'df_oil.csv',index_col=0)


# ### code

# [Revenue box subplots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-customization?ex=2)
# 
# > -   Set up the subplots in a 2x2 grid.
# > -   Set the titles to be the industries ('Tech', 'Professional Services', 'Retail', 'Oil').

# In[5]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[6]:


# Set up the subplots grid
fig = make_subplots(rows=2, cols=2, 
                    # Set the subplot titles
                    subplot_titles=['Tech', 'Professional Services', 'Retail', 'Oil'])


# [Revenue box subplots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-customization?ex=2)
# 
# > Add the box plots in order to the subplot squares. You have DataFrames `df_tech`, `df_prof_serve`, `df_retail`, and `df_oil` already subsetted for you.

# In[7]:


# Add the Tech trace
fig.add_trace(go.Box(x=df_tech.Revenue, name='', showlegend=False), row=1, col=1)
# Add the Professional Services trace
fig.add_trace(go.Box(x=df_prof_serve.Revenue, name='', showlegend=False), row=1, col=2)
# Add the Retail trace
fig.add_trace(go.Box(x=df_retail.Revenue, name='', showlegend=False), row=2, col=1)
# Add the Oil trace
fig.add_trace(go.Box(x=df_oil.Revenue, name='', showlegend=False), row=2, col=2)


# [Revenue box subplots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-customization?ex=2)
# 
# > -   Add a plot title `Box plots of company revenues` and position it 50% along the x-axis and 90% up the y-axis. Note that `name` is an empty string and legend is removed since our subplot titles already nicely label our plots.

# In[8]:


# Add a title (and show)
fig.update_layout({'title': {'text': 'Box plots of company revenues', 'x': .5, 'y': .9}})
fig.show()


# ## Revenue histogram subplots
# > 
# > The revenue histogram with colors by industry (with stacked bars) you created for the The New York Stock exchange firm was enlightening for which industries tended to be in which area of the histogram.
# > 
# > However, the firm wishes to understand the distribution of each industry without having to hover to see. For this analysis, the previous histogram has too much in a single plot, but they don't want multiple plots. How can you help solve this conundrum?
# > 
# > Your task is to create a histogram of company revenues by industry as a stacked subplot and a shared x-axis to allow meaningful comparison of industries.
# > 
# > You have a `revenues` DataFrame loaded for you.

# ### init

# In[9]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(revenues)


"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'revenues.csv': 'https://file.io/dlEcHZMRDI2s'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
revenues = pd.read_csv(prefix+'revenues.csv',index_col=0)


# ### code

# [Revenue histogram subplots | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-customization?ex=3)
# 
# > -   Create the subplots grid with 3 rows and 1 column and make the x-axis shared.
# > -   Loop through the desired industries, adding a histogram trace each time with the industry name.
# > -   Position the trace in the appropriate place in the subplot grid.

# In[10]:


# Create the subplots
fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

# Loop through the industries
row_num = 1
for industry in ['Tech', 'Retail', 'Professional Services']:
    df = revenues[revenues.Industry == industry]
    # Add a histogram using subsetted df
    fig.add_trace(go.Histogram(x=df['Revenue'], name=industry),
    # Position the trace
    row=row_num, col=1)
    row_num +=1

# Show the plot
fig.show()


# # Layering multiple plots

# ```python
# # GDP growth layered plot
# 
# fig = go.Figure()
# fig.add_trace(go.Bar(x=gdp['Date'],
#     y=gdp['Quarterly growth (%)'],
#     name='Quarterly Growth (%)'))
# fig.add_trace(go.Scatter(x=gdp['Date'],
#     y=gdp['Rolling YTD growth (%)'],
#     name='Rolling YTD Growth (%)',
#     mode='lines+markers'))
# fig.show()
# 
# ```

# ## Species on different islands
# > 
# > The Antarctic research scientists are back with another brief. They want to be able to visualize how their data collection counts differ between species and islands.
# > 
# > Specifically, they want to easily compare islands based on the count of different species of penguins they recorded there.
# > 
# > You have the perfect plot - you will layer several bar charts together for easy comparison!
# > 
# > You have been provided a `penguins_grouped` DataFrame that has the count of samples for each species at each island as well as an `islands` list of the different islands where research was undertaken.

# ### init

# In[11]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(penguins_grouped)


"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'penguins_grouped.csv': 'https://file.io/qQYKMx95AMRF'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
penguins_grouped = pd.read_csv(prefix+'penguins_grouped.csv',index_col=0)


# In[12]:


islands = ['Torgersen', 'Biscoe', 'Dream']


# ### code

# [Species on different islands | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-customization?ex=5)
# 
# > -   View the printed out `penguins_grouped` DataFrame to see its structure.
# > -   Create an empty figure object.
# > -   Loop through the `species`, adding a bar chart trace to the base figure.
# > -   Set the appropriate y subset and name for the bar chart trace being added.

# In[14]:


penguins_grouped.head()


# In[15]:


# Create the base figure
fig = go.Figure()

# Loop through the species
for species in ['Adelie', 'Chinstrap', 'Gentoo']:
  # Add a bar chart trace
  fig.add_trace(go.Bar(x=islands,
    # Set the appropriate y subset and name
    y=penguins_grouped[penguins_grouped.Species == species]['Count'],
    name=species))
# Show the figure
fig.show()


# ## Monthly temperatures layered
# > 
# > The Australian Bureau Of Meteorology has tasked you with helping them build some nice interactive plots for their website.
# > 
# > They want to look at both the daily temperature from January to July this year and smooth out all the data points with a nice trend line of the monthly average temperature.
# > 
# > This would be an excellent opportunity to layer two plots together to achieve the desired outcome.
# > 
# > You have been provided a `temp_syd` DataFrame that contains the daily (max) temperature from January to July in 2020. You also have a `temp_syd_avg` DataFrame containing each month's average daily (max) temperature.

# ### init

# In[16]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(temp_syd)
uploadToFileIO(temp_syd_avg)

"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'temp_syd.csv': 'https://file.io/s5pQ3gGDFld9',
'temp_syd_avg.csv': 'https://file.io/HdHbo2bmJrhO'}}

"""
prefixToc='2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
temp_syd = pd.read_csv(prefix+'temp_syd.csv',index_col=0)
temp_syd_avg = pd.read_csv(prefix+'temp_syd_avg.csv',index_col=0)


# ### code

# [Monthly temperatures layered | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-customization?ex=6)
# 
# > -   Create an empty figure object.
# > -   Add a bar chart trace for `temp_syd`, using `Date` on the x-axis and `Temp` as the y-axis with a `name` of `Daily Max Temperature`.
# > -   Add a line chart trace for `temp_syd_avg` using `Date` on the x-axis and `Average` as the y-axis with a `name` of `Average Monthly Temperature`.

# In[17]:


# Create the base figure
fig = go.Figure()

# Add the bar graph of daily temperatures
fig.add_trace(
  go.Bar(x=temp_syd['Date'], y=temp_syd['Temp'], name='Daily Max Temperature'))

# Add the monthly average line graph
fig.add_trace(
  go.Scatter(x=temp_syd_avg['Date'], y=temp_syd_avg['Average'], name='Average Monthly Temperature'))

# Show the plot
fig.show()


# # Time buttons

# ```python
# 
# date_buttons = [
# {'count': 6, 'step': "month", 'stepmode': "todate", 'label': "6MTD"},
# {'count': 14, 'step': "day", 'stepmode': "todate", 'label': "2WTD"}
# ]
# 
# fig = px.line(data_frame=rain, x='Date',
#     y='Rainfall',
#     title="Rainfall (mm) in Sydney")
# fig.update_layout(
#     {'xaxis':
#         {'rangeselector':
#             {'buttons': date_buttons}
#     }})
# fig.show()
# 
# ```
# ![image.png](attachment:image.png)

# ## Time buttons on our rainfall graph
# > 
# > The local news station is wanting to update the graphics in the weather section of their website. They have contacted you to assist in jazzing up the old images and tables they have.
# > 
# > They have requested a line chart, but with the ability to filter the data for the last 4 weeks (`4WTD`), last 48 hours (`48HR`) and the year to date (`YTD`).
# > 
# > In this exercise, you will help the news station by building their line chart with the requested buttons.
# > 
# > You have a `rain` DataFrame available that contains the necessary data.

# ### init

# In[19]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(rain)


"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'rain.csv': 'https://file.io/lqk25Pq6T6jN'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
rain = pd.read_csv(prefix+'rain.csv',index_col=0)


# In[20]:


rain.head()


# In[22]:


import plotly.express as px


# ### code

# [Time buttons on our rainfall graph | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-customization?ex=9)
# 
# > -   Create the list of buttons specified with the names noted above.
# > -   Create the basic line chart using the `rain` DataFrame, appropriately using the `Date` and `Rainfall` columns.
# > -   Update the figure using `update_layout()` to construct buttons using the list you just created.

# In[23]:


# Create the buttons
date_buttons = [
{'count': 4*7, 'label': '4WTD', 'step': "day", 'stepmode': "todate"},
{'count': 48, 'label': '48HR', 'step': "hour", 'stepmode': "todate"},
{'count': 1, 'label': 'YTD', 'step': "year", 'stepmode': "todate"}]

# Create the basic line chart
fig = px.line(data_frame=rain, x='Date', y='Rainfall', 
              title="Rainfall (mm)")

# Add the buttons and show
fig.update_layout(
  	{'xaxis':
    {'rangeselector': {'buttons': date_buttons}}})
fig.show()


# ## Finance line chart with custom time buttons
# > 
# > You have been engaged by an Excel-savvy finance trading company to help them jazz up their data visualization capabilities. Safe to say, Excel graphics aren't cutting it for them!
# > 
# > The fund is particularly interested in the electric vehicle company Tesla and how it has performed this year and wants a tool that can help them zoom to view key timeframes.
# > 
# > In this exercise, you will help the trading company by visualizing the opening stock price of Tesla over 2020 and create the following date-filter buttons:
# > 
# > -   **1WTD** = The previous week (7 days to date)
# > -   **6MTD** = The previous 6 months week (6 months to date)
# > -   **YTD** = The current year to date
# > 
# > You have a `tesla` DataFrame available that contains the necessary data.

# ### init

# In[24]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(tesla)


"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'tesla.csv': 'https://file.io/DmlhIGyHOosO'}}
"""
prefixToc='3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
tesla = pd.read_csv(prefix+'tesla.csv',index_col=0)


# In[25]:


tesla.head()


# ### code

# [Finance line chart with custom time buttons | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-plotly-in-python/advanced-customization?ex=10)
# 
# > -   Create a basic line chart of the `tesla` DataFrame using the `Date` and `Open` columns.
# > -   Create a list called `fin_buttons` containing the custom date-filter buttons mentioned above.
# > -   Update the figure using `.update_layout()` to construct buttons using the list you just created.

# In[27]:


# Create the basic line chart
fig = px.line(data_frame=tesla, x='Date', y='Open', title="Tesla Opening Stock Prices")

# Create the financial buttons
fin_buttons = [
  {'count': 7, 'label': "1WTD", 'step': 'day', 'stepmode': 'todate'},
  {'count': 6, 'label': "6MTD", 'step': 'month', 'stepmode': 'todate'},
  {'count': 1, 'label': "YTD", 'step': 'year', 'stepmode': 'todate'}
]

# Create the date range buttons & show the plot
fig.update_layout({'xaxis': {'rangeselector': {'buttons': fin_buttons}}})
fig.show()


# In[ ]:




