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

# In[ ]:




