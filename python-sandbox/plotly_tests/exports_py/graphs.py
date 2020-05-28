#!/usr/bin/env python
# coding: utf-8

# # Forget Matplotlib! You Should Be Using Plotly

# The following examples are used to demonstrate why I prefer Plotly to Matplotlib. First, we plot some simple lines, then extend that to bubbles. Next, we show Plotly's interactivity (although all its plots here are interactive - by default). Finally, Plotly's animations are demonstrated.

# In[1]:


x = [i for i in range(-10, 10)]
y = [2*i**3 + 3*i + 3 for i in x]
z = [i**2 + 2 for i in x]


# ## Matplotlib - Line

# In[6]:


import matplotlib.pyplot as plt

plt.plot(x, y);


# ## Plotly - Line

# In[8]:


import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter(x=x, y=y)])
fig.show()


# ## Matplotlib - Bubble Plot

# In[9]:


plt.scatter(x, y, s=z, alpha=0.5)
plt.show()


# ## Plotly - Bubble Plot

# In[10]:


fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', marker_size=z))
fig.show()


# ## Plotly - Interactivity

# In[11]:


import plotly.graph_objects as go

x = [i/10 for i in range(-100, 100)]
y1 = [i**2 for i in x]
y2 = [i**3 for i in x]
y3 = [i**4 for i in x]

fig = go.Figure(data=go.Scatter(x=x, y=y1))
fig.add_trace(go.Scatter(x=x, y=y2))
fig.add_trace(go.Scatter(x=x, y=y3))

fig.show()


# ## Plotly - Animations

# In[12]:


import plotly.express as px

df = px.data.gapminder()

fig = px.bar(df,
             x='continent',
             y='gdpPercap',
             color='continent',
             animation_frame='year',
             animation_group='country',
             range_y=[0, 1000000])
fig.show()


# In[ ]:




