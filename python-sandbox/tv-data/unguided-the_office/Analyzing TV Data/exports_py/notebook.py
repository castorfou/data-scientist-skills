#!/usr/bin/env python
# coding: utf-8

# [Projects](https://projects.datacamp.com/projects/1170)
# 
# > # Instructions
# > 
# > Data visualization is often a great way to start exploring your data and uncovering insights. In this notebook, you will initiate this process by creating an informative plot of the episode data provided to you. In doing so, you're going to work on several different variables, including the episode number, the viewership, the fan rating, and guest appearances. Here are the requirements needed to pass this project:
# > 
# > 1.  Create a `matplotlib` **scatter plot** of the data that contains the following attributes:
# >     
# >     -   Each episode's **episode number plotted along the x-axis**
# >     -   Each episode's **viewership (in millions) plotted along the y-axis**
# >     -   A **color scheme** reflecting the **scaled ratings** (not the regular ratings) of each episode, such that:
# >         -   Ratings < 0.25 are colored `"red"`
# >         -   Ratings >= 0.25 and < 0.50 are colored `"orange"`
# >         -   Ratings >= 0.50 and < 0.75 are colored `"lightgreen"`
# >         -   Ratings >= 0.75 are colored `"darkgreen"`
# >     -   A **sizing system**, such that episodes with guest appearances have a marker size of `250` and episodes without are sized `25`
# >     -   A **title**, reading `"Popularity, Quality, and Guest Appearances on the Office"`
# >     -   An **x-axis label** reading `"Episode Number"`
# >     -   A **y-axis label** reading `"Viewership (Millions)"`
# > 2.  Provide the name of one of the guest stars (hint, there were multiple!) who was in the most watched Office episode. Save it as a string in the variable `top_star` (e.g. `top_star = "Will Ferrell"`).
# >     
# > 
# > ### Important!
# > 
# > To test your `matplotlib` plot, you will need to initalize a `matplotlib.pyplot` `fig` object, which you can do using the code `fig = plt.figure()` (provided you have imported `matplotlib.pyplot` as `plt`). In addition, in order to test it correctly, **please make sure to specify your plot (including the type, data, labels, etc) in the same cell as the one you initialize your figure** (`fig`)! _You are still free to use other cells to load data, experiment, and answer Question 2._
# > 
# > _In addition, if you want to be able to see a larger version of your plot, you can set the figure size parameters using this code (provided again you have imported `matplotlib.pyplot` as `plt`):_
# > 
# > `plt.rcParams['figure.figsize'] = [11, 7]`
# > 
# > ## Bonus Step!
# > 
# > Although it was not taught in Intermediate Python, a useful skill for visualizing different data points is to use a different marker. You can learn more about them via the [Matplotlib documentation](https://matplotlib.org/api/markers_api.html) or via our course [Introduction to Data Visualization with Matplotlib](https://learn.datacamp.com/courses/introduction-to-data-visualization-with-matplotlib). Thus, as a bonus step, try to differentiate guest appearances not just with size, but also with a star!
# > 
# > All other attributes still apply (data on the axes, color scheme, sizes for guest appearances, title, and axis labels).

# ## 1. Welcome!
# <p><img src="https://assets.datacamp.com/production/project_1170/img/office_cast.jpeg" alt="Markdown">.</p>
# <p><strong>The Office!</strong> What started as a British mockumentary series about office culture in 2001 has since spawned ten other variants across the world, including an Israeli version (2010-13), a Hindi version (2019-), and even a French Canadian variant (2006-2007). Of all these iterations (including the original), the American series has been the longest-running, spanning 201 episodes over nine seasons.</p>
# <p>In this notebook, we will take a look at a dataset of The Office episodes, and try to understand how the popularity and quality of the series varied over time. To do so, we will use the following dataset: <code>datasets/office_episodes.csv</code>, which was downloaded from Kaggle <a href="https://www.kaggle.com/nehaprabhavalkar/the-office-dataset">here</a>.</p>
# <p>This dataset contains information on a variety of characteristics of each episode. In detail, these are:
# <br></p>
# <div style="background-color: #efebe4; color: #05192d; text-align:left; vertical-align: middle; padding: 15px 25px 15px 25px; line-height: 1.6;">
#     <div style="font-size:20px"><b>datasets/office_episodes.csv</b></div>
# <ul>
#     <li><b>episode_number:</b> Canonical episode number.</li>
#     <li><b>season:</b> Season in which the episode appeared.</li>
#     <li><b>episode_title:</b> Title of the episode.</li>
#     <li><b>description:</b> Description of the episode.</li>
#     <li><b>ratings:</b> Average IMDB rating.</li>
#     <li><b>votes:</b> Number of votes.</li>
#     <li><b>viewership_mil:</b> Number of US viewers in millions.</li>
#     <li><b>duration:</b> Duration in number of minutes.</li>
#     <li><b>release_date:</b> Airdate.</li>
#     <li><b>guest_stars:</b> Guest stars in the episode (if any).</li>
#     <li><b>director:</b> Director of the episode.</li>
#     <li><b>writers:</b> Writers of the episode.</li>
#     <li><b>has_guests:</b> True/False column for whether the episode contained guest stars.</li>
#     <li><b>scaled_ratings:</b> The ratings scaled from 0 (worst-reviewed) to 1 (best-reviewed).</li>
# </ul>
#     </div>

# # mon projet

# In[68]:


# Use this cell to begin your analysis, and add as many as you would like!
import pandas as pd

office_episodes=pd.read_csv('datasets/office_episodes.csv')


# In[69]:


office_episodes.head()


# In[76]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [11, 7]


# ## Create a matplotlib scatter plot of the data 

# In[81]:


fig = plt.figure()
plt.style.use('fivethirtyeight')

office_episodes['marker_size']=office_episodes['has_guests'].apply(lambda x: 250 if x==True else 25)
office_episodes['color']=pd.cut(office_episodes['scaled_ratings'], bins=[0,0.25,0.5,0.75, 1], include_lowest=True, right=True, labels=["red", "orange", "Lightgreen", "darkgreen"])

office_episodes_guests = office_episodes[office_episodes.has_guests == True]
office_episodes_noguests = office_episodes[office_episodes.has_guests == False]

plt.scatter(x='episode_number', y="viewership_mil", c='color', s='marker_size', marker='*', data=office_episodes_guests)
plt.scatter(x='episode_number', y="viewership_mil", c='color', s='marker_size',  data=office_episodes_noguests)
plt.xlabel('Episode Number')

plt.ylabel('Viewership (Millions)')
plt.title("Popularity, Quality, and Guest Appearances on the Office")
plt.show()


# ## Provide the name of one of the guest stars (hint, there were multiple!)

# In[41]:


max_viewership = max(office_episodes['viewership_mil'])
top_star = office_episodes[office_episodes.viewership_mil == max_viewership]['guest_stars'].values[0].split(',')[0]
top_star


# In[ ]:




