#!/usr/bin/env python
# coding: utf-8

# # Preparing your figures to share with others
# 
# ```python
# 
# # Choosing a style
# plt.style.use("ggplot")
# fig, ax = plt.subplots()
# ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"
# ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
# ax.set_xlabel("Time (months)")
# ax.set_ylabel("Average temperature (Fahrenheit degrees)")
# plt.show()
#                                                   
# # Back to the default
# plt.style.use("default")                                                  
# ```
# [available styles](https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html)
# 
# ```python
# # The "bmh" style
# plt.style.use("bmh")
# fig, ax = plt.subplots()
# ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"
# ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
# ax.set_xlabel("Time (months)")
# ax.set_ylabel("Average temperature (Fahrenheit degrees)")
# plt.show()
#                                                   
# # Seaborn styles
# plt.style.use("seaborn-colorblind")
# fig, ax = plt.subplots()
# ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"
# ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
# ax.set_xlabel("Time (months)")
# ax.set_ylabel("Average temperature (Fahrenheit degrees)")
# plt.show()                                                  
# ```

# [Selecting a style for printing | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-matplotlib/sharing-visualizations-with-others?ex=2)
# 
# > ## Selecting a style for printing
# > 
# > You are creating a figure that will be included in a leaflet printed on a black-and-white printer. What style should you choose for your figures?
# > 
# > In the console, we have loaded the medals dataset. Before initializing Axes and Figure objects and plotting them, you can try setting a style to use.

# [Switching between styles | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-matplotlib/sharing-visualizations-with-others?ex=3)
# 
# > ## Switching between styles
# > 
# > Selecting a style to use affects all of the visualizations that are created after this style is selected.
# > 
# > Here, you will practice plotting data in two different styles. The data you will use is the same weather data we used in the first lesson: you will have available to you the DataFrame `seattle_weather` and the DataFrame `austin_weather`, both with records of the average temperature in every month.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(seattle_weather, austin_weather)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'austin_weather.csv': 'https://file.io/7sbKoeGgGfDF',
  'seattle_weather.csv': 'https://file.io/PbWuH0NN7xAq'}}
"""
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
seattle_weather = pd.read_csv(prefix+'seattle_weather.csv',index_col=0)
austin_weather = pd.read_csv(prefix+'austin_weather.csv',index_col=0)


# ### code

# > Select the `'ggplot'` style, create a new Figure called `fig`, and a new Axes object called `ax` with `plt.subplots`.

# In[2]:


import matplotlib.pyplot as plt


# In[3]:


# Use the "ggplot" style and create new Figure/Axes
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
plt.show()


# > Select the `'Solarize_Light2'` style, create a new Figure called `fig`, and a new Axes object called `ax` with `plt.subplots`.

# In[4]:


# Use the "Solarize_Light2" style and create new Figure/Axes
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()


# # Saving your visualizations
# 
# ```python
# 
# # Saving the figure to file
# fig, ax = plt.subplots()
# ax.bar(medals.index, medals["Gold"])
# ax.set_xticklabels(medals.index, rotation=90)
# ax.set_ylabel("Number of medals")
# fig.savefig("gold_medals.png")
# 
# # Different file formats
# fig.savefig("gold_medals.jpg")
# fig.savefig("gold_medals.jpg", quality=50)
# fig.savefig("gold_medals.svg")
# 
# # Resolution
# fig.savefig("gold_medals.png", dpi=300)
# 
# # Size
# fig.set_size_inches([5, 3])
# 
# # Another aspect ratio
# fig.set_size_inches([3, 5])
# ```

# [Saving a file several times | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-matplotlib/sharing-visualizations-with-others?ex=5)
# 
# > ## Saving a file several times
# > 
# > If you want to share your visualizations with others, you will need to save them into files. Matplotlib provides as way to do that, through the `savefig` method of the `Figure` object. In this exercise, you will save a figure several times. Each time setting the parameters to something slightly different. We have provided and already created `Figure` object.

# > Examine the figure by calling the `plt.show()` function.

# ### init

# In[5]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(medals)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'medals.csv': 'https://file.io/FZDQId8VkVDY'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
medals = pd.read_csv(prefix+'medals.csv',index_col=0)


# ### code

# In[11]:


fig, ax = plt.subplots()
fig.set_size_inches([5, 5])
plt.style.use("default")  

# Plot a bar-chart of gold medals as a function of country
ax.bar(medals.index, medals["Gold"])

# Set the x-axis tick labels to the country names
ax.set_xticklabels(medals.index, rotation=90)

# Set the y-axis label
ax.set_ylabel('Number of medals')
# Show the figure
plt.show()


# > Save the figure into the file `my_figure.png`, using the default resolution.

# In[12]:


# Save as a PNG file
fig.savefig('my_figure.png')


# > Save the figure into the file `my_figure_300dpi.png` and set the resolution to 300 dpi.

# In[13]:


# Save as a PNG file with 300 dpi
fig.savefig("my_figure_300dpi.png", dpi=300)


# [Save a figure with different sizes | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-matplotlib/sharing-visualizations-with-others?ex=6)
# 
# > ## Save a figure with different sizes
# > 
# > Before saving your visualization, you might want to also set the size that the figure will have on the page. To do so, you can use the `Figure` object's `set_size_inches` method. This method takes a sequence of two values. The first sets the width and the second sets the height of the figure.
# > 
# > Here, you will again have a `Figure` object called `fig` already provided (you can run `plt.show` if you want to see its contents). Use the `Figure` methods `set_size_inches` and `savefig` to change its size and save two different versions of this figure.

# > Set the figure size as width of 3 inches and height of 5 inches and save it as `'figure_3_5.png'` with default resolution.

# In[14]:


# Set figure dimensions and save as a PNG
fig.set_size_inches([3, 5])
fig.savefig('figure_3_5.png')


# > Set the figure size to width of 5 inches and height of 3 inches and save it as `'figure_5_3.png'` with default settings.

# In[15]:


# Set figure dimensions and save as a PNG
fig.set_size_inches([5, 3])
fig.savefig('figure_5_3.png')


# # Automating figures from data
# 
# ```python
# 
# # Getting unique values of a column
# sports = summer_2016_medals["Sport"].unique()
# 
# # Bar-chart of heights for all sports
# fig, ax = plt.subplots()
# for sport in sports:
# sport_df = summer_2016_medals[summer_2016_medals["Sport"] == spor
# ax.bar(sport, sport_df["Height"].mean(),
# yerr=sport_df["Height"].std())
# ax.set_ylabel("Height (cm)")
# ax.set_xticklabels(sports, rotation=90)
# plt.show()
# ```

# [Unique values of a column | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-matplotlib/sharing-visualizations-with-others?ex=8)
# 
# > ## Unique values of a column
# > 
# > One of the main strengths of Matplotlib is that it can be automated to adapt to the data that it receives as input. For example, if you receive data that has an unknown number of categories, you can still create a bar plot that has bars for each category.
# > 
# > In this exercise and the next, you will be visualizing the weight of medalis in the 2016 summer Olympic Games again, from a dataset that has some unknown number of branches of sports in it. This will be loaded into memory as a Pandas `DataFrame` object called `summer_2016_medals`, which has a column called `"Sport"` that tells you to which branch of sport each row corresponds. There is also a `"Weight"` column that tells you the weight of each athlete.
# > 
# > In this exercise, we will extract the unique values of the `"Sport"` column

# ### init

# In[16]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(summer_2016_medals)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'summer_2016_medals.csv': 'https://file.io/z4U7sKkYwZ1y'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
summer_2016_medals = pd.read_csv(prefix+'summer_2016_medals.csv',index_col=0)


#  ### code

# > -   Create a variable called `sports_column` that holds the data from the `"Sport"` column of the `DataFrame` object.
# > -   Use the `unique` method of this variable to find all the unique different sports that are present in this data, and assign these values into a new variable called `sports`.
# > -   Print the `sports` variable to the console.

# In[18]:


summer_2016_medals.head()


# In[19]:


# Extract the "Sport" column
sports_column = summer_2016_medals['Sport']

# Find the unique values of the "Sport" column
sports = sports_column.unique()

# Print out the unique sports values
print(sports)


# [Automate your visualization | Python](https://campus.datacamp.com/courses/introduction-to-data-visualization-with-matplotlib/sharing-visualizations-with-others?ex=9)
# 
# > ## Automate your visualization
# > 
# > One of the main strengths of Matplotlib is that it can be automated to adapt to the data that it receives as input. For example, if you receive data that has an unknown number of categories, you can still create a bar plot that has bars for each category.
# > 
# > This is what you will do in this exercise. You will be visualizing data about medal winners in the 2016 summer Olympic Games again, but this time you will have a dataset that has some unknown number of branches of sports in it. This will be loaded into memory as a Pandas `DataFrame` object called `summer_2016_medals`, which has a column called `"Sport"` that tells you to which branch of sport each row corresponds. There is also a `"Weight"` column that tells you the weight of each athlete.

# > -   Iterate over the values of `sports` setting `sport` as your loop variable.
# > -   In each iteration, extract the rows where the `"Sport"` column is equal to `sport`.
# > -   Add a bar to the provided `ax` object, labeled with the sport name, with the mean of the `"Weight"` column as its height, and the standard deviation as a y-axis error bar.
# > -   Save the figure into the file `"sports_weights.png"`.

# In[24]:


fig, ax = plt.subplots()

# Loop over the different sports branches
for sport in sports:
  # Extract the rows only for this sport
  sport_df = summer_2016_medals[summer_2016_medals['Sport']==sport]
  # Add a bar for the "Weight" mean with std y error bar
  ax.bar(sport, sport_df['Weight'].mean(), yerr=sport_df['Weight'].std())

ax.set_ylabel("Weight")
ax.set_xticklabels(sports, rotation=90)

# Save the figure to file
fig.savefig("sports_weights.png")


# In[ ]:




