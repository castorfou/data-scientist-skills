#!/usr/bin/env python
# coding: utf-8

# # Measures of center

# ## Mean and median
# 
# In this chapter, you'll be working with the [2018 Food Carbon Footprint Index](https://www.nu3.de/blogs/nutrition/food-carbon-footprint-index-2018) from nu3. The `food_consumption` dataset contains information about the kilograms of food consumed per person per year in each country in each food category (`consumption`) as well as information about the carbon footprint of that food category (`co2_emissions`) measured in kilograms of carbon dioxide, or CO², per person per year in each country.
# 
# In this exercise, you'll compute measures of center to compare food consumption in the US and Belgium using your `pandas` and `numpy` skills.
# 
# `pandas` is imported as `pd` for you and `food_consumption` is pre-loaded.

# ### init

# In[1]:


import pandas as pd


# In[8]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(food_consumption)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'food_consumption.csv': 'https://file.io/ZTSaN4ceMIst'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
food_consumption = pd.read_csv(prefix+'food_consumption.csv',index_col=0)


# ### code

# In[10]:


# Import numpy with alias np
import numpy as np

# Filter for Belgium
be_consumption = food_consumption[food_consumption.country=='Belgium']

# Filter for USA
usa_consumption = food_consumption[food_consumption.country=='USA']

# Calculate mean and median consumption in Belgium
print(np.mean(be_consumption.consumption))
print(np.median(be_consumption.consumption))

# Calculate mean and median consumption in USA
print(np.mean(usa_consumption.consumption))
print(np.median(usa_consumption.consumption))


# In[11]:


# Import numpy as np
import numpy as np

# Subset for Belgium and USA only
be_and_usa = food_consumption[(food_consumption.country == 'Belgium') | (food_consumption.country == 'USA')]

# Group by country, select consumption column, and compute mean and median
print(be_and_usa.groupby('country')['consumption'].agg([np.mean, np.median]))


# ## Mean vs. median
# 
# In the video, you learned that the mean is the sum of all the data points divided by the total number of data points, and the median is the middle value of the dataset where 50% of the data is less than the median, and 50% of the data is greater than the median. In this exercise, you'll compare these two measures of center.
# 
# pandas is loaded as pd, numpy is loaded as np, and food_consumption is available.

# In[12]:


# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Subset for food_category equals rice
rice_consumption = food_consumption[food_consumption.food_category == 'rice']

# Histogram of co2_emission for rice and show plot
rice_consumption['co2_emission'].hist()
plt.show()


# In[13]:


# Subset for food_category equals rice
rice_consumption = food_consumption[food_consumption['food_category'] == 'rice']

# Calculate mean and median of co2_emission with .agg()
print(rice_consumption['co2_emission'].agg([np.mean, np.median]))


# # Measures of spread 

# ## Quartiles, quantiles, and quintiles
# 
# Quantiles are a great way of summarizing numerical data since they can be used to measure center and spread, as well as to get a sense of where a data point stands in relation to the rest of the data set. For example, you might want to give a discount to the 10% most active users on a website.
# 
# In this exercise, you'll calculate quartiles, quintiles, and deciles, which split up a dataset into 4, 5, and 10 pieces, respectively.
# 
# Both pandas as pd and numpy as np are loaded and `food_consumption` is available.

# In[14]:


# Calculate the quartiles of co2_emission
print(np.quantile(food_consumption['co2_emission'], np.linspace(0,1,5)))


# In[15]:


# Calculate the quintiles of co2_emission
print(np.quantile(food_consumption['co2_emission'], np.linspace(0,1,6)))


# In[16]:


# Calculate the deciles of co2_emission
print(np.quantile(food_consumption['co2_emission'], np.linspace(0,1,10)))


# ## Variance and standard deviation
# 
# Variance and standard deviation are two of the most common ways to measure the spread of a variable, and you'll practice calculating these in this exercise. Spread is important since it can help inform expectations. For example, if a salesperson sells a mean of 20 products a day, but has a standard deviation of 10 products, there will probably be days where they sell 40 products, but also days where they only sell one or two. Information like this is important, especially when making predictions.
# 
# Both pandas as pd and numpy as np are loaded, and `food_consumption` is available.

# In[17]:


# Print variance and sd of co2_emission for each food_category
print(food_consumption.groupby('food_category')['co2_emission'].agg([np.var, np.std]))

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Create histogram of co2_emission for food_category 'beef'
food_consumption[food_consumption.food_category == 'beef']['co2_emission'].hist()
# Show plot
plt.show()

# Create histogram of co2_emission for food_category 'eggs'
food_consumption[food_consumption.food_category == 'eggs']['co2_emission'].hist()
# Show plot
plt.show()


# ## Finding outliers using IQR
# 
# Outliers can have big effects on statistics like mean, as well as statistics that rely on the mean, such as variance and standard deviation. Interquartile range, or IQR, is another way of measuring spread that's less influenced by outliers. IQR is also often used to find outliers. If a value is less than <math xmlns="http://www.w3.org/1998/Math/MathML">
#   <mtext>Q1</mtext>
#   <mo>&#x2212;</mo>
#   <mn>1.5</mn>
#   <mo>&#xD7;</mo>
#   <mtext>IQR</mtext>
# </math>
# or greater than <math xmlns="http://www.w3.org/1998/Math/MathML">
#   <mtext>Q3</mtext>
#   <mo>+</mo>
#   <mn>1.5</mn>
#   <mo>&#xD7;</mo>
#   <mtext>IQR</mtext>
# </math>
# 
# , it's considered an outlier. In fact, this is how the lengths of the whiskers in a matplotlib box plot are calculated.
# 
# ![image.png](attachment:image.png)
# Diagram of a box plot showing median, quartiles, and outliers
# 
# In this exercise, you'll calculate IQR and use it to find some outliers. pandas as pd and numpy as np are loaded and `food_consumption` is available.

# In[18]:


# Calculate total co2_emission per country: emissions_by_country
emissions_by_country = food_consumption.groupby('country')['co2_emission'].agg('sum')

print(emissions_by_country)


# In[19]:


# Calculate total co2_emission per country: emissions_by_country
emissions_by_country = food_consumption.groupby('country')['co2_emission'].sum()

# Compute the first and third quartiles and IQR of emissions_by_country
q1 = np.quantile(emissions_by_country, 0.25)
q3 = np.quantile(emissions_by_country, 0.75)
iqr = q3-q1


# In[20]:



# Calculate the lower and upper cutoffs for outliers
lower = q1-1.5*iqr
upper = q3+1.5*iqr


# In[21]:


# Subset emissions_by_country to find outliers
outliers = emissions_by_country[(emissions_by_country.values > upper) | (emissions_by_country.values < lower)]
print(outliers)


# In[ ]:




