#!/usr/bin/env python
# coding: utf-8

# # Basics of k-means clustering

# ## K-means clustering: first exercise
# This exercise will familiarize you with the usage of k-means clustering on a dataset. Let us use the Comic Con dataset and check how k-means clustering works on it.
# 
# Recall the two steps of k-means clustering:
# 
# Define cluster centers through kmeans() function. It has two required arguments: observations and number of clusters.
# Assign cluster labels through the vq() function. It has two required arguments: observations and cluster centers.
# The data is stored in a Pandas data frame, comic_con. x_scaled and y_scaled are the column names of the standardized X and Y coordinates of people at a given point in time.
# 
# 

# ### init: 1 dataframe

# In[1]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(comic_con)
tobedownloaded="{pandas.core.frame.DataFrame: {'comic_con.csv': 'https://file.io/l1eRcR'}}"
prefix='data_from_datacamp/Chap3-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[2]:


import pandas as pd
comic_con=pd.read_csv(prefix+'comic_con.csv',index_col=0)


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### code

# In[5]:


# Import the kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

# Generate cluster centers
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], 2)

# Assign cluster labels
comic_con['cluster_labels'], distortion_list = vq(comic_con[['x_scaled', 'y_scaled']], cluster_centers)

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', 
                hue='cluster_labels', data = comic_con)
plt.show()


# ## Runtime of k-means clustering
# Recall that it took a significantly long time to run hierarchical clustering. How long does it take to run the kmeans() function on the FIFA dataset?
# 
# The data is stored in a Pandas data frame, fifa. scaled_sliding_tackle and scaled_aggression are the relevant scaled columns. timeit and kmeans have been imported.
# 
# Cluster centers are defined through the kmeans() function. It has two required arguments: observations and number of clusters. You can use %timeit before a piece of code to check how long it takes to run. You can time the kmeans() function for three clusters on the fifa dataset.

# ### init: 1 dataframe, timeit

# In[7]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(fifa)
tobedownloaded="{pandas.core.frame.DataFrame: {'fifa.csv': 'https://file.io/1gvitY'}}"
prefix='data_from_datacamp/Chap3-Exercise1.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[8]:


import pandas as pd
fifa=pd.read_csv(prefix+'fifa.csv',index_col=0)


# In[9]:


import timeit


# ### code

# In[10]:


get_ipython().run_line_magic('timeit', "kmeans(fifa[['scaled_sliding_tackle', 'scaled_aggression']], 3)")


# # How many clusters?

# ## Elbow method on distinct clusters
# Let us use the comic con data set to see how the elbow plot looks on a data set with distinct, well-defined clusters. You may want to display the data points before proceeding with the exercise.
# 
# The data is stored in a Pandas data frame, comic_con. x_scaled and y_scaled are the column names of the standardized X and Y coordinates of people at a given point in time.

# ### code

# In[11]:


distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']],i)
    distortions.append(distortion)

# Create a data frame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.show()


# ## Elbow method on uniform data
# In the earlier exercise, you constructed an elbow plot on data with well-defined clusters. Let us now see how the elbow plot looks on a data set with uniformly distributed points. You may want to display the data points on the console before proceeding with the exercise.
# 
# The data is stored in a Pandas data frame, uniform_data. x_scaled and y_scaled are the column names of the standardized X and Y coordinates of points.

# ### init: 1 dataframe

# In[12]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(uniform_data)
tobedownloaded="{pandas.core.frame.DataFrame: {'uniform_data.csv': 'https://file.io/oLAe6S'}}"
prefix='data_from_datacamp/Chap3-Exercise2.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[13]:


import pandas as pd
uniform_data=pd.read_csv(prefix+'uniform_data.csv',index_col=0)


# ### code

# In[14]:


distortions = []
num_clusters = range(2, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(uniform_data[['x_scaled', 'y_scaled']],i)
    distortions.append(distortion)

# Create a data frame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.show()


# In[15]:


# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', 
                 data = uniform_data)
plt.show()


# # Limitations of k-means clustering
# 

# ## Impact of seeds on distinct clusters
# You noticed the impact of seeds on a dataset that did not have well-defined groups of clusters. In this exercise, you will explore whether seeds impact the clusters in the Comic Con data, where the clusters are well-defined.
# 
# The data is stored in a Pandas data frame, comic_con. x_scaled and y_scaled are the column names of the standardized X and Y coordinates of people at a given point in time.

# ### code

# In[16]:


# Import random class
from numpy import random

# Initialize seed
random.seed(0)

# Run kmeans clustering
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], 2)
comic_con['cluster_labels'], distortion_list = vq(comic_con[['x_scaled', 'y_scaled']], cluster_centers)

# Plot the scatterplot
sns.scatterplot(x='x_scaled', y='y_scaled', 
                hue='cluster_labels', data = comic_con)
plt.show()


# In[17]:


# Import random class
from numpy import random

# Initialize seed
random.seed([1, 2, 1000])

# Run kmeans clustering
cluster_centers, distortion = kmeans(comic_con[['x_scaled', 'y_scaled']], 2)
comic_con['cluster_labels'], distortion_list = vq(comic_con[['x_scaled', 'y_scaled']], cluster_centers)

# Plot the scatterplot
sns.scatterplot(x='x_scaled', y='y_scaled', 
                hue='cluster_labels', data = comic_con)
plt.show()


# ## Uniform clustering patterns
# Now that you are familiar with the impact of seeds, let us look at the bias in k-means clustering towards the formation of uniform clusters.
# 
# Let us use a mouse-like dataset for our next exercise. A mouse-like dataset is a group of points that resemble the head of a mouse: it has three clusters of points arranged in circles, one each for the face and two ears of a mouse.
# 
# Here is how a typical mouse-like dataset looks like (Source).
# 
# The data is stored in a Pandas data frame, mouse. x_scaled and y_scaled are the column names of the standardized X and Y coordinates of the data points.

# ### init: 1 dataframe

# In[18]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(mouse)
tobedownloaded="{pandas.core.frame.DataFrame: {'mouse.csv': 'https://file.io/YpVaxH'}}"
prefix='data_from_datacamp/Chap3-Exercise3.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[19]:


import pandas as pd
mouse=pd.read_csv(prefix+'mouse.csv',index_col=0)


# ### code

# In[20]:


# Import the kmeans and vq functions
from scipy.cluster.vq import kmeans, vq


# Generate cluster centers
cluster_centers, distortion = kmeans(mouse[['x_scaled', 'y_scaled']], 3)

# Assign cluster labels
mouse['cluster_labels'], distortion_list = vq(mouse[['x_scaled', 'y_scaled']], cluster_centers)

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', 
                hue='cluster_labels', data = mouse)
plt.show()


# ![image.png](attachment:image.png)

# ## FIFA 18: defenders revisited
# In the FIFA 18 dataset, various attributes of players are present. Two such attributes are:
# 
# defending: a number which signifies the defending attributes of a player
# physical: a number which signifies the physical attributes of a player
# These are typically defense-minded players. In this exercise, you will perform clustering based on these attributes in the data.
# 
# The following modules have been pre-loaded: kmeans, vq from scipy.cluster.vq, matplotlib.pyplot as plt, seaborn as sns. The data for this exercise is stored in a Pandas dataframe, fifa. The scaled variables are scaled_def and scaled_phy.

# ### init: 1 dataframe

# In[25]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(fifa)
tobedownloaded="{pandas.core.frame.DataFrame: {'fifa.csv': 'https://file.io/6NM6qh'}}"
prefix='data_from_datacamp/Chap3-Exercise3.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[26]:


import pandas as pd
fifa=pd.read_csv(prefix+'fifa.csv',index_col=0)


# ### code

# Initialize the random seed to the list [1000,2000].

# In[27]:


# Set up a random seed in numpy
random.seed([1000,2000])


# Fit the scaled data in columns scaled_def and scaled_phy into a k-means clustering algorithm with 3 clusters and assign cluster labels.

# In[28]:


# Fit the data into a k-means algorithm
cluster_centers,_ = kmeans(fifa[['scaled_def', 'scaled_phy']],3)

# Assign cluster labels
fifa['cluster_labels'],_ = vq(fifa[['scaled_def', 'scaled_phy']],cluster_centers)


# Display cluster centers of each cluster with respect to the scaled columns by calculating the mean value for each cluster.

# In[29]:


# Display cluster centers 
print(fifa[['scaled_def', 'scaled_phy', 'cluster_labels']].groupby('cluster_labels').mean())


# Create a seaborn scatter plot with scaled_def on the x-axis and scaled_phy on the y-axis, with each cluster represented by a different color.

# In[30]:


# Create a scatter plot through seaborn
sns.scatterplot(x='scaled_def', y='scaled_phy', hue='cluster_labels', data=fifa)
plt.show()


# In[ ]:




