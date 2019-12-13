#!/usr/bin/env python
# coding: utf-8

# # Dominant colors in images
# 

# ## Extract RGB values from image
# There are broadly three steps to find the dominant colors in an image:
# 
# - Extract RGB values into three lists.
# - Perform k-means clustering on scaled RGB values.
# - Display the colors of cluster centers.
# 
# To extract RGB values, we use the imread() function of the image class of matplotlib. Empty lists, r, g and b have been initialized.
# 
# For the purpose of finding dominant colors, we will be using the following image.
# ![image.png](attachment:image.png)

# ### code

# In[2]:


r,g,b=[],[],[]


# In[4]:


# Import image class of matplotlib
import matplotlib.image as img

# Read batman image and print dimensions
batman_image = img.imread('batman.jpg')
print(batman_image.shape)

# Store RGB values of all pixels in lists r, g and b
for row in batman_image:
    for temp_r, temp_g, temp_b in row:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)


# ## How many dominant colors?
# We have loaded the following image using the imread() function of the image class of matplotlib.
# 
# 
# 
# The RGB values are stored in a data frame, batman_df. The RGB values have been standardized used the whiten() function, stored in columns, scaled_red, scaled_blue and scaled_green.
# 
# Construct an elbow plot with the data frame. How many dominant colors are present?

# ### init: 1 dataframe

# In[7]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(batman_df)
tobedownloaded="{pandas.core.frame.DataFrame: {'batman_df.csv': 'https://file.io/9Bii8D'}}"
prefix='data_from_datacamp/Chap4-Exercise1.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[8]:


import pandas as pd
batman_df=pd.read_csv(prefix+'batman_df.csv',index_col=0)


# In[11]:


# Import the kmeans and vq functions
from scipy.cluster.vq import kmeans, vq
import seaborn as sns
import matplotlib.pyplot as plt


# ### code

# In[12]:


distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(batman_df[['scaled_red', 'scaled_blue', 'scaled_green']], i)
    distortions.append(distortion)

# Create a data frame with two lists, num_clusters and distortions
elbow_plot = pd.DataFrame({'num_clusters':num_clusters, 'distortions':distortions})

# Create a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.show()


# ## Display dominant colors
# We have loaded the following image using the imread() function of the image class of matplotlib.
# 
# 
# 
# To display the dominant colors, convert the colors of the cluster centers to their raw values and then converted them to the range of 0-1, using the following formula: converted_pixel = standardized_pixel * pixel_std / 255
# 
# The RGB values are stored in a data frame, batman_df. The scaled RGB values are stored in columns, scaled_red, scaled_blue and scaled_green. The cluster centers are stored in the variable cluster_centers, which were generated using the kmeans() function with three clusters.

# ### code

# - Get standard deviations of each color from the data frame and store it in r_std, g_std, b_std.
# - For each cluster center, convert the standardized RGB values to scaled values in the range of 0-1.
# - Display the colors of the cluster centers.
# 

# In[19]:


colors=[]
cluster_centers, distortion = kmeans(batman_df[['scaled_red', 'scaled_blue', 'scaled_green']], 3)


# In[20]:


# Get standard deviations of each color
r_std, g_std, b_std = batman_df[['red', 'green', 'blue']].std()

for cluster_center in cluster_centers:
    scaled_r, scaled_g, scaled_b = cluster_center
    # Convert each standardized value to scaled value
    colors.append((
        scaled_r * r_std / 255,
        scaled_g * g_std / 255,
        scaled_b * b_std / 255
    ))

# Display colors of cluster centers
plt.imshow([colors])
plt.show()


# # Document clustering
# 

# ## TF-IDF of movie plots
# Let us use the plots of randomly selected movies to perform document clustering on. Before performing clustering on documents, they need to be cleaned of any unwanted noise (such as special characters and stop words) and converted into a sparse matrix through TF-IDF of the documents.
# 
# Use the TfidfVectorizer class to perform the TF-IDF of movie plots stored in the list plots. The remove_noise() function is available to use as a tokenizer in the TfidfVectorizer class. The .fit_transform() method fits the data into the TfidfVectorizer objects and then generates the TF-IDF sparse matrix.
# 
# Note: It takes a few seconds to run the .fit_transform() method.

# ### init: 1 list

# In[21]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(plots)
tobedownloaded="{list: {'plots.txt': 'https://file.io/iuCa4u'}}"
prefix='data_from_datacamp/Chap4-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[22]:


from uploadfromdatacamp import loadListFromTxt

plots=loadListFromTxt(prefix+'plots.txt')


# In[29]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(stop_words_2)
tobedownloaded="{list: {'stop_words_2.txt': 'https://file.io/phH5EW'}}"
prefix='data_from_datacamp/Chap4-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[30]:


stop_words_2=loadListFromTxt(prefix+'stop_words_2.txt')


# In[31]:


#import inspect
#print(inspect.getsource(remove_noise))
def remove_noise(text, stop_words = stop_words_2):

    tokens = word_tokenize(text)

    cleaned_tokens = []

    for token in tokens:

        token = re.sub('[^A-Za-z0-9]+', '', token)

        if len(token) > 1 and token.lower() not in stop_words:
            # Get lowercase
            cleaned_tokens.append(token.lower())

    return cleaned_tokens


# In[39]:


from nltk import word_tokenize
import nltk
nltk.download('punkt')
import re


# ### code

# - Import TfidfVectorizer class from sklearn.
# - Initialize the TfidfVectorizer class with minimum and maximum frequencies of 0.1 and 0.75, and 50 maximum features.
# - Use the fit_transform() method on the initialized TfidfVectorizer class with the list plots.

# In[40]:


# Import TfidfVectorizer class from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.75, max_features=50, tokenizer=remove_noise)

# Use the .fit_transform() method on the list plots
tfidf_matrix = tfidf_vectorizer.fit_transform(plots)


# ## Top terms in movie clusters
# Now that you have created a sparse matrix, generate cluster centers and print the top three terms in each cluster. Use the .todense() method to convert the sparse matrix, tfidf_matrix to a normal matrix for the kmeans() function to process. Then, use the .get_feature_names() method to get a list of terms in the tfidf_vectorizer object. The zip() function in Python joins two lists.
# 
# The tfidf_vectorizer object and sparse matrix, tfidf_matrix, from the previous have been retained in this exercise. kmeans has been imported from SciPy.
# 
# With a higher number of data points, the clusters formed would be defined more clearly. However, this requires some computational power, making it difficult to accomplish in an exercise here.

# ### code

# In[42]:


num_clusters = 2

# Generate cluster centers through the kmeans function
cluster_centers, distortion = kmeans(tfidf_matrix.todense(), num_clusters)

# Generate terms from the tfidf_vectorizer object
terms = tfidf_vectorizer.get_feature_names()

for i in range(num_clusters):
    # Sort the terms and print top 3 terms
    center_terms = dict(zip(terms, list(cluster_centers[i])))
    sorted_terms = sorted(center_terms, key=center_terms.get, reverse=True)
    print(sorted_terms[:3])


# # Clustering with multiple features
# 

# ## Basic checks on clusters
# In the FIFA 18 dataset, we have concentrated on defenders in previous exercises. Let us try to focus on attacking attributes of a player. Pace (pac), Dribbling (dri) and Shooting (sho) are features that are present in attack minded players. In this exercise, k-means clustering has already been applied on the data using the scaled values of these three attributes. Try some basic checks on the clusters so formed.
# 
# The data is stored in a Pandas data frame, fifa. The scaled column names are present in a list scaled_features. The cluster labels are stored in the cluster_labels column. Recall the .count() and .mean() methods in Pandas help you find the number of observations and mean of observations in a data frame.

# ### init: 1 dataframe

# In[45]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(fifa)
tobedownloaded="{pandas.core.frame.DataFrame: {'fifa.csv': 'https://file.io/xJwn1n'}}"
prefix='data_from_datacamp/Chap4-Exercise3.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[46]:


import pandas as pd
fifa=pd.read_csv(prefix+'fifa.csv',index_col=0)


# ### code

# In[53]:


# Print the size of the clusters
print(fifa.groupby('cluster_labels')['ID'].count())

# Print the mean value of wages in each cluster
print(fifa.groupby('cluster_labels')['eur_wage'].mean())


# ## FIFA 18: what makes a complete player?
# The overall level of a player in FIFA 18 is defined by six characteristics: pace (pac), shooting (sho), passing (pas), dribbling (dri), defending (def), physical (phy).
# 
# Here is a sample card:
# 
# ![image.png](attachment:image.png)
# 
# Eden Hazard Player Card
# 
# In this exercise, you will use all six characteristics to create clusters. The data for this exercise is stored in a Pandas dataframe, fifa. features is the list of these column names and scaled_features is the list of columns which contains their scaled values. The following have been pre-loaded: kmeans, vq from scipy.cluster.vq, matplotlib.pyplot as plt, seaborn as sns.

# ### init: 1 dataframe

# In[60]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(fifa)
tobedownloaded="{pandas.core.frame.DataFrame: {'fifa.csv': 'https://file.io/SqaovY'}}"
prefix='data_from_datacamp/Chap4-Exercise3.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[61]:


import pandas as pd
fifa=pd.read_csv(prefix+'fifa.csv',index_col=0)


# ### code

# Use the kmeans() algorithm to create 2 clusters using the scaled features.

# In[62]:


from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
import seaborn as sns

scaled_features=['scaled_pac',
 'scaled_sho',
 'scaled_pas',
 'scaled_dri',
 'scaled_def',
 'scaled_phy']


# In[65]:


# Create centroids with kmeans for 2 clusters
cluster_centers,_ = kmeans(fifa[scaled_features], 2)


# Assign cluster labels to each row and print cluster centers using the .mean() method of Pandas.
# 

# In[66]:


# Assign cluster labels and print cluster centers
fifa['cluster_labels'], _ = vq(fifa[scaled_features],cluster_centers)
print(fifa.groupby('cluster_labels')[scaled_features].mean())


# Plot a bar chart of scaled attributes of each cluster center using the .plot() method of Pandas.

# In[67]:


# Plot cluster centers to visualize clusters
fifa.groupby('cluster_labels')[scaled_features].mean().plot(legend=True, kind='bar')
plt.show()


# Print the names of top 5 players in each cluster, using the name column.

# In[69]:


# Get the name column of top 5 players in each cluster
for cluster in fifa['cluster_labels'].unique():
    print(cluster, fifa[fifa['cluster_labels'] == cluster]['name'].values[:5])


# ![image.png](attachment:image.png)

# In[ ]:




