#!/usr/bin/env python
# coding: utf-8

# # chargement du dataset

# In[18]:


import pandas as pd

dataset = pd.read_excel('Clustering BMM.xlsx')

colonnes_violettes = ['Cond TH', 'Masse V', 'Chaleur M', 'K (Pa.s)','n','Alpha','T alpha (K)']
dataset


# In[19]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dataset[colonnes_violettes])
dataset_scaled = scaler.transform(dataset[colonnes_violettes])


# # Visualizing hierarchies

# ## dendrograms entre FAMILLE et les colonnes VIOLETTES

# In[20]:


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
plt.figure(figsize=[20, 16])
mergings = linkage(dataset[colonnes_violettes], method='complete')
dendrogram(mergings,
           labels=dataset[['FAMILLE']].values,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()


# ### sur données scalées

# In[35]:


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
plt.figure(figsize=[20, 10])
mergings = linkage(dataset_scaled, method='complete')
dendrogram(mergings,
           labels=dataset[['FAMILLE']].values,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()


# ## tSNE

# In[55]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(dataset[colonnes_violettes])
labels = pipeline.predict(dataset[colonnes_violettes])


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette(sns.color_palette("Set2", n_colors=6, desat=.5), n_colors=6)
plt.figure(figsize=[20, 10])

from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
sns.set_style("whitegrid")

transformed = model.fit_transform(dataset_scaled)
xs = transformed[:,0]
ys = transformed[:,1]
#plt.scatter(xs, ys, c=labels)
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
g = sns.scatterplot(x=xs, y=ys, hue=labels, palette=flatui, legend='full', s=200)
plt.show()


# In[56]:


dataset


# In[59]:


dataset['label']=labels


# In[60]:


dataset


# In[65]:


dataset.to_csv('pour_gaetan.csv')


# In[64]:


dataset['x']=xs
dataset['y']=ys


# In[ ]:




