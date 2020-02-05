#!/usr/bin/env python
# coding: utf-8

# In[63]:


#chargement du dataset
import pandas as pd

dataset = pd.read_excel('Clustering BMM.xlsx')

colonnes_violettes = ['Cond TH', 'Masse V', 'Chaleur M', 'K (Pa.s)','n','Alpha','T alpha (K)']
dataset


# # Unsupervised learning

# ## k-means clustering with scikit-learn

# In[64]:


from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(dataset[colonnes_violettes])


# In[65]:


dataset['labels'] = model.predict(dataset[colonnes_violettes])


# ## Scatter plots

# In[66]:


import matplotlib.pyplot as plt
xs = dataset[colonnes_violettes[0]]
ys = dataset[colonnes_violettes[3]]
plt.scatter(xs, ys, c=dataset['labels'], alpha=0.5)
centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()


# In[67]:


centroids


# ## Cross tabulation with pandas

# In[68]:


ct = pd.crosstab(dataset['FILIERE'], dataset['labels'])
print(ct)


# In[69]:


ct = pd.crosstab(dataset['POSTE'], dataset['labels'])
print(ct)


# In[70]:


ct = pd.crosstab(dataset['FILIERE PMU'], dataset['labels'])
print(ct)


# ## Measuring cluster quality - cluster 4 semble etre une bonne valeur

# In[71]:


from sklearn.cluster import KMeans

inertia=[]
ks= range(1,15)
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(dataset[colonnes_violettes])
    inertia.append(model.inertia_)

plt.plot(ks, inertia, '-o')
plt.xlabel('number of clusers, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# # Transforming features for better clusterings

# In[72]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dataset[colonnes_violettes])
dataset_scaled = scaler.transform(dataset[colonnes_violettes])

inertia=[]
ks= range(1,15)
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(dataset_scaled)
    inertia.append(model.inertia_)

plt.plot(ks, inertia, '-o')
plt.xlabel('number of clusers, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[73]:


kmeans = KMeans(n_clusters=6)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(dataset[colonnes_violettes])
labels = pipeline.predict(dataset[colonnes_violettes])


# In[74]:


labels


# In[75]:


kmeans.inertia_


# In[76]:


df = pd.DataFrame({'labels': labels, 'filiere': dataset['FILIERE'], 'poste': dataset['POSTE'], 'filiere_pmu': dataset['FILIERE PMU']})
ct_filiere = pd.crosstab(df['labels'], df['filiere'])
print(ct_filiere)
ct_poste = pd.crosstab(df['labels'], df['poste'])
print(ct_poste)
ct_filiere_pmu = pd.crosstab(df['labels'], df['filiere_pmu'])
print(ct_filiere_pmu)
ct_filiere_pmu


# 
