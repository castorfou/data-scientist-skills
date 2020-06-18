#!/usr/bin/env python
# coding: utf-8

# # Question de gaetan

# https://www.yammer.com/michelin.com/#/Threads/show?threadId=728531149766656

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # 1ere option avec seaborn

# ## chargement des données

# In[1]:


import pandas as pd

filename = 'DATASET_ML25307.xlsx'

df_ml25307 = pd.read_excel(filename, parse_dates=[['Date', 'Heure']])


# In[2]:


df_ml25307.info()


# In[3]:


df_ml25307.Date_Heure.min(), df_ml25307.Date_Heure.max(), 


# In[4]:


df_ml25307['SITE Z'].value_counts()


# In[5]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15,7))
df_ml25307[['Date_Heure', 'TEMCHIMI']].set_index('Date_Heure').plot(ax=ax)
plt.show()


# ### jour Michelin 

# In[6]:


def ajoute_un_jour_Michelin(date, heureEquipeA):
    heure = date.hour
    if (heure < heureEquipeA):
        return date+np.timedelta64(24, 'h')
    return date

#print(ajoute_un_jour_Michelin(pd.to_datetime('2019-10-05 05:59:00'), heureEquipeA = 6))
#print(ajoute_un_jour_Michelin(pd.to_datetime('2019-10-05 06:01:00'), heureEquipeA = 6))

def dateMichelin_vers_dateCalendaire(df, heureEquipeA = 6):
    df_cop = df.copy()
    df_cop['Date_Heure']=df_cop['Date_Heure'].apply(lambda row:ajoute_un_jour_Michelin(row, heureEquipeA))
    return df_cop


# In[7]:


df_ml25307_date_calendaire = dateMichelin_vers_dateCalendaire(df_ml25307, heureEquipeA=6)


# In[8]:


fig, ax = plt.subplots(figsize=(15,7))
df_ml25307_date_calendaire[['Date_Heure', 'TEMCHIMI']].set_index('Date_Heure').plot(ax=ax)
plt.show()


# ### split date et heure

# In[9]:


df_ml25307_date_calendaire['Date'] = pd.to_datetime(df_ml25307_date_calendaire['Date_Heure']).dt.date
df_ml25307_date_calendaire['Time'] = pd.to_datetime(df_ml25307_date_calendaire['Date_Heure']).dt.time
df_ml25307_date_calendaire['Date'] = pd.to_datetime(df_ml25307_date_calendaire['Date'])


# In[10]:


df_ml25307_date_calendaire.sort_values(by='Date_Heure', axis=0, ascending=True, inplace=True)
df_ml25307_date_calendaire.reset_index(inplace=True)


# In[11]:


df_ml25307_date_calendaire


# ### trim classement

# In[12]:


df_ml25307_date_calendaire['Clt'] = df_ml25307_date_calendaire['Clt'].str.strip()


# ## seaborn

# ### boxplot + swarmplot

# In[13]:


import seaborn as sns


df_sous_ensemble = df_ml25307_date_calendaire[df_ml25307_date_calendaire.Date >= '2019-09-10']

sns.set(style="ticks", palette="pastel")

fig_dims = (17, 8)
fig, ax = plt.subplots(figsize=fig_dims)
#hue_order=['D', 'C', 'P 5', 'P 10', 'P 15', 'H'],
g1 = sns.boxplot(x="Date", y="TEMCHIMI", hue="Clt", data=df_sous_ensemble, ax=ax)
g2 = sns.swarmplot(x="Date", y="TEMCHIMI", hue="Clt", data=df_sous_ensemble, ax=ax)
xlabels =[pd.to_datetime(str(x)).strftime("%d-%m") for x in set(df_sous_ensemble['Date'].values)]
g1.set_xticklabels(xlabels, rotation=30)
g2.set_xticklabels(xlabels, rotation=30);


# ### violinplot + swarmplot

# In[14]:


import seaborn as sns


df_sous_ensemble = df_ml25307_date_calendaire[df_ml25307_date_calendaire.Date >= '2019-09-10']
#df_sous_ensemble = df_sous_ensemble[df_sous_ensemble.Clt.isin(['D', 'C'])]

sns.set(style="ticks", palette="pastel")

fig_dims = (17, 8)
fig, ax = plt.subplots(figsize=fig_dims)
#hue_order=['D', 'C', 'P 5', 'P 10', 'P 15', 'H'],
g1 = sns.violinplot(x="Date", y="TEMCHIMI", hue="Clt", data=df_sous_ensemble, ax=ax)
g2 = sns.swarmplot(x="Date", y="TEMCHIMI", hue="Clt", data=df_sous_ensemble, ax=ax)
xlabels =[pd.to_datetime(str(x)).strftime("%d-%m") for x in set(df_sous_ensemble['Date'].values)]
g1.set_xticklabels(xlabels, rotation=30)
g2.set_xticklabels(xlabels, rotation=30);


# ### violinplot with 2 sides

# In[16]:


import seaborn as sns


df_sous_ensemble = df_ml25307_date_calendaire[df_ml25307_date_calendaire.Date >= '2019-09-10']
df_sous_ensemble_DC = df_sous_ensemble[df_sous_ensemble.Clt.isin(['D', 'C'])]

sns.set(style="ticks", palette="pastel")

fig_dims = (17, 8)
fig, ax = plt.subplots(figsize=fig_dims)
#hue_order=['D', 'C', 'P 5', 'P 10', 'P 15', 'H'],
g1 = sns.violinplot(x="Date", y="TEMCHIMI", hue="Clt", split=True, inner="quart",data=df_sous_ensemble_DC, ax=ax, palette='muted')
g2 = sns.swarmplot(x="Date", y="TEMCHIMI", hue="Clt", data=df_sous_ensemble_DC, ax=ax)
xlabels =[pd.to_datetime(str(x)).strftime("%d-%m") for x in set(df_sous_ensemble_DC['Date'].values)]
g1.set_xticklabels(xlabels, rotation=30)
g2.set_xticklabels(xlabels, rotation=30);


# ### violinplot + standard plot of TEMCHIMI

# In[20]:


import seaborn as sns


df_sous_ensemble = df_ml25307_date_calendaire[df_ml25307_date_calendaire.Date >= '2019-09-10']
df_sous_ensemble_DC = df_sous_ensemble[df_sous_ensemble.Clt.isin(['D', 'C'])]

sns.set(style="ticks", palette="pastel")

fig_dims = (17, 8)
fig, ax = plt.subplots(figsize=fig_dims)
#hue_order=['D', 'C', 'P 5', 'P 10', 'P 15', 'H'],
g1 = sns.violinplot(x="Date", y="TEMCHIMI", hue="Clt", split=True, inner="quart",data=df_sous_ensemble_DC, ax=ax, palette='muted')
g2 = sns.swarmplot(x="Date", y="TEMCHIMI", hue="Clt", data=df_sous_ensemble_DC, ax=ax)
xlabels =[pd.to_datetime(str(x)).strftime("%d-%m") for x in set(df_sous_ensemble_DC['Date'].values)]
g1.set_xticklabels(xlabels, rotation=30)
g2.set_xticklabels(xlabels, rotation=30);

ax2 = ax.twinx()
sns.lineplot(x="Date_Heure", y="TEMCHIMI", ax=ax2, data=df_sous_ensemble_DC)
plt.show()


# In[30]:


import seaborn as sns


df_sous_ensemble = df_ml25307_date_calendaire[df_ml25307_date_calendaire.Date >= '2019-09-10']
df_sous_ensemble_DC = df_sous_ensemble[df_sous_ensemble.Clt.isin(['D', 'C'])]

sns.set(style="ticks", palette="pastel")

fig_dims = (17, 8)
fig, ax = plt.subplots(figsize=fig_dims)

sns.lineplot(x="Date_Heure", y="TEMCHIMI", ax=ax, data=df_sous_ensemble_DC)
ax2 = ax.twinx()
g1 = sns.violinplot(x="Date", y="TEMCHIMI", hue="Clt", split=True, inner="quart",data=df_sous_ensemble_DC, ax=ax2, palette='muted')
xlabels =[pd.to_datetime(str(x)).strftime("%d-%m") for x in set(df_sous_ensemble_DC['Date'].values)]
g1.set_xticklabels(xlabels, rotation=30)

plt.show()


# In[ ]:




