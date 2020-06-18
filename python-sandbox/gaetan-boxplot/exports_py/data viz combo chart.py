#!/usr/bin/env python
# coding: utf-8

# # Question de gaetan

# https://www.yammer.com/michelin.com/#/Threads/show?threadId=728531149766656

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # 1ere option avec seaborn

# ## chargement des données

# In[34]:


import pandas as pd

filename = 'DATASET_ML25307.xlsx'

df_ml25307 = pd.read_excel(filename, parse_dates=[['Date', 'Heure']])


# In[35]:


df_ml25307.info()


# In[36]:


df_ml25307.Date_Heure.min(), df_ml25307.Date_Heure.max(), 


# In[37]:


df_ml25307['SITE Z'].value_counts()


# In[38]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15,7))
df_ml25307[['Date_Heure', 'TEMCHIMI']].set_index('Date_Heure').plot(ax=ax)
plt.show()


# ### jour Michelin 

# In[39]:


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


# In[40]:


df_ml25307_date_calendaire = dateMichelin_vers_dateCalendaire(df_ml25307, heureEquipeA=6)


# In[41]:


fig, ax = plt.subplots(figsize=(15,7))
df_ml25307_date_calendaire[['Date_Heure', 'TEMCHIMI']].set_index('Date_Heure').plot(ax=ax)
plt.show()


# ### split date et heure

# In[42]:


df_ml25307_date_calendaire['Date'] = pd.to_datetime(df_ml25307_date_calendaire['Date_Heure']).dt.date
df_ml25307_date_calendaire['Time'] = pd.to_datetime(df_ml25307_date_calendaire['Date_Heure']).dt.time
df_ml25307_date_calendaire['Date'] = pd.to_datetime(df_ml25307_date_calendaire['Date'])


# In[43]:


df_ml25307_date_calendaire.sort_values(by='Date_Heure', axis=0, ascending=True, inplace=True)
df_ml25307_date_calendaire.reset_index(inplace=True)


# In[44]:


df_ml25307_date_calendaire


# ### trim classement

# In[45]:


df_ml25307_date_calendaire['Clt'] = df_ml25307_date_calendaire['Clt'].str.strip()


# ## seaborn

# ### boxplot + swarmplot

# In[46]:


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

# In[47]:


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

# In[53]:


import seaborn as sns


df_sous_ensemble = df_ml25307_date_calendaire[df_ml25307_date_calendaire.Date >= '2019-09-10']
df_sous_ensemble_DC = df_sous_ensemble[df_sous_ensemble.Clt.isin(['D', 'C'])]

sns.set(style="ticks", palette="pastel")

fig_dims = (17, 8)
fig, ax = plt.subplots(figsize=fig_dims)
#hue_order=['D', 'C', 'P 5', 'P 10', 'P 15', 'H'],
g1 = sns.violinplot(x="Date", y="TEMCHIMI", hue="Clt", split=True, inner="quart",data=df_sous_ensemble_DC, ax=ax)
g2 = sns.swarmplot(x="Date", y="TEMCHIMI", hue="Clt", data=df_sous_ensemble_DC, ax=ax)
xlabels =[pd.to_datetime(str(x)).strftime("%d-%m") for x in set(df_sous_ensemble_DC['Date'].values)]
g1.set_xticklabels(xlabels, rotation=30)
g2.set_xticklabels(xlabels, rotation=30);


# In[ ]:




