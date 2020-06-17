#!/usr/bin/env python
# coding: utf-8

# # Question de gaetan

# https://www.yammer.com/michelin.com/#/Threads/show?threadId=728531149766656

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # 1ere option avec seaborn

# ## chargement des données

# In[7]:


import pandas as pd

filename = 'DATASET_ML25307.xlsx'

df_ml25307 = pd.read_excel(filename, parse_dates=[['Date', 'Heure']])


# In[8]:


df_ml25307.info()


# In[11]:


df_ml25307.Date_Heure.min(), df_ml25307.Date_Heure.max(), 


# In[6]:


df_ml25307['SITE Z'].value_counts()


# In[15]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15,7))
df_ml25307[['Date_Heure', 'TEMCHIMI']].set_index('Date_Heure').plot(ax=ax)
plt.show()


# ### jour Michelin 

# In[16]:


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


# In[17]:


df_ml25307_date_calendaire = dateMichelin_vers_dateCalendaire(df_ml25307, heureEquipeA=6)


# In[18]:


fig, ax = plt.subplots(figsize=(15,7))
df_ml25307_date_calendaire[['Date_Heure', 'TEMCHIMI']].set_index('Date_Heure').plot(ax=ax)
plt.show()


# ### split date et heure

# In[29]:


df_ml25307_date_calendaire['Date'] = pd.to_datetime(df_ml25307_date_calendaire['Date_Heure']).dt.date
df_ml25307_date_calendaire['Time'] = pd.to_datetime(df_ml25307_date_calendaire['Date_Heure']).dt.time
df_ml25307_date_calendaire['Date'] = pd.to_datetime(df_ml25307_date_calendaire['Date'])


# In[24]:


df_ml25307_date_calendaire.sort_values(by='Date_Heure', axis=0, ascending=True, inplace=True)
df_ml25307_date_calendaire.reset_index(inplace=True)


# In[25]:


df_ml25307_date_calendaire


# ## seaborn

# In[66]:


import seaborn as sns


df_sous_ensemble = df_ml25307_date_calendaire[df_ml25307_date_calendaire.Date >= '2019-09-10']

sns.set(style="ticks", palette="pastel")

fig_dims = (17, 8)
fig, ax = plt.subplots(figsize=fig_dims)
#hue_order=['D', 'C', 'P 5', 'P 10', 'P 15', 'H'],
g = sns.boxplot(x="Date", y="TEMCHIMI", 
            hue="Clt",             data=df_sous_ensemble, ax=ax)
g = sns.swarmplot(x="Date", y="TEMCHIMI", 
            hue="Clt",             data=df_sous_ensemble, ax=ax, color=".25")
#sns.despine(offset=10, trim=True)
#plt.xticks(rotation=45)

xlabels =[pd.to_datetime(str(x)).strftime("%d-%m") for x in set(df_sous_ensemble['Date'].values)]
g.set_xticklabels(xlabels, rotation=30)


# In[ ]:




