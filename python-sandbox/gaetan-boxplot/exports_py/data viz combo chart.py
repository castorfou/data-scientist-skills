#!/usr/bin/env python
# coding: utf-8

# # Question de gaetan

# https://www.yammer.com/michelin.com/#/Threads/show?threadId=728531149766656

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # 1ere option avec seaborn

# ## chargement des données

# In[2]:


import pandas as pd

filename = 'DATASET_ML25307.xlsx'

df_ml25307 = pd.read_excel(filename, parse_dates=[['Date', 'Heure']])


# In[3]:


df_ml25307.info()


# In[4]:


df_ml25307.Date_Heure.min(), df_ml25307.Date_Heure.max(), 


# In[5]:


df_ml25307['SITE Z'].value_counts()


# In[6]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15,7))
df_ml25307[['Date_Heure', 'TEMCHIMI']].set_index('Date_Heure').plot(ax=ax)
plt.show()


# ### jour Michelin 

# In[7]:


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


# In[8]:


df_ml25307_date_calendaire = dateMichelin_vers_dateCalendaire(df_ml25307, heureEquipeA=6)


# In[9]:


fig, ax = plt.subplots(figsize=(15,7))
df_ml25307_date_calendaire[['Date_Heure', 'TEMCHIMI']].set_index('Date_Heure').plot(ax=ax)
plt.show()


# ### split date et heure

# In[10]:


df_ml25307_date_calendaire['Date'] = pd.to_datetime(df_ml25307_date_calendaire['Date_Heure']).dt.date
df_ml25307_date_calendaire['Time'] = pd.to_datetime(df_ml25307_date_calendaire['Date_Heure']).dt.time
df_ml25307_date_calendaire['Date'] = pd.to_datetime(df_ml25307_date_calendaire['Date'])


# In[11]:


df_ml25307_date_calendaire.sort_values(by='Date_Heure', axis=0, ascending=True, inplace=True)
df_ml25307_date_calendaire.reset_index(inplace=True)


# In[12]:


df_ml25307_date_calendaire


# ### trim classement

# In[13]:


df_ml25307_date_calendaire['Clt'] = df_ml25307_date_calendaire['Clt'].str.strip()


# ## seaborn

# ### boxplot + swarmplot

# In[14]:


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

# In[15]:


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


# ### violinplot + standard plot of TEMCHIMI - ne marche pas ;(

# In[17]:


import seaborn as sns


df_sous_ensemble = df_ml25307_date_calendaire[df_ml25307_date_calendaire.Date >= '2019-09-10']
df_sous_ensemble_DC = df_sous_ensemble[df_sous_ensemble.Clt.isin(['D', 'C'])]

sns.set(style="ticks", palette="pastel")

fig_dims = (17, 8)
fig, ax = plt.subplots(figsize=fig_dims)
g1 = sns.violinplot(x="Date", y="TEMCHIMI", hue="Clt", split=True, inner="quart",data=df_sous_ensemble_DC, ax=ax, palette='muted')
g2 = sns.swarmplot(x="Date", y="TEMCHIMI", hue="Clt", data=df_sous_ensemble_DC, ax=ax)
xlabels =[pd.to_datetime(str(x)).strftime("%d-%m") for x in set(df_sous_ensemble_DC['Date'].values)]
g1.set_xticklabels(xlabels, rotation=30)
g2.set_xticklabels(xlabels, rotation=30);

ax2 = ax.twinx()
sns.lineplot(x="Date_Heure", y="TEMCHIMI", ax=ax2, data=df_sous_ensemble_DC)
plt.show()


# ## display dates with missing values

# In[27]:


import seaborn as sns


df_sous_ensemble = df_ml25307_date_calendaire[df_ml25307_date_calendaire.Date >= '2019-09-10']
df_sous_ensemble_DC = df_sous_ensemble[df_sous_ensemble.Clt.isin(['D', 'C'])]

sns.set(style="ticks", palette="pastel")
fig_dims = (17, 8)
fig, ax = plt.subplots(figsize=fig_dims)

g1 = sns.lineplot(x="Date_Heure", y="TEMCHIMI", hue="Clt", data=df_sous_ensemble_DC, ax=ax)


# In[28]:


import seaborn as sns


df_sous_ensemble = df_ml25307_date_calendaire[df_ml25307_date_calendaire.Date >= '2019-09-10']
df_sous_ensemble_DC = df_sous_ensemble[df_sous_ensemble.Clt.isin(['D', 'C'])]

sns.set(style="ticks", palette="pastel")
fig_dims = (17, 8)
fig, ax = plt.subplots(figsize=fig_dims)

g1 = sns.lineplot(x="Date", y="TEMCHIMI", hue="Clt", data=df_sous_ensemble_DC, ax=ax)


# ### bokeh

# In[64]:


from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.palettes import Dark2_5 as palette
from bokeh.transform import factor_cmap, factor_mark

import itertools  

output_notebook()


# In[72]:


TOOLS = "pan,xwheel_zoom,box_zoom,reset,save"
p = figure(plot_width=900, plot_height=600, title="Pour toi Gaetan", x_axis_type='datetime', 
           tools=TOOLS)

df_sous_ensemble = df_ml25307_date_calendaire[df_ml25307_date_calendaire.Date >= '2019-09-10']
df_sous_ensemble_DC = df_sous_ensemble[df_sous_ensemble.Clt.isin(['D', 'C'])]
source = ColumnDataSource(df_sous_ensemble_DC[['Date_Heure', 'TEMCHIMI', 'Clt']])

CLASSTS = ['D', 'C']
MARKERS = ['triangle', 'hex' ]

p.line(x='Date_Heure', y='TEMCHIMI', source=source, color='grey',alpha=0.9, )
p.scatter(x='Date_Heure', y='TEMCHIMI', source=source, legend="Clt", size=10, 
         fill_alpha=0.6, line_color=None, color=factor_cmap('Clt', 'Category10_3', CLASSTS),
        marker=factor_mark('Clt', MARKERS, CLASSTS),)
p.xaxis.axis_label = 'Date Heure de la tombée'
p.yaxis.axis_label = 'TEMPCHIMI'

# create a color iterator
colors = itertools.cycle(palette)    

hover = HoverTool(mode='mouse')
hover.tooltips=[
    ('Date et Heure', '@Date_Heure{%Y-%m-%d %H:%M}'),
    ('TEMCHIMI', '@TEMCHIMI'),
    ('Classement', '@Clt'),                    
]
hover.formatters={'@Date_Heure':'datetime'}

p.add_tools(hover)


show(p) # show the results


# In[ ]:




