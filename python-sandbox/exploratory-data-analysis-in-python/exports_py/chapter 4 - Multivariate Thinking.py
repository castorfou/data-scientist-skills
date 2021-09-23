#!/usr/bin/env python
# coding: utf-8

# # Limits of simple regression
# 
# ```python
# Multiple regression
# import statsmodels.formula.api as smf
# results = smf.ols('INCOME2 ~ _VEGESU1', data=brfss).fit()
# results.params
# ```

# ## Using StatsModels
# > 
# > Let's run the same regression using SciPy and StatsModels, and confirm we get the same results.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(brfss)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'brfss.csv': 'https://file.io/kPaa3aCDYGrm'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
brfss = pd.read_csv(prefix+'brfss.csv',index_col=0)


# ### code

# [Using StatsModels | Python](https://campus.datacamp.com/courses/exploratory-data-analysis-in-python/multivariate-thinking?ex=3)
# 
# > -   Compute the regression of `'_VEGESU1'` as a function of `'INCOME2'` using SciPy's `linregress()`.
# > -   Compute the regression of `'_VEGESU1'` as a function of `'INCOME2'` using StatsModels' `smf.ols()`.

# In[4]:


from scipy.stats import linregress
import statsmodels.formula.api as smf

# Run regression with linregress
subset = brfss.dropna(subset=['INCOME2', '_VEGESU1'])
xs = subset['INCOME2']
ys = subset['_VEGESU1']
res = linregress(xs, ys)
print(res)

# Run regression with StatsModels
results = smf.ols('_VEGESU1 ~ INCOME2', data=brfss).fit()
print(results.params)


# # Multiple regression
# 
# ```python
# 
# # Income and education + age
# gss = pd.read_hdf('gss.hdf5', 'gss')
# results = smf.ols('realinc ~ educ + age', data=gss).fit()
# results.params
# 
# # Income and age
# grouped = gss.groupby('age')
# mean_income_by_age = grouped['realinc'].mean()
# plt.plot(mean_income_by_age, 'o', alpha=0.5)
# plt.xlabel('Age (years)')
# plt.ylabel('Income (1986 \$)')
# 
# # Adding a quadratic term
# gss['age2'] = gss['age']**2
# model = smf.ols('realinc ~ educ + age + age2', data=gss)
# results = model.fit()
# results.params
# 
# ```

# ## Plot income and education
# > 
# > To get a closer look at the relationship between income and education, let's use the variable `'educ'` to group the data, then plot mean income in each group.
# > 
# > Here, the GSS dataset has been pre-loaded into a DataFrame called `gss`.

# ### init

# In[5]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(gss)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'gss.csv': 'https://file.io/6oSBxfKK3qsa'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
gss = pd.read_csv(prefix+'gss.csv',index_col=0)


# ### code

# In[7]:


import matplotlib.pyplot as plt


# In[8]:


# Group by educ
grouped = gss.groupby('educ')

# Compute mean income in each group
mean_income_by_educ = grouped['realinc'].mean()

# Plot mean income as a scatter plot
plt.plot(mean_income_by_educ, 'o', alpha=0.5)

# Label the axes
plt.xlabel('Education (years)')
plt.ylabel('Income (1986 $)')
plt.show()


# ## Non-linear model of education
# > 
# > The graph in the previous exercise suggests that the relationship between income and education is non-linear. So let's try fitting a non-linear model.

# [Non-linear model of education | Python](https://campus.datacamp.com/courses/exploratory-data-analysis-in-python/multivariate-thinking?ex=6)
# 
# > -   Add a column named `'educ2'` to the `gss` DataFrame; it should contain the values from `'educ'` squared.
# > -   Run a regression model that uses `'educ'`, `'educ2'`, `'age'`, and `'age2'` to predict `'realinc'`.

# In[13]:


import statsmodels.formula.api as smf

# Add a new column with educ squared
gss['educ2'] = gss['educ']**2
gss['age2'] = gss['age']**2

# Run a regression model with educ, educ2, age, and age2
results = smf.ols('realinc ~ educ + educ2 +age +age2', data=gss).fit()

# Print the estimated parameters
print(results.params)


# # Visualizing regression results
# 
# ```python
# # Generating predictions
# df = pd.DataFrame()
# df['age'] = np.linspace(18, 85)
# df['age2'] = df['age']**2
# df['educ'] = 12
# df['educ2'] = df['educ']**2
# pred12 = results.predict(df)
# 
# # Plotting predictions
# plt.plot(df['age'], pred12, label='High school')
# plt.plot(mean_income_by_age, 'o', alpha=0.5)
# plt.xlabel('Age (years)')
# plt.ylabel('Income (1986 $)')
# plt.legend()
# ```

# ## Making predictions
# > 
# > At this point, we have a model that predicts income using age, education, and sex.
# > 
# > Let's see what it predicts for different levels of education, holding `age` constant.

# In[14]:


import numpy as np


# In[15]:


# Run a regression model with educ, educ2, age, and age2
results = smf.ols('realinc ~ educ + educ2 + age + age2', data=gss).fit()

# Make the DataFrame
df = pd.DataFrame()
df['educ'] = np.linspace(0,20)
df['age'] = 30
df['educ2'] = df['educ']**2
df['age2'] = df['age']**2

# Generate and plot the predictions
pred = results.predict(df)
print(pred.head())


# ## Visualizing predictions
# > 
# > Now let's visualize the results from the previous exercise!

# In[16]:


# Plot mean income in each age group
plt.clf()
grouped = gss.groupby('educ')
mean_income_by_educ = grouped['realinc'].mean()
plt.plot(mean_income_by_educ, 'o', alpha=0.5)

# Plot the predictions
pred = results.predict(df)
plt.plot(df['educ'], pred, label='Age 30')

# Label axes
plt.xlabel('Education (years)')
plt.ylabel('Income (1986 $)')
plt.legend()
plt.show()


# # Logistic regression
# 
# ```python
# # Sex and income - Categorial variable
# formula = 'realinc ~ educ + educ2 + age + age2 + C(sex)'
# results = smf.ols(formula, data=gss).fit()
# results.params
# 
# # Logistic regression
# formula = 'gunlaw ~ age + age2 + educ + educ2 + C(sex)'
# results = smf.logit(formula, data=gss).fit()
# results.params
# 
# # Generating predictions
# df = pd.DataFrame()
# df['age'] = np.linspace(18, 89)
# df['educ'] = 12
# df['age2'] = df['age']**2
# df['educ2'] = df['educ']**2
# df['sex'] = 1
# pred1 = results.predict(df)
# df['sex'] = 2
# pred2 = results.predict(df)
# 
# # Visualizing results
# grouped = gss.groupby('age')
# favor_by_age = grouped['gunlaw'].mean()
# plt.plot(favor_by_age, 'o', alpha=0.5)
# plt.plot(df['age'], pred1, label='Male')
# plt.plot(df['age'], pred2, label='Female')
# plt.xlabel('Age')
# plt.ylabel('Probability of favoring gun law')
# plt.legend()
# ```

# ## Predicting a binary variable
# > 
# > Let's use logistic regression to predict a binary variable. Specifically, we'll use age, sex, and education level to predict support for legalizing cannabis (marijuana) in the U.S.
# > 
# > In the GSS dataset, the variable `grass` records the answer to the question "Do you think the use of marijuana should be made legal or not?"

# ### init

# In[19]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(gss)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'gss.csv': 'https://file.io/FVsd2jjPcmMg'}}
"""
prefixToc='4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
gss = pd.read_csv(prefix+'gss.csv',index_col=0)


# ### code

# In[22]:


# Recode grass
gss['grass'].replace(2, 0, inplace=True)

# Run logistic regression
results = smf.logit('grass ~ age +age2+ educ +educ2 +C(sex)', data=gss).fit()
results.params


# In[23]:


# Make a DataFrame with a range of ages
df = pd.DataFrame()
df['age'] = np.linspace(18, 89)
df['age2'] = df['age']**2

# Set the education level to 12
df['educ'] = 12
df['educ2'] = 12**2


# In[24]:


# Generate predictions for men and women
df['sex'] = 1
pred1 = results.predict(df)

df['sex'] = 2
pred2 = results.predict(df)


# In[25]:


plt.clf()
grouped = gss.groupby('age')
favor_by_age = grouped.mean()
plt.plot(favor_by_age, 'o', alpha=0.5)

plt.plot(df['age'], pred1, label='Male')
plt.plot(df['age'], pred2, label='Female')

plt.xlabel('Age')
plt.ylabel('Probability of favoring legalization')
plt.legend()
plt.show()


# In[ ]:




