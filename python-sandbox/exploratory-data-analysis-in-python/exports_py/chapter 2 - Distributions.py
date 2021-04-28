#!/usr/bin/env python
# coding: utf-8

# # Probability mass functions
# 
# ```python
# # PMF
# pmf_educ = Pmf(educ, normalize=False)
# pmf_educ.head()
# 
# # PMF barchart
# pmf_educ.bar(label='educ')
# plt.xlabel('Years of education')
# plt.ylabel('PMF')
# plt.show()
# ```

# ## Make a PMF
# > 
# > The GSS dataset has been pre-loaded for you into a DataFrame called `gss`. You can explore it in the IPython Shell to get familiar with it.
# > 
# > In this exercise, you'll focus on one variable in this dataset, `'year'`, which represents the year each respondent was interviewed.
# > 
# > The `Pmf` class you saw in the video has already been created for you. You can access it outside of DataCamp via the [`empiricaldist`](https://pypi.org/project/empiricaldist/) library.

# ### init

# In[8]:


from empiricaldist import Pmf


# In[17]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(gss)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'gss.csv': 'https://file.io/s6LCYCI931wu'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
gss = pd.read_csv(prefix+'gss.csv',index_col=0)


# ### code

# [Make a PMF | Python](https://campus.datacamp.com/courses/exploratory-data-analysis-in-python/distributions?ex=2)
# 
# > Make a PMF for `year` with `normalize=False` and display the result.

# In[22]:


# Compute the PMF for year
pmf_year = Pmf(gss['year'], normalize=False)

# Print the result
print(pmf_year)


# In[4]:


get_ipython().run_line_magic('pinfo2', 'Pmf')


# ## Plot a PMF
# > 
# > Now let's plot a PMF for the age of the respondents in the GSS dataset. The variable `'age'` contains respondents' age in years.

# In[5]:


import matplotlib.pyplot as plt


# In[ ]:


# Select the age column
age = gss['age']

# Make a PMF of age
pmf_age = Pmf(age)

# Plot the PMF
pmf_age.bar()

# Label the axes
plt.xlabel('Age')
plt.ylabel('PMF')
plt.show()


# # Cumulative distribution functions
# 
# ```python
# # CDF
# cdf = Cdf(gss['age'])
# cdf.plot()
# plt.xlabel('Age')
# plt.ylabel('CDF')
# plt.show()
# 
# # Evaluating the CDF
# q = 51
# p = cdf(q)
# print(p)
# 
# # Evaluating the inverse CDF
# p = 0.25
# q = cdf.inverse(p)
# print(q)
# ```

# ## Make a CDF
# > 
# > In this exercise, you'll make a CDF and use it to determine the fraction of respondents in the GSS dataset who are OLDER than 30.
# > 
# > The GSS dataset has been preloaded for you into a DataFrame called `gss`.
# > 
# > As with the `Pmf` class from the previous lesson, the `Cdf` class you just saw in the video has been created for you, and you can access it outside of DataCamp via the [`empiricaldist`](https://pypi.org/project/empiricaldist/) library.

# In[7]:


# Select the age column
age = gss['age']


# In[10]:


from empiricaldist import Cdf

# Compute the CDF of age
cdf_age = Cdf(age)

# Calculate the CDF of 30
print(cdf_age(30))


# ## Compute IQR
# > 
# > Recall from the video that the interquartile range (IQR) is the difference between the 75th and 25th percentiles. It is a measure of variability that is robust in the presence of errors or extreme values.
# > 
# > In this exercise, you'll compute the interquartile range of income in the GSS dataset. Income is stored in the `'realinc'` column, and the CDF of income has already been computed and stored in `cdf_income`.

# [Compute IQR | Python](https://campus.datacamp.com/courses/exploratory-data-analysis-in-python/distributions?ex=6)
# 
# > Calculate the 75th percentile of income and store it in `percentile_75th`.

# In[18]:


realinc=gss['realinc']
cdf_income=Cdf(realinc)

percentile_75th = cdf_income.inverse(0.75)
percentile_75th


# In[19]:


percentile_25th = cdf_income.inverse(0.25)
percentile_25th


# In[20]:



# Calculate the interquartile range
iqr = percentile_75th - percentile_25th

# Print the interquartile range
print(iqr)


# # Comparing distributions

# # Modeling distributions
# 
# ```python
# # The normal distribution
# sample = np.random.normal(size=1000)
# Cdf(sample).plot()
# 
# # The normal CDF
# from scipy.stats import norm
# xs = np.linspace(-3, 3)
# ys = norm(0, 1).cdf(xs)
# plt.plot(xs, ys, color='gray')
# Cdf(sample).plot()
# 
# # The bell curve
# xs = np.linspace(-3, 3)
# ys = norm(0,1).pdf(xs)
# plt.plot(xs, ys, color='gray')
# 
# # KDE plot
# import seaborn as sns
# sns.kdeplot(sample)
# 
# 
# ```
# 
# PMF, CDF, KDE
# Use CDFs for exploration.
# Use PMFs if there are a small number of unique values.
# Use KDE if there are a lot of values.

# ## Distribution of income
# > 
# > In many datasets, the distribution of income is approximately lognormal, which means that the logarithms of the incomes fit a normal distribution. We'll see whether that's true for the GSS data. As a first step, you'll compute the mean and standard deviation of the log of incomes using NumPy's `np.log10()` function.
# > 
# > Then, you'll use the computed mean and standard deviation to make a `norm` object using the [`scipy.stats.norm()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html) function.

# In[24]:


import numpy as np

# Extract realinc and compute its log
income = gss['realinc']
log_income = np.log10(income)

# Compute mean and standard deviation
mean = log_income.mean()
std = log_income.std()
print(mean, std)

# Make a norm object
from scipy.stats import norm
dist = norm(mean,std)


# ## Comparing CDFs
# > 
# > To see whether the distribution of income is well modeled by a lognormal distribution, we'll compare the CDF of the logarithm of the data to a normal distribution with the same mean and standard deviation. These variables from the previous exercise are available for use:
# > 
# >     # Extract realinc and compute its log
# >     log_income = np.log10(gss['realinc'])
# >     
# >     # Compute mean and standard deviation
# >     mean, std = log_income.mean(), log_income.std()
# >     
# >     # Make a norm object
# >     from scipy.stats import norm
# >     dist = norm(mean, std)
# >     
# > 
# > `dist` is a `scipy.stats.norm` object with the same mean and standard deviation as the data. It provides `.cdf()`, which evaluates the normal cumulative distribution function.
# > 
# > Be careful with capitalization: `Cdf()`, with an uppercase `C`, creates `Cdf` objects. `dist.cdf()`, with a lowercase `c`, evaluates the normal cumulative distribution function.

# In[25]:


# Evaluate the model CDF
xs = np.linspace(2, 5.5)
ys = dist.cdf(xs)

# Plot the model CDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Create and plot the Cdf of log_income
Cdf(log_income).plot()
    
# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('CDF')
plt.show()


# ## Comparing PDFs
# > 
# > In the previous exercise, we used CDFs to see if the distribution of income is lognormal. We can make the same comparison using a PDF and KDE. That's what you'll do in this exercise!
# > 
# > As before, the `norm` object `dist` is available in your workspace:
# > 
# >     from scipy.stats import norm
# >     dist = norm(mean, std)
# >     
# > 
# > Just as all `norm` objects have a `.cdf()` method, they also have a `.pdf()` method.
# > 
# > To create a KDE plot, you can use Seaborn's [`kdeplot()`](https://seaborn.pydata.org/generated/seaborn.kdeplot.html) function. To learn more about this function and Seaborn, you can check out DataCamp's [Data Visualization with Seaborn](https://www.datacamp.com/courses/data-visualization-with-seaborn) course. Here, Seaborn has been imported for you as `sns`.

# In[32]:


from scipy.stats import norm
dist = norm(mean, std)
import seaborn as sns


# In[34]:


# Evaluate the normal PDF
xs = np.linspace(2, 5.5)
ys = dist.pdf(xs)

# Plot the model PDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Plot the data KDE
sns.kdeplot(log_income)

# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('PDF')
plt.show()


# In[ ]:




