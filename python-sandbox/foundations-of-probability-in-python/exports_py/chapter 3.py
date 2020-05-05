#!/usr/bin/env python
# coding: utf-8

# # Normal distributions
# 

# ## Plotting normal distributions
# A certain restaurant chain has been collecting data about customer spending. The data shows that the spending is approximately normally distributed, with a mean of \\$3.15 and a standard deviation of \\$1.50 per customer.

# ### code

# In[2]:


# Import norm, matplotlib.pyplot, and seaborn
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Create the sample using norm.rvs()
sample = norm.rvs(loc=3.15, scale=1.5, size=10000, random_state=13)

# Plot the sample
sns.distplot(sample)
plt.show()


# # Normal probabilities
# 

# ## Restaurant spending example
# Let's go back to the example of the restaurant chain that has been collecting data about customer spending. Recall that the data shows that the spending is approximately normally distributed, with a mean of 3.15 and a standard deviation of 1.5 per customer, as pictured in the plot.
# 
# ![image.png](attachment:image.png)
# Spending distribution
# 
# We can use the already imported norm object from scipy.stats to answer several questions about customer spending at this restaurant chain.

# ### code

# In[5]:


# Probability of spending $3 or less
spending = norm.cdf(3, loc=3.15, scale=1.5)
print(spending)


# In[6]:


# Probability of spending more than $5
spending = norm.sf(5, loc=3.15, scale=1.5)
print(spending)


# In[7]:


# Probability of spending more than $2.15 and $4.15 or less
spending_4 = norm.cdf(4.15, loc=3.15, scale=1.5)
spending_2 = norm.cdf(2.15, loc=3.15, scale=1.5)
print(spending_4 - spending_2)


# In[8]:


# Probability of spending $2.15 or less or more than $4.15
spending_2 = norm.cdf(2.15, loc=3.15, scale=1.5)
spending_over_4 = norm.sf(4.15, loc=3.15, scale=1.5) 
print(spending_2 + spending_over_4)


# ## Smartphone battery example
# One of the most important things to consider when buying a smartphone is how long the battery will last.
# 
# Suppose the period of time between charges can be modeled with a normal distribution with a mean of 5 hours and a standard deviation of 1.5 hours.
# 
# A friend wants to buy a smartphone and is asking you the following questions.

# ### code

# In[9]:


# Probability that battery will last less than 3 hours
less_than_3h = norm.cdf(3, loc=5, scale=1.5)
print(less_than_3h)


# In[10]:


# Probability that battery will last more than 3 hours
more_than_3h = norm.sf(3, loc=5, scale=1.5)
print(more_than_3h)


# In[11]:


# Probability that battery will last between 5 and 7 hours
P_less_than_7h = norm.cdf(7, loc=5, scale=1.5)
P_less_than_5h = norm.cdf(5, loc=5, scale=1.5)
print(P_less_than_7h - P_less_than_5h)


# ## Adults' heights example
# The heights of adults aged between 18 and 35 years are normally distributed. For males, the mean height is 70 inches with a standard deviation of 4. Adult females have a mean height of 65 inches with a standard deviation of 3.5. You can see how the heights are distributed in this plot:
# 
# ![image.png](attachment:image.png)
# Adults heights distribution for male and female
# 
# Using the previous information, complete the following exercises.
# 
# For your convenience, norm has been imported from the library scipy.stats.

# ### code

# In[16]:


# Values one standard deviation from mean height for females
interval = norm.interval(0.68, loc=65, scale=3.5)
print(interval)


# In[20]:


# Value where the tallest males fall with 0.01 probability
tallest = norm.ppf(0.99, loc=70, scale=4)
print(tallest)


# In[21]:


# Probability of being taller than 73 inches for males and females
P_taller_male = norm.sf(73, loc=70, scale=4)
P_taller_female = norm.sf(73, loc=65, scale=3.5)
print(P_taller_male, P_taller_female)


# In[22]:


# Probability of being shorter than 61 inches for males and females
P_shorter_male = norm.cdf(61, loc=70, scale=4)
P_shorter_female = norm.cdf(61, loc=65, scale=3.5)
print(P_shorter_male, P_shorter_female)


# # Poisson distributions
# 

# ## ATM example
# If you know how many specific events occurred per unit of measure, you can assume that the distribution of the random variable follows a Poisson distribution to study the phenomenon.
# 
# Consider an ATM (automatic teller machine) at a very busy shopping mall. The bank wants to avoid making customers wait in line to use the ATM. It has been observed that the average number of customers making withdrawals between 10:00 a.m. and 10:05 a.m. on any given day is 1.
# 
# As a data analyst at the bank, you are asked what the probability is that the bank will need to install another ATM to handle the load.
# 
# To answer the question, you need to calculate the probability of getting more than one customer during that time period.

# ### code

# In[24]:


# Import poisson from scipy.stats
from scipy.stats import poisson

# Probability of more than 1 customer
probability = poisson.sf(k=1, mu=1)

# Print the result
print(probability)


# ## Highway accidents example
# On a certain turn on a very busy highway, there are 2 accidents per day. Let's assume the number of accidents per day can be modeled as a Poisson random variable and is distributed as in the following plot:
# 
# ![image.png](attachment:image.png)
# Probability mass function
# 
# For your convenience, the poisson object has already been imported from the scipy.stats library.
# 
# Aiming to improve road safety, the transportation agency of the regional government has assigned you the following tasks.

# ### code

# In[26]:


# Import the poisson object
from scipy.stats import poisson

# Probability of 5 accidents any day
P_five_accidents = poisson.pmf(k=5, mu=2)

# Print the result
print(P_five_accidents)


# In[28]:


# Import the poisson object
from scipy.stats import poisson

# Probability of having 4 or 5 accidents on any day
P_less_than_6 = poisson.cdf(k=5, mu=2)
P_less_than_4 = poisson.cdf(k=3, mu=2)

# Print the result
print(P_less_than_6 - P_less_than_4)


# In[29]:


# Import the poisson object
from scipy.stats import poisson

# Probability of more than 3 accidents any day
P_more_than_3 = poisson.sf(k=3, mu=2)

# Print the result
print(P_more_than_3)


# In[31]:


# Import the poisson object
from scipy.stats import poisson

# Number of accidents with 0.75 probability
accidents = poisson.ppf(q=0.75, mu=2)

# Print the result
print(accidents)


# ![image.png](attachment:image.png)

# ## Generating and plotting Poisson distributions
# In the previous exercise, you calculated some probabilities. Now let's plot that distribution.
# 
# Recall that on a certain highway turn, there are 2 accidents per day on average. Assuming the number of accidents per day can be modeled as a Poisson random variable, let's plot the distribution.
# 

# ### code

# In[32]:


# Import poisson, matplotlib.pyplot, and seaborn
from scipy.stats import poisson
import matplotlib.pyplot as plt 
import seaborn as sns

# Create the sample
sample = poisson.rvs(mu=2, size=10000, random_state=13)

# Plot the sample
sns.distplot(sample, kde=False)
plt.show()


# # Geometric distributions
# 

# ## Catching salmon example
# Every fall the salmon run occurs -- this is the time when salmon swim back upriver from the ocean to spawn. While swimming back to the upper river (usually to the place where they were spawned), the fish may encounter grizzly bears. Some of these bears can eat 18 salmon in 3 hours, and they have a 0.0333 probability of success in their attempts to catch a fish.
# 
# ![image.png](attachment:image.png)
# Grizzly bears catching salmons
# 
# We can model a grizzly bear catching salmon with a geometric distribution.
# 
# For the following exercises, the geom object from scipy.stats has already been loaded for your convenience.

# ### code

# In[34]:


from scipy.stats import geom


# In[35]:


# Getting a salmon on the third attempt
probability = geom.pmf(k=3, p=0.0333)

# Print the result
print(probability)


# In[37]:


# Probability of getting a salmon in less than 5 attempts
probability = geom.cdf(k=4, p=0.0333)

# Print the result
print(probability)


# In[38]:


# Probability of getting a salmon in less than 21 attempts
probability = geom.cdf(k=20, p=0.0333)

# Print the result
print(probability)


# In[39]:


# Attempts for 0.9 probability of catching a salmon
attempts = geom.ppf(q=0.9, p=0.0333)

# Print the result
print(attempts)


# ## Free throws example
# Suppose you know that a basketball player has a 0.3 probability of scoring a free throw. What is the probability of them missing with the first throw and scoring with the second?

# ### code

# In[40]:


# Import geom from scipy.stats
from scipy.stats import geom

# Probability of missing first and scoring on second throw
probability = geom.pmf(k=2, p=0.3)

# Print the result
print(probability)


# ## Generating and plotting geometric distributions
# In sports it is common for players to make multiple attempts to score points for themselves or their teams. Each single attempt can have two possible outcomes, scoring or not scoring. Those situations can be modeled with geometric distributions. With scipy.stats you can generate samples using the rvs() function for each distribution.
# 
# Consider the previous example of a basketball player who scores free throws with a probability of 0.3. Generate a sample, and plot it.
# 
# numpy has been imported for you with the standard alias np.

# ### code

# In[41]:


import numpy as np


# In[42]:


# Import geom, matplotlib.pyplot, and seaborn
from scipy.stats import geom
import matplotlib.pyplot as plt
import seaborn as sns

# Create the sample
sample = geom.rvs(p=0.3, size=10000, random_state=13)

# Plot the sample
sns.distplot(sample, bins = np.linspace(0,20,21), kde=False)
plt.show()


# In[ ]:




