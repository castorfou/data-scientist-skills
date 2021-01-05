#!/usr/bin/env python
# coding: utf-8

# # The normal distribution
# 
# ![image.png](attachment:image.png)
# 
# ![image-2.png](attachment:image-2.png)

# [Distribution of Amir's sales | Python](https://campus.datacamp.com/courses/introduction-to-statistics-in-python/more-distributions-and-the-central-limit-theorem-3?ex=2)
# 
# > ## Distribution of Amir's sales
# > 
# > Since each deal Amir worked on (both won and lost) was different, each was worth a different amount of money. These values are stored in the `amount` column of `amir_deals` As part of Amir's performance review, you want to be able to estimate the probability of him selling different amounts, but before you can do this, you'll need to determine what kind of distribution the `amount` variable follows.
# > 
# > Both `pandas` as `pd` and `matplotlib.pyplot` as `plt` are loaded and `amir_deals` is available.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(amir_deals)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'amir_deals.csv': 'https://file.io/7Mntc9wVbV7P'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
amir_deals = pd.read_csv(prefix+'amir_deals.csv',index_col=0)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### code
# 

# > Create a histogram with 10 bins to visualize the distribution of the `amount`. Show the plot.

# In[3]:


# Histogram of amount with 10 bins and show plot
amir_deals['amount'].hist(bins=10)
plt.show()


# [Probabilities from the normal distribution | Python](https://campus.datacamp.com/courses/introduction-to-statistics-in-python/more-distributions-and-the-central-limit-theorem-3?ex=3)
# 
# > ## Probabilities from the normal distribution
# > 
# > Since each deal Amir worked on (both won and lost) was different, each was worth a different amount of money. These values are stored in the `amount` column of `amir_deals` and follow a normal distribution with a mean of 5000 dollars and a standard deviation of 2000 dollars. As part of his performance metrics, you want to calculate the probability of Amir closing a deal worth various amounts.
# > 
# > `norm` from `scipy.stats` is imported as well as `pandas` as `pd`. The DataFrame `amir_deals` is loaded.

# In[5]:


from scipy.stats import norm


# > What's the probability of Amir closing a deal worth less than $7500?

# In[6]:


# Probability of deal < 7500
prob_less_7500 = norm.cdf(7500, 5000, 2000)

print(prob_less_7500)


# > What's the probability of Amir closing a deal worth more than $1000?

# In[7]:


# Probability of deal > 1000
prob_over_1000 = 1-norm.cdf(1000, 5000, 2000)

print(prob_over_1000)


# > What's the probability of Amir closing a deal worth between \\$3000 and \\$7000?

# In[8]:


# Probability of deal between 3000 and 7000
prob_3000_to_7000 = norm.cdf(7000, 5000, 2000) - norm.cdf(3000, 5000, 2000)

print(prob_3000_to_7000)


# > What amount will 25% of Amir's sales be _less than_?

# In[9]:


# Calculate amount that 25% of deals will be less than
pct_25 = norm.ppf(0.25, 5000, 2000)

print(pct_25)


# [Simulating sales under new market conditions | Python](https://campus.datacamp.com/courses/introduction-to-statistics-in-python/more-distributions-and-the-central-limit-theorem-3?ex=4)
# 
# > ## Simulating sales under new market conditions
# > 
# > The company's financial analyst is predicting that next quarter, the worth of each sale will increase by 20% and the volatility, or standard deviation, of each sale's worth will increase by 30%. To see what Amir's sales might look like next quarter under these new market conditions, you'll simulate new sales amounts using the normal distribution and store these in the `new_sales` DataFrame, which has already been created for you.
# > 
# > In addition, `norm` from `scipy.stats`, `pandas` as `pd`, and `matplotlib.pyplot` as `plt` are loaded.

# > -   Currently, Amir's average sale amount is \\$5000. Calculate what his new average amount will be if it increases by 20% and store this in `new_mean`.
# > -   Amir's current standard deviation is \\$2000. Calculate what his new standard deviation will be if it increases by 30% and store this in `new_sd`.
# > -   Create a variable called `new_sales`, which contains 36 simulated amounts from a normal distribution with a mean of `new_mean` and a standard deviation of `new_sd`.
# > -   Plot the distribution of the `new_sales` `amount`s using a histogram and show the plot.

# In[10]:


# Calculate new average amount
new_mean = 5000*1.2

# Calculate new standard deviation
new_sd = 2000*1.3

# Simulate 36 new sales
new_sales = norm.rvs(new_mean, new_sd, size=36)

# Create histogram and show
plt.hist(new_sales)
plt.show()


# [Which market is better? | Python](https://campus.datacamp.com/courses/introduction-to-statistics-in-python/more-distributions-and-the-central-limit-theorem-3?ex=5)
# 
# > ## Which market is better?
# > 
# > The key metric that the company uses to evaluate salespeople is the percent of sales they make over \\$1000 since the time put into each sale is usually worth a bit more than that, so the higher this metric, the better the salesperson is performing.
# > 
# > Recall that Amir's current sales amounts have a mean of \\$5000 and a standard deviation of \\$2000, and Amir's predicted amounts in next quarter's market have a mean of \\$6000 and a standard deviation of \\$2600.
# > 
# > `norm` from `scipy.stats` is imported.
# > 
# > Based **_only_** on the metric of **percent of sales over \\$1000**, does Amir perform better in the current market or the predicted market?

# In[11]:


previous_quarter = norm.rvs(5000, 2000, size=10000)
next_quarter = norm.rvs(6000, 2600, size=10000)


# In[15]:


np.mean(previous_quarter[previous_quarter>1000])


# In[16]:


np.mean(next_quarter[next_quarter>1000])


# In[19]:


#probability of sales over $1000
1-norm.cdf(1000, 5000, 2000)


# In[20]:


#probability of sales over $1000
1-norm.cdf(1000, 6000, 2600)


# ![image.png](attachment:image.png)

# # The central limit theorem
# 
# ![image.png](attachment:image.png)

# [The CLT in action | Python](https://campus.datacamp.com/courses/introduction-to-statistics-in-python/more-distributions-and-the-central-limit-theorem-3?ex=8)
# 
# > ## The CLT in action
# > 
# > The central limit theorem states that a sampling distribution of a sample statistic approaches the normal distribution as you take more samples, no matter the original distribution being sampled from.
# > 
# > In this exercise, you'll focus on the sample mean and see the central limit theorem in action while examining the `num_users` column of `amir_deals` more closely, which contains the number of people who intend to use the product Amir is selling.
# > 
# > `pandas` as `pd`, `numpy` as `np`, and `matplotlib.pyplot` as `plt` are loaded and `amir_deals` is available.

# > Create a histogram of the `num_users` column of `amir_deals` and show the plot.

# In[22]:


# Create a histogram of num_users and show
amir_deals.num_users.hist()
plt.show()


# > -   Set the seed to `104`.
# > -   Take a sample of size `20` with replacement from the `num_users` column of `amir_deals`, and take the mean.

# In[24]:


# Set seed to 104
np.random.seed(104)

# Sample 20 num_users with replacement from amir_deals
samp_20 = amir_deals['num_users'].sample(n=20, replace=True)

# Take mean of samp_20
print(np.mean(samp_20))


# > Repeat this 100 times using a `for` loop and store as `sample_means`. This will take 100 different samples and calculate the mean of each.

# In[25]:


sample_means = []
# Loop 100 times
for i in range(100):
  # Take sample of 20 num_users
  samp_20 = amir_deals['num_users'].sample(n=20, replace=True)
  # Calculate mean of samp_20
  samp_20_mean = np.mean(samp_20)
  # Append samp_20_mean to sample_means
  sample_means.append(samp_20_mean)
  
print(sample_means)


# > Convert `sample_means` into a `pd.Series`, create a histogram of the `sample_means`, and show the plot.

# In[26]:


# Convert to Series and plot histogram
sample_means_series = pd.Series(sample_means)
sample_means_series.hist()
# Show plot
plt.show()


# [The mean of means | Python](https://campus.datacamp.com/courses/introduction-to-statistics-in-python/more-distributions-and-the-central-limit-theorem-3?ex=9)
# 
# > ## The mean of means
# > 
# > You want to know what the average number of users (`num_users`) is per deal, but you want to know this number for the entire company so that you can see if Amir's deals have more or fewer users than the company's average deal. The problem is that over the past year, the company has worked on more than ten thousand deals, so it's not realistic to compile all the data. Instead, you'll estimate the mean by taking several random samples of deals, since this is much easier than collecting data from everyone in the company.
# > 
# > `amir_deals` is available and the user data for all the company's deals is available in `all_deals`. Both `pandas` as `pd` and `numpy` as `np` are loaded.

# ### init

# In[27]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(all_deals)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'all_deals.csv': 'https://file.io/P8gmG2I3ovIe'}}
"""
prefixToc='2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
all_deals = pd.read_csv(prefix+'all_deals.csv',index_col=0)


# ### code

# > -   Set the random seed to `321`.
# > -   Take 30 samples (with replacement) of size 20 from `all_deals['num_users']` and take the mean of each sample. Store the sample means in `sample_means`.
# > -   Print the mean of `sample_means`.
# > -   Print the mean of the `num_users` column of `amir_deals`.

# In[29]:


# Set seed to 321
np.random.seed(321)

sample_means = []
# Loop 30 times to take 30 means
for i in range(30):
  # Take sample of size 20 from num_users col of all_deals with replacement
  cur_sample = all_deals['num_users'].sample(n=20, replace=True)
  # Take mean of cur_sample
  cur_mean = np.mean(cur_sample)
  # Append cur_mean to sample_means
  sample_means.append(cur_mean)

# Print mean of sample_means
print(np.mean(sample_means))

# Print mean of num_users in amir_deals
print(np.mean(amir_deals['num_users']))


# # The Poisson distribution
# 
# ![image.png](attachment:image.png)
# 
# ![image-2.png](attachment:image-2.png)

# [Tracking lead responses | Python](https://campus.datacamp.com/courses/introduction-to-statistics-in-python/more-distributions-and-the-central-limit-theorem-3?ex=12)
# 
# > ## Tracking lead responses
# > 
# > Your company uses sales software to keep track of new sales leads. It organizes them into a queue so that anyone can follow up on one when they have a bit of free time. Since the number of lead responses is a countable outcome over a period of time, this scenario corresponds to a Poisson distribution. On average, Amir responds to 4 leads each day. In this exercise, you'll calculate probabilities of Amir responding to different numbers of leads.

# > Import `poisson` from `scipy.stats` and calculate the probability that Amir responds to 5 leads in a day, given that he responds to an average of 4.

# In[30]:


# Import poisson from scipy.stats
from scipy.stats import poisson

# Probability of 5 responses
prob_5 = poisson.pmf(5,4)

print(prob_5)


# > Amir's coworker responds to an average of 5.5 leads per day. What is the probability that she answers 5 leads in a day?

# In[33]:


# Probability of 5 responses
prob_coworker = poisson.pmf(5, 5.5)

print(prob_coworker)


# > What's the probability that Amir responds to 2 or fewer leads in a day?

# In[35]:


# Probability of 2 or fewer responses
prob_2_or_less = poisson.cdf(2, 4)

print(prob_2_or_less)


# > What's the probability that Amir responds to more than 10 leads in a day?

# In[36]:


# Probability of > 10 responses
prob_over_10 = 1-poisson.cdf(10, 4)

print(prob_over_10)


# # More probability distributions
# 
# ![image-2.png](attachment:image-2.png)
# 
# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# [Modeling time between leads | Python](https://campus.datacamp.com/courses/introduction-to-statistics-in-python/more-distributions-and-the-central-limit-theorem-3?ex=15)
# 
# > ## Modeling time between leads
# > 
# > To further evaluate Amir's performance, you want to know how much time it takes him to respond to a lead after he opens it. On average, it takes 2.5 hours for him to respond. In this exercise, you'll calculate probabilities of different amounts of time passing between Amir receiving a lead and sending a response.

# > Import `expon` from `scipy.stats`. What's the probability it takes Amir less than an hour to respond to a lead?

# In[37]:


# Import expon from scipy.stats
from scipy.stats import expon

# Print probability response takes < 1 hour
print(expon.cdf(1, scale=2.5))


# > What's the probability it takes Amir more than 4 hours to respond to a lead?

# In[40]:


# Print probability response takes > 4 hours
print(1-expon.cdf(4, scale=2.5))


# > -   What's the probability it takes Amir 3-4 hours to respond to a lead?

# In[41]:


# Print probability response takes 3-4 hours
print(expon.cdf(4, scale=2.5) - expon.cdf(3, scale=2.5))


# In[ ]:




