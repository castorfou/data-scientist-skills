#!/usr/bin/env python
# coding: utf-8

# # Let’s flip a coin in python

# ## Flipping coins
# This exercise requires the bernoulli object from the scipy.stats library to simulate the two possible outcomes from a coin flip, 1 ("heads") or 0 ("tails"), and the numpy library (loaded as np) to set the random generator seed.
# 
# You'll use the bernoulli.rvs() function to simulate coin flips using the size argument.
# 
# You will set the random seed so you can reproduce the results for the random experiment in each exercise.
# 
# From each experiment, you will get the values of each coin flip. You can add the coin flips to get the number of heads after flipping 10 coins using the sum() function.

# ### code

# In[1]:


import numpy as np


# In[2]:


# Import the bernoulli object from scipy.stats
from scipy.stats import bernoulli

# Set the random seed to reproduce the results
np.random.seed(42)

# Simulate one coin flip with 35% chance of getting heads
coin_flip = bernoulli.rvs(p=0.35, size=1)
print(coin_flip)


# In[3]:


# Simulate ten coin flips and get the number of heads
ten_coin_flips = bernoulli.rvs(p=0.35, size=10)
coin_flips_sum = sum(ten_coin_flips)
print(coin_flips_sum)


# In[5]:


# Simulate ten coin flips and get the number of heads
five_coin_flips = bernoulli.rvs(p=0.5, size=5)
coin_flips_sum = sum(five_coin_flips)
print(coin_flips_sum)


# ## Using binom to flip even more coins
# Previously, you simulated 10 coin flips with a 35% chance of getting heads using bernoulli.rvs().
# 
# This exercise loads the binom object from scipy.stats so you can use binom.rvs() to simulate 20 trials of 10 coin flips with a 35% chance of getting heads on each coin flip.

# ### code

# In[6]:


from scipy.stats import binom


# In[8]:


# Set the random seed to reproduce the results
np.random.seed(42)

# Simulate 20 trials of 10 coin flips 
draws = binom.rvs(n=10, p=0.35, size=20)
print(draws)


# # Probability mass and distribution functions
# 

# ## Predicting the probability of defects
# Any situation with exactly two possible outcomes can be modeled with binomial random variables. For example, you could model if someone likes or dislikes a product, or if they voted or not.
# 
# Let's model whether or not a component from a supplier comes with a defect. From the thousands of components that we got from a supplier, we are going to take a sample of 50, selected randomly. The agreed and accepted defect rate is 2%.
# 
# We import the binom object from scipy.stats.
# 
# Recall that:
# 
# - binom.pmf() calculates the probability of having exactly k heads out of n coin flips.
# - binom.cdf() calculates the probability of having k heads or less out of n coin flips.
# - binom.sf() calculates the probability of having more than k heads out of n coin flips.
# 

# ### code

# In[9]:


from scipy.stats import binom


# Let's answer a simple question before we start calculating probabilities:
# 
# What is the probability of getting more than 20 heads from a fair coin after 30 coin flips?

# In[10]:


binom.sf(k=20, p=0.5, n=30)


# Let's get started with our model for defective components.
# 
# First, calculate the probability of getting exactly 1 defective component.

# In[12]:


# Probability of getting exactly 1 defective component
prob_one_defect = binom.pmf(k=1, n=50, p=0.02)
print(prob_one_defect)


# Next, calculate the probability of not getting any defective components.
# 
# 

# In[13]:


# Probability of not getting any defective components
prob_no_defects = binom.pmf(k=0, n=50, p=0.02)
print(prob_no_defects)


# Now calculate the probability of getting 2 or fewer defective components out of 50.
# 
# 

# In[14]:


# Probability of getting 2 or less defective components
prob_two_or_less_defects = binom.cdf(k=2, n=50, p=0.02)
print(prob_two_or_less_defects)


# ## Predicting employment status
# Consider a survey about employment that contains the question "Are you employed?" It is known that 65% of respondents will answer "yes." Eight survey responses have been collected.
# 
# We load the binom object from scipy.stats with the following code: from scipy.stats import binom
# 
# Answer the following questions using pmf(), cdf(), and sf().

# ### code

# Calculate the probability of getting exactly 5 yes responses.

# In[15]:


# Calculate the probability of getting exactly 5 yes responses
prob_five_yes = binom.pmf(k=5, n=8, p=0.65)
print(prob_five_yes)


# Calculate the probability of getting 3 or fewer no responses.

# In[18]:


# Calculate the probability of getting 3 or less no responses
prob_three_or_less_no = 1-binom.cdf(k=3, n=8, p=0.65)
print(prob_three_or_less_no)


# Calculate the probability of getting more than 3 yes responses.

# In[19]:


# Calculate the probability of getting more than 3 yes responses
prob_more_than_three_yes = binom.sf(k=3, n=8, p=0.65)
print(prob_more_than_three_yes)


# ## Predicting burglary conviction rate
# There are many situations that can be modeled with only two outcomes: success or failure. This exercise presents a situation that can be modeled with a binomial distribution and gives you the opportunity to calculate probabilities using binom.pmf(), binom.cdf(), and binom.sf().
# 
# The binom object from scipy.stats has been loaded for your convenience.
# 
# Imagine that in your town there are many crimes, including burglaries, but only 20% of them get solved. Last week, there were 9 burglaries. Answer the following questions.

# ### code

# What is the probability of solving exactly 4 of the 9 total burglaries?
# 
# 

# In[20]:


# What is the probability of solving 4 burglaries?
four_solved = binom.pmf(k=4, n=9, p=0.2)
print(four_solved)


# What is the probability of solving more than 3 of the 9 burglaries?

# In[21]:


# What is the probability of solving more than 3 burglaries?
more_than_three_solved = binom.sf(k=3, n=9, p=0.2)
print(more_than_three_solved)


# What is the probability of solving exactly 2 or 3 of the 9 burglaries?

# In[23]:


# What is the probability of solving 2 or 3 burglaries?
two_or_three_solved = binom.pmf(k=2, n=9, p=0.2) + binom.pmf(k=3, n=9, p=0.2)
print(two_or_three_solved)


# What is the probability of solving 1 or fewer or more than 7 of the 9 burglaries?

# In[24]:


# What is the probability of solving 1 or fewer or more than 7 burglaries?
tail_probabilities = binom.cdf(k=1, n=9, p=0.2) + binom.sf(k=7, n=9, p=0.2)
print(tail_probabilities)


# # Expected value, mean, and variance
# 

# ## Calculating the sample mean
# Simulation involves generating samples and then measuring. In this exercise, we'll generate some samples and calculate the sample mean with the describe() method. See what you observe about the sample mean as the number of samples increases.
# 
# We've preloaded the binom object and the describe() method from scipy.stats for you, so you can calculate some values.

# ### code

# Generate a sample of 100 fair coin flips using .rvs() and calculate the sample mean using describe().

# In[26]:


from scipy.stats import describe


# In[29]:


# Sample mean from a generated sample of 100 fair coin flips
sample_of_100_flips = binom.rvs(n=1, p=0.5, size=100)
sample_mean_100_flips = describe(sample_of_100_flips).mean
print(sample_mean_100_flips)


# Generate a sample of 1,000 fair coin flips and calculate the sample mean.
# 
# 

# In[30]:


# Sample mean from a generated sample of 1,000 fair coin flips
sample_mean_1000_flips = describe(binom.rvs(n=1, p=0.5, size=1000)).mean
print(sample_mean_1000_flips)


# Generate a sample of 2,000 fair coin flips and calculate the sample mean.

# In[31]:


# Sample mean from a generated sample of 2,000 fair coin flips
sample_mean_2000_flips = describe(binom.rvs(n=1, p=0.5, size=2000)).mean
print(sample_mean_2000_flips)


# ## Checking the result
# Now try generating some samples and calculating the expected value and variance yourself, then using the method provided by binom to check if the sample values match the theoretical values.
# 
# The binom object and describe() method from scipy.stats are already loaded, so you can make the calculations.

# ### code

# In[32]:


sample = binom.rvs(n=10, p=0.3, size=2000)

# Calculate the sample mean and variance from the sample variable
sample_describe = describe(sample)

# Calculate the sample mean using the values of n and p
mean = 10*0.3

# Calculate the sample variance using the value of 1-p
variance = mean*(1-0.3)

# Calculate the sample mean and variance for 10 coin flips with p=0.3
binom_stats = binom.stats(n=10, p=0.3)

print(sample_describe.mean, sample_describe.variance, mean, variance, binom_stats)


# ## Calculating the mean and variance of a sample
# Now that you're familiar with working with coin flips using the binom object and calculating the mean and variance, let's try simulating a larger number of coin flips and calculating the sample mean and variance. Comparing this with the theoretical mean and variance will allow you to check if your simulated data follows the distribution you want.
# 
# We've preloaded the binom object and the describe() method from scipy.stats for you, as well as creating an empty list called averages to store the mean of the sample variable and a variable called variances to store the variance of the sample variable.

# ### code

# In[37]:


averages = []
variances = []


# In[38]:


for i in range(0, 1500):
    # 10 trials of 10 coin flips with 25% probability of heads
    sample = binom.rvs(n=10, p=0.25, size=10)
    # Mean and variance of the values in the sample variable
    averages.append(describe(sample).mean)
    variances.append(describe(sample).variance)


# In[39]:


# Calculate the mean of the averages variable
print("Mean {}".format(describe(averages).mean))

# Calculate the mean of the variances variable
print("Variance {}".format(describe(variances).mean))


# In[40]:


# Calculate the mean and variance
print(binom.stats(n=10, p=0.25))


# In[ ]:




