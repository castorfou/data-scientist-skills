#!/usr/bin/env python
# coding: utf-8

# # Who is Bayes? What is Bayes?

# ```python
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.kdeplot(draws, shade=True)
# plt.show()
# ```

# ## Bayesians vs. Frequentists
# > 
# > The Bayesian approach is a different way to look at statistical inference than what is typically taught in STATS101 classes. The latter is known as frequentist or classical statistics and is quite different from the Bayesian approach.
# > 
# > Let's see if you recognize the differences between these two worlds!

# ![image.png](attachment:image.png)

# ## Probability distributions
# > 
# > Well done on the previous exercise! Now you have the general idea of what the Bayesian approach is all about. Among other things, you know that for a Bayesian, parameters of statistical models are random variables which can be described by probability distributions.
# > 
# > This exercise will test your ability to visualize and interpret probability distributions. You have been given a long list of draws from a distribution of the heights of plants in centimeters, contained in the variable `draws`. `seaborn` and `matplotlib.pyplot` have been imported for you as `sns` and `plt`, respectively. Time to get your hands dirty with data!

# ### init

# In[1]:


###################
##### liste de nombres (list)
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(draws)
"""

tobedownloaded="""
{list: {'draws.txt': 'https://file.io/Cf41p9Vnm2cD'}}
"""
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
from downloadfromFileIO import loadListFromTxt
draws = loadListFromTxt(prefix+'draws.txt')


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt


# ### code

# [Probability distributions | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=3)
# 
# > -   Print the list of `draws`.
# > -   Print the length of `draws`.

# In[4]:


# Print the list of draws
print(draws)

# Print the length of draws
print(len(draws))


# [Probability distributions | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=3)
# 
# > Draw a density plot of `draws`, shading the distribution.

# In[5]:


# Plot the density of draws
sns.kdeplot(draws, shade=True)
plt.show()


# # Probability and Bayes' Theorem

# $$P(A|B) = \frac{P(B|A)*P(A)}{P(B)}$$

# ## Let's play cards
# > 
# > You have a regular deck of 52 well-shuffled playing cards. The deck consists of 4 suits, and there are 13 cards in each suite: ranks 2 through 10, a jack, a queen, a king, and an ace. This means that in the whole deck of 52, there are four of each distinct rank: four aces, four kings, four tens, four fives, etc.
# > 
# > Since there are 52 distinct cards, the probability of drawing any one particular card is 1/52. Using the two rules of probability you've learned about in the last video, calculate the probabilities of drawing some specific combinations of cards, as described in the instructions.

# [Let's play cards | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=5)
# 
# > Calculate the probability of drawing a king or a queen, assign it to the variable `p_king_or_queen` and print it.

# In[6]:


# Calculate probability of drawing a king or queen
p_king_or_queen = 1/13+1/13
print(p_king_or_queen)


# [Let's play cards | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=5)
# 
# > Calculate the probability of drawing a numbered rank lesser than or equal to 5, assign it to the variable `p_five_or_less` and print it.

# In[7]:


# Calculate probability of drawing <= 5
p_five_or_less = 4*(1/13)
print(p_five_or_less)


# [Let's play cards | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=5)
# 
# > Calculate the probability of drawing all four aces in a row, assign it to the variable `p_four_aces`and print it.

# In[8]:


# Calculate probability of drawing four aces
p_all_four_aces = (4/52)*(3/51)*(2/50)*(1/49)
print(p_all_four_aces)


# ## Bayesian spam filter
# > 
# > Well done on the previous exercise! Let's now tackle the famous Bayes' Theorem and use it for a simple but important task: spam detection.
# > 
# > While browsing your inbox, you have figured out that quite a few of the emails you would rather not waste your time on reading contain exclamatory statements, such as "BUY NOW!!!". You start thinking that the presence of three exclamation marks next to each other might be a good spam predictor! Hence you've prepared a DataFrame called `emails` with two variables: `spam`, whether the email was spam, and `contains_3_exlc`, whether it contains the string "!!!". The head of the data looks like this:
# > 
# >          spam    contains_3_excl
# >     0    False             False
# >     1    False             False
# >     2    True              False
# >     3    False             False
# >     4    False             False
# >     
# > 
# > Your job is to calculate the probability of the email being spam given that it contains three exclamation marks. Let's tackle it step by step! Here is Bayes' formula for your reference:
# 
# $$P(A|B) = \frac{P(B|A)*P(A)}{P(B)}$$
# 
# 
# $$P(accident|slippery) = \frac{P(slippery|accident)*P(accident)}{P(slippery)}$$
# 
# ```python
# # Unconditional probability of an accident
# p_accident = road_conditions["accident"].mean()
# # 0.0625
# # Unconditional probability of the road being slippery
# p_slippery = road_conditions["slippery"].mean()
# # 0.0892
# # Probability of the road being slippery given there is an accident
# p_slippery_given_accident = road_conditions.loc[road_conditions["accident"]]["slippery"].mean()
# # 0.7142
# # Probability of an accident given the road is slippery
# p_accident_given_slippery = p_slippery_given_accident * p_accident / p_slippery
# # 0.5
# ```

# ### init

# In[9]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(emails)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'emails.csv': 'https://file.io/4zp54jBonEyM'}}
"""
prefixToc='2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
emails = pd.read_csv(prefix+'emails.csv',index_col=0)


# ### code

# [Bayesian spam filter | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=6)
# 
# > Calculate the unconditional probability of the email being spam, assign it to the variable `p_spam` and print it.

# In[14]:


# Calculate and print the unconditional probability of spam
p_spam = emails['spam'].mean()
print(p_spam)


# [Bayesian spam filter | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=6)
# 
# > Calculate the unconditional probability of the email containing "!!!", assign it to the variable `p_3_excl` and print it.

# In[16]:


# Calculate and print the unconditional probability of "!!!"
p_3_excl = emails['contains_3_excl'].mean()
print(p_3_excl)


# [Bayesian spam filter | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=6)
# 
# > Calculate the probability of the email containing "!!!" given that it is spam, assign it to the variable `p_3_excl_given_spam` and print it.

# In[17]:


# Calculate and print the probability of "!!!" given spam
p_3_excl_given_spam = emails.loc[emails['spam']]['contains_3_excl'].mean()
print(p_3_excl_given_spam)


# [Bayesian spam filter | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=6)
# 
# > Calculate the probability of the email being spam given that it contains "!!!", assign it to the variable `p_spam_given_3_excl` and print it.

# In[19]:


# Calculate and print the probability of spam given "!!!"
p_spam_given_3_excl = p_3_excl_given_spam * p_spam / p_3_excl
print(p_spam_given_3_excl)


# $$P(spam|!!!) = \frac{P(!!!|spam)*P(spam)}{P(!!!)}$$
# 

# ## What does the test say?
# > 
# > A doctor suspects a disease in their patient, so they run a medical test. The test's manufacturer claims that 99% of sick patients test positive, while the doctor has observed that the test comes back positive in 2% of all cases. The suspected disease is quite rare: only 1 in 1000 people suffer from it.
# > 
# > The test result came back positive. **What is the probability that the patient is indeed sick?** You can use Bayes' Theorem to answer this question. Here is what you should calculate:
# 
# $$P(sick|positive) = \frac{P(positive|sick)*P(sick)}{P(positive)}$$
#  
# > Feel free to do the calculations in the console.

# In[21]:


p_positive_when_sick = 0.99
p_positive = 0.02
p_sick = 1/1000
p_sick_when_positive = p_positive_when_sick * p_sick / p_positive
p_sick_when_positive


# # Tasting the Bayes
# 
# ```python
# # Binomial distribution in Python
# 
# # Number of successes in 100 trials:
# import numpy as np 
# np.random.binomial(100, 0.5) # 51
# 
# # Get draws from a binomial:
# import numpy as np
# np.random.binomial(1, 0.5, size=5) # array([1, 0, 0, 1, 1])
# 
# ```

# ## Tossing a coin
# > 
# > In the video, you have seen our custom `get_heads_prob()` function that estimates the probability of success of a binomial distribution. In this exercise, you will use it yourself and verify whether it does its job well in a coin-flipping experiment.
# > 
# > Watch out for the confusion: there are two different probability distributions involved! One is the binomial, which we use to model the coin-flipping. It's a discrete distribution with two possible values (heads or tails) parametrized with the probability of success (tossing heads). The Bayesian estimate of this parameter is another, continuous probability distribution. We don't know what kind of distribution it is, but we can estimate it with `get_heads_prob()` and visualize it.
# > 
# > `numpy` and `seaborn` have been imported for you as `np` and `sns`, respectively.

# ### init

# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[23]:


###################
##### inspect Function
###################

""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(get_heads_prob)
"""
def get_heads_prob(tosses):
    num_heads = np.sum(tosses)
    return np.random.beta(num_heads + 1, 
                          len(tosses) - num_heads + 1, 
                          1000)


# ### code

# [Tossing a coin | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=9)
# 
# > -   Generate a list of 1000 coin tosses (0s and 1s) with 50% chance of tossing heads, and assign to the variable `tosses`.
# > -   Use the `tosses` and the `get_heads_prob()` function to estimate the heads probability, and assign the result to `heads_prob`.
# > -   Draw a density plot of the distribution of the heads probability you have just estimated.

# In[26]:


# Generate 1000 coin tosses
tosses = np.random.binomial(1, .5, 1000)

# Estimate the heads probability
heads_prob = get_heads_prob(tosses)

# Plot the distribution of heads probability
sns.kdeplot(heads_prob, shade=True, label="heads probabilty")
plt.show()


# ## The more you toss, the more you learn
# > 
# > Imagine you are a frequentist (just for a day), and you've been tasked with estimating the probability of tossing heads with a (possibly biased) coin, but without observing any tosses. What would you say? It's impossible, there is no data! Then, you are allowed to flip the coin once. You get tails. What do you say now? Well, if that's all your data, you'd say the heads probability is 0%.
# > 
# > You can probably feel deep inside that these answers are not the best ones. But what would be better? What would a Bayesian say? Let's find out! `numpy` and `seaborn` have been imported for you as `np` and `sns`, respectively.

# [The more you toss, the more you learn | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=10)
# 
# > Estimate the heads probability using `get_heads_prob()` based on an empty list, assign the result to `heads_prob_nodata` and visualize it on a density plot.

# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[32]:


# Estimate and plot heads probability based on no data
heads_prob_nodata = get_heads_prob([])
sns.kdeplot(heads_prob_nodata, shade=True, label="no data")
plt.show()


# [The more you toss, the more you learn | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=10)
# 
# > Do the same, this time based on a single tails (a list with a single `0`) and assigning the result to `heads_prob_onetails`.

# In[33]:


# Estimate and plot heads probability based on a single tails
heads_prob_onetails = get_heads_prob([0])
sns.kdeplot(heads_prob_onetails, shade=True, label="single tails")
plt.show()


# [The more you toss, the more you learn | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=10)
# 
# > -   Generate a list of 1000 tosses with a biased coin which comes up heads only 5% of all times and assign the result to `biased_tosses`.
# > -   Estimate the heads probability based on `biased_tosses`, assign the result to `heads_prob_biased` and visualize it on a density plot.

# In[34]:


# Estimate and plot heads probability based on 1000 tosses with a biased coin
biased_tosses = np.random.binomial(1,.05,1000)
heads_prob_biased = get_heads_prob(biased_tosses)
sns.kdeplot(heads_prob_biased, shade=True, label="biased coin")
plt.show()


# ## Hey, is this coin fair?
# > 
# > In the last two exercises, you have examined the `get_heads_prob()` function to discover how the model estimates the probability of tossing heads and how it updates its estimate as more data comes in.
# > 
# > Now, let's get down to some serious stuff: would you like to play coin flipping against your friend? She is willing to play, as long as you use her special lucky coin. The `tosses` variable contains a list of 1000 results of tossing her coin. Will you play?
# > 
# > In this exercise, you will be doing some plotting with the `seaborn` package again, which has been imported for you as `sns`.

# ### init

# In[35]:


###################
##### numpy ndarray float
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(tosses)
"""

tobedownloaded="""
{numpy.ndarray: {'tosses.csv': 'https://file.io/wUepQSphoeZo'}}
"""
prefixToc='3.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

from downloadfromFileIO import loadNDArrayFromCsv
tosses = loadNDArrayFromCsv(prefix+'tosses.csv')


# ### code

# [Hey, is this coin fair? | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=11)
# 
# > Assign the first 10 and the first 100 `tosses` to the variables `tosses_first_10` and `tosses_first_100`, respectively.

# In[37]:


# Assign first 10 and 100 tosses to separate variables
tosses_first_10 = tosses[:10]
tosses_first_100 = tosses[:100]


# [Hey, is this coin fair? | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=11)
# 
# > Use `get_heads_prob()` to obtain the probability of tossing a head for the first 10, first 100, and all tosses, assigning the results to `heads_prob_first_10`, `heads_prob_first_100`, and `heads_prob_all`, respectively.

# In[38]:


# Get head probabilities for first 10, first 100, and all tossses
heads_prob_first_10 = get_heads_prob(tosses_first_10)
heads_prob_first_100 = get_heads_prob(tosses_first_100)
heads_prob_all = get_heads_prob(tosses)


# [Hey, is this coin fair? | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/the-bayesian-way?ex=11)
# 
# > Plot the density of head probability for each subset of tosses, passing `"first_10"`, `"first_100"`, and `"all"` as the respective `label` argument.

# In[39]:


# Plot density of head probability for each subset of tosses
sns.kdeplot(heads_prob_first_10, shade=True, label='first_10')
sns.kdeplot(heads_prob_first_100, shade=True, label='first_100')
sns.kdeplot(heads_prob_all, shade=True, label='all')
plt.show()


# In[ ]:




