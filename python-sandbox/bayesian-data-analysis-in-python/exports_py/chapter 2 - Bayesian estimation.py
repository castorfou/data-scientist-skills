#!/usr/bin/env python
# coding: utf-8

# # Under the Bayesian hood
# 
# ```python
# # Grid approximation
# from scipy.stats import binom
# from scipy.stats import uniform
# num_heads = np.arange(0, 101, 1)
# head_prob = np.arange(0, 1.01, 0.01)
# coin = pd.DataFrame([(x, y) for x in num_heads for y in head_prob])
# coin.columns = ["num_heads", "head_prob"]
# coin["prior"] = uniform.pdf(coin["head_prob"])
# coin["likelihood"] = binom.pmf(coin["num_heads"], 100, coin["head_prob"])
# coin["posterior_prob"] = coin["prior"] * coin["likelihood"]
# coin["posterior_prob"] /= coin["posterior_prob"].sum()
# 
# # Plotting posterior distribution
# # Q: What's the probability of tossing heads with a coin, if we observed 75 heads in 100 tosses?
# import seaborn as sns
# import matplotlib.pyplot as plt
# heads75 = coin.loc[coin["num_heads"] == 75]
# heads75["posterior_prob"] /= heads75["posterior_prob"].sum()
# sns.lineplot(heads75["head_prob"], heads75["posterior_prob"])
# plt.show()
# 
# ```

# ## Towards grid approximation
# > 
# > Congratulations! You have just been hired as a data analyst at your government's Department of Health. The cabinet is considering the purchase of a brand-new drug against a deadly and contagious virus. There are some doubts, however, regarding how effective the new drug is against the virus. You have been tasked with estimating the drug's efficacy rate, i.e. the percentage of patients cured by the drug.
# > 
# > An experiment was quickly set up in which 10 sick patients have been treated with the drug. Once you know how many of them are cured, you can use the binomial distribution with a cured patient being a "success" and the efficacy rate being the "probability of success". While you are waiting for the experiment's results, you decide to prepare the parameter grid.
# > 
# > `numpy` and `pandas` have been imported for you as `np` and `pd`, respectively.

# [Towards grid approximation | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=2)
# 
# > -   Using `np.arange()`, create an array of all possible numbers of patients cured (from 0 to 10) and assign it to `num_patients_cured`.
# > -   Using `np.arange()`, create an array of all possible values for the efficacy rate (from 0 to 1, by 0.01) and assign it to `efficacy_rate`.
# > -   Combine `num_patients_cured` and `efficacy_rate` into a DataFrame called `df`, listing all possible combinations of the two.
# > -   Assign `["num_patients_cured", "efficacy_rate"]` to `df`'s columns and print it.

# In[1]:


import numpy as np
import pandas as pd


# In[8]:


# Create cured patients array from 0 to 10
num_patients_cured = np.arange(0, 11, 1)

# Create efficacy rate array from 0 to 1 by 0.01
efficacy_rate = np.arange(0, 1.01, 0.01)

# Combine the two arrays in one DataFrame
df = pd.DataFrame([(x, y) for x in num_patients_cured for y in efficacy_rate])

# Name the columns
df.columns = ["num_patients_cured", "efficacy_rate"]

# Print df
print(df)


# ## Grid approximation without prior knowledge
# > 
# > According to the experiment's outcomes, out of 10 sick patients treated with the drug, 9 have been cured. What can you say about the drug's efficacy rate based on such a small sample? Assume you have no prior knowledge whatsoever regarding how good the drug is.
# > 
# > A DataFrame `df` with all possible combinations of the number of patients cured and the efficacy rate which you created in the previous exercise is available in the workspace.
# > 
# > `uniform` and `binom` have been imported for you from `scipy.stats`. Also, `pandas` and `seaborn` are imported as `pd` and `sns`, respectively.

# ### init

# In[9]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/P830G2BHHb6o'}}
"""
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
df = pd.read_csv(prefix+'df.csv',index_col=0)


# In[19]:


import numpy as np
import pandas as pd
from scipy.stats import binom
from scipy.stats import uniform
import seaborn as sns
import matplotlib.pyplot as plt


# ### code

# [Grid approximation without prior knowledge | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=3)
# 
# > -   Calculate a uniform `prior` for `efficacy_rate` using `uniform.pdf()` and assign it as a new column to `df`.
# > -   Calculate the binomial `likelihood` using `binom.pmf()` by passing the number of "successes", the total number of observations, and the "probability of success", and assign the result as a new column to `df`.

# In[12]:


# Calculate the prior efficacy rate and the likelihood
df["prior"] = uniform.pdf(df['efficacy_rate'])
df["likelihood"] = binom.pmf(df['num_patients_cured'], 10, df['efficacy_rate'])


# [Grid approximation without prior knowledge | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=3)
# 
# > Calculate the posterior probability for efficacy rate, assign it to a new column called `posterior_prob` in `df`, and scale it so that it sums up to 1.

# In[13]:


# Calculate the posterior efficacy rate and scale it to sum up to one
df["posterior_prob"] = df['prior']*df['likelihood']
df["posterior_prob"] /= df['posterior_prob'].sum()


# [Grid approximation without prior knowledge | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=3)
# 
# > Filter `df` to keep only rows where the number of patients cured is 9, assign the result to `df_9_of_10_cured`, and scale the `posterior_prob` so that it sums up to 1.

# In[15]:


# Compute the posterior probability of observing 9 cured patients
df_9_of_10_cured = df.loc[df.num_patients_cured == 9]
df_9_of_10_cured["posterior_prob"] /= df_9_of_10_cured["posterior_prob"].sum()


# [Grid approximation without prior knowledge | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=3)
# 
# > Plot the drug's posterior efficacy rate having seen 9 out of 10 patients cured.

# In[21]:


# Plot the drug's posterior efficacy rate
sns.lineplot(df_9_of_10_cured['efficacy_rate'], df_9_of_10_cured['posterior_prob'])
plt.show()


# ## Updating posterior belief
# > 
# > Well done on estimating the posterior distribution of the efficacy rate in the previous exercise! Unfortunately, due to a small data sample, this distribution is quite wide, indicating much uncertainty regarding the drug's quality. Luckily, testing of the drug continues, and a group of another 12 sick patients have been treated, 10 of whom were cured. We need to update our posterior distribution with these new data!
# > 
# > This is easy to do with the Bayesian approach. We simply need to run the grid approximation similarly as before, but with a different prior. We can use all our knowledge about the efficacy rate (embodied by the posterior distribution from the previous exercise) as a new prior! Then, we recompute the likelihood for the new data, and get the new posterior!
# > 
# > The DataFrame you created in the previous exercise, `df`, is available in the workspace and `binom` has been imported for you from `scipy.stats`.

# ### init

# In[22]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/4dUNY2vK5hDx'}}
"""
prefixToc='1.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
df = pd.read_csv(prefix+'df.csv',index_col=0)


# In[23]:


from scipy.stats import binom


# ### code

# [Updating posterior belief | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=4)
# 
# > -   Assign `posterior_prob` from `df` to a new column called `new_prior`.
# > -   Calculate the `new_likelihood` using `binom.pmf()` based on the new data and assign it as a new column to `df`.

# In[28]:


# Assign old posterior to new prior and calculate likelihood
df["new_prior"] = df['posterior_prob']
df["new_likelihood"] = binom.pmf(df["num_patients_cured"], 12, df["efficacy_rate"])


# [Updating posterior belief | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=4)
# 
# > Calculate the `new_posterior_prob` using `new_prior` and `new_likelihood`, assign it as a new column to `df`, and scale it by its sum.

# In[30]:


# Calculate new posterior and scale it
df["new_posterior_prob"] = df['new_prior'] * df['new_likelihood']
df["new_posterior_prob"] /= df["new_posterior_prob"].sum()


# [Updating posterior belief | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=4)
# 
# > Filter `df` to keep only rows with 10 cured patients, assign the result to `df_10_of_12_cured`, and scale the `new_posterior_prob` so that it sums up to 1.

# In[31]:


# Compute the posterior probability of observing 10 cured patients
df_10_of_12_cured = df.loc[df.num_patients_cured == 10]
df_10_of_12_cured["new_posterior_prob"] /= df_10_of_12_cured["new_posterior_prob"].sum()


# [Updating posterior belief | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=4)
# 
# > #### Question
# > 
# > We have two posterior distributions for the efficacy rate now:
# > 
# > 1.  The one from the previous exercise (without prior knowledge, after seeing 9 out of 10 patients cured) which you have used as a new prior in this exercise.
# > 2.  The updated one you have just calculated (after seeing another 10 out of 12 patients cured).
# > 
# > You can plot them on top of each other using the following code chunk:
# > 
# >     sns.lineplot(df_10_of_12_cured["efficacy_rate"], 
# >                  df_10_of_12_cured["new_posterior_prob"], 
# >                  label="new posterior")
# >     sns.lineplot(df_9_of_10_cured["efficacy_rate"], 
# >                  df_9_of_10_cured["posterior_prob"], 
# >                  label="old posterior = new prior")
# >     plt.show()
# >     
# > 
# > Based on the plot, which of the following statements is **false**?

# In[32]:


sns.lineplot(df_10_of_12_cured["efficacy_rate"], 
             df_10_of_12_cured["new_posterior_prob"], 
             label="new posterior")
sns.lineplot(df_9_of_10_cured["efficacy_rate"], 
             df_9_of_10_cured["posterior_prob"], 
             label="old posterior = new prior")
plt.show()


# # Prior belief

# ## Simulating posterior draws
# > 
# > You have just decided to use a Beta(5, 2) prior for the efficacy rate. You are also using the binomial distribution to model the data (curing a sick patient is a "success", remember?). Since the beta distribution is a conjugate prior for the binomial likelihood, you can simply simulate the posterior!
# > 
# > You know that if the prior is $$Beta(a,b)$$
# > 
# > , then the posterior is $$Beta(x, y)$$, with:
# > 
# > $$x = NumberOfSuccesses + a$$,
# > 
# > $$y=NumberOfObservations-NumberOfSuccess+b$$.
# > 
# > Can you simulate the posterior distribution? Recall that altogether you have data on 22 patients, 19 of whom have been cured. `numpy` and `seaborn` have been imported for you as `np` and `sns`, respectively.

# In[33]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# [Simulating posterior draws | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=8)
# 
# > -   Assign the numbers of patients treated and cured to `num_patients_treated` and `num_patients_cured`, respectively.
# > -   Use the appropriate `numpy` function to sample from the posterior distribution and assign the result to `posterior_draws`.
# > -   Plot the posterior distribution using the appropriate `seaborn` function.

# In[34]:


# Define the number of patients treated and cured
num_patients_treated = 22
num_patients_cured = 19

# Simulate 10000 draws from the posterior distribution
posterior_draws = np.random.beta(19 + 5, 22 - 19 + 2, 10000)

# Plot the posterior distribution
sns.kdeplot(posterior_draws, shade=True)
plt.show()


# # Reporting Bayesian results
# 
# ```python
# # Bayesian point estimates
# posterior_mean = np.mean(posterior_draws)
# posterior_median = np.median(posterior_draws)
# posterior_p75 = np.percentile(posterior_draws, 75)
# 
# # Highest Posterior Density (HPD)
# import pymc3 as pm
# hpd = pm.hpd(posterior_draws, hdi_prob=0.9)
# print(hpd)
# ```

# ## Point estimates
# > 
# > You continue working at your government's Department of Health. You have been tasked with filling the following memo with numbers, before it is sent to the secretary.
# > 
# > > Based on the experiments carried out by ourselves and neighboring countries, should we distribute the drug, we can expect \_\_\_ infected people to be cured. There is a 50% probability the number of cured infections will amount to at least \_\_\_, and with 90% probability it will not be less than \_\_\_.
# > 
# > The array of posterior draws of the drug's efficacy rate you have estimated before is available to you as `drug_efficacy_posterior_draws`.
# > 
# > Calculate the three numbers needed to fill in the memo, knowing there are 100,000 infections at the moment. `numpy` has been imported for you as `np`.

# ### init

# In[2]:


###################
##### numpy ndarray float
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(drug_efficacy_posterior_draws)
"""

tobedownloaded="""
{numpy.ndarray: {'drug_efficacy_posterior_draws.csv': 'https://file.io/cCvkzC0Q3jE3'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

from downloadfromFileIO import loadNDArrayFromCsv
drug_efficacy_posterior_draws = loadNDArrayFromCsv(prefix+'drug_efficacy_posterior_draws.csv')


# In[37]:


import numpy as np


# ### code

# [Point estimates | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=10)
# 
# > Calculate the expected number of cured infections by multiplying the drug's expected efficacy rate (`drug_efficacy_posterior_draws`) by the number of infected people (100,000) and assign the result to `cured_expected`.

# In[39]:


# Calculate the expected number of people cured
cured_expected = np.mean(drug_efficacy_posterior_draws) * 100000


# [Point estimates | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=10)
# 
# > Calculate the minimum number of infections that will be cured with 50% probability and assign it to `min_cured_50_perc`.

# In[43]:


# Calculate the minimum number of people cured with 50% probability
min_cured_50_perc = np.percentile(drug_efficacy_posterior_draws, 50)*100000


# [Point estimates | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=10)
# 
# > Calculate the minimum number of infections that will be cured with 90% probability and assign it to `min_cured_90_perc`.

# In[45]:


# Calculate the minimum number of people cured with 90% probability
min_cured_90_perc = np.percentile(drug_efficacy_posterior_draws, 10)*100000

# Print the filled-in memo
print(f"Based on the experiments carried out by ourselves and neighboring countries, \nshould we distribute the drug, we can expect {int(cured_expected)} infected people to be cured. \nThere is a 50% probability the number of cured infections \nwill amount to at least {int(min_cured_50_perc)}, and with 90% probability \nit will not be less than {int(min_cured_90_perc)}.")


# ## Highest Posterior Density credible intervals
# > 
# > You know that reporting bare point estimates is not enough. It would be great to provide a measure of uncertainty in the drug's efficacy rate estimate, and you have all the means to do so. You decide to add the following to the memo.
# > 
# > > The experimental results indicate that with a 90% probability the new drug's efficacy rate is between \_\_\_ and \_\_\_, and with a 95% probability it is between \_\_\_ and \_\_\_.
# > 
# > You will need to calculate two credible intervals: one of 90% and another of 95% probability. The `drug_efficacy_posterior_draws` array is still available in your workspace.

# [Highest Posterior Density credible intervals | Python](https://campus.datacamp.com/courses/bayesian-data-analysis-in-python/bayesian-estimation?ex=11)
# 
# > -   Import the `pymc3` package as `pm`.
# > -   Calculate the Highest Posterior Density credible interval of 90% and assign it to `ci_90`.
# > -   Calculate the Highest Posterior Density credible interval of 95% and assign it to `ci_95`.

# In[4]:


import numpy as np


# In[5]:


# Import pymc3 as pm
import pymc3 as pm

# Calculate HPD credible interval of 90%
ci_90 = pm.hpd(drug_efficacy_posterior_draws, hdi_prob=0.9)

# Calculate HPD credible interval of 95%
ci_95 = pm.hpd(drug_efficacy_posterior_draws, hdi_prob=0.95)

# Print the memo
print(f"The experimental results indicate that with a 90% probability \nthe new drug's efficacy rate is between {np.round(ci_90[0], 2)} and {np.round(ci_90[1], 2)}, \nand with a 95% probability it is between {np.round(ci_95[0], 2)} and {np.round(ci_95[1], 2)}.")


# In[ ]:




