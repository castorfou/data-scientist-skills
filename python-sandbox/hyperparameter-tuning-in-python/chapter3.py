#!/usr/bin/env python
# coding: utf-8

# # Introducing Random Search
# 

# ## Randomly Sample Hyperparameters
# To undertake a random search, we firstly need to undertake a random sampling of our hyperparameter space.
# 
# In this exercise, you will firstly create some lists of hyperparameters that can be zipped up to a list of lists. Then you will randomly sample hyperparameter combinations preparation for running a random search.
# 
# You will use just the hyperparameters learning_rate and min_samples_leaf of the GBM algorithm to keep the example illustrative and not overly complicated.

# ### code

# - Create a list of 200 values for the learning_rate hyperparameter between 0.01 and 1.5 and assign to the list learn_rate_list.
# - Create a list of values between 10 and 40 inclusive for the hyperparameter min_samples_leaf and assign to the list min_samples_list.
# - Combine these lists into a list of lists to sample from.
# - Randomly sample 250 models from these hyperparameter combinations and print the result.

# In[2]:


from itertools import product


# In[4]:


# Create a list of values for the learning_rate hyperparameter
learn_rate_list = np.linspace(0.01,1.5,200)

# Create a list of values for the min_samples_leaf hyperparameter
min_samples_list = list(range(10,41))

# Combination list
combinations_list = [list(x) for x in product(learn_rate_list, min_samples_list)]

# Sample hyperparameter combinations for a random search.
random_combinations_index = np.random.choice(range(0, len(combinations_list)), 250, replace=False)
combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]

# Print the result
print(combinations_random_chosen)


# ## Randomly Search with Random Forest
# To solidify your knowledge of random sampling, let's try a similar exercise but using different hyperparameters and a different algorithm.
# 
# As before, create some lists of hyperparameters that can be zipped up to a list of lists. You will use the hyperparameters criterion, max_depth and max_features of the random forest algorithm. Then you will randomly sample hyperparameter combinations in preparation for running a random search.
# 
# You will use a slightly different package for sampling in this task, random.sample().

# ### code

# - Create lists of the values 'gini' and 'entropy' for criterion & "auto", "sqrt", "log2", None for max_features.
# - Create a list of values between 3 and 55 inclusive for the hyperparameter max_depth and assign to the list max_depth_list. - Remember that range(N,M) will create a list from N to M-1.
# - Combine these lists into a list of lists to sample from using product().
# - Randomly sample 150 models from the combined list and print the result.
# 

# In[6]:


import random


# In[9]:


# Create lists for criterion and max_features
criterion_list = ['gini', 'entropy']
max_feature_list = ['auto', 'sqrt', 'log2', None]

# Create a list of values for the max_depth hyperparameter
max_depth_list = list(range(3,56))

# Combination list
combinations_list = [list(x) for x in product(criterion_list, max_feature_list, max_depth_list)]

# Sample hyperparameter combinations for a random search
combinations_random_chosen = random.sample(combinations_list, 150)

# Print the result
print(combinations_random_chosen)


# ## Visualizing a Random Search
# Visualizing the search space of random search allows you to easily see the coverage of this technique and therefore allows you to see the effect of your sampling on the search space.
# 
# In this exercise you will use several different samples of hyperparameter combinations and produce visualizations of the search space.
# 
# You have been provided the function sample_hyperparameters() which does the work you undertook in the previous exercise to create a (global) random sample of hyperparameter combinations from a (reduced) combinations_list you created previously. It takes an argument n_samples for how many combinations to sample.
# 
# You have been also provided the function visualize_search() which will produce a graph of the sample combinations. It takes no arguments.

# ### init: 1 list

# In[16]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(combinations_list)
tobedownloaded="{list: {'combinations_list.txt': 'https://file.io/7IOkDG'}}"
prefix='data_from_datacamp/Chap3-Exercise1.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[17]:


from uploadfromdatacamp import loadListFromTxt
combinations_list=loadListFromTxt(prefix+'combinations_list.txt')


# In[20]:


x_lims=[0.01, 1.5]
y_lims=[10,29]


# ### code

# In[21]:


import matplotlib.pyplot as plt
def sample_hyperparameters(n_samples):
  global combinations_random_chosen
  
  if n_samples == len(combinations_list):
    combinations_random_chosen = combinations_list
    return
  
  combinations_random_chosen = []
  random_combinations_index = np.random.choice(range(0, len(combinations_list)), n_samples, replace=False)
  combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]
  return

def visualize_search():
  rand_y, rand_x = [x[0] for x in combinations_random_chosen], [x[1] for x in combinations_random_chosen]

  # Plot all together
  plt.clf() 
  plt.scatter(rand_y, rand_x, c=['blue']*len(combinations_random_chosen))
  plt.gca().set(xlabel='learn_rate', ylabel='min_samples_leaf', title='Random Search Hyperparameters')
  plt.gca().set_xlim(x_lims)
  plt.gca().set_ylim(y_lims)
  plt.show()


# In[23]:


# Confirm how hyperparameter combinations & print
number_combs = len(combinations_list)
print(number_combs)

# Sample and visualise combinations
for x in [50, 500, 1500]:
    sample_hyperparameters(x)
    visualize_search()

# Sample all the hyperparameter combinations & visualise
sample_hyperparameters(number_combs)
visualize_search()


# # Random Search in Scikit Learn
# 

# ## The RandomizedSearchCV Object
# Just like the GridSearchCV library from Scikit Learn, RandomizedSearchCV provides many useful features to assist with efficiently undertaking a random search. You're going to create a RandomizedSearchCV object, making the small adjustment needed from the GridSearchCV object.
# 
# The desired options are:
# 
# A default Gradient Boosting Classifier Estimator
# 5-fold cross validation
# Use accuracy to score the models
# Use 4 cores for processing in parallel
# Ensure you refit the best model and return training scores
# Randomly sample 10 models
# The hyperparameter grid should be for learning_rate (150 values between 0.1 and 2) and min_samples_leaf (all values between 20 and 65).
# 
# You will have available X_train & y_train datasets.

# ### init: 1 dataframe, 1 array

# In[24]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train,  y_train)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_train.csv': 'https://file.io/38dUor'}, numpy.ndarray: {'y_train.csv': 'https://file.io/5eUHpe'}}"
prefix='data_from_datacamp/Chap3-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[25]:


import pandas as pd
from uploadfromdatacamp import loadNDArrayFromCsv

X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')


# ### code

# In[28]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier


# In[30]:


# Create the parameter grid
param_grid = {'learning_rate': np.linspace(0.1,2,150), 'min_samples_leaf': list(range(20,65))} 

# Create a random search object
random_GBM_class = RandomizedSearchCV(
    estimator = GradientBoostingClassifier(),
    param_distributions = param_grid,
    n_iter = 10,
    scoring='accuracy', n_jobs=4, cv = 5, refit=True, return_train_score = True)

# Fit to the training data
random_GBM_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_GBM_class.cv_results_['param_learning_rate'])
print(random_GBM_class.cv_results_['param_min_samples_leaf'])


# ## RandomSearchCV in Scikit Learn
# Let's practice building a RandomizedSearchCV object using Scikit Learn. Instead this time we will use a different algorithm and some different instructions.
# 
# The desired options are:
# 
# A RandomForestClassifier Estimator with default 80 estimators
# 3-fold cross validation
# Use AUC to score the models
# Use 4 cores for processing in parallel
# Ensure you refit the best model and return training scores
# Randomly sample 5 models for processing efficiency
# The hyperparameter grid should be for max_depth (all values between 5 and 25) and max_features ('auto' and 'sqrt').
# 
# You will have available X_train & y_train datasets.

# ### code

# In[33]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import AUC


# In[41]:


# Create the parameter grid
param_grid = {'max_depth':list(range(5,26)), 'max_features': ['auto', 'sqrt']} 

# Create a random search object
random_rf_class = RandomizedSearchCV(
    estimator = RandomForestClassifier(n_estimators=80),
    param_distributions = param_grid, n_iter = 5,
    scoring='roc_auc', n_jobs=4, cv = 3, refit=True, return_train_score = True)

# Fit to the training data
random_rf_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_rf_class.cv_results_['param_max_depth'])
print(random_rf_class.cv_results_['param_max_features'])


# # Comparing Grid and Random Search
# 

# ## Grid and Random Search Side by Side
# Visualizing the search space of random and grid search together allows you to easily see the coverage that each technique has and therefore brings to life their specific advantages and disadvantages.
# 
# In this exercise, you will sample hyperparameter combinations in a grid search way as well as a random search way, then plot these to see the difference.
# 
# You will have available:
# 
# combinations_list which is a list of combinations of learn_rate and min_samples_leaf for this algorithm
# The function visualize_search() which will make your hyperparameter combinations into X and Y coordinates and plot both grid and random search combinations on the same graph. It takes as input two lists of hyperparameter combinations.

# ### init: 1 list

# In[42]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(combinations_list)
tobedownloaded="{list: {'combinations_list.txt': 'https://file.io/xpmTfc'}}"
prefix='data_from_datacamp/Chap3-Exercise3.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[43]:


from uploadfromdatacamp import loadListFromTxt
combinations_list=loadListFromTxt(prefix+'combinations_list.txt')


# In[44]:


x_lims=[0.01, 3.0]
y_lims=[5,24]


# ### code

# In[45]:


# Sample grid coordinates
grid_combinations_chosen = combinations_list[0:300]

# Print result
print(grid_combinations_chosen)


# In[46]:


# Create a list of sample indexes
sample_indexes = list(range(0,len(combinations_list)))

# Randomly sample 300 indexes
random_indexes = np.random.choice(sample_indexes, 300, replace=False)


# In[47]:


# Use indexes to create random sample
random_combinations_chosen = [combinations_list[index] for index in random_indexes]


# In[49]:


def visualize_search(grid_combinations_chosen, random_combinations_chosen):
  grid_y, grid_x = [x[0] for x in grid_combinations_chosen], [x[1] for x in grid_combinations_chosen]
  rand_y, rand_x = [x[0] for x in random_combinations_chosen], [x[1] for x in random_combinations_chosen]

  # Plot all together
  plt.scatter(grid_y + rand_y, grid_x + rand_x, c=['red']*300 + ['blue']*300)
  plt.gca().set(xlabel='learn_rate', ylabel='min_samples_leaf', title='Grid and Random Search Hyperparameters')
  plt.gca().set_xlim(x_lims)
  plt.gca().set_ylim(y_lims)
  plt.show()


# In[51]:


# Call the function to produce the visualization
visualize_search(grid_combinations_chosen, random_combinations_chosen)


# ![image.png](attachment:image.png)

# In[ ]:




