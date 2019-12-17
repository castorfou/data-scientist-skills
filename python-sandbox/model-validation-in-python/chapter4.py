#!/usr/bin/env python
# coding: utf-8

# # Introduction to hyperparameter tuning
# 

# ## Creating Hyperparameters
# For a school assignment, your professor has asked your class to create a random forest model to predict the average test score for the final exam.
# 
# After developing an initial random forest model, you are unsatisfied with the overall accuracy. You realize that there are too many hyperparameters to choose from, and each one has a lot of possible values. You have decided to make a list of possible ranges for the hyperparameters you might use in your next model.
# 
# Your professor has provided de-identified data for the last ten quizzes to act as the training data. There are 30 students in your class.

# ### init

# In[3]:


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(max_depth=None, n_estimators='warn', random_state=1111)


# ### code

# - Print.get_params() in the console to review the possible parameters of the model that you can tune.

# In[4]:


# Review the parameters of rfr
print(rfr.get_params())


# - Create a maximum depth list, [4, 8, 12] and a minimum samples list [2, 5, 10] that specify possible values for each hyperparameter.
# 

# In[5]:


# Maximum Depth
max_depth = [4,8,12]

# Minimum samples for a split
min_samples_split = [2,5,10]


# - Create one final list to use for the maximum features.
# Use values 4 through the maximum number of features possible (10), by 2.
# 

# In[7]:


# Max features 
max_features = np.arange(4,12,2)


# ## Running a model using ranges
# You have just finished creating a list of hyperparameters and ranges to use when tuning a predictive model for an assignment. You have used max_depth, min_samples_split, and max_features as your range variable names.

# ### code

# In[9]:


import random


# In[10]:


from sklearn.ensemble import RandomForestRegressor

# Fill in rfr using your variables
rfr = RandomForestRegressor(
    n_estimators=100,
    max_depth=random.choice(max_depth),
    min_samples_split=random.choice(min_samples_split),
    max_features=random.choice(max_features))

# Print out the parameters
print(rfr.get_params())


# # RandomizedSearchCV

# ## Preparing for RandomizedSearch
# Last semester your professor challenged your class to build a predictive model to predict final exam test scores. You tried running a few different models by randomly selecting hyperparameters. However, running each model required you to code it individually.
# 
# After learning about RandomizedSearchCV(), you're revisiting your professors challenge to build the best model. In this exercise, you will prepare the three necessary inputs for completing a random search.

# ### code

# In[11]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Finish the dictionary by adding the max_depth parameter
param_dist = {"max_depth": [2,4,6,8],
              "max_features": [2, 4, 6, 8, 10],
              "min_samples_split": [2, 4, 8, 16]}

# Create a random forest regression model
rfr = RandomForestRegressor(n_estimators=10, random_state=1111)

# Create a scorer to use (use the mean squared error)
scorer = make_scorer(mean_squared_error)


# ## Implementing RandomizedSearchCV
# You are hoping that using a random search algorithm will help you improve predictions for a class assignment. You professor has challenged your class to predict the overall final exam average score.
# 
# In preparation for completing a random search, you have created:
# 
# param_dist: the hyperparameter distributions
# rfr: a random forest regression model
# scorer: a scoring method to use

# ### code

# In[13]:


# Import the method for random search
from sklearn.model_selection import RandomizedSearchCV

# Build a random search using param_dist, rfr, and scorer
random_search =    RandomizedSearchCV(
        estimator=rfr,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring=scorer)


# # Selecting your final model
#  

# ![image.png](attachment:image.png)

# ## Best classification accuracy
# You are in a competition at work to build the best model for predicting the winner of a Tic-Tac-Toe game. You already ran a random search and saved the results of the most accurate model to rs.
# 
# Which parameter set produces the best classification accuracy?

# ### code

# ![image.png](attachment:image.png)

# ## Selecting the best precision model
# Your boss has offered to pay for you to see three sports games this year. Of the 41 home games your favorite team plays, you want to ensure you go to three home games that they will definitely win. You build a model to decide which games your team will win.
# 
# To do this, you will build a random search algorithm and focus on model precision (to ensure your team wins). You also want to keep track of your best model and best parameters, so that you can use them again next year (if the model does well, of course). You have already decided on using the random forest classification model rfc and generated a parameter distribution param_dist.

# ### init: 1 array, 1 list

# In[16]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X, y)
tobedownloaded="{numpy.ndarray: {'X.csv': 'https://file.io/byrCnX'}, list: {'y.txt': 'https://file.io/SH4sVA'}}"
prefix='data_from_datacamp/Chap4-Exercise3.2_'
#(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[17]:


from uploadfromdatacamp import loadNDArrayFromCsv, loadListFromTxt
X=loadNDArrayFromCsv(prefix+'X.csv', dtype='str')
y=loadListFromTxt(prefix+'y.txt')


# ### code

# In[15]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(max_depth=None, n_estimators='warn', random_state=False)
param_dist= {'max_depth': range(2, 12, 2), 'min_samples_split': range(2, 12, 2), 'n_estimators': [10, 25, 50]}


# In[18]:


from sklearn.metrics import precision_score, make_scorer

# Create a precision scorer
precision = make_scorer(precision_score)
# Finalize the random search
rs = RandomizedSearchCV(
  estimator=rfc, param_distributions=param_dist,
  scoring = precision,
  cv=5, n_iter=10, random_state=1111)
rs.fit(X, y)

# print the mean test scores:
print('The accuracy for each run was: {}.'.format(rs.cv_results_['mean_test_score']))
# print the best model score:
print('The best accuracy for a single model was: {}'.format(rs.best_score_))


# In[ ]:




