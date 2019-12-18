#!/usr/bin/env python
# coding: utf-8

# # Introducing Grid Search
# 

# ## Build Grid Search functions
# In data science it is a great idea to try building algorithms, models and processes 'from scratch' so you can really understand what is happening at a deeper level. Of course there are great packages and libraries for this work (and we will get to that very soon!) but building from scratch will give you a great edge in your data science work.
# 
# In this exercise, you will create a function to take in 2 hyperparameters, build models and return results. You will use this function in a future exercise.
# 
# You will have available the X_train, X_test, y_train and y_test datasets available.

# ### init: 2 dataframes, 2 arrays

# In[7]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/KNty1a',  'X_train.csv': 'https://file.io/bCWSyc'}, numpy.ndarray: {'y_test.csv': 'https://file.io/wZlesY',  'y_train.csv': 'https://file.io/sNmdly'}}"
prefix='data_from_datacamp/Chap2-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[8]:


import pandas as pd
from uploadfromdatacamp import loadNDArrayFromCsv

X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')


# ### code

# - Build a function that takes two parameters called learn_rate and max_depth for the learning rate and maximum depth.
# - Add capability in the function to build a GBM model and fit it to the data with the input hyperparameters.
# - Have the function return the results of that model and the chosen hyperparameters (learn_rate and max_depth).

# In[12]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# In[13]:


# Create the function
def gbm_grid_search(learn_rate, max_depth):

	# Create the model
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth)
    
    # Use the model to make predictions
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Return the hyperparameters and score
    return([learn_rate, max_depth, accuracy_score(y_test, predictions)])


# ## Iteratively tune multiple hyperparameters
# In this exercise, you will build on the function you previously created to take in 2 hyperparameters, build a model and return the results. You will now use that to loop through some values and then extend this function and loop with another hyperparameter.
# 
# The function gbm_grid_search(learn_rate, max_depth) is available in this exercise.
# 
# If you need to remind yourself of the function you can run the function print_func() that has been created for you

# ### code

# - Write a for-loop to test the values (0.01, 0.1, 0.5) for the learning_rate and (2, 4, 6) for the max_depth using the function you created gbm_grid_search and print the results.
# 

# In[14]:


# Create the relevant lists
results_list = []
learn_rate_list = [0.01,0.1,0.5]
max_depth_list = [2,4,6]

# Create the for loop
for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        results_list.append(gbm_grid_search(learn_rate,max_depth))

# Print the results
print(results_list)   


# Extend the gbm_grid_search function to include the hyperparameter subsample. Name this new function gbm_grid_search_extended.

# In[15]:


# Extend the function input
def gbm_grid_search_extended(learn_rate, max_depth, subsample):

	# Extend the model creation section
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth, subsample=subsample)
    
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Extend the return part
    return([learn_rate, max_depth, subsample, accuracy_score(y_test, predictions)])       


# Extend your loop to call gbm_grid_search (available in your console), then test the values [0.4 , 0.6] for the subsample hyperparameter and print the results. max_depth_list & learn_rate_list are available in your environment.

# In[17]:


results_list = []

# Create the new list to test
subsample_list = [0.4, 0.6]

for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
    
    	# Extend the for loop
        for subsample in subsample_list:
        	
            # Extend the results to include the new hyperparameter
            results_list.append(gbm_grid_search_extended(learn_rate, max_depth, subsample))
            
# Print results
print(results_list)            


# ## How Many Models?
# Adding more hyperparameters or values, you increase the amount of models created but the increases is not linear it is proportional to how many values and hyperparameters you already have.
# 
# How many models would be created when running a grid search over the following hyperparameters and values for a GBM algorithm?
# 
# learning_rate = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2]
# max_depth = [4,6,8,10,12,14,16,18, 20]
# subsample = [0.4, 0.6, 0.7, 0.8, 0.9]
# max_features = ['auto', 'sqrt', 'log2']
# These lists are in your console so you can utilize properties of them to help you!

# ### code

# In[18]:


learning_rate = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2] 
max_depth = [4,6,8,10,12,14,16,18, 20] 
subsample = [0.4, 0.6, 0.7, 0.8, 0.9] 
max_features = ['auto', 'sqrt', 'log2']


# In[24]:


len(learning_rate)*len(max_depth)*len(subsample)*len(max_features)


# # Grid Search with Scikit Learn
# 

# ## GridSearchCV with Scikit Learn
# The GridSearchCV module from Scikit Learn provides many useful features to assist with efficiently undertaking a grid search. You will now put your learning into practice by creating a GridSearchCV object with certain parameters.
# 
# The desired options are:
# 
# A Random Forest Estimator, with the split criterion as 'entropy'
# 5-fold cross validation
# The hyperparameters max_depth (2, 4, 8, 15) and max_features ('auto' vs 'sqrt')
# Use roc_auc to score the models
# Use 4 cores for processing in parallel
# Ensure you refit the best model and return training scores
# You will have available X_train, X_test, y_train & y_test datasets.

# ### init: 2 dataframes, 2 arrays

# In[26]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/Lprt4S',  'X_train.csv': 'https://file.io/SeJEvv'}, numpy.ndarray: {'y_test.csv': 'https://file.io/fJZGcM',  'y_train.csv': 'https://file.io/LFExum'}}"
prefix='data_from_datacamp/Chap2-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[27]:


import pandas as pd
from uploadfromdatacamp import loadNDArrayFromCsv

X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')


# ### code

# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[31]:


# Create a Random Forest Classifier with specified criterion
rf_class = RandomForestClassifier(criterion='entropy')

# Create the parameter grid
param_grid = {'max_depth': [2,4,8,15], 'max_features': ['auto', 'sqrt']} 

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(
    estimator=rf_class,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=4,
    cv=5,
    refit=True, return_train_score=True)
print(grid_rf_class)


# # Understanding a grid search output
# 

# ## Exploring the grid search results
# You will now explore the cv_results_ property of the GridSearchCV object defined in the video. This is a dictionary that we can read into a pandas DataFrame and contains a lot of useful information about the grid search we just undertook.
# 
# A reminder of the different column types in this property:
# 
# time_columns
# params_ columns and the params column
# test_score columns for each cv fold including the mean_test_score and std_test_score columns
# a rank_test_score column
# train_score columns for each cv fold including the mean_train_score and std_train_score columns
# We will firstly read the cv_results property into a DataFrame, then extract and explore different elements using pandas loc & iloc attributes.

# ### code

# In[33]:


grid_rf_class.fit(X_train, y_train)


# In[36]:


# Read the cv_results property into a dataframe & print it out
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
print(cv_results_df)

# Get and show the column with dictionaries of the hyperparameters used
column = cv_results_df.loc[:, ['params']]
print(column)

# Get and show the row that had the best mean test score
best_row = cv_results_df[cv_results_df['rank_test_score'] == 1 ]
print(best_row)


# ## Analyzing the best results
# The cv_results_ property has a lot of information in it. At the end of the day, we primarily care about the best square on that grid search. Luckily Scikit Learn's gridSearchCv objects also have a number of parameters that provide key information on just the best square (or row in cv_results_).
# 
# These three properties are:
# 
# - best_params_ – Which is a dictionary of the parameters that gave the best score.
# - best_score_ – The actual best score.
# - best_index_ – The index of the row in cv_results_ that was the best.
# 
# The grid search object grid_rf_class has been loaded for you.

# ### code

# In[46]:


# Print out the ROC_AUC score from the best grid search square
best_score = grid_rf_class.best_score_
print(best_score)

# Recreate the best_row variable
best_row = cv_results_df.loc[[grid_rf_class.best_index_]]
print(best_row)

# Get the n_estimators from the best grid search
best_n_estimators = grid_rf_class.best_estimator_.get_params()['n_estimators']


# ## Using the best results
# While it is interesting to analyze the results of our grid search, our final goal is practical in nature; we want to make predictions on our test set using our estimator object.
# 
# We can access this object through the best_estimator_ property of our grid search object.
# 
# In this exercise we will take a look inside the best_estimator_ property and then use this to make predictions on our test set for credit card defaults and generate a variety of scores. Remember to use predict_proba rather than predict since we need probability values rather than class labels for our roc_auc score. We use a slice [:,1] to get probabilities of the positive class.
# 
# You have available the X_test and y_test datasets to use and the grid_rf_class object from previous exercises.

# ### code

# In[52]:


from sklearn.metrics import confusion_matrix, roc_auc_score


# In[53]:


# See what type of object the best_estimator_ property is
print(type(grid_rf_class.best_estimator_))

# Create an array of predictions directly using the best_estimator_ property
predictions = grid_rf_class.best_estimator_.predict(X_test)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# Now create a confusion matrix 
print("Confustion Matrix \n", confusion_matrix(y_test, predictions))

# Get the ROC-AUC score
predictions_proba = grid_rf_class.best_estimator_.predict_proba(X_test)[:,1]
print("ROC-AUC Score \n", roc_auc_score(y_test, predictions_proba))


# In[ ]:




