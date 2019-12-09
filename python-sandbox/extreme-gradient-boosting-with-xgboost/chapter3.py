#!/usr/bin/env python
# coding: utf-8

# # Why tune your model?
# 

# ## Tuning the number of boosting rounds
# Let's start with parameter tuning by seeing how the number of boosting rounds (number of trees you build) impacts the out-of-sample performance of your XGBoost model. You'll use xgb.cv() inside a for loop and build one model per num_boost_round parameter.
# 
# Here, you'll continue working with the Ames housing dataset. The features are available in the array X, and the target vector is contained in y.

# ### init: 1 dataframe, 1 serie

# In[2]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X, y)
tobedownloaded="{pandas.core.frame.DataFrame: {'X.csv': 'https://file.io/001oTj'}, pandas.core.series.Series: {'y.csv': 'https://file.io/nm1NHn'}}"
prefix='data_from_datacamp/Chap3-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[3]:


import pandas as pd
X=pd.read_csv(prefix+'X.csv',index_col=0)
y=pd.read_csv(prefix+'y.csv',index_col=0, header=None,squeeze=True)


# ### code

# - Create a DMatrix called housing_dmatrix from X and y.
# - Create a parameter dictionary called params, passing in the appropriate "objective" ("reg:linear") and "max_depth" (set it to 3).
# - Iterate over num_rounds inside a for loop and perform 3-fold cross-validation. In each iteration of the loop, pass in the current number of boosting rounds (curr_num_rounds) to xgb.cv() as the argument to num_boost_round.
# - Append the final boosting round RMSE for each cross-validated XGBoost model to the final_rmse_per_round list. 
# - num_rounds and final_rmse_per_round have been zipped and converted into a DataFrame so you can easily see how the model performs with each boosting round. Hit 'Submit Answer' to see the results!
# 

# In[4]:


import xgboost as xgb


# In[5]:


# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params 
params = {"objective":"reg:linear", "max_depth":3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)
    
    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))


# ## Automated boosting round selection using early_stopping
# Now, instead of attempting to cherry pick the best possible number of boosting rounds, you can very easily have XGBoost automatically select the number of boosting rounds for you within xgb.cv(). This is done using a technique called early stopping.
# 
# Early stopping works by testing the XGBoost model after every boosting round against a hold-out dataset and stopping the creation of additional boosting rounds (thereby finishing training of the model early) if the hold-out metric ("rmse" in our case) does not improve for a given number of rounds. Here you will use the early_stopping_rounds parameter in xgb.cv() with a large possible number of boosting rounds (50). Bear in mind that if the holdout metric continuously improves up through when num_boosting_rounds is reached, then early stopping does not occur.
# 
# Here, the DMatrix and parameter dictionary have been created for you. Your task is to use cross-validation with early stopping. Go for it!

# ### code

# - Perform 3-fold cross-validation with early stopping and "rmse" as your metric. Use 10 early stopping rounds and 50 boosting rounds. Specify a seed of 123 and make sure the output is a pandas DataFrame. Remember to specify the other parameters such as dtrain, params, and metrics.
# - Print cv_results.

# In[7]:


# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)


# # Overview of XGBoost's hyperparameters
# 

# ![image.png](attachment:image.png)

# ## Tuning eta
# It's time to practice tuning other XGBoost hyperparameters in earnest and observing their effect on model performance! You'll begin by tuning the "eta", also known as the learning rate.
# 
# The learning rate in XGBoost is a parameter that can range between 0 and 1, with higher values of "eta" penalizing feature weights more strongly, causing much stronger regularization.

# ### code

# - Create a list called eta_vals to store the following "eta" values: 0.001, 0.01, and 0.1.
# - Iterate over your eta_vals list using a for loop.
# - In each iteration of the for loop, set the "eta" key of params to be equal to curr_val. Then, perform 3-fold cross-validation with early stopping (5 rounds), 10 boosting rounds, a metric of "rmse", and a seed of 123. Ensure the output is a DataFrame.
# - Append the final round RMSE to the best_rmse list.

# In[8]:


# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree (boosting round)
params = {"objective":"reg:linear", "max_depth":3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematically vary the eta 
for curr_val in eta_vals:

    params["eta"] = curr_val
    
    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, early_stopping_rounds=5, num_boost_round=10, metrics='rmse', seed=123, as_pandas=True)
    
    
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))


# ## Tuning max_depth
# In this exercise, your job is to tune max_depth, which is the parameter that dictates the maximum depth that each tree in a boosting round can grow to. Smaller values will lead to shallower trees, and larger values to deeper trees.

# - Create a list called max_depths to store the following "max_depth" values: 2, 5, 10, and 20.
# - Iterate over your max_depths list using a for loop.
# - Systematically vary "max_depth" in each iteration of the for loop and perform 2-fold cross-validation with early stopping (5 rounds), 10 boosting rounds, a metric of "rmse", and a seed of 123. Ensure the output is a DataFrame.

# In[9]:


# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary
params = {"objective":"reg:linear"}

# Create list of max_depth values
max_depths = [2,5,10,20]
best_rmse = []

# Systematically vary the max_depth
for curr_val in max_depths:

    params["max_depth"] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, early_stopping_rounds=5,                         num_boost_round=10, metrics='rmse', seed=123, as_pandas=True)
    
    
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(max_depths, best_rmse)),columns=["max_depth","best_rmse"]))


# ## Tuning colsample_bytree
# Now, it's time to tune "colsample_bytree". You've already seen this if you've ever worked with scikit-learn's RandomForestClassifier or RandomForestRegressor, where it just was called max_features. In both xgboost and sklearn, this parameter (although named differently) simply specifies the fraction of features to choose from at every split in a given tree. In xgboost, colsample_bytree must be specified as a float between 0 and 1.

# ### code

# - Create a list called colsample_bytree_vals to store the values 0.1, 0.5, 0.8, and 1.
# - Systematically vary "colsample_bytree" and perform cross-validation, exactly as you did with max_depth and eta previously.
# 

# In[10]:


# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary
params={"objective":"reg:linear","max_depth":3}

# Create list of hyperparameter values: colsample_bytree_vals
colsample_bytree_vals = [0.1,0.5,0.8,1]
best_rmse = []

# Systematically vary the hyperparameter value 
for curr_val in colsample_bytree_vals:

    params['colsample_bytree'] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                 num_boost_round=10, early_stopping_rounds=5,
                 metrics="rmse", as_pandas=True, seed=123)
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=["colsample_bytree","best_rmse"]))


# ![image.png](attachment:image.png)

# # Review of grid search and random search
# 

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ## Grid search with XGBoost
# Now that you've learned how to tune parameters individually with XGBoost, let's take your parameter tuning to the next level by using scikit-learn's GridSearch and RandomizedSearch capabilities with internal cross-validation using the GridSearchCV and RandomizedSearchCV functions. You will use these to find the best model exhaustively from a collection of possible parameter values across multiple parameters simultaneously. Let's get to work, starting with GridSearchCV!

# ### code

# - Create a parameter grid called gbm_param_grid that contains a list of "colsample_bytree" values (0.3, 0.7), a list with a single value for "n_estimators" (50), and a list of 2 "max_depth" (2, 5) values.
# - Instantiate an XGBRegressor object called gbm.
# - Create a GridSearchCV object called grid_mse, passing in: the parameter grid to param_grid, the XGBRegressor to estimator, "neg_mean_squared_error" to scoring, and 4 to cv. Also specify verbose=1 so you can better understand the output.
# - Fit the GridSearchCV object to X and y.
# - Print the best parameter values and lowest RMSE, using the .best_params_ and .best_score_ attributes, respectively, of grid_mse.

# In[11]:


from sklearn.model_selection import GridSearchCV


# In[12]:


# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(param_grid=gbm_param_grid, estimator=gbm, scoring='neg_mean_squared_error', cv=4, verbose=1)


# Fit grid_mse to the data
grid_mse.fit(X,y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))


# ## Random search with XGBoost
# Often, GridSearchCV can be really time consuming, so in practice, you may want to use RandomizedSearchCV instead, as you will do in this exercise. The good news is you only have to make a few modifications to your GridSearchCV code to do RandomizedSearchCV. The key difference is you have to specify a param_distributions parameter instead of a param_grid parameter.

# ### code

# - Create a parameter grid called gbm_param_grid that contains a list with a single value for 'n_estimators' (25), and a list of 'max_depth' values between 2 and 11 for 'max_depth' - use range(2, 12) for this.
# - Create a RandomizedSearchCV object called randomized_mse, passing in: the parameter grid to param_distributions, the XGBRegressor to estimator, "neg_mean_squared_error" to scoring, 5 to n_iter, and 4 to cv. Also specify verbose=1 so you can better understand the output.
# - Fit the RandomizedSearchCV object to X and y.

# In[13]:


from sklearn.model_selection import RandomizedSearchCV


# In[15]:


# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor(n_estimators=10)

# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(param_distributions=gbm_param_grid, estimator=gbm, scoring='neg_mean_squared_error', n_iter=5, cv=4, verbose=1)


# Fit randomized_mse to the data
randomized_mse.fit(X,y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))


# # Limits of grid search and random search
# 

# ![image.png](attachment:image.png)

# In[ ]:




