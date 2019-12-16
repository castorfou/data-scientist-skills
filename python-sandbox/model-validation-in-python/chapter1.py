#!/usr/bin/env python
# coding: utf-8

# # Introduction to model validation
# 

# ##  Seen vs. unseen data
# Model's tend to have higher accuracy on observations they have seen before. In the candy dataset, predicting the popularity of Skittles will likely have higher accuracy than predicting the popularity of Andes Mints; Skittles is in the dataset, and Andes Mints is not.
# 
# You've built a model based on 50 candies using the dataset X_train and need to report how accurate the model is at predicting the popularity of the 50 candies the model was built on, and the 35 candies (X_test) it has never seen. You will use the mean absolute error, mae(), as the accuracy metric.

# ### init: 4 arrays

# In[1]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, X_test, y_train,  y_test)
tobedownloaded="{numpy.ndarray: {'X_test.csv': 'https://file.io/yJMpfW',  'X_train.csv': 'https://file.io/8e5dGJ',  'y_test.csv': 'https://file.io/vI4Zbk',  'y_train.csv': 'https://file.io/6PZK0J'}}"
prefix='data_from_datacamp/Chap1-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[2]:


from uploadfromdatacamp import loadNDArrayFromCsv
import pandas as pd
X_train=loadNDArrayFromCsv(prefix+'X_train.csv')
X_test=loadNDArrayFromCsv(prefix+'X_test.csv')
y_train=loadNDArrayFromCsv(prefix+'y_train.csv')
y_test=loadNDArrayFromCsv(prefix+'y_test.csv')


# In[7]:


from sklearn.ensemble import RandomForestRegressor 
from  sklearn.metrics import mean_absolute_error as mae
model = RandomForestRegressor(n_estimators=500, random_state=1111)


# ### code

# In[8]:


# The model is fit using X_train and y_train
model.fit(X_train, y_train)

# Create vectors of predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Train/Test Errors
train_error = mae(y_true=y_train, y_pred=train_predictions)
test_error = mae(y_true=y_test, y_pred=test_predictions)

# Print the accuracy for seen and unseen data
print("Model error on seen data: {0:.2f}.".format(train_error))
print("Model error on unseen data: {0:.2f}.".format(test_error))


# # Regression models
# 

# ## Set parameters and fit a model
# Predictive tasks fall into one of two categories: regression or classification. In the candy dataset, the outcome is a continuous variable describing how often the candy was chosen over another candy in a series of 1-on-1 match-ups. To predict this value (the win-percentage), you will use a regression model.
# 
# In this exercise, you will specify a few parameters using a random forest regression model rfr.

# ### init: 2 arrays

# In[9]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, y_train)
tobedownloaded="{numpy.ndarray: {'X_train.csv': 'https://file.io/4CbXRq',  'y_train.csv': 'https://file.io/10itAS'}}"
prefix='data_from_datacamp/Chap1-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[10]:


from uploadfromdatacamp import loadNDArrayFromCsv
import pandas as pd
X_train=loadNDArrayFromCsv(prefix+'X_train.csv')
y_train=loadNDArrayFromCsv(prefix+'y_train.csv')


# In[11]:


from sklearn.ensemble import RandomForestRegressor 
rfr = RandomForestRegressor(random_state=1111)


# ### code

# - Add a parameter to rfr so that the number of trees built is 100 and the maximum depth of these trees is 6.
# - Make sure the model is reproducible by adding a random state of 1111.
# - Use the .fit() method to train the random forest regression model with X_train as the input data and y_train as the response.

# In[12]:


# Set the number of trees
rfr.n_estimators = 100

# Add a maximum depth
rfr.max_depth = 6

# Set the random state
rfr.random_state = 1111

# Fit the model
rfr.fit(X_train, y_train)


# ## Feature importances
# Although some candy attributes, such as chocolate, may be extremely popular, it doesn't mean they will be important to model prediction. After a random forest model has been fit, you can review the model's attribute, .feature_importances_, to see which variables had the biggest impact. You can check how important each variable was in the model by looping over the feature importance array using enumerate().
# 
# If you are unfamiliar with Python's enumerate() function, it can loop over a list while also creating an automatic counter.
# 
# 

# ### init: 1 dataframe, 1 array

# In[14]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, y_train)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_train.csv': 'https://file.io/81uIvS'}, numpy.ndarray: {'y_train.csv': 'https://file.io/XiAQQ2'}}"
prefix='data_from_datacamp/Chap1-Exercise2.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[15]:


import pandas as pd
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
y_train=loadNDArrayFromCsv(prefix+'y_train.csv')


# ### code

# In[16]:


# Fit the model using X and y
rfr.fit(X_train, y_train)

# Print how important each column is to the model
for i, item in enumerate(rfr.feature_importances_):
      # Use i and item to print out the feature importance of each column
    print("{0:s}: {1:.2f}".format(X_train.columns[i], item))


# # Classification models

# ## Classification predictions
# In model validation, it is often important to know more about the predictions than just the final classification. When predicting who will win a game, most people are also interested in how likely it is a team will win.
# 
# ![image.png](attachment:image.png)
# 
# In this exercise, you look at the methods, .predict() and .predict_proba() using the tic_tac_toe dataset. The first method will give a prediction of whether Player One will win the game, and the second method will provide the probability of Player One winning. Use rfc as the random forest classification model.

# ### init: 4 arrays

# In[17]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, X_test, y_train,  y_test)
tobedownloaded="{numpy.ndarray: {'X_test.csv': 'https://file.io/ArIDN8',  'X_train.csv': 'https://file.io/5EbeEd',  'y_test.csv': 'https://file.io/832aZH',  'y_train.csv': 'https://file.io/eJMzic'}}"
prefix='data_from_datacamp/Chap1-Exercise3.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[18]:


from uploadfromdatacamp import loadNDArrayFromCsv
import pandas as pd
X_train=loadNDArrayFromCsv(prefix+'X_train.csv')
X_test=loadNDArrayFromCsv(prefix+'X_test.csv')
y_train=loadNDArrayFromCsv(prefix+'y_train.csv')
y_test=loadNDArrayFromCsv(prefix+'y_test.csv')


# In[25]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(max_depth=6, random_state=1111, n_estimators=50)


# ### code

# In[26]:


# Fit the rfc model. 
rfc.fit(X_train, y_train)

# Create arrays of predictions
classification_predictions = rfc.predict(X_test)
probability_predictions = rfc.predict_proba(X_test)

# Print out count of binary predictions
print(pd.Series(classification_predictions).value_counts())

# Print the first value from probability_predictions
print('The first predicted probabilities are: {}'.format(probability_predictions[0]))


# ## Reusing model parameters
# Replicating model performance is vital in model validation. Replication is also important when sharing models with co-workers, reusing models on new data or asking questions on a website such as Stack Overflow. You might use such a site to ask other coders about model errors, output, or performance. The best way to do this is to replicate your work by reusing model parameters.
# 
# In this exercise, you use various methods to recall which parameters were used in a model.

# ### code

# In[28]:


rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Print the classification model
print(rfc)

# Print the classification model's random state parameter
print('The random state is: {}'.format(rfc.random_state))

# Print all parameters
print('Printing the parameters dictionary: {}'.format(rfc.get_params()))


# ## Random forest classifier
# This exercise reviews the four modeling steps discussed throughout this chapter using a random forest classification model. You will:
# 
# Create a random forest classification model.
# Fit the model using the tic_tac_toe dataset.
# Make predictions on whether Player One will win (1) or lose (0) the current game.
# Finally, you will evaluate the overall accuracy of the model.
# Let's get started!

# ### init: 1 dataframe

# In[29]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(tic_tac_toe)
tobedownloaded="{pandas.core.frame.DataFrame: {'tic_tac_toe.csv': 'https://file.io/MdFujs'}}"
prefix='data_from_datacamp/Chap1-Exercise3.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[30]:


import pandas as pd
tic_tac_toe=pd.read_csv(prefix+'tic_tac_toe.csv',index_col=0)


# ### code

# - Create rfc using the scikit-learn implementation of random forest classifiers and set a random state of 1111.
# 

# In[31]:


from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)


# In[32]:


# Fit rfc using X_train and y_train
rfc.fit(X_train, y_train)


# In[33]:


# Create predictions on X_test
predictions = rfc.predict(X_test)
print(predictions[0:5])


# In[35]:


# Print model accuracy using score() and the testing data
print(rfc.score(X_test, y_test))


# In[ ]:




