#!/usr/bin/env python
# coding: utf-8

# # Creating train, test, and validation datasets
# 

# ## Create one holdout set
# Your boss has asked you to create a simple random forest model on the tic_tac_toe dataset. She doesn't want you to spend much time selecting parameters; rather she wants to know how well the model will perform on future data. For future Tic-Tac-Toe games, it would be nice to know if your model can predict which player will win.
# 
# The dataset tic_tac_toe has been loaded for your use.
# 
# Note that in Python, =\ indicates the code was too long for one line and has been split across two lines.

# ### init: 1 dataframe

# In[4]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(tic_tac_toe)
tobedownloaded="{pandas.core.frame.DataFrame: {'tic_tac_toe.csv': 'https://file.io/MdFujs'}}"
prefix='data_from_datacamp/Chap1-Exercise3.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[5]:


import pandas as pd
tic_tac_toe=pd.read_csv(prefix+'tic_tac_toe.csv',index_col=0)


# ### code

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[8]:


# Create dummy variables using pandas
X = pd.get_dummies(tic_tac_toe.iloc[:,0:9])
y = tic_tac_toe.iloc[:, 9]

# Create training and testing datasets. Use 10% for the test set
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.1, random_state=1111)


# ## Create two holdout sets
# You recently created a simple random forest model to predict Tic-Tac-Toe game wins for your boss, and at her request, you did not do any parameter tuning. Unfortunately, the overall model accuracy was too low for her standards. This time around, she has asked you to focus on model performance.
# 
# Before you start testing different models and parameter sets, you will need to split the data into training, validation, and testing datasets. Remember that after splitting the data into training and testing datasets, the validation dataset is created by splitting the training dataset.
# 
# The datasets X and y have been loaded for your use.

# ### init: 1 dataframe, 1 serie

# In[9]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X,y)
tobedownloaded="{pandas.core.frame.DataFrame: {'X.csv': 'https://file.io/sAkMXs'}, pandas.core.series.Series: {'y.csv': 'https://file.io/ludL04'}}"
prefix='data_from_datacamp/Chap2-Exercise1.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[10]:


import pandas as pd
X=pd.read_csv(prefix+'X.csv',index_col=0)
y=pd.read_csv(prefix+'y.csv',index_col=0, header=None,squeeze=True)


# ### code

# In[11]:


# Create temporary training and final testing datasets
X_temp, X_test, y_temp, y_test  =    train_test_split(X, y, test_size=0.2, random_state=1111)

# Create the final training and validation datasets
X_train, X_val, y_train, y_val =    train_test_split(X_temp, y_temp, test_size=0.25, random_state=1111)


# # Accuracy metrics: regression models
# 

# ## Mean absolute error
# Communicating modeling results can be difficult. However, most clients understand that on average, a predictive model was off by some number. This makes explaining the mean absolute error easy. For example, when predicting the number of wins for a basketball team, if you predict 42, and they end up with 40, you can easily explain that the error was two wins.
# 
# In this exercise, you are interviewing for a new position and are provided with two arrays. y_test, the true number of wins for all 30 NBA teams in 2017 and predictions, which contains a prediction for each team. To test your understanding, you are asked to both manually calculate the MAE and use sklearn.

# ### init: 2 arrays

# In[12]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(y_test, predictions)
tobedownloaded="{numpy.ndarray: {'predictions.csv': 'https://file.io/6x1eBd',  'y_test.csv': 'https://file.io/7OmmwQ'}}"
prefix='data_from_datacamp/Chap2-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[13]:


from uploadfromdatacamp import loadNDArrayFromCsv
import pandas as pd
y_test=loadNDArrayFromCsv(prefix+'y_test.csv')
predictions=loadNDArrayFromCsv(prefix+'predictions.csv')


# ### code

# In[14]:


from sklearn.metrics import mean_absolute_error

# Manually calculate the MAE
n = len(predictions)
mae_one = sum(abs(y_test - predictions)) / n
print('With a manual calculation, the error is {}'.format(mae_one))

# Use scikit-learn to calculate the MAE
mae_two = mean_absolute_error(y_test, predictions)
print('Using scikit-lean, the error is {}'.format(mae_two))


# ## Mean squared error
# Let's focus on the 2017 NBA predictions again. Every year, there are at least a couple of NBA teams that win way more games than expected. If you use the MAE, this accuracy metric does not reflect the bad predictions as much as if you use the MSE. Squaring the large errors from bad predictions will make the accuracy look worse.
# 
# In this example, NBA executives want to better predict team wins. You will use the mean squared error to calculate the prediction error. The actual wins are loaded as y_test and the predictions as predictions.

# ### code

# In[16]:


from sklearn.metrics import mean_squared_error

n = len(predictions)
# Finish the manual calculation of the MSE
mse_one = sum((y_test - predictions)**2) / n
print('With a manual calculation, the error is {}'.format(mse_one))

# Use the scikit-learn function to calculate MSE
mse_two = mean_squared_error(y_test, predictions)
print('Using scikit-lean, the error is {}'.format(mse_two))


# ## Performance on data subsets
# In professional basketball, there are two conferences, the East and the West. Coaches and fans often only care about how teams in their own conference will do this year.
# 
# You have been working on an NBA prediction model and would like to determine if the predictions were better for the East or West conference. You added a third array to your data called labels, which contains an "E" for the East teams, and a "W" for the West. y_test and predictions have again been loaded for your use.

# ### init: 1 array

# In[17]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(labels)
tobedownloaded="{numpy.ndarray: {'labels.csv': 'https://file.io/BVKz4T'}}"
prefix='data_from_datacamp/Chap2-Exercise2.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[20]:


from uploadfromdatacamp import loadNDArrayFromCsv
import pandas as pd
labels=loadNDArrayFromCsv(prefix+'labels.csv', dtype='str')


# ### code

# In[22]:


# Find the East conference teams
east_teams = labels == 'E'


# In[23]:


# Create arrays for the true and predicted values
true_east = y_test[east_teams]
preds_east = predictions[east_teams]


# In[24]:


# Print the accuracy metrics
print('The MAE for East teams is {}'.format(
    mean_absolute_error(true_east, preds_east)))


# In[25]:


west_error = 5.01


# In[26]:


# Print the West accuracy
print('The MAE for West conference is {}'.format(west_error))


# ![image.png](attachment:image.png)

# # Classification metrics
# 

# ## Confusion matrices
# Confusion matrices are a great way to start exploring your model's accuracy. They provide the values needed to calculate a wide range of metrics, including sensitivity, specificity, and the F1-score.
# 
# You have built a classification model to predict if a person has a broken arm based on an X-ray image. On the testing set, you have the following confusion matrix:
# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ### code

# In[28]:


# Calculate and print the accuracy
accuracy = (324 + 491) / (953)
print("The overall accuracy is {0: 0.2f}".format(accuracy))

# Calculate and print the precision
precision = (491) / (491 + 15)
print("The precision is {0: 0.2f}".format(precision))

# Calculate and print the recall
recall = (491) / (491 + 123)
print("The recall is {0: 0.2f}".format(recall))


# ## Confusion matrices, again
# Creating a confusion matrix in Python is simple. The biggest challenge will be making sure you understand the orientation of the matrix. This exercise makes sure you understand the sklearn implementation of confusion matrices. Here, you have created a random forest model using the tic_tac_toe dataset rfc to predict outcomes of 0 (loss) or 1 (a win) for Player One.
# 
# Note: If you read about confusion matrices on another website or for another programming language, the values might be reversed.

# ### init: 2 dataframes, 2 series

# In[37]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/VbhpWZ',  'X_train.csv': 'https://file.io/TE63Re'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/Kq5PHY',  'y_train.csv': 'https://file.io/FJglZh'}}"
prefix='data_from_datacamp/Chap2-Exercise3.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[38]:


import pandas as pd
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_train=pd.read_csv(prefix+'y_train.csv',index_col=0, header=None,squeeze=True)
y_test=pd.read_csv(prefix+'y_test.csv',index_col=0, header=None,squeeze=True)


# ### code

# In[39]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(max_depth=None, random_state=1111, n_estimators=500)
rfc.fit(X_train, y_train)


# In[41]:


from sklearn.metrics import confusion_matrix

# Create predictions
test_predictions = rfc.predict(X_test)

# Create and print the confusion matrix
cm = confusion_matrix(y_test, test_predictions)
print(cm)

# Print the true positives (actual 1s that were predicted 1s)
print("The number of true positives is: {}".format(cm[1, 1]))


# ## Precision vs. recall
# The accuracy metrics you use to evaluate your model should always be based on the specific application. For this example, let's assume you are a really sore loser when it comes to playing Tic-Tac-Toe, but only when you are certain that you are going to win.
# 
# Choose the most appropriate accuracy metric, either precision or recall, to complete this example. But remember, if you think you are going to win, you better win!
# 
# Use rfc, which is a random forest classification model built on the tic_tac_toe dataset.

# ### init: 2 dataframes, 2 lists

# In[47]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/daKnn7',  'X_train.csv': 'https://file.io/ziL6xl'}, list: {'y_test.txt': 'https://file.io/kDC4MO',  'y_train.txt': 'https://file.io/kypujf'}}"
prefix='data_from_datacamp/Chap2-Exercise3.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[49]:


from uploadfromdatacamp import loadListFromTxt
import pandas as pd
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_train=loadListFromTxt(prefix+'y_train.txt')
y_test=loadListFromTxt(prefix+'y_test.txt')


# In[51]:


rfc.fit(X_train, y_train)


# ### code

# In[52]:


from sklearn.metrics import precision_score

test_predictions = rfc.predict(X_test)

# Create precision or recall score based on the metric you imported
score = precision_score(y_test, test_predictions)

# Print the final result
print("The precision value is {0:.2f}".format(score))


# ![image.png](attachment:image.png)

# # The bias-variance tradeoff
# 

# ## Error due to under/over-fitting
# The candy dataset is prime for overfitting. With only 85 observations, if you use 20% for the testing dataset, you are losing a lot of vital data that could be used for modeling. Imagine the scenario where most of the chocolate candies ended up in the training data and very few in the holdout sample. Our model might only see that chocolate is a vital factor, but fail to find that other attributes are also important. In this exercise, you'll explore how using too many features (columns) in a random forest model can lead to overfitting.
# 
# A feature represents which columns of the data are used in a decision tree. The parameter max_features limits the number of features available.

# ### init: 2 dataframes, 2 arrays

# In[53]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/MRtGoK',  'X_train.csv': 'https://file.io/6VdyQw'}, numpy.ndarray: {'y_test.csv': 'https://file.io/FR6jTo',  'y_train.csv': 'https://file.io/XJRtVO'}}"
prefix='data_from_datacamp/Chap2-Exercise4.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[55]:


from uploadfromdatacamp import loadNDArrayFromCsv
import pandas as pd
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_train=loadNDArrayFromCsv(prefix+'y_train.csv', dtype='float')
y_test=loadNDArrayFromCsv(prefix+'y_test.csv', dtype='float')


# In[58]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae


# ### code
# 

# Create a random forest model with 25 trees, a random state of 1111, and max_features of 2. Read the print statements.

# In[60]:


# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=2)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies 
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))


# Set max_features to 11 (the number of columns in the dataset). Read the print statements.
# 
# 

# In[61]:


# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=11)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies 
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))


# Set max_features equal to 4. Read the print statements.
# 
# 

# In[62]:


# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=4)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies 
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))


# ![image.png](attachment:image.png)

# ## Am I underfitting?
# You are creating a random forest model to predict if you will win a future game of Tic-Tac-Toe. Using the tic_tac_toe dataset, you have created training and testing datasets, X_train, X_test, y_train, and y_test.
# 
# You have decided to create a bunch of random forest models with varying amounts of trees (1, 2, 3, 4, 5, 10, 20, and 50). The more trees you use, the longer your random forest model will take to run. However, if you don't use enough trees, you risk underfitting. You have created a for loop to test your model at the different number of trees.

# ### init : 2 dataframes, 2 series

# In[63]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/UXZJ82',  'X_train.csv': 'https://file.io/aYlLfV'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/RZeykk',  'y_train.csv': 'https://file.io/SMmqcY'}}"
prefix='data_from_datacamp/Chap2-Exercise4.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[64]:


import pandas as pd
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_train=pd.read_csv(prefix+'y_train.csv',index_col=0, header=None,squeeze=True)
y_test=pd.read_csv(prefix+'y_test.csv',index_col=0, header=None,squeeze=True)


# ### code

# In[65]:


from sklearn.metrics import accuracy_score

test_scores, train_scores = [], []
for i in [1, 2, 3, 4, 5, 10, 20, 50]:
    rfc = RandomForestClassifier(n_estimators=i, random_state=1111)
    rfc.fit(X_train, y_train)
    # Create predictions for the X_train and X_test datasets.
    train_predictions = rfc.predict(X_train)
    test_predictions = rfc.predict(X_test)
    # Append the accuracy score for the test and train predictions.
    train_scores.append(round(accuracy_score(y_train, train_predictions), 2))
    test_scores.append(round(accuracy_score(y_test, test_predictions), 2))
# Print the train and test scores.
print("The training scores were: {}".format(train_scores))
print("The testing scores were: {}".format(test_scores))


# In[ ]:




