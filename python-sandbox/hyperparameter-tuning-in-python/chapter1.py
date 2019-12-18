#!/usr/bin/env python
# coding: utf-8

# # Introduction & 'Parameters'
# 

# ## Parameters in Logistic Regression
# Now that you have had a chance to explore what a parameter is, let us apply this knowledge. It is important to be able to review any new algorithm and identify which elements are parameters and hyperparameters.
# 
# Which of the following is a parameter for the Scikit Learn logistic regression model?

# ![image.png](attachment:image.png)

# ## Extracting a Logistic Regression parameter
# You are now going to practice extracting an important parameter of the logistic regression model. The logistic regression has a few other parameters you will not explore here but you can review them in the scikit-learn.org documentation for the LogisticRegression() module under 'Attributes'.
# 
# This parameter is important for understanding the direction and magnitude of the effect the variables have on the target.
# 
# In this exercise we will extract the coefficient parameter (found in the coef_ attribute), zip it up with the original column names, and see which variables had the largest positive effect on the target variable.
# 
# You will have available:
# 
# A logistic regression model object named log_reg_clf
# The X_train DataFrame
# sklearn and pandas have been imported for you.

# ### init: 1 dataframe, sklearn, LogisticRegression (from pickle), pandas

# In[1]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_train.csv': 'https://file.io/tk4jXt'}}"
prefix='data_from_datacamp/Chap1-Exercise1.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[2]:


import pandas as pd
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)


# In[3]:


from sklearn.linear_model import LogisticRegression
log_reg_clf = LogisticRegression()


# In[9]:


filename = 'finalized_model.sav'
#pickle.dump(log_reg_clf, open(filename, 'wb'))
#uploadToFileIO_pushto_fileio(filename,proxy='')
import pickle
tobedownloaded='https://file.io/qgRvnh'
# load the model from disk
log_reg_clf = pickle.load(open('data_from_datacamp/'+filename, 'rb'))


# ### code

# In[11]:


# Create a list of original variable names from the training DataFrame
original_variables = X_train.columns

# Extract the coefficients of the logistic regression estimator
model_coefficients = log_reg_clf.coef_[0]

# Create a dataframe of the variables and coefficients & print it out
coefficient_df = pd.DataFrame({"Variable" : original_variables, "Coefficient": model_coefficients})
print(coefficient_df)

# Print out the top 3 positive variables
top_three_df = coefficient_df.sort_values(by=['Coefficient'], axis=0, ascending=False)[0:3]
print(top_three_df)


# ![image.png](attachment:image.png)

# ## Extracting a Random Forest parameter
# You will now translate the work previously undertaken on the logistic regression model to a random forest model. A parameter of this model is, for a given tree, how it decided to split at each level.
# 
# This analysis is not as useful as the coefficients of logistic regression as you will be unlikely to ever explore every split and every tree in a random forest model. However, it is a very useful exercise to peak under the hood at what the model is doing.
# 
# In this exercise we will extract a single tree from our random forest model, visualize it and programmatically extract one of the splits.
# 
# You have available:
# 
# - A random forest model object named rf_clf
# - An image of the chosen decision tree, tree_viz
# - The X_train DataFrame & the original_variables list

# ### init: 1 dataframe, 1 pandas.index, RandomForestClassifier, image of chosen decision tree

# In[15]:


from uploadfromdatacamp import saveFromFileIO

#tree_viz2D = tree_viz.reshape((778,2130*4))
#uploadToFileIO(X_train, tree_viz2D)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_train.csv': 'https://file.io/9zMdAC'}, numpy.ndarray: {'tree_viz2D.csv': 'https://file.io/6zT4Tt'}}"
prefix='data_from_datacamp/Chap1-Exercise1.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[16]:


import pandas as pd
from uploadfromdatacamp import loadNDArrayFromCsv

X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
tree_viz2D = loadNDArrayFromCsv(prefix+'tree_viz2D.csv')


# In[17]:


tree_viz=tree_viz2D.reshape((778,2130,4
                            ))


# In[18]:


original_variables=X_train.columns


# In[22]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(max_depth=4, n_estimators=10, n_jobs=None, random_state=None)

filename = 'rf_clf.pkl'
#pickle.dump(rf_clf, open(filename, 'wb'))
#uploadToFileIO_pushto_fileio(filename,proxy='')
import pickle
tobedownloaded="{manual_download: {'rf_clf.pkl': 'https://file.io/FU9od4'}}" 
prefix='data_from_datacamp/'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

# load the model from disk
rf_clf = pickle.load(open(prefix+filename, 'rb'))


# In[24]:


import matplotlib.pyplot as plt


# ### code

# In[30]:


# Extract the 7th (index 6) tree from the random forest
chosen_tree = rf_clf.estimators_[6]

# Visualize the graph using the provided image
fig, ax = plt.subplots(figsize=(20, 8))

imgplot = plt.imshow(tree_viz)
plt.show()

# Extract the parameters and level of the top (index 0) node
split_column = chosen_tree.tree_.feature[0]
split_column_name = X_train.columns[split_column]
split_value = chosen_tree.tree_.threshold[0]

# Print out the feature and level
print("This node split on feature {}, at a value of {}".format(split_column_name, split_value))


# # Introducing Hyperparameters
# 

# ## Hyperparameters in Random Forests
# As you saw, there are many different hyperparameters available in a Random Forest model using Scikit Learn. Here you can remind yourself how to differentiate between a hyperparameter and a parameter, and easily check whether something is a hyperparameter.
# 
# You can create a random forest estimator yourself from the imported Scikit Learn package. Then print this estimator out to see the hyperparameters and their values.
# 
# Which of the following is a hyperparameter for the Scikit Learn random forest model?

# ### code

# In[31]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()


# In[32]:


print(rf_clf)


# ![image.png](attachment:image.png)

# ## Exploring Random Forest Hyperparameters
# Understanding what hyperparameters are available and the impact of different hyperparameters is a core skill for any data scientist. As models become more complex, there are many different settings you can set, but only some will have a large impact on your model.
# 
# You will now assess an existing random forest model (it has some bad choices for hyperparameters!) and then make better choices for a new random forest model and assess its performance.
# 
# You will have available:
# 
# - X_train, X_test, y_train, y_test DataFrames
# - An existing pre-trained random forest estimator, rf_clf_old
# - The predictions of the existing random forest estimator on the test set, rf_old_predictions

# ### init: 2 dataframes, 3 arrays

# In[33]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, X_test, y_train, y_test, rf_old_predictions)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/ZaP7R1',  'X_train.csv': 'https://file.io/3q6HuL'}, numpy.ndarray: {'rf_old_predictions.csv': 'https://file.io/qTDX1d',  'y_test.csv': 'https://file.io/snmaka',  'y_train.csv': 'https://file.io/Yvb8O8'}}"
prefix='data_from_datacamp/Chap1-Exercise2.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[34]:


import pandas as pd
from uploadfromdatacamp import loadNDArrayFromCsv

X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')
rf_old_predictions = loadNDArrayFromCsv(prefix+'rf_old_predictions.csv')


# In[35]:


from sklearn.ensemble import RandomForestClassifier

filename = 'rf_clf_old.pkl'
#pickle.dump(rf_clf_old, open(filename, 'wb'))
#uploadToFileIO_pushto_fileio(filename,proxy='')
import pickle
tobedownloaded="{manual_download: {'rf_clf_old.pkl': 'https://file.io/vurS8w'}}" 
prefix='data_from_datacamp/'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

# load the model from disk
rf_clf_old = pickle.load(open(prefix+filename, 'rb'))


# ### code

# In[39]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[41]:


# Print out the old estimator, notice which hyperparameter is badly set
print(rf_clf_old)

# Get confusion matrix & accuracy for the old rf_model
print("Confusion Matrix: \n\n {} \n Accuracy Score: \n\n {}".format(confusion_matrix(y_test, rf_old_predictions),  accuracy_score(y_test, rf_old_predictions))) 


# Create a new random forest classifier with a better n_estimators (try 500) then fit this to the data and obtain predictions.

# In[43]:


# Create a new random forest classifier with better hyperparamaters
rf_clf_new = RandomForestClassifier(n_estimators=500)

# Fit this to the data and obtain predictions
rf_new_predictions = rf_clf_new.fit(X_train, y_train).predict(X_test)


# In[44]:


# Assess the new model
print("Confusion Matrix: \n\n", confusion_matrix(y_test, rf_new_predictions))
print("Accuracy Score: \n\n", accuracy_score(y_test, rf_new_predictions))


# ## Hyperparameters of KNN
# To apply the concepts learned in the prior exercise, it is good practice to try out learnings on a new algorithm. The k-nearest-neighbors algorithm is not as popular as it used to be but can still be an excellent choice for data that has groups of data that behave similarly. Could this be the case for our credit card users?
# 
# In this case you will try out several different values for one of the core hyperparameters for the knn algorithm and compare performance.
# 
# You will have available:
# 
# X_train, X_test, y_train, y_test DataFrames

# ### init: 2 dataframes, 2 arrays

# In[45]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/iNRhpq',  'X_train.csv': 'https://file.io/wpF9s3'}, numpy.ndarray: {'y_test.csv': 'https://file.io/oJcNcR',  'y_train.csv': 'https://file.io/dl5hP9'}}"
prefix='data_from_datacamp/Chap1-Exercise2.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[46]:


import pandas as pd
from uploadfromdatacamp import loadNDArrayFromCsv

X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')


# In[48]:


from sklearn.neighbors import KNeighborsClassifier


# ### code

# In[49]:


# Build a knn estimator for each value of n_neighbours
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_20 = KNeighborsClassifier(n_neighbors=20)

# Fit each to the training data & produce predictions
knn_5_predictions = knn_5.fit(X_train, y_train).predict(X_test)
knn_10_predictions = knn_10.fit(X_train, y_train).predict(X_test)
knn_20_predictions = knn_20.fit(X_train, y_train).predict(X_test)

# Get an accuracy score for each of the models
knn_5_accuracy = accuracy_score(y_test, knn_5_predictions)
knn_10_accuracy = accuracy_score(y_test, knn_10_predictions)
knn_20_accuracy = accuracy_score(y_test, knn_20_predictions)
print("The accuracy of 5, 10, 20 neighbours was {}, {}, {}".format(knn_5_accuracy, knn_10_accuracy, knn_20_accuracy))


# # Setting & Analyzing Hyperparameter Values
# 

# ## Automating Hyperparameter Choice
# Finding the best hyperparameter of interest without writing hundreds of lines of code for hundreds of models is an important efficiency gain that will greatly assist your future machine learning model building.
# 
# An important hyperparameter for the GBM algorithm is the learning rate. But which learning rate is best for this problem? By writing a loop to search through a number of possibilities, collating these and viewing them you can find the best one.
# 
# Possible learning rates to try include 0.001, 0.01, 0.05, 0.1, 0.2 and 0.5
# 
# You will have available X_train, X_test, y_train & y_test datasets, and GradientBoostingClassifier has been imported for you.

# ### init: 2 dataframes, 2 arrays

# In[50]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/ExyoQH',  'X_train.csv': 'https://file.io/aXpdge'}, numpy.ndarray: {'y_test.csv': 'https://file.io/Gwj3z7',  'y_train.csv': 'https://file.io/mLUq2a'}}"
prefix='data_from_datacamp/Chap1-Exercise3.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[51]:


import pandas as pd
from uploadfromdatacamp import loadNDArrayFromCsv

X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')


# In[52]:


from sklearn.ensemble import GradientBoostingClassifier


# ### code

# In[53]:


# Set the learning rates & results storage
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2 ,0.5]
results_list = []

# Create the for loop to evaluate model predictions for each learning rate
for learning_rate in learning_rates:
    model = GradientBoostingClassifier(learning_rate=learning_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    # Save the learning rate and accuracy score
    results_list.append([learning_rate, accuracy_score(y_test, predictions)])

# Gather everything into a DataFrame
results_df = pd.DataFrame(results_list, columns=['learning_rate', 'accuracy'])
print(results_df)


# ## Building Learning Curves
# If we want to test many different values for a single hyperparameter it can be difficult to easily view that in the form of a DataFrame. Previously you learned about a nice trick to analyze this. A graph called a 'learning curve' can nicely demonstrate the effect of increasing or decreasing a particular hyperparameter on the final result.
# 
# Instead of testing only a few values for the learning rate, you will test many to easily see the effect of this hyperparameter across a large range of values. A useful function from NumPy is np.linspace(start, end, num) which allows you to create a number of values (num) evenly spread within an interval (start, end) that you specify.
# 
# You will have available X_train, X_test, y_train & y_test datasets.

# ### code

# - Create a list of 30 learning rates evenly spread between 0.01 and 2.
# - Create a similar loop to last exercise but just save out accuracy scores to a list.
# - Plot the learning rates against the accuracy score.
# 

# In[54]:


# Set the learning rates & accuracies list
learn_rates = np.linspace(0.01, 2, num=30)
accuracies = []

# Create the for loop
for learn_rate in learn_rates:
  	# Create the model, predictions & save the accuracies as before
    model = GradientBoostingClassifier(learning_rate=learn_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

# Plot results    
plt.plot(learn_rates, accuracies)
plt.gca().set(xlabel='learning_rate', ylabel='Accuracy', title='Accuracy for different learning_rates')
plt.show()


# ![image.png](attachment:image.png)

# In[ ]:




