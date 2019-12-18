#!/usr/bin/env python
# coding: utf-8

# # Informed Search: Coarse to Fine
# 

# ## Visualizing Coarse to Fine
# You're going to undertake the first part of a Coarse to Fine search. This involves analyzing the results of an initial random search that took place over a large search space, then deciding what would be the next logical step to make your hyperparameter search finer.
# 
# You have available:
# 
# - combinations_list - a list of the possible hyperparameter combinations the random search was undertaken on.
# - results_df - a DataFrame that has each hyperparameter combination and the resulting accuracy of all 500 trials. Each hyperparameter is a column, with the header the hyperparameter name.
# - visualize_hyperparameter() - a function that takes in a column of the DataFrame (as a string) and produces a scatter plot of this column's values compared to the accuracy scores. An example call of the function would be visualize_hyperparameter('accuracy')

# ### init: 1 dataframe, 1 list, visualize_hyperparameter

# In[1]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(results_df, combinations_list)
tobedownloaded="{pandas.core.frame.DataFrame: {'results_df.csv': 'https://file.io/rHSjOn'}, list: {'combinations_list.txt': 'https://file.io/I9bei1'}}"
prefix='data_from_datacamp/Chap4-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[2]:


from uploadfromdatacamp import loadListFromTxt
combinations_list=loadListFromTxt(prefix+'combinations_list.txt')


# In[3]:


import pandas as pd

results_df=pd.read_csv(prefix+'results_df.csv',index_col=0)


# In[4]:


#print_func(visualize_hyperparameter)
import matplotlib.pyplot as plt
def visualize_hyperparameter(name):
  plt.clf()
  plt.scatter(results_df[name],results_df['accuracy'], c=['blue']*500)
  plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
  plt.gca().set_ylim([0,100])
  plt.show()


# ### code

# In[6]:


# Confirm the size of the combinations_list
print(len(combinations_list))

# Sort the results_df by accuracy and print the top 10 rows
print(results_df.sort_values(by='accuracy', ascending=False).head(10))

# Confirm which hyperparameters were used in this search
print(results_df.columns)

# Call visualize_hyperparameter() with each hyperparameter in turn
visualize_hyperparameter('max_depth')
visualize_hyperparameter('min_samples_leaf')
visualize_hyperparameter('learn_rate')


# ## Coarse to Fine Iterations
# You will now visualize the first random search undertaken, construct a tighter grid and check the results. You will have available:
# 
# results_df - a DataFrame that has the hyperparameter combination and the resulting accuracy of all 500 trials. Only the hyperparameters that had the strongest visualizations from the previous exercise are included (max_depth and learn_rate)
# visualize_first() - This function takes no arguments but will visualize each of your hyperparameters against accuracy for your first random search.

# ### init: 1 dataframe, visualize_first()

# In[9]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(results_df)
tobedownloaded="{pandas.core.frame.DataFrame: {'results_df.csv': 'https://file.io/D2Qh8s'}}"
prefix='data_from_datacamp/Chap4-Exercise1.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[10]:


import pandas as pd

results_df=pd.read_csv(prefix+'results_df.csv',index_col=0)


# In[11]:


#print_func(visualize_first)
import matplotlib.pyplot as plt
def visualize_first():
  for name in results_df.columns[0:2]:
    plt.clf()
    plt.scatter(results_df[name],results_df['accuracy'], c=['blue']*500)
    plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
    plt.gca().set_ylim([0,100])
    x_line = 20
    if name == "learn_rate":
      	x_line = 1
    plt.axvline(x=x_line, color="red", linewidth=4)
    plt.show()


# ### code

# Use the visualize_first() function to check the values of max_depth and learn_rate that tend to perform better. A convenient red line will be added to make this explicit.

# In[12]:


visualize_first()


# Now create a more narrow grid search, testing for max_depth values between 1 and 20 and for 50 learning rates between 0.001 and 1.

# In[13]:


# Use the provided function to visualize the first results
# visualize_first()

# Create some combinations lists & combine
max_depth_list = list(range(1, 21))
learn_rate_list = np.linspace(0.001, 1, 50)


# In[14]:


def visualize_second():
  for name in results_df2.columns[0:2]:
    plt.clf()
    plt.scatter(results_df2[name],results_df2['accuracy'], c=['blue']*1000)
    plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
    plt.gca().set_ylim([0,100])
    plt.show()


# In[16]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(results_df2)
tobedownloaded="{pandas.core.frame.DataFrame: {'results_df2.csv': 'https://file.io/nJrq9l'}}"
prefix='data_from_datacamp/Chap4-Exercise1.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[17]:


import pandas as pd

results_df2=pd.read_csv(prefix+'results_df2.csv',index_col=0)


# We ran the 1,000 model grid search in the background based on those new combinations. Now use the visualize_second() function to visualize the second iteration (grid search) and see if there is any improved results.

# In[18]:


# Call the function to visualize the second results
visualize_second()


# ![image.png](attachment:image.png)

# # Informed Search: Bayesian Statistics
# 

# ## Bayes Rule in Python
# In this exercise you will undertake a practical example of setting up Bayes formula, obtaining new evidence and updating your 'beliefs' in order to get a more accurate result. The example will relate to the likelihood that someone will close their account for your online software product.
# 
# These are the probabilities we know:
# 
# 7% (0.07) of people are likely to close their account next month
# 15% (0.15) of people with accounts are unhappy with your product (you don't know who though!)
# 35% (0.35) of people who are likely to close their account are unhappy with your product

# ### code

# Assign the different probabilities (as decimals) to variables. p_unhappy is the likelihood someone is unhappy, p_unhappy_close is the probability that someone is unhappy with the product, given they are going to close their account.
# 

# In[19]:


# Assign probabilities to variables 
p_unhappy = 0.15
p_unhappy_close = 0.35


# Assign the probability that someone will close their account next month to the variable p_close as a decimal.

# In[20]:


# Probabiliy someone will close
p_close = 0.07


# You interview one of your customers and discover they are unhappy. What is the probability they will close their account, now that you know this evidence? Assign the result to p_close_unhappy and print it.

# In[21]:


# Probability unhappy person will close
p_close_unhappy = (p_close * p_unhappy_close) / p_unhappy
print(p_close_unhappy)


# ![image.png](attachment:image.png)

# ## Bayesian Hyperparameter tuning with Hyperopt
# In this example you will set up and run a bayesian hyperparameter optimization process using the package Hyperopt (already imported as hp for you). You will set up the domain (which is similar to setting up the grid for a grid search), then set up the objective function. Finally, you will run the optimizer over 20 iterations.
# 
# You will need to set up the domain using values:
# 
# - max_depth using quniform distribution (between 2 and 10, increasing by 2)
# - learning_rate using uniform distribution (0.001 to 0.9)
# 
# Note that for the purpose of this exercise, this process was reduced in data sample size and hyperopt & GBM iterations. If you are trying out this method by yourself on your own machine, try a larger search space, more trials, more cvs and a larger dataset size to really see this in action!

# ### init : hyperopt, 1 dataframe, 1 array

# In[8]:


from hyperopt import fmin, tpe, hp
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import cross_val_score


# In[1]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X_train,  y_train)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_train.csv': 'https://file.io/joKq3s'}, numpy.ndarray: {'y_train.csv': 'https://file.io/uWGVOz'}}"
prefix='data_from_datacamp/Chap4-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[11]:


import pandas as pd
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
from uploadfromdatacamp import loadNDArrayFromCsv
y_train=loadNDArrayFromCsv(prefix+'y_train.csv')


# ### code

# In[12]:


# Set up space dictionary with specified hyperparameters
space = {'max_depth': hp.quniform('max_depth', 2, 10, 2),'learning_rate': hp.uniform('learning_rate', 0.001,0.9)}

# Set up objective function
def objective(params):
    params = {'max_depth': int(params['max_depth']),'learning_rate': params['learning_rate']}
    gbm_clf = GradientBoostingClassifier(n_estimators=100, **params) 
    best_score = cross_val_score(gbm_clf, X_train, y_train, scoring='accuracy', cv=2, n_jobs=4).mean()
    loss = 1 - best_score
    return loss

# Run the algorithm
best = fmin(fn=objective,space=space, max_evals=20, rstate=np.random.RandomState(42), algo=tpe.suggest)
print(best)


# ![image.png](attachment:image.png)

# # Informed Search: Genetic Algorithms
# 

# In[1]:


from tpot import TPOTClassifier


# ## Genetic Hyperparameter Tuning with TPOT
# You're going to undertake a simple example of genetic hyperparameter tuning. TPOT is a very powerful library that has a lot of features. You're just scratching the surface in this lesson, but you are highly encouraged to explore in your own time.
# 
# This is a very small example. In real life, TPOT is designed to be run for many hours to find the best model. You would have a much larger population and offspring size as well as hundreds more generations to find a good model.
# 
# You will create the estimator, fit the estimator to the training data and then score this on the test data.
# 
# For this example we wish to use:
# 
# - 3 generations
# - 4 in the population size
# - 3 offspring in each generation
# - accuracy for scoring
# 
# A random_state of 2 has been set for consistency of results.

# ### init: 2 dataframes, 2 arrays

# In[3]:


#from uploadfromdatacamp import saveFromFileIO
#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/sKD2Cs',  'X_train.csv': 'https://file.io/78zPge'}, numpy.ndarray: {'y_test.csv': 'https://file.io/i4qBh3',  'y_train.csv': 'https://file.io/TMvOog'}}"
prefix='data_from_datacamp/Chap4-Exercise3.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[5]:


import pandas as pd
from uploadfromdatacamp import loadNDArrayFromCsv

X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')


# ### code

# In[8]:


# Assign the values outlined to the inputs
number_generations = 3
population_size = 4
offspring_size = 3
scoring_function = 'accuracy'

# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=number_generations, population_size=population_size,
                          offspring_size=offspring_size, scoring=scoring_function,
                          verbosity=2, random_state=2, cv=2)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


# ![image.png](attachment:image.png)

# ## Analysing TPOT's stability
# You will now see the random nature of TPOT by constructing the classifier with different random states and seeing what model is found to be best by the algorithm. This assists to see that TPOT is quite unstable when not run for a reasonable amount of time.

# ### code

# Create the TPOT classifier, fit to the data and score using a random_state of 42.

# In[9]:


# Create the tpot classifier 
tpot_clf = TPOTClassifier(generations=2, population_size=4, offspring_size=3, scoring='accuracy', cv=2,
                          verbosity=2, random_state=42)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


# Now try using a random_state of 122. The numbers don't mean anything special, but should produce different results.

# In[10]:


# Create the tpot classifier 
tpot_clf = TPOTClassifier(generations=2, population_size=4, offspring_size=3, scoring='accuracy', cv=2,
                          verbosity=2, random_state=122)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


# Finally try using the random_state of 99. See how there is a different result again?
# 

# In[11]:


# Create the tpot classifier 
tpot_clf = TPOTClassifier(generations=2, population_size=4, offspring_size=3, scoring='accuracy', cv=2,
                          verbosity=2, random_state=99)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


# ![image.png](attachment:image.png)

# In[ ]:




