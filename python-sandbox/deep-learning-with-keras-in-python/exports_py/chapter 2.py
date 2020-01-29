#!/usr/bin/env python
# coding: utf-8

# # Binary classification

# ## Exploring dollar bills
# You will practice building classification models in Keras with the Banknote Authentication dataset.
# 
# Your goal is to distinguish between real and fake dollar bills. In order to do this, the dataset comes with 4 variables: variance,skewness,curtosis and entropy. These variables are calculated by applying mathematical operations over the dollar bill images. The labels are found in the class variable.
# 
# 
# The dataset is pre-loaded in your workspace as banknotes, let's do some data exploration!

# ### init

# In[1]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(banknotes)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'banknotes.csv': 'https://file.io/fHmR9K'}}
"""
prefix='data_from_datacamp/Chap2-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
banknotes = pd.read_csv(prefix+'banknotes.csv',index_col=0)


# ### code

# In[4]:


import matplotlib.pyplot as plt


# In[5]:


# Import seaborn
import seaborn as sns

# Use pairplot and set the hue to be our class
sns.pairplot(banknotes, hue='class') 

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', banknotes.describe())

# Count the number of observations of each class
print('Observations per class: \n', banknotes['class'].value_counts())


# ## A binary classification model
# Now that you know what the Banknote Authentication dataset looks like, we'll build a simple model to distinguish between real and fake bills.
# 
# You will perform binary classification by using a single neuron as an output. The input layer will have 4 neurons since we have 4 features in our dataset. The model output will be a value constrained between 0 and 1.
# 
# We will interpret this number as the probability of our input variables coming from a fake dollar bill, with 1 meaning we are certain it's fake.

# ### code

# In[6]:


# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a sequential model
model = Sequential()

# Add a dense layer 
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))

# Compile your model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Display a summary of your model
model.summary()


# ## Is this dollar bill fake ?
# You are now ready to train your model and check how well it performs when classifying new bills! The dataset has already been partitioned as X_train, X_test,y_train and y_test.

# ### init

# In[7]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_train, y_train, X_test, y_test)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test.csv': 'https://file.io/DVXWIF',
  'X_train.csv': 'https://file.io/MnMhX3',
  'y_test.csv': 'https://file.io/p98Xak',
  'y_train.csv': 'https://file.io/lB6OLl'}}
"""
prefix='data_from_datacamp/Chap2-Exercise1.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X_train = loadNDArrayFromCsv(prefix+'X_train.csv')
X_test = loadNDArrayFromCsv(prefix+'X_test.csv')
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')


# ### code

# In[8]:


# Train your model for 20 epochs
model.fit(X_train, y_train, epochs=20)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test, y_test)[1]

# Print accuracy
print('Accuracy:',accuracy)


# ## A multi-class model
# You're going to build a model that predicts who threw which dart only based on where that dart landed! (That is the dart's x and y coordinates.)
# 
# This problem is a multi-class classification problem since each dart can only be thrown by one of 4 competitors. So classes are mutually exclusive, and therefore we can build a neuron with as many output as competitors and use the softmax activation function to achieve a total sum of probabilities of 1 over all competitors.
# 
# Keras Sequential model and Dense layer are already loaded for you to use.

# ### code

# In[1]:


# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


# Instantiate a sequential model
model = Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
  
# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation='softmax'))
  
# Compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# ## Prepare your dataset
# In the console you can check that your labels, darts.competitor are not yet in a format to be understood by your network. They contain the names of the competitors as strings. You will first turn these competitors into unique numbers,then use the to_categorical() function from keras.utils to turn these numbers into their one-hot encoded representation.
# 
# This is useful for multi-class classification problems, since there are as many output neurons as classes and for every observation in our dataset we just want one of the neurons to be activated.
# 
# The dart's dataset is loaded as darts. Pandas is imported as pd. Let's prepare this dataset!

# ### init

# In[3]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(darts)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'darts.csv': 'https://file.io/GfUaR9'}}
"""
prefix='data_from_datacamp/Chap2-Exercise1.5_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
darts = pd.read_csv(prefix+'darts.csv',index_col=0)


# ### code

# In[9]:


import keras
import pandas as pd


# In[13]:


# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes 

# Print the label encoded competitors
print('Label encoded competitors: \n',darts.competitor.head())


# In[15]:


# Import to_categorical from keras utils module
from keras.utils import to_categorical

# Use to_categorical on your labels
coordinates = darts.drop(['competitor'], axis=1)
competitors = to_categorical(darts.competitor)

# Now print the to_categorical() result
print('One-hot encoded competitors: \n',competitors)


# ## Training on dart throwers
# Your model is now ready, just as your dataset. It's time to train!
# 
# The coordinates and competitors variables you just transformed have been partitioned into coord_train,competitors_train, coord_test and competitors_test. Your model is also loaded. Feel free to visualize your training data or model.summary() in the console.

# ### init

# In[ ]:




