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


# In[3]:


model.summary()


# ## Prepare your dataset
# In the console you can check that your labels, darts.competitor are not yet in a format to be understood by your network. They contain the names of the competitors as strings. You will first turn these competitors into unique numbers,then use the to_categorical() function from keras.utils to turn these numbers into their one-hot encoded representation.
# 
# This is useful for multi-class classification problems, since there are as many output neurons as classes and for every observation in our dataset we just want one of the neurons to be activated.
# 
# The dart's dataset is loaded as darts. Pandas is imported as pd. Let's prepare this dataset!

# ### init

# In[4]:


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

# In[5]:


import keras
import pandas as pd


# In[6]:


# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes 

# Print the label encoded competitors
print('Label encoded competitors: \n',darts.competitor.head())


# In[7]:


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

# In[8]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(coord_train, competitors_train, coord_test, competitors_test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'coord_test.csv': 'https://file.io/sEhbzO',
  'coord_train.csv': 'https://file.io/JMKwDm'},
 numpy.ndarray: {'competitors_test.csv': 'https://file.io/coqrtC',
  'competitors_train.csv': 'https://file.io/FNOLXS'}}
"""
prefix='data_from_datacamp/Chap2-Exercise1.6_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
coord_test = pd.read_csv(prefix+'coord_test.csv',index_col=0)
coord_train = pd.read_csv(prefix+'coord_train.csv',index_col=0)
from uploadfromdatacamp import loadNDArrayFromCsv
competitors_test = loadNDArrayFromCsv(prefix+'competitors_test.csv')
competitors_train = loadNDArrayFromCsv(prefix+'competitors_train.csv')



# ### code

# In[11]:


# Train your model on the training data for 200 epochs
model.fit(coord_train,competitors_train,epochs=200)

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(coord_test, competitors_test)[1]

# Print accuracy
print('Accuracy:', accuracy)


# ## Softmax predictions
# Your recently trained model is loaded for you. This model is generalizing well!, that's why you got a high accuracy on the test set.
# 
# Since you used the softmax activation function, for every input of 2 coordinates provided to your model there's an output vector of 4 numbers. Each of these numbers encodes the probability of a given dart being thrown by one of the 4 possible competitors.
# 
# When computing accuracy with the model's .evaluate() method, your model takes the class with the highest probability as the prediction. np.argmax() can help you do this since it returns the index with the highest value in an array.
# 
# Use the collection of test throws stored in coords_small_test and np.argmax()to check this out!

# ### init

# In[14]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(coords_small_test, competitors_small_test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'coords_small_test.csv': 'https://file.io/Fw7Guv'},
 numpy.ndarray: {'competitors_small_test.csv': 'https://file.io/68SB0B'}}
"""
prefix='data_from_datacamp/Chap2-Exercise1.7_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
coords_small_test = pd.read_csv(prefix+'coords_small_test.csv',index_col=0)
from uploadfromdatacamp import loadNDArrayFromCsv
competitors_small_test = loadNDArrayFromCsv(prefix+'competitors_small_test.csv')


# ### code

# In[15]:


# Predict on coords_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))


# In[16]:


# Extract the indexes of the highest probable predictions
preds = [np.argmax(pred) for pred in preds]

# Print preds vs true values
print("{:10} | {}".format('Rounded Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{:25} | {}".format(pred,competitors_small_test[i]))


# # Multi-label classification
# 

# ## An irrigation machine
# You're going to automate the watering of parcels by making an intelligent irrigation machine. Multi-label classification problems differ from multi-class problems in that each observation can be labeled with zero or more classes. So classes are not mutually exclusive.
# 
# To account for this behavior what we do is have an output layer with as many neurons as classes but this time, unlike in multi-class problems, each output neuron has a sigmoid activation function. This makes the output layer able to output a number between 0 and 1 in any of its neurons.
# 
# Keras Sequential() model and Dense() layers are preloaded. It's time to build an intelligent irrigation machine!

# ### code

# - Instantiate a Sequential() model.
# - Add a hidden layer of 64 neurons with as many input neurons as there are sensors and relu activation.
# - Add an output layer with as many neurons as parcels and sigmoidactivation.
# - Compile your model with adam and binary_crossentropy loss.

# In[18]:


# Instantiate a Sequential model
model = Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(64, input_shape=(20,), activation='relu'))

# Add an output layer of 3 neurons with sigmoid activation
model.add(Dense(3, activation='sigmoid'))

# Compile your model with adam and binary crossentropy loss
model.compile(optimizer='adam',
           loss='binary_crossentropy',
           metrics=['accuracy'])

model.summary()


# ## Training with multiple labels
# An output of your multi-label model could look like this: [0.76 , 0.99 , 0.66 ]. If we round up probabilities higher than 0.5, this observation will be classified as containing all 3 possible labels [1,1,1]. For this particular problem, this would mean watering all 3 parcels in your field is the right thing to do given the input sensor measurements.
# 
# You will now train and predict with the model you just built. sensors_train, parcels_train, sensors_test and parcels_test are already loaded for you to use. Let's see how well your machine performs!

# ### init

# In[19]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(sensors_train, parcels_train, sensors_test , parcels_test)
"""

tobedownloaded="""
{numpy.ndarray: {'parcels_test.csv': 'https://file.io/Hsuc4N',
  'parcels_train.csv': 'https://file.io/nlHd4z',
  'sensors_test.csv': 'https://file.io/Gkwpyh',
  'sensors_train.csv': 'https://file.io/HhouaJ'}}
"""
prefix='data_from_datacamp/Chap2-Exercise2.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
parcels_test = loadNDArrayFromCsv(prefix+'parcels_test.csv')
parcels_train = loadNDArrayFromCsv(prefix+'parcels_train.csv')
sensors_test = loadNDArrayFromCsv(prefix+'sensors_test.csv')
sensors_train = loadNDArrayFromCsv(prefix+'sensors_train.csv')


# ### code

# In[25]:


# Train for 100 epochs using a validation split of 0.2
model.fit(sensors_train, parcels_train, epochs = 100, validation_split = 0.2)

# Predict on sensors_test and round up the predictions
preds = model.predict(sensors_test)
preds_rounded = np.round(preds)

# Print rounded preds
print('Rounded Predictions: \n', preds_rounded)

# Evaluate your model's accuracy on the test data
accuracy = model.evaluate(sensors_test, parcels_test)[1]

# Print accuracy
print('Accuracy:', accuracy)


# # Keras callbacks
# 

# ## The history callback
# The history callback is returned by default every time you train a model with the .fit() method. To access these metrics you can access the history dictionary inside the returned callback object and the corresponding keys.
# 
# The irrigation machine model you built in the previous lesson is loaded for you to train, along with its features and labels (X and y). This time you will store the model's historycallback and use the validation_data parameter as it trains.
# 
# You will plot the results stored in history with plot_accuracy() and plot_loss(), two simple matplotlib functions. You can check their code in the console by typing print(inspect.getsource(plot_loss)).
# 
# Let's see the behind the scenes of our training!

# ### init

# In[26]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_train, y_train, X_test, y_test)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test.csv': 'https://file.io/6rioyr',
  'X_train.csv': 'https://file.io/im3td2',
  'y_test.csv': 'https://file.io/NGEAO4',
  'y_train.csv': 'https://file.io/C8dgEs'}}
"""
prefix='data_from_datacamp/Chap1-Exercise3.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X_test = loadNDArrayFromCsv(prefix+'X_test.csv')
X_train = loadNDArrayFromCsv(prefix+'X_train.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')


# In[27]:


import matplotlib.pyplot as plt
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(plot_loss)
"""
def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()

    
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(plot_accuracy)
"""   

def plot_accuracy(acc,val_acc):
  # Plot training & validation accuracy values
  plt.figure()
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()


# ### code

# In[28]:


# Train your model and save it's history
history = model.fit(X_train, y_train, epochs = 50,
               validation_data=(X_test, y_test))

# Plot train vs test loss during training
plot_loss(history.history['loss'], history.history['val_loss'])

# Plot train vs test accuracy during training
plot_accuracy(history.history['acc'], history.history['val_acc'])


# In[29]:


model.summary()


# ## Early stopping your model
# The early stopping callback is useful since it allows for you to stop the model training if it no longer improves after a given number of epochs. To make use of this functionality you need to pass the callback inside a list to the model's callback parameter in the .fit() method.
# 
# The model you built to detect fake dollar bills is loaded for you to train, this time with early stopping. X_train, y_train, X_test and y_test are also available for you to use.

# ### init

# In[30]:


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


# In[31]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_train, y_train, X_test, y_test)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test.csv': 'https://file.io/A9eTiX',
  'X_train.csv': 'https://file.io/wSe8NV',
  'y_test.csv': 'https://file.io/s4rYWQ',
  'y_train.csv': 'https://file.io/4VMwvv'}}
"""
prefix='data_from_datacamp/Chap1-Exercise3.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X_test = loadNDArrayFromCsv(prefix+'X_test.csv')
X_train = loadNDArrayFromCsv(prefix+'X_train.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')


# ### code

# In[32]:


# Import the early stopping callback
from keras.callbacks import EarlyStopping

# Define a callback to monitor val_acc
monitor_val_acc = EarlyStopping(monitor='val_acc', 
                       patience=5)

# Train your model using the early stopping callback
model.fit(X_train, y_train, 
           epochs=1000, validation_data=(X_test, y_test),
           callbacks=[monitor_val_acc])


# ## A combination of callbacks
# Deep learning models can take a long time to train, especially when you move to deeper architectures and bigger datasets. Saving your model every time it improves as well as stopping it when it no longer does allows you to worry less about choosing the number of epochs to train for. You can also restore a saved model anytime.
# 
# The model training and validation data are available in your workspace as X_train, X_test, y_train, and y_test.
# 
# Use the EarlyStopping() and the ModelCheckpoint() callbacks so that you can go eat a jar of cookies while you leave your computer to work!

# ### code

# In[33]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[34]:


# Import the EarlyStopping and ModelCheckpoint callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor = 'val_acc', patience=3)

# Save the best model as best_banknote_model.hdf5
modelCheckpoint = ModelCheckpoint('best_banknote_model.hdf5', save_best_only = True)

# Fit your model for a stupid amount of epochs
history = model.fit(X_train, y_train,
                    epochs = 10000000,
                    callbacks = [monitor_val_acc, modelCheckpoint],
                    validation_data = (X_test, y_test))


# In[ ]:




