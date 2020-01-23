#!/usr/bin/env python
# coding: utf-8

# # What is Keras?
# 

# # Your first neural network
# 

# ## Hello nets!
# You're going to build a simple neural network to get a feeling for how quickly it is to accomplish in Keras.
# 
# You will build a network that takes two numbers as input, passes them through a hidden layer of 10 neurons, and finally outputs a single non-constrained number.
# 
# A non-constrained output can be obtained by avoiding setting an activation function in the output layer. This is useful for problems like regression, when we want our output to be able to take any value.

# ### code

# In[3]:


# Import the Sequential model and Dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add an input layer and a hidden layer with 10 neurons
model.add(Dense(10, input_shape=(2,), activation="relu"))

# Add a 1-neuron output layer
model.add(Dense(1))

# Summarise your model
model.summary()


# ## Counting parameters
# You've just created a neural network. Create a new one now and take some time to think about the weights of each layer. The Keras Dense layer and the Sequential model are already loaded for you to use.
# 
# This is the network you will be creating:

# ### code

# In[4]:


# Instantiate a new Sequential model
model = Sequential()

# Add a Dense layer with five neurons and three inputs
model.add(Dense(5, input_shape=(3,), activation="relu"))

# Add a final Dense layer with one neuron and no activation
model.add(Dense(1))

# Summarize your model
model.summary()


# ## Build as shown!
# You will take on a final challenge before moving on to the next lesson. Build the network shown in the picture below. Prove your mastered Keras basics in no time!

# ### code

# In[5]:


from keras.models import Sequential
from keras.layers import Dense

# Instantiate a Sequential model
model = Sequential()

# Build the input and hidden layer
model.add(Dense(3, input_shape=(2,), activation="relu"))

# Add the ouput layer
model.add(Dense(1))


# # Surviving a meteor strike
# 

# ## Specifying a model
# You will build a simple regression model to forecast the orbit of the meteor!
# 
# Your training data consist of measurements taken at time steps from -10 minutes before the impact region to +10 minutes after. Each time step can be viewed as an X coordinate in our graph, which has an associated position Y for the meteor at that time step.
# 
# Note that you can view this problem as approximating a quadratic function via the use of neural networks.

# This data is stored in two numpy arrays: one called time_steps , containing the features, and another called y_positions, with the labels.
# 
# Feel free to look at these arrays in the console anytime, then build your model! Keras Sequential model and Dense layers are available for you to use.

# ### init

# In[6]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(time_steps, y_positions)
"""

tobedownloaded="""
{numpy.ndarray: {'time_steps.csv': 'https://file.io/FNc6kh',
  'y_positions.csv': 'https://file.io/cBAFRl'}}
"""
prefix='data_from_datacamp/Chap1-Exercise3.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
time_steps = loadNDArrayFromCsv(prefix+'time_steps.csv')
y_positions = loadNDArrayFromCsv(prefix+'y_positions.csv')


# ### code

# In[7]:


# Instantiate a Sequential model
model = Sequential()

# Add a Dense layer with 50 neurons and an input of 1 neuron
model.add(Dense(50, input_shape=(1,), activation='relu'))

# Add two Dense layers with 50 neurons and relu activation
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))

# End your model with a Dense layer and no activation
model.add(Dense(1))


# ## Training
# You're going to train your first model in this course, and for a good cause!
# 
# Remember that before training your Keras models you need to compile them. This can be done with the .compile() method. The .compile() method takes arguments such as the optimizer, used for weight updating, and the loss function, which is what we want to minimize. Training your model is as easy as calling the .fit() method, passing on the features, labels and number of epochs to train for.
# 
# The model you built in the previous exercise is loaded for you to use, along with the time_steps and y_positions data.

# ### code

# In[8]:


# Compile your model
model.compile(optimizer= 'adam', loss = 'mse')

print("Training started..., this can take a while:")

# Fit your model on your data for 30 epochs
model.fit(time_steps,y_positions, epochs = 30)

# Evaluate your model 
print("Final lost value:",model.evaluate(time_steps, y_positions))


# ## Predicting the orbit!
# You've already trained a model that approximates the orbit of the meteor approaching earth and it's loaded for you to use.
# 
# Since you trained your model for values between -10 and 10 minutes, your model hasn't yet seen any other values for different time steps. You will visualize how your model behaves on unseen data.
# 
# To see the source code of plot_orbit, type the following print(inspect.getsource(plot_orbit)) in the console.
# 
# Remember np.arange(x,y) produces a range of values from x to y-1.
# 
# Hurry up, you're running out of time!

# ### init

# In[11]:


import matplotlib.pyplot as plt
def plot_orbit(model_preds):
  axeslim = int(len(model_preds)/2)
  plt.plot(np.arange(-axeslim, axeslim + 1),np.arange(-axeslim, axeslim + 1)**2,color="mediumslateblue")
  plt.plot(np.arange(-axeslim, axeslim + 1),model_preds,color="orange")
  plt.axis([-40, 41, -5, 550])
  plt.legend(["Scientist's Orbit", 'Your orbit'],loc="lower left")
  plt.title("Model orbit vs Scientist's Orbit")
  plt.show()


# ### code

# In[13]:


# Predict the twenty minutes orbit
twenty_min_orbit = model.predict(np.arange(-10, 11))

# Plot the twenty minute orbit 
plot_orbit(twenty_min_orbit)


# In[14]:


# Predict the eighty minute orbit
eighty_min_orbit =  model.predict(np.arange(-40, 41))

# Plot the eighty minute orbit 
plot_orbit(eighty_min_orbit)


# In[ ]:




