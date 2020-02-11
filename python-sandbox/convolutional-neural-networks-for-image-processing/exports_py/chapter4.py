#!/usr/bin/env python
# coding: utf-8

# # Tracking learning
# 

# ## Plot the learning curves
# During learning, the model will store the loss function evaluated in each epoch. Looking at the learning curves can tell us quite a bit about the learning process. In this exercise, you will plot the learning and validation loss curves for a model that you will train.

# ### init

# In[2]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(model)
"""

tobedownloaded="""
{keras.engine.sequential.Sequential: {'model.h5': 'https://file.io/E4OdhL'}}
"""
prefixToc = '1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc,  proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import loadModelFromH5
model = loadModelFromH5(prefix+'model.h5')


# In[3]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO( train_data , train_labels, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'train_data[50_28_28_1].csv': 'https://file.io/fmE3XW',
  'train_labels[50_3].csv': 'https://file.io/SZYWO5'}}
"""
prefixToc = '1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
train_data = getImage(prefix+'train_data[50_28_28_1].csv', dtype='float64')
train_labels = getImage(prefix+'train_labels[50_3].csv', dtype='float64')


# In[4]:


import matplotlib.pyplot as plt

def show_image(image, title='Image', cmap_type='gray', interpolation=None):
    plt.imshow(image, cmap=cmap_type)    
    plt.title(title)
    plt.axis('off')
    plt.show()


# In[5]:


show_image(train_data[49,:,:,0])


# ### code

# In[7]:


import matplotlib.pyplot as plt

# Train the model and store the training object
training = model.fit(train_data, train_labels, validation_split=0.2, epochs=3, batch_size=10)

# Extract the history from the training object
history = training.history

# Plot the training loss 
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

# Show the figure
plt.show()


# ## Using stored weights to predict in a test set
# Model weights stored in an hdf5 file can be reused to populate an untrained model. Once the weights are loaded into this model, it behaves just like a model that has been trained to reach these weights. For example, you can use this model to make predictions from an unseen data set (e.g. test_data).

# ### init

# In[8]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('weights.hdf5')
"""

tobedownloaded="""
{numpy.ndarray: {'weights.hdf5': 'https://file.io/2bILQA'}}
"""
prefixToc = '1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# In[9]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO( test_data ,  image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'test_data[10_28_28_1].csv': 'https://file.io/l43ApV'}}
"""
prefixToc = '1.2'
prefix2 = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
test_data = getImage(prefix2+'test_data[10_28_28_1].csv', dtype='float64')


# ### code

# In[17]:


from keras.models import Sequential, load_model

# Load the weights from file
model = load_model(prefix+'weights.hdf5')
model.load_weights(prefix+'weights.hdf5')

# Predict from the first three images in the test data
model.predict(test_data[:3])


# In[20]:


show_image(test_data[0,:,:,0])


# In[21]:


show_image(test_data[1,:,:,0])


# In[22]:


show_image(test_data[2,:,:,0])


# # Regularization

# ## Adding dropout to your network
# Dropout is a form of regularization that removes a different random subset of the units in a layer in each round of training. In this exercise, we will add dropout to the convolutional neural network that we have used in previous exercises:
# 
# - Convolution (15 units, kernel size 2, 'relu' activation)
# - Dropout (20%)
# - Convolution (5 units, kernel size 2, 'relu' activation)
# - Flatten
# - Dense (3 units, 'softmax' activation)
# 
# A Sequential model along with Dense, Conv2D, Flatten, and Dropout objects are available in your workspace.

# ### code

# In[24]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
model = Sequential()

img_rows, img_cols = (28,28)


# In[25]:


# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))

# Add a dropout layer
model.add(Dropout(0.20))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


# ## Add batch normalization to your network
# Batch normalization is another form of regularization that rescales the outputs of a layer to make sure that they have mean 0 and standard deviation 1. In this exercise, we will add batch normalization to the convolutional neural network that we have used in previous exercises:
# 
# - Convolution (15 units, kernel size 2, 'relu' activation)
# - Batch normalization
# - Convolution (5 unites, kernel size 2, 'relu' activation)
# - Flatten
# - Dense (3 units, 'softmax' activation)
# 
# A Sequential model along with Dense, Conv2D, Flatten, and Dropout objects are available in your workspace.

# ### code

# In[27]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
model = Sequential()

img_rows, img_cols = (28,28)


# In[28]:


# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', input_shape=(img_rows, img_cols, 1)))


# Add batch normalization layer
model.add(BatchNormalization())

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


# # Interpreting the model
# 

# ## Extracting a kernel from a trained network
# One way to interpret models is to examine the properties of the kernels in the convolutional layers. In this exercise, you will extract one of the kernels from a convolutional neural network with weights that you saved in a hdf5 file.

# ### init

# In[29]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('weights.hdf5')
"""

tobedownloaded="""
{numpy.ndarray: {'weights.hdf5': 'https://file.io/5TZVjK'}}
"""
prefixToc = '1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# ### code

# In[30]:


from keras.models import Sequential, load_model

# Load the weights from file
model = load_model(prefix+'weights.hdf5')

# Get the first convolutional layer from the model
c1 = model.layers[0]

# Get the weights of the first convolutional layer
weights1 = c1.get_weights()

# Pull out the first channel of the first kernel in the first layer
kernel = weights1[0][...,0, 0]
print(kernel)


# ## Visualizing kernel responses
# One of the ways to interpret the weights of a neural network is to see how the kernels stored in these weights "see" the world. That is, what properties of an image are emphasized by this kernel. In this exercise, we will do that by convolving an image with the kernel and visualizing the result. Given images in the test_data variable, a function called extract_kernel() that extracts a kernel from the provided network, and the function called convolution() that we defined in the first chapter, extract the kernel, load the data from a file and visualize it with matplotlib.
# 
# A deep CNN model, a function convolution(), along with the kernel you extracted in an earlier exercise is available in your workspace.
# 
# Ready to take your deep learning to the next level? Check out Advanced Deep Learning with Keras in Python to see how the Keras functional API lets you build domain knowledge to solve new types of problems.

# ### init

# In[31]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(model)
"""

tobedownloaded="""
{keras.engine.sequential.Sequential: {'model.h5': 'https://file.io/V0ekQV'}}
"""
prefixToc = '3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc,  proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import loadModelFromH5
model = loadModelFromH5(prefix+'model.h5')


# In[32]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO( test_data ,  image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'test_data[10_28_28_1].csv': 'https://file.io/FaUrVD'}}
"""
prefixToc = '3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
test_data = getImage(prefix+'test_data[10_28_28_1].csv', dtype='float64')


# In[33]:


def convolution(image, kernel):
    kernel = kernel - kernel.mean()
    result = np.zeros(image.shape)

    for ii in range(image.shape[0]-2):
        for jj in range(image.shape[1]-2):
            result[ii, jj] = np.sum(image[ii:ii+2, jj:jj+2] * kernel)

    return result



# ### code

# In[34]:


import matplotlib.pyplot as plt

# Convolve with the fourth image in test_data
out = convolution(test_data[3, :, :, 0], kernel)

# Visualize the result
plt.imshow(out)
plt.show()


# In[ ]:




