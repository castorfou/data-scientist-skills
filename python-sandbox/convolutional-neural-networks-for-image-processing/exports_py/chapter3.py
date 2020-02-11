#!/usr/bin/env python
# coding: utf-8

# # Going deeper

# ## Creating a deep learning network
# A deep convolutional neural network is a network that has more than one layer. Each layer in a deep network receives its input from the preceding layer, with the very first layer receiving its input from the images used as training or test data.
# 
# Here, you will create a network that has two convolutional layers.

# ### code

# In[1]:


img_rows, img_cols = (28,28)


# In[3]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

model = Sequential()

# Add a convolutional layer (15 units)
model.add(Conv2D(15, kernel_size=2, activation='relu', input_shape=(img_rows, img_cols, 1), padding='same'))


# Add another convolutional layer (5 units)
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


# ## Train a deep CNN to classify clothing images
# Training a deep learning model is very similar to training a single layer network. Once the model is constructed (as you have done in the previous exercise), the model needs to be compiled with the right set of parameters. Then, the model is fit by providing it with training data, as well as training labels. After training is done, the model can be evaluated on test data.
# 
# The model you built in the previous exercise is available in your workspace.

# ### init

# In[11]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(train_data, train_labels, test_data , test_labels, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'test_data[10_28_28_1].csv': 'https://file.io/HwAXmA',
  'test_labels[10_3].csv': 'https://file.io/h0icLz',
  'train_data[50_28_28_1].csv': 'https://file.io/y5bzoT',
  'train_labels[50_3].csv': 'https://file.io/kiiiaS'}}
"""
prefixToc = '1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
test_data = getImage(prefix+'test_data[10_28_28_1].csv', dtype='float64')
test_labels = getImage(prefix+'test_labels[10_3].csv', dtype='float64')
train_data = getImage(prefix+'train_data[50_28_28_1].csv', dtype='float64')
train_labels = getImage(prefix+'train_labels[50_3].csv', dtype='float64')


# In[12]:


import matplotlib.pyplot as plt

def show_image(image, title='Image', cmap_type='gray', interpolation=None):
    plt.imshow(image, cmap=cmap_type)    
    plt.title(title)
    plt.axis('off')
    plt.show()


# In[13]:


show_image(train_data[49,:,:,0])


# ### code

# In[14]:


# Compile model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit the model to training data 
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=3, batch_size=10)

# Evaluate the model on test data
model.evaluate(test_data, test_labels, batch_size=10)


# # How many parameters?
# 

# ## How many parameters in a deep CNN?
# In this exercise, you will use Keras to calculate the total number of parameters along with the number of parameters in each layer of the network.
# 
# We have already provided code that builds a deep CNN for you.

# ### code

# In[15]:


# CNN model
model = Sequential()
model.add(Conv2D(10, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))
model.add(Conv2D(10, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Summarize the model 
model.summary()


# # Pooling operations
# 

# ## Write your own pooling operation
# As we have seen before, CNNs can have a lot of parameters. Pooling layers are often added between the convolutional layers of a neural network to summarize their outputs in a condensed manner, and reduce the number of parameters in the next layer in the network. This can help us if we want to train the network more rapidly, or if we don't have enough data to learn a very large number of parameters.
# 
# A pooling layer can be described as a particular kind of convolution. For every window in the input it finds the maximal pixel value and passes only this pixel through. In this exercise, you will write your own max pooling operation, based on the code that you previously used to write a two-dimensional convolution operation.

# ### init

# In[16]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(im)
"""

tobedownloaded="""
{numpy.ndarray: {'im.csv': 'https://file.io/0ux0X7'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import loadNDArrayFromCsv
im = loadNDArrayFromCsv(prefix+'im.csv')


# In[17]:


show_image(im)


# ### code

# In[18]:


# Result placeholder
result = np.zeros((im.shape[0]//2, im.shape[1]//2))

# Pooling operation
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii*2:ii*2+2,jj*2:jj*2+2])


# ## Keras pooling layers
# Keras implements a pooling operation as a layer that can be added to CNNs between other layers. In this exercise, you will construct a convolutional neural network similar to the one you have constructed before:
# 
# Convolution => Convolution => Flatten => Dense
# 
# However, you will also add a pooling layer. The architecture will add a single max-pooling layer between the convolutional layer and the dense layer with a pooling of 2x2:
# 
# Convolution => Max pooling => Convolution => Flatten => Dense
# 
# A Sequential model along with Dense, Conv2D, Flatten, and MaxPool2D objects are available in your workspace.

# ### code

# In[20]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
model = Sequential()

img_rows, img_cols = (28,28)


# In[21]:


# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu', input_shape=(img_rows, img_cols, 1)))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()


# ## Train a deep CNN with pooling to classify images
# Training a CNN with pooling layers is very similar to training of the deep networks that y have seen before. Once the network is constructed (as you did in the previous exercise), the model needs to be appropriately compiled, and then training data needs to be provided, together with the other arguments that control the fitting procedure.
# 
# The following model from the previous exercise is available in your workspace:
# 
# Convolution => Max pooling => Convolution => Flatten => Dense

# ### init

# In[22]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(train_data, train_labels, test_data , test_labels, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'test_data[10_28_28_1].csv': 'https://file.io/h1Lp1O',
  'test_labels[10_3].csv': 'https://file.io/zs8BJD',
  'train_data[50_28_28_1].csv': 'https://file.io/2DI2S4',
  'train_labels[50_3].csv': 'https://file.io/7ApdTs'}}
"""
prefixToc = '3.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
test_data = getImage(prefix+'test_data[10_28_28_1].csv', dtype='float64')
test_labels = getImage(prefix+'test_labels[10_3].csv', dtype='float64')
train_data = getImage(prefix+'train_data[50_28_28_1].csv', dtype='float64')
train_labels = getImage(prefix+'train_labels[50_3].csv', dtype='float64')


# In[23]:


show_image(train_data[49,:,:,0])


# ### code

# In[25]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit to training data
model.fit(train_data, train_labels, epochs=3, batch_size=10, validation_split=0.2)

# Evaluate on test data 
model.evaluate(test_data, test_labels, batch_size=10)


# In[ ]:




