#!/usr/bin/env python
# coding: utf-8

# # Tensors, layers, and autoencoders
# 

# ## It's a flow of tensors
# If you have already built a model, you can use the model.layers and the keras.backend to build functions that, provided with a valid input tensor, return the corresponding output tensor.
# 
# This is a useful tool when trying to understand what is going on inside the layers of a neural network.
# 
# For instance, if you get the input and output from the first layer of a network, you can build an inp_to_out function that returns the result of carrying out forward propagation through only the first layer for a given input tensor.
# 
# So that's what you're going to do right now!
# 
# X_test from the Banknote Authentication dataset and its model are preloaded. Type model.summary() in the console to check it.

# ### init

# In[1]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_test)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test.csv': 'https://file.io/GdBFuo'}}
"""
prefix='data_from_datacamp/Chap4-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X_test = loadNDArrayFromCsv(prefix+'X_test.csv')


# In[2]:


# Import the Sequential model and Dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add an input layer and a hidden layer with 10 neurons
model.add(Dense(2, input_shape=(4,), activation="relu"))

# Add a 1-neuron output layer
model.add(Dense(1))

# Summarise your model
model.summary()


# ### code

# In[4]:


# Import keras backend
import keras.backend as K

# Input tensor from the 1st layer of the model
inp = model.layers[0].input

# Output tensor from the 1st layer of the model
out = model.layers[0].output

# Define a function from inputs to outputs
inp_to_out = K.function([inp], [out])

# Print the results of passing X_test through the 1st layer
print(inp_to_out([X_test]))


# ## Neural separation
# Neurons learn by updating their weights to output values that help them distinguish between the input classes. So put on your gloves because you're going to perform brain surgery!
# 
# You will make use of the inp_to_out() function you just built to visualize the output of two neurons in the first layer of the Banknote Authentication model as epochs go by. Plotting the outputs of both of these neurons against each other will show you the difference in output depending on whether each bill was real or fake.
# 
# The model you built in chapter 2 is ready for you to use, just like X_test and y_test. Copy print(inspect.getsource(plot)) in the console if you want to check plot().
# 
# You're performing heavy duty, once it's done, take a look at the graphs to watch the separation live!

# ### init

# In[5]:


#print(inspect.getsource(plot))
import matplotlib.pyplot as plt
def plot():
  fig, ax = plt.subplots()
  plt.scatter(layer_output[:, 0], layer_output[:, 1],c=y_test,edgecolors='none')
  plt.title('Epoch: {}, Test Acc: {:3.1f} %'.format(i+1, test_accuracy * 100.0))
  plt.show()
    


# In[11]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_train, y_train, y_test, X_test)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test.csv': 'https://file.io/N8tcpv',
  'X_train.csv': 'https://file.io/xastru',
  'y_test.csv': 'https://file.io/zJehFm',
  'y_train.csv': 'https://file.io/TocYUT'}}
"""
prefix='data_from_datacamp/Chap4-Exercise1.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X_train = loadNDArrayFromCsv(prefix+'X_train.csv')
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
X_test = loadNDArrayFromCsv(prefix+'X_test.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# ### code

# In[12]:


for i in range(0, 21):
    # Train model for 1 epoch
    h = model.fit(X_train, y_train, batch_size=16, epochs=1,verbose=0)
    if i%4==0: 
      # Get the output of the first layer
      layer_output = inp_to_out([X_test])[0]
      
      # Evaluate model accuracy for this epoch
      test_accuracy = model.evaluate(X_test, y_test)[1] 
      
      # Plot 1st vs 2nd neuron output
      plot()


# ## Building an autoencoder
# Autoencoders have several interesting applications like anomaly detection or image denoising. They aim at producing an output identical to its inputs. The input will be compressed into a lower dimensional space, encoded. The model then learns to decode it back to its original form.
# 
# You will encode and decode the MNIST dataset of handwritten digits, the hidden layer will encode a 32-dimensional representation of the image, which originally consists of 784 pixels.
# 
# The Sequential model and Dense layers are ready for you to use.
# 
# Let's build an autoencoder!

# ### code

# In[14]:


# Start with a sequential model
autoencoder = Sequential()

# Add a dense layer with the original image as input
autoencoder.add(Dense(32, input_shape=(784, ), activation="relu"))

# Add an output layer with as many nodes as the image
autoencoder.add(Dense(784, activation="sigmoid"))

# Compile your model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Take a look at your model structure
autoencoder.summary()


# ## De-noising like an autoencoder
# Okay, you have just built an autoencoder model. Let's see how it handles a more challenging task.
# 
# First, you will build a model that encodes images, and you will check how different digits are represented with show_encodings(). You can change the number parameter of this function to check other digits in the console.
# 
# Then, you will apply your autoencoder to noisy images from MNIST, it should be able to clean the noisy artifacts.
# 
# X_test_noise is loaded in your workspace. The digits in this data look like this:
# 
# 
# 
# Apply the power of the autoencoder!

# ### init

# In[28]:


""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(show_encodings)
"""
import matplotlib.pyplot as plt
def show_encodings(encoded_imgs,number=4):
    n = 5  # how many digits we will display
    original = X_test_noise
    original = original[np.where(y_test == number)]
    encoded_imgs = encoded_imgs[np.where(y_test==number)]
    plt.figure(figsize=(20, 4))
    #plt.title('Original '+str(number)+' vs Encoded representation')
    for i in range(min(n,len(original))):
        # display original imgs
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display encoded imgs
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(np.tile(encoded_imgs[i],(32,1)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
def compare_plot(original,decoded_imgs):
    n = 4  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.title('Noisy vs Decoded images')
    plt.show()


# In[19]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_test_noise)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test_noise.csv': 'https://file.io/LEbYEg'}}
"""
prefix='data_from_datacamp/Chap4-Exercise1.4_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X_test_noise = loadNDArrayFromCsv(prefix+'X_test_noise.csv')


# In[30]:


autoencoder.fit(X_train, y_train, validation_date=(X_test, y_test), epochs=10, verbose=0 )


# ### code

# In[25]:


# Build your encoder
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# Encode the images and show the encodings
preds = encoder.predict(X_test_noise)
show_encodings(preds)


# In[29]:


# Predict on the noisy images with your autoencoder
decoded_imgs = autoencoder.predict(X_test_noise)

# Plot noisy vs decoded images
compare_plot(X_test_noise, decoded_imgs)


# ## Intro to CNNs

# ## Building a CNN model
# Building a CNN model in Keras isn't much more difficult than building any of the models you've already built throughout the course! You just need to make use of convolutional layers.
# 
# You're going to build a shallow convolutional model that classifies the MNIST dataset of digits. The same one you de-noised with your autoencoder!. The images are 28x28 pixels and just have one channel.
# 
# Go ahead and build this small convolutional model!

# ### code

# In[3]:


# Import the Conv2D and Flatten layers and instantiate model
from keras.layers import Conv2D,Flatten
model = Sequential()

# Add a convolutional layer of 32 filters of size 3x3
model.add(Conv2D(32, input_shape=(28, 28, 1), kernel_size=3, activation='relu'))

# Add a convolutional layer of 16 filters of size 3x3
model.add(Conv2D(16, kernel_size=3, activation='relu'))

# Flatten the previous layer output
model.add(Flatten())

# Add as many outputs as classes with softmax activation
model.add(Dense(10, activation='softmax'))


# ## Looking at convolutions
# Inspecting the activations of a convolutional layer is a cool thing. You have to do it at least once in your lifetime!
# 
# To do so, you will build a new model with the Keras Model object, which takes in a list of inputs and a list of outputs. The output you will provide to this new model is the first convolutional layer outputs when given an MNIST digit as input image.
# 
# The convolutional model you built in the previous exercise has already been trained for you. You can check it with model.summary() in the console.
# 
# Let's look at a couple convolutional masks that were learned in the first convolutional layer of this model!

# ### init

# In[11]:


###################
##### Keras Sequential model
###################

#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(model)
"""

tobedownloaded="""
{keras.engine.sequential.Sequential: {'model.h5': 'https://file.io/5NUMGz'}}
"""
prefix='data_from_datacamp/Chap4-Exercise1.7_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadModelFromH5
model = loadModelFromH5(prefix+'model.h5')


# In[15]:


from keras.models import Model


# In[19]:


###################
##### numpy ndarray float N-dimensional n>2
###################

#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
X_test_100_28_28_1 = X_test.flatten()
uploadToFileIO(X_test_100_28_28_1)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test_100_28_28_1.csv': 'https://file.io/jdynnD'}}
"""
prefix='data_from_datacamp/Chap4-Exercise1.7_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from uploadfromdatacamp import loadNDArrayFromCsv
X_test_100_28_28_1 = loadNDArrayFromCsv(prefix+'X_test_100_28_28_1.csv')
X_test = np.reshape(X_test_100_28_28_1, (100,28,28,1))


# ### code

# In[31]:


import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=1, ncols=2)

# Obtain a reference to the outputs of the first layer
layer_output = model.layers[0].output

# Build a model using the model's input and the first layer output
first_layer_model = Model(inputs = model.layers[0].input, outputs = layer_output)

# Use this model to predict on X_test
activations = first_layer_model.predict(X_test)

# Plot the activations of first digit of X_test for the 15th filter
axs[0].matshow(activations[0,:,:,14], cmap = 'viridis')

# Do the same but for the 18th filter now
axs[1].matshow(activations[0,:,:,18], cmap = 'viridis')
plt.show()


# ![image.png](attachment:image.png)

# ## Preparing your input image
# When using an already trained model like ResNet50, we need to make sure that we fit the network the way it was originally trained. So if we want to use a trained model on our custom images, these images need to have the same dimensions as the one used in the original model.
# 
# The original ResNet50 model was trained with images of size 224x224 pixels and a number of preprocessing operations; like the subtraction of the mean pixel value in the training set for all training images.
# 
# You will go over these preprocessing steps as you prepare this dog's (named Ivy) image into one that can be classified by ResNet50.
# ![image.png](attachment:image.png)

# ### init

# In[2]:


img_path = '/usr/local/share/datasets/dog.jpg'
'''
uploadToFileIO_pushto_fileio(img_path)
{"success":true,"key":"KNLDiN","link":"https://file.io/KNLDiN","expiry":"14 days"}
'''

img_path='data_from_datacamp\dog.jpg'


# ### code

# In[3]:


# Import image and preprocess_input
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input   

# Load the image with the right target size for your model
img = image.load_img(img_path, target_size=(224, 224))

# Turn it into an array
img_array = image.img_to_array(img)

# Expand the dimensions of the image
img_expanded = np.expand_dims(img_array, axis = 0)

# Pre-process the img in the same way original images were
img_ready = preprocess_input(img_expanded)


# ## Using a real world model
# Okay, so Ivy's picture is ready to be used by ResNet50. It is stored in img_ready and now looks like this:
# 
# 
# ResNet50 is a model trained on the Imagenet dataset that is able to distinguish between 1000 different objects. ResNet50 is a deep model with 50 layers, you can check it in 3D here.
# 
# ResNet50 and decode_predictions have both been imported from keras.applications.resnet50 for you.
# 
# It's time to use this trained model to find out Ivy's breed!

# ### init

# In[6]:


'''telechargement de resnet50
https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5
à télécharger dans c:/users/f279814/.keras/models
et 
https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
'''


# ![image.png](attachment:image.png)

# ### code

# In[4]:


from keras.applications.resnet50 import ResNet50, decode_predictions


# In[12]:


# Instantiate a ResNet50 model with imagenet weights
model = ResNet50(weights='imagenet')

# Predict with ResNet50 on your already processed img
preds = model.predict(img_ready)

# Decode predictions
print('Predicted:', decode_predictions(preds, top=3)[0])


# In[18]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from uploadfromdatacamp import saveFromFileIO2
       

tobedownloaded="""
{keras.engine.sequential.Sequential: {'model.h5': 'https://file.io/5NUMGz'}}
"""
prefix='data_from_datacamp/Chap4-Exercise1.7_'
saveFromFileIO2(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# # Intro to LSTMs
# 

# ## Text prediction with LSTMs
# During the following exercises you will build an LSTM model that is able to predict the next word using a small text dataset. This dataset consist of cleaned quotes from the The Lord of the Ring movies. You can find them in the text variable.
# 
# You will turn this text into sequences of length 4 and make use of the Keras Tokenizer to prepare the features and labels for your model!
# 
# The Keras Tokenizer is already imported for you to use. It assigns a unique number to each unique word, and stores the mappings in a dictionary. This is important since the model deals with numbers but we later will want to decode the output numbers back into words.

# ### init

# In[19]:


text = 'it is not the strength of the body but the strength of the spirit it is useless to meet revenge with revenge it will heal nothing even the smallest person can change the course of history all we have to decide is what to do with the time that is given us the burned hand teaches best after that advice about fire goes to the heart'


# In[20]:


# Import Tokenizer from keras preprocessing text
from keras.preprocessing.text import Tokenizer


# ### code

# In[21]:


# Split text into an array of words 
words = text.split()

# Make lines of 4 words each, moving one word at a time
lines = []
for i in range(4, len(words)):
  lines.append(' '.join(words[i-4:i]))

# Instantiate a Tokenizer, then fit it on the lines
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)

# Turn lines into a sequence of numbers
sequences = tokenizer.texts_to_sequences(lines)
print("Lines: \n {} \n Sequences: \n {}".format(lines[:5],sequences[:5]))


# ## Build your LSTM model
# You've already prepared your sequences of text, with each of the sequences consisting of four words. It's time to build your LSTM model!
# 
# Your model will be trained on the first three words of each sequence, predicting the 4th one. You are going to use an Embedding layer that will essentially learn to turn words into vectors. These vectors will then be passed to a simple LSTM layer. Our output is a Dense layer with as many neurons as words in the vocabulary and softmax activation. This is because we want to obtain the highest probable next word out of all possible words.
# 
# The size of the vocabulary of words (the unique number of words) is stored in vocab_size.

# ### init

# In[28]:


vocab_size = 44
from keras.models import Sequential


# ### code

# In[29]:


# Import the Embedding, LSTM and Dense layer
from keras.layers import Embedding, LSTM, Dense

model = Sequential()

# Add an Embedding layer with the right parameters
model.add(Embedding(input_dim=vocab_size, output_dim=8, input_length=3))

# Add a 32 unit LSTM layer
model.add(LSTM(32))

# Add a hidden Dense layer of 32 units and an output layer of vocab_size with softmax
model.add(Dense(32, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()


# ## Decode your predictions
# Your LSTM model has already been trained for you so that you don't have to wait. It's time to define a function that decodes its predictions.
# 
# Since you are predicting on a model that uses the softmax function,argmax() is used to obtain the position of the output layer with the highest probability, that is the index representing the most probable next word.
# 
# The tokenizer you previously created and fitted, is loaded for you. You will be making use of its internal index_word dictionary to turn the model's next word prediction (which is an integer) into the actual written word it represents.
# 
# You're very close to experimenting with your model!

# ### init

# In[32]:


###################
##### Keras Sequential model
###################

#upload and download

from downloadfromFileIO import saveFromFileIO2
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(model)
"""

tobedownloaded="""
{keras.engine.sequential.Sequential: {'model.h5': 'https://file.io/J3B2WY'}}
"""
prefix='data_from_datacamp/Chap4-Exercise2.3_'
saveFromFileIO2(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import loadModelFromH5
model = loadModelFromH5(prefix+'model.h5')


# ### code

# In[33]:


def predict_text(test_text):
  if len(test_text.split())!=3:
    print('Text input should be 3 words!')
    return False
  
  # Turn the test_text into a sequence of numbers
  test_seq = tokenizer.texts_to_sequences([test_text])
  test_seq = np.array(test_seq)
  
  # Get the model's next word prediction by passing in test_seq
  pred = model.predict_proba(test_seq).argmax(axis = 1)[0]
  
  # Return the word associated to the predicted index
  return tokenizer.index_word[pred]


# In[34]:


predict_text('meet revenge with')


# In[35]:


predict_text('the course of')


# In[37]:


predict_text('strength of the')


# In[ ]:




