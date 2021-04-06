# -*- coding: utf-8 -*-
"""
Chapter 4
Created on Wed Jul 24 09:06:22 2019

@author: N561507
"""

#%% The sequential API - Building a sequential model

#The sequential API
#Input layer
#Hidden layers
#Output layer
#Ordered in sequence

# Import tensorflow
from tensorflow import keras
# Define a sequential model
model = keras.Sequential()
# Define first hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(28*28,)))
# Define second hidden layer
model.add(keras.layers.Dense(8, activation='relu'))
# Define output layer
model.add(keras.layers.Dense(4, activation='softmax'))
# Compile the model
model.compile('adam', loss='categorical_crossentropy')
# Summarize the model
print(model.summary())

#%% The functional API - Using the functional API
# Import tensorflow
import tensorflow as tf
# Define model 1 input layer shape
model1_inputs = tf.keras.Input(shape=(28*28,))
# Define model 2 input layer shape
model2_inputs = tf.keras.Input(shape=(10,))
# Define layer 1 for model 1
model1_layer1 = tf.keras.layers.Dense(12, activation='relu')(model1_inputs)
# Define layer 2 for model 1
model1_layer2 = tf.keras.layers.Dense(4, activation='softmax')(model1_layer1)
# Define layer 1 for model 2
model2_layer1 = tf.keras.layers.Dense(8, activation='relu')(model2_inputs)
# Define layer 2 for model 2
model2_layer2 = tf.keras.layers.Dense(4, activation='softmax')(model2_layer1)
# Merge model 1 and model 2
merged = tf.keras.layers.add([model1_layer2, model2_layer2])
# Define a functional model
model = tf.keras.Model(inputs=[model1_inputs, model2_inputs], outputs=merged)
# Compile the model
model.compile('adam', loss='categorical_crossentropy')

#%% Exercise - The sequential model in Keras
from tensorflow import keras

# Define a Keras sequential model
model = keras.Sequential()
# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu',))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Print the model architecture
print(model.summary())

#%% Exercise - Compiling a sequential model
from tensorflow import keras
# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(784,)))

# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model.summary())

#%% Exercise - Defining a multiple input model
from tensorflow import keras

# Import tensorflow
import tensorflow as tf
# Define model 1 input layer shape
m1_inputs = tf.keras.Input(shape=(784,))
# Define model 2 input layer shape
m2_inputs = tf.keras.Input(shape=(784,))

# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())


#%% How to train a model
# Import tensorflow
import tensorflow as tf
# Define a sequential model
model = tf.keras.Sequential()
# Define the hidden layer
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(784,)))
# Define the output layer
model.add(tf.keras.layers.Dense(4, activation='softmax'))
# Compile model
model.compile('adam', loss='categorical_crossentropy')
# Train model
model.fit(image_features, image_labels)

#%% Performing validation
# Train model with validation split
model.fit(features, labels, epochs=10, validation_split=0.20)

#%% Changing the metric
# Recomile the model with the accuracy metric
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train model with validation split
model.fit(features, labels, epochs=10, validation_split=0.20)

#%% The evaluation() operation
# Evaluate the test set
model.evaluate(test)

#%% Exercise - Training with Keras - initial data
import numpy as np
sign_language_labels=np.loadtxt("sign_language_labels.txt")
sign_language_features=np.loadtxt("sign_language_features.txt")
print(sign_language_labels.shape, sign_language_features.shape)


#%% Exercise - Training with Keras
# Import tensorflow
import tensorflow as tf
from tensorflow import keras

# Define a sequential model
model = keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)

#%% Exercise - Metrics and validation with Keras
# Import keras and tensorflow
import tensorflow as tf
from tensorflow import keras

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.1)

#%% Exercise - Overfitting detection
# Import keras and tensorflow
import tensorflow as tf
from tensorflow import keras

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024, activation='relu',input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.01), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=200, validation_split=0.5)