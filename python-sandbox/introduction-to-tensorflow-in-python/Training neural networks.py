# -*- coding: utf-8 -*-
"""
Training neural networks
Created on Tue Jul 23 15:33:31 2019

@author: N561507
"""


#%% Initialization in TensorFlow
from tensorflow import float32, Variable, random
import tensorflow as tf

# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))

# Initialize the layer 1 bias
b1 = Variable(tf.ones([7]))

# Define the layer 2 weights
w2 = Variable(random.normal([7, 1]))

# Define the layer 2 bias
b2 = Variable(0.0, float32)

#%% initial data
import numpy as np
default=np.loadtxt("default.txt",'f')
borrower_features=np.loadtxt("borrower_features2.txt",'f')
print(borrower_features.shape)


#%% Defining the model and loss function

# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
	# Apply relu activation functions to layer 1
	layer1 = tf.keras.activations.relu(tf.matmul(features, w1) + b1)
    # Apply dropout
	dropout = tf.keras.layers.Dropout(0.25)(layer1)
	return tf.keras.activations.sigmoid(tf.matmul(dropout, w2) + b2)

# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
	predictions = model(w1, b1, w2, b2)
	# Pass targets and predictions to the cross entropy loss
	return tf.keras.losses.binary_crossentropy(targets, predictions)


#%% initial data
import numpy as np
test_features=np.loadtxt("test_features.txt", dtype='f')
test_targets=np.loadtxt("test_targets.txt", dtype='f')
print(test_features.shape, test_targets.shape)

#%% Training neural networks with TensorFlow
import tensorflow as tf

opt=tf.keras.optimizers.Adam()
# Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), 
                 var_list=[w1, b1, w2, b2])

# Make predictions with model
model_predictions = model(w1, b1, w2, b2, test_features)

# Construct the confusion matrix
tf.confusion_matrix(test_targets, model_predictions)
