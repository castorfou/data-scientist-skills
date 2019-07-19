# -*- coding: utf-8 -*-
"""
Chapiter1

Created on Fri Jul 19 14:11:15 2019

@author: N561507
"""

#%% import
import tensorflow as tf
from tensorflow import constant

tf.enable_eager_execution()


#%% defining tensors in Tensorflow
# 0D Tensor
d0 = tf.ones((1,))

# 1D Tensor
d1 = tf.ones((2,))

# 2D Tensor
d2 = tf.ones((2, 2))

# 3D Tensor
d3 = tf.ones((2, 2, 2))

print(type(d3))

#Print the 3D tensor
#print(d3.numpy())
#Obsolete?

#%% Defining constants in TensorFlow
# Define a 2x3 constant.
a = constant(3, shape=[2, 3])
# Define a 2x2 constant.
b = constant([1, 2, 3, 4], shape=[2, 2])

#%% Defining and initializing variables

# Define a variable
a0 = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.float32)
a1 = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.int16)
# Define a constant
b = tf.constant(2, tf.float32)
# Compute their product
c0 = tf.multiply(a0, b)
c1 = a0*b
print(a0,a1,b,c0,c1)

#%% Exercise 1 - Defining constants with convenience functions
# Define a 3x4 tensor with all values equal to 9
from tensorflow import fill
from tensorflow import ones_like
from tensorflow import constant


A34 = fill([3, 4], 9)

# Define a tensor of ones with the same shape as A34
B34 = ones_like(A34)

# Define the one-dimensional vector, C1
C1 = constant([1, 2, 3, 4])

# Print C1 as a numpy array
#print(C1.numpy())
#'Tensor' object has no attribute 'numpy'
#
#If you are using tensorflow 2.0, then eager execution is enabled by default, so you can just call tensor.numpy() to get a NumPy array as shown in this answer. – cs95 Jun 12 at 19:10
#https://www.tensorflow.org/guide/eager


#%% Exercise 2 - Defining variables
from tensorflow import Variable
tf.executing_eagerly()

# Define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])

# Print the variable A1
print(A1)

# Convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()

# Print B1
print(B1)

#%% Exercise 3 - Checking properties of tensors

#%% Applying the addition operator

#Import constant and add from tensorflow
from tensorflow import constant, add
# Define 0-dimensional tensors
A0 = constant([1])
B0 = constant([2])
# Define 1-dimensional tensors
A1 = constant([1, 2])
B1 = constant([3, 4])
# Define 2-dimensional tensors
A2 = constant([[1, 2], [3, 4]])
B2 = constant([[5, 6], [7, 8]])

# Perform tensor addition with add()
C0 = add(A0, B0)
C1 = add(A1, B1)
C2 = add(A2, B2)

#%% Applying the multiplication operators
# Import operators from tensorflow
from tensorflow import ones, matmul, multiply
# Define tensors
A0 = ones(1)
A31 = ones([3, 1])
A34 = ones([3, 4])
A43 = ones([4, 3])

A0A0 = multiply(A0, A0)
A31A31 = multiply(A31,A31)

A43A34 = matmul(A43,A34)

#%% Summing over tensor dimensions
# Import operations from tensorflow
from tensorflow import ones, reduce_sum
# Define a 2x3x4 tensor of ones
A = ones([2, 3, 4])
# Sum over all dimensions
B = reduce_sum(A)
# Sum over dimensions 0, 1, and 2
B0 = reduce_sum(A, 0)
B1 = reduce_sum(A, 1)
B2 = reduce_sum(A, 2)

#%% Exercise 4 - Performing element-wise multiplication
from tensorflow import constant, multiply

# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = ones_like(A1)
B23 = ones_like(A23)

# Perform element-wise multiplication
C1 = multiply(A1, B1)
C23 = multiply(A23, B23)

# Print the tensors C1 and C23
print(C1.numpy())
print(C23.numpy())

#%% Exercise 5 - Making predictions with matrix multiplication
from tensorflow import constant, matmul
# Define X, b, and y as constants
X = constant([[1, 2], [2, 1], [5, 8], [6, 10]])
b = constant([[1], [2]])
y = constant([[6], [4], [20], [23]])

# Compute ypred using X and b
ypred = matmul(X,b)

# Compute and print the error
error = y - ypred
print(error.numpy())

#%% Exercise 6 - Summing over tensor dimensions
from tensorflow import constant

wealth = constant([[11, 7, 4, 3, 25], [50, 2, 60, 0, 10]])
print(wealth)
print(reduce_sum(wealth))
print(reduce_sum(wealth,0))
print(reduce_sum(wealth,1))

#%% Gradients in TensorFlow
# Import tensorflow under the alias tf
import tensorflow as tf
# Define x
x = tf.Variable(-1.0)
# Define y within instance of GradientTape
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.multiply(x, x)
# Evaluate the gradient of y at x = -1
g = tape.gradient(y, x)
#print(g.numpy())
print(g)

#%% How to reshape a grayscale image
# Import tensorflow as alias tf
import tensorflow as tf
# Generate grayscale image
gray = tf.random.uniform([2, 2], maxval=255, dtype='int32')
# Reshape grayscale image
gray = tf.reshape(gray, [2*2, 1])
print(gray)

#%% How to reshape a color image
# Import tensorflow as alias tf
import tensorflow as tf
# Generate color image
color = tf.random.uniform([2, 2, 3], maxval=255, dtype='int32')
# Reshape color image
color = tf.reshape(color, [2*2, 3])

#%% Exercise 7 - Reshaping tensors
from tensorflow import ones, reshape
# Define input image
image = ones([16, 16])

# Reshape image into a vector
image_vector = reshape(image, (256, 1))

# Reshape image into a higher dimensional tensor
image_tensor = reshape(image, (4, 4, 4, 4))

# Add three color channels
image = ones([16, 16, 3])

# Reshape image into a vector
image_vector = reshape(image, (768, 1))

# Reshape image into a higher dimensional tensor
image_tensor = reshape(image, (4, 4, 4, 4, 3))

#%% Exercise 8 - Optimizing with gradients
from tensorflow import GradientTape, multiply, Variable

def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x, x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))

#%% Exercise 9 - Working with image data
# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (3, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print(prediction.numpy())
