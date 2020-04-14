#!/usr/bin/env python
# coding: utf-8

# # Intro to Object Oriented Programming in Python
# 

# ## Creating functions
# In this exercise, we will review functions, as they are key building blocks of object-oriented programs.
# 
# For this, we will create a simple function average_numbers() which averages a list of numbers. Remember that lists are a basic data type in Python that we can build using the [] bracket notation.
# 
# Here is an example of a function that returns the square of an integer:
# ```
# def square_function(x):
#     x_squared =  x**2
#     return x_squared
#     ```

# ### code

# In[1]:


# Create function that returns the average of an integer list
def average_numbers(num_list): 
    avg = sum(num_list)/float(len(num_list)) # divide by length of list
    return avg

# Take the average of a list: my_avg
my_avg = average_numbers([1, 2, 3, 4, 5, 6])

# Print out my_avg
print(my_avg)


# ## Creating a complex data type
# In this exercise, we'll take a closer look at the flexibility of the list data type, by creating a list of lists.
# 
# In Python, lists usually look like our list example below, and can be made up of either simple strings, integers, or a combination of both.
# ```
# list = [1,2]
# ```
# In creating a list of lists, we're building up to the concept of a NumPy array.

# ### code

# In[2]:


# Create a list that contains two lists: matrix
matrix = [[1,2,3,4] , [5,6,7,8]]

# Print the matrix list
print(matrix)


# # Introduction to NumPy Internals
# 

# ## Create a function that returns a NumPy array
# In this exercise, we'll continue working with the numpy package and our previous structures.
# 
# We'll create a NumPy array of the float (numerical) data type so that we can work with a multi-dimensional data objects, much like columns and rows in a spreadsheet.

# In[3]:


# Import numpy as np
import numpy as np

# List input: my_matrix
my_matrix = [[1,2,3,4], [5,6,7,8]] 

# Function that converts lists to arrays: return_array
def return_array(matrix):
    array = np.array(matrix, dtype = float)
    return array
    
# Call return_array on my_matrix, and print the output
print(return_array(my_matrix))


# # Introduction to Objects
# 

# ## Creating a class
# We're going to be working on building a class, which is a way to organize functions and variables in Python. To start with, let's look at the simplest possible way to create a class.

# ### code

# In[4]:


# Create a class: DataShell
class DataShell: 
    pass


# In[ ]:




