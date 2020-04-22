#!/usr/bin/env python
# coding: utf-8

# # Inheritance

# ## Animal Inheritance
# In this exercise we will code a simple example of an abstract class, and two other classes that inherit from it.
# 
# To focus on the concept of inheritance, we will introduce another set of classes: Animal, Mammal, and Reptile.
# 
# More specifically, Animal will be our abstract class, and both Mammal and Reptile will inherit from it.

# ### code

# In[ ]:


# Create a class Animal
class Animal:
	def __init__(self, name):
		self.name = name

# Create a class Mammal, which inherits from Animal
class Mammal(____):
	def __init__(self, ____, ____):
		self.____ = ____

# Create a class Reptile, which also inherits from Animal
class ____(____):
	def __init__(self, ____, ____):
		self.____ = ____

# Instantiate a mammal with name 'Daisy' and animal_type 'dog': daisy
daisy = ____(____, ____)

# Instantiate a reptile with name 'Stella' and animal_type 'alligator': stella
stella = ____(____, ____)

# Print both objects
print(daisy)
print(stella)

