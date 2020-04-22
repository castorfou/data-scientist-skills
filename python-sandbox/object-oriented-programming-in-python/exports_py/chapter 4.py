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

# In[1]:


# Create a class Animal
class Animal:
	def __init__(self, name):
		self.name = name

# Create a class Mammal, which inherits from Animal
class Mammal(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Create a class Reptile, which also inherits from Animal
class Reptile(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Instantiate a mammal with name 'Daisy' and animal_type 'dog': daisy
daisy = Mammal('Daisy', 'dog')

# Instantiate a reptile with name 'Stella' and animal_type 'alligator': stella
stella = Reptile('Stella', 'alligator')

# Print both objects
print(daisy)
print(stella)


# ## Vertebrate Inheritance
# In the previous exercise, it seemed almost unnecessary to have an abstract class, as it did not do anything particularly interesting (other than begin to learn inheritance).
# 
# In this exercise, we will refine our abstract class and include some class variables in our abstract class so that they can be passed down to our other classes.
# 
# Additionally from inheritance, in this exercise we are seeing another powerful object-oriented programming concept: polymorphism. As you explore your code while writing the Mammal and Reptile classes, notice their differences. Because they both inherit from the Vertebrate class, and because they are different, we say that they are polymorphic. How cool!

# ### code

# In[2]:


# Create a class Vertebrate
class Vertebrate:
    spinal_cord = True
    def __init__(self, name):
        self.name = name

# Create a class Mammal, which inherits from Vertebrate
class Mammal(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = True

# Create a class Reptile, which also inherits from Vertebrate
class Reptile(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = False

# Instantiate a mammal with name 'Daisy' and animal_type 'dog': daisy
daisy = Mammal('Daisy', 'dog')

# Instantiate a reptile with name 'Stella' and animal_type 'alligator': stella
stella = Reptile('Stella', 'alligator')

# Print stella's attributes spinal_cord and temperature_regulation
print("Stella Spinal cord: " + str(stella.spinal_cord))
print("Stella temperature regulation: " + str(stella.temperature_regulation))

# Print daisy's attributes spinal_cord and temperature_regulation
print("Daisy Spinal cord: " + str(daisy.spinal_cord))
print("Daisy temperature regulation: " + str(daisy.temperature_regulation))


# # Inheritance with DataShells
# 

# ## Abstract Class DataShell I
# We will now switch back to working on our DataShell class. Specifically, we will create an abstract class, such that we can create other classes that then inherit from it!
# 
# For this reason, our abstract DataShell class will not do much, resembling some of the earlier exercises in this course.

# ### code

# In[3]:


us_life_expectancy = 'https://assets.datacamp.com/production/repositories/2097/datasets/5dd3a8250688a4f08306206fa1d40f63b66bc8a9/us_life_expectancy.csv'


# In[4]:


# Load numpy as np and pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
    def __init__(self, inputFile):
        self.file = inputFile

# Instantiate DataShell as my_data_shell
my_data_shell = DataShell(us_life_expectancy)

# Print my_data_shell
print(my_data_shell)


# ## Abstract Class DataShell II
# Now that we have our abstract class DataShell, we can now create a second class that inherits from it.
# 
# Specifically, we will define a class called CsvDataShell. This class will have the ability to import a CSV file. In the following exercises we will add a bit more functionality to make our classes more sophisticated!

# ### code

# In[5]:


# Load numpy as np and pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
    def __init__(self, inputFile):
        self.file = inputFile

# Create class CsvDataShell, which inherits from DataShell
class CsvDataShell(DataShell):
    # Initialization method with arguments self, inputFile
    def __init__(self, inputFile):
        # Instance variable data
        self.data = pd.read_csv(inputFile)

# Instantiate CsvDataShell as us_data_shell, passing us_life_expectancy as argument
us_data_shell = CsvDataShell(us_life_expectancy)

# Print us_data_shell.data
print(us_data_shell.data)


# # Composition

# ## Composition and Inheritance I
# As you may have noticed, we have already been using composition in our classes, we just have not been explicit about it. More specifically, we have been relying on functionality from the pandas package.
# 
# In this exercise, we will combine inheritance and composition as we define a class that 1) inherits from another class, and 2) uses functionality from other classes.

# ### code

# In[6]:


# Define abstract class DataShell
class DataShell:
    # Class variable family
    family = 'DataShell'
    # Initialization method with arguments, and instance variables
    def __init__(self, name, filepath): 
        self.name = name
        self.filepath = filepath

# Define class CsvDataShell      
class CsvDataShell(DataShell):
    # Initialization method with arguments self, name, filepath
    def __init__(self, name, filepath):
        # Instance variable data
        self.data = pd.read_csv(filepath)
        # Instance variable stats
        self.stats = self.data.describe()

# Instantiate CsvDataShell as us_data_shell
us_data_shell = CsvDataShell("US", us_life_expectancy)

# Print us_data_shell.stats
print(us_data_shell.stats)


# ## Composition and Inheritance II
# In this exercise, we will create another class TsvDataShell that inherits from our abstract class DataShell, which also uses composition in recycling functionality from pandas objects.
# 
# Specifically, our new class will be able to read in TSV files, and also give us a description of the data it stores.

# ### code

# In[7]:


france_life_expectancy = 'https://assets.datacamp.com/production/repositories/2097/datasets/e3620bc33a17d7ce5cf0ae87e723171284c81df3/france_life_expectancy.csv'


# In[8]:


# Define abstract class DataShell
class DataShell:
    family = 'DataShell'
    def __init__(self, name, filepath): 
        self.name = name
        self.filepath = filepath

# Define class CsvDataShell
class CsvDataShell(DataShell):
    def __init__(self, name, filepath):
        self.data = pd.read_csv(filepath)
        self.stats = self.data.describe()

# Define class TsvDataShell
class TsvDataShell(DataShell):
    # Initialization method with arguments self, name, filepath
    def __init__(self, name, filepath):
        # Instance variable data
        self.data = pd.read_table(filepath)
        # Instance variable stats
        self.stats = self.data.describe()

# Instantiate CsvDataShell as us_data_shell, print us_data_shell.stats
us_data_shell = CsvDataShell("US", us_life_expectancy)
print(us_data_shell.stats)

# Instantiate TsvDataShell as france_data_shell, print france_data_shell.stats
france_data_shell = TsvDataShell("FR", france_life_expectancy)
print(france_data_shell.stats)


# In[ ]:




