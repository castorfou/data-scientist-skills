#!/usr/bin/env python
# coding: utf-8

# # Intro to Classes
# 

# ## Object: Instance of a Class
# As we learned earlier, a class is like a blueprint: we can make many copies of our class.
# 
# When we do this, we say that we are instantiating our class. These instances are called objects.
# 
# Here is an example of class instantiation:
# 
# object_name = ClassName()

# ### code

# In[1]:


# Create empty class: DataShell
class DataShell:
  
    # Pass statement
    pass

# Instantiate DataShell: my_data_shell
my_data_shell = DataShell()

# Print my_data_shell
print(my_data_shell)


# # Initializing a Class and Self
# 

# ## The Init Method
# Now it's time to explore the special ```__init__``` method!
# 
# ```__init__``` is an initialization method used to construct class instances in custom ways. In this exercise we will simply introduce the utilization of the method, and in subsequent ones we will do fancier things.

# ### code

# In[2]:


# Create class: DataShell
class DataShell:
  
	# Initialize class with self argument
    def __init__(self):
      
        # Pass statement
        pass

# Instantiate DataShell: my_data_shell
my_data_shell = DataShell()

# Print my_data_shell
print(my_data_shell)


# ## Instance Variables
# Class instances are useful in that we can store values in them at the time of instantiation. We store these values in instance variables. This means that we can have many instances of the same class whose instance variables hold different values!

# ### code

# In[3]:


# Create class: DataShell
class DataShell:
  
	# Initialize class with self and integerInput arguments
    def __init__(self, integerInput):
      
		# Set data as instance variable, and assign the value of integerInput
        self.data = integerInput

# Declare variable x with value of 10
x = 10      

# Instantiate DataShell passing x as argument: my_data_shell
my_data_shell = DataShell(x)

# Print my_data_shell
print(my_data_shell.data)


# ## Multiple Instance Variables
# We are not limited to declaring only one instance variable; in fact, we can declare many!
# 
# In this exercise we will declare two instance variables: identifier and data. Their values will be specified by the values passed to the initialization method, as before.

# ### code

# In[4]:


# Create class: DataShell
class DataShell:
  
	# Initialize class with self, identifier and data arguments
    def __init__(self, identifier, data):
      
		# Set identifier and data as instance variables, assigning value of input arguments
        self.identifier = identifier
        self.data = data

# Declare variable x with value of 100, and y with list of integers from 1 to 5
x = 100
y = [1, 2, 3, 4, 5]

# Instantiate DataShell passing x and y as arguments: my_data_shell
my_data_shell = DataShell(x, y)

# Print my_data_shell.identifier
print(my_data_shell.identifier)

# Print my_data_shell.data
print(my_data_shell.data)


# # More on Self and Passing in Variables
# 

# ## Class Variables
# We saw that we can specify different instance variables.
# 
# But, what if we want any instance of a class to hold the same value for a specific variable? Enter class variables.
# 
# Class variables must not be specified at the time of instantiation and instead, are declared/specified at the class definition phase.

# ### code

# In[5]:


# Create class: DataShell
class DataShell:
  
    # Declare a class variable family, and assign value of "DataShell"
    family = "DataShell"
    
    # Initialize class with self, identifier arguments
    def __init__(self, identifier):
      
        # Set identifier as instance variable of input argument
        self.identifier = identifier

# Declare variable x with value of 100
x = 100

# Instantiate DataShell passing x as argument: my_data_shell
my_data_shell = DataShell(x)

# Print my_data_shell class variable family
print(my_data_shell.family)


# ## Overriding Class Variables
# Sometimes our object instances have class variables whose values are not correct, and hence, not useful. For this reason it makes sense to modify our object's class variables.
# 
# In this exercise, we will do just that: override class variables with values of our own!

# ### code

# In[6]:


# Create class: DataShell
class DataShell:
  
    # Declare a class variable family, and assign value of "DataShell"
    family = "DataShell"
    
    # Initialize class with self, identifier arguments
    def __init__(self, identifier):
      
        # Set identifier as instance variable of input argument
        self.identifier = identifier

# Declare variable x with value of 100
x = 100

# Instantiate DataShell passing x as argument: my_data_shell
my_data_shell = DataShell(x)

# Print my_data_shell class variable family
print(my_data_shell.family)

# Override the my_data_shell.family value with "NotDataShell"
my_data_shell.family = "NotDataShell"

# Print my_data_shell class variable family once again
print(my_data_shell.family)


# # Methods in Classes
# 

# ## Methods I
# Not only are we able to declare both instance variables and class variables in our objects, we can also cook functions right into our objects as well. These object-contained functions are called methods.

# ### code

# In[7]:


# Create class: DataShell
class DataShell:
  
	# Initialize class with self argument
    def __init__(self):
        pass
      
	# Define class method which takes self argument: print_static
    def print_static(self):
        # Print string
        print("You just executed a class method!")
        
# Instantiate DataShell taking no arguments: my_data_shell
my_data_shell = DataShell()

# Call the print_static method of your newly created object
my_data_shell.print_static()


# ## Methods II
# In the previous exercise our print_static() method was kind of boring.
# 
# We can do more interesting things with our objects' methods. For example, we can interact with our objects' data. In this exercise we will declare a method that prints the value of one of our instance variables.

# ### code

# In[8]:


# Create class: DataShell
class DataShell:
  
	# Initialize class with self and dataList as arguments
    def __init__(self, dataList):
      	# Set data as instance variable, and assign it the value of dataList
        self.data = dataList
        
	# Define class method which takes self argument: show
    def show(self):
        # Print the instance variable data
        print(self.data)

# Declare variable with list of integers from 1 to 10: integer_list   
integer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Call the show method of your newly created object
my_data_shell.show()


# ## Methods III
# In the last exercise our method simply printed out the value of instance variables.
# 
# In this one, we'll do something more interesting. We will add another method, avg(), which takes a list of integers, calculates the average value, and prints it out. To make things even more interesting, the list of integers for which avg() does this operations, is one of our object's instance variables.
# 
# This means that our object can not only store data, but also can store procedures it can execute on its own data. Awesome.
# 
# Note that the variable integer_list has already been loaded for you.

# ### code

# In[9]:


integer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# In[11]:


# Create class: DataShell
class DataShell:
  
	# Initialize class with self and dataList as arguments
    def __init__(self, dataList):
      	# Set data as instance variable, and assign it the value of dataList
        self.data = dataList
        
	# Define method that prints data: show
    def show(self):
        print(self.data)
        
    # Define method that prints average of data: avg 
    def avg(self):
        # Declare avg and assign it the average of data
        avg = sum(self.data)/float(len(self.data))
        # Print avg
        print(avg)
        
# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Call the show and avg methods of your newly created object
my_data_shell.show()
my_data_shell.avg()


# In[ ]:




