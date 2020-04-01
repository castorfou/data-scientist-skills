#!/usr/bin/env python
# coding: utf-8

# # Intro
# 

# ## A taste of things to come
# In this exercise, you'll explore both the Non-Pythonic and Pythonic ways of looping over a list.
# 
# names = ['Jerry', 'Kramer', 'Elaine', 'George', 'Newman']
# Suppose you wanted to collect the names in the above list that have six letters or more. In other programming languages, the typical approach is to create an index variable (i), use i to iterate over the list, and use an if statement to collect the names with six letters or more:
# 
# i = 0
# new_list= []
# while i < len(names):
#     if len(names[i]) >= 6:
#         new_list.append(names[i])
#     i += 1
# Let's explore some more Pythonic ways of doing this.

# ### init

# In[1]:


names = ['Jerry', 'Kramer', 'Elaine', 'George', 'Newman']


# ### code

# Print the list, new_list, that was created using a Non-Pythonic approach.

# In[2]:


# Print the list created using the Non-Pythonic approach
i = 0
new_list= []
while i < len(names):
    if len(names[i]) >= 6:
        new_list.append(names[i])
    i += 1
print(new_list)


# A more Pythonic approach would loop over the contents of names, rather than using an index variable. Print better_list.

# In[3]:


# Print the list created by looping over the contents of names
better_list = []
for name in names:
    if len(name) >= 6:
        better_list.append(name)
print(better_list)


# The best Pythonic way of doing this is by using list comprehension. Print best_list.

# In[4]:


# Print the list created by using list comprehension
best_list = [name for name in names if len(name) >= 6]
print(best_list)


# ## Zen of Python
# In the video, we covered the Zen of Python written by Tim Peters, which lists 19 idioms that serve as guiding principles for any Pythonista. Python has hundreds of Python Enhancement Proposals, commonly referred to as PEPs. The Zen of Python is one of these PEPs and is documented as [PEP20](https://www.python.org/dev/peps/pep-0020/).
# 
# One little Easter Egg in Python is the ability to print the Zen of Python using the command import this. Let's take a look at one of the idioms listed in these guiding principles.
# 
# Type and run the command import this within your IPython console and answer the following question:
# 
# What is the 7th idiom of the Zen of Python?

# In[5]:


import this


# # Building with built-ins
# 

# ## Built-in practice: range()
# In this exercise, you will practice using Python's built-in function range(). Remember that you can use range() in a few different ways:
# 
# a) Create a sequence of numbers from 0 to a stop value (which is exclusive). This is useful when you want to create a simple sequence of numbers starting at zero:
# 
# ```
# range(stop)
# # Example
# list(range(11))
# ```
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 
# b) Create a sequence of numbers from a start value to a stop value (which is exclusive) with a step size. This is useful when you want to create a sequence of numbers that increments by some value other than one. For example, a list of even numbers:
# 
# ```
# range(start, stop, step)
# # Example
# list(range(2,11,2))
# ```
# [2, 4, 6, 8, 10]

# ### code

# In[7]:


# Create a range object that goes from 0 to 5
nums = range(6)
print(type(nums))

# Convert nums to a list
nums_list = list(nums)
print(nums_list)

# Create a new list of odd numbers from 1 to 11 by unpacking a range object
nums_list2 = [*range(1,12,2)]
print(nums_list2)


# ## Built-in practice: enumerate()
# In this exercise, you'll practice using Python's built-in function enumerate(). This function is useful for obtaining an indexed list. For example, suppose you had a list of people that arrived at a party you are hosting. The list is ordered by arrival (Jerry was the first to arrive, followed by Kramer, etc.):
# ```
# names = ['Jerry', 'Kramer', 'Elaine', 'George', 'Newman']
# ```
# If you wanted to attach an index representing a person's arrival order, you could use the following for loop:
# ```
# indexed_names = []
# for i in range(len(names)):
#     index_name = (i, names[i])
#     indexed_names.append(index_name)
# 
# [(0,'Jerry'),(1,'Kramer'),(2,'Elaine'),(3,'George'),(4,'Newman')]
# ```
# But, that's not the most efficient solution. Let's explore how to use enumerate() to make this more efficient.

# ### code

# In[8]:


names = ['Jerry', 'Kramer', 'Elaine', 'George', 'Newman']


# In[10]:


# Rewrite the for loop to use enumerate
indexed_names = []
for i,name in enumerate(names):
    index_name = (i,name)
    indexed_names.append(index_name) 
print(indexed_names)

# Rewrite the above for loop using list comprehension
indexed_names_comp = [(i,name) for i,name in enumerate(names)]
print(indexed_names_comp)

# Unpack an enumerate object with a starting index of one
indexed_names_unpack = [*enumerate(names, 1)]
print(indexed_names_unpack)


# ## Built-in practice: map()
# In this exercise, you'll practice using Python's built-in map() function to apply a function to every element of an object. Let's look at a list of party guests:
# ```
# names = ['Jerry', 'Kramer', 'Elaine', 'George', 'Newman']
# ```
# Suppose you wanted to create a new list (called names_uppercase) that converted all the letters in each name to uppercase. you could accomplish this with the below for loop:
# ```
# names_uppercase = []
# 
# for name in names:
#   names_uppercase.append(name.upper())
# 
# ['JERRY', 'KRAMER', 'ELAINE', 'GEORGE', 'NEWMAN']
# ```
# Let's explore using the map() function to do this more efficiently in one line of code.

# ### code

# In[11]:


# Use map to apply str.upper to each element in names
names_map  = map(str.upper, names)

# Print the type of the names_map
print(type(names_map))

# Unpack names_map into a list
names_uppercase = [*names_map]

# Print the list created above
print(names_uppercase)


# # The power of NumPy arrays
# 

# ## Practice with NumPy arrays
# Let's practice slicing `numpy` arrays and using NumPy's broadcasting concept. Remember, broadcasting refers to a `numpy` array's ability to vectorize operations, so they are performed on all elements of an object at once.
# 
# A two-dimensional `numpy` array has been loaded into your session (called `nums`) and printed into the console for your convenience. `numpy` has been imported into your session as `np`.

# ### init

# In[16]:


import numpy as np
nums = np.array([[ 1,  2,  3,  4,  5],
 [ 6,  7,  8,  9, 10]])
nums


# ### code

# In[15]:


# Print second row of nums
print(nums[1,:])

# Print all elements of nums that are greater than six
print(nums[nums > 6])

# Double every element of nums
nums_dbl = nums * 2
print(nums_dbl)

# Replace the third column of nums
nums[:,2] = nums[:,2] + 1
print(nums)


# ## Bringing it all together: Festivus!
# In this exercise, you will be throwing a party—a Festivus if you will!
# 
# You have a list of guests (the `names` list). Each guest, for whatever reason, has decided to show up to the party in 10-minute increments. For example, Jerry shows up to Festivus 10 minutes into the party's start time, Kramer shows up 20 minutes into the party, and so on and so forth.
# 
# We want to write a few simple lines of code, using the built-ins we have covered, to welcome each of your guests and let them know how many minutes late they are to your party. Note that `numpy` has been imported into your session as `np` and the `names` list has been loaded as well.
# 
# Let's welcome your guests!

# ### init

# In[17]:


import numpy as np
names = ['Jerry', 'Kramer', 'Elaine', 'George', 'Newman']


# ### code

# In[19]:


# Create a list of arrival times
arrival_times = [*range(10, 51, 10)]

print(arrival_times)


# In[20]:


# Convert arrival_times to an array and update the times
arrival_times_np = np.array(arrival_times)
new_times = arrival_times_np - 3

print(new_times)


# In[21]:


# Use list comprehension and enumerate to pair guests to new times
guest_arrivals = [(names[i],time) for i,time in enumerate(new_times)]

print(guest_arrivals)


# In[22]:


""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(welcome_guest)
"""
def welcome_guest(guest_and_time):
    """
    Returns a welcome string for the guest_and_time tuple.
    
    Args:
        guest_and_time (tuple): The guest and time tuple to create
            a welcome string for.
            
    Returns:
        welcome_string (str): A string welcoming the guest to Festivus.
        'Welcome to Festivus {guest}... You're {time} min late.'
    
    """
    guest = guest_and_time[0]
    arrival_time = guest_and_time[1]
    welcome_string = "Welcome to Festivus {}... You're {} min late.".format(guest,arrival_time)
    return welcome_string


# In[23]:


# Map the welcome_guest function to each (guest,time) pair
welcome_map = map(welcome_guest, guest_arrivals)

guest_welcomes = [*welcome_map]
print(*guest_welcomes, sep='\n')


# In[ ]:




