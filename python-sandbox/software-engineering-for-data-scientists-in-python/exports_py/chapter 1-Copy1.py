#!/usr/bin/env python
# coding: utf-8

# # Python, data science, & software engineering
# 

# ## Python modularity in the wild
# In the slides, we covered 3 ways that you can write modular code with Python: packages, classes, and methods. For reference, you can see the example code we reviewed below.
# 
# ```
# #Import the pandas PACKAGE
# import pandas as pd
# 
# #Create some example data
# data = {'x': [1, 2, 3, 4], 
#         'y': [20.1, 62.5, 34.8, 42.7]}
# 
# #Create a dataframe CLASS object
# df = pd.DataFrame(data)
# 
# #Use the plot METHOD
# df.plot('x', 'y')
# ```
# In this exercise, you'll utilize a class & a method from the popular package numpy.

# ### code

# In[1]:


# import the numpy package
import numpy as np

# create an array class object
arr = np.array([8, 6, 7, 5, 3, 0, 9])

# use the sort method
arr.sort()

# print the sorted array
print(arr)


# # Introduction to packages & documentation
# 

# ## Leveraging documentation
# When writing code for Data Science, it's inevitable that you'll need to install and use someone else's code. You'll quickly learn that using someone else's code is much more pleasant when they use good software engineering practices. In particular, good documentation makes the right way to call a function obvious. In this exercise you'll use python's help() method to view a function's documentation so you can determine how to correctly call a new method.
# 
# The list words has been loaded in your session.

# ### init

# In[3]:


###################
##### liste de mots (list)
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(words)
"""

tobedownloaded="""
{list: {'words.txt': 'https://file.io/JEWojpBH'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
from downloadfromFileIO import loadListFromTxt
words = loadListFromTxt(prefix+'words.txt')


# ### code

# In[5]:


# load the Counter function into our environment
from collections import Counter

# View the documentation for Counter.most_common
help(Counter.most_common)


# In[6]:


# use Counter to find the top 5 most common words
top_5_words = Counter(words).most_common(5)

# display the top 5 most common words
print(top_5_words)


# # Conventions and PEP 8
# 

# ## Using pycodestyle
# We saw earlier that pycodestyle can be run from the command line to check a file for PEP 8 compliance. Sometimes it's useful to run this kind of check from a Python script.
# 
# In this exercise, you'll use pycodestyle's StyleGuide class to check multiple files for PEP 8 compliance. Both files accomplish the same task, but they differ greatly in formatting and readability. You can view the contents of the files by following their links below.

# ### init

# In[16]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('nay_pep8.py')
uploadToFileIO_pushto_fileio('yay_pep8.py')
"""
"""
{"success":true,"key":"XyZxB1J3","link":"https://file.io/XyZxB1J3","expiry":"14 days"}
{"success":true,"key":"Lrae3qMc","link":"https://file.io/Lrae3qMc","expiry":"14 days"}
"""

tobedownloaded="""
{numpy.ndarray: {'nay_pep8.py': 'https://file.io/XyZxB1J3', 'yay_pep8.py': 'https://file.io/Lrae3qMc'}}
"""
prefixToc = '3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)


# ### code

# In[17]:


# Import needed package
import pycodestyle

# Create a StyleGuide instance
style_checker = pycodestyle.StyleGuide()

# Run PEP 8 check on multiple files
result = style_checker.check_files([prefix+'nay_pep8.py', prefix+'yay_pep8.py'])

# Print result of PEP 8 style check
print(result.messages)


# ## Conforming to PEP 8
# As we've covered, there are tools available to check if your code conforms to the PEP 8 guidelines. One possible way to stay compliant is to use an IDE that warns you when you accidentally stray from the style guide. Another way to check code is to use the pycodestyle package.
# 
# The results below show the output of running pycodestyle check against the code shown in your editor. The leading number in each line shows how many occurrences there were of that particular violation.
# ```
# my_script.py:2:2:  E225 missing whitespace around operator
# my_script.py:2:7:  E231 missing whitespace after ','
# my_script.py:2:9:  E231 missing whitespace after ','
# my_script.py:5:7:  E201 whitespace after '('
# my_script.py:5:11: E202 whitespace before ')'
# ```

# ### code

# In[18]:


# Assign data to x
x=[8,3,4]

# Print the data
print(   x )


# In[19]:


# Assign data to x
x = [8, 3, 4]

# Print the data
print(x)


# ## PEP 8 in documentation
# So far we've focused on how PEP 8 affects functional pieces of code. There are also rules to help make comments and documentation more readable. In this exercise, you'll be fixing various types of comments to be PEP 8 compliant.
# 
# The result of a pycodestyle style check on the code can be seen below.
# ```
# my_script.py:2:15: E261 at least two spaces before inline comment
# my_script.py:5:16: E262 inline comment should start with '# '
# my_script.py:11:1: E265 block comment should start with '# '
# my_script.py:13:2: E114 indentation is not a multiple of four (comment)
# my_script.py:13:2: E116 unexpected indentation (comment)
# ```

# ### code

# In[20]:


def print_phrase(phrase, polite=True, shout=False):
    if polite:# It's generally polite to say please
        phrase = 'Please ' + phrase

    if shout:  #All caps looks like a written shout
        phrase = phrase.upper() + '!!'

    print(phrase)


#Politely ask for help
print_phrase('help me', polite=True)
 # Shout about a discovery
print_phrase('eureka', shout=True)


# In[21]:


def print_phrase(phrase, polite=True, shout=False):
    if polite:  # It's generally polite to say please
        phrase = 'Please ' + phrase

    if shout:  # All caps looks like a written shout
        phrase = phrase.upper() + '!!'

    print(phrase)


# Politely ask for help
print_phrase('help me', polite=True)
# Shout about a discovery
print_phrase('eureka', shout=True)


# # Writing your first package
# 

# ## Naming packages
# We covered the PEP 8 guidelines for naming packages. In this exercise, you'll use that knowledge to identify a package following the requirements.
# 
# For additional reference, you can view the PEP 8 section on package naming here

# ### code

# The possible package names to import are the following: 
# 
# text_analyzer, textAnalyzer, TextAnalyzer, & __text_analyzer__.
# 
# 
# import the package from the list above that follows the PEP 8 naming conventions.

# In[22]:


# Import the package with a name that follows PEP 8
import text_analyzer


# ## Recognizing packages
# The structure of your directory tree is printed below. You'll be working in the file my_script.py that you can see in the tree.
# ```
# recognizing_packages
# ├── MY_PACKAGE
# │   └── _init_.py
# ├── package
# │   └── __init__.py
# ├── package_py
# │   └── __init__
# │       └── __init__.py
# ├── py_package
# │   └── __init__.py
# ├── pyackage
# │   └── init.py
# └── my_script.py
# ```

# In[ ]:


# Import local packages
import package
import py_package

# View the help for each package
help(package)
help(py_package)


# ## Adding functionality to your package
# Thanks to your work before, you already have a skeleton for your python package. In this exercise, you will work to define the functions needed for a text analysis of word usage.
# 
# In the file counter_utils.py, you will write 2 functions to be a part of your package: plot_counter and sum_counters. The structure of your package can be seen in the tree below. For the coding portions of this exercise, you will be working in the file counter_utils.py.
# ```
# text_analyzer
# ├── __init__.py
# └── counter_utils.py
# ```

# ### code

# In[23]:


# Import needed functionality
from collections import Counter

def plot_counter(counter, n_most_common=5):
  # Subset the n_most_common items from the input counter
  top_items = counter.most_common(n_most_common)
  # Plot `top_items`
  plot_counter_most_common(top_items)


# In[24]:


# Import needed functionality
from collections import Counter

def sum_counters(counters):
  # Sum the inputted counters
  return sum(counters, Counter())


# ![image.png](attachment:image.png)

# ## Using your package's new functionality
# You've now created some great functionality for text analysis to your package. In this exercise, you'll leverage your package to analyze some tweets written by DataCamp & DataCamp users.
# 
# The object word_counts is loaded into your environment. It contains a list of Counter objects that contain word counts from a sample of DataCamp tweets.
# 
# The structure you've created can be seen in the tree below. You'll be working in my_script.py.
# ```
# working_dir
# ├── text_analyzer
# │    ├── __init__.py
# │    ├── counter_utils.py
# └── my_script.py
# ```

# ### init

# In[25]:


###################
##### liste de mots (list)
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(word_counts)
"""

tobedownloaded="""
{list: {'word_counts.txt': 'https://file.io/O5WKNyox'}}
"""
prefixToc='4.4'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

import pandas as pd
from downloadfromFileIO import loadListFromTxt
word_counts = loadListFromTxt(prefix+'word_counts.txt')


# ### code

# In[26]:


# Import local package
import text_analyzer

# Sum word_counts using sum_counters from text_analyzer
word_count_totals = text_analyzer.sum_counters(word_counts)

# Plot word_count_totals using plot_counter from text_analyzer
text_analyzer.plot_counter(word_count_totals)


# # Making your package portable
# 

# ## Writing requirements.txt
# We covered how having a requirements.txt file can help your package be more portable by allowing your users to easily recreate its intended environment. In this exercise, you will be writing the contents of a requirements file to a python variable.
# 
# Note, in practice, the code you write in this exercise would be written to it's own txt file instead of a variable in your python session.

# ### code

# ![image.png](attachment:image.png)

# In[27]:


requirements = """
matplotlib>=3.0.0
numpy==1.15.4
pandas<=0.22.0
pycodestyle
"""


# ## Creating setup.py
# In order to make your package installable by pip you need to create a setup.py file. In this exercise you will create this file for the text_analyzer package you've been building.

# ### code

# In[28]:


# Import needed function from setuptools
from setuptools import setup

# Create proper setup to be used by pip
setup(name='text_analyzer',
      version='0.0.1',
      description='Perform and visualize a text anaylsis.',
      author='Guillaume Ramelet',
      packages=['text_analyzer'])


# ## Listing requirements in setup.py
# We created a setup.py file earlier, but we forgot to list our dependency on matplotlib in the install_requires argument. In this exercise you will practice listing your version specific dependencies by correcting the setup.py you previously wrote for your text_analyzer package.

# ### code

# In[29]:


# Import needed function from setuptools
from setuptools import setup

# Create proper setup to be used by pip
setup(name='text_analyzer',
      version='0.0.1',
      description='Perform and visualize a text anaylsis.',
      author='Guillaume Ramelet',
      packages=['text_analyzer'],
      install_requires=['matplotlib>=3.0.0'])


# In[ ]:




