#!/usr/bin/env python
# coding: utf-8

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




