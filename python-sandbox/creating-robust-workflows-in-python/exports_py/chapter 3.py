#!/usr/bin/env python
# coding: utf-8

# # Command-line interfaces
# 

# ## Argparse nbuild()
# We will use the standard library argparse module to make a general command-line interface (CLI) function called argparse_cli() and apply this function to the nbuild() function from the previous chapter.
# 
# If you do not remember how nbuild() works, call help(nbuild).
# 
# In short, nbuild() returns a notebook object that contains one cell for each input file it receives.
# 
# We want argparse_cli() to be able to handle an indeterminate number of shell arguments.
# 
# To do this with argparse, we need to pass nargs='*' to the add_argument() method of an instance of the ArgumentParser class.
# 
# We will use our CLI to pass shell arguments to nbuild(), so that we can focus on CLI design and not on how the shell arguments are used.

# ### code

# In[5]:


import argparse 
from typing import Callable

from typing import List
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

def nbuild(filenames: List[str]) -> nbformat.notebooknode.NotebookNode:
    '''Create a Jupyter notebook from text files and Python scripts'''
    # Create a new notebook object
    nb = new_notebook()
    nb.cells.extend(
        # Create new code cells from files that end in .py
        new_code_cell(Path(name).read_text())
        if name.endswith('.py')
        # Create new markdown cells from all other files
        else new_markdown_cell(Path(name).read_text()) 
        for name in filenames
    )
    return nb


# In[6]:


def argparse_cli(func: Callable) -> None:
    # Instantiate the parser object
    parser = argparse.ArgumentParser()
    # Add an argument called in_files to the parser object
    parser.add_argument("in_files", nargs="*")
    args = parser.parse_args()
    print(func(args.in_files))

if __name__ == "__main__":
    argparse_cli(nbuild)


# ## Docopt nbuild()
# If you love docstrings, you are likely to be a fan of docopt CLIs.
# 
# The docstring in our docopt_cli.py file is only one line, but it includes all the details we need to pass a list of shell arguments to any function.
# 
# More specifically, the docstring determines that our IN_FILES variable is
# 
# optional and
# represents a list of arguments
# In docopt docstrings, optional arguments are wrapped in square brackets ([]), while lists of arguments are followed by ellipses (...).
# 
# We have already imported the docopt() function from the docopt module for use in our docopt_cli() function.

# ### code

# In[10]:


from docopt import docopt


# In[11]:


# Add the section title in the docstring below
"""Usage: docopt_cli.py [IN_FILES...]"""

def docopt_cli(func: Callable) -> None:
    # Assign the shell arguments to "args"
    args = docopt(__doc__)
    print(func(args["IN_FILES"]))

if __name__ == "__main__":
    docopt_cli(nbuild)


# # Git version control
# 

# ## Commit added files
# GitPython gives us building blocks that we can use to build Python scripts that make our use of version control faster, easier, and more efficient.
# 
# Version controlled projects usually start with initializing or cloning repositories.
# 
# After a repository is set up, the standard cycle of commands is add and commit changes.
# 
# In this exercise, we will focus on the first two steps: adding changes to the index and committing them to version control history.
# 
# The commit message is created by an f-string, which evaluates the code inside curly braces ({}).
# 
# With GitPython, we can initialize a new repository and instantiate the Repo class in one line of code.
# 
# We can then check for untracked files, add files to the index, commit changes, and list all of the newly tracked files.

# ### code

# In[12]:


import git


# In[ ]:


# Initialize a new repo in the current folder
repo = git.Repo.init()

# Obtain a list of untracked files
untracked = repo.untracked_files

# Add all untracked files to the index
repo.index.add(untracked)

# Commit newly added files to version control history
repo.index.commit(f"Added {', '.join(untracked)}")
print(repo.head.commit.message)


# ## Commit modified files
# Since the previous exercise, we have made some changes and now we want to get a list of the files that have changed.
# 
# We will include the list of changed files in our next commit message.
# 
# A major advantage of GitPython is that it allows us to programmatically access information on the status of our repositories.
# 
# Here, we will use the diff() method to obtain a list of files with changes since the latest commit.
# 
# Our code should work regardless of how many or which files have been modified.
# 
# For an example of how GitPython can be applied in the real world, take a look at the Gitone Python package.
# 
# 

# ### code

# In[ ]:


changed_files = [file.b_path
                 # Iterate over items in the diff object
                 for file in repo.index.diff(None)
                 # Include only modified files
                 .iter_change_type("M")]

repo.index.add(changed_files)
repo.index.commit(f"Modified {', '.join(changed_files)}")
for number, commit in enumerate(repo.iter_commits()):
    print(number, commit.message)


# # Virtual environments
# 

# ## List installed packages
# In this exercise, we will create a venv virtual environment and then make sure we are using a relatively recent of version pandas.
# 
# We will use the pip list command to pick out pandas from a list of all the packages available in our virtual environment.

# ### code

# In[ ]:


# Create an virtual environment
venv.create(".venv")

# Run pip list and obtain a CompletedProcess instance
cp = subprocess.run([".venv/bin/python", "-m", "pip", "list"], stdout=-1)

for line in cp.stdout.decode().split("\n"):
    if "pandas" in line:
        print(line)


# ## Show package information
# In this exercise, we will use the pip install command to install a local package called aadvark.
# 
# The requirements.txt file in the current working directory is already set up to install any local packages that can be found.
# 
# To confirm that the installation worked, and the pip show command to access information on the aadvark package.
# 
# The code in this exercise can be used as part of a script to set up a virtual environment and install local packages in any directory.

# ###  code

# In[ ]:


print(run(
    # Install project dependencies
    [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
    stdout=-1
).stdout.decode())

print(run(
    # Show information on the aardvark package
    [".venv/bin/python", "-m", "pip", "show", "aardvark"], stdout=-1
).stdout.decode())


# # Persistence and packaging
# 

# ## Pickle dataframes
# In this exercise, we will
# 
# - create a Pandas dataframe from the diabetes dataset,
# - add column names based on the dataset documentation, and then
# - pickle and unpickle the dataframe using Pandas methods.
# 
# Finally, we will create a Pandas scatterplot to look at the relationship between a diabetes dataset predictor and the target variable.

# ### code

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()


# In[20]:


pd.DataFrame(
    np.c_[(diabetes.data, diabetes.target)],
    columns="age sex bmi map tc ldl hdl tch ltg glu target".split()
    # Pickle the diabetes dataframe with zip compression
    ).to_pickle("diabetes.pkl.zip")
                  
# Unpickle the diabetes dataframe
df = pd.read_pickle("diabetes.pkl.zip")
df.plot.scatter(x="ltg", y="target", c="age", colormap="viridis")
plt.show()


# ## Pickle models
# In our final persistence exercise, we will pickle and unpickle a scikit-learn model with joblib.
# 
# The training and test sets we will need to train the model and make predictions are already loaded.
# 
# Once we have the predictions, we will plot them against the measured values to assess model fit.

# ### init

# In[27]:


###################
##### numpy ndarray float
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(x_train, y_train, x_test, y_test)
"""

tobedownloaded="""
{numpy.ndarray: {'x_train.csv': 'https://file.io/gFit48qz',
  'y_train.csv': 'https://file.io/OYu4Rkm8',
  'x_test.csv': 'https://file.io/kliLVJ7u',
  'y_test.csv': 'https://file.io/tgWnV75i'}}
"""
prefixToc='4.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

from downloadfromFileIO import loadNDArrayFromCsv
x_train = loadNDArrayFromCsv(prefix+'x_train.csv')
y_train = loadNDArrayFromCsv(prefix+'y_train.csv')
x_test = loadNDArrayFromCsv(prefix+'x_test.csv')
y_test = loadNDArrayFromCsv(prefix+'y_test.csv')


# ### code

# In[28]:


import joblib
from sklearn.linear_model import LinearRegression


# In[29]:


# Train and pickle a linear model
joblib.dump(LinearRegression().fit(x_train, y_train), "linear.pkl")

# Unpickle the linear model
linear = joblib.load("linear.pkl")
predictions = linear.predict(x_test)
plt.scatter(y_test, predictions, edgecolors=(0, 0, 0))
min_max = [y_test.min(), y_test.max()]
plt.plot(min_max, min_max, "--", lw=3)
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.show()


# In[ ]:




