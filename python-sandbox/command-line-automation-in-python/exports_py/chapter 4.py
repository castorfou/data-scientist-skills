#!/usr/bin/env python
# coding: utf-8

# # Using functions for automation
# 

# ## Funky clusters
# You need to write an integration test that verifies that your cloud environment can run KMeans clustering algorithms. One issue you have had in the past is that it gets tedious to keep rewriting code that needs minor changes. You have learned that you can accomplish these small changes by creating a command line tool instead. To prepare your code to become a command line tool, first you must refactor it into functions. Write two functions that make your code ready to be run by a command line tool library.
# 
# The make_blobs and KMeans modules have been imported for you. These modules are from sklearn a widely used machine learning library.

# ### code

# In[1]:


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# In[2]:


# create sample blobs from sklearn datasets
def blobs():
    X, y = make_blobs(n_samples=10, centers=3, n_features=2,random_state=0)
    return X,y
  
# Perform KMeans cluster
def cluster(X, random_state=170, num=2):
    return KMeans(n_clusters=num, random_state=random_state).fit_predict(X) # Returns cluster assignment

#Run everything:  Call both functions. `X` creates the data and `cluster`, clusters the data.
def main():
    X,_ = blobs()
    return cluster(X)

print(main()) #print the KMeans cluster assignments


# ## Hello decorator
# You are working more and more with decorators and you want to ensure that decorators you write remember to use the @wraps functionality to preserve the name of the function and the docstring. One idea you have to verify this is to create an integration test that loops over decorators you create and prints out the names of the decorated functions. Use this approach to verify two decorated function names.

# ### init

# In[6]:


###################
##### inspect Function
###################

""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(nothing)
"""

from functools import wraps


def nothing(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        return f(*args, **kwds)
    return wrapper


# ### code

# In[7]:


# decorate first function
@nothing
def something():
    pass

# decorate second function
@nothing
def another():
    pass

# Put uncalled function into a list and print name  
funcs = [something, another]
for func in funcs:
    print(f"function name: {func.__name__}")


# ## Debugging decorator
# Your company has an internship program that involves training non-developers how to program in Python. One of the challenges is that it is often difficult to explain how to debug code to the interns. You have an idea to write a debugging decorator that interns can use that will print out both the arguments and the keyword arguments when they are applied to a function. You are confident this will be a very useful skill to teach them because of how much you use decorators to debug code and enhance automation tasks.
# 
# Write a decorator that will wrap a function and print any *args or **kw arguments out. Remember a decorator must return the the function is wraps. This is the last line of a decorator.

# ### code

# In[8]:


# create decorator
def debug(f):
	@wraps(f)
	def wrap(*args, **kw):
		result = f(*args, **kw)
		print(f"function name: {f.__name__}, args: [{args}], kwargs: [{kw}]")
		return result
	return wrap
  
# apply decorator
@debug
def mult(x, y=10):
	return x*y
print(mult(5, y=5))


# # Understand script input
# 

# ## Using python command-line tools
# There is a command-line executable written in python called findit.py. Use it to find files from the IPython terminal.
# You can run the findit.py script like this in the IPython prompt (note, you typically would run this in the bash shell):
# 
# !python3.6 findit.py
# 
# It takes two arguments using sys.argv to process them. The first argument is the path to search (i.e. /some/path) and the second is an optional argument that finds a file extension (i.e. .pdf). Use this script to search for files with the extension .txt in the test_dir.
# 
# Which is the name of one of the files it finds?

# ## Backwards day
# A colleague has written a Python script that reverse all lines in a file and prints them out one by one. This is an integration tool for some NLP (Natural Language Processing) work your department is involved in. You want to call their script, reverseit.py from a python program you are writing so you can use it as well. Use your knowledge of sys.argv and subprocess to write a file, then pass that file to the script that processes it.

# ### code

# In[10]:


import subprocess

# Write a file
with open("input.txt", "w") as input_file:
  input_file.write("Reverse this string\n")
  input_file.write("Reverse this too!")

# runs python script that reverse strings in a file line by line
run_script = subprocess.Popen(
    ["/usr/bin/python3", 'reverseit.py', 'input.txt'], stdout=subprocess.PIPE)

# print out the script output
for line in run_script.stdout:
  print(line)


# # Introduction to Click
# 

# ## Simple yet true
# You need to convince your company to both use the click command-line tool and to open their new headquarters in an affordable city. Use this opportunity to create a random city selector and then print the results to stdout using click.

# ### code

# In[11]:


import click
import random

# Create random values to choose from
values = ["Nashville", "Austin", "Denver", "Cleveland"]

# Select a random choice
result = random.choice(values)

# Print the random choice using click echo
click.echo(f"My choice is: {result}")


# ## Running a click application from subprocess
# You are building many click command line applications and you realize that it would be powerful to automate their execution by not only humans but other scripts. Take an existing script that performs KMeans clustering and execute it with two different options: help and num. Run this inside of subprocess.run and print both outputs to standard out.

# ### code

# In[12]:


import subprocess

# run help for click tool
help_out = subprocess.run(["/usr/bin/python3.6", "./cluster.py", "--help"],
                     stdout=subprocess.PIPE)

# run cluster
cluster2 = subprocess.run(["/usr/bin/python3.6", "./cluster.py", "--num", "2"],
                     stdout=subprocess.PIPE)

# print help
print(help_out.stdout)

# print cluster output
print(cluster2.stdout)


# # Using click to write command line tools
# 

# ## Got a ticket to write
# You have been reading through the click documentation while you are working on an data science project around housing prices in the United states. One thing you discover is that it incredibly easy to write to files from click. Take a few random words that come to mind and write them out via click. The click module has been imported for you.

# ### code

# In[13]:


import click


# In[14]:


# Setup
words = ["Asset", "Bubble", "10", "Year"]
filename = "words.txt"

# Write with click.open()
with click.open_file(filename, 'w') as f:

# Loop over words with a for loop
    for word in words:
        f.write(f'{word}\n')

# Read it back
with open(filename) as output_file:
    print(output_file.read())


# ## Invoking command line tests
# Not only has your department widely adopted click, but you are writing click applications so quickly that you need to ensure you know how to test them as well. The click framework has the ability to test itself using the CliRunner method. Lead by example and extend this click application so it can be tested properly and integrated into your continuous integration system.
# 
# Remember that click flags like --something need to be passed into functions as something to use them in the function.

# ### code

# In[18]:


from click.testing import CliRunner


# In[19]:


#define the click command
@click.command()
@click.option("--num", default=2, help="Number of clusters")
def run_cluster(num):
    result = main(num)
    click.echo(f'Cluster assignments: {result} for total clusters [{num}]')

# Create the click test runner
runner = CliRunner()

# Run the click app and assert it runs without error
result = runner.invoke(run_cluster, ['--num', '2'])
assert result.exit_code == 0
print(result.output)


# In[ ]:




