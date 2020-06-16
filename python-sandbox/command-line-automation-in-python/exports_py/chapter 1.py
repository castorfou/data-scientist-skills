#!/usr/bin/env python
# coding: utf-8

# # Learn the Python interpreter
# 

# ## Execute Python commands
# Your boss has the habit of tapping you on the shoulder and asking you to do random tasks. They always ask which day you can get it done by. You have noticed that you keep saying "Friday", but this is catching up to you. Create a random day generator to give your boss an answer. This way your work will be evenly distributed throughout the week. This solution is almost complete: "from random import choices;days = ['Mo', 'Tu', 'We', 'Th', 'Fr']"
# 
# What is the correct syntax for a "one liner" using python that will print a random day of the week? Feel free to experiment in the IPython console!
# 
# Note that in this chapter you'll be learning many nifty IPython commands to perform basic operating system tasks. As a result, most exercises will be Multiple Choice. But don't get too comfortable because towards the end of the chapter and throughout the rest of the course you'll be writing your very own python scripts to automate command line tasks!

# ### code

# In[1]:


from random import choices;days = ['Mo', 'Tu', 'We', 'Th', 'Fr']


# In[3]:


choices(days)


# In[5]:


get_ipython().system('python -c "from random import choices;days = [\'Mo\', \'Tu\', \'We\', \'Th\', \'Fr\'];print(choices(days))"')


# ## Execute IPython shell commands
# One of your coworkers has mentioned that you can do very powerful shell operations inside of the IPython terminal. You decide to try this out to solve a problem you are currently working on. You use this approach to see if you can determine how many files of a specific type live in a directory by using this along with the built-in len() method. The output of !ls will return a list which you can store as a variable.
# 
# How many total files with the extension .csv are in the test_dir directory? Make sure you store the results of command to variable and run len() on that variable. You can store a variable from a shell command in IPython like this: var = !ls -h *.png.

# ### code

# In[6]:


var = get_ipython().getoutput('ls -h test_dir/*.csv')

len(var)


# # Capture IPython Shell output
# 

# ## Using Bash Magic command
# You have an existing bash script that you need to run in a hurry and then capture the output to continue working on it in Python. You remember that you can use %%bash magic syntax to capture the output of a script in IPython. What is the proper way to run this script and capture the output as a Python variable?
# 
# Select the correct method of running a bash script in IPython. You can try this out in the IPython terminal if you get stuck. You will need to hit shift+enter to return to the next line. After you run the magic command, print the output variable to see the results.

# In[9]:


get_ipython().run_cell_magic('bash', '--out output', 'ls')


# In[10]:


output


# ## Using the ! operator in IPython
# You have a directory full of files you want the size of. You need a technique that will allow you to filter patterns. You want to do this in Python. Use the ! operator to create a command that sums the total size of the files in a directory. The piped command will use this awk snippet awk '{ SUM+=$5} END {print SUM}'. Awk is a tool that is used often on the Unix command line because it understands how to deal with whitespace delimited output from shell commands. The awk command works well at grabbing fields from a string.
# 
# Make sure you try the commands out in the IPython terminal!

# In[11]:


get_ipython().system("ls -l | awk '{SUM+=$5} END {print SUM}'")


# In[12]:


get_ipython().run_cell_magic('bash', '--out output', "ls -l | awk '{SUM+=$5} END {print SUM}'")


# In[14]:


output


# # Automate with SList
# 

# In[ ]:




