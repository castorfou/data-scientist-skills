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

# ##  Use SList fields to parse shell output
# A Data Scientist who you highly respect at work mentioned that IPython has a powerful data type called SList that enables a user to perform powerful operations on shell commands. In particular they mention that there were able to easily extract fields from the output of the df command. In this exercise you investigate what you can accomplish with the SList data type. You will start from this command: disk_space = !df -h
# 
# Using the .fields method on the df variable, select the column that shows total size of the mounted volumes.

# In[1]:


disk_space = get_ipython().getoutput('df -hl')


# In[2]:


disk_space


# In[4]:


disk_space.fields(1)


# ## Find Python files using SLIST grep
# A coworker has written a script that has gone haywire. You both have been working in a directory src and this is where you store your python scripts. Your coworker accidentally wrote 250 text files that have similar names to your python files. Help clean up the mess by using the SList .grep() method to filter for files only containing the pattern .py.
# 
# What are the names of only the Python source code files in the src directory? Remember to store the output of !ls src. The .grep() method will accept a file extension as a pattern.

# In[ ]:


src = get_ipython().getoutput('ls src')
src.grep('.py')
Out[3]: ['apple.py', 'banana.py', 'orange.py']


# ## Using SList to grep
# You get woken up in the middle of the night by a frantic phone call from a co-worker. There is a rogue process running in production that is generating hundreds of extra backup files. It was discovered when your co-worker tried to restore from backup, but found hundreds of backup files that are corrupt. You need to write a script to isolate the correct backup file.
# 
# Use the SList object to find all files with the number 2 in them and print out their full path so the backups can be inspected. The reason 2 is so important is this corresponds to the second day of the week Tuesday. This is the last time the backups worked properly.

# ### code

# In[5]:


# Use the results of an SList object
root = "test_dir"

# Find the backups with "_2" in them
result = slist_out.grep('_2')

# Extract the filenames
for res in result:
	filename = res.split()[-1]
    
	# Create the full path
	fullpath = os.path.join(root, filename)
	print(f"fullpath of backup file: {fullpath}")


# In[ ]:




