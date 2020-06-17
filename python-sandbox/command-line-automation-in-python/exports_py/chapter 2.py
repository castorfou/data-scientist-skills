#!/usr/bin/env python
# coding: utf-8

# # Execute shell commands in subprocess
# 

# ## Permissions Check
# As the CTO of your small startup you often have to perform many roles. Some days this means being the data scientists, data engineer, DevOps engineer and machine learning engineer all at once. Recently you setup a large network file system in your cloud deployment that all of the instances that perform machine learning will communicate with. You are a strong believe in IaC (Infrastructure as Code). As a result you want to verify that the when the network filesystem is mounted on a new system that each worker node is able to create files with the correct permissions.
# 
# Write a script that will check for this by using subprocess.Popen and os.stat. Be sure to use the variables in setup in your script!

# ### code

# In[1]:


import subprocess
import os

#setup
file_location = "/tmp/file.txt"
expected_uid = 1000
#touch a file
proc = subprocess.run(["touch", file_location])

#check user permissions
stat = os.stat(file_location)
if stat.st_uid == expected_uid:
  print(f"File System exported properly: {expected_uid} == {stat.st_uid}")
else:
  print(f"File System NOT exported properly: {expected_uid} != {stat.st_uid}")


# ## Reading a creepy AI poem
# As a mad scientists working on AGI (Artificial General Intelligence) in your underground bunker in Siberia, you have come up with a program that appears to show signs of human level intelligence. Your program was trained to write poems and initially showed signs of true brilliance. You read one of the poems and it seemed a bit creepy and repetitive. You need to write a script that inspects some of the output of the computer generated poems it is writing. The Unix command head will read the first few lines. The Unix command wc -w will count the total number of words. The name of the poem is called poem.txt.
# 
# Use subprocess.Popen to run each of these shell commands print the results. You must pass stdout=subprocess.PIPE into Popen to capture the output of wc.

# ### init

# In[6]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('poem.txt')

"""

tobedownloaded="""
{numpy.ndarray: {'poem.txt': 'https://file.io/hIuUKx08'}}
"""
prefixToc = '1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# In[3]:


from subprocess import (Popen, PIPE)


# ### code

# In[7]:


import subprocess

# Execute Unix command `head` safely as items in a list
with subprocess.Popen(["head", prefix+"poem.txt"], stdout=PIPE) as head:
  
  # Print each line of list returned by `stdout.readlines()`
  for line in head.stdout.readlines():
    print(line)
    
# Execute Unix command `wc -w` safely as items in a list
with subprocess.Popen(['wc', '-w', prefix+"poem.txt"], stdout=PIPE) as word_count:
  
  # Print the string output of standard out of `wc -w`
  print(word_count.stdout.read())


# ## Running processes script
# The year is 2040 and you share a Unix terminal with a mixture of AI and human co-workers. Recently your AI boss has been shorter than usual in your telepathic exchanges. You notice your daily meal biscuit quality has been downgraded to "sufficient" from "good". You are concerned your AI boss may recommend a transfer to the social media news content moderation team. One day your AI boss asks you to monitor CPU usage. They suspect someone is using CPU processing power to run other code instead of writing unit tests for the code the AI bots produce.
# 
# Write a script using subprocess.run and ps aux that discards all CPU output with the string 'python' in it. This will hide your secret Python scripts from the AI.

# ### code

# In[16]:


import subprocess

# Use subprocess to run the `ps aux` command that lists running processes
with subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE) as proc:
    process_output = proc.stdout.readlines()
    
# Look through each line in the output and skip it if it contains "python"
for line in process_output:
    if b'python' in line:
        continue
    print(line)


# # Capture output of shell commands
# 

# ## Using subprocess Popen
# A coworker is proficient in Bash tells you that most data engineering tasks should be done in the shell. You mention a scripting language like Python can build robust production systems that have high quality. The code is often easier in practice to write and maintain, even if you are directly calling shell commands. You demonstrate how this works using a small Python script that you write that finds all of the Python packages installed.
# 
# Use Python, subprocess.Popen(), and the bash command pip list --format=json command, to find all of the installed packages. The pip tool is a common method of installing Python packages. The result will be a byte string, a Python3 construct. The Popen command will use PIPE to send the JSON output to stdout.

# ### code

# In[20]:


from subprocess import Popen, PIPE
import json
import pprint

# Use the with context manager to run subprocess.Popen()
with Popen(["pip","list","--format=json"], stdout=PIPE) as proc:
  # Pipe the output of subprocess.Popen() to stdout
  result = proc.stdout.readlines()
  
# Convert the JSON payload to a Python dictionary
# JSON is a datastructure similar to a Python dictionary
converted_result = json.loads(result[0])

# Display the result in the IPython terminal
pprint.pprint(converted_result)


# ## Waiting for processes
# In the real-world code is messy. There are edge cases that have to be handled, and things don't always go as planned. Dealing with data increases the complexity of the mess.
# 
# In this example you will be using the subprocess module to launch a "misbehaving" process that will run for six seconds. This will be simulated by using linux sleep command. The sleep command will suspend execution of a shell for a period of time. You will use the subprocess.communicate() method to wait for the command to finish running for up to five seconds. The process will then timeout and it will return an Exception: i.e. error detected during execution, which will be caught and the process will be cleaned up by proc.kill(). Popen, PIPE, and TimeoutExpired have all been imported for you.

# ### code

# In[24]:


from subprocess import (Popen, PIPE, TimeoutExpired)


# In[29]:


# Start a long running process using subprocess.Popen()
proc = Popen(["sleep", "6"], stderr=PIPE, stdout=PIPE)

# Use subprocess.communicate() to create a timeout 
try:
    output, error = proc.communicate(timeout=5)
    
except TimeoutExpired:

	# Cleanup the process if it takes longer than the timeout
    proc.kill()
    
    # Read standard out and standard error streams and print
    output, error = proc.communicate()
    print(f"Process timed out with output: {output}, error: {error}")


# ## Detecting duplicate files with subprocess
# Imagine you are a new data scientist at a startup, and you have been tasked with doing machine learning on Terabytes of data. The CEO has mentioned they have a small budget to train your model. You notice many duplicate files when manually inspecting. If you can identify the duplicate files before you begin training, this potentially saves 50% of the cost of training.
# 
# In this exercise, you will find duplicate files by using the subprocess.Popen() module and capturing the output of the md5sum command. The md5sum utility is a shell command that finds the unique hash of each file. There is a list of files available via the files variable that you can iterate over. Popen and PIPE have already been imported for you from the subprocess module. ['file_8.csv'...]

# ### init

# In[31]:


files=['test_dir/file_0.csv',
 'test_dir/file_2.csv',
 'test_dir/file_3.csv',
 'test_dir/file_9.csv',
 'test_dir/file_7.csv',
 'test_dir/file_4.csv',
 'test_dir/file_8.csv',
 'test_dir/file_5.csv',
 'test_dir/file_6.csv',
 'test_dir/file_1.csv']


# ### code

# In[32]:


checksums = {}
duplicates = []

# Iterate over the list of files filenames
for filename in files:
  	# Use Popen to call the md5sum utility
    with Popen(['md5sum', filename], stdout=PIPE) as proc:
        checksum, _ = proc.stdout.read().split()
        
        # Append duplicate to a list if the checksum is found
        if checksum in checksums:
            duplicates.append(filename)
        checksums[checksum] = filename

print(f"Found Duplicates: {duplicates}")


# # Sending input to processes
# 

# ## Counting files in a directory tree
# After the last bad experience with corrupt backups at your company, you decide to rewrite the backup script from scratch in Python. One of the improvements you want to make is to audit the number of files in a subdirectory and count them. You will then ensure the exact same number of files exists in a directory tree before and after the backup. This will create a validation step that was missing in the last script.
# 
# Use subprocess.run to pipe the output of the find command to wc -l to print the numbers of files in the directory tree.

# ### code

# In[33]:


import subprocess

# runs find command to search for files
find = subprocess.Popen(
    ["find", ".", "-type", "f", "-print"], stdout=PIPE)

# runs wc and counts the number of lines
word_count = subprocess.Popen(
    ["wc", "-l"], stdin=find.stdout, stdout=PIPE)

# print the decoded and formatted output
output = word_count.stdout.read()
print(output.decode('utf-8').strip())


# ## Running a health check
# The data science team at your company has been working closely with the DevOps team to ensure the production machine learning systems are reliable, elastic and fault-tolerant. Recently, there was an outage of a critical system that cost the company hundreds of thousands of dollars in lost revenue when a machine learning model began throwing exceptions instead of offering recommendations to shoppers. One solution that can be implemented is to run periodic health checks on production systems to ensure they have the correct environment. The DevOps team has written several bash scripts that your team will need to invoke from Python and run periodically.
# 
# Send the output of an echo 'python3' command to a healthcheck.sh script.

# ### init

# In[34]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('healthcheck.sh')
"""

tobedownloaded="""
{numpy.ndarray: {'healthcheck.sh': 'https://file.io/ycp4puf8'}}
"""
prefixToc = '3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[37]:


import subprocess

# equivalent to 'echo "python3"'
echo = subprocess.Popen(
    ['echo', 'python'], stdout=PIPE)

# equivalent to: echo "python3" | ./healthcheck.sh
path = subprocess.Popen(
    [prefix+'healthcheck.sh'], stdin=echo.stdout, stdout=PIPE)

full_path = path.stdout.read().decode("utf-8")
print(f"...Health Check Output...\n\n {full_path}")

# The assertion will fail if python3 executable path is not found
assert "python" in full_path


# # Passing arguments safely to shell commands
# 

# ## Safely find directories
# At the film studio you work at there is a need to create a tool for animators to search incredibly large file systems as quickly as possible and find all of the directories in them. You benchmarked both regular python code and python code that uses the Unix find command to see which performed better at searches. You have determined for this particular problem that the Unix find performs the best. One concern you have though is ensuring that the command safely processes user input. In the past tools have been released that accidentally deleted large sections of the file server because a user accidentally put the wrong string into a tool.
# 
# Write a tool that safely processes user input and searches a file system for all directories using find and subprocess.Popen.

# ### code

# In[39]:


import subprocess

#Accepts user input
print("Enter a path to search for directories: \n")
user_input = "."
print(f"directory to process: {user_input}")

#Pass safe user input into subprocess
with subprocess.Popen(["find", user_input, "-type", "d"], stdout=subprocess.PIPE) as find:
    result = find.stdout.readlines()
    
    #Process each line and decode it and strip it
    for line in result:
        formatted_line = line.decode("utf-8").strip()
        print(f"Found Directory: {formatted_line}")


# ## Directory summarizer
# Your high performance laptop with solid state drives and a GPU capable of doing deep learning was a great investment. One issue you have though is that your hard drive keeps filling up with machine learning data. You need to write a script that will calculate the total disk usage from an arbitrary amount of directories you pass in. After you finish this script locally you plan on using it on your work file system as well. Eventually you then turn it into a sophisticated tool that manages disk storage in Python or you would just use the Unix du command alone. You are very concerned about accepting user input that could permanently delete user data or cause a security hole.
# 
# Use shlex and subprocess to get the total storage of a list of directories.

# ### code

# In[40]:


import shlex


# In[42]:


print("Enter a list of directories to calculate storage total: \n")
user_input = "pluto mars jupiter"

# Sanitize the user input
sanitized_user_input = shlex.split(user_input)
print(f"raw_user_input: {user_input} |  sanitized_user_input: {sanitized_user_input}")

# Safely Extend the command with sanitized input
cmd = ["du", "-sh", "--total"]
cmd.extend(sanitized_user_input)
print(f"cmd: {cmd}")

# Print the totals out
disk_total = subprocess.run(cmd, stdout=PIPE)
print(disk_total.stdout.decode("utf-8"))


# In[ ]:




