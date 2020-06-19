#!/usr/bin/env python
# coding: utf-8

# # Dealing with file systems
# 

# ## Double trouble
# The CEO at your startup has been very happy with previous Data Engineering solution you created that eliminates duplicates in a tree full of Terabytes of data. You have been tasked with another similar task of finding all of the .csv files in your company's data lake. These files will need to later move to a specific directory for a machine learning task. Your code could save hours of time if it performs as expected.
# 
# In this exercise, you will search for files that match specific patterns in a directory test_dir. The os module has already been imported for you.

# ### code

# In[1]:


import os


# In[2]:


matches = []
# Walk the filesystem starting at the test_dir
for root, _, files in os.walk('test_dir'):
    for name in files:
      	# Create the full path to the file by using os.path.join()
        fullpath = os.path.join(root, name)
        print(f"Processing file: {fullpath}")
        # Split off the extension and discard the rest of the path
        _, ext = os.path.splitext(fullpath)
        # Match the extension pattern .csv
        if ext == ".csv":
            matches.append(fullpath)
            
# Print the matches you find          
print(matches)


# ## Y'all got some renaming to do
# After escaping the oppressive rent of the Bay Area by moving to Texas, you got a job at a dude ranch as their only programmer. Once a year they sell all of their cattle in an auction. You wrote their inventory control system from scratch in Python and the CEO said, "Not bad for a kid from 'Frisco'". The day before the auction the CEO comes up to you in a panic because the names of all of the cattle are wrong in the inventory control system. The CEO tells you, "Y'all got some renaming to do!".
# 
# longhorn
# 
# Rename all of the files in the cattle directory by replacing the phrase 'shorthorn' with 'longhorn'. The os and pathlib modules have been imported for you. Remember that the name variable will need to be split to be renamed.

# ### init

# In[1]:


### sur datacamp

"""
!zip -r cattle.zip cattle
!tar zcvf cattle.tar.gz cattle
"""

###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('cattle.tar.gz')
"""

tobedownloaded="""
{numpy.ndarray: {'cattle.tar.gz': 'https://file.io/QUWGSDa8'}}
"""
prefixToc = '1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[4]:


import os
import pathlib 


# In[10]:


# Walk the filesystem starting at the test_dir
for root, _, files in os.walk('data_from_datacamp/'+'cattle'):
    for name in files:
      	
        # Create the full path to the file by using os.path.join()
        fullpath = os.path.join(root, name)
        print(f"Processing file: {fullpath}")
        
        # Rename file
        if "shorthorn" in name:
            p = pathlib.Path(fullpath)
            shortname = name.split("_")[0] # You need to split the name by underscore
            new_name = f"{shortname}_longhorn"
            print(f"Renaming file {name} to {new_name}")
            p.rename(new_name)


# ## Sweet pickle
# "It was the best of times, it was the worst of times..", Charles Dickens said in a Tale of Two Cities. He could also be talking about your startup. Initially things were amazing and you and your co-workers laughed in delight as the CTO churned out machine learning models dozens by the day. Often this would be at 2AM and you would arrive in the morning and find the serialized sklearn models waiting for the Data Science team to deploy to production.
# 
# Unfortunately, this was in fact too good to be true. Many of the models had serious flaws and this ultimately led to the CTO stepping down. IT Auditors want to determine how flawed these ML models were and back test the predictions for accuracy.
# 
# Use the os.walk module to find serialized models and test them for accuracy.

# ### init

# In[11]:


### sur datacamp

"""
!tar zcvf my.tar.gz my
"""

###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('my.tar.gz')
"""

tobedownloaded="""
{numpy.ndarray: {'my.tar.gz': 'https://file.io/16RyfkJp'}}
"""
prefixToc = '1.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# In[13]:


###################
##### numpy ndarray float
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_digits)
"""

tobedownloaded="""
 {numpy.ndarray: {'X_digits.csv': 'https://file.io/by0qFE2F'}}
 """
prefixToc='1.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadNDArrayFromCsv
X_digits = loadNDArrayFromCsv(prefix+'X_digits.csv')


# In[17]:


import joblib
import sklearn


# ### code

# In[18]:


# Walk the filesystem starting at the my path
for root, _, files in os.walk('data_from_datacamp/'+'my'):
    for name in files:
      	# Create the full path to the file by using os.path.join()
        fullpath = os.path.join(root, name)
        print(f"Processing file: {fullpath}")
        _, ext = os.path.splitext(fullpath)
        # Match the extension pattern .joblib
        if ext == ".joblib":
            clf = joblib.load(fullpath)
            break

# Predict from pickled model
print(clf.predict(X_digits))


# # Find files matching a pattern
# 

# ## Rogue founder code
# Working as employee number 10 at a small startup with millions in funding seemed like a dream job. Now it seems like a nightmare. There are constant outages in production and during the middle of those outages one of the founders builds Java .jar files on their laptop and via ssh loads the .jar files into production servers thinking this will fix the problem. You have mentioned that all software should go through a continuous deployment system.
# 
# After a several day continuous outage that caused permanent customer data loss caused by the founder's rogue coding practices, the founder finally listens to you. They ask you to help them identify all of the .jar files located on servers in the prod directory. Make sure you use the powerful recursive glob technique.

# ### init

# In[19]:


### sur datacamp

"""
!tar zcvf prod.tar.gz prod
"""

###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('prod.tar.gz')
"""

tobedownloaded="""
{numpy.ndarray: {'prod.tar.gz': 'https://file.io/sOhWKx9k'}}
"""
prefixToc = '2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[20]:


import pathlib
import os

path = pathlib.Path('data_from_datacamp/'+'prod')
matches = sorted(path.glob('*.jar'))
for match in matches:
  print(f"Found rogue .jar file in production: {match}")


# ## Is this pattern True?
# As the head of data science you often get called in to help a new data scientists track down an important intermediate csv file that has gone missing. This has taken so much time, that you have decided to write an automated script that identify all csv files that are created by a user and then copy them to centralized storage. As the first step to create this near line backup solution you have to write a function that can filter and return only csv matches.
# 
# Use the fnmatch.filter function to filter for csv files from a list of files. Make sure you write a Python function so it can be portable code that a larger system can be built from.

# ### code

# In[22]:


import fnmatch

# List of file names to process
files = ["data1.csv", "script.py", "image.png", "data2.csv", "all.py"]

# Function that returns 
def csv_matches(list_of_files):
    """Return matches for csv files"""

    matches = fnmatch.filter(list_of_files, "*.csv")
    return matches

# Call function to find matches
matches = csv_matches(files)
print(f"Found matches: {matches}")


# # High-level file and directory operations
# 

# ## Goons over my shammy
# You are many light years from earth. Python programming aliens (Goons) abducted you and are forcing you to wash computer monitors. Goons keep stepping on your shammy (towel) and stopping you from working. You tell the Goons if they quit stepping on your shammy, you will write a Python script that programmatically creates self-destructing files and directories.
# 
# One of the uses for self-destructing files is to create integration tests. Integration tests can use temporary directory and files as a way of validating that processes in an application is doing what is expected. They tell you, "you had us at self-destruction". The tempfile and os module have been imported for you. Remember that tempfile.NameTemporaryFile object has many useful methods on it including .name.

# ### code

# In[23]:


import tempfile, os


# In[31]:


# Create a self-destructing temporary file
with tempfile.NamedTemporaryFile() as exploding_file:
  	# This file will be deleted automatically after the with statement block
    print(f"Temp file created: {exploding_file.name}")
    exploding_file.write(b"This message will self-destruct in 5....4...\n")
    
    # Get to the top of the file
    exploding_file.seek(0)

    #Print the message
    print(exploding_file.read())

# Check to sure file self-destructed
if not os.path.exists(exploding_file.name): 
    print(f"self-destruction verified: {exploding_file.name}")


# ## Archive Users
# At your university they have hired you to be an assistance for the machine learning course. It has been a very rewarding job, but some parts of the job are frustrating. You get frequent requests to retrieve items from a user's project after the course is over. This has consumed much of your time. You believe you can write an automation script that will archive all user folders and email them the archived copy. If you can do this, it will eliminate about 80% of the work you perform each semester. You are hoping to spend the recovered time helping the lead instructor deliver machine learning content. Use the shutil.archive function to archive a user directory. You will create two archive types: gztar and zip. make_archive and rmtree have been imported for you.

# ### code

# In[32]:


import shutil


# In[35]:


username = "user1"
root_dir = "/tmp"
# archive root
apath = "/tmp/archive"
# archive base
final_archive_base = f"{apath}/{username}"

# create tar and gzipped archive
shutil.make_archive(final_archive_base, 'gztar', apath)

# create zip archive
shutil.make_archive(final_archive_base, 'zip', apath)

# print out archives
print(os.listdir(apath))


# # Using pathlib
# 

# ## Does it even exist?
# Your a social media company with a huge problem. Your NoSQL database cluster was upgraded to 1.0.0-beta because it had some really cool new features. Around this same time files started disappearing in production and social media posts were suddenly vanishing. It turns out the beta version of the database was actually deleting data between cluster nodes, not syncing data. Even worse, the backups were never tested and the same backup from a year previous was being run over and over again. You have a list of all social media posts that should exist in production, and you need to write a script that audits which files actually exist.
# 
# Write a script using pathlib that validates if a list of files exists on disk. Remember you can explore pathlib.Path in IPython.

# ### init

# In[38]:


### sur datacamp

"""
!tar zcvf socialposts.tar.gz *
"""

###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('socialposts.tar.gz')
"""

tobedownloaded="""
{numpy.ndarray: {'socialposts.tar.gz': 'https://file.io/iwg30QD8'}}
"""
prefixToc = '4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[40]:


import pathlib

# Read the index of social media posts
with open('data_from_datacamp/'+"posts_index.txt") as posts:
  for post in posts.readlines():
    
    # Create a pathlib object
    path = pathlib.Path(post.strip())
    
    # Check if the social media post still exists on disk
    if path.exists():
      print(f"Found active post: {post}")
    else:
      print(f"Post is missing: {post}")


# ## File writing one-liner
# As a Data Engineer at a Fortune 500 company you need to make sure your large scale data pipeline is running smoothly. Recently you have been experienced unpredictable errors when running Spark Python jobs. You want to write an integration test that programatically creates Python files, gives them executable permission and the runs them. You want to run this everytime you create IaC (Infrastructure as Code) scripts that provision new Spark clusters.
# 
# Create an integration script that creates several Python files and writes Python to them. After that run them all with python3 and subprocess to obtain the scripts' output. The Path module is imported for you.

# ### code

# In[46]:


from pathlib import Path


# In[48]:


from subprocess import run, PIPE

# Find all the python files you created and print them out
for i in range(3):
  path = Path(f"/tmp/test/file_{i}.py")
  path.write_text("#!/usr/bin/env python\n")
  path.write_text("import datetime;print(datetime.datetime.now())")
  

# Find all the python files you created and print them out
for file in Path("/tmp/test/").glob("*.py"):
  # gets the resolved full path
  fullpath = str(file.resolve())
  proc = run(["python3", fullpath], stdout=PIPE)
  print(proc)


# In[ ]:




