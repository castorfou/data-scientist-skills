#!/usr/bin/env python
# coding: utf-8

# # Machine Learning & Spark
# 

# # Connecting to Spark
# 

# In[2]:


import findspark
findspark.init()

import numpy as np
import pandas as pd
import pyspark
pyspark.__version__


# In[3]:


from pyspark.sql import SparkSession


# In[4]:


spark  = SparkSession.builder.master('local[*]').appName('first_spark-application').getOrCreate()


# ## Creating a SparkSession
# In this exercise, you'll spin up a local Spark cluster using all available cores. The cluster will be accessible via a SparkSession object.
# 
# The SparkSession class has a builder attribute, which is an instance of the Builder class. The Builder class exposes three important methods that let you:
# 
# specify the location of the master node;
# name the application (optional); and
# retrieve an existing SparkSession or, if there is none, create a new one.
# The SparkSession class has a version attribute which gives the version of Spark.
# 
# Find out more about SparkSession here.
# 
# Once you are finished with the cluster, it's a good idea to shut it down, which will free up its resources, making them available for other processes.
# 
# Note:: You might find it useful to revise the slides from the lessons in the Slides panel next to the IPython Shell.

# ### code

# In[10]:


# Import the PySpark module
from pyspark.sql import SparkSession
# Create SparkSession object
spark = SparkSession.builder                     .master('local[*]')                     .appName('test')                     .getOrCreate()

# What version of Spark?
print(spark.version)

# Terminate the cluster
spark.stop()


# # Loading Data
# 

# ## Loading flights data
# In this exercise you're going to load some airline flight data from a CSV file. To ensure that the exercise runs quickly these data have been trimmed down to only 50 000 records. You can get a larger dataset in the same format here.
# 
# Notes on CSV format:
# 
# fields are separated by a comma (this is the default separator) and
# missing data are denoted by the string 'NA'.
# Data dictionary:
# 
# - mon — month (integer between 1 and 12)
# - dom — day of month (integer between 1 and 31)
# - dow — day of week (integer; 1 = Monday and 7 = Sunday)
# - org — origin airport (IATA code)
# - mile — distance (miles)
# - carrier — carrier (IATA code)
# - depart — departure time (decimal hour)
# - duration — expected duration (minutes)
# - delay — delay (minutes)
# 
# pyspark has been imported for you and the session has been initialized.
# 
# Note: The data have been aggressively down-sampled.

# ### init

# In[11]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('flights.csv')
"""

tobedownloaded="""
{numpy.ndarray: {'flights.csv': 'https://file.io/XfVNqs'}}
"""
prefixToc = '3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# In[12]:


# Import the PySpark module
from pyspark.sql import SparkSession
# Create SparkSession object
spark = SparkSession.builder                     .master('local[*]')                     .appName('test')                     .getOrCreate()

# What version of Spark?
print(spark.version)


# ### code

# In[13]:


# Read data from CSV file
flights = spark.read.csv(prefix+'flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)

# Check column data types
flights.dtypes


# ## Loading SMS spam data
# You've seen that it's possible to infer data types directly from the data. Sometimes it's convenient to have direct control over the column types. You do this by defining an explicit schema.
# 
# The file sms.csv contains a selection of SMS messages which have been classified as either 'spam' or 'ham'. These data have been adapted from the UCI Machine Learning Repository. There are a total of 5574 SMS, of which 747 have been labelled as spam.
# 
# Notes on CSV format:
# 
# - no header record and
# - fields are separated by a semicolon (this is not the default separator).
# 
# Data dictionary:
# 
# - id — record identifier
# - text — content of SMS message
# - label — spam or ham (integer; 0 = ham and 1 = spam)

# ### init

# In[14]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('sms.csv')
"""

tobedownloaded="""
{numpy.ndarray: {'sms.csv': 'https://file.io/D78i7N'}}
"""
prefixToc = '3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# ### code

# In[18]:


from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Specify column names and types
schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])

# Load data from a delimited file
sms = spark.read.csv(prefix+'sms.csv', sep=';', header=False, schema=schema)

# Print schema of DataFrame
sms.printSchema()


# In[ ]:




