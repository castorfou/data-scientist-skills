#!/usr/bin/env python
# coding: utf-8

# # One-Hot Encoding
# 

# ## Encoding flight origin
# The org column in the flights data is a categorical variable giving the airport from which a flight departs.
# 
# - ORD — O'Hare International Airport (Chicago)
# - SFO — San Francisco International Airport
# - JFK — John F Kennedy International Airport (New York)
# - LGA — La Guardia Airport (New York)
# - SMF — Sacramento
# - SJC — San Jose
# - TUS — Tucson International Airport
# - OGG — Kahului (Hawaii)
# 
# Obviously this is only a small subset of airports. Nevertheless, since this is a categorical variable, it needs to be one-hot encoded before it can be used in a regression model.
# 
# The data are in a variable called flights. You have already used a string indexer to create a column of indexed values corresponding to the strings in org.
# 
# Note:: You might find it useful to revise the slides from the lessons in the Slides panel next to the IPython Shell.

# ### init

# In[1]:


import findspark
findspark.init()

import numpy as np
import pandas as pd
import pyspark
pyspark.__version__
# Import the PySpark module
from pyspark.sql import SparkSession
# Create SparkSession object
spark = SparkSession.builder                     .master('local[*]')                     .appName('test')                     .getOrCreate()

# What version of Spark?
print(spark.version)


# In[6]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
df_flights = flights.toPandas()
uploadToFileIO(df_flights)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df_flights.csv': 'https://file.io/zm7yeJ'}}
"""
prefixToc = '1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# Read data from CSV file
flights = spark.read.csv(prefix+'df_flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

flights=flights.drop('_c0')
# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)

flights = flights.withColumn("delay", flights.delay.cast('integer'))

# Check column data types
flights.dtypes


# ### code

# In[8]:


# Import the one hot encoder class
from pyspark.ml.feature import OneHotEncoderEstimator

# Create an instance of the one hot encoder
onehot = OneHotEncoderEstimator(inputCols=['org_idx'], outputCols=['org_dummy'])

# Apply the one hot encoder to the flights data
onehot = onehot.fit(flights)
flights_onehot = onehot.transform(flights)

# Check the results
flights_onehot.select('org', 'org_idx', 'org_dummy').distinct().sort('org_idx').show()


# ## Encoding shirt sizes
# You have data for a consignment of t-shirts. The data includes the size of the shirt, which is given as either S, M, L or XL.
# 
# Here are the counts for the different sizes:
# 
# +----+-----+
# |size|count|
# +----+-----+
# |   S|    8|
# |   M|   15|
# |   L|   20|
# |  XL|    7|
# +----+-----+
# The sizes are first converted to an index using StringIndexer and then one-hot encoded using OneHotEncoderEstimator.
# 
# Which of the following is not true:

# # Regression

# ## Flight duration model: Just distance
# In this exercise you'll build a regression model to predict flight duration (the duration column).
# 
# For the moment you'll keep the model simple, including only the distance of the flight (the km column) as a predictor.
# 
# The data are in flights. The first few records are displayed in the terminal. These data have also been split into training and testing sets and are available as flights_train and flights_test.

# ### init

# In[11]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
df_flights = flights.toPandas()
df_flights_train = flights_train.toPandas()
df_flights_test = flights_test.toPandas()
uploadToFileIO(df_flights, df_flights_train, df_flights_test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df_flights.csv': 'https://file.io/EVQEya',
  'df_flights_test.csv': 'https://file.io/k8CBKx',
  'df_flights_train.csv': 'https://file.io/qNQAx6'}}
"""
prefixToc = '2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# Read data from CSV file
flights = spark.read.csv(prefix+'df_flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')
flights_train = spark.read.csv(prefix+'df_flights_train.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')
flights_test = spark.read.csv(prefix+'df_flights_test.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

flights=flights.drop('_c0')
flights_train=flights.drop('_c0')
flights_test=flights.drop('_c0')
# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)
flights_train.show(5)
flights_test.show(5)


flights = flights.withColumn("delay", flights.delay.cast('integer'))
flights_train = flights_train.withColumn("delay", flights_train.delay.cast('integer'))
flights_test = flights_test.withColumn("delay", flights_test.delay.cast('integer'))

# Check column data types
flights.dtypes


# In[26]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

def turnToVector(dataset):
    # Repeat the process for the other categorical feature
    dataset_indexed = StringIndexer(inputCol='org', outputCol='org_idx2').fit(dataset).transform(dataset)
    dataset_indexed = dataset_indexed.drop('org_dummy', 'features')
    # Create a OneHotEncoder
    dataset_indexed = OneHotEncoder(inputCol='org_idx2', outputCol='org_dummy').transform(dataset_indexed)
    dataset = dataset_indexed.drop('org_idx2')
    # Create an assembler object
    assembler = VectorAssembler(inputCols=[
        'km'
    ], outputCol='features')
    # Consolidate predictor columns
    dataset = assembler.transform(dataset)
    return dataset

flights=turnToVector(flights)
flights_train=turnToVector(flights_train)
flights_test=turnToVector(flights_test)


# ### code

# In[31]:


from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a regression object and train on training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
predictions = regression.transform(flights_test)
predictions.select('duration', 'prediction').show(5, False)

# Calculate the RMSE
RegressionEvaluator(labelCol='duration').evaluate(predictions)


# ## Interpreting the coefficients
# The linear regression model for flight duration as a function of distance takes the form
# 
# duration=α+β×distance
# where
# 
# - α — intercept (component of duration which does not depend on distance) and
# - β — coefficient (rate at which duration increases as a function of distance; also called the slope).
# 
# By looking at the coefficients of your model you will be able to infer
# 
# - how much of the average flight duration is actually spent on the ground and
# - what the average speed is during a flight.
# 
# The linear regression model is available as regression.

# ### code

# In[33]:


# Intercept (average minutes on ground)
inter = regression.intercept
print(inter)

# Coefficients
coefs = regression.coefficients
print(coefs)

# Average minutes per km
minutes_per_km = regression.coefficients[0]
print(minutes_per_km)

# Average speed in km per hour
avg_speed = 60 / minutes_per_km
print(avg_speed)


# ## Flight duration model: Adding origin airport
# Some airports are busier than others. Some airports are bigger than others too. Flights departing from large or busy airports are likely to spend more time taxiing or waiting for their takeoff slot. So it stands to reason that the duration of a flight might depend not only on the distance being covered but also the airport from which the flight departs.
# 
# You are going to make the regression model a little more sophisticated by including the departure airport as a predictor.
# 
# These data have been split into training and testing sets and are available as flights_train and flights_test. The origin airport, stored in the org column, has been indexed into org_idx, which in turn has been one-hot encoded into org_dummy. The first few records are displayed in the terminal.
# 

# In[34]:


flights.show(
)


# ### init

# In[51]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
df_flights = flights.toPandas()
df_flights_train = flights_train.toPandas()
df_flights_test = flights_test.toPandas()
uploadToFileIO(df_flights, df_flights_train, df_flights_test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df_flights.csv': 'https://file.io/CEJm6A',
  'df_flights_test.csv': 'https://file.io/Fw6X72',
  'df_flights_train.csv': 'https://file.io/tSbDDp'}}
"""
prefixToc = '2.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# Read data from CSV file
flights = spark.read.csv(prefix+'df_flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')
flights_train = spark.read.csv(prefix+'df_flights_train.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')
flights_test = spark.read.csv(prefix+'df_flights_test.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

flights=flights.drop('_c0')
flights_train=flights.drop('_c0')
flights_test=flights.drop('_c0')
# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)
flights_train.show(5)
flights_test.show(5)


flights = flights.withColumn("delay", flights.delay.cast('integer'))
flights_train = flights_train.withColumn("delay", flights_train.delay.cast('integer'))
flights_test = flights_test.withColumn("delay", flights_test.delay.cast('integer'))

# Check column data types
flights.dtypes


# In[67]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

def turnOrgKMToVector(dataset):
    # Repeat the process for the other categorical feature
    dataset_indexed = StringIndexer(inputCol='org', outputCol='org_idx2').fit(dataset).transform(dataset)
    dataset_indexed = dataset_indexed.drop('org_dummy', 'features')
    # Create a OneHotEncoder
    dataset_indexed = OneHotEncoder(inputCol='org_idx2', outputCol='org_dummy').transform(dataset_indexed)
    dataset = dataset_indexed.drop('org_idx2')
    # Create an assembler object
    assembler = VectorAssembler(inputCols=['km','org_dummy'], outputCol='features')
    dataset = assembler.transform(dataset)
    return dataset

# Create an assembler object
flights=turnOrgKMToVector(flights)
flights_train=turnOrgKMToVector(flights_train)
flights_test=turnOrgKMToVector(flights_test)


# In[68]:


flights.dtypes


# In[70]:


flights.show()


# ### code

# In[72]:


from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a regression object and train on training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Create predictions for the testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE on testing data
RegressionEvaluator(labelCol='duration').evaluate(predictions)


# ## Interpreting coefficients
# Remember that origin airport, org, has eight possible values (ORD, SFO, JFK, LGA, SMF, SJC, TUS and OGG) which have been one-hot encoded to seven dummy variables in org_dummy.
# 
# The values for km and org_dummy have been assembled into features, which has eight columns with sparse representation. Column indices in features are as follows:
# 
# - 0 — km
# - 1 — ORD
# - 2 — SFO
# - 3 — JFK
# - 4 — LGA
# - 5 — SMF
# - 6 — SJC and
# - 7 — TUS.
# 
# Note that OGG does not appear in this list because it is the reference level for the origin airport category.
# 
# In this exercise you'll be using the intercept and coefficients attributes to interpret the model.
# 
# The coefficients attribute is a list, where the first element indicates how flight duration changes with flight distance.

# ### code

# In[74]:


# Average speed in km per hour
avg_speed_hour = 60 / regression.coefficients[0]
print(avg_speed_hour)

# Average minutes on ground at OGG
inter = regression.intercept
print(inter)

# Average minutes on ground at JFK
avg_ground_jfk = inter + regression.coefficients[3]
print(avg_ground_jfk)

# Average minutes on ground at LGA
avg_ground_lga = inter + regression.coefficients[4]
print(avg_ground_lga)


# # Bucketing & Engineering
# 

# ## Bucketing departure time
# Time of day data are a challenge with regression models. They are also a great candidate for bucketing.
# 
# In this lesson you will convert the flight departure times from numeric values between 0 (corresponding to 00:00) and 24 (corresponding to 24:00) to binned values. You'll then take those binned values and one-hot encode them.
# 

# ### init

# In[75]:


prefix = 'data_from_datacamp/chapter1-Exercise3.1_'
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


# ### code

# In[78]:


from pyspark.ml.feature import Bucketizer, OneHotEncoderEstimator

# Create buckets at 3 hour intervals through the day
buckets = Bucketizer(splits=[0,3,6,9,12,15,18,21,24], inputCol='depart', outputCol='depart_bucket')

# Bucket the departure times
bucketed = buckets.transform(flights)
bucketed.select('depart', 'depart_bucket').show(5)

# Create a one-hot encoder
onehot = OneHotEncoderEstimator(inputCols=['depart_bucket'], outputCols=['depart_dummy'])

# One-hot encode the bucketed departure times
flights_onehot = onehot.fit(bucketed).transform(bucketed)
flights_onehot.select('depart', 'depart_bucket', 'depart_dummy').show(5)


# ## Flight duration model: Adding departure time
# In the previous exercise the departure time was bucketed and converted to dummy variables. Now you're going to include those dummy variables in a regression model for flight duration.
# 
# The data are in flights. The km, org_dummy and depart_dummy columns have been assembled into features, where km is index 0, org_dummy runs from index 1 to 7 and depart_dummy from index 8 to 14.
# 
# The data have been split into training and testing sets and a linear regression model, regression, has been built on the training data. Predictions have been made on the testing data and are available as predictions.
# ![image.png](attachment:image.png)

# ### init

# In[85]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
df_predictions = predictions.toPandas()
df_flights_train = flights_train.toPandas()
df_flights_test = flights_test.toPandas()
uploadToFileIO(df_predictions, df_flights_train, df_flights_test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df_flights_test.csv': 'https://file.io/7hLGqq',
  'df_flights_train.csv': 'https://file.io/ICJOgb',
  'df_predictions.csv': 'https://file.io/JqLtv5'}}
"""
prefixToc = '3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# Read data from CSV file
predictions = spark.read.csv(prefix+'df_predictions.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')
flights_train = spark.read.csv(prefix+'df_flights_train.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')
flights_test = spark.read.csv(prefix+'df_flights_test.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

predictions=predictions.drop('_c0')
flights_train=flights_train.drop('_c0')
flights_test=flights_test.drop('_c0')
# Get number of records
print("The data predictions contain %d records." % predictions.count())
print("The data flights_train contain %d records." % flights_train.count())
print("The data flights_test contain %d records." % flights_test.count())

# View the first five records
predictions.show(5)
flights_train.show(5)
flights_test.show(5)


predictions = predictions.withColumn("delay", predictions.delay.cast('integer'))
flights_train = flights_train.withColumn("delay", flights_train.delay.cast('integer'))
flights_test = flights_test.withColumn("delay", flights_test.delay.cast('integer'))

# Check column data types
predictions.dtypes


# In[93]:


def transformToVector(dataset):
    # Create a one-hot encoder
    dataset = dataset.drop('depart_dummy', 'org_dummy', 'features')

    # Create instances of one hot encoders
    onehot_depart = OneHotEncoderEstimator(inputCols=['depart_bucket'], outputCols=['depart_dummy'])
    onehot_org = OneHotEncoderEstimator(inputCols=['org_idx'], outputCols=['org_dummy'])

    # One-hot encode the bucketed departure times
    dataset = onehot_depart.fit(dataset).transform(dataset)
    # One-hot encode the org
    dataset = onehot_org.fit(dataset).transform(dataset)
    
    assembler = VectorAssembler(inputCols=['km','org_dummy', 'depart_dummy'], outputCol='features')
    dataset = assembler.transform(dataset)
    return dataset

predictions = transformToVector(predictions)
flights_train = transformToVector(flights_train)
flights_test = transformToVector(flights_test)


# In[94]:


predictions.show(2)


# In[95]:


from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a regression object and train on training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Create predictions for the testing data
predictions2 = regression.transform(flights_test)


# In[96]:


predictions2.show(2)


# ### code

# In[97]:


# Find the RMSE on testing data
from pyspark.ml.evaluation import RegressionEvaluator
RegressionEvaluator(labelCol='duration').evaluate(predictions)

# Average minutes on ground at OGG for flights departing between 21:00 and 24:00
avg_eve_ogg = regression.intercept
print(avg_eve_ogg)

# Average minutes on ground at OGG for flights departing between 00:00 and 03:00
avg_night_ogg = regression.intercept + regression.coefficients[8]
print(avg_night_ogg)

# Average minutes on ground at JFK for flights departing between 00:00 and 03:00
avg_night_jfk = regression.intercept + regression.coefficients[3] + regression.coefficients[8]
print(avg_night_jfk)


# # Regularization

# ## Flight duration model: More features!
# Let's add more features to our model. This will not necessarily result in a better model. Adding some features might improve the model. Adding other features might make it worse.
# 
# More features will always make the model more complicated and difficult to interpret.
# 
# These are the features you'll include in the next model:
# 
# - km
# - org (origin airport, one-hot encoded, 8 levels)
# - depart (departure time, binned in 3 hour intervals, one-hot encoded, 8 levels)
# - dow (departure day of week, one-hot encoded, 7 levels) and
# - mon (departure month, one-hot encoded, 12 levels).
# 
# These have been assembled into the features column, which is a sparse representation of 32 columns (remember one-hot encoding produces a number of columns which is one fewer than the number of levels).
# 
# The data are available as flights, randomly split into flights_train and flights_test. The object predictions is also available.

# ### init

# In[98]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
df_predictions = predictions.toPandas()
df_flights_train = flights_train.toPandas()
df_flights_test = flights_test.toPandas()
uploadToFileIO(df_predictions, df_flights_train, df_flights_test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df_flights_test.csv': 'https://file.io/z8Wz1a',
  'df_flights_train.csv': 'https://file.io/USXK5j',
  'df_predictions.csv': 'https://file.io/XzrTZy'}}
"""
prefixToc = '4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# Read data from CSV file
predictions = spark.read.csv(prefix+'df_predictions.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')
flights_train = spark.read.csv(prefix+'df_flights_train.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')
flights_test = spark.read.csv(prefix+'df_flights_test.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

predictions=predictions.drop('_c0')
flights_train=flights_train.drop('_c0')
flights_test=flights_test.drop('_c0')
# Get number of records
print("The data predictions contain %d records." % predictions.count())
print("The data flights_train contain %d records." % flights_train.count())
print("The data flights_test contain %d records." % flights_test.count())

# View the first five records
predictions.show(5)
flights_train.show(5)
flights_test.show(5)


predictions = predictions.withColumn("delay", predictions.delay.cast('integer'))
flights_train = flights_train.withColumn("delay", flights_train.delay.cast('integer'))
flights_test = flights_test.withColumn("delay", flights_test.delay.cast('integer'))

# Check column data types
predictions.dtypes


# In[104]:


def transformToVector(dataset):
    # Create a one-hot encoder
    dataset = dataset.drop('depart_dummy', 'org_dummy', 'features', 'dow_dummy', 'mon_dummy')

    # Create instances of one hot encoders
    onehot_depart = OneHotEncoderEstimator(inputCols=['depart_bucket'], outputCols=['depart_dummy'])
    onehot_org = OneHotEncoderEstimator(inputCols=['org_idx'], outputCols=['org_dummy'])
    onehot_dow = OneHotEncoderEstimator(inputCols=['dow'], outputCols=['dow_dummy'])
    onehot_mon = OneHotEncoderEstimator(inputCols=['mon'], outputCols=['mon_dummy'])

    # One-hot encode the bucketed departure times
    dataset = onehot_depart.fit(dataset).transform(dataset)
    # One-hot encode the org
    dataset = onehot_org.fit(dataset).transform(dataset)
    dataset = onehot_dow.fit(dataset).transform(dataset)
    dataset = onehot_mon.fit(dataset).transform(dataset)
    
    assembler = VectorAssembler(inputCols=['km','org_dummy', 'depart_dummy', 'dow_dummy', 'mon_dummy'], outputCol='features')
    dataset = assembler.transform(dataset)
    dataset = dataset.drop('dow_dummy', 'mon_dummy')
    return dataset

predictions = transformToVector(predictions)
flights_train = transformToVector(flights_train)
flights_test = transformToVector(flights_test)


# In[106]:


flights_test.select('features','duration').show(5, False)


# ### code

# In[107]:


from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Fit linear regression model to training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Make predictions on testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE on testing data
rmse = RegressionEvaluator(labelCol='duration').evaluate(predictions)
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)


# ## Flight duration model: Regularisation!
# In the previous exercise you added more predictors to the flight duration model. The model performed well on testing data, but with so many coefficients it was difficult to interpret.
# 
# In this exercise you'll use Lasso regression (regularized with a L1 penalty) to create a more parsimonious model. Many of the coefficients in the resulting model will be set to zero. This means that only a subset of the predictors actually contribute to the model. Despite the simpler model, it still produces a good RMSE on the testing data.
# 
# You'll use a specific value for the regularization strength. Later you'll learn how to find the best value using cross validation.
# 
# The data (same as previous exercise) are available as flights, randomly split into flights_train and flights_test.

# ### code

# In[110]:


from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Fit Lasso model (α = 1) to training data
regression = LinearRegression(labelCol='duration', regParam=1, elasticNetParam=1).fit(flights_train)

# Calculate the RMSE on testing data
rmse = RegressionEvaluator(labelCol='duration').evaluate(regression.transform(flights_test))
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)

# Number of zero coefficients
zero_coeff = sum([beta==0 for beta in regression.coefficients])
print("Number of ceofficients equal to 0:", zero_coeff)


# In[ ]:




