#!/usr/bin/env python
# coding: utf-8

# # Data Preparation
# 

# ## Removing columns and rows
# You previously loaded airline flight data from a CSV file. You're going to develop a model which will predict whether or not a given flight will be delayed.
# 
# In this exercise you need to trim those data down by:
# 
# - removing an uninformative column and
# - removing rows which do not have information about whether or not a flight was delayed.
# 
# Note:: You might find it useful to revise the slides from the lessons in the Slides panel next to the IPython Shell.

# ### init

# In[3]:


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


from pyspark.sql.types import StructType, StructField, IntegerType, StringType

prefix='data_from_datacamp/chapter1-Exercise3.1_'
flights = spark.read.csv(prefix+'flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])

# Load data from a delimited file
prefix='data_from_datacamp/chapter1-Exercise3.2_'
sms = spark.read.csv(prefix+'sms.csv', sep=';', header=False, schema=schema)


# ### code

# In[7]:


# Remove the 'flight' column
flights = flights.drop('flight')

# Number of records with missing 'delay' values
flights.filter('delay IS NULL').count()

# Remove records with missing 'delay' values
flights = flights.filter('delay IS NOT NULL')

# Remove records with missing values in any column and get the number of remaining rows
flights = flights.dropna()
print(flights.count())


# ## Column manipulation
# The Federal Aviation Administration (FAA) considers a flight to be "delayed" when it arrives 15 minutes or more after its scheduled time.
# 
# The next step of preparing the flight data has two parts:
# 
# - convert the units of distance, replacing the mile column with a kmcolumn; and
# - create a Boolean column indicating whether or not a flight was delayed.

# ### code

# In[16]:


# Import the required function
from pyspark.sql.functions import round

# Convert 'mile' to 'km' and drop 'mile' column
flights_km = flights.withColumn('km', round(flights.mile * 1.60934, 0))                     .drop('flights.mile')

# Create 'label' column indicating whether flight delayed (1) or not (0)
flights_km = flights_km.withColumn('label', (flights_km.delay >= 15).cast('integer'))

# Check first five records
flights_km.show(5)


# ## Categorical columns
# In the flights data there are two columns, carrier and org, which hold categorical data. You need to transform those columns into indexed numerical values.

# ### code

# In[21]:


from pyspark.ml.feature import StringIndexer

# Create an indexer
indexer = StringIndexer(inputCol='carrier', outputCol='carrier_idx')

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights_km)

# Indexer creates a new column with numeric index values
flights_indexed = indexer_model.transform(flights_km)

# Repeat the process for the other categorical feature
flights_indexed = StringIndexer(inputCol='org', outputCol='org_idx').fit(flights_indexed).transform(flights_indexed)


# ## Assembling columns
# The final stage of data preparation is to consolidate all of the predictor columns into a single column.
# 
# At present our data has the following predictor columns:
# 
# - mon, dom and dow
# - carrier_idx (derived from carrier)
# - org_idx (derived from org)
# - km
# - depart
# - duration

# ### code

# In[22]:


# Import the necessary class
from pyspark.ml.feature import VectorAssembler

# Create an assembler object
assembler = VectorAssembler(inputCols=[
    'mon', 'dom', 'dow', 'carrier_idx', 'org_idx', 'km', 'depart', 'duration'
], outputCol='features')

# Consolidate predictor columns
flights_assembled = assembler.transform(flights_indexed)

# Check the resulting column
flights_assembled.select('features', 'delay').show(5, truncate=False)


# # Decision Tree
# 

# ## Train/test split
# To objectively assess a Machine Learning model you need to be able to test it on an independent set of data. You can't use the same data that you used to train the model: of course the model will perform (relatively) well on those data!
# 
# You will split the data into two components:
# 
# training data (used to train the model) and
# testing data (used to test the model).

# ### code

# In[28]:


# Split into training and testing sets in a 80:20 ratio
flights_train, flights_test = flights_assembled.randomSplit([0.8, 0.2], seed=17)

# Check that training set has around 80% of records
training_ratio = flights_train.count() / flights.count()
print(training_ratio)


# ## Build a Decision Tree
# Now that you've split the flights data into training and testing sets, you can use the training set to fit a Decision Tree model.
# 
# The data are available as flights_train and flights_test.
# 
# NOTE: It will take a few seconds for the model to train... please be patient!

# ### code

# In[30]:


# Import the Decision Tree Classifier class
from pyspark.ml.classification import DecisionTreeClassifier

# Create a classifier object and fit to the training data
tree = DecisionTreeClassifier()
tree_model = tree.fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
prediction = tree_model.transform(flights_test)
prediction.select('label', 'prediction', 'probability').show(5, False)


# ## Evaluate the Decision Tree
# You can assess the quality of your model by evaluating how well it performs on the testing data. Because the model was not trained on these data, this represents an objective assessment of the model.
# 
# A confusion matrix gives a useful breakdown of predictions versus known values. It has four cells which represent the counts of:
# 
# - True Negatives (TN) — model predicts negative outcome & known outcome is negative
# - True Positives (TP) — model predicts positive outcome & known outcome is positive
# - False Negatives (FN) — model predicts negative outcome but known outcome is positive
# - False Positives (FP) — model predicts positive outcome but known outcome is negative.

# ### code

# In[31]:


# Create a confusion matrix
prediction.groupBy('label', 'prediction').count().show()

# Calculate the elements of the confusion matrix
TN = prediction.filter('prediction = 0 AND label = 0').count()
TP = prediction.filter('prediction = 1 AND label = 1').count()
FN = prediction.filter('prediction = 0 AND label = 1').count()
FP = prediction.filter('prediction = 1 AND label = 0').count()

# Accuracy measures the proportion of correct predictions
accuracy = (TN+TP)/(TN+TP+FP+FN)
print(accuracy)


# # Logistic Regression
# 

# ## Build a Logistic Regression model
# You've already built a Decision Tree model using the flights data. Now you're going to create a Logistic Regression model on the same data.
# 
# The objective is to predict whether a flight is likely to be delayed by at least 15 minutes (label 1) or not (label 0).
# 
# Although you have a variety of predictors at your disposal, you'll only use the mon, depart and duration columns for the moment. These are numerical features which can immediately be used for a Logistic Regression model. You'll need to do a little more work before you can include categorical features. Stay tuned!
# 
# The data have been split into training and testing sets and are available as flights_train and flights_test.

# ### code

# In[32]:


# Import the logistic regression class
from pyspark.ml.classification import LogisticRegression

# Create a classifier object and train on training data
logistic = LogisticRegression().fit(flights_train)

# Create predictions for the testing data and show confusion matrix
prediction = logistic.transform(flights_test)
prediction.groupBy('label', 'prediction').count().show()


# ## Evaluate the Logistic Regression model
# Accuracy is generally not a very reliable metric because it can be biased by the most common target class.
# 
# There are two other useful metrics:
# 
# - precision and
# - recall.
# Check the slides for this lesson to get the relevant expressions.
# 
# Precision is the proportion of positive predictions which are correct. For all flights which are predicted to be delayed, what proportion is actually delayed?
# 
# Recall is the proportion of positives outcomes which are correctly predicted. For all delayed flights, what proportion is correctly predicted by the model?
# 
# The precision and recall are generally formulated in terms of the positive target class. But it's also possible to calculate weighted versions of these metrics which look at both target classes.
# 
# The components of the confusion matrix are available as TN, TP, FN and FP, as well as the object prediction.

# ### code

# In[ ]:


TN, TP, FN,FP = (288, 277, 195, 201)


# In[34]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Calculate precision and recall
precision = prediction.filter('prediction = 1 AND label = 1').count() / prediction.filter('prediction = 1').count()
recall = prediction.filter('prediction = 1 AND label = 1').count() / prediction.filter('prediction = label').count()
print('precision = {:.2f}\nrecall    = {:.2f}'.format(precision, recall))

# Find weighted precision
multi_evaluator = MulticlassClassificationEvaluator()
weighted_precision = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedPrecision"})

# Find AUC
binary_evaluator = BinaryClassificationEvaluator()
auc = binary_evaluator.evaluate(prediction, {multi_evaluator.metricName: "areaUnderROC"})


# # Turning Text into Tables
# 

# ## Punctuation, numbers and tokens
# At the end of the previous chapter you loaded a dataset of SMS messages which had been labeled as either "spam" (label 1) or "ham" (label 0). You're now going to use those data to build a classifier model.
# 
# But first you'll need to prepare the SMS messages as follows:
# 
# - remove punctuation and numbers
# - tokenize (split into individual words)
# - remove stop words
# - apply the hashing trick
# - convert to TF-IDF representation.
# 
# In this exercise you'll remove punctuation and numbers, then tokenize the messages.
# 
# The SMS data are available as sms.

# ### init

# In[36]:


from downloadfromFileIO import saveFromFileIO

#from datacamp
"""
df_sms = sms.toPandas()
uploadToFileIO(df_sms)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df_sms.csv': 'https://file.io/uPLoQZ'}}
"""
prefixToc='4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
df_sms = pd.read_csv(prefix+'df_sms.csv',index_col=0)


# In[37]:


sms = spark.createDataFrame(df_sms)


# In[38]:


sms.show()


# ### code

# In[40]:


# Import the necessary functions
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer

# Remove punctuation (REGEX provided) and numbers
wrangled = sms.withColumn('text', regexp_replace(sms.text, '[_():;,.!?\\-]', ' '))
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, '[0-9]', ' '))

# Merge multiple spaces
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, ' +', ' '))

# Split the text into words
wrangled = Tokenizer(inputCol='text', outputCol='words').transform(wrangled)

wrangled.show(4, truncate=False)


# ## Stop words and hashing
# The next steps will be to remove stop words and then apply the hashing trick, converting the results into a TF-IDF.
# 
# A quick reminder about these concepts:
# 
# The hashing trick provides a fast and space-efficient way to map a very large (possibly infinite) set of items (in this case, all words contained in the SMS messages) onto a smaller, finite number of values.
# The TF-IDF matrix reflects how important a word is to each document. It takes into account both the frequency of the word within each document but also the frequency of the word across all of the documents in the collection.
# The tokenized SMS data are stored in sms in a column named words. You've cleaned up the handling of spaces in the data so that the tokenized text is neater.

# ### code

# In[43]:


from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF

# Remove stop words.
wrangled = StopWordsRemover(inputCol='words', outputCol='terms')      .transform(wrangled)

# Apply the hashing trick
wrangled = HashingTF(inputCol='terms', outputCol='hash', numFeatures=1024)      .transform(wrangled)

# Convert hashed symbols to TF-IDF
tf_idf = IDF(inputCol='hash', outputCol='features')      .fit(wrangled).transform(wrangled)
      
tf_idf.select('terms', 'features').show(4, truncate=False)


# ## Training a spam classifier
# The SMS data have now been prepared for building a classifier. Specifically, this is what you have done:
# 
# - removed numbers and punctuation
# - split the messages into words (or "tokens")
# - removed stop words
# - applied the hashing trick and
# - converted to a TF-IDF representation.
# 
# Next you'll need to split the TF-IDF data into training and testing sets. Then you'll use the training data to fit a Logistic Regression model and finally evaluate the performance of that model on the testing data.
# 
# The data are stored in sms and LogisticRegression has been imported for you.

# ### init
# 

# In[44]:


from downloadfromFileIO import saveFromFileIO

#from datacamp
"""
df_sms = sms.toPandas()
uploadToFileIO(df_sms)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df_sms.csv': 'https://file.io/wixV72'}}
"""
prefixToc='4.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
df_sms = pd.read_csv(prefix+'df_sms.csv',index_col=0)
sms = spark.createDataFrame(df_sms)


# In[45]:


sms.show()


# In[46]:


# Import the logistic regression class
from pyspark.ml.classification import LogisticRegression


# ### code

# In[48]:


# Split the data into training and testing sets
sms_train, sms_test = sms.randomSplit([0.8, 0.2], seed=13)

# Fit a Logistic Regression model to the training data
logistic = LogisticRegression(regParam=0.2).fit(sms_train)

# Make predictions on the testing data
prediction = logistic.transform(sms_test)

# Create a confusion matrix, comparing predictions to known labels
prediction.groupBy('label', 'prediction').count().show()


# In[ ]:




