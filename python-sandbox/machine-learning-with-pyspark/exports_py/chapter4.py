#!/usr/bin/env python
# coding: utf-8

# # Pipeline

# In[2]:


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


# ## Flight duration model: Pipeline stages
# You're going to create the stages for the flights duration model pipeline. You will use these in the next exercise to build a pipeline and to create a regression model.

# ### code

# In[7]:


from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import LinearRegression


# In[9]:


# Convert categorical strings to index values
indexer = StringIndexer(inputCol='org', outputCol='org_idx')

# One-hot encode index values
onehot = OneHotEncoderEstimator(
    inputCols=['org_idx', 'dow'],
    outputCols=['org_dummy', 'dow_dummy']
)

# Assemble predictors into a single column
assembler = VectorAssembler(inputCols=['km', 'org_dummy', 'dow_dummy'], outputCol='features')

# A linear regression object
regression = LinearRegression(labelCol='duration')


# ## Flight duration model: Pipeline model
# You're now ready to put those stages together in a pipeline.
# 
# You'll construct the pipeline and then train the pipeline on the training data. This will apply each of the individual stages in the pipeline to the training data in turn. None of the stages will be exposed to the testing data at all: there will be no leakage!
# 
# Once the entire pipeline has been trained it will then be used to make predictions on the testing data.
# 
# The data are available as flights, which has been randomly split into flights_train and flights_test.

# ### init

# In[10]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
df_flights_train = flights_train.toPandas()
df_flights_test = flights_test.toPandas()
uploadToFileIO(df_flights_train, df_flights_test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df_flights_test.csv': 'https://file.io/qvGJBW',
  'df_flights_train.csv': 'https://file.io/BPoVCe'}}
"""
prefixToc = '1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# Read data from CSV file
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

flights_train=flights_train.drop('_c0')
flights_test=flights_test.drop('_c0')
# Get number of records
print("The data flights_train contain %d records." % flights_train.count())
print("The data flights_test contain %d records." % flights_test.count())

# View the first five records
flights_train.show(5)
flights_test.show(5)


flights_train = flights_train.withColumn("delay", flights_train.delay.cast('integer'))
flights_test = flights_test.withColumn("delay", flights_test.delay.cast('integer'))

# Check column data types
flights_train.dtypes


# ### code

# In[11]:


# Import class for creating a pipeline
from pyspark.ml import Pipeline

# Construct a pipeline
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])

# Train the pipeline on the training data
pipeline = pipeline.fit(flights_train)

# Make predictions on the testing data
predictions = pipeline.transform(flights_test)


# ## SMS spam pipeline
# You haven't looked at the SMS data for quite a while. Last time we did the following:
# 
# - split the text into tokens
# - removed stop words
# - applied the hashing trick
# -converted the data from counts to IDF and
# - trained a linear regression model.
# 
# Each of these steps was done independently. This seems like a great application for a pipeline!

# ### code

# In[14]:


from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression


# Break text into tokens at non-word characters
tokenizer = Tokenizer(inputCol='text', outputCol='words')

# Remove stop words
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='terms')

# Apply the hashing trick and transform to TF-IDF
hasher = HashingTF(inputCol=remover.getOutputCol(), outputCol="hash")
idf = IDF(inputCol=hasher.getOutputCol(), outputCol="features")

# Create a logistic regression object and add everything to a pipeline
logistic = LogisticRegression()
pipeline = Pipeline(stages=[tokenizer, remover, hasher, idf, logistic])


# # Cross-Validation
# 

# ## Cross validating simple flight duration model
# You've already built a few models for predicting flight duration and evaluated them with a simple train/test split. However, cross-validation provides a much better way to evaluate model performance.
# 
# In this exercise you're going to train a simple model for flight duration using cross-validation. Travel time is usually strongly correlated with distance, so using the km column alone should give a decent model.
# 
# The data have been randomly split into flights_train and flights_test.
# 
# The following classes have already been imported: LinearRegression, RegressionEvaluator, ParamGridBuilder and CrossValidator.

# ### init

# In[15]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
df_flights_train = flights_train.toPandas()
df_flights_test = flights_test.toPandas()
uploadToFileIO(df_flights_train, df_flights_test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df_flights_test.csv': 'https://file.io/EPILsY',
  'df_flights_train.csv': 'https://file.io/WdzcXN'}}
"""
prefixToc = '2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# Read data from CSV file
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

flights_train=flights_train.drop('_c0')
flights_test=flights_test.drop('_c0')
# Get number of records
print("The data flights_train contain %d records." % flights_train.count())
print("The data flights_test contain %d records." % flights_test.count())

# View the first five records
flights_train.show(5)
flights_test.show(5)


flights_train = flights_train.withColumn("delay", flights_train.delay.cast('integer'))
flights_test = flights_test.withColumn("delay", flights_test.delay.cast('integer'))

# Check column data types
flights_train.dtypes


# In[16]:


def transformToVector(dataset):
    # Create a one-hot encoder
    dataset = dataset.drop('features')
    assembler = VectorAssembler(inputCols=['km'], outputCol='features')
    dataset = assembler.transform(dataset)
    return dataset

flights_train = transformToVector(flights_train)
flights_test = transformToVector(flights_test)


# In[20]:


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# ### code

# In[22]:


# Create an empty parameter grid
params = ParamGridBuilder().build()

# Create objects for building and evaluating a regression model
regression = LinearRegression(labelCol='duration')
evaluator = RegressionEvaluator(labelCol='duration')

# Create a cross validator
cv = CrossValidator(estimator=regression, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)

# Train and test model on multiple folds of the training data
cv = cv.fit(flights_train)

# NOTE: Since cross-valdiation builds multiple models, the fit() method can take a little while to complete.


# ## Cross validating flight duration model pipeline
# The cross-validated model that you just built was simple, using km alone to predict duration.
# 
# Another important predictor of flight duration is the origin airport. Flights generally take longer to get into the air from busy airports. Let's see if adding this predictor improves the model!
# 
# In this exercise you'll add the org field to the model. However, since org is categorical, there's more work to be done before it can be included: it must first be transformed to an index and then one-hot encoded before being assembled with km and used to build the regression model. We'll wrap these operations up in a pipeline.
# 
# The following objects have already been created:
# 
# - params — an empty parameter grid
# - evaluator — a regression evaluator
# - regression — a LinearRegression object with labelCol='duration'.
# 
# All of the required classes have already been imported.

# ### code

# In[27]:


# Create an indexer for the org field
indexer = StringIndexer(inputCol='org', outputCol='org_idx')

# Create an one-hot encoder for the indexed org field
onehot = OneHotEncoderEstimator(inputCols=['org_idx'], outputCols=['org_dummy'])

# Assemble the km and one-hot encoded fields
assembler = VectorAssembler(inputCols=['km', 'org_dummy'], outputCol='features')

# Create a pipeline and cross-validator.
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])
cv = CrossValidator(estimator=pipeline,
          estimatorParamMaps=params,
          evaluator=evaluator)


# # Grid Search
# 

# ## Optimizing flights linear regression
# Up until now you've been using the default parameters when building your models. In this exercise you'll use cross validation to choose an optimal (or close to optimal) set of model parameters.
# 
# The following have already been created:
# 
# - regression — a LinearRegression object
# - pipeline — a pipeline with string indexer, one-hot encoder, vector assembler and linear regression and
# - evaluator — a RegressionEvaluator object.
# 

# ### code

# In[28]:


from pyspark.ml.tuning import ParamGridBuilder


# In[30]:


# Create parameter grid
params = ParamGridBuilder()

# Add grids for two parameters
params = params.addGrid(regression.regParam, [0.01, 0.1, 1.0, 10.0])                .addGrid(regression.elasticNetParam, [0.0, 0.5, 1.0])

# Build the parameter grid
params = params.build()
print('Number of models to be tested: ', len(params))

# Create cross-validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)


# ## Dissecting the best flight duration model
# You just set up a CrossValidator to find good parameters for the linear regression model predicting flight duration.
# 
# Now you're going to take a closer look at the resulting model, split out the stages and use it to make predictions on the testing data.
# 
# The following have already been created:
# 
# - cv — a trained CrossValidatorModel object and
# - evaluator — a RegressionEvaluator object.
# 
# The flights data have been randomly split into flights_train and flights_test.

# ### init

# In[38]:


flights_train = flights_train.drop('features')
flights_test = flights_test.drop('features')
cv.fit(flights_train)


# ### code

# In[39]:


# Get the best model from cross validation
best_model = cv.bestModel

# Look at the stages in the best model
print(best_model.stages)

# Get the parameters for the LinearRegression object in the best model
best_model.stages[3].extractParamMap()

# Generate predictions on testing data using the best model then calculate RMSE
predictions = best_model.transform(flights_test)
evaluator.evaluate(predictions)


# ## SMS spam optimised
# The pipeline you built earlier for the SMS spam model used the default parameters for all of the elements in the pipeline. It's very unlikely that these parameters will give a particularly good model though.
# 
# In this exercise you'll set up a parameter grid which can be used with cross validation to choose a good set of parameters for the SMS spam classifier.
# 
# The following are already defined:
# 
# - hasher — a HashingTF object and
# - logistic — a LogisticRegression object.

# ### init

# In[40]:


hasher = HashingTF(inputCol='terms', outputCol="hash")
# Create a logistic regression object and add everything to a pipeline
logistic = LogisticRegression()


# ### code

# In[41]:


# Create parameter grid
params = ParamGridBuilder()

# Add grid for hashing trick parameters
params = params.addGrid(hasher.numFeatures, [1024, 4096, 16384])                .addGrid(hasher.binary, [True, False])

# Add grid for logistic regression parameters
params = params.addGrid(logistic.regParam, [0.01, 0.1, 1.0, 10.0])                .addGrid(logistic.elasticNetParam, [0.0, 0.5, 1.0])

# Build parameter grid
params = params.build()


# # Ensemble

# ## Delayed flights with Gradient-Boosted Trees
# You've previously built a classifier for flights likely to be delayed using a Decision Tree. In this exercise you'll compare a Decision Tree model to a Gradient-Boosted Trees model.
# 
# The flights data have been randomly split into flights_train and flights_test.

# ### init

# In[42]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
df_flights_train = flights_train.toPandas()
df_flights_test = flights_test.toPandas()
uploadToFileIO(df_flights_train, df_flights_test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df_flights_test.csv': 'https://file.io/PepwHu',
  'df_flights_train.csv': 'https://file.io/69v2YN'}}
"""
prefixToc = '4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# Read data from CSV file
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

flights_train=flights_train.drop('_c0')
flights_test=flights_test.drop('_c0')
# Get number of records
print("The data flights_train contain %d records." % flights_train.count())
print("The data flights_test contain %d records." % flights_test.count())

# View the first five records
flights_train.show(2)
flights_test.show(2)


# Check column data types
flights_train.dtypes


# In[43]:


def transformToVector(dataset):
    # Create a one-hot encoder
    dataset = dataset.drop('features')
    assembler = VectorAssembler(inputCols=['mon', 'depart','duration'], outputCol='features')
    dataset = assembler.transform(dataset)
    return dataset

flights_train = transformToVector(flights_train)
flights_test = transformToVector(flights_test)


# In[44]:


flights_train.dtypes


# In[45]:


flights_train.show(2)


# ### code

# In[49]:


# Import the classes required
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create model objects and train on training data
tree = DecisionTreeClassifier().fit(flights_train)
gbt = GBTClassifier().fit(flights_train)

# Compare AUC on testing data
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(tree.transform(flights_test))
evaluator.evaluate(gbt.transform(flights_test))

# Find the number of trees and the relative importance of features
print(len(gbt.trees))
print(gbt.featureImportances)


# ## Delayed flights with a Random Forest
# In this exercise you'll bring together cross validation and ensemble methods. You'll be training a Random Forest classifier to predict delayed flights, using cross validation to choose the best values for model parameters.
# 
# You'll find good values for the following parameters:
# 
# - featureSubsetStrategy — the number of features to consider for splitting at each node and
# - maxDepth — the maximum number of splits along any branch.
# 
# Unfortunately building this model takes too long, so we won't be running the .fit() method on the pipeline.

# ### code

# In[50]:


from pyspark.ml.classification import RandomForestClassifier


# In[51]:


# Create a random forest classifier
forest = RandomForestClassifier()

# Create a parameter grid
params = ParamGridBuilder()             .addGrid(forest.featureSubsetStrategy, ['all', 'onethird', 'sqrt', 'log2'])             .addGrid(forest.maxDepth, [2, 5, 10])             .build()

# Create a binary classification evaluator
evaluator = BinaryClassificationEvaluator()

# Create a cross-validator
cv = CrossValidator(evaluator=evaluator, estimator=forest, estimatorParamMaps=params, numFolds=5)


# ## Evaluating Random Forest
# In this final exercise you'll be evaluating the results of cross-validation on a Random Forest model.
# 
# The following have already been created:
# 
# - cv - a cross-validator which has already been fit to the training data
# - evaluator — a BinaryClassificationEvaluator object and
# - flights_test — the testing data.

# ### code

# In[52]:


# Average AUC for each parameter combination in grid
avg_auc = cv.avgMetrics

# Average AUC for the best model
best_model_auc =  max(avg_auc)

# What's the optimal parameter value?
opt_max_depth = cv.bestModel.explainParam('maxDepth')
opt_feat_substrat = cv.bestModel.explainParam('featureSubsetStrategy')

# AUC for best model on testing data
best_auc = evaluator.evaluate(cv.transform(flights_test))


# In[ ]:




