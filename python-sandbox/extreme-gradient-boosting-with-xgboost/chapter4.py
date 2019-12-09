#!/usr/bin/env python
# coding: utf-8

# # Review of pipelines using sklearn
# 

# ![image.png](attachment:image.png)

# ## Exploratory data analysis
# Before diving into the nitty gritty of pipelines and preprocessing, let's do some exploratory analysis of the original, unprocessed Ames housing dataset. When you worked with this data in previous chapters, we preprocessed it for you so you could focus on the core XGBoost concepts. In this chapter, you'll do the preprocessing yourself!
# 
# A smaller version of this original, unprocessed dataset has been pre-loaded into a pandas DataFrame called df. Your task is to explore df in the Shell and pick the option that is incorrect. The larger purpose of this exercise is to understand the kinds of transformations you will need to perform in order to be able to use XGBoost.

# ### init: 1 dataframe

# In[1]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(df)
tobedownloaded="{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/OlAR3N'}}"
prefix='data_from_datacamp/Chap4-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[2]:


import pandas as pd
df=pd.read_csv(prefix+'df.csv',index_col=0)


# ### code

# In[3]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# ## Encoding categorical columns I: LabelEncoder
# Now that you've seen what will need to be done to get the housing data ready for XGBoost, let's go through the process step-by-step.
# 
# First, you will need to fill in missing values - as you saw previously, the column LotFrontage has many missing values. Then, you will need to encode any categorical columns in the dataset using one-hot encoding so that they are encoded numerically. You can watch this video from Supervised Learning with scikit-learn for a refresher on the idea.
# 
# The data has five categorical columns: MSZoning, PavedDrive, Neighborhood, BldgType, and HouseStyle. Scikit-learn has a LabelEncoder function that converts the values in each categorical column into integers. You'll practice using this here.µ

# ### code

# - Import LabelEncoder from sklearn.preprocessing.
# - Fill in missing values in the LotFrontage column with 0 using .fillna().
# - Create a boolean mask for categorical columns. You can do this by checking for whether df.dtypes equals object.
# - Create a LabelEncoder object. You can do this in the same way you instantiate any scikit-learn estimator.
# - Encode all of the categorical columns into integers using LabelEncoder(). To do this, use the .fit_transform() method of le in the provided lambda function.

# In[11]:


# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == object)

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())


# ## Encoding categorical columns II: OneHotEncoder
# Okay - so you have your categorical columns encoded numerically. Can you now move onto using pipelines and XGBoost? Not yet! In the categorical columns of this dataset, there is no natural ordering between the entries. As an example: Using LabelEncoder, the CollgCr Neighborhood was encoded as 5, while the Veenker Neighborhood was encoded as 24, and Crawfor as 6. Is Veenker "greater" than Crawfor and CollgCr? No - and allowing the model to assume this natural ordering may result in poor performance.
# 
# As a result, there is another step needed: You have to apply a one-hot encoding to create binary, or "dummy" variables. You can do this using scikit-learn's OneHotEncoder.

# ### code

# - Import OneHotEncoder from sklearn.preprocessing.
# - Instantiate a OneHotEncoder object called ohe. Specify the keyword arguments categorical_features=categorical_mask and sparse=False.
# - Using its .fit_transform() method, apply the OneHotEncoder to df and save the result as df_encoded. The output will be a NumPy array.
# - Print the first 5 rows of df_encoded, and then the shape of df as well as df_encoded to compare the difference.

# In[12]:


# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categorical_features=categorical_mask, sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape of the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)


# ## Encoding categorical columns III: DictVectorizer
# Alright, one final trick before you dive into pipelines. The two step process you just went through - LabelEncoder followed by OneHotEncoder - can be simplified by using a DictVectorizer.
# 
# Using a DictVectorizer on a DataFrame that has been converted to a dictionary allows you to get label encoding as well as one-hot encoding in one go.
# 
# Your task is to work through this strategy in this exercise!

# ### code

# - Import DictVectorizer from sklearn.feature_extraction.
# - Convert df into a dictionary called df_dict using its .to_dict() method with "records" as the argument.
# - Instantiate a DictVectorizer object called dv with the keyword argument sparse=False.
# - Apply the DictVectorizer on df_dict by using its .fit_transform() method.
# - Hit 'Submit Answer' to print the resulting first five rows and the vocabulary.

# In[14]:


# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary: df_dict
df_dict = df.to_dict('records')

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5,:])

# Print the vocabulary
print(dv.vocabulary_)


# ## Preprocessing within a pipeline
# Now that you've seen what steps need to be taken individually to properly process the Ames housing data, let's use the much cleaner and more succinct DictVectorizer approach and put it alongside an XGBoostRegressor inside of a scikit-learn pipeline.

# ### init: 1 dataframe, 1 serie

# In[17]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X, y)
tobedownloaded="{pandas.core.frame.DataFrame: {'X.csv': 'https://file.io/kPI9z5'}, pandas.core.series.Series: {'y.csv': 'https://file.io/dh9M1U'}}"
prefix='data_from_datacamp/Chap4-Exercise1.5_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[18]:


import pandas as pd
X=pd.read_csv(prefix+'X.csv',index_col=0)
y=pd.read_csv(prefix+'y.csv',index_col=0, header=None,squeeze=True)


# ### code

# - Import DictVectorizer from sklearn.feature_extraction and Pipeline from sklearn.pipeline.
# - Fill in any missing values in the LotFrontage column of X with 0.
# - Complete the steps of the pipeline with DictVectorizer(sparse=False) for "ohe_onestep" and xgb.XGBRegressor() for "xgb_model".
# - Create the pipeline using Pipeline() and steps.
# - Fit the Pipeline. Don't forget to convert X into a format that DictVectorizer understands by calling the to_dict("records") method on X.

# In[20]:


import xgboost as xgb


# In[23]:


# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps=steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict("records"), y)


# # Incorporating XGBoost into pipelines
# 

# ## Cross-validating your XGBoost model
# In this exercise, you'll go one step further by using the pipeline you've created to preprocess and cross-validate your model.

# ### code

# - Create a pipeline called xgb_pipeline using steps.
# - Perform 10-fold cross-validation using cross_val_score(). You'll have to pass in the pipeline, X (as a dictionary, using .to_dict("records")), y, the number of folds you want to use, and scoring ("neg_mean_squared_error").
# - Print the 10-fold RMSE.
# 

# In[27]:


# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps=steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict('records'), y, scoring='neg_mean_squared_error', cv=10)

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))


# ## Kidney disease case study I: Categorical Imputer
# You'll now continue your exploration of using pipelines with a dataset that requires significantly more wrangling. The chronic kidney disease dataset contains both categorical and numeric features, but contains lots of missing values. The goal here is to predict who has chronic kidney disease given various blood indicators as features.
# 
# As Sergey mentioned in the video, you'll be introduced to a new library, sklearn_pandas, that allows you to chain many more processing steps inside of a pipeline than are currently supported in scikit-learn. Specifically, you'll be able to impute missing categorical values directly using the Categorical_Imputer() class in sklearn_pandas, and the DataFrameMapper() class to apply any arbitrary sklearn-compatible transformer on DataFrame columns, where the resulting output can be either a NumPy array or DataFrame.
# 
# We've also created a transformer called a Dictifier that encapsulates converting a DataFrame using .to_dict("records") without you having to do it explicitly (and so that it works in a pipeline). Finally, we've also provided the list of feature names in kidney_feature_names, the target name in kidney_target_name, the features in X, and the target in y.
# 
# In this exercise, your task is to apply the CategoricalImputer to impute all of the categorical columns in the dataset. You can refer to how the numeric imputation mapper was created as a template. Notice the keyword arguments input_df=True and df_out=True? This is so that you can work with DataFrames instead of arrays. By default, the transformers are passed a numpy array of the selected columns as input, and as a result, the output of the DataFrame mapper is also an array. Scikit-learn transformers have historically been designed to work with numpy arrays, not pandas DataFrames, even though their basic indexing interfaces are similar.

# ### init: 1 dataframe, 1 array, sklearn_pandas

# In[28]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X, y)
tobedownloaded="{pandas.core.frame.DataFrame: {'X.csv': 'https://file.io/LWNS8F'}, numpy.ndarray: {'y.csv': 'https://file.io/rmeb12'}}"
prefix='data_from_datacamp/Chap4-Exercise2.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[30]:


import pandas as pd
X=pd.read_csv(prefix+'X.csv',index_col=0)
from uploadfromdatacamp import loadNDArrayFromCsv
y=loadNDArrayFromCsv(prefix+'y.csv')


# ### code

# - Apply the categorical imputer using DataFrameMapper() and CategoricalImputer(). CategoricalImputer() does not need any arguments to be passed in. The columns are contained in categorical_columns. Be sure to specify input_df=True and df_out=True, and use category_feature as your iterator variable in the list comprehension.

# In[36]:


# Import necessary modules
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing import Imputer

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
                                            [([numeric_feature], Imputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                            input_df=True,
                                            df_out=True
                                           )

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
                                                [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
                                                input_df=True,
                                                df_out=True
                                               )


# ## Kidney disease case study II: Feature Union
# Having separately imputed numeric as well as categorical columns, your task is now to use scikit-learn's FeatureUnion to concatenate their results, which are contained in two separate transformer objects - numeric_imputation_mapper, and categorical_imputation_mapper, respectively.
# 
# You may have already encountered FeatureUnion in Machine Learning with the Experts: School Budgets. Just like with pipelines, you have to pass it a list of (string, transformer) tuples, where the first half of each tuple is the name of the transformer.

# ### code

# - Import FeatureUnion from sklearn.pipeline.
# - Combine the results of numeric_imputation_mapper and categorical_imputation_mapper using FeatureUnion(), with the names "num_mapper" and "cat_mapper" respectively.

# In[37]:


# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
                                          ("num_mapper", numeric_imputation_mapper),
                                          ("cat_mapper", categorical_imputation_mapper)
                                         ])


# ## Kidney disease case study III: Full pipeline
# It's time to piece together all of the transforms along with an XGBClassifier to build the full pipeline!
# 
# Besides the numeric_categorical_union that you created in the previous exercise, there are two other transforms needed: the Dictifier() transform which we created for you, and the DictVectorizer().
# 
# After creating the pipeline, your task is to cross-validate it to see how well it performs.

# ### code

# - Create the pipeline using the numeric_categorical_union, Dictifier(), and DictVectorizer(sort=False) transforms, and xgb.XGBClassifier() estimator with max_depth=3. Name the transforms "featureunion", "dictifier" "vectorizer", and the estimator "clf".
# - Perform 3-fold cross-validation on the pipeline using cross_val_score(). Pass it the pipeline, pipeline, the features, kidney_data, the outcomes, y. Also set scoring to "roc_auc" and cv to 3.
# 

# In[38]:


# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier())
                    ])

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, kidney_data, y , scoring="roc_auc", cv=3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))


# # Tuning XGBoost hyperparameters
# 

# ![image.png](attachment:image.png)

# ## Bringing it all together
# Alright, it's time to bring together everything you've learned so far! In this final exercise of the course, you will combine your work from the previous exercises into one end-to-end XGBoost pipeline to really cement your understanding of preprocessing and pipelines in XGBoost.
# 
# Your work from the previous 3 exercises, where you preprocessed the data and set up your pipeline, has been pre-loaded. Your job is to perform a randomized search and identify the best hyperparameters.

# ### init

# - Set up the parameter grid to tune 'clf__learning_rate' (from 0.05 to 1 in increments of 0.05), 'clf__max_depth' (from 3 to 10 in increments of 1), and 'clf__n_estimators' (from 50 to 200 in increments of 50).
# - Using your pipeline as the estimator, perform 2-fold RandomizedSearchCV with an n_iter of 2. Use "roc_auc" as the metric, and set verbose to 1 so the output is more detailed. Store the result in randomized_roc_auc.
# - Fit randomized_roc_auc to X and y.
# - Compute the best score and best estimator of randomized_roc_auc.

# In[39]:


from sklearn.model_selection import RandomizedSearchCV


# In[40]:


# Create the parameter grid
gbm_param_grid = {
    'clf__learning_rate': np.arange(0.05, 1, 0.05),
    'clf__max_depth': np.arange(3, 10, 1),
    'clf__n_estimators': np.arange(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(param_distributions=gbm_param_grid, estimator=pipeline, n_iter=2, scoring='roc_auc', verbose=1)

# Fit the estimator
randomized_roc_auc.fit(X, y)

# Compute metrics
print(randomized_roc_auc.best_score_)
print(randomized_roc_auc.best_estimator_)


# ![image.png](attachment:image.png)

# # Final Thoughts
# 

# In[ ]:




