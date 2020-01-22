#!/usr/bin/env python
# coding: utf-8

# # Building a bag of words model
# 

# ## BoW model for movie taglines
# In this exercise, you have been provided with a corpus of more than 7000 movie tag lines. Your job is to generate the bag of words representation bow_matrix for these taglines. For this exercise, we will ignore the text preprocessing step and generate bow_matrix directly.
# 
# We will also investigate the shape of the resultant bow_matrix. The first five taglines in corpus have been printed to the console for you to examine.

# ### init

# In[1]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(corpus)
"""

tobedownloaded="""
{pandas.core.series.Series: {'corpus.csv': 'https://file.io/eVXVNs'}}
"""
prefix='data_from_datacamp/Chap3-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
corpus = pd.read_csv(prefix+'corpus.csv',index_col=0, header=None,squeeze=True)


# ### code

# In[2]:


# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Print the shape of bow_matrix
print(bow_matrix.shape)


# ## Analyzing dimensionality and preprocessing
# In this exercise, you have been provided with a lem_corpus which contains the pre-processed versions of the movie taglines from the previous exercise. In other words, the taglines have been lowercased and lemmatized, and stopwords have been removed.
# 
# Your job is to generate the bag of words representation bow_lem_matrix for these lemmatized taglines and compare its shape with that of bow_matrix obtained in the previous exercise. The first five lemmatized taglines in lem_corpus have been printed to the console for you to examine.

# ### init

# In[3]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(lem_corpus)
"""

tobedownloaded="""
{pandas.core.series.Series: {'lem_corpus.csv': 'https://file.io/3fzva3'}}
"""
prefix='data_from_datacamp/Chap3-Exercise1.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
lem_corpus = pd.read_csv(prefix+'lem_corpus.csv',index_col=0, header=None,squeeze=True)


# ### code

# In[4]:


# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_lem_matrix = vectorizer.fit_transform(lem_corpus)

# Print the shape of bow_lem_matrix
print(bow_lem_matrix.shape)


# ## Mapping feature indices with feature names
# In the lesson video, we had seen that CountVectorizer doesn't necessarily index the vocabulary in alphabetical order. In this exercise, we will learn to map each feature index to its corresponding feature name from the vocabulary.
# 
# We will use the same three sentences on lions from the video. The sentences are available in a list named corpus and has already been printed to the console.

# ### code

# In[5]:


corpus = ['The lion is the king of the jungle',
 'Lions have lifespans of a decade',
 'The lion is an endangered species']


# In[6]:


# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Map the column names to vocabulary 
bow_df.columns = vectorizer.get_feature_names()

# Print bow_df
print(bow_df)


# # Building a BoW Naive Bayes classifier
# 

# ## BoW vectors for movie reviews
# In this exercise, you have been given two pandas Series, X_train and X_test, which consist of movie reviews. They represent the training and the test review data respectively. Your task is to preprocess the reviews and generate BoW vectors for these two sets using CountVectorizer.
# 
# Once we have generated the BoW vector matrices X_train_bow and X_test_bow, we will be in a very good position to apply a machine learning model to it and conduct sentiment analysis.

# ### init

# In[7]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_train, X_test)
"""

tobedownloaded="""
{pandas.core.series.Series: {'X_test.csv': 'https://file.io/VMCtL5',
  'X_train.csv': 'https://file.io/zBLW39'}}
"""
prefix='data_from_datacamp/Chap3-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
X_train = pd.read_csv(prefix+'X_train.csv',index_col=0, header=None,squeeze=True)
X_test = pd.read_csv(prefix+'X_test.csv',index_col=0, header=None,squeeze=True)


# ### code

# In[8]:


# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer object
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Fit and transform X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transform X_test
X_test_bow = vectorizer.transform(X_test)

# Print shape of X_train_bow and X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)


# ## Predicting the sentiment of a movie review
# In the previous exercise, you generated the bag-of-words representations for the training and test movie review data. In this exercise, we will use this model to train a Naive Bayes classifier that can detect the sentiment of a movie review and compute its accuracy. Note that since this is a binary classification problem, the model is only capable of classifying a review as either positive (1) or negative (0). It is incapable of detecting neutral reviews.
# 
# In case you don't recall, the training and test BoW vectors are available as X_train_bow and X_test_bow respectively. The corresponding labels are available as y_train and y_test respectively. Also, for you reference, the original movie review dataset is available as df.

# ### init

# In[10]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(df, y_train, y_test)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/xRv5dz'},
 pandas.core.series.Series: {'y_test.csv': 'https://file.io/dp0Ah3',
  'y_train.csv': 'https://file.io/MbLw61'}}
"""
prefix='data_from_datacamp/Chap3-Exercise2.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
df = pd.read_csv(prefix+'df.csv',index_col=0)
y_train = pd.read_csv(prefix+'y_train.csv',index_col=0, header=None,squeeze=True)
y_test = pd.read_csv(prefix+'y_test.csv',index_col=0, header=None,squeeze=True)


# ### code

# In[11]:


from sklearn.naive_bayes import MultinomialNB


# In[12]:


# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the classifier
clf.fit(X_train_bow, y_train)

# Measure the accuracy
accuracy = clf.score(X_test_bow, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))


# # Building n-gram models
# 

# ## n-gram models for movie tag lines
# In this exercise, we have been provided with a corpus of more than 9000 movie tag lines. Our job is to generate n-gram models up to n equal to 1, n equal to 2 and n equal to 3 for this data and discover the number of features for each model.
# 
# We will then compare the number of features generated for each model.

# ### init

# In[16]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(corpus)
"""

tobedownloaded="""
{pandas.core.series.Series: {'corpus.csv': 'https://file.io/KkQJic'}}
"""
prefix='data_from_datacamp/Chap3-Exercise3.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
corpus = pd.read_csv(prefix+'corpus.csv',index_col=0, header=None,squeeze=True)

corpus.dropna(inplace=True)


# ### code

# In[17]:


# Generate n-grams upto n=1
vectorizer_ng1 = CountVectorizer(ngram_range=(1,1))
ng1 = vectorizer_ng1.fit_transform(corpus)

# Generate n-grams upto n=2
vectorizer_ng2 = CountVectorizer(ngram_range=(1,2))
ng2 = vectorizer_ng2.fit_transform(corpus)

# Generate n-grams upto n=3
vectorizer_ng3 = CountVectorizer(ngram_range=(1, 3))
ng3 = vectorizer_ng3.fit_transform(corpus)

# Print the number of features for each model
print("ng1, ng2 and ng3 have %i, %i and %i features respectively" % (ng1.shape[1], ng2.shape[1], ng3.shape[1]))


# ## Higher order n-grams for sentiment analysis
# Similar to a previous exercise, we are going to build a classifier that can detect if the review of a particular movie is positive or negative. However, this time, we will use n-grams up to n=2 for the task.
# 
# The n-gram training reviews are available as X_train_ng. The corresponding test reviews are available as X_test_ng. Finally, use y_train and y_test to access the training and test sentiment classes respectively.

# ### init

# In[18]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(X_train, X_test, y_train, y_test)
"""

tobedownloaded="""
{pandas.core.series.Series: {'X_test.csv': 'https://file.io/nTwjmp',
  'X_train.csv': 'https://file.io/zqbOGc',
  'y_test.csv': 'https://file.io/0IS0B6',
  'y_train.csv': 'https://file.io/xPeTm2'}}
"""
prefix='data_from_datacamp/Chap3-Exercise3.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
X_train = pd.read_csv(prefix+'X_train.csv',index_col=0, header=None,squeeze=True)
X_test = pd.read_csv(prefix+'X_test.csv',index_col=0, header=None,squeeze=True)
y_train = pd.read_csv(prefix+'y_train.csv',index_col=0, header=None,squeeze=True)
y_test = pd.read_csv(prefix+'y_test.csv',index_col=0, header=None,squeeze=True)


# In[26]:


# Generate n-grams upto n=2
ng_vectorizer = CountVectorizer(ngram_range=(1,2))
X_train_ng = ng_vectorizer.fit_transform(X_train)
X_test_ng = ng_vectorizer.transform(X_test)


# ### code

# In[27]:


# Define an instance of MultinomialNB 
clf_ng = MultinomialNB()

# Fit the classifier 
clf_ng.fit(X_train_ng, y_train)

# Measure the accuracy 
accuracy = clf_ng.score(X_test_ng, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was not good. The plot had several holes and the acting lacked panache."
prediction = clf_ng.predict(ng_vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))


# ## Comparing performance of n-gram models
# You now know how to conduct sentiment analysis by converting text into various n-gram representations and feeding them to a classifier. In this exercise, we will conduct sentiment analysis for the same movie reviews from before using two n-gram models: unigrams and n-grams upto n equal to 3.
# 
# We will then compare the performance using three criteria: accuracy of the model on the test set, time taken to execute the program and the number of features created when generating the n-gram representation.

# ### init

# In[28]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/SzR8im'}}
"""
prefix='data_from_datacamp/Chap3-Exercise3.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
df = pd.read_csv(prefix+'df.csv',index_col=0)


# ### code

# In[32]:


import time
from sklearn.model_selection import train_test_split 


# In[33]:


start_time = time.time()
# Splitting the data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(df['review'], df['sentiment'], test_size=0.5, random_state=42, stratify=df['sentiment'])

# Generating ngrams
vectorizer = CountVectorizer(ngram_range=(1,1))
train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

# Fit classifier
clf = MultinomialNB()
clf.fit(train_X, train_y)

# Print accuracy, time and number of dimensions
print("The program took %.3f seconds to complete. The accuracy on the test set is %.2f. The ngram representation had %i features." % (time.time() - start_time, clf.score(test_X, test_y), train_X.shape[1]))


# In[34]:


start_time = time.time()
# Splitting the data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(df['review'], df['sentiment'], test_size=0.5, random_state=42, stratify=df['sentiment'])

# Generating ngrams
vectorizer = CountVectorizer(ngram_range=(1,3))
train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

# Fit classifier
clf = MultinomialNB()
clf.fit(train_X, train_y)

# Print accuracy, time and number of dimensions
print("The program took %.3f seconds to complete. The accuracy on the test set is %.2f. The ngram representation had %i features." % (time.time() - start_time, clf.score(test_X, test_y), train_X.shape[1]))


# In[ ]:




