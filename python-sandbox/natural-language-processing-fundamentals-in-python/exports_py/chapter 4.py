#!/usr/bin/env python
# coding: utf-8

# # Building word count vectors with scikit-learn
# 

# ## CountVectorizer for text classification
# It's time to begin building your text classifier! The data has been loaded into a DataFrame called df. Explore it in the IPython Shell to investigate what columns you can use. The .head() method is particularly informative.
# 
# In this exercise, you'll use pandas alongside scikit-learn to create a sparse text vectorizer you can use to train and test a simple supervised model. To begin, you'll set up a CountVectorizer and investigate some of its features.

# ### code : 1 dataframe

# In[5]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(df)
tobedownloaded="{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/wmCKCD'}}"
prefix='data_from_datacamp/Chap4-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[6]:


import pandas as pd

df=pd.read_csv(prefix+'df.csv',index_col=0, sep="|")


# In[7]:


df


# ### code

# In[9]:


# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df["label"]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], y, test_size=0.33, random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])


# ## TfidfVectorizer for text classification
# Similar to the sparse CountVectorizer created in the previous exercise, you'll work on creating tf-idf vectors for your documents. You'll set up a TfidfVectorizer and investigate some of its features.
# 
# In this exercise, you'll use pandas and sklearn along with the same X_train, y_train and X_test, y_test DataFrames and Series you created in the last exercise.

# ### code

# In[10]:


# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])


# ## Inspecting the vectors
# To get a better idea of how the vectors work, you'll investigate them by converting them into pandas DataFrames.
# 
# Here, you'll use the same data structures you created in the previous two exercises (count_train, count_vectorizer, tfidf_train, tfidf_vectorizer) as well as pandas, which is imported as pd.

# ### code

# In[11]:


# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(tfidf_df.columns) - set(count_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))


# # Training and testing a classification model with scikit-learn
# 

# ## Training and testing the "fake news" model with CountVectorizer
# Now it's your turn to train the "fake news" model using the features you identified and extracted. In this first exercise you'll train and test a Naive Bayes model using the CountVectorizer data.
# 
# The training and test sets have been created, and count_vectorizer, count_train, and count_test have been computed.

# ### init

# In[12]:


from uploadfromdatacamp import saveFromFileIO
#import scipy
#scipy.sparse.save_npz('count_test.npz', count_test)
#scipy.sparse.save_npz('count_train.npz', count_train)
#print(uploadToFileIO_pushto_fileio('count_test.npz'))
#print(uploadToFileIO_pushto_fileio('count_train.npz'))

#uploadToFileIO(y_train, y_test)

tobedownloaded="""{pandas.core.series.Series: {'y_test.csv': 'https://file.io/Ox7O8S',
  'y_train.csv': 'https://file.io/BByC19'},
                csr_matrix:{
                    'count_test.npz': 'https://file.io/AxwzIb',
                    'count_train.npz': 'https://file.io/LkaENZ'
                }}"""
prefix='data_from_datacamp/Chap4-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[14]:


import scipy

count_test = scipy.sparse.load_npz(prefix+'count_test.npz')
count_train = scipy.sparse.load_npz(prefix+'count_train.npz')

y_test  =pd.read_csv(prefix+'y_test.csv', index_col=0, header=None,squeeze=True)
y_train =pd.read_csv(prefix+'y_train.csv', index_col=0, header=None,squeeze=True)


# In[15]:


count_vectorizer = CountVectorizer(stop_words='english')


# ### code

# In[17]:


# Import the necessary modules
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)


# ## Training and testing the "fake news" model with TfidfVectorizer
# Now that you have evaluated the model using the CountVectorizer, you'll do the same using the TfidfVectorizer with a Naive Bayes model.
# 
# The training and test sets have been created, and tfidf_vectorizer, tfidf_train, and tfidf_test have been computed. Additionally, MultinomialNB and metrics have been imported from, respectively, sklearn.naive_bayes and sklearn.

# ### init

# In[18]:


from uploadfromdatacamp import saveFromFileIO

fromDatacamp = """
import scipy
scipy.sparse.save_npz('tfidf_test.npz', tfidf_test)
scipy.sparse.save_npz('tfidf_train.npz', tfidf_train)
uploadToFileIO_pushto_fileio('tfidf_test.npz')
uploadToFileIO_pushto_fileio('tfidf_train.npz')
uploadToFileIO(y_train, y_test)
"""

tobedownloaded="""{pandas.core.series.Series: {'y_test.csv': 'https://file.io/mzBimd',
  'y_train.csv': 'https://file.io/1Ox2bh'},
                csr_matrix:{
                    'tfidf_test.npz': 'https://file.io/2xhPOZ',
                    'tfidf_train.npz': 'https://file.io/WAM179'
                }}"""
prefix='data_from_datacamp/Chap4-Exercise2.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[19]:


import scipy

tfidf_test = scipy.sparse.load_npz(prefix+'tfidf_test.npz')
tfidf_train = scipy.sparse.load_npz(prefix+'tfidf_train.npz')

y_test  =pd.read_csv(prefix+'y_test.csv', index_col=0, header=None,squeeze=True)
y_train =pd.read_csv(prefix+'y_train.csv', index_col=0, header=None,squeeze=True)


# In[20]:


# Import the necessary modules
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# ### code

# In[21]:


# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)


# # Simple NLP, complex problems
# 

# ## Improving your model
# Your job in this exercise is to test a few different alpha levels using the Tfidf vectors to determine if there is a better performing combination.
# 
# The training and test sets have been created, and tfidf_vectorizer, tfidf_train, and tfidf_test have been computed.

# ### code

# In[24]:


# Create the list of alphas: alphas
alphas = np.arange(0,1,0.1)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test, pred)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()


# ## Inspecting your model
# Now that you have built a "fake news" classifier, you'll investigate what it has learned. You can map the important vector weights back to actual words using some simple inspection techniques.
# 
# You have your well performing tfidf Naive Bayes classifier available as nb_classifier, and the vectors as tfidf_vectorizer.

# ### code

# In[25]:


# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])


# In[ ]:




