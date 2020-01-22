#!/usr/bin/env python
# coding: utf-8

# # Building tf-idf document vectors
# 

# ## tf-idf vectors for TED talks
# In this exercise, you have been given a corpus ted which contains the transcripts of 500 TED Talks. Your task is to generate the tf-idf vectors for these talks.
# 
# In a later lesson, we will use these vectors to generate recommendations of similar talks based on the transcript.

# ### init

# In[1]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(ted)
"""

tobedownloaded="""
 {pandas.core.series.Series: {'ted.csv': 'https://file.io/TyIcyb'}}
"""
prefix='data_from_datacamp/Chap4-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
ted = pd.read_csv(prefix+'ted.csv',index_col=0, header=None,squeeze=True)


# ### code

# In[2]:


# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(ted)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)


# # Cosine similarity
# 

# ## Computing dot product
# In this exercise, we will learn to compute the dot product between two vectors, A = (1, 3) and B = (-2, 2), using the numpy library. More specifically, we will use the np.dot() function to compute the dot product of two numpy arrays.

# ### code

# In[3]:


# Initialize numpy vectors
A = np.array([1,3])
B = np.array([-2,2])

# Compute dot product
dot_prod = np.dot(A, B)

# Print dot product
print(dot_prod)


# ## Cosine similarity matrix of a corpus
# In this exercise, you have been given a corpus, which is a list containing five sentences. The corpus is printed in the console. You have to compute the cosine similarity matrix which contains the pairwise cosine similarity score for every pair of sentences (vectorized using tf-idf).
# 
# Remember, the value corresponding to the ith row and jth column of a similarity matrix denotes the similarity score for the ith and jth vector.

# ### code

# In[4]:


corpus = ['The sun is the largest celestial body in the solar system', 'The solar system consists of the sun and eight revolving planets', 'Ra was the Egyptian Sun God', 'The Pyramids were the pinnacle of Egyptian architecture', 'The quick brown fox jumps over the lazy dog']


# In[5]:


from sklearn.metrics.pairwise import cosine_similarity


# In[7]:


# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Compute and print the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)


# # Building a plot line based recommender
# 

# ## Comparing linear_kernel and cosine_similarity
# In this exercise, you have been given tfidf_matrix which contains the tf-idf vectors of a thousand documents. Your task is to generate the cosine similarity matrix for these vectors first using cosine_similarity and then, using linear_kernel.
# 
# We will then compare the computation times for both functions.

# ### code

# In[ ]:


# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" %(time.time() - start))


# In[ ]:


# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" %(time.time() - start))


# ## Plot recommendation engine
# In this exercise, we will build a recommendation engine that suggests movies based on similarity of plot lines. You have been given a get_recommendations() function that takes in the title of a movie, a similarity matrix and an indices series as its arguments and outputs a list of most similar movies. indices has already been provided to you.
# 
# You have also been given a movie_plots Series that contains the plot lines of several movies. Your task is to generate a cosine similarity matrix for the tf-idf vectors of these plots.
# 
# Consequently, we will check the potency of our engine by generating recommendations for one of my favorite movies, The Dark Knight Rises.

# ### init

# In[16]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(indices, movie_plots)
"""

tobedownloaded="""
{pandas.core.series.Series: {'indices.csv': 'https://file.io/iTkOnT',
  'movie_plots.csv': 'https://file.io/c4BzRd'}}
"""
prefix='data_from_datacamp/Chap4-Exercise3.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
indices = pd.read_csv(prefix+'indices.csv',index_col=0, header=None,squeeze=True)
movie_plots = pd.read_csv(prefix+'movie_plots.csv',index_col=0, header=None,squeeze=True)


# In[22]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(metadata)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'metadata.csv': 'https://file.io/QGNgSl'}}
"""
prefix='data_from_datacamp/Chap4-Exercise3.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
metadata = pd.read_csv(prefix+'metadata.csv',index_col=0)


# In[23]:


movie_plots[movie_plots.isnull()]


# In[24]:


""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(get_recommendations)
"""
def get_recommendations(title, cosine_sim, indices):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

from sklearn.metrics.pairwise import linear_kernel


# ### code

# In[25]:


# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movie_plots)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
 
# Generate recommendations 
print(get_recommendations('The Dark Knight Rises', cosine_sim, indices))


# ## The recommender function
# In this exercise, we will build a recommender function get_recommendations(), as discussed in the lesson and the previous exercise. As we know, it takes in a title, a cosine similarity matrix, and a movie title and index mapping as arguments and outputs a list of 10 titles most similar to the original title (excluding the title itself).
# 
# You have been given a dataset metadata that consists of the movie titles and overviews. The head of this dataset has been printed to console.

# ### init

# In[28]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(metadata)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'metadata.csv': 'https://file.io/bhVPmN'}}
"""
prefix='data_from_datacamp/Chap4-Exercise3.3_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
metadata = pd.read_csv(prefix+'metadata.csv',index_col=0)


# In[29]:


metadata.head()


# ### code

# In[31]:


# Generate mapping between titles and index
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

def get_recommendations(title, cosine_sim, indices):
    # Get index of movie that matches title
    idx = indices[title]
    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]


# ## TED talk recommender
# In this exercise, we will build a recommendation system that suggests TED Talks based on their transcripts. You have been given a get_recommendations() function that takes in the title of a talk, a similarity matrix and an indices series as its arguments, and outputs a list of most similar talks. indices has already been provided to you.
# 
# You have also been given a transcripts series that contains the transcripts of around 500 TED talks. Your task is to generate a cosine similarity matrix for the tf-idf vectors of the talk transcripts.
# 
# Consequently, we will generate recommendations for a talk titled '5 ways to kill your dreams' by Brazilian entrepreneur Bel Pesce.

# ### init

# In[32]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(indices, transcripts)
"""

tobedownloaded="""
{pandas.core.series.Series: {'indices.csv': 'https://file.io/WS3cZP',
  'transcripts.csv': 'https://file.io/6L6VlS'}}
"""
prefix='data_from_datacamp/Chap4-Exercise3.4_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
indices = pd.read_csv(prefix+'indices.csv',index_col=0, header=None,squeeze=True)
transcripts = pd.read_csv(prefix+'transcripts.csv',index_col=0, header=None,squeeze=True)


# In[40]:


#upload and download

from uploadfromdatacamp import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(ted)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'ted.csv': 'https://file.io/80JHyW'}}
"""
prefix='data_from_datacamp/Chap4-Exercise3.4_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
ted = pd.read_csv(prefix+'ted.csv',index_col=0)


# In[41]:


def get_recommendations(title, cosine_sim, indices):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    talk_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return ted['title'].iloc[talk_indices]


# ### code

# In[42]:


# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(transcripts)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix)
 
# Generate recommendations 
print(get_recommendations('5 ways to kill your dreams', cosine_sim, indices))


# # Beyond n-grams: word embeddings
# 

# ## Generating word vectors
# In this exercise, we will generate the pairwise similarity scores of all the words in a sentence. The sentence is available as sent and has been printed to the console for your convenience.

# ### code

# In[49]:


import en_core_web_sm
nlp = en_core_web_sm.load()


# In[50]:


sent = 'I like apples and oranges'
import spacy
# Load model and create Doc object
nlp = spacy.load('en_core_web_sm')


# In[51]:


# Create the doc object
doc = nlp(sent)

# Compute pairwise similarity scores
for token1 in doc:
  for token2 in doc:
    print(token1.text, token2.text, token1.similarity(token2))


# ## Computing similarity of Pink Floyd songs
# In this final exercise, you have been given lyrics of three songs by the British band Pink Floyd, namely 'High Hopes', 'Hey You' and 'Mother'. The lyrics to these songs are available as hopes, hey and mother respectively.
# 
# Your task is to compute the pairwise similarity between mother and hopes, and mother and hey.

# ### init

# In[52]:


hopes = "\nBeyond the horizon of the place we lived when we were young\nIn a world of magnets and miracles\nOur thoughts strayed constantly and without boundary\nThe ringing of the division bell had begun\nAlong the Long Road and on down the Causeway\nDo they still meet there by the Cut\nThere was a ragged band that followed in our footsteps\nRunning before times took our dreams away\nLeaving the myriad small creatures trying to tie us to the ground\nTo a life consumed by slow decay\nThe grass was greener\nThe light was brighter\nWhen friends surrounded\nThe nights of wonder\nLooking beyond the embers of bridges glowing behind us\nTo a glimpse of how green it was on the other side\nSteps taken forwards but sleepwalking back again\nDragged by the force of some in a tide\nAt a higher altitude with flag unfurled\nWe reached the dizzy heights of that dreamed of world\nEncumbered forever by desire and ambition\nThere's a hunger still unsatisfied\nOur weary eyes still stray to the horizon\nThough down this road we've been so many times\nThe grass was greener\nThe light was brighter\nThe taste was sweeter\nThe nights of wonder\nWith friends surrounded\nThe dawn mist glowing\nThe water flowing\nThe endless river\nForever and ever\n"


# In[53]:


mother = "\nMother do you think they'll drop the bomb?\nMother do you think they'll like this song?\nMother do you think they'll try to break my balls?\nOoh, ah\nMother should I build the wall?\nMother should I run for President?\nMother should I trust the government?\nMother will they put me in the firing mine?\nOoh ah,\nIs it just a waste of time?\nHush now baby, baby, don't you cry.\nMama's gonna make all your nightmares come true.\nMama's gonna put all her fears into you.\nMama's gonna keep you right here under her wing.\nShe won't let you fly, but she might let you sing.\nMama's gonna keep baby cozy and warm.\nOoh baby, ooh baby, ooh baby,\nOf course mama's gonna help build the wall.\nMother do you think she's good enough, for me?\nMother do you think she's dangerous, to me?\nMother will she tear your little boy apart?\nOoh ah,\nMother will she break my heart?\nHush now baby, baby don't you cry.\nMama's gonna check out all your girlfriends for you.\nMama won't let anyone dirty get through.\nMama's gonna wait up until you get in.\nMama will always find out where you've been.\nMama's gonna keep baby healthy and clean.\nOoh baby, ooh baby, ooh baby,\nYou'll always be baby to me.\nMother, did it need to be so high?\n"


# In[54]:


hey = "\nHey you, out there in the cold\nGetting lonely, getting old\nCan you feel me?\nHey you, standing in the aisles\nWith itchy feet and fading smiles\nCan you feel me?\nHey you, don't help them to bury the light\nDon't give in without a fight\nHey you out there on your own\nSitting naked by the phone\nWould you touch me?\nHey you with you ear against the wall\nWaiting for someone to call out\nWould you touch me?\nHey you, would you help me to carry the stone?\nOpen your heart, I'm coming home\nBut it was only fantasy\nThe wall was too high\nAs you can see\nNo matter how he tried\nHe could not break free\nAnd the worms ate into his brain\nHey you, out there on the road\nAlways doing what you're told\nCan you help me?\nHey you, out there beyond the wall\nBreaking bottles in the hall\nCan you help me?\nHey you, don't tell me there's no hope at all\nTogether we stand, divided we fall\n"


# ### code

# In[56]:


# Create Doc objects
mother_doc = nlp(mother)
hopes_doc = nlp(hopes)
hey_doc = nlp(hey)

# Print similarity between mother and hopes
print(mother_doc.similarity(hopes_doc))

# Print similarity between mother and hey
print(mother_doc.similarity(hey_doc))


# In[ ]:




