#!/usr/bin/env python
# coding: utf-8

# # Creating transcription helper functions
# 

# ## Converting audio to the right format
# Acme Studios have asked you to do a proof of concept to find out more about their audio files.
# 
# After exploring them briefly, you find there's a few calls but they're in the wrong file format for transcription.
# 
# As you'll be interacting with many audio files, you decide to begin by creating some helper functions.
# 
# The first one, convert_to_wav(filename) takes a file path and uses PyDub to convert it from a non-wav format to .wav format.
# 
# Once it's built, we'll use the function to convert Acme's first call, call_1.mp3, from .mp3 format to .wav.
# 
# PyDub's AudioSegment class has already been imported. Remember, to work with non-wav files, you'll need ffmpeg.

# ### init

# In[1]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('call_1.mp3')
"""

tobedownloaded="""
{numpy.ndarray: {'call_1.mp3': 'https://file.io/LR9nRLl5'}}
"""
prefixToc = '1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# In[2]:


# Import AudioSegment from Pydub
from pydub import AudioSegment


# ### code

# In[10]:


import os

# Create function to convert audio file to wav
def convert_to_wav(filename):
  """Takes an audio file of non .wav format and converts to .wav"""
  # Import audio file
  audio = AudioSegment.from_file(filename)
  
  # Create new filename
  new_filename = os.path.splitext(filename)[0] + ".wav"
  
  # Export file as .wav
  audio.export(new_filename, format='wav')
  print(f"Converting {filename} to {new_filename}...")
 
# Test the function
convert_to_wav(prefix+'call_1.mp3')


# ## Finding PyDub stats
# You decide it'll be helpful to know the audio attributes of any given file easily. This will be especially helpful for finding out how many channels an audio file has or if the frame rate is adequate for transcription.
# 
# In this exercise, we'll create show_pydub_stats() which takes a filename of an audio file as input. It then imports the audio as a PyDub AudioSegment instance and prints attributes such as number of channels, length and more.
# 
# It then returns the AudioSegment instance so it can be used later on.
# 
# We'll use our function on the newly converted .wav file, call_1.wav
# 
# AudioSegment has already imported from PyDub.

# ### code

# In[11]:


def show_pydub_stats(filename):
  """Returns different audio attributes related to an audio file."""
  # Create AudioSegment instance
  audio_segment = AudioSegment.from_file(filename)
  
  # Print audio attributes and return AudioSegment instance
  print(f"Channels: {audio_segment.channels}")
  print(f"Sample width: {audio_segment.sample_width}")
  print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
  print(f"Frame width: {audio_segment.frame_width}")
  print(f"Length (ms): {len(audio_segment)}")
  return audio_segment

# Try the function
call_1_audio_segment = show_pydub_stats(prefix+'call_1.wav')


# ## Transcribing audio with one line
# Alright, now you've got functions to convert audio files and find out their attributes, it's time to build one to transcribe them.
# 
# In this exercise, you'll build transcribe_audio() which takes a filename as input, imports the filename using speech_recognition's AudioFile class and then transcribes it using recognize_google().
# 
# You've seen these functions before but now we'll put them together so they're accessible in a function.
# 
# To test it out, we'll transcribe Acme's first call, "call_1.wav".
# 
# speech_recognition has been imported as sr.

# ### code

# In[12]:


import speech_recognition as sr


# In[13]:


def transcribe_audio(filename):
  """Takes a .wav format audio file and transcribes it to text."""
  # Setup a recognizer instance
  recognizer = sr.Recognizer()
  
  # Import the audio file and convert to audio data
  audio_file = sr.AudioFile(filename)
  with audio_file as source:
    audio_data = recognizer.record(source)
  
  # Return the transcribed text
  return recognizer.recognize_google(audio_data)

# Test the function
print(transcribe_audio(prefix+'call_1.wav'))


# ## Using the helper functions you've built
# Okay, now we've got some helper functions ready to go, it's time to put them to use!
# 
# You'll first use convert_to_wav() to convert Acme's call_1.mp3 to .wav format and save it as call_1.wav
# 
# Using show_pydub_stats() you find call_1.wav has 2 channels so you decide to split them using PyDub's split_to_mono(). Acme tells you the customer channel is likely channel 2. So you export channel 2 using PyDub's .export().
# 
# Finally, you'll use transcribe_audio() to transcribe channel 2 only.

# ### init

# In[15]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('call_1.mp3')
"""

tobedownloaded="""
{numpy.ndarray: {'call_1.mp3': 'https://file.io/vegb3SL2'}}
"""
prefixToc = '1.4'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[16]:


# Convert mp3 file to wav
convert_to_wav(prefix+'call_1.mp3')

# Check the stats of new file
call_1 = show_pydub_stats(prefix+'call_1.wav')


# In[17]:


# Split call_1 to mono
call_1_split = call_1.split_to_mono()

# Export channel 2 (the customer channel)
call_1_split[1].export(prefix+"call_1_channel_2.wav",
                       format='wav')


# In[18]:


# Transcribe the single channel
print(transcribe_audio(prefix+"call_1_channel_2.wav"))


# # Sentiment analysis on spoken language text
# 

# ## Analyzing sentiment of a phone call
# Once you've transcribed the text from an audio file, it's possible to perform natural language processing on the text.
# 
# In this exercise, we'll use NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to analyze the sentiment of the transcribed text of call_2.wav.
# 
# To transcribe the text, we'll use the transcribe_audio() function we created earlier.
# 
# Once we have the text, we'll use NLTK's SentimentIntensityAnalyzer() class to obtain a sentiment polarity score.
# 
# .polarity_scores(text) returns a value for pos (positive), neu (neutral), neg (negative) and compound. Compound is a mixture of the other three values. The higher it is, the more positive the text. Lower means more negative.

# ### init

# In[19]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('call_2.wav')
"""

tobedownloaded="""
{numpy.ndarray: {'call_2.wav': 'https://file.io/4Sjun5W4'}}
"""
prefixToc = '2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# In[21]:


import nltk
nltk.download('vader_lexicon')


# In[23]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('call_2_channel_2.wav')
"""

tobedownloaded="""
{numpy.ndarray: {'call_2_channel_2.wav': 'https://file.io/5cOeP97R'}}
"""
prefixToc = '2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[22]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create SentimentIntensityAnalyzer instance
sid = SentimentIntensityAnalyzer()

# Let's try it on one of our phone calls
call_2_text = transcribe_audio(prefix+'call_2.wav')

# Display text and sentiment polarity scores
print(call_2_text)
print(sid.polarity_scores(call_2_text))


# In[24]:


# Transcribe customer channel of call 2
call_2_channel_2_text = transcribe_audio(prefix+'call_2_channel_2.wav')

# Display text and sentiment polarity scores
print(call_2_channel_2_text)
print(sid.polarity_scores(call_2_channel_2_text))


# In[25]:


# Import sent tokenizer
from nltk.tokenize import sent_tokenize


# In[26]:


# Split call 2 channel 2 into sentences and score each
for sentence in sent_tokenize(call_2_channel_2_text):
    print(sentence)
    print(sid.polarity_scores(sentence))


# In[27]:


call_2_channel_2_paid_api_text = "Hello and welcome to acme studios. My name's Daniel. How can I best help you? Hi Diane. This is paid on this call up to see the status of my, I'm proctor mortars at three weeks ago, and then service is terrible. Okay, Peter, sorry to hear about that. Hey, Peter, before we go on, do you mind just, uh, is there something going on with your microphone? I can't quite hear you. Is this any better? Yeah, that's much better. And sorry, what was, what was it that you said when you first first started speaking?  So I ordered a product from you guys three weeks ago and, uh, it's, it's currently on July 1st and I haven't received a provocative, again, three weeks to a full four weeks down line. This service is terrible. Okay. Well, what's your order id? I'll, uh, I'll start looking into that for you. Six, nine, eight, seven five. Okay. Thank you."


# In[28]:


# Split channel 2 paid text into sentences and score each
for sentence in sent_tokenize(call_2_channel_2_paid_api_text):
    print(sentence)
    print(sid.polarity_scores(sentence))


# # Named entity recognition on transcribed text
# 

# ## Named entity recognition in spaCy
# Named entities are real-world objects which have names, such as, cities, people, dates or times. We can use spaCy to find named entities in our transcribed text.
# 
# In this exercise, you'll transcribe call_4_channel_2.wav using transcribe_audio() and then use spaCy's language model, en_core_web_sm to convert the transcribed text to a spaCy doc.
# 
# Transforming text to a spaCy doc allows us to leverage spaCy's built-in features for analyzing text, such as, .text for tokens (single words), .sents for sentences and .ents for named entities.

# ### init

# In[1]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('call_4_channel_2.wav')
uploadToFileIO_pushto_fileio('call_4.wav')

{"success":true,"key":"NQK8vWrq","link":"https://file.io/NQK8vWrq","expiry":"14 days"}
{"success":true,"key":"rItZyWvo","link":"https://file.io/rItZyWvo","expiry":"14 days"}

"""

tobedownloaded="""
{numpy.ndarray: {'call_4_channel_2.wav': 'https://file.io/NQK8vWrq', 'call_4.wav': 'https://file.io/rItZyWvo'}}
"""
prefixToc = '3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# In[4]:


import speech_recognition as sr

def transcribe_audio(filename):
  """Takes a .wav format audio file and transcribes it to text."""
  # Setup a recognizer instance
  recognizer = sr.Recognizer()
  
  # Import the audio file and convert to audio data
  audio_file = sr.AudioFile(filename)
  with audio_file as source:
    audio_data = recognizer.record(source)
  
  # Return the transcribed text
  return recognizer.recognize_google(audio_data)

# Test the function
print(transcribe_audio(prefix+'call_4_channel_2.wav'))


# ### code

# In[5]:


import spacy

# Transcribe call 4 channel 2
call_4_channel_2_text = transcribe_audio(prefix+"call_4_channel_2.wav")

# Create a spaCy language model instance
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc with call 4 channel 2 text
doc = nlp(call_4_channel_2_text)

# Check the type of doc
print(type(doc))


# In[7]:


# Show tokens in doc
for token in doc:
    print(token.text, token.idx)


# In[9]:


# Show sentences in doc
for sentence in doc.sents:
    print(sentence)


# In[10]:


# Show named entities and their labels
for entity in doc.ents:
    print(entity.text, entity.label_)


# ## Creating a custom named entity in spaCy
# If spaCy's built-in named entities aren't enough, you can make your own using spaCy's EntityRuler() class.
# 
# EntityRuler() allows you to create your own entities to add to a spaCy pipeline.
# 
# You start by creating an instance of EntityRuler() and passing it the current pipeline, nlp.
# 
# You can then call add_patterns() on the instance and pass it a dictionary of the text pattern you'd like to label with an entity.
# 
# Once you've setup a pattern you can add it the nlp pipeline using add_pipe().
# 
# Since Acme is a technology company, you decide to tag the pattern "smartphone" with the "PRODUCT" entity tag.
# 
# spaCy has been imported and a doc already exists containing the transcribed text from call_4_channel_2.wav.

# ### code

# In[11]:


# Import EntityRuler class
from spacy.pipeline import EntityRuler

# Create EntityRuler instance
ruler = EntityRuler(nlp)

# Define pattern for new entity
ruler.add_patterns([{"label": "PRODUCT", "pattern": "smartphone"}])

# Update existing pipeline
nlp.add_pipe(ruler, before="ner")

# Test new entity
for entity in doc.ents:
  print(entity.text, entity.label_)


# # Classifying transcribed speech with Sklearn
# 

# ## Preparing audio files for text classification
# Acme are very impressed with your work so far. So they've sent over two more folders of audio files.
# 
# One folder is called pre_purchase and contains audio snippets from customers who are pre-purchase, like pre_purchase_audio_25.mp3.
# 
# And the other is called post_purchase and contains audio snippets from customers who have made a purchase (post-purchase), like post_purchase_audio_27.mp3.
# 
# Upon inspecting the files you find there's about 50 in each and they're in the .mp3 format.
# 
# Acme want to know if you can build a classifier to classify future calls. You tell them you sure can.
# 
# So in this exercise, you'll go through each folder and convert the audio files to .wav format using convert_to_wav() so you can transcribe them.

# ### code

# In[12]:


# Convert post purchase
for file in post_purchase:
    print(f"Converting {file} to .wav...")
    convert_to_wav(file)

# Convert pre purchase
for file in pre_purchase:
    print(f"Converting {file} to .wav...")
    convert_to_wav(file)


# ## Transcribing phone call excerpts
# In this exercise, we'll transcribe the audio files we converted to .wav format to text using transcribe_audio().
# 
# Since there's lots of them and there could be more, we'll build a function create_test_list() which takes a list of filenames of audio files as input and goes through each file transcribing the text.
# 
# create_test_list() uses our transcribe_audio() function we created earlier and returns a list of strings containing the transcribed text from each audio file.
# 
# pre_purchase_wav_files and post_purchase_wav_files are lists of audio snippet filenames.

# ### code

# In[13]:


def create_text_list(folder):
  # Create empty list
  text_list = []
  
  # Go through each file
  for file in folder:
    # Make sure the file is .wav
    if file.endswith(".wav"):
      print(f"Transcribing file: {file}...")
      
      # Transcribe audio and append text to list
      text_list.append(transcribe_audio(file))   
  return text_list

create_text_list(folder)


# In[ ]:


# Transcribe post and pre purchase text
post_purchase_text = create_text_list(post_purchase_wav_files)
pre_purchase_text = create_text_list(pre_purchase_wav_files)

# Inspect the first transcription of post purchase
print(post_purchase_text[0])


# ## Organizing transcribed phone call data
# We're almost ready to build a text classifier. But right now, all of our transcribed text data is in two lists, pre_purchase_text and post_purchase_text.
# 
# To organize it better for building a text classifier as well as for future use, we'll put it together into a pandas DataFrame.
# 
# To start we'll import pandas as pd then we'll create a post purchase dataframe, post_purchase_df using pd.DataFrame().
# 
# We'll pass pd.DataFrame() a dictionary containing a "label" key with a value of "post_purchase" and a "text" key with a value of our post_purchase_text list.
# 
# We'll do the same for pre_purchase_df except with pre_purchase_text.
# 
# To have all the data in one place, we'll use pd.concat() and pass it the pre and post purchase DataFrames.

# ### init

# In[14]:


pre_purchase_text = ['yeah hi John just calling in regards to a recent order I just placed I found a cheaper product online and I was wondering if I could cancel that',
 "I was looking online it says that you're only size is available a large and small I was wondering if you'll have any mediums in soon",
 'hi I was just wondering if you have the extra large tea and blue',
 'yeah hey Steve just calling in regards to a recent order I just placed I was wondering if I could cancel that order',
 'hi I just ordered a new phone and I was just wondering if I could cancel out order and organise a refund',
 'hi I just ordered a new t-shirt and I was wondering if I could cancel an order and organise a refund',
 'accidentally made some errors and order I recently just placed I was wondering if you could help me',
 "I just placed an order online and I was just wondering when I'll get my confirmation email",
 "hey mate I just finished paying for my order and I was just wondering when I'm going to get that email to confirm it",
 'hey I was wondering if you know where my new phone is that I just recently ordered',
 'do you currently offer any new promotions at the moment',
 "hi I just pre-ordered the nudity and this is my order number but doctor I was just wondering if you know where abouts it isn't shipment",
 'your hi Jacob looking to make an order but just have a few questions regarding some products that you have online',
 'hi I just recently placed an order with your company I was just wondering if you know the status of my shipment',
 "Archie thank god I'm free been on hold for the last 30 minutes yeah got a couple of complaints made about this order I just posted",
 "hi just calling in regards to my order on November the 3rd I was just wondering when that's going to leave your office",
 "just looking to get some more information on the current promotions you're offering right now before I place my order",
 "hi I recently ordered a new phone and I'm just wondering where I could find my reference number for the delivery",
 'hey mate just looking to make some alterations to my order I just placed',
 'hey just looking to place this order but I see that you have a promotion still running can you give me some more details behind this promotion',
 "hi I placed an order a couple days ago and I was just wondering why my tracking number isn't working",
 'hi I just realised I ordered the wrong computer I was wondering if I could just cancel that and organise a refund',
 "yeah I just placed an all this you guys and I was wondering if I could change a few things before it's shift out",
 "how's it going after I just placed an order with you guys and I accidentally sent it to the wrong address can you please help me change this",
 "hey Polly just looking to place an order but before I proceed I'm just wondering if this offer still stands",
 'yeah hi Tommy I just placed an order with you guys but I use the wrong payment processing method I was wondering if I could change that',
 'hi Michael just looking to enquire about a few things before I placed an order I was wondering if you could help me',
 'hi I saw your new phone on your website I was wondering if you have any setup tips for',
 "I just ordered the new remote control car off you website I was just didn't see how many horsepower it has can you tell me",
 'hi just about to order these shoes online I was just wondering if you have any different sizes in store',
 'I just placed an order and I was wondering if I could change my shipping time from standard business days to rush if possible',
 'hey I just ordered the new phone and I was wondering if I could get airpods put into that order just before you guys send it',
 'hi Jacob I just placed an order with you guys but I found the same product online it and other store for a cheaper price I was wondering if you could price match it or could I cancel this order',
 'it says here you have the iPhone x l and X I was wondering if you still stock the iPhone 10',
 'hey I was just looking online at your shoes and I was wondering if you have this brand in Pink',
 'I just placed an order I was wondering how long shipping time would be expected to be',
 "hey mate just have a few questions regarding the recent order I just posted it shows that it's coming from overseas however when I looked at the Australian soccer shop online it says that there's current stock in store for the Australian store",
 'hi I just ordered some shoes and I was just wondering if I could cancel that order and make a refund',
 'hey I just ordered the blue and yellow shoes off your website and I was wondering if I could cancel that order and organise a refund',
 'hey so I just placed an order with your company and I was just wondering where I can find my reference number',
 'hey I was just wondering about the sizing on your shirts it says us as how does that relate to AUD',
 "hi Tony I just placed an order I'm currently having a few problems I was wondering if you could help me",
 'yeah hi David I just placed an order online and I was wondering if I could make an alteration to that order before you send it off',
 'hi I was just looking at finding a new phone I was wondering if you could recommend anything to me',
 'I I just ordered the green and blue shoes off your website and I was wondering if I could add a shirt to my order before you send it']

post_purchase_text = ['hey man I just bought a product from you guys and I think is amazing but I leave a little help setting it up',
 'these clothes I just bought from you guys too small is there any way I can change the size',
 "I recently got these pair of shoes but they're too big can I change the size",
 "I bought a pair of pants from you guys but they're way too small",
 "I bought a pair of pants and they're the wrong colour is there any chance I can change that",
 "hey mate how you doing I'm just calling in regards the product that god it's faulty and doesn't work",
 "just wondering if there's any tutorials on how to set up my device I just received",
 "hey I'm just not happy with the product that you guys send me there any chance I can swap it out for another one",
 'I bought a pair of pants from you guys and they are just a bit too long do you guys do Hemi',
 'is there anybody that can help me set up this product or any how to use',
 "hey mate I just bought a product from you guys and I'm just unhappy with the pop the product can I return it",
 "just received the product from you guys and it didn't meet my expectations can I please get a refund",
 "what's the process I have to go through to send my product back for a swap",
 "hey mate how are you doing just wanting to know if there's any support I can get on this device how to set it up",
 "what's your refund policy on items that I've purchased from you guys",
 "hey how we doing I just put a cat from you guys and it's just the Wrong Colours is there any chance I can change that",
 "call me on to talk about a package I got yesterday it's I got it but I need to do I need some help with setting it up",
 "I got my order yesterday and the order number is 1863 3845 I'm just calling up to to check some more details on that",
 'I would have a couple of things from you guys the other day and two it two of them two of them and great and I love them but the other one is is not the right thing',
 "yeah hello I'm just wondering if I can speak to someone about an order I received yesterday",
 'wrong package delivered',
 "hey I ordered something yesterday and it arrived it arrived this morning but it seems like there's a few a few extra things in there that I didn't really order is there someone that I can talk to you to fix this up",
 "hey I bought something from your website the other day and it arrived but it's it's not the thing that I ordered is there someone I can talk to her to fix this up",
 "hello someone from your team delivered my package today but it's it's got a problem with it",
 "my shipment arrived this afternoon but it's wrong size is there anyone I can talk to you to change it",
 'I just bought a item from you guys and ID want to know if I can swap it for a different colour',
 "hey I received my order but it's the wrong size can I get a refund please",
 "hey my order arrived today but it's it's there's a it's I don't think it's the one that I ordered I check the receipt and it doesnt match what what a right",
 "hey I'm calling up to to see if I can talk to someone to help with her a shipment that I received yesterday",
 "I just received this device and I'd love some supported to be able to set it up",
 "I just bought a product from you guys and I wouldn't want to know if I can send it back to get a colour change",
 "I purchase something from your online store yesterday but the receipt didn't come through can can I get another receipt emailed please",
 'the product arrived and there was a few things in the box but two of them the wrong is there someone I can talk to about fixing up my order',
 "I'm just happy with the colour that I got from you guys so is there any chance I can change it for a different one",
 "a couple of days ago I got a message saying that my package have been delivered it wasn't delivered that day but it still hasn't arrived there someone I can talk to about my order",
 "my shipment arrived yesterday but it's not the right thing is there someone I can talk to you to fix it up",
 "my shipment arrived yesterday but it's not the right thing is there someone I can talk to you to fix it up",
 "my package was supposed to be delivered yesterday but it it didn't arrive is there someone I can talk to about my order",
 "my package was supposed to be delivered yesterday but it it didn't arrive is there someone I can talk to about my order",
 "I bought a hat from you guys and it's just too big is there anyway I can get it down size and what's your policies on that",
 'calling in regards to the order I just got would love some support',
 "my order a 64321 arrived this morning but it's something wrong with it is there someone I can talk to to fix it",
 "yeah hello someone this morning delivered a package but I think it's I think it's not the right one that I ordered is there someone I can talk to you too to change it",
 "on the box that you sent me yesterday arrived but it's damaged the someone I can talk to her about replacement",
 "I've just bought a product can you guys and I want to know what your return keys and Caesar",
 "my order a 64321 arrived this morning but it's something wrong with it is there someone I can talk to to fix it",
 "hey my name is Daniel I received my shipment yesterday but it's wrong can I change it",
 "all the things I received the my order yesterday would damaged I'm not sure what happened to delivery is there someone that can give me a hand",
 'the shipment I received is wrong',
 "yeah hey I need I need some help with her with an order that I ordered the other day it it came and it wasn't it wasn't correct",
 "yeah hello someone this morning delivered a package but I think it's I think it's not the right one that I ordered is there someone I can talk to you too to change it",
 'the shipment I received is wrong',
 "yeah hello I'm just wondering if I can speak to someone about an order I received yesterday",
 "my shipment arrived this afternoon but it's wrong size is there anyone I can talk to you to change it",
 "all the things I received the my order yesterday would damaged I'm not sure what happened to delivery is there someone that can give me a hand",
 'hey mate the must have been a problem with the shipping because the product I just received from you is damaged',
 "hey mate how you doing just calling in regards to the phone I just purchased from you guys faulty not working and now he's damaged on the way here"]


# ### code

# In[16]:


import pandas as pd

# Make dataframes with the text
post_purchase_df = pd.DataFrame({"label": "post_purchase",
                                 "text": post_purchase_text})
pre_purchase_df = pd.DataFrame({"label": "pre_purchase",
                                "text": pre_purchase_text})

# Combine DataFrames
df = pd.concat([post_purchase_df, pre_purchase_df])

# Print the combined DataFrame
print(df.head())


# ## Create a spoken language text classifier
# Now you've transcribed some customer call audio data, we'll build a model to classify whether the text from the customer call is pre_purchase or post_purchase.
# 
# We've got 45 examples of pre_purchase calls and 57 examples of post_purchase calls.
# 
# The data the model will train on is stored in train_df and the data the model will predict on is stored in test_df.
# 
# Try printing the .head() of each of these to the console.
# 
# We'll build an sklearn pipeline using CountVectorizer() and TfidfTransformer() to convert our text samples to numbers and then use a MultinomialNB() classifier to learn what category each sample belongs to.
# 
# This model will work well on our small example here but for larger amounts of text, you may want to consider something more sophisticated.

# ### init

# In[25]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(train_df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'train_df.csv': 'https://file.io/YhAD08rE'}}
"""
prefixToc='4.4'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
train_df = pd.read_csv(prefix+'train_df.csv',index_col=0)


# In[28]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(test_df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'test_df.csv': 'https://file.io/0Y7QdGXP'}}
"""
prefixToc='4.4'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
test_df = pd.read_csv(prefix+'test_df.csv',index_col=0)


# In[26]:


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split


# ### code

# In[27]:


# Build the text_classifier as an sklearn pipeline
text_classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

# Fit the classifier pipeline on the training data
text_classifier.fit(train_df.text, train_df.label)


# In[29]:


# Evaluate the MultinomialNB model
predicted = text_classifier.predict(test_df.text)
accuracy = 100 * np.mean(predicted == test_df.label)
print(f'The model is {accuracy}% accurate')


# In[ ]:




