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

# In[ ]:




