#!/usr/bin/env python
# coding: utf-8

# # Introduction to PyDub
# 

# ## Import an audio file with PyDub
# PyDub's AudioSegment class makes it easy to import and manipulate audio files with Python.
# 
# In this exercise, we'll import an audio file of interest by creating an instance of AudioSegment.
# 
# To import an audio file, you can use the from_file() function on AudioSegment and pass it your target audio file's pathname as a string. The format parameter gives you an option to specify the format of your audio file, however, this is optional as PyDub will automatically infer it.
# 
# PyDub works with .wav files without any extra dependencies but for other file types like .mp3, you'll need to install ffmpeg.
# 
# A sample audio file has been setup as wav_file.wav, you can listen to it here.

# ### init

# In[1]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('wav_file.wav')
"""

tobedownloaded="""
{numpy.ndarray: {'wav_file.wav': 'https://file.io/5IoS8lku'}}
"""
prefixToc = '1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[2]:


# Import AudioSegment from Pydub
from pydub import AudioSegment

# Create an AudioSegment instance
wav_file = AudioSegment.from_file(file=prefix+'wav_file.wav', 
                                  format="wav")

# Check the type
print(type(wav_file))


# ## Play an audio file with PyDub
# If you're working with audio files, chances are you want to listen to them.
# 
# PyDub's playback module provides a function called play() which can be passed an AudioSegment. Running the play() function with an AudioSegment passed in will play the AudioSegment out loud.
# 
# This can be helpful to check the quality of your audio files and assess any changes you need to make.
# 
# In this exercise you'll see how simple it is to use the play() function.
# 
# Remember: to use the play() function, you'll need simpleaudio or pyaudio installed for .wav files and ffmpeg for other kinds of files.

# ### code

# In[3]:


# Import AudioSegment and play
from pydub import AudioSegment
from pydub.playback import play

# Create an AudioSegment instance
wav_file = AudioSegment.from_file(file=prefix+"wav_file.wav", 
                                  format="wav")

# Play the audio file
play(wav_file)


# ## Audio parameters with PyDub
# Every audio file you work with will have a number of characteristics associated with them, such as, channels, frame rate (or sample rate), sample width and more.
# 
# Knowing these parameters is useful to ensure your audio files are compatible with various API requirements for speech transcription.
# 
# For example, many APIs recommend a minimum frame rate (wav_file.frame_rate) of 16,000 Hz.
# 
# When you create an instance of AudioSegment, PyDub automatically infers these parameters from your audio files and saves them as attributes.
# 
# In this exercise, we'll explore these attributes.

# ### code

# In[4]:


# Import audio file
wav_file = AudioSegment.from_file(file=prefix+"wav_file.wav")

# Find the frame rate
print(wav_file.frame_rate)


# In[5]:


# Find the number of channels
print(wav_file.channels)


# In[6]:


# Find the max amplitude
print(wav_file.max)


# In[7]:


# Find the length
print(len(wav_file))


# ## Adjusting audio parameters
# During your exploratory data analysis, you may find some of the parameters of your audio files differ or are incompatible with speech recognition APIs.
# 
# Don't worry, PyDub has built-in functionality which allows you to change various attributes.
# 
# For example, you can set the frame rate of your audio file calling set_frame_rate() on your AudioSegment instance and passing it an integer of the desired frame rate measured in Hertz.
# 
# In this exercise, we'll practice altering some audio attributes.
# 

# ### code

# In[9]:


# Import audio file
wav_file = AudioSegment.from_file(file=prefix+"wav_file.wav")

# Create a new wav file with adjusted frame rate
wav_file_16k = wav_file.set_frame_rate(16000)

# Check the frame rate of the new wav file
print(wav_file_16k.frame_rate)


# In[10]:


# Set number of channels to 1
wav_file_1_ch = wav_file.set_channels(1)

# Check the number of channels
print(wav_file_1_ch.channels)


# In[11]:


# Print sample_width
print(f"Old sample width: {wav_file.sample_width}")

# Set sample_width to 1
wav_file_sw_1 = wav_file.set_sample_width(1)

# Check new sample_width
print(f"New sample width: {wav_file_sw_1.sample_width}")


# In[ ]:




