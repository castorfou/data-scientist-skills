#!/usr/bin/env python
# coding: utf-8

# # Introduction to audio data in Python
# 

# ## Importing an audio file with Python
# You've seen how there are different kinds of audio files and how streaming music and spoken language have different sampling rates. But now we want to start working with these files.
# 
# To begin, we're going to import the good_morning.wav audio file using Python's in-built wave library. Then we'll see what it looks like in byte form using the built-in readframes() method.
# 
# You can listen to good_morning.wav here.
# 
# Remember, good_morning.wav is only a few seconds long but at 48 kHz, that means it contains 48,000 pieces of information per second.

# ### init

# In[1]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('good_morning.wav')
"""

tobedownloaded="""
{numpy.ndarray: {'good_morning.wav': 'https://file.io/bE49PYiU'}}
"""
prefixToc = '1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[2]:


import wave

# Create audio file wave object
good_morning = wave.open(prefix+'good_morning.wav', 'r')

# Read all frames from wave object 
signal_gm = good_morning.readframes(-1)

# View first 10
print(signal_gm[:10])


# # Converting sound wave bytes to integers
# 

# ## Bytes to integers
# You've seen how to import and read an audio file using Python's wave module and the readframes() method. But doing that results in an array of bytes.
# 
# To convert the bytes into something more useful, we'll use NumPy's frombuffer() method.
# 
# Passing frombuffer() our sound waves bytes and indicating a dtype of 'int16', we can convert our bytes to integers. Integers are much easier to work with than bytes.
# 
# The Python wave library has already been imported along with the good_morning.wav audio file.

# ### code

# In[3]:


import numpy as np

# Open good morning sound wave and read frames as bytes
good_morning = wave.open(prefix+'good_morning.wav', 'r')
signal_gm = good_morning.readframes(-1)

# Convert good morning audio bytes to integers
soundwave_gm = np.frombuffer(signal_gm, dtype='int16')

# View the first 10 sound wave values
print(soundwave_gm[:10])


# ## Finding the time stamps
# We know the frequency of our sound wave is 48 kHz, but what if we didn't? We could find it by dividing the length of our sound wave array by the duration of our sound wave. However, Python's wave module has a better way. Calling getframerate() on a wave object returns the frame rate of that wave object.
# 
# We can then use NumPy's linspace() method to find the time stamp of each integer in our sound wave array. This will help us visualize our sound wave in the future.
# 
# The linspace() method takes start, stop and num parameters and returns num evenly spaced values between start and stop.
# 
# In our case, start will be zero, stop will be the length of our sound wave array over the frame rate (or the duration of our audio file) and num will be the length of our sound wave array.

# ### code

# In[4]:


# Read in sound wave and convert from bytes to integers
good_morning = wave.open(prefix+'good_morning.wav', 'r')
signal_gm = good_morning.readframes(-1)
soundwave_gm = np.frombuffer(signal_gm, dtype='int16')

# Get the sound wave frame rate
framerate_gm = good_morning.getframerate()

# Find the sound wave timestamps
time_gm = np.linspace(start=0,
                      stop=len(soundwave_gm)/framerate_gm,
					  num=len(soundwave_gm))

# Print the first 10 timestamps
print(time_gm[:10])


# In[7]:


petit_duc = wave.open('petit-duc.wav', 'r')
signal_pd = petit_duc.readframes(-1)
soundwave_pd = np.frombuffer(signal_pd, dtype='int16')
framerate_pd = petit_duc.getframerate()
print('Frequence d\'enregistrement du petit duc : '+str(framerate_pd))


# # Visualizing sound waves
# 

# ## Processing audio data with Python
# You've seen how a sound waves can be turned into numbers but what does all that conversion look like?
# 
# And how about another similar sound wave? One slightly different?
# 
# In this exercise, we're going to use MatPlotLib to plot the sound wave of good_morning against good_afternoon.
# 
# To have the good_morning and good_afternoon sound waves on the same plot and distinguishable from each other, we'll use MatPlotLib's alpha parameter.
# 
# You can listen to the good_morning audio here and good_afternoon audio here.

# ### init

# In[10]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('good-afternoon.wav')
"""

tobedownloaded="""
{numpy.ndarray: {'good-afternoon.wav': 'https://file.io/A1j6pvtC'}}
"""
prefixToc = '3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


from downloadfromFileIO import getPrefixfromTOC
prefix_afternoon = getPrefixfromTOC('3.1')
prefix_morning = getPrefixfromTOC('1.1')


# In[15]:


# Open good morning sound wave and read frames as bytes
good_morning = wave.open(prefix_morning+'good_morning.wav', 'r')
signal_gm = good_morning.readframes(-1)
# Convert good morning audio bytes to integers
soundwave_gm = np.frombuffer(signal_gm, dtype='int16')
# Get the sound wave frame rate
framerate_gm = good_morning.getframerate()
# Find the sound wave timestamps
time_gm = np.linspace(start=0,
                      stop=len(soundwave_gm)/framerate_gm,
					  num=len(soundwave_gm))

# Open sound wave and read frames as bytes
good_afternoon = wave.open(prefix_afternoon+'good-afternoon.wav', 'r')
signal_ga = good_afternoon.readframes(-1)
# Convert audio bytes to integers
soundwave_ga = np.frombuffer(signal_ga, dtype='int16')
# Get the sound wave frame rate
framerate_ga = good_afternoon.getframerate()
# Find the sound wave timestamps
time_ga = np.linspace(start=0,
                      stop=len(soundwave_ga)/framerate_ga,
					  num=len(soundwave_ga))

import matplotlib.pyplot as plt


# ### code

# In[16]:


# Setup the title and axis titles
plt.title('Good Afternoon vs. Good Morning')
plt.ylabel('Amplitude')
plt.xlabel('Time (seconds)')

# Add the Good Afternoon data to the plot
plt.plot(time_ga, soundwave_ga, label='Good Afternoon')

# Add the Good Morning data to the plot
plt.plot(time_gm, soundwave_gm, label='Good Morning',
   # Set the alpha variable to 0.5
   alpha=0.5)

plt.legend()
plt.show()


# ### visu petit duc

# In[17]:


# Open sound wave and read frames as bytes
petit_duc = wave.open('petit-duc.wav', 'r')
signal_pd = petit_duc.readframes(-1)
# Convert audio bytes to integers
soundwave_pd = np.frombuffer(signal_pd, dtype='int16')
# Get the sound wave frame rate
framerate_pd = petit_duc.getframerate()
# Find the sound wave timestamps
time_pd = np.linspace(start=0,
                      stop=len(soundwave_pd)/framerate_pd,
					  num=len(soundwave_pd))


# In[18]:


# Setup the title and axis titles
plt.title('Petit Duc')
plt.ylabel('Amplitude')
plt.xlabel('Time (seconds)')

# Add the Good Afternoon data to the plot
plt.plot(time_pd, soundwave_pd, label='Petit Duc')


plt.legend()
plt.show()


# In[19]:


soundwave_pd


# In[20]:


import pandas as pd
pandas_pd = pd.DataFrame(soundwave_pd)


# In[22]:


pandas_pd.describe()


# In[33]:


threeshold = 100
soundwave_pd_filtered = soundwave_pd.copy()
soundwave_pd_filtered[(soundwave_pd_filtered < threeshold) & (soundwave_pd_filtered > -threeshold)] =0


# In[34]:


# Setup the title and axis titles
plt.title('Petit Duc')
plt.ylabel('Amplitude')
plt.xlabel('Time (seconds)')

# Add the Good Afternoon data to the plot
plt.plot(time_pd, soundwave_pd_filtered, label='Petit Duc')


plt.legend()
plt.show()


# In[38]:


from scipy.io.wavfile import write
samplerate = 44100; 
write("petit_duc_filtré.wav", samplerate, soundwave_pd_filtered)


# In[37]:


framerate_pd


# In[ ]:




