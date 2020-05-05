#!/usr/bin/env python
# coding: utf-8

# # SpeechRecognition Python library
# 

# ## sing the SpeechRecognition library
# To save typing speech_recognition every time, we'll import it as sr.
# 
# We'll also setup an instance of the Recognizer class to use later.
# 
# The energy_threshold is a number between 0 and 4000 for how much the Recognizer class should listen to an audio file.
# 
# energy_threshold will dynamically adjust whilst the recognizer class listens to audio.

# ### code

# In[1]:


# Importing the speech_recognition library
import speech_recognition as sr

# Create an instance of the Recognizer class
recognizer = sr.Recognizer()

# Set the energy threshold
recognizer.energy_threshold = 300


# ## Using the Recognizer class
# Now you've created an instance of the Recognizer class we'll use the recognize_google() method on it to access the Google web speech API and turn spoken language into text.
# 
# recognize_google() requires an argument audio_data otherwise it will return an error.
# 
# US English is the default language. If your audio file isn't in US English, you can change the language with the language argument. A list of language codes can be seen here.
# 
# An audio file containing English speech has been imported as clean_support_call_audio. You can listen to the audio file here. SpeechRecognition has also been imported as sr.
# 
# To avoid hitting the API request limit of Google's web API, we've mocked the Recognizer class to work with our audio files. This means some functionality will be limited.

# ### init

# In[2]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('clean_support_call.wav')
"""

tobedownloaded="""
{numpy.ndarray: {'clean_support_call.wav': 'https://file.io/d6HVQN1G'}}
"""
prefixToc = '1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# In[11]:


# Create a recognizer class
recognizer = sr.Recognizer()

# Read audio data
with sr.AudioFile(prefix+'clean_support_call.wav') as source:
    clean_support_call_audio = recognizer.record(source)  # read the entire audio file


# ### code

# In[12]:


# Create a recognizer class
recognizer = sr.Recognizer()

# Transcribe the support call audio
text = recognizer.recognize_google(
  audio_data=clean_support_call_audio, 
  language='en-US')

print(text)


# ### test avec enregistrement micro

# https://stackabuse.com/introduction-to-speech-recognition-with-python/
# 
# conda install PyAudio

# In[15]:


mic = sr.Microphone()


# In[16]:


sr.Microphone.list_microphone_names()


# In[18]:


with mic as audio_file:
    print("Speak Please")

    recognizer.adjust_for_ambient_noise(audio_file)
    audio = recognizer.listen(audio_file)

    print("fin d'enregistrement...")


# In[19]:


type(audio)


# In[38]:


with mic as audio_file:
    print("Parlez svp")

    recognizer.adjust_for_ambient_noise(audio_file)
    audio = recognizer.listen(audio_file)

    print("Converting Speech to Text...")

    try:
        print("You said: " + recognizer.recognize_google(audio, 'en-US'))
    except Exception as e:
        print("Error: " + str(e))


# # Reading audio files with SpeechRecognition
# 

# ## From AudioFile to AudioData
# As you saw earlier, there are some transformation steps we have to take to make our audio data useful. The same goes for SpeechRecognition.
# 
# In this exercise, we'll import the clean_support_call.wav audio file and get it ready to be recognized.
# 
# We first read our audio file using the AudioFile class. But the recognize_google() method requires an input of type AudioData.
# 
# To convert our AudioFile to AudioData, we'll use the Recognizer class's method record() along with a context manager. The record() method takes an AudioFile as input and converts it to AudioData, ready to be used with recognize_google().
# 
# SpeechRecognition has already been imported as sr.

# ### code

# In[22]:


# Instantiate Recognizer
recognizer = sr.Recognizer()

# Convert audio to AudioFile
clean_support_call = sr.AudioFile(prefix+'clean_support_call.wav')

# Convert AudioFile to AudioData
with clean_support_call as source:
    clean_support_call_audio = recognizer.record(source)

# Transcribe AudioData to text
text = recognizer.recognize_google(clean_support_call_audio,
                                   language="en-US")
print(text)


# ## Recording the audio we need
# Sometimes you may not want the entire audio file you're working with. The duration and offset parameters of the record() method can help with this.
# 
# After exploring your dataset, you find there's one file, imported as nothing_at_end which has 30-seconds of silence at the end and a support call file, imported as out_of_warranty has 3-seconds of static at the front.
# 
# Setting duration and offset means the record() method will record up to duration audio starting at offset. They're both measured in seconds.

# ### init

# In[23]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('30_seconds_of_nothing.wav')
uploadToFileIO_pushto_fileio('static_out_of_warranty.wav')

{"success":true,"key":"4NjpePKk","link":"https://file.io/4NjpePKk","expiry":"14 days"}
{"success":true,"key":"gcaUrCI7","link":"https://file.io/gcaUrCI7","expiry":"14 days"}

"""

tobedownloaded="""
{numpy.ndarray: {'30_seconds_of_nothing.wav': 'https://file.io/4NjpePKk', 
'static_out_of_warranty.wav': 'https://file.io/gcaUrCI7'}}
"""
prefixToc = '2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# In[30]:


def get_audio_from_wav(filename):
    source_wav = sr.AudioFile(filename)
    with source_wav as source:
        return recognizer.record(source)


# In[24]:


# Convert audio to AudioFile
nothing_at_end_wav = sr.AudioFile(prefix+'30_seconds_of_nothing.wav')
# Convert AudioFile to AudioData
with nothing_at_end_wav as source:
    nothing_at_end = recognizer.record(source)

# Convert audio to AudioFile
out_of_warranty_wav = sr.AudioFile(prefix+'static_out_of_warranty.wav')
# Convert AudioFile to AudioData
with out_of_warranty_wav as source:
    out_of_warranty = recognizer.record(source)


# ### code

# In[25]:


# Convert AudioFile to AudioData
with nothing_at_end as source:
    nothing_at_end_audio = recognizer.record(source,
                                             duration=10,
                                             offset=None)

# Transcribe AudioData to text
text = recognizer.recognize_google(nothing_at_end_audio,
                                   language="en-US")

print(text)


# In[28]:


# Convert AudioFile to AudioData
with out_of_warranty as source:
    static_art_start_audio = recognizer.record(source,
                                               duration=None,
                                               offset=3)

# Transcribe AudioData to text
text = recognizer.recognize_google(static_art_start_audio,
                                   language="en-US")

print(text)


# # Dealing with different kinds of audio
# 

# ## Different kinds of audio
# Now you've seen an example of how the Recognizer class works. Let's try a few more. How about speech from a different language?
# 
# What do you think will happen when we call the recognize_google() function on a Japanese version of good_morning.wav (japanese_audio)?
# 
# The default language is "en-US", are the results the same with the "ja" tag?
# 
# How about non-speech audio? Like this leopard roaring (leopard_audio).
# 
# Or speech where the sounds may not be real words, such as a baby talking (charlie_audio)?
# 
# To familiarize more with the Recognizer class, we'll look at an example of each of these.

# ### init

# In[29]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('charlie.wav')
uploadToFileIO_pushto_fileio('japanese_good_morning.wav')
uploadToFileIO_pushto_fileio('leopard.wav')

{"success":true,"key":"aOAxAIEu","link":"https://file.io/aOAxAIEu","expiry":"14 days"}
{"success":true,"key":"IIe7UoDf","link":"https://file.io/IIe7UoDf","expiry":"14 days"}
{"success":true,"key":"8o17be2D","link":"https://file.io/8o17be2D","expiry":"14 days"}

"""

tobedownloaded="""
{numpy.ndarray: {'charlie.wav': 'https://file.io/aOAxAIEu', 
'japanese_good_morning.wav': 'https://file.io/IIe7UoDf', 
'leopard.wav': 'https://file.io/8o17be2D'}}
"""
prefixToc = '3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# In[32]:


japanese_audio = get_audio_from_wav(prefix+'japanese_good_morning.wav')
charlie_audio =  get_audio_from_wav(prefix+'charlie.wav')
leopard_audio = get_audio_from_wav(prefix+'leopard.wav')


# ### code

# In[33]:


# Create a recognizer class
recognizer = sr.Recognizer()

# Pass the Japanese audio to recognize_google
text = recognizer.recognize_google(japanese_audio, language='en-US')

# Print the text
print(text)


# In[34]:


# Create a recognizer class
recognizer = sr.Recognizer()

# Pass the Japanese audio to recognize_google
text = recognizer.recognize_google(japanese_audio, language='ja')

# Print the text
print(text)


# In[35]:


# Create a recognizer class
recognizer = sr.Recognizer()

# Pass the leopard roar audio to recognize_google
text = recognizer.recognize_google(leopard_audio, 
                                   language="en-US", 
                                   show_all=True)

# Print the text
print(text)


# In[36]:


# Create a recognizer class
recognizer = sr.Recognizer()

# Pass charlie_audio to recognize_google
text = recognizer.recognize_google(charlie_audio, 
                                   language="en-US")

# Print the text
print(text)


# ## Multiple Speakers 1
# If your goal is to transcribe conversations, there will be more than one speaker. However, as you'll see, the recognize_google() function will only transcribe speech into a single block of text.
# 
# You can hear in this audio file there are three different speakers.
# 
# But if you transcribe it on its own, recognize_google() returns a single block of text. Which is still useful but it doesn't let you know which speaker said what.
# 
# We'll see an alternative to this in the next exercise.
# 
# The multiple speakers audio file has been imported and converted to AudioData as multiple_speakers.

# ### init

# In[40]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('multiple_speakers.wav')


{"success":true,"key":"MDkwS70q","link":"https://file.io/MDkwS70q","expiry":"14 days"}

"""

tobedownloaded="""
{numpy.ndarray: {'multiple_speakers.wav': 'https://file.io/MDkwS70q'}}
"""
prefixToc = '3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# In[41]:


multiple_speakers = get_audio_from_wav(prefix+'multiple_speakers.wav')


# ### code

# In[42]:


# Create a recognizer class
recognizer = sr.Recognizer()

# Recognize the multiple speaker AudioData
text = recognizer.recognize_google(multiple_speakers, 
                       			   language='en-US')

# Print the text
print(text)


# ## Multiple Speakers 2
# Deciphering between multiple speakers in one audio file is called speaker diarization. However, you've seen the free function we've been using, recognize_google() doesn't have the ability to transcribe different speakers.
# 
# One way around this, without using one of the paid speech to text services, is to ensure your audio files are single speaker.
# 
# This means if you were working with phone call data, you would make sure the caller and receiver are recorded separately. Then you could transcribe each file individually.
# 
# In this exercise, we'll transcribe each of the speakers in our multiple speakers audio file individually.

# ### init

# In[44]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('speaker_0.wav')
uploadToFileIO_pushto_fileio('speaker_1.wav')
uploadToFileIO_pushto_fileio('speaker_2.wav')

{"success":true,"key":"okzGvNrJ","link":"https://file.io/okzGvNrJ","expiry":"14 days"}
{"success":true,"key":"JiZRvwHs","link":"https://file.io/JiZRvwHs","expiry":"14 days"}
{"success":true,"key":"CrbOlD0x","link":"https://file.io/CrbOlD0x","expiry":"14 days"}

"""

tobedownloaded="""
{numpy.ndarray: {'speaker_0.wav': 'https://file.io/okzGvNrJ', 
'speaker_1.wav': 'https://file.io/JiZRvwHs', 
'speaker_2.wav': 'https://file.io/CrbOlD0x'}}
"""
prefixToc = '3.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[45]:


recognizer = sr.Recognizer()

# Multiple speakers on different files
speakers = [sr.AudioFile(prefix+"speaker_0.wav"), 
            sr.AudioFile(prefix+"speaker_1.wav"), 
            sr.AudioFile(prefix+"speaker_2.wav")]

# Transcribe each speaker individually
for i, speaker in enumerate(speakers):
    with speaker as source:
        speaker_audio = recognizer.record(source)
    print(f"Text from speaker {i}:")
    print(recognizer.recognize_google(speaker_audio,
         				  language="en-US"))


# ## Working with noisy audio
# In this exercise, we'll start by transcribing a clean speech sample to text and then see what happens when we add some background noise.
# 
# A clean audio sample has been imported as clean_support_call.
# 
# Play clean support call.
# 
# We'll then do the same with the noisy audio file saved as noisy_support_call. It has the same speech as clean_support_call but with additional background noise.
# 
# Play noisy support call.
# 
# To try and negate the background noise, we'll take advantage of Recognizer's adjust_for_ambient_noise() function.

# ### init

# In[46]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('clean_support_call.wav')
uploadToFileIO_pushto_fileio('noisy_support_call.wav')

{"success":true,"key":"W1i13sVb","link":"https://file.io/W1i13sVb","expiry":"14 days"}
{"success":true,"key":"2xLA74St","link":"https://file.io/2xLA74St","expiry":"14 days"}

"""

tobedownloaded="""
{numpy.ndarray: {'clean_support_call.wav': 'https://file.io/W1i13sVb', 
'noisy_support_call.wav': 'https://file.io/2xLA74St'}}
"""
prefixToc = '3.4'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[47]:


clean_support_call = sr.AudioFile(prefix+"clean_support_call.wav")
noisy_support_call = sr.AudioFile(prefix+"noisy_support_call.wav")


# In[50]:


recognizer = sr.Recognizer()

# Record the audio from the clean support call
with clean_support_call as source:
  clean_support_call_audio = recognizer.record(source)

# Transcribe the speech from the clean support call
text = recognizer.recognize_google(clean_support_call_audio,
					   language="en-US")

print(text)


# In[51]:


recognizer = sr.Recognizer()

# Record the audio from the noisy support call
with noisy_support_call as source:
  noisy_support_call_audio = recognizer.record(source)

# Transcribe the speech from the noisy support call
text = recognizer.recognize_google(noisy_support_call_audio,
                         language="en-US",
                         show_all=True)

print(text)


# In[52]:


recognizer = sr.Recognizer()

# Record the audio from the noisy support call
with noisy_support_call as source:
	# Adjust the recognizer energy threshold for ambient noise
    recognizer.adjust_for_ambient_noise(source, duration=1)
    noisy_support_call_audio = recognizer.record(noisy_support_call)
 
# Transcribe the speech from the noisy support call
text = recognizer.recognize_google(noisy_support_call_audio,
                                   language="en-US",
                                   show_all=True)

print(text)


# In[53]:


recognizer = sr.Recognizer()

# Record the audio from the noisy support call
with noisy_support_call as source:
	# Adjust the recognizer energy threshold for ambient noise
    recognizer.adjust_for_ambient_noise(source, duration=0.5)
    noisy_support_call_audio = recognizer.record(noisy_support_call)
 
# Transcribe the speech from the noisy support call
text = recognizer.recognize_google(noisy_support_call_audio,
                                   language="en-US",
                                   show_all=True)

print(text)


# In[ ]:




