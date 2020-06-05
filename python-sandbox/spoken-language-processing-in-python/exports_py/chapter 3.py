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


# # Manipulating audio files with PyDub
# 

# ## Turning it down... then up
# Speech recognition works best on clean, audible speech. If your audio files are too quiet or too loud, it can hinder transcription.
# 
# In this exercise, you'll see how to make an AudioSegment quieter or louder.
# 
# Since the play() function won't play your changes in the DataCamp classroom.
# 
# The baseline audio file, volume_adjusted.wav can be heard here.

# ### init

# In[2]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('loud_volume_adjusted.wav')
uploadToFileIO_pushto_fileio('quiet_volume_adjusted.wav')
uploadToFileIO_pushto_fileio('volume_adjusted.wav')
"""
tobedownloaded="""
{numpy.ndarray: {'loud_volume_adjusted.wav': 'https://file.io/y4XU2Fv7',
                'quiet_volume_adjusted.wav': 'https://file.io/FNCEO5vX',
                'volume_adjusted.wav': 'https://file.io/9Wx5UZ9J'}}
"""
prefixToc = '2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[3]:


from pydub import AudioSegment

# Import audio file
volume_adjusted = AudioSegment.from_file(prefix+'volume_adjusted.wav')

# Lower the volume by 60 dB
quiet_volume_adjusted = volume_adjusted - 60


# In[6]:


from pydub.playback import play
play(volume_adjusted)
play(quiet_volume_adjusted)


# In[7]:


# Increase the volume by 15 dB
louder_volume_adjusted = volume_adjusted + 15


# In[9]:


play(louder_volume_adjusted)


# ## Normalizing an audio file with PyDub
# Sometimes you'll have audio files where the speech is loud in some portions and quiet in others. Having this variance in volume can hinder transcription.
# 
# Luckily, PyDub's effects module has a function called normalize() which finds the maximum volume of an AudioSegment, then adjusts the rest of the AudioSegment to be in proportion. This means the quiet parts will get a volume boost.
# 
# You can listen to an example of an audio file which starts as loud then goes quiet, loud_then_quiet.wav, here.
# 
# In this exercise, you'll use normalize() to normalize the volume of our file, making it sound more like this.

# ### init

# In[1]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('loud_then_quiet.wav')
uploadToFileIO_pushto_fileio('normalized_loud_then_quiet.wav')
"""

tobedownloaded="""
{numpy.ndarray: {'loud_then_quiet.wav': 'https://file.io/VCzqYS75',
                'normalized_loud_then_quiet.wav': 'https://file.io/CPnzOIdI'}}
"""
prefixToc = '2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[2]:


# Import AudioSegment and normalize
from pydub import AudioSegment
from pydub.effects import normalize

# Import target audio file
loud_then_quiet = AudioSegment.from_file(prefix+'loud_then_quiet.wav')

# Normalize target audio file
normalized_loud_then_quiet = normalize(loud_then_quiet)


# ## Chopping and changing audio files
# Some of your audio files may have sections of redundancy. For example, you might find at the beginning of each file, there's a few seconds of static.
# 
# Instead of wasting compute trying to transcribe static, you can remove it.
# 
# Since an AudioSegment is iterable, and measured in milliseconds, you can use slicing to alter the length.
# 
# To get the first 3-seconds of wav_file, you'd use wav_file[:3000].
# 
# You can also add two AudioSegment's together using the addition operator. This is helpful if you need to combine several audio files.
# 
# To practice both of these, we're going to remove the first four seconds of part1.wav, and add the remainder to part2.wav. Leaving the end result sounding like part_3.wav.

# ### init

# In[3]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('part_1.wav')
uploadToFileIO_pushto_fileio('part_2.wav')
uploadToFileIO_pushto_fileio('part_3.wav')
"""
tobedownloaded="""
{numpy.ndarray: {'part_1.wav': 'https://file.io/fa11ygAd',
                'part_2.wav': 'https://file.io/SHHhlDRR',
                'part_3.wav': 'https://file.io/aSHreuCX'}}
"""
prefixToc = '2.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[4]:


from pydub import AudioSegment

# Import part 1 and part 2 audio files
part_1 = AudioSegment.from_file(prefix+'part_1.wav')
part_2 = AudioSegment.from_file(prefix+'part_2.wav')

# Remove the first four seconds of part 1
part_1_removed = part_1[4000:]

# Add the remainder of part 1 and part 2 together
part_3 = part_1_removed + part_2


# In[5]:


from pydub.playback import play
play(part_1)
play(part_2)
play(part_3)


# ## Splitting stereo audio to mono with PyDub
# If you're trying to transcribe phone calls, there's a chance they've been recorded in stereo format, with one speaker on each channel.
# 
# As you've seen, it's hard to transcribe an audio file with more than one speaker. One solution is to split the audio file with multiple speakers into single files with individual speakers.
# 
# PyDub's split_to_mono() function can help with this. When called on an AudioSegment recorded in stereo, it returns a list of two separate AudioSegment's in mono format, one for each channel.
# 
# In this exercise, you'll practice this by splitting this stereo phone call (stereo_phone_call.wav) recording into channel 1 and channel 2. This separates the two speakers, allowing for easier transcription.

# ### init

# In[22]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('channel_1.wav')
uploadToFileIO_pushto_fileio('channel_2.wav')
uploadToFileIO_pushto_fileio('stereo_phone_call.wav')
"""

tobedownloaded="""
{numpy.ndarray: {'channel_1.wav': 'https://file.io/1a9blKUP',
                'channel_2.wav': 'https://file.io/ScV8smSp',
                'stereo_phone_call.wav': 'https://file.io/MqVMM0wB'}}
"""
prefixToc = '2.4'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ###  code

# In[24]:


# Import AudioSegment
from pydub import AudioSegment

# Import stereo audio file and check channels
stereo_phone_call = AudioSegment.from_file(prefix+'stereo_phone_call.wav')
print(f"Stereo number channels: {stereo_phone_call.channels}")

# Split stereo phone call and check channels
channels = stereo_phone_call.split_to_mono()
print(f"Split number channels: {channels[0].channels}, {channels[1].channels}")

# Save new channels separately
phone_call_channel_1 = channels[0]
phone_call_channel_2 = channels[1]


# # Converting and saving audio files with PyDub
# 

# ## Exporting and reformatting audio files
# If you've made some changes to your audio files, or if they've got the wrong file extension, you can use PyDub to export and save them as new audio files.
# 
# You can do this by using the .export() function on any instance of an AudioSegment you've created. The export() function takes two parameters, out_f, or the destination file path of your audio file and format, the format you'd like your new audio file to be. Both of these are strings. format is "mp3" by default so be sure to change it if you need.
# 
# In this exercise, you'll import this .mp3 file (mp3_file.mp3) and then export it with the .wav extension using .export().
# 
# Remember, to work with files other than .wav, you'll need ffmpeg.

# ### init

# In[25]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('mp3_file.mp3')
"""

tobedownloaded="""
{numpy.ndarray: {'mp3_file.mp3': 'https://file.io/ub6XXC2Q'}}
"""
prefixToc = '3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[26]:


from pydub import AudioSegment

# Import the .mp3 file
mp3_file = AudioSegment.from_file(prefix+'mp3_file.mp3')

# Export the .mp3 file as wav
mp3_file.export(out_f=pefix+'mp3_file.wav',
                format='wav')


# ## Manipulating multiple audio files with PyDub
# You've seen how to convert a single file using PyDub but what if you had a folder with multiple different file types?
# 
# For this exercise, we've setup a folder which has .mp3, .m4a and .aac versions of the good-afternoon audio file.
# 
# We'll use PyDub to open each of the files and export them as .wav format so they're compatible with speech recognition APIs.

# ### init

# In[27]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('good_afternoon_aac.aac')
uploadToFileIO_pushto_fileio('good_afternoon_m4a.m4a')
uploadToFileIO_pushto_fileio('good_afternoon_mp3.mp3')

{"success":true,"key":"2tJ7yoQw","link":"https://file.io/2tJ7yoQw","expiry":"14 days"}
{"success":true,"key":"4KXZgu7V","link":"https://file.io/4KXZgu7V","expiry":"14 days"}
{"success":true,"key":"CPLuMuGc","link":"https://file.io/CPLuMuGc","expiry":"14 days"}
"""

tobedownloaded="""
{numpy.ndarray: {'good_afternoon_aac.aac': 'https://file.io/2tJ7yoQw',
                'good_afternoon_m4a.m4a': 'https://file.io/4KXZgu7V',
                'good_afternoon_mp3.mp3': 'https://file.io/CPLuMuGc'}}
"""
prefixToc = '3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# In[30]:


folder = [prefix+'good_afternoon_mp3.mp3', prefix+'good_afternoon_m4a.m4a', prefix+'good_afternoon_aac.aac']


import os


# ### code

# In[32]:


# Loop through the files in the folder
for audio_file in folder:
    
	# Create the new .wav filename
    wav_filename = os.path.splitext(os.path.basename(audio_file))[0] + ".wav"
        
    # Read audio_file and export it in wav format
    AudioSegment.from_file(audio_file).export(out_f=wav_filename, 
                                      format='wav')
        
    print(f"Creating {wav_filename}...")


# ## An audio processing workflow
# You've seen how to import and manipulate a single audio file using PyDub. But what if you had a folder with multiple audio files you needed to convert?
# 
# In this exercise we'll use PyDub to format a folder of files to be ready to use with speech_recognition.
# 
# You've found your customer call files all have 3-seconds of static at the start and are quieter than they could be.
# 
# To fix this, we'll use PyDub to cut the static, increase the sound level and convert them to the .wav extension.
# 
# You can listen to an unformatted example here.

# ### init

# In[36]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('account_help.mp3')
uploadToFileIO_pushto_fileio('order_issue.mp3')
uploadToFileIO_pushto_fileio('out_warranty.mp3')
uploadToFileIO_pushto_fileio('make_complaint.mp3')
uploadToFileIO_pushto_fileio('order_number.mp3')
uploadToFileIO_pushto_fileio('speech_recognition.mp3')

{"success":true,"key":"tdQo36Cj","link":"https://file.io/tdQo36Cj","expiry":"14 days"}
{"success":true,"key":"JMi1X1qY","link":"https://file.io/JMi1X1qY","expiry":"14 days"}
{"success":true,"key":"s9ntbUiz","link":"https://file.io/s9ntbUiz","expiry":"14 days"}
{"success":true,"key":"2jRxX7dI","link":"https://file.io/2jRxX7dI","expiry":"14 days"}
{"success":true,"key":"IZjZH3dw","link":"https://file.io/IZjZH3dw","expiry":"14 days"}
{"success":true,"key":"9DFmsmNq","link":"https://file.io/9DFmsmNq","expiry":"14 days"}


"""

tobedownloaded="""
{numpy.ndarray: {'account_help.mp3': 'https://file.io/tdQo36Cj',
                'order_issue.mp3': 'https://file.io/JMi1X1qY',
                'out_warranty.mp3': 'https://file.io/s9ntbUiz',
                'make_complaint.mp3': 'https://file.io/2jRxX7dI',
                'order_number.mp3': 'https://file.io/IZjZH3dw',
                'speech_recognition.mp3': 'https://file.io/9DFmsmNq'
                }}
"""
prefixToc = '3.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")


# ### code

# In[37]:


file_with_static = AudioSegment.from_file(prefix+'account_help.mp3')

# Cut the first 3-seconds of static off
file_without_static = file_with_static[3000:]


# In[38]:


# Increase the volume by 10dB
louder_file_without_static = file_without_static + 10


# In[39]:


folder = ['account_help.mp3',
 'make_complaint.mp3',
 'order_issue.mp3',
 'order_number.mp3',
 'out_warranty.mp3',
 'speech_recognition.mp3']


# In[41]:


for audio_file in folder:
    file_with_static = AudioSegment.from_file(prefix+audio_file)

    # Cut the 3-seconds of static off
    file_without_static = file_with_static[3000:]

    # Increase the volume by 10dB
    louder_file_without_static = file_without_static + 10
    
    # Create the .wav filename for export
    wav_filename = os.path.splitext(os.path.basename(audio_file))[0] + ".wav"
    
    # Export the louder file without static as .wav
    louder_file_without_static.export(wav_filename, format='wav')
    print(f"Creating {wav_filename}...")


# In[ ]:




