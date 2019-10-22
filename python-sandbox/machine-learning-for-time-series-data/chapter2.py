# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:59:47 2019

@author: F279814
"""

#%% Exercise - Many repetitions of sounds - init

#import inspect
#print(inspect.getsource(show_plot_and_make_titles))

def show_plot_and_make_titles():
   axs[0, 0].set(title="Normal Heartbeats")
   axs[0, 1].set(title="Abnormal Heartbeats")
   plt.tight_layout()
   plt.show()

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(normal, abnormal)
tobedownloaded="{pandas.core.frame.DataFrame: {'abnormal.csv': 'https://file.io/Q9G3vd',  'normal.csv': 'https://file.io/rEwjIg'}}"
prefix='data_from_datacamp/ZZZ_Chap21_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

normal=pd.read_csv(prefix+'normal.csv',index_col=0)
abnormal=pd.read_csv(prefix+'abnormal.csv',index_col=0)
sfreq=2205

#%% Exercise - Many repetitions of sounds

fig, axs = plt.subplots(3, 2, figsize=(15, 7), sharex=True, sharey=True)

# Calculate the time array
time = np.arange(normal.shape[0]) / sfreq

# Stack the normal/abnormal audio so you can loop and plot
stacked_audio = np.hstack([normal, abnormal]).T

# Loop through each audio file / ax object and plot
# .T.ravel() transposes the array, then unravels it into a 1-D vector for looping
for iaudio, ax in zip(stacked_audio, axs.T.ravel()):
    ax.plot(time, iaudio)
show_plot_and_make_titles()

#%% Exercise - Invariance in time - init


import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from uploadfromdatacamp import saveFromFileIO
from uploadfromdatacamp import loadNDArrayFromCsv

prefix_old='data_from_datacamp/ZZZ_Chap21_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

normal=pd.read_csv(prefix_old+'normal.csv',index_col=0)
abnormal=pd.read_csv(prefix_old+'abnormal.csv',index_col=0)

#uploadToFileIO(time)
tobedownloaded="{numpy.ndarray: {'time.csv': 'https://file.io/8g5yBo'}}"
prefix='data_from_datacamp/ZZZ_Chap22_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

time=loadNDArrayFromCsv(prefix+'time.csv')


#%% Exercise - Invariance in time

# Average across the audio files of each DataFrame
mean_normal = np.mean(normal, axis=1)
mean_abnormal = np.mean(abnormal, axis=1)

# Plot each average over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
ax1.plot(time, mean_normal)
ax1.set(title="Normal Data")
ax2.plot(time, mean_abnormal)
ax2.set(title="Abnormal Data")
plt.show()

#%% Exercise - Build a classification model - init

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from uploadfromdatacamp import saveFromFileIO
from uploadfromdatacamp import loadNDArrayFromCsv

#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{numpy.ndarray: {'X_test.csv': 'https://file.io/5ONvUO', 'X_train.csv': 'https://file.io/iKq5GT'}}"
prefix='data_from_datacamp/ZZZ_Chap23_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

X_train=loadNDArrayFromCsv(prefix+'X_train.csv')
X_test=loadNDArrayFromCsv(prefix+'X_test.csv')

tobedownloaded="{numpy.ndarray: {'y_train.csv': 'https://file.io/YP9Y3d', 'y_test.csv': 'https://file.io/8ouPaK'}}"
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

y_train=loadNDArrayFromCsv(prefix+'y_train.csv', dtype='<U8')
y_test=loadNDArrayFromCsv(prefix+'y_test.csv', dtype='<U8')


#%% Exercise - Build a classification model

from sklearn.svm import LinearSVC

# Initialize and fit the model
model = LinearSVC()
model.fit(X_train, y_train)

# Generate predictions and score them manually
predictions = model.predict(X_test)
print(sum(predictions == y_test.squeeze()) / len(y_test))


#%% Exercise - Calculating the envelope of sound - init

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(audio)
tobedownloaded="{pandas.core.series.Series: {'audio.csv': 'https://file.io/ZLRIHE'}}"
prefix='data_from_datacamp/ZZZ_Chap24_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

audio=pd.read_csv(prefix+'audio.csv', index_col=0, header=None,squeeze=True)


#%% Exercise - Calculating the envelope of sound

# Plot the raw data first
audio.plot(figsize=(10, 5))
plt.show()

# Rectify the audio signal
audio_rectified = audio.apply(np.abs)

# Plot the result
audio_rectified.plot(figsize=(10, 5))
plt.show()

# Smooth by applying a rolling mean
audio_rectified_smooth = audio_rectified.rolling(50).mean()

# Plot the result
audio_rectified_smooth.plot(figsize=(10, 5))
plt.show()

#%% Exercise - Calculating features from the envelope - init

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from uploadfromdatacamp import saveFromFileIO
from uploadfromdatacamp import loadNDArrayFromCsv
from sklearn.svm import LinearSVC

#uploadToFileIO(labels)
tobedownloaded="{numpy.ndarray: {'labels.csv': 'https://file.io/YHcJpx'}}"
prefix='data_from_datacamp/ZZZ_Chap25_'

#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

labels=loadNDArrayFromCsv(prefix+'labels.csv', dtype='<U8')
model=LinearSVC()

#uploadToFileIO(audio_rectified_smooth)
tobedownloaded="{pandas.core.frame.DataFrame: {'audio_rectified_smooth.csv': 'https://file.io/yGeyEV'}}"
prefix='data_from_datacamp/ZZZ_Chap25_'

#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

audio_rectified_smooth=pd.read_csv(prefix+'audio_rectified_smooth.csv', index_col=0)

#uploadToFileIO(audio)
tobedownloaded="{pandas.core.frame.DataFrame: {'audio.csv': 'https://file.io/tHgHiM'}}"
prefix='data_from_datacamp/ZZZ_Chap25_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#%% Exercise - Calculating features from the envelope

# Calculate stats
means = np.mean(audio_rectified_smooth, axis=0)
stds = np.std(audio_rectified_smooth, axis=0)
maxs = np.max(audio_rectified_smooth, axis=0)

# Create the X and y arrays
X = np.column_stack([means, stds, maxs])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
from sklearn.model_selection import cross_val_score
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

#%% Exercise - Derivative features: The tempogram - init

import pandas as pd
import numpy as np
from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(audio)
tobedownloaded="{pandas.core.frame.DataFrame: {'audio.csv': 'https://file.io/mC0lPn'}}"
prefix='data_from_datacamp/ZZZ_Chap26_'

#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

audio=pd.read_csv(prefix+'audio.csv',index_col=0)

import librosa as lr
sfreq=2205


#%% Exercise - Derivative features: The tempogram

# Calculate the tempo of the sounds
tempos = []
for col, i_audio in audio.items():
    tempos.append(lr.beat.tempo(i_audio.values, sr=sfreq, hop_length=2**6, aggregate=None))

# Convert the list to an array so you can manipulate it more easily
tempos = np.array(tempos)

# Calculate statistics of each tempo
tempos_mean = tempos.mean(axis=-1)
tempos_std = tempos.std(axis=-1)
tempos_max = tempos.max(axis=-1)

# Create the X and y arrays
X = np.column_stack([means, stds, maxs, tempos_mean,tempos_std, tempos_max])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))


#%% Exercise - Spectrograms of heartbeat audio - init

from uploadfromdatacamp import saveFromFileIO, loadNDArrayFromCsv
import matplotlib.pyplot as plt

#uploadToFileIO(audio, time)
tobedownloaded="{numpy.ndarray: {'audio.csv': 'https://file.io/HtFYDY','time.csv': 'https://file.io/qDvUW1'}}"
prefix='data_from_datacamp/ZZZ_Chap27_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

audio=loadNDArrayFromCsv(prefix+'audio.csv')
time=loadNDArrayFromCsv(prefix+'time.csv')
sfreq=2205

#%% Exercise - Spectrograms of heartbeat audio

# Import the stft function
from librosa.core import stft

# Prepare the STFT
HOP_LENGTH = 2**4
spec = stft(audio, hop_length=HOP_LENGTH, n_fft=2**7)

from librosa.core import amplitude_to_db
from librosa.display import specshow

# Convert into decibels
spec_db = amplitude_to_db(spec)

# Compare the raw audio to the spectrogram of the audio
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axs[0].plot(time, audio)
specshow(spec_db, sr=sfreq, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
plt.show()

#%% Exercise - Engineering spectral features - init

import numpy as np

from uploadfromdatacamp import saveFromFileIO, loadNDArrayFromCsv
import matplotlib.pyplot as plt

#uploadToFileIO(spec,times_spec )
tobedownloaded="{numpy.ndarray: {'spec.csv': 'https://file.io/GqqSoz',  'times_spec.csv': 'https://file.io/xMyQno'}}"
prefix='data_from_datacamp/ZZZ_Chap28_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

spec=loadNDArrayFromCsv(prefix+'spec.csv')
times_spec=loadNDArrayFromCsv(prefix+'times_spec.csv')
sfreq=2205


#%% Exercise - Engineering spectral features 

import librosa as lr

# Calculate the spectral centroid and bandwidth for the spectrogram
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]

from librosa.core import amplitude_to_db
from librosa.display import specshow

# Convert spectrogram to decibels for visualization
spec_db = amplitude_to_db(spec)

# Display these features on top of the spectrogram
fig, ax = plt.subplots(figsize=(10, 5))
ax = specshow(spec_db, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
ax.plot(times_spec, centroids)
ax.fill_between(times_spec, centroids - bandwidths / 2, centroids + bandwidths / 2, alpha=.5)
ax.set(ylim=[None, 6000])
plt.show()

#%% Exercise - Combining many features in a classifier - init

#dict_spectrograms={i:spectrograms[i] for i in range(len(spectrograms))}
import numpy as np
import pandas as pd
from uploadfromdatacamp import saveFromFileIO, loadNDArrayFromCsv
import librosa as lr


#flat_spectro=np.reshape(spectrograms,(65*60,552))
#uploadToFileIO(flat_spectro)
tobedownloaded="{numpy.ndarray: {'flat_spectro.csv': 'https://file.io/CoxZDy'}}"
prefix='data_from_datacamp/ZZZ_Chap29_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

flat_spectro=loadNDArrayFromCsv(prefix+'flat_spectro.csv')
spect=np.reshape(flat_spectro,(60,65,-1))

spectrograms=[spect[i,:,:] for i in range(60)]


#%% Exercise - Combining many features in a classifier

# Loop through each spectrogram
bandwidths = []
centroids = []

for spec in spectrograms:
    # Calculate the mean spectral bandwidth
    this_mean_bandwidth = np.mean(lr.feature.spectral_bandwidth(S=spec))
    # Calculate the mean spectral centroid
    this_mean_centroid = np.mean(lr.feature.spectral_centroid(S=spec))
    # Collect the values
    bandwidths.append(this_mean_bandwidth)  
    centroids.append(this_mean_centroid)
    
# Create X and y arrays
X = np.column_stack([means, stds, maxs, tempos_mean, tempos_max, tempos_std, bandwidths, centroids])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))    