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

