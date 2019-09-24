# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:14:50 2019

@author: F279814
"""

#%% Exercise 41 - A simple outlier - init
import pandas as pd

#%% Exercise 41 - A simple outlier

# Import the LocalOutlierFactor module
from sklearn.neighbors import LocalOutlierFactor as lof

# Create the list [1.0, 1.0, ..., 1.0, 10.0] as explained
x = [1.0]*30
x.append(10)

# Cast to a data frame
X = pd.DataFrame(x)

# Fit the local outlier factor and print the outlier scores
print(lof().fit_predict(X))

#%% Exercise 42 - LoF contamination - init
from sklearn.neighbors import LocalOutlierFactor as lof
import numpy as np
from uploadfromdatacamp import saveFromFileIO
from sklearn.metrics import confusion_matrix
import pandas as pd


#uploadToFileIO(ground_truth, X)
tobedownloaded="{pandas.core.series.Series: {'ground_truth.csv': 'https://file.io/7PQgYu'}, pandas.core.frame.DataFrame: {'X.csv': 'https://file.io/WZsl8V'}}"
prefix='Chap42_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

ground_truth=pd.read_csv(prefix+'ground_truth.csv', index_col=0, header=None,squeeze=True)
X=pd.read_csv(prefix+'X.csv',index_col=0)


#%% Exercise 42 - LoF contamination
# Fit the local outlier factor and output predictions
preds = lof().fit_predict(X)

# Print the confusion matrix
print(confusion_matrix(ground_truth, preds))

#Repeat but now set the proportion of datapoints to be flagged as outliers to 0.2. Print the confusion matrix.
# Set the contamination parameter to 0.2
preds = lof(contamination=0.2).fit_predict(X)

# Print the confusion matrix
print(confusion_matrix(ground_truth, preds))

#Now set the contamination to be equal to the actual proportion of outliers in the data.
# Contamination to match outlier frequency in ground_truth
preds = lof(
  contamination=np.mean(ground_truth == -1)).fit_predict(X)

# Print the confusion matrix
print(confusion_matrix(ground_truth, preds))

#%% Exercise 43 - A simple novelty - init
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor as lof


#%% Exercise 43 - A simple novelty 

# Create a list of thirty 1s and cast to a dataframe
X = pd.DataFrame([1.0]*30)

# Create an instance of a lof novelty detector
detector = lof(novelty=True)

# Fit the detector to the data
detector.fit(X)

# Use it to predict the label of an example with value 10.0
print(detector.predict(pd.DataFrame([10.0])))

#%%Exercise 44 - Three novelty detectors - init
import numpy as np
import pandas as pd
from uploadfromdatacamp import saveFromFileIO


#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/ELu3tS',  'X_train.csv': 'https://file.io/6fgJdT'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/uZs0Tf',  'y_train.csv': 'https://file.io/4S2zDL'}}"
prefix='Chap44_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

y_train=pd.read_csv(prefix+'y_train.csv', index_col=0, header=None,squeeze=True)
y_test=pd.read_csv(prefix+'y_test.csv', index_col=0, header=None,squeeze=True)
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)

#%%Exercise 44 - Three novelty detectors

# Import the novelty detector
from sklearn.svm import OneClassSVM as onesvm

# Fit it to the training data and score the test data
svm_detector = onesvm().fit(X_train)
scores = svm_detector.score_samples(X_test)

#Adapt your code to import the isolation forest from the ensemble module as isof, fit it and score the test data.

# Import the isolation forest
from sklearn.ensemble import IsolationForest as isof

# Fit it to the training data and score the test data
isof_detector = isof().fit(X_train)
scores = isof_detector.score_samples(X_test)

#Adapt your code to import the LocalOutlierFactor module as lof, fit it to the training data, 
#and score the test data.

# Import the novelty detector
from sklearn.neighbors import LocalOutlierFactor as lof

# Fit it to the training data and score the test data
lof_detector = lof(novelty=True).fit(X_train)
scores = lof_detector.score_samples(X_test)

#%% Exercise 45 - Contamination revisited - init
import numpy as np
import pandas as pd
from uploadfromdatacamp import saveFromFileIO
from sklearn.svm import OneClassSVM as onesvm
from sklearn.metrics import confusion_matrix

#uploadToFileIO(X_train, X_test, y_train, y_test)
tobedownloaded="{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/t2IEfl',  'X_train.csv': 'https://file.io/vbsAxO'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/jHTrPT',  'y_train.csv': 'https://file.io/DUkEmW'}}"
prefix='Chap45_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

y_train=pd.read_csv(prefix+'y_train.csv', index_col=0, header=None,squeeze=True)
y_test=pd.read_csv(prefix+'y_test.csv', index_col=0, header=None,squeeze=True)
X_train=pd.read_csv(prefix+'X_train.csv',index_col=0)
X_test=pd.read_csv(prefix+'X_test.csv',index_col=0)


#%% Exercise 45 - Contamination revisited

# Fit a one-class SVM detector and score the test data
nov_det = onesvm().fit(X_train)
scores = nov_det.score_samples(X_test)

# Find the observed proportion of outliers in the test data
prop = np.mean(y_test==1)

# Compute the appropriate threshold
threshold = np.quantile(scores, prop)

# Print the confusion matrix for the thresholded scores
print(confusion_matrix(y_test, scores > threshold))


#%% Exercise 46 - Find the neighbor - init
import pandas as pd
import numpy as np
from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(features, labels)
tobedownloaded="{pandas.core.frame.DataFrame: {'features.csv': 'https://file.io/Hx52vt'}, pandas.core.series.Series: {'labels.csv': 'https://file.io/begEm1'}}"
prefix='ZZZ_Chap46_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

labels=pd.read_csv(prefix+'labels.csv', index_col=0, header=None,squeeze=True)
features=pd.read_csv(prefix+'features.csv',index_col=0)



#%% Exercise 46 - Find the neighbor

# Import DistanceMetric as dm
from sklearn.neighbors import DistanceMetric as dm

# Find the Euclidean distance between all pairs
dist_eucl = dm.get_metric('euclidean').pairwise(features)

# Find the Hamming distance between all pairs
dist_hamm = dm.get_metric('hamming').pairwise(features)

# Find the Chebyshev distance between all pairs
dist_cheb = dm.get_metric('chebyshev').pairwise(features)

#%% Exercise 47 - Not all metrics agree - init
from sklearn.neighbors import LocalOutlierFactor as lof
import pandas as pd

#local
#!cp features.csv vers Chap47
prefix='ZZZ_Chap47_'
features=pd.read_csv(prefix+'features.csv',index_col=0)


#%% Exercise 47 - Not all metrics agree
# Compute outliers according to the euclidean metric
out_eucl = lof(metric='euclidean').fit_predict(features)

# Compute outliers according to the hamming metric
out_hamm = lof(metric='hamming').fit_predict(features)

# Compute outliers according to the jaccard metric
out_jacc  = lof(metric='jaccard').fit_predict(features)

# Find if the metrics agree on any one datapoint
print(any(out_eucl + out_hamm + out_jacc == -3))

#%% Exercise 48 - Restricted Levenshtein - init
from sklearn.metrics import accuracy_score as accuracy
import stringdist
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import LocalOutlierFactor as lof
from uploadfromdatacamp import saveFromFileIO


#uploadToFileIO(proteins)
tobedownloaded="{pandas.core.frame.DataFrame: {'proteins.csv': 'https://file.io/JVthuR'}}"
prefix='ZZZ_Chap48_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
proteins=pd.read_csv(prefix+'proteins.csv',index_col=0)


#%% Exercise 48 - Restricted Levenshtein

# Wrap the RD-Levenshtein metric in a custom function
def my_rdlevenshtein(u, v):
    return stringdist.rdlevenshtein(u[0], v[0])

# Reshape the array into a numpy matrix
sequences = np.array(proteins['seq']).reshape(-1, 1)

# Compute the pairwise distance matrix in square form
M = squareform(pdist(sequences, my_rdlevenshtein))

# Run a LoF algorithm on the precomputed distance matrix
preds = lof(metric='precomputed').fit_predict(M)

# Compute the accuracy of the outlier predictions
print(accuracy(proteins['label'] == 'VIRUS', preds == -1))

#%% Exercise 49 - Bringing it all together - init
import pandas as pd
import numpy as np
import pickle
from uploadfromdatacamp import saveFromFileIO
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score as auc
from sklearn.svm import OneClassSVM 

#uploadToFileIO(proteins)
tobedownloaded="{pandas.core.frame.DataFrame: {'proteins.csv': 'https://file.io/7Akwc9'}}"
prefix='ZZZ_Chap49_'
#○saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")
proteins=pd.read_csv(prefix+'proteins.csv',index_col=0)

#with open('lof_detector.pkl', 'wb') as file:
#	pickle.dump(lof_detector, file=file)
#uploadToFileIO_pushto_fileio('lof_detector.pkl')

url='https://file.io/Jfe3CX'
tobesaved_as='lof_detector.pkl'
prefix='ZZZ_Chap49_'
tobedownloaded="{lof:{'"+tobesaved_as+"': '"+url+"'}}"
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

with open(prefix+tobesaved_as, 'rb') as file:
	lof_detector = pickle.load(file)


#%% Exercise 49 - Bringing it all together
# Create a feature that contains the length of the string
proteins['len'] = proteins['seq'].apply(lambda x: len(x))

# Create a feature encoding the first letter of the string
#For a string s, list(s) returns a list of its characters. Use this to extract the first letter of each sequence, and encode it using LabelEncoder().
proteins['first'] =  LabelEncoder().fit_transform(
  proteins['seq'].apply(lambda s: list(s)[0]))

# Extract scores from the fitted LoF object, compute its AUC
#LoF scores are in the negative_outlier_factor_ attribute. Compute their AUC.
scores_lof = lof_detector.negative_outlier_factor_
print(auc(proteins['label']=='IMMUNE SYSTEM', scores_lof))

# Fit a 1-class SVM, extract its scores, and compute its AUC
svm = OneClassSVM().fit(proteins[['len', 'first']])
scores_svm = svm.score_samples(proteins[['len', 'first']])
print(auc(proteins['label']=='IMMUNE SYSTEM', scores_svm))
