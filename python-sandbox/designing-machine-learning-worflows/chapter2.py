# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:29:10 2019

@author: F279814
"""

#%% Exercise - Is the source or the destination bad? - init
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
#import pandas as pd
#flows.to_csv('flows.csv')
#!curl -F "file=@flows.csv" https://file.io
#{"success":true,"key":"ilu4i4","link":"https://file.io/ilu4i4","expiry":"14 days"}
import pandas as pd
flows=pd.read_csv('flows.csv', index_col=0)
bads={'C4773', 'C4159', 'C3199', 'C17425', 'C16401', 'C6513', 'C19932', 'C977', 'C504', 'C706', 'C1616', 'C1776', 'C126', 'C3699', 'C791', 'C20966', 'C17640', 'C1461', 'C21814', 'C3380', 'C2849', 'C243', 'C882', 'C1319', 'C3292', 'C3173', 'C5343', 'C3521', 'C2254', 'C22766', 'C2844', 'C18113', 'C1503', 'C1382', 'C19803', 'C2578', 'C3610', 'C553', 'C246', 'C1191', 'C15197', 'C5653', 'C1042', 'C633', 'C10005', 'C5439', 'C21946', 'C4280', 'C1022', 'C2877', 'C96', 'C2091', 'C1980', 'C42', 'C5453', 'C586', 'C1224', 'C20677', 'C12320', 'C1183', 'C21349', 'C1484', 'C22174', 'C3153', 'C1952', 'C89', 'C1302', 'C1046', 'C3305', 'C423', 'C16088', 'C687', 'C1014', 'C2846', 'C1125', 'C2013', 'C90', 'C685', 'C779', 'C3422', 'C366', 'C1269', 'C143', 'C2079', 'C2669', 'C2378', 'C115', 'C395', 'C13713', 'C513', 'C2057', 'C1964', 'C15', 'C1222', 'C486', 'C1737', 'C18626', 'C15232', 'C21963', 'C1570', 'C16467', 'C881', 'C1906', 'C10817', 'C359', 'C917', 'C1215', 'C22275', 'C1173', 'C1482', 'C721', 'C10405', 'C8751', 'C2519', 'C2725', 'C3888', 'C1509', 'C2648', 'C92', 'C9945', 'C294', 'C17860', 'C528', 'C4554', 'C1549', 'C2196', 'C4610', 'C1089', 'C113', 'C20455', 'C1581', 'C1611', 'C1275', 'C3019', 'C1626', 'C1823', 'C1096', 'C78', 'C430', 'C17776', 'C1432', 'C19444', 'C923', 'C1003', 'C728', 'C3388', 'C19356', 'C2388', 'C17636', 'C400', 'C1479', 'C849', 'C1961', 'C3249', 'C612', 'C17600', 'C1065', 'C2944', 'C9006', 'C7503', 'C177', 'C2609', 'C5618', 'C1506', 'C625', 'C3170', 'C8490', 'C1448', 'C3629', 'C3288', 'C11039', 'C4161', 'C18872', 'C886', 'C1784', 'C1966', 'C1936', 'C4845', 'C8209', 'C231', 'C305', 'C853', 'C5030', 'C370', 'C4934', 'C1006', 'C20203', 'C353', 'C313', 'C1415', 'C152', 'C7464', 'C1710', 'C3601', 'C801', 'C6487', 'C2816', 'C464', 'C765', 'C1632', 'C3455', 'C636', 'C1567', 'C1477', 'C1119', 'C3491', 'C16563', 'C754', 'C307', 'C742', 'C7597', 'C22409', 'C1493', 'C12682', 'C1944', 'C2058', 'C492', 'C306', 'C1732', 'C398', 'C7782', 'C12116', 'C626', 'C828', 'C11727', 'C583', 'C2914', 'C529', 'C3758', 'C1500', 'C2341', 'C10', 'C18190', 'C368', 'C11178', 'C2597', 'C108', 'C883', 'C458', 'C338', 'C17806', 'C1124', 'C332', 'C457', 'C3813', 'C3586', 'C452', 'C17693', 'C9723', 'C1797', 'C52', 'C3635', 'C102', 'C5111', 'C3303', 'C19038', 'C346', 'C8585', 'C148', 'C1268', 'C302', 'C1028', 'C18464', 'C61', 'C12448', 'C2604', 'C3597', 'C3437', 'C3037', 'C20819', 'C10577', 'C21919', 'C467', 'C1555', 'C2012', 'C12512', 'C46', 'C4106', 'C21664', 'C5693', 'C1015', 'C7131', 'C18025', 'C798', 'C385', 'C3435', 'C8172', 'C506', 'C1', 'C1438', 'C1610', 'C1887', 'C477', 'C155', 'C19156', 'C2085', 'C3755', 'C1085', 'C4403', 'C11194', 'C429', 'C1810', 'C22176', 'C965', 'C9692'}
#print(type(bads))

def featurize(df):
    return {
        'unique_ports': len(set(df['destination_port'])),
        'average_packet': np.mean(df['packet_count']),
        'average_duration': np.mean(df['duration'])
    }

#%% Exercise - Is the source or the destination bad?
# Group by source computer, and apply the feature extractor
out = flows.groupby('source_computer').apply(featurize)

# Convert the iterator to a dataframe by calling list on it
X = pd.DataFrame(list(out), index=out.index)

# Check which sources in X.index are bad to create labels
y = [x in bads for x in X.index]

# Report the average accuracy of Adaboost over 3-fold CV
print(np.mean(cross_val_score(AdaBoostClassifier(), X, y)))

#%% Exercise - Feature engineering on grouped data
# Create a feature counting unique protocols per source
protocols = flows.groupby('source_computer').apply(
  lambda df: len(set(df['protocol'])))

# Convert this feature into a dataframe, naming the column
protocols_DF = pd.DataFrame(
  protocols, index=protocols.index, columns=['protocol'])

# Now concatenate this feature with the previous dataset, X
X_more = pd.concat([X, protocols_DF], axis=1)

# Refit the classifier and report its accuracy
print(np.mean(cross_val_score(AdaBoostClassifier(), X_more, y)))

#%% Exercise - Turning a heuristic into a classifier - init
import numpy as np
from sklearn.metrics import accuracy_score
from uploadfromdatacamp import loadListFromTxt

#uploadFromDatacamp(X_train, X_test, y_train, y_test)
#saveFromFileIO("{'X_test.csv': 'https://file.io/AIQv4v', 'X_train.csv': 'https://file.io/kqaUY3', 'y_test.txt': 'https://file.io/cisNoL', 'y_train.txt': 'https://file.io/r5pL9X'}")
X_train=pd.read_csv('X_train.csv', index_col=0)
X_test=pd.read_csv('X_test.csv', index_col=0)
y_train=loadListFromTxt('y_train.txt')
y_test=loadListFromTxt('y_test.txt')


#%% Exercise - Turning a heuristic into a classifier

# Create a new dataset X_train_bad by subselecting bad hosts
X_train_bad = X_train[y_train]

# Calculate the average of unique_ports in bad examples
avg_bad_ports = np.mean(X_train_bad['unique_ports'])

# Label as positive sources that use more ports than that
pred_port = X_test['unique_ports'] > avg_bad_ports

# Print the accuracy of the heuristic
print(accuracy_score(y_test, pred_port))

#%%Exercise - Combining heuristics - init
import numpy as np
from sklearn.metrics import accuracy_score
from uploadfromdatacamp import loadListFromTxt

#uploadFromDatacamp(X_train, X_test, y_train, y_test)
#saveFromFileIO("{'X_test.csv': 'https://file.io/lcjuIS', 'X_train.csv': 'https://file.io/K665Su', 'y_test.txt': 'https://file.io/o4Zg2o', 'y_train.txt': 'https://file.io/UXJui7'}")
X_train=pd.read_csv('X_train.csv',index_col=0)
X_test=pd.read_csv('X_test.csv',index_col=0)
y_train=loadListFromTxt('y_train.txt')
y_test=loadListFromTxt('y_test.txt')

#%%Exercise - Combining heuristics
# Compute the mean of average_packet for bad sources
avg_bad_packet = np.mean(X_train[y_train]['average_packet'])

# Label as positive if average_packet is lower than that
pred_packet = X_test['average_packet'] < avg_bad_packet

# Find indices where pred_port and pred_packet both True
pred_port = X_test['unique_ports'] > avg_bad_ports
pred_both = pred_packet & pred_port

# Ports only produced an accuracy of 0.919. Is this better?
print(accuracy_score(y_test, pred_both))

#%% Exercise - Dealing with label noise - init
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from uploadfromdatacamp import loadListFromTxt

#uploadFromDatacamp(X_train, X_test, y_train, y_test)
#saveFromFileIO("{'X_test.csv': 'https://file.io/AIQv4v', 'X_train.csv': 'https://file.io/kqaUY3', 'y_test.txt': 'https://file.io/cisNoL', 'y_train.txt': 'https://file.io/r5pL9X'}")
X_train=pd.read_csv('X_train.csv',index_col=0)
X_test=pd.read_csv('X_test.csv',index_col=0)
y_train=loadListFromTxt('y_train.txt')
y_test=loadListFromTxt('y_test.txt')
#uploadFromDatacamp(X_train, X_test, y_train_noisy, y_test)
#saveFromFileIO("{'X_test.csv': 'https://file.io/oN4jvF', 'X_train.csv': 'https://file.io/nzL4W3', 'y_test.txt': 'https://file.io/VYXSP0', 'y_train_noisy.txt': 'https://file.io/5nzj3Y'}")
y_train_noisy=loadListFromTxt('y_train_noisy.txt')
X_train=pd.read_csv('X_train.csv',index_col=0)
X_test=pd.read_csv('X_test.csv',index_col=0)
y_test=loadListFromTxt('y_test.txt')


#%%Exercise - Dealing with label noise
# Fit a Gaussian Naive Bayes classifier to the training data
clf = GaussianNB().fit(X_train, y_train_noisy)

# Report its accuracy on the test data
print(accuracy_score(y_test, clf.predict(X_test)))

# Assign half the weight to the first 100 noisy examples
weights = [0.5]*100 + [1.0]*(len(y_train_noisy)-100)

# Refit using weights and report accuracy. Has it improved?
clf_weights = GaussianNB().fit(X_train, y_train_noisy, sample_weight=weights)
print(accuracy_score(y_test, clf_weights.predict(X_test)))

#%% Exercise - Reminder of performance metrics - init
from sklearn.metrics import precision_score, f1_score
from uploadfromdatacamp import loadNDArrayFromCsv
(tp,fp,fn,tn)=(155, 23, 48, 24)
#uploadFromDatacamp(y_test, preds)
#{pandas.core.series.Series: {'y_test.csv': 'https://file.io/BXr70M'}, numpy.ndarray: {'preds.csv': 'https://file.io/Yi4jj3'}}
#saveFromFileIO("{pandas.core.series.Series: {'y_test.csv': 'https://file.io/BXr70M'}, numpy.ndarray: {'preds.csv': 'https://file.io/Yi4jj3'}}")
import pandas as pd
y_test=pd.read_csv('y_test.csv', index_col=0, header=None,squeeze=True)
preds=loadNDArrayFromCsv('preds.csv').astype(bool)

#%% Exercise - Reminder of performance metrics
print(f1_score(y_test, preds))
print(precision_score(y_test, preds))
print((tp + tn)/len(y_test))

#%% Exercise - Real-world cost analysis - init
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from uploadfromdatacamp import loadNDArrayFromCsv
from uploadfromdatacamp import loadListFromTxt
#from uploadfromdatacamp import saveFromFileIO

#uploadFromDatacamp(X_train, X_test, y_train, y_test)
#{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/z7L9od',  'X_train.csv': 'https://file.io/qVqpz2'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/t6vojs',  'y_train.csv': 'https://file.io/n2A02H'}}
#saveFromFileIO("{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/z7L9od',  'X_train.csv': 'https://file.io/qVqpz2'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/t6vojs',  'y_train.csv': 'https://file.io/n2A02H'}}")
import pandas as pd
X_train=pd.read_csv('X_train.csv',index_col=0)
X_test=pd.read_csv('X_test.csv',index_col=0)
y_test=pd.read_csv('y_test.csv', index_col=0, header=None,squeeze=True)
y_train=pd.read_csv('y_train.csv', index_col=0, header=None,squeeze=True)

#%% Exercise - Real-world cost analysis

# Fit a random forest classifier to the training data
clf = RandomForestClassifier(random_state=2).fit(X_train, y_train)

# Label the test data
preds = clf.predict(X_test)

# Get false positives/negatives from the confusion matrix
tp, fp, fn, tn = confusion_matrix(y_test, preds).ravel()

# Now compute the cost using the manager's advice
cost = fp*10 + fn*150


#%% Exercise - Default thresholding - init
from sklearn.tree import DecisionTreeClassifier
#from uploadfromdatacamp import saveFromFileIO

#uploadFromDatacamp(X_train, X_test, y_train, y_test)
#{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/6Rrga4',  'X_train.csv': 'https://file.io/VV8MB3'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/63mvgM',  'y_train.csv': 'https://file.io/iqMPkE'}}
#saveFromFileIO("{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/6Rrga4',  'X_train.csv': 'https://file.io/VV8MB3'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/63mvgM',  'y_train.csv': 'https://file.io/iqMPkE'}}")
import pandas as pd
X_train=pd.read_csv('X_train.csv',index_col=0)
X_test=pd.read_csv('X_test.csv',index_col=0)
y_test=pd.read_csv('y_test.csv', index_col=0, header=None,squeeze=True)
y_train=pd.read_csv('y_train.csv', index_col=0, header=None,squeeze=True)

clf=DecisionTreeClassifier().fit(X_train, y_train)


#%% Exercise - Default thresholding
# Score the test data using the given classifier
scores = clf.predict_proba(X_test)

# Get labels from the scores using the default threshold
preds = [s[1] > 0.5 for s in scores]

# Use the predict method to label the test data again
preds_default = clf.predict(X_test)

# Compare the two sets of predictions
all(preds == preds_default)

#%% Exercise - Optimizing the threshold - init
import numpy as np
#from uploadfromdatacamp import saveFromFileIO
import pandas as pd
from uploadfromdatacamp import loadNDArrayFromCsv
from sklearn.metrics import accuracy_score, f1_score
from numpy import argmax

#uploadFromDatacamp(scores, y_test)
#{numpy.ndarray: {'scores.csv': 'https://file.io/c68R60'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/zjMwXn'}}
#saveFromFileIO("{numpy.ndarray: {'scores.csv': 'https://file.io/c68R60'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/zjMwXn'}}")
y_test=pd.read_csv('y_test.csv', index_col=0, header=None,squeeze=True)
scores=loadNDArrayFromCsv('scores.csv')

#%% Exercise - Optimizing the threshold
# Create a range of equally spaced threshold values
t_range = [0.0, 0.25, 0.5, 0.75, 1.0]

# Store the predicted labels for each value of the threshold
preds = [[s[1] > thr for s in scores] for thr in t_range]

# Compute the accuracy for each threshold
accuracies = [accuracy_score(y_test, p) for p in preds]

# Compute the F1 score for each threshold
f1_scores = [f1_score(y_test, p) for p in preds]

# Report the optimal threshold for accuracy, and for F1
print(t_range[argmax(accuracies)], t_range[argmax(f1_scores)])

#%% Exercise - Bringing it all together - init
import numpy as np
#from uploadfromdatacamp import saveFromFileIO
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#uploadFromDatacamp(X_train, X_test, y_train, y_test)
#{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/TYEDSm',  'X_train.csv': 'https://file.io/g6QkrP'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/8XOOnh',  'y_train.csv': 'https://file.io/2O44ua'}}
#saveFromFileIO("{pandas.core.frame.DataFrame: {'X_test.csv': 'https://file.io/TYEDSm',  'X_train.csv': 'https://file.io/g6QkrP'}, pandas.core.series.Series: {'y_test.csv': 'https://file.io/8XOOnh',  'y_train.csv': 'https://file.io/2O44ua'}}")
X_train=pd.read_csv('X_train.csv',index_col=0)
X_test=pd.read_csv('X_test.csv',index_col=0)
y_test=pd.read_csv('y_test.csv', index_col=0, header=None,squeeze=True)
y_train=pd.read_csv('y_train.csv', index_col=0, header=None,squeeze=True)


#%% Exercise - Bringing it all together
# Create a scorer assigning more cost to false positives
def my_scorer(y_test, y_est, cost_fp=10.0, cost_fn=1.0):
    tn, fp, fn, tp = confusion_matrix(y_test, y_est).ravel()
    return fp*cost_fp+fn*cost_fn

# Fit a DecisionTreeClassifier to the data and compute the loss
clf = DecisionTreeClassifier(random_state=2).fit(X_train, y_train)
print(my_scorer(y_test, clf.predict(X_test)))

# Refit with same seed, downweighting subjects weighing > 80
weights = [0.5 if w > 80 else 1.0 for w in X_train.weight]
clf_weighted = DecisionTreeClassifier(random_state=2).fit(X_train,y_train,sample_weight=weights)
print(my_scorer(y_test, clf_weighted.predict(X_test)))
