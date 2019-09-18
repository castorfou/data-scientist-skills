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

#uploadFromDatacamp(X_train, X_test)

#%% Exercise - Turning a heuristic into a classifier

