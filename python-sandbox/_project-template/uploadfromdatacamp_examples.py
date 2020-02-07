###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(df)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'df.csv': 'https://file.io/y7Rwj6'}}
"""
prefix='data_from_datacamp/Chap1-Exercise1.1_'
saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
df = pd.read_csv(prefix+'df.csv',index_col=0)

###################
##### pandas Serie
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(y)
"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'y.csv': 'https://file.io/y7Rwj6'}}
"""
prefix='data_from_datacamp/Chap1-Exercise1.1_'
saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

import pandas as pd
y = pd.read_csv(prefix+'y.csv',index_col=0, header=None,squeeze=True)

###################
##### inspect Function
###################

""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(get_recommendations)
"""

###################
##### numpy ndarray float
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(time_steps)
"""

tobedownloaded="""
{numpy.ndarray: {'time_steps.csv': 'https://file.io/FNc6kh}
"""
prefix='data_from_datacamp/Chap1-Exercise3.1_'
saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import loadNDArrayFromCsv
time_steps = loadNDArrayFromCsv(prefix+'time_steps.csv')

###################
##### Keras model
###################

#upload and download

from downloadfromFileIO import saveFromFileIO2
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(model)
"""

tobedownloaded="""
{keras.engine.sequential.Sequential: {'model.h5': 'https://file.io/J3B2WY'}}
"""
prefix='data_from_datacamp/Chap4-Exercise2.3_'
saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import loadModelFromH5
model = loadModelFromH5(prefix+'model.h5')


###################
##### numpy ndarray float N-dimensional n>2
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
X_test_100_28_28_1 = X_test.flatten()
uploadToFileIO(X_test_100_28_28_1)
"""

tobedownloaded="""
{numpy.ndarray: {'X_test_100_28_28_1.csv': 'https://file.io/jdynnD'}}
"""
prefix='data_from_datacamp/Chap1-Exercise3.1_'
saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import loadNDArrayFromCsv
X_test_100_28_28_1 = loadNDArrayFromCsv(prefix+'X_test_100_28_28_1.txt')
X_test = np.reshape(X_test_100_28_28_1, (100,28,28,1))