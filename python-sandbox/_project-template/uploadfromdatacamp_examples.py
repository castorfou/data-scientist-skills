###################
##### Dataframe
###################

#upload and download

from uploadfromdatacamp import saveFromFileIO
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

from uploadfromdatacamp import saveFromFileIO
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
