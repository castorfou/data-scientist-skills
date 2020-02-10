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
prefixToc = '1.1'
saveFromFileIO(tobedownloaded, proxy="10.225.92.1:80", prefixToc=prefixToc)

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
uploadToFileIO(mask)
"""

tobedownloaded="""
{numpy.ndarray: {'mask.csv': 'https://file.io/6USsXM'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import loadNDArrayFromCsv
mask = loadNDArrayFromCsv(prefix+'mask.csv')

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
image_256_256_3 = image.flatten()
uploadToFileIO(image_256_256_3)
"""

tobedownloaded="""
{numpy.ndarray: {'image_256_256_3.csv': 'https://file.io/0e8NyX'}}
"""
prefixToc = '2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import loadNDArrayFromCsv
image_256_256_3 = loadNDArrayFromCsv(prefix+'image_256_256_3.csv')
image = np.reshape(image_256_256_3, (256,256,3))
image =image.astype('uint8')

###################
##### image (numpy ndarray)
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(page_image, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'page_image[172_448].csv': 'https://file.io/Qmazrg'}}
"""
prefixToc = '3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
page_image = getImage(prefix+'page_image[172_448].csv')

