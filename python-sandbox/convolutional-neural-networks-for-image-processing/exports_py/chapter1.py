#!/usr/bin/env python
# coding: utf-8

# # Introducing convolutional neural networks
# 

# ## Images as data: visualizations
# To display image data, you will rely on Python's Matplotlib library, and specifically use matplotlib's pyplot sub-module, that contains many plotting commands. Some of these commands allow you to display the content of images stored in arrays.

# ### init

# In[2]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('bricks.png')
"""

tobedownloaded="""
{numpy.ndarray: {'bricks.png': 'https://file.io/GJ8xXK'}}
"""
prefixToc = '1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")


# ### code

# In[7]:


# Import matplotlib
import matplotlib.pyplot as plt

# Load the image
data = plt.imread(prefix+'bricks.png')

# Display the image
plt.imshow(data)
plt.show()


# ## Images as data: changing images
# To modify an image, you can modify the existing numbers in the array. In a color image, you can change the values in one of the color channels without affecting the other colors, by indexing on the last dimension of the array.
# 
# The image you imported in the previous exercise is available in data.

# ### code

# In[8]:


# Set the red channel in this part of the image to 1
data[:10,:10,0] = 1

# Set the green channel in this part of the image to 0
data[:10,:10,1] = 0

# Set the blue channel in this part of the image to 0
data[:10,:10,2] = 0

# Visualize the result
plt.imshow(data)
plt.show()


# In[ ]:




