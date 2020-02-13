#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
pixel_gabrou = plt.imread('pixel-gabrou.jpg')


# In[3]:


type(pixel_gabrou)


# In[4]:



import matplotlib.pyplot as plt
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()


# In[5]:


show_image(pixel_gabrou)


# In[6]:


from skimage.transform import rotate, rescale
rotated_pixel_gabrou = rotate(pixel_gabrou, -90)
show_image(rotated_pixel_gabrou)


# In[8]:


# Import the slic function from segmentation module
from skimage.segmentation import slic

# Import the label2rgb function from color module
from skimage.color import label2rgb
import matplotlib


segments_possibles=[32000]

def batchTraitementImage(listeSegments):
    if (type(listeSegments)==(int)):
        listeSegments=[listeSegments]
    for n_segments in listeSegments:
        # Obtain the segmentation with 400 regions
        segments = slic(rotated_pixel_gabrou, n_segments= n_segments)

        # Put segments on top of original image to compare
        segmented_image = label2rgb(segments, rotated_pixel_gabrou, kind='avg')
        matplotlib.image.imsave('pix-gab'+str(n_segments)+'.png', segmented_image)

        # Show the segmented image
        show_image(segmented_image, "Segmented image, "+str(n_segments)+" superpixels")


# In[19]:


segments_possibles=[100,200, 500, 1000, 2000, 4000]
batchTraitementImage(segments_possibles)


# In[7]:


segments_possibles=[32000]
batchTraitementImage(segments_possibles)


# In[10]:


for i in range(5,100,10):
    batchTraitementImage(i)


# In[ ]:




