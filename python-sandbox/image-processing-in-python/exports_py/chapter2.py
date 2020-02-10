#!/usr/bin/env python
# coding: utf-8

# # Jump into filtering
# 

# ## Edge detection
# In this exercise, you'll detect edges in an image by applying the Sobel filter.
# 
# Soap pills of heart and rectangle shapes in blue background
# Image preloaded as soaps_image.
# ![image.png](attachment:image.png)
# Theshow_image() function has been already loaded for you.
# 
# Let's see if it spots all the figures in the image.

# ### init

# In[1]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(soaps_image, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'soaps_image[417_626_3].csv': 'https://file.io/ltoDac'}}
"""
prefixToc = '1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
soaps_image = getImage(prefix+'soaps_image[417_626_3].csv')


# In[2]:


""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(show_image)
"""
import matplotlib.pyplot as plt

def show_image(image, title='Image', cmap_type='gray', interpolation=None):
    plt.imshow(image, cmap=cmap_type)    
    plt.title(title)
    plt.axis('off')
    plt.show()


# ### code

# In[3]:


# Import the color module
from skimage import color

# Import the filters module and sobel function
from skimage.filters import sobel

# Make the image grayscale
soaps_image_gray = color.rgb2gray(soaps_image)

# Apply edge detection filter
edge_sobel = sobel(soaps_image_gray)

# Show original and resulting image to compare
show_image(soaps_image, "Original")
show_image(edge_sobel, "Edges with Sobel")


# ## Blurring to reduce noise
# In this exercise you will reduce the sharpness of an image of a building taken during a London trip, through filtering.
# 
# ![image.png](attachment:image.png)
# Building in Lodon
# Image loaded as building_image.

# ### init

# In[4]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(building_image, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'building_image[1728_1152_3].csv': 'https://file.io/9Niw4b'}}
"""
prefixToc = '1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
building_image = getImage(prefix+'building_image[1728_1152_3].csv')


# ### code

# In[5]:


# Import Gaussian filter 
from skimage.filters import gaussian

# Apply filter
gaussian_image = gaussian(building_image, multichannel=True)

# Show original and resulting image to compare
show_image(building_image, "Original")
show_image(gaussian_image, "Reduced sharpness Gaussian")


# # Contrast enhancement
# 

# ## What's the contrast of this image?
# ![image.png](attachment:image.png)
# Black and white clock hanging and moving Histogram of the clock's image
# 
# The histogram tell us.
# 
# Just as we saw previously, you can calculate the contrast by calculating the range of the pixel intensities i.e. by subtracting the minimum pixel intensity value from the histogram to the maximum one.
# 
# You can obtain the maximum pixel intensity of the image by using the np.max() method from NumPy and the minimum with np.min() in the console.
# 
# The image has already been loaded as clock_image, NumPy as np and the show_image() function.

# ### init
# 

# In[6]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(clock_image, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'clock_image[300_400].csv': 'https://file.io/iGNdND'}}
"""
prefixToc = '2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
clock_image = getImage(prefix+'clock_image[300_400].csv')


# ### code

# In[8]:


import numpy as np
np.min(clock_image), np.max(clock_image)


# ## Medical images
# You are trying to improve the tools of a hospital by pre-processing the X-ray images so that doctors have a higher chance of spotting relevant details. You'll test our code on a chest X-ray image from the National Institutes of Health Chest X-Ray Dataset
# X-ray chest image
# ![image.png](attachment:image.png)
# Image loaded as chest_xray_image.
# First, you'll check the histogram of the image and then apply standard histogram equalization to improve the contrast. Remember we obtain the histogram by using the hist() function from Matplotlib, which has been already imported as plt.

# ### init

# In[9]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(chest_xray_image, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'chest_xray_image[1024_1024].csv': 'https://file.io/VkFXbZ'}}
"""
prefixToc = '2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
chest_xray_image = getImage(prefix+'chest_xray_image[1024_1024].csv')


# ### code

# In[10]:


# Import the required module
from skimage import exposure


# In[11]:


# Show original x-ray image and its histogram
show_image(chest_xray_image, 'Original x-ray')

plt.title('Histogram of image')
plt.hist(chest_xray_image.ravel(), bins=256)
plt.show()


# In[12]:


# Use histogram equalization to improve the contrast
xray_image_eq =  exposure.equalize_hist(chest_xray_image)


# In[13]:


# Show the resulting image
show_image(xray_image_eq, 'Resulting image')


# ## Aerial image
# In this exercise, we will improve the quality of an aerial image of a city. The image has low contrast and therefore we can not distinguish all the elements in it.
# ![image.png](attachment:image.png)
# Aerial image, airport taken from the air
# Image loaded as image_aerial.
# For this we will use the normal or standard technique of Histogram Equalization.

# ### init

# In[14]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(image_aerial, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'image_aerial[511_512].csv': 'https://file.io/1hdNyC'}}
"""
prefixToc = '2.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
image_aerial = getImage(prefix+'image_aerial[511_512].csv')


# ### code

# In[15]:


# Import the required module
from skimage import exposure

# Use histogram equalization to improve the contrast
image_eq =  exposure.equalize_hist(image_aerial)

# Show the original and resulting image
show_image(image_aerial, 'Original')
show_image(image_eq, 'Resulting image')


# ## Let's add some impact and contrast
# Have you ever wanted to enhance the contrast of your photos so that they appear more dramatic?
# 
# In this exercise, you'll increase the contrast of a cup of coffee. Something you could share with your friends on social media. Don't forget to use #ImageProcessingDatacamp as hashtag!
# 
# Even though this is not our Sunday morning coffee cup, you can still apply the same methods to any of our photos.
# ![image.png](attachment:image.png)
# Cup of coffee
# A function called show_image(), that displays an image using Matplotlib, has already been defined. It has the arguments image and title, with title being 'Original' by default.

# ### code

# In[16]:


# Import the necessary modules
from skimage import data, exposure

# Load the image
original_image = data.coffee()

# Apply the adaptive equalization on the original image
adapthist_eq_image = exposure.equalize_adapthist(original_image, clip_limit=0.03)

# Compare the original image to the equalized
show_image(original_image)
show_image(adapthist_eq_image, '#ImageProcessingDatacamp')


# # Transformations

# ## Aliasing, rotating and rescaling
# Let's look at the impact of aliasing on images.
# 
# Remember that aliasing is an effect that causes different signals, in this case pixels, to become indistinguishable or distorted.
# 
# You'll make this cat image upright by rotating it 90 degrees and then rescaling it two times. Once with the anti aliasing filter applied before rescaling and a second time without it, so you can compare them.
# ![image.png](attachment:image.png)
# Little cute cat
# Image preloaded as image_cat.

# ### init

# In[17]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(image_cat, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'image_cat[612_640_3].csv': 'https://file.io/JHvMte'}}
"""
prefixToc = '3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
image_cat = getImage(prefix+'image_cat[612_640_3].csv')


# ### code

# In[18]:


# Import the module and the rotate and rescale functions
from skimage.transform import rotate, rescale


# In[19]:


# Rotate the image 90 degrees clockwise 
rotated_cat_image = rotate(image_cat, -90)


# In[20]:


# Rescale with anti aliasing
rescaled_with_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=True, multichannel=True)


# In[21]:


# Rescale without anti aliasing
rescaled_without_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=False, multichannel=True)

# Show the resulting images
show_image(rescaled_with_aa, "Transformed with anti aliasing")
show_image(rescaled_without_aa, "Transformed without anti aliasing")


# ## Enlarging images
# Have you ever tried resizing an image to make it larger? This usually results in loss of quality, with the enlarged image looking blurry.
# 
# The good news is that the algorithm used by scikit-image works very well for enlarging images up to a certain point.
# 
# In this exercise you'll enlarge an image four times!!
# 
# You'll do this by rescaling the image of a rocket, that will be loaded from the data module.
# ![image.png](attachment:image.png)
# Rocket

# ### code

# In[22]:


# Import the module and function to enlarge images
from skimage.transform import rescale

# Import the data module
from skimage import data

# Load the image from data
rocket_image = data.rocket()

# Enlarge the image so it is 4 times bigger
enlarged_rocket_image = rescale(rocket_image, 4, anti_aliasing=True, multichannel=True)

# Show original and resulting image
show_image(rocket_image)
show_image(enlarged_rocket_image, "4 times enlarged image")


# ## Proportionally resizing
# We want to downscale the images of a veterinary blog website so all of them have the same compressed size.
# 
# It's important that you do this proportionally, meaning that these are not distorted.
# 
# First, you'll try it out for one image so you know what code to test later in the rest of the pictures.
# 
# ![image.png](attachment:image.png)
# The image preloaded as dogs_banner.
# Remember that by looking at the shape of the image, you can know its width and height.

# ### init

# In[23]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(dogs_banner, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'dogs_banner[423_640_3].csv': 'https://file.io/N7wi52'}}
"""
prefixToc = '3.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
dogs_banner = getImage(prefix+'dogs_banner[423_640_3].csv')


# ### code

# In[24]:


# Import the module and function
from skimage.transform import resize

# Set proportional height so its half its size
height = int(dogs_banner.shape[0] / 2)
width = int(dogs_banner.shape[1] / 2)

# Resize using the calculated proportional height and width
image_resized = resize(dogs_banner, (height, width),
                       anti_aliasing=True)

# Show the original and rotated image
show_image(dogs_banner, 'Original')
show_image(image_resized, 'Resized image')


# # Morphology

# ## Handwritten letters
# A very interesting use of computer vision in real-life solutions is performing Optical Character Recognition (OCR) to distinguish printed or handwritten text characters inside digital images of physical documents.
# 
# Let's try to improve the definition of this handwritten letter so that it's easier to classify.
# 
# ![image.png](attachment:image.png)
# As we can see it's the letter R, already binary, with with some noise in it. It's already loaded as upper_r_image.
# 
# Apply the morphological operation that will discard the pixels near the letter boundaries.

# ### init

# In[39]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(upper_r_image, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'upper_r_image[70_80].csv': 'https://file.io/vWp2T9'}}
"""
prefixToc = '4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
upper_r_image = getImage(prefix+'upper_r_image[70_80].csv', dtype='float64')
show_image(upper_r_image)


# ### code

# In[40]:


# Import the morphology module
from skimage import morphology

# Obtain the eroded shape 
eroded_image_shape = morphology.binary_erosion(upper_r_image) 

# See results
show_image(upper_r_image, 'Original')
show_image(eroded_image_shape, 'Eroded image')


# ## Improving thresholded image
# In this exercise, we'll try to reduce the noise of a thresholded image using the dilation morphological operation.
# ![image.png](attachment:image.png)
# World map
# Image already loaded as world_image.
# This operation, in a way, expands the objects in the image.

# ### init

# In[32]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(world_image, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'world_image[327_640].csv': 'https://file.io/qG88O5'}}
"""
prefixToc = '4.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
world_image = getImage(prefix+'world_image[327_640].csv', dtype='bool')


# In[33]:


show_image(world_image)


# ### code

# In[37]:


# Import the module
from skimage import morphology

# Obtain the dilated image 
dilated_image = morphology.binary_dilation(world_image)

# See results
show_image(world_image, 'Original')
show_image(dilated_image, 'Dilated image')


# In[ ]:




