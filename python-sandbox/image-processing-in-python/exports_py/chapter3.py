#!/usr/bin/env python
# coding: utf-8

# # Image restoration

# ## Let's restore a damaged image
# In this exercise, we'll restore an image that has missing parts in it, using the inpaint_biharmonic() function.
# 
# ![image.png](attachment:image.png)
# Small cute puppy
# Loaded as defect_image.
# We'll work on an image from the data module, obtained by data.astronaut(). Some of the pixels have been replaced by 1s using a binary mask, on purpose, to simulate a damaged image. Replacing pixels with 1s turns them totally black. The defective image is saved as an array called defect_image.
# 
# The mask is a black and white image with patches that have the position of the image bits that have been corrupted. We can apply the restoration function on these areas.
# 
# Remember that inpainting is the process of reconstructing lost or deteriorated parts of images and videos.

# ### init

# In[1]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(defect_image, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'defect_image[512_512_3].csv': 'https://file.io/Fdt0o3'}}
"""
prefixToc = '1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
defect_image = getImage(prefix+'defect_image[512_512_3].csv')


# In[2]:


import matplotlib.pyplot as plt

def show_image(image, title='Image', cmap_type='gray', interpolation=None):
    plt.imshow(image, cmap=cmap_type)    
    plt.title(title)
    plt.axis('off')
    plt.show()


# In[3]:


show_image(defect_image)


# In[6]:


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


# ### code

# In[4]:


# Import the module from restoration
from skimage.restoration import inpaint


# In[5]:


# Show the defective image
show_image(defect_image, 'Image to restore')


# In[8]:


# Apply the restoration function to the image using the mask
restored_image = inpaint.inpaint_biharmonic(defect_image, mask, multichannel=True)
show_image(restored_image)


# ## Removing logos
# As we saw in the video, another use of image restoration is removing objects from an scene. In this exercise, we'll remove the Datacamp logo from an image.
# ![image.png](attachment:image.png)
# Landscape with small datacamp logo
# Image loaded as image_with_logo.
# You will create and set the mask to be able to erase the logo by inpainting this area.
# 
# Remember that when you want to remove an object from an image you can either manually delineate that object or run some image analysis algorithm to find it.

# ### init

# In[9]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(image_with_logo, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'image_with_logo[296_512_3].csv': 'https://file.io/D9GPzt'}}
"""
prefixToc = '1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
image_with_logo = getImage(prefix+'image_with_logo[296_512_3].csv')


# In[10]:


show_image(image_with_logo)


# ### code

# In[11]:


# Initialize the mask
mask = np.zeros(image_with_logo.shape[:-1])

# Set the pixels where the logo is to 1
mask[210:272, 360:425] = 1

# Apply inpainting to remove the logo
image_logo_removed = inpaint.inpaint_biharmonic(image_with_logo,
                                  mask,
                                  multichannel=True)

# Show the original and logo removed images
show_image(image_with_logo, 'Image with logo')
show_image(image_logo_removed, 'Image with logo removed')


# # Noise

# ## Let's make some noise!
# In this exercise, we'll practice adding noise to a fruit image.
# ![image.png](attachment:image.png)
# Various fruits
# Image preloaded as fruit_image.

# ### init

# In[12]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(fruit_image, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'fruit_image[417_417_3].csv': 'https://file.io/vYJxw2'}}
"""
prefixToc = '2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
fruit_image = getImage(prefix+'fruit_image[417_417_3].csv')


# In[13]:


show_image(fruit_image)


# ### code

# In[14]:


# Import the module and function
from skimage.util import random_noise

# Add noise to the image
noisy_image = random_noise(fruit_image)

# Show original and resulting image
show_image(fruit_image, 'Original')
show_image(noisy_image, 'Noisy image')


# ## Reducing noise
# We have a noisy image that we want to improve by removing the noise in it.
# ![image.png](attachment:image.png)
# Small cute puppy
# Preloaded as noisy_image.
# Use total variation filter denoising to accomplish this.

# ### init

# In[19]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(noisy_image, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'noisy_image[1200_1600_3].csv': 'https://file.io/wysLvC'}}
"""
prefixToc = '2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
noisy_image = getImage(prefix+'noisy_image[1200_1600_3].csv', dtype='float64')


# In[20]:


show_image(noisy_image)


# ### code

# In[21]:


# Import the module and function
from skimage.restoration import denoise_tv_chambolle

# Apply total variation filter denoising
denoised_image = denoise_tv_chambolle(noisy_image, 
                                      multichannel=True)

# Show the noisy and denoised images
show_image(noisy_image, 'Noisy')
show_image(denoised_image, 'Denoised image')


# ## Reducing noise while preserving edges
# In this exercise, you will reduce the noise in this landscape picture.
# ![image.png](attachment:image.png)
# Landscape of a river
# Preloaded as landscape_image.
# Since we prefer to preserve the edges in the image, we'll use the bilateral denoising filter.

# ### init

# In[22]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(landscape_image, image=True)
"""

tobedownloaded="""
 {numpy.ndarray: {'landscape_image[1216_1824_3].csv': 'https://file.io/gsTIkk'}}
"""
prefixToc = '2.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
landscape_image = getImage(prefix+'landscape_image[1216_1824_3].csv')
show_image(landscape_image)


# ### code

# In[24]:


# Import bilateral denoising function
from skimage.restoration import denoise_bilateral

# Apply bilateral filter denoising
denoised_image = denoise_bilateral(landscape_image, 
                                   multichannel=True)

# Show original and resulting images
show_image(landscape_image, 'Noisy image')
show_image(denoised_image, 'Denoised image')


# # Superpixels & segmentation
# 

# ## Number of pixels
# Let's calculate the total number of pixels in this image.
# ![image.png](attachment:image.png)
# Young woman
# Image preloaded as face_image
# The total amount of pixel is its resolution. Given by Height×Width.
# 
# Use .shape from NumPy which is preloaded as , in the console to check the width and height of the image.

# ### init

# In[25]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(face_image, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'face_image[265_191_3].csv': 'https://file.io/PiYj8g'}}
"""
prefixToc = '3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
face_image = getImage(prefix+'face_image[265_191_3].csv')


# In[26]:


show_image(face_image)


# ### code

# In[27]:


face_image.shape


# In[28]:


face_image.shape[0]*face_image.shape[1]


# ## Superpixel segmentation
# In this exercise, you will apply unsupervised segmentation to the same image, before it's passed to a face detection machine learning model.
# 
# So you will reduce this image from 265×191=50,615 pixels down to 400 regions.
# 
# Young woman
# Already preloaded as face_image.
# The show_image() function has been preloaded for you as well.

# ### code

# In[30]:


# Import the slic function from segmentation module
from skimage.segmentation import slic

# Import the label2rgb function from color module
from skimage.color import label2rgb

# Obtain the segmentation with 400 regions
segments = slic(face_image, n_segments= 400)

# Put segments on top of original image to compare
segmented_image = label2rgb(segments, face_image, kind='avg')

# Show the segmented image
show_image(segmented_image, "Segmented image, 400 superpixels")


# # Finding contours
# 

# ## Contouring shapes
# In this exercise we'll find the contour of a horse.
# 
# For that we will make use of a binarized image provided by scikit-image in its data module. Binarized images are easier to process when finding contours with this algorithm. Remember that contour finding only supports 2D image arrays.
# 
# Once the contour is detected, we will display it together with the original image. That way we can check if our analysis was correct!
# 
# show_image_contour(image, contours) is a preloaded function that displays the image with all contours found using Matplotlib.
# ![image.png](attachment:image.png)
# Shape of a horse in black and white
# Remember you can use the find_contours() function from the measure module, by passing the thresholded image and a constant value.

# ### init

# In[32]:


def show_image_contour(image, contours):
    plt.figure()
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3)
    plt.imshow(image, interpolation='nearest', cmap='gray_r')
    plt.title('Contours')
    plt.axis('off')
    plt.show()


# ### code

# In[33]:


# Import the modules
from skimage import measure, data

# Obtain the horse image
horse_image = data.horse()

# Find the contours with a constant level value of 0.8
contours = measure.find_contours(horse_image, 0.8)

# Shows the image with contours found
show_image_contour(horse_image, contours)


# ## Find contours of an image that is not binary
# Let's work a bit more on how to prepare an image to be able to find its contours and extract information from it.
# 
# We'll process an image of two purple dices loaded as image_dices and determine what number was rolled for each dice.
# ![image.png](attachment:image.png)
# Purple dices
# In this case, the image is not grayscale or binary yet. This means we need to perform some image pre-processing steps before looking for the contours. First, we'll transform the image to a 2D array grayscale image and next apply thresholding. Finally, the contours are displayed together with the original image.
# 
# color, measure and filters modules are already imported so you can use the functions to find contours and apply thresholding.
# 
# We also import io module to load the image_dices from local memory, using imread. Read more here.

# ### init

# In[34]:


#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(image_dices, image=True)
"""

tobedownloaded="""
{numpy.ndarray: {'image_dices[120_120_4].csv': 'https://file.io/QIdFYO'}}
"""
prefixToc = '4.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="10.225.92.1:80")

#initialisation

from downloadfromFileIO import getImage
image_dices = getImage(prefix+'image_dices[120_120_4].csv')

show_image(image_dices)


# In[37]:


from skimage import color
from skimage.filters import threshold_otsu


# ### code

# In[36]:


# Make the image grayscale
image_dices = color.rgb2gray(image_dices)


# In[38]:


# Obtain the optimal thresh value
thresh = threshold_otsu(image_dices)


# In[39]:


# Apply thresholding
binary = image_dices > thresh


# In[40]:


# Find contours at a constant value of 0.8
contours = measure.find_contours(binary, 0.8)

# Show the image
show_image_contour(image_dices, contours)


# ## Count the dots in a dice's image
# Now we have found the contours, we can extract information from it.
# 
# In the previous exercise, we prepared a purple dices image to find its contours:
# ![image.png](attachment:image.png)
# 3 images showing the steps to find contours
# 
# This time we'll determine what number was rolled for the dice, by counting the dots in the image.
# 
# The contours found in the previous exercise are preloaded as contours.
# 
# Create a list with all contour's shapes as shape_contours. You can see all the contours shapes by calling shape_contours in the console, once you have created it.
# 
# Check that most of the contours aren't bigger in size than 50. If you count them, they are the exact number of dots in the image.
# 
# show_image_contour(image, contours) is a preloaded function that displays the image with all contours found using Matplotlib.

# ### code

# In[41]:


# Create list with the shape of each contour
shape_contours = [cnt.shape[0] for cnt in contours]

# Set 50 as the maximum size of the dots shape
max_dots_shape = 50

# Count dots in contours excluding bigger than dots size
dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] < max_dots_shape]

# Shows all contours found 
show_image_contour(binary, contours)

# Print the dice's number
print("Dice's dots number: {}. ".format(len(dots_contours)))


# In[ ]:




