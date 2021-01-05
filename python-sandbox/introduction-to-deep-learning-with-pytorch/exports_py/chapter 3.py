#!/usr/bin/env python
# coding: utf-8

# # Convolution operator

# ## Convolution operator - OOP way
# 
# Let's kick off this chapter by using convolution operator from the torch.nn package. You are going to create a random tensor which will represent your image and random filters to convolve the image with. Then you'll apply those images.
# 
# The torch library and the torch.nn package have already been imported for you.

# 
#     Create 10 images with shape (1, 28, 28).
#     Build 6 convolutional filters of size (3, 3) with stride set to 1 and padding set to 1.
#     Apply the filters in the image and print the shape of the feature map.
# 

# In[1]:


#init
import torch
import torch.nn 


# In[2]:


# Create 10 random images of shape (1, 28, 28)
images = torch.rand(10, 1, 28, 28)

# Build 6 conv. filters
conv_filters = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3,  padding=1, stride=1)

# Convolve the image with the filters 
output_feature = conv_filters(images)
print(output_feature.shape)


# ## Convolution operator - Functional way
# 
# While I and most of PyTorch practitioners love the torch.nn package (OOP way), other practitioners prefer building neural network models in a more functional way, using torch.nn.functional. More importantly, it is possible to mix the concepts and use both libraries at the same time (we have already done it in the previous chapter). You are going to build the same neural network you built in the previous exercise, but this time using the functional way.
# 
# As before, we have already imported the torch library and torch.nn.functional as F.

# In[4]:


#init
import torch
import torch.nn.functional as F


# In[8]:


# Create 10 random images
image = torch.rand(10, 1, 28, 28)

# Create 6 filters
filters = torch.rand(6, 1, 3, 3)

# Convolve the image with the filters
output_feature = F.conv2d(image, filters, stride=1, padding=1)
print(output_feature.shape)


# # Pooling operators

# ## Max-pooling operator
# 
# Here you are going to practice using max-pooling in both OOP and functional way, and see for yourself that the produced results are the same. We have already created and printed the image for you, and imported torch library in addition to torch.nn and torch.nn.Functional as F packages.

# In[10]:


#init
import torch
import torch.nn
import torch.nn.functional as F

im=torch.tensor([[[[ 8.,  1.,  2.,  5.,  3.,  1.],
          [ 6.,  0.,  0., -5.,  7.,  9.],
          [ 1.,  9., -1., -2.,  2.,  6.],
          [ 0.,  4.,  2., -3.,  4.,  3.],
          [ 2., -1.,  4., -1., -2.,  3.],
          [ 2., -4.,  5.,  9., -7.,  8.]]]])


# In[11]:


# Build a pooling operator with size `2`.
max_pooling = torch.nn.MaxPool2d(2)

# Apply the pooling operator
output_feature = max_pooling(im)

# Use pooling operator in the image
output_feature_F = F.max_pool2d(im, 2)

# print the results of both cases
print(output_feature)
print(output_feature_F)


# ## Average-pooling operator
# 
# After coding the max-pooling operator, you are now going to code the average-pooling operator. You just need to replace max-pooling with average pooling.

# In[13]:


# Build a pooling operator with size `2`.
avg_pooling = torch.nn.AvgPool2d(2)

# Apply the pooling operator
output_feature = avg_pooling(im)

# Use pooling operator in the image
output_feature_F = F.avg_pool2d(im, 2)

# print the results of both cases
print(output_feature)
print(output_feature_F)


# # Convolutional Neural Networks

# ## Your first CNN - __init__ method
# 
# You are going to build your first convolutional neural network. You're going to use the MNIST dataset as the dataset, which is made of handwritten digits from 0 to 9. The convolutional neural network is going to have 2 convolutional layers, each followed by a ReLU nonlinearity, and a fully connected layer. We have already imported torch and torch.nn as nn. Remember that each pooling layer halves both the height and the width of the image, so by using 2 pooling layers, the height and width are 1/4 of the original sizes. MNIST images have shape (1, 28, 28)
# 
# For the moment, you are going to implement the __init__ method of the net. In the next exercise, you will implement the .forward() method.
# 
# NB: We need 2 pooling layers, but we only need to instantiate a pooling layer once, because each pooling layer will have the same configuration. Instead, we will use self.pool twice in the next exercise.

# 
#     Instantiate two convolutional filters: the first one should have 5 channels, while the second one should have 10 channels. The kernel_size for both of them should be 3, and both should use padding=1. Use the names of the arguments (instead of using 1, use padding=1).
#     Instantiate a ReLU() nonlinearity.
#     Instantiate a max pooling layer which halves the size of the image in both directions.
#     Instantiate a fully connected layer which connects the units with the number of classes (we are using MNIST, so there are 10 classes).
# 

# In[15]:


#init
import torch
import torch.nn as nn


# In[17]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)


# ## Your first CNN - forward() method
# 
# Now that you have declared all the parameters of your CNN, all you need to do is to implement the net's forward() method, and voila, you have your very first PyTorch CNN.
# 
# Note: for evaluation purposes, the entire code of the class needs to be in the script. We are using the __init__ method as you have coded it on the previous exercise, while you are going to code the .forward() method here.

# 
#     Apply the first convolutional layer, followed by the relu nonlinearity, then in the next line apply max-pooling layer.
#     Apply the second convolutional layer, followed by the relu nonlinearity, then in the next line apply max-pooling layer.
#     Transform the feature map from 4 dimensional to 2 dimensional space. The first dimension contains the batch size (-1), deduct the second dimension, by multiplying the values for height, width and depth.
#     Apply the fully-connected layer and return the result.
# 

# In[28]:


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
		
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)

    def forward(self, x):

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Prepare the image for the fully connected layer
        x = x.view(-1, 7*7*10)

        # Apply the fully connected layer and return the result
        return self.fc(x)


# # Training Convolutional Neural Networks

# ## Training CNNs
# 
# Similarly to what you did in Chapter 2, you are going to train a neural network. This time however, you will train the CNN you built in the previous lesson, instead of a fully connected network. The packages you need have been imported for you and the network (called net) instantiated. The cross-entropy loss function (called criterion) and the Adam optimizer (called optimizer) are also available. We have subsampled the training set so that the training goes faster, and you are going to use a single epoch.

# In[30]:


#init
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim

# Transform the data to torch tensors and normalize it 
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.1307), ((0.3081)))])

# Prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, 
									  download=True, transform=transform)
testset = torchvision.datasets.MNIST('mnist', train=False,
			   download=True, transform=transform)

# Prepare training loader and testing loader
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
										 shuffle=False, num_workers=0) 

net=Net(1)
optimizer = optim.Adam(net.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()


# In[31]:


type(train_loader)


# In[32]:


train_loader.dataset.train_data.shape


# In[34]:


for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    optimizer.zero_grad()

    # Compute the forward pass
    outputs = net(inputs)
        
    # Compute the loss function
    loss = criterion(outputs, labels)
        
    # Compute the gradients
    loss.backward()
    
    # Update the weights
    optimizer.step()


# ## Using CNNs to make predictions
# 
# Building and training neural networks is a very exciting job (trust me, I do it every day)! However, the main utility of neural networks is to make predictions. This is the entire reason why the field of deep learning has bloomed in the last few years, as neural networks predictions are extremely accurate. On this exercise, we are going to use the convolutional neural network you already trained in order to make predictions on the MNIST dataset.
# 
# Remember that torch.max() takes two arguments: -output.data - the tensor which contains the data.
# 
#     Either 1 to do argmax or 0 to do max.
# 

# In[36]:


# Iterate over the data in the test_loader
for i, data in enumerate(test_loader):
  
    # Get the image and label from data
    inputs, label = data
    
    # Make a forward pass in the net with your image
    output = net(inputs)
    
    # Argmax the results of the net
    _, predicted = torch.max(output.data, 1)
    if predicted == label:
        print("Yipes, your net made the right prediction " + str(predicted))
    else:
        print("Your net prediction was " + str(predicted) + ", but the correct label is: " + str(label))


# In[ ]:




