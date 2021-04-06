# -*- coding: utf-8 -*-
"""
test plot
Created on Tue Jul 23 14:20:57 2019

@author: N561507
"""

#%% plot loos_function
import math
import numpy as np

def loss_function(x0):
    result0=x0-1
    result1=4.0*np.cos(result0)
    result2=np.cos(2.0*np.pi*x0)
    result = result1+np.divide(result2,x0)
    return result

x= np.arange(0.01,6.0,0.1)
y=loss_function(x)
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.title('Plot of input values against losses')
plt.xlabel('x')
plt.ylabel('loss_function(x)')
plt.show()