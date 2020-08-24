# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:22:51 2020

@author: F279814
"""

x = 12
def g(x):
    x = x + 1
    def h(y):
        return x + y
    return h(6)
print(g(x))