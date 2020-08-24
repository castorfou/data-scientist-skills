# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:20:09 2020

@author: F279814
"""

def a(x, y, z):
     if x:
         return y
     else:
         return z

def b(q, r):
    return a(q>r, q, r)