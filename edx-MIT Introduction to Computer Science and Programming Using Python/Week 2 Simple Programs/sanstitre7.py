# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:56:00 2020

@author: F279814
"""

def applyToEach(L, f):
    for i in range(len(L)):
        L[i] = f(L[i])
        
        
testList = [1, -4, 8, -9]

def timesFive(a):
    return a * 5

testList = [1, -4, 8, -9]
applyToEach(testList, timesFive)
print(testList)

testList = [1, -4, 8, -9]
applyToEach(testList, abs)
print(testList)

testList = [1, -4, 8, -9]
def plus_one(n):
    return n+1
applyToEach(testList, plus_one)
print(testList)

testList = [1, -4, 8, -9]
def square(n):
    return n**2
applyToEach(testList, square)
print(testList)
