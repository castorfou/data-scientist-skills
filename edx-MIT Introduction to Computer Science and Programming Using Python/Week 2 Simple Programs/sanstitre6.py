# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:31:23 2020

@author: F279814
"""

def oddTuples(aTup):
    '''
    aTup: a tuple
    
    returns: tuple, every other element of aTup. 
    '''
    # Your Code Here
    keep = ()
    rankT = 0
    for t in aTup:
        if (rankT % 2) == 0:
            keep += (t,)
        rankT += 1
    return keep


testTuple = (1, 2, 'tr', 4, 'jhr', True, False)

print(oddTuples(testTuple))