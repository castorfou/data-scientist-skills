# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 08:56:01 2020

@author: F279814
"""

a = 100
b = 45

print('reste',a % b )
print('division entiere', a // b)

reste = a % b
div_entier = a // b

print (a,'=',b,'x',div_entier,'+',reste)


def gcdRecur(a, b):
    '''
    a, b: positive integers
    
    returns: a positive integer, the greatest common divisor of a & b.
    '''
    # Your code here
    a, b = max(a,b), min(a,b)
    
    if (b==0): return a
    return gcdRecur(b, a%b)
    
print(gcdRecur(a,b))