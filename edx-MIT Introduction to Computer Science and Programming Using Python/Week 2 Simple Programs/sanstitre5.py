# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:01:51 2020

@author: F279814
"""

import math

#nb de cote
n = 4
#length
s = 10

def polysum(n, s):
    area = 0.25 * n * (s ** 2)/math.tan(math.pi/n)
    perimeter = n * s
    return round(area, 4)+perimeter**2
    
print(polysum(n,s))