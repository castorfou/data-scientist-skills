# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:24:21 2020

@author: F279814
"""


test = 'acdefdgsdhnhhglkgvxwguy'
caractere = 'b'

def isIn(char, aStr):
    '''
    char: a single character
    aStr: an alphabetized string
    
    returns: True if char is in aStr; False otherwise
    '''
    # Your code here
    #if string empty
    if not aStr:
        return False
    #if string is a char
    if len(aStr) == 1:
        if aStr == char:
            return True
        else:
            return False
    longueur = len(aStr) // 2
    charMilieu = aStr[longueur]
    subStr1 = aStr[0:longueur]
    subStr2 = aStr[longueur+1: len(aStr)]
    
    print('longueur',longueur)
    if (char == aStr[longueur]):
        return True
    else:
        if char < aStr[longueur]:
            return isIn(char, subStr1)
        else:
            return isIn(char, subStr2)
        
        
def sortString(unSortedString):
    sorted_characters = sorted(unSortedString)
    a_string = "".join(sorted_characters)
    return a_string
    
    
print(sortString(test))   
print(isIn(caractere, sortString(test)))
        