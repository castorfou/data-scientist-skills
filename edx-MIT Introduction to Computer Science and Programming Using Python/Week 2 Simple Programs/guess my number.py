# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""


low=0
high=100
hlc=''

def getHLC(ans):
    print('Is your secret number '+str(ans)+'?')
    hlc = input("Enter 'h' to indicate the guess is too high. Enter 'l' to indicate the guess is too low. Enter 'c' to indicate I guessed correctly. ")
    while(hlc not in 'hlc' or hlc == ''):
        print("Sorry, I did not understand your input.")
        hlc = input("Enter 'h' to indicate the guess is too high. Enter 'l' to indicate the guess is too low. Enter 'c' to indicate I guessed correctly. ")
    return hlc


ans = (low + high)/2
print("Please think of a number between 0 and 100!")

while(hlc != 'c'):
    
    hlc = getHLC(ans)
    if hlc == 'c':
        print('Game over. Your secret number was:',ans)
    if hlc == 'l':
        low = ans
        ans = int((low + high)/2)
    if hlc == 'h':
        high = ans
        ans = int((low + high)/2)
        
        