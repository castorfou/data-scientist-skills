# Import needed packages
import re


# Define helper function
def tokenize(text, regex):
    """function to tokenize text with a regex"""
    return re.findall(regex, text)


# Define song lyric lines to tokenize
lines = ['Row, row, row your boat',
         'Gently down the stream',
         'Merrily, merrily, merrily, merrily',
         'Life is but a dream']

# Iterate over each line, tokenize, and print result
for line in lines:
    tokens = tokenize(text=line, regex=r'[a-zA-Z]+')
    print(tokens)
