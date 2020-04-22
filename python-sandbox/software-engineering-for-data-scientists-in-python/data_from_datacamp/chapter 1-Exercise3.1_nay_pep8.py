#define song lyric lines to tokenize
lines=[
  'Row, row, row your boat',
 'Gently down the stream',
     'Merrily, merrily, merrily, merrily',
'Life is but a dream']
#import needed packages
import re
#define helper function
def tokenize(text,regex):
 'function to tokenize text with a regex'
 return re.findall(regex, text)

#iterate over each line, tokenize, and print result
for line in lines:
          tokens=tokenize(text=  line,regex  = r'[a-zA-Z]+')
          print(  tokens )