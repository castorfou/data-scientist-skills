#!/usr/bin/env python
# coding: utf-8

# # Documentation

# ![image.png](attachment:image.png)

# ## Identifying good comments
# We learned about what characteristics make a 'good' comment. In this exercise, you'll apply this knowledge to identify a function that utilizes comment best practices.

# ### code

# In[1]:


text = 'Our competitor pricing is $10.50 an inch. Our price is $125.00 a foot.'


# In[2]:


import re

def extract_0(text):
    # match and extract dollar amounts from the text
    return re.findall(r'\$\d+\.\d\d', text)

def extract_1(text):
    # return all matches to regex pattern
    return re.findall(r'\$\d+\.\d\d', text)

# Print the text
print(text)

# Print the results of the function with better commenting
print(extract_0(text))


# ![image.png](attachment:image.png)

# ## Identifying proper docstrings
# We covered how to write fully-fledged docstrings. Before writing one of your own, this exercise will help you practice by having you identify a properly formatted docstring.
# 
# In this exercise, you'll be using the functions goldilocks(), rapunzel(), mary(), and sleeping_beauty() which have been loaded in your environment.

# ### init

# In[4]:


"""
import inspect
print_func(goldilocks)
"""
def goldilocks(bear=3):
    """
    :param bear: which number bear's food are you trying to eat; valid numbers: [1, 2, 3]
    :return: description of how the food's temperature is

    >>> goldilocks(bear=1)
    'too hot'
    """
    if bear == 1:
        return 'too hot'
    elif bear == 2:
        return 'too cold'
    elif bear == 3:
        return 'just right'
    else:
        ValueError('There are only 3 bears!')


# In[6]:


"""
import inspect
print_func(rapunzel)
"""
def rapunzel(hair_len=20):
    """Lets down hair from tower to be used as climbing rope

    :param hair_len: length of hair (cannot be negative)
    :return: strand of hair that is hair_len characters long

    >>> rapunzel(hair_len=15)
    '~~~~~~~~~~~~~~~'
    """
    if hair_len < 0:
        ValueError('hair_len cannot be negative!')

    return "~" * hair_len


# In[7]:


"""
import inspect
print_func(mary)
"""
def mary(white_as='snow'):
    """How white was mary's little lamb?

    >>> mary(white_as='salt')
    'Mary had a little lamb whose fleece was white as salt'
    """
    return "Mary had a little lamb whose fleece was white as {}".format(white_as)


# In[8]:


"""
import inspect
print_func(sleeping_beauty)
"""
def sleeping_beauty(awake=False):
    """Should Sleeping Beauty wake up?

    :param awake: if True then wake up; else snooze
    :return: string showing sleepiness or wakefulness
    """
    if awake is True:
        return 'o_o'

    return 'Zzzzz'


# ### code

# In[9]:


# Run the help on all 4 functions
help(goldilocks)
help(rapunzel)
help(mary)
help(sleeping_beauty)


# In[10]:


# Execute the function with most complete docstring
result = rapunzel()

# Print the result
print(result)


# ## Writing docstrings
# We just learned some about the benefits of docstrings. In this exercise, you will practice writing docstrings that can be utilized by a documentation generator like Sphinx.
# 
# Note that your docstring submission must match the solution exactly. If you find yourself getting it wrong several times, it may be a good idea to refresh the sample code and start over.

# ### code

# In[12]:


# Complete the function's docstring
def tokenize(text, regex=r'[a-zA-z]+'):
  """Split text into tokens using a regular expression

  :param text: text to be tokenized
  :param regex: regular expression used to match tokens using re.findall 
  :return: a list of resulting tokens

  >>> tokenize('the rain in spain')
  ['the', 'rain', 'in', 'spain']
  """
  return re.findall(regex, text, flags=re.IGNORECASE)

# Print the docstring
help(tokenize)


# # Readability counts
# 

# ## Using good function names
# A good function name can go a long way for both user and maintainer understanding. A good function name is descriptive and describes what a function does. In this exercise, you'll choose a name for a function that will help aid in its readability when used.

# ### code

# In[13]:


import math


# In[14]:


def hypotenuse_length(leg_a, leg_b):
    """Find the length of a right triangle's hypotenuse

    :param leg_a: length of one leg of triangle
    :param leg_b: length of other leg of triangle
    :return: length of hypotenuse
    
    >>> hypotenuse_length(3, 4)
    5
    """
    return math.sqrt(leg_a**2 + leg_b**2)


# Print the length of the hypotenuse with legs 6 & 8
print(hypotenuse_length(6, 8))


# ## Using good variable names
# Just like functions, descriptive variable names can make your code much more readable. In this exercise, you'll write some code using good variable naming practices.
# 
# There's not always a clear best name for a variable. The exercise has been written to try and make a clear best choice from the provided options.

# ### code

# In[15]:


from statistics import mean

# Sample measurements of pupil diameter in mm
pupil_diameter = [3.3, 6.8, 7.0, 5.4, 2.7]

# Average pupil diameter from sample
mean_diameter = mean(pupil_diameter)

print(mean_diameter)


# ## Refactoring for readability
# Refactoring longer functions into smaller units can help with both readability and modularity. In this exercise, you will refactor a function into smaller units. The function you will be refactoring is shown below. Note, in the exercise, you won't be using docstrings for the sake of space; in a real application, you should include documentation!
# 
# ```
# def polygon_area(n_sides, side_len):
#     """Find the area of a regular polygon
# 
#     :param n_sides: number of sides
#     :param side_len: length of polygon sides
#     :return: area of polygon
# 
#     >>> round(polygon_area(4, 5))
#     25
#     """
#     perimeter = n_sides * side_len
# 
#     apothem_denominator = 2 * math.tan(math.pi / n_sides)
#     apothem = side_len / apothem_denominator
# 
#     return perimeter * apothem / 2
# ```

# ### code

# In[16]:


def polygon_perimeter(n_sides, side_len):
    return n_sides * side_len

def polygon_apothem(n_sides, side_len):
    denominator = 2 * math.tan(math.pi / n_sides)
    return side_len / denominator

def polygon_area(n_sides, side_len):
    perimeter = polygon_perimeter(n_sides, side_len)
    apothem = polygon_apothem(n_sides, side_len)

    return perimeter * apothem / 2

# Print the area of a hexagon with legs of size 10
print(polygon_area(n_sides=6, side_len=10))


# # Unit testing
# 

# ## Using doctest
# We just learned about doctest, which, if you're writing full docstrings with examples, is a simple way to minimally test your functions. In this exercise, you'll get some hands-on practice testing and debugging with doctest.
# 
# The following have all be pre-loaded in your environment: doctest, Counter, and text_analyzer.
# 
# Note that your docstring submission must match the solution exactly. If you find yourself getting it wrong several times, it may be a good idea to refresh the sample code and start over.

# ### code

# In[18]:


def sum_counters(counters):
    """Aggregate collections.Counter objects by summing counts

    :param counters: list/tuple of counters to sum
    :return: aggregated counters with counts summed

    >>> d1 = text_analyzer.Document('1 2 fizz 4 buzz fizz 7 8')
    >>> d2 = text_analyzer.Document('fizz buzz 11 fizz 13 14')
    >>> sum_counters([d1.word_counts, d2.word_counts])
    Counter({'buzz': 2, 'fizz': 4})
    """
    return sum(counters, Counter())

doctest.testmod()


# ## Using pytest
# doctest is a great tool, but it's not nearly as powerful as pytest. In this exercise, you'll write tests for your SocialMedia class using the pytest framework.

# ### code

# In[ ]:


from collections import Counter
from text_analyzer import SocialMedia

# Create an instance of SocialMedia for testing
test_post = 'learning #python & #rstats is awesome! thanks @datacamp!'
sm_post = SocialMedia(test_post)

# Test hashtag counts are created properly
def test_social_media_hashtags():
    expected_hashtag_counts = Counter({'#python': 1, '#rstats': 1})
    assert sm_post.hashtag_counts == expected_hashtag_counts


# # Documentation & testing in practice
# 

# ## Documenting classes for Sphinx
# sphinx is a great tool for rendering documentation as HTML. In this exercise, you'll write a docstring for a class that can be taken advantage of by sphinx.
# 
# Note that your docstring submission must match the solution exactly. If you find yourself getting it wrong several times, it may be a good idea to refresh the sample code and start over.

# ### code

# In[19]:


from text_analyzer import Document

class SocialMedia(Document):
    """Analyze text data from social media
    
    :param text: social media text to analyze

    :ivar hashtag_counts: Counter object containing counts of hashtags used in text
    :ivar mention_counts: Counter object containing counts of @mentions used in text
    """
    def __init__(self, text):
        Document.__init__(self, text)
        self.hashtag_counts = self._count_hashtags()
        self.mention_counts = self._count_mentions()


# In[ ]:




