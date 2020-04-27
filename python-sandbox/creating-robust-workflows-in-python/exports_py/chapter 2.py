#!/usr/bin/env python
# coding: utf-8

# # Type hints
# 

# ## TextFile hints
# In this exercise, we will go back to the example of a TextFile class that can represent any text file.
# 
# As in the previous chapter, TextFile stores file contents in an instance variable called text.
# 
# This time our TextFile class will have a get_lines() method that returns all of the lines from the text file used to instantiate the TextFile class.
# 
# The TextFile class definition is ready, but we want to add type hints, so that calling help() on the TextFile class will provide type annotation information to users.
# 
# To help getting things started, we've already imported the List class from the typing module.

# ### code

# In[2]:


from typing import List


# In[3]:


class TextFile:
  	# Add type hints to TextFile"s __init__() method
    def __init__(self, name: str) -> None:
        self.text = Path(name).read_text()

	# Type annotate TextFile"s get_lines() method
    def get_lines(self) -> List[str]:
        return self.text.split("\n")

help(TextFile)


# ## MatchFinder hints
# In the video, we discussed how we can introduce flexibility into type hints with the Optional (None and one of any other type) class.
# 
# In this exercise, we'll design a class called MatchFinder that has a method called get_matches().
# 
# MatchFinder should only accept a list of strings as its strings argument and then store the input list in an instance variable called strings.
# 
# The get_matches() method returns a list of either
# 
# every string in strings that contains the query argument or
# all strings in strings if the match argument is None.
# The typing module's List and Optional classes have already been imported.

# ### code

# In[5]:


from typing import Optional


# In[6]:


class MatchFinder:
  	# Add type hints to __init__()'s strings argument
    def __init__(self, strings: List[str]) -> None:
        self.strings = strings

	# Type annotate get_matches()'s query argument
    def get_matches(self, query: Optional[str] = None) -> List[str]:
        return [s for s in self.strings if query in s] if query else self.strings

help(MatchFinder)


# # Docstrings

# ## Get matches docstring
# We'll add a docstring with doctest examples to the get_matches() function.
# 
# This time, we've built the docstring from multiple single-quoted strings.
# 
# The three strings that will be combined thanks to the wrapping parentheses.
# 
# The superpowers of docstrings stem from their location, not from triple quotes!
# 
# The doctest examples will show how to use get_matches() to find a matching character in a list of strings.
# 
# Call help() on get_matches() if you want to view the final docstring without the newline ('\n') and tab ('\t') characters and interspersed code comments.
# 
# If you get stuck, run the example code in the IPython console!

# ### code

# In[7]:


def get_matches(word_list: List[str], query:str) -> List[str]:
    ("Find lines containing the query string.\nExamples:\n\t"
     # Complete the docstring example below
     ">>> get_matches(['a', 'list', 'of', 'words'], 's')\n\t"
     # Fill in the expected result of the function call
     "['list', 'words']")
    return [line for line in word_list if query in line]

help(get_matches)


# ## Obtain words docstring
# In the obtain_words() function's docstring, we will obtain the title of a poem about Python.
# 
# Later, we can use doctest to compare the actual and expected result.
# 
# If you are not sure what the result should be, run the the first line of code from the Examples section of the docstring in the IPython console!
# 
# Examples:
#     >>> from this import s
# Call help() on obtain_words() to see the final docstring without the interspersed code comments and the tab ('\t') and newline ('\n') characters.

# ### code

# In[8]:


def obtain_words(string: str) -> List[str]:
    ("Get the top words in a word list.\nExamples:\n\t"
     ">>> from this import s\n\t>>> from codecs import decode\n\t"
     # Use obtain_words() in the docstring example below
     ">>> obtain_words(decode(s, encoding='rot13'))[:4]\n\t"
     # Fill in the expected result of the function call
     "['The', 'Zen', 'of', 'Python']") 
    return ''.join(char if char.isalpha() else ' ' for char in string).split()
  
help(obtain_words)


# # Reports

# ## Build notebooks
# The first function we will define for our new Python package is called nbuild().
# 
# nbuild() will
# 
# - Create a new notebook with the new_notebook() function from the v4 module of nbformat
# - Read the file contents from a list of source files with the read_file() function that we have used in previous exercises
# - Pass the file contents to the new_code_cell() or new_markdown_cell() functions from the v4 module of nbformat
# - Assign a list of the resulting cells to the 'cells' key in the new notebook
# - Return the notebook instance
# 
# 
# We've already imported nbformat and all of the above functions.
# 
# With nbuild(), we will be able to create Jupyter notebooks from small, modular files!

# ### init

# In[11]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
"intro.md", "plot.py", "discussion.md"

uploadToFileIO_pushto_fileio('intro.md')
uploadToFileIO_pushto_fileio('plot.py')
uploadToFileIO_pushto_fileio('discussion.md')

{"success":true,"key":"x7i5BTVc","link":"https://file.io/x7i5BTVc","expiry":"14 days"}
{"success":true,"key":"y8MaoFUf","link":"https://file.io/y8MaoFUf","expiry":"14 days"}
{"success":true,"key":"8H08SuIG","link":"https://file.io/8H08SuIG","expiry":"14 days"}

"""

tobedownloaded="""
{files: {'intro.md': 'https://file.io/x7i5BTVc',
'plot.py': 'https://file.io/y8MaoFUf',
'discussion.md': 'https://file.io/8H08SuIG'}}
"""
prefixToc = '3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)


# ### code

# In[10]:


from nbformat.v4 import (new_notebook,
new_code_cell, new_markdown_cell)
import nbformat
from nbconvert.exporters import HTMLExporter
from nbconvert.exporters import get_exporter
from pathlib import Path

from pprint import pprint


# In[13]:


def nbuild(filenames: List[str]) -> nbformat.notebooknode.NotebookNode:
    """Create a Jupyter notebook from text files and Python scripts."""
    nb = new_notebook()
    nb.cells = [
        # Create new code cells from files that end in .py
        new_code_cell(Path(name).read_text()) 
        if name.endswith(".py")
        # Create new markdown cells from all other files
        else new_markdown_cell(Path(name).read_text()) 
        for name in filenames
    ]
    return nb
    
pprint(nbuild([prefix+"intro.md", prefix+"plot.py", prefix+"discussion.md"]))


# ## Convert notebooks
# nbconvert is very flexible and includes exporter classes that can convert notebooks into many formats, including: 'asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', and 'slides'.
# 
# We'll write a single function, called nbconv(), that can export to any of these formats.
# 
# To do this, we'll use the get_exporter() function from the exporters module of nbconvert.
# 
# After instantiating an exporter, we'll use its from_filename() method to obtain the contents of the converted file that will be returned by nbconv().
# 
# The from_filename() method also produces a dictionary of metadata that we will not use in this exercise.
# 
# Unlike nbuild(), nbconv() will return a string, rather than a NotebookNode object.

# ### init

# In[14]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
"mynotebook.ipynb "

uploadToFileIO_pushto_fileio('mynotebook.ipynb')

{"success":true,"key":"yO8AAj4Z","link":"https://file.io/yO8AAj4Z","expiry":"14 days"}
"""

tobedownloaded="""
{files: {'mynotebook.ipynb': 'https://file.io/yO8AAj4Z'}}
"""
prefixToc = '3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)


# ### code

# In[15]:


def nbconv(nb_name: str, exporter: str = "script") -> str:
    """Convert a notebook into various formats using different exporters."""
    # Instantiate the specified exporter class
    exp = get_exporter(exporter)()
    # Return the converted file"s contents string 
    return exp.from_filename(nb_name)[0]
    
pprint(nbconv(nb_name=prefix+"mynotebook.ipynb", exporter="html"))


# # Pytest

# ## Parametrize
# Docstring examples are great, because they are included in Sphinx documentation and testable with doctest, but now we are ready to take our testing to the next level with pytest.
# 
# Writing pytest tests
# 
# - is less cumbersome than writing docstring examples (no need for >>> or ...)
# - allows us to leverage awesome features like the parametrize decorator.
# 
# The arguments we will pass to parametrize are
# 
# a name for the list of arguments and
# the list of arguments itself.
# In this exercise, we will define a test_nbuild() function to pass three different file types to the nbuild() function and confirm that the output notebook contains the input file in its first cell.
# 
# We will use a custom function called show_test_output() to see the test output.

# ### init

# In[47]:


###################
##### inspect Function
###################

""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(show_test_output)
print_func(check_output)

"""
import inspect
from subprocess import CalledProcessError, check_output
def show_test_output(func):
    source = inspect.getsource(func)
    with open('test_file.py', 'w') as file:
        file.write("import pytest\n\nfrom nbuild import nbuild\nfrom pathlib import Path\n\n")
        file.write(source)

    try:
        output = check_output(["pytest", "test_file.py"])
        print(output.decode())
    except CalledProcessError as e:
        print(e.output.decode())

        
from typing import List
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

def nbuild(filenames: List[str]) -> nbformat.notebooknode.NotebookNode:
    '''Create a Jupyter notebook from text files and Python scripts'''
    # Create a new notebook object
    nb = new_notebook()
    nb.cells.extend(
        # Create new code cells from files that end in .py
        new_code_cell(Path(name).read_text())
        if name.endswith('.py')
        # Create new markdown cells from all other files
        else new_markdown_cell(Path(name).read_text()) 
        for name in filenames
    )
    return nb
       


# ### code

# In[48]:


import pytest


# In[49]:


# Fill in the decorator for the test_nbuild() function 
@pytest.mark.parametrize("inputs", ["intro.md", "plot.py", "discussion.md"])
# Pass the argument set to the test_nbuild() function
def test_nbuild(inputs):
    assert nbuild([inputs]).cells[0].source == Path(inputs).read_text()

show_test_output(test_nbuild)


# ## Raises
# In this coding exercise, we will define a test function called test_nbconv() that will use the
# 
# @parametrize decorator to pass three unsupported arguments to our nbconv() function
# raises() function to make sure that passing each incorrect argument to nbconv() results in a ValueError
# As in the previous exercise, we will use show_test_output() to see the test output.
# 
# To see an implementation of this test and others, checkout the Nbless package documentation.

# ### code

# In[50]:


@pytest.mark.parametrize("not_exporters", ["htm", "ipython", "markup"])
# Pass the argument set to the test_nbconv() function
def test_nbconv(not_exporters):
     # Use pytest to confirm that a ValueError is raised
    with pytest.raises(ValueError):
        nbconv(nb_name="mynotebook.ipynb", exporter=not_exporters)

show_test_output(test_nbconv)


# In[ ]:




