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
       