#!/usr/bin/env python
# coding: utf-8

# # Working with a DataSet to Create DataFrames
# 

# ## Return Statement II: The Return of the DataShell
# Let's now go back to the class DataShell that we were working with earlier, and refactor it such that it uses the return statement instead of the print() function.
# 
# Notice that since we are now using the return statement, we need to include our calls to object methods within the print() function.

# ### code

# In[2]:


integer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# In[3]:


# Create class: DataShell
class DataShell:
  
	# Initialize class with self and dataList as arguments
    def __init__(self, dataList):
      	# Set data as instance variable, and assign it the value of dataList
        self.data = dataList
        
	# Define method that returns data: show
    def show(self):
        return self.data
        
    # Define method that prints average of data: avg 
    def avg(self):
        # Declare avg and assign it the average of data
        avg = sum(self.data)/float(len(self.data))
        # Return avg
        return avg
        
# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Print output of your object's show method
print(my_data_shell.show())

# Print output of your object's avg method
print(my_data_shell.avg())


# ## Return Statement III: A More Powerful DataShell
# In this exercise our DataShell class will evolve from simply consuming lists to consuming CSV files so that we can do more sophisticated things.
# 
# For this, we will employ the return statement once again. Additionally, we will leverage some neat functionality from both the numpy and pandas packages.

# ### code

# In[4]:


us_life_expectancy = 'https://assets.datacamp.com/production/repositories/2097/datasets/5dd3a8250688a4f08306206fa1d40f63b66bc8a9/us_life_expectancy.csv'


# In[5]:


# Load numpy as np and pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
  
    # Initialize class with self and inputFile
    def __init__(self, inputFile):
        self.file = inputFile
        
    # Define generate_csv method, with self argument
    def generate_csv(self):
        self.data_as_csv = pd.read_csv(self.file)
        return self.data_as_csv

# Instantiate DataShell with us_life_expectancy as input argument
data_shell = DataShell(us_life_expectancy)

# Call data_shell's generate_csv method, assign it to df
df = data_shell.generate_csv()

# Print df
print(df)


# # Renaming Columns and the Five-Figure Summary
# 

# ## Data as Attributes
# In the previous coding exercise you wrote a method within your DataShell class that returns a Pandas Dataframe.
# 
# In this one, we will cook the data into our class, as an instance variable. This is so that we can do fancy things later, such as renaming columns, as well as getting descriptive statistics.
# 
# The object us_life_expectancy is loaded and available in your workspace.

# ### code

# In[6]:


us_life_expectancy = 'https://assets.datacamp.com/production/repositories/2097/datasets/5dd3a8250688a4f08306206fa1d40f63b66bc8a9/us_life_expectancy.csv'


# In[7]:


# Import numpy as np, pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
  
    # Define initialization method
    def __init__(self, filepath):
        # Set filepath as instance variable  
        self.filepath = filepath
        # Set data_as_csv as instance variable
        self.data_as_csv = pd.read_csv(filepath)

# Instantiate DataShell as us_data_shell
us_data_shell = DataShell(us_life_expectancy)

# Print your object's data_as_csv attribute
print(us_data_shell.data_as_csv)


# ## Renaming Columns
# Methods can be especially useful to manipulate their object's data. In this exercise, we will create a method inside of our DataShell class, so that we can rename our data columns.
# 
# numpy and pandas are already imported in your workspace as np and pd, respectively.

# In[8]:


# Create class DataShell
class DataShell:
  
    # Define initialization method
    def __init__(self, filepath):
        self.filepath = filepath
        self.data_as_csv = pd.read_csv(filepath)
    
    # Define method rename_column, with arguments self, column_name, and new_column_name
    def rename_column(self, column_name, new_column_name):
        self.data_as_csv.columns = self.data_as_csv.columns.str.replace(column_name, new_column_name)

# Instantiate DataShell as us_data_shell with argument us_life_expectancy
us_data_shell = DataShell(us_life_expectancy)

# Print the datatype of your object's data_as_csv attribute
print(us_data_shell.data_as_csv.dtypes)

# Rename your objects column 'code' to 'country_code'
us_data_shell.rename_column('code', 'country_code')

# Again, print the datatype of your object's data_as_csv attribute
print(us_data_shell.data_as_csv.dtypes)


# ## Self-Describing DataShells
# In this exercise you will add functionality to your DataShell class such that it returns information about itself.
# 
# numpy and pandas are already imported in your workspace as np and pd, respectively.

# ### code

# In[9]:


# Create class DataShell
class DataShell:

    # Define initialization method
    def __init__(self, filepath):
        self.filepath = filepath
        self.data_as_csv = pd.read_csv(filepath)

    # Define method rename_column, with arguments self, column_name, and new_column_name
    def rename_column(self, column_name, new_column_name):
        self.data_as_csv.columns = self.data_as_csv.columns.str.replace(column_name, new_column_name)
        
    # Define get_stats method, with argument self
    def get_stats(self):
        # Return a description data_as_csv
        return self.data_as_csv.describe()
    
# Instantiate DataShell as us_data_shell
us_data_shell = DataShell(us_life_expectancy)

# Print the output of your objects get_stats method
print(us_data_shell.get_stats())


# # OOP Best Practices
# 

# In[ ]:




