#!/usr/bin/env python
# coding: utf-8

# # Project templates
# 

# ## Set up template
# For our first project template exercise, we will write a cookiecutter.json file that will contain defaults for our project template.
# 
# Our cookiecutter.json file will contain three keys:
# 
# - project
# - package
# - license
# 
# The package key's value is a Jinja2 template string that will use the project key's value to create a snake_case package name by converting the input string to lowercase and replacing spaces with underscores.
# 
# Inside the double curly braces ({{}}) of the Jinja2 template string, we can use any Python code necessary to create the desired final value.
# 
# The license key's value is a list of possible license types:
# 
# - MIT
# - BSD
# - GPL3

# ### code

# In[1]:


from pprint import pprint


# In[ ]:


json_path.write_text(json.dumps({
    "project": "Creating Robust Python Workflows",
  	# Convert the project name into snake_case
    "package": "{{ cookiecutter.project.lower().replace(' ', '_') }}",
    # Fill in the default license value
    "license": ["MIT", "BSD", "GPL3"]
}))

pprint(json.loads(json_path.read_text()))


# ## Create project
# In this project template exercise, we will first list the keys in a local cookiecutter.json file.
# 
# The paths to the template directory and its cookiecutter.json are assigned to template_root and json_path variables, respectively.
# 
# While template_root is a string, json_path is a pathlib.Path object.
# 
# We will use the json module to obtain cookiecutter.json file contents as a Python dictionary and unpack this dictionary into a list to see its keys.
# 
# We need to see the keys in the cookiecutter.json file to know how override the default project name in the template, because the key in the extra_context argument passed to the cookiecutter() function must match the corresponding key in cookiecutter.json.

# ### code

# In[ ]:


# Obtain keys from the local template's cookiecutter.json
keys = [*json.load(json_path.open())]
vals = "Your name here", "My Amazing Python Project"

# Create a cookiecutter project without prompting for input
main.cookiecutter(template_root.as_posix(), no_input=True,
                  extra_context=dict(zip(keys, vals)))

for path in pathlib.Path.cwd().glob("**"):
    print(path)


# # Executable projects
# 

# ## Zipapp
# In this exercise, we will
# 
# zip up a project called myproject
# make the zipped project command-line executable
# create a __main__.py file in the zipped project
# all with a single call to the create_archive() function from the standard library zipapp module.
# 
# The python interpreter we want to use is /usr/bin/env python,
# 
# while the function we want __main__.py to run is called print_name_and_file():
# 
# def print_name_and_file():
#     print(f"Name is {__name__}. File is {__file__}.")
# The print_name_and_file() function is in the mymodule.py file inside the top-level mypackage directory, as shown below:
# 
# myproject
# └── mypackage
#     ├── __init__.py
#     └── mymodule.pyµ

# ### code

# In[ ]:


zipapp.create_archive(
    # Zip up a project called "myproject"
    "myproject",                    
    interpreter="/usr/bin/env python",
    # Generate a __main__.py file
    main="mypackage.mymodule:print_name_and_file")

print(subprocess.run([".venv/bin/python", "myproject.pyz"],
                     stdout=-1).stdout.decode())


# ## Argparse main()
# Next, we'll create a __main__.py file to pass shell arguments to classify() or regress(), functions based on code we wrote in Chapter 1.
# 
# We will provide default values for all arguments, so that the code can run even if no shell arguments are provided.
# 
# To do this, we'll instantiate the ArgumentParser class from the argparse module as parser and use its add_argument() method to create arguments called dataset and model with the following defaults: diabetes and linear_model.Ridge.
# 
# Setting nargs to ? means that each argument can accept either one value or none at all.
# 
# We will create a keyword arguments (kwargs) variable and unpack kwargs into the classify() or regress() functions in the main() function's return statement.

# ### code

# In[ ]:


def main():
    parser = argparse.ArgumentParser(description="Scikit datasets only!")
    # Set the default for the dataset argument
    parser.add_argument("dataset", nargs="?", default="diabetes")
    parser.add_argument("model", nargs="?", default="linear_model.Ridge")
    args = parser.parse_args()
    # Create a dictionary of the shell arguments
    kwargs = dict(dataset=args.dataset, model=args.model)
    return (classify(**kwargs) if args.dataset in ("digits", "iris", "wine")
            else regress(**kwargs) if args.dataset in ("boston", "diabetes")
            else print(f"{args.dataset} is not a supported dataset!"))

if __name__ == "__main__":
    main()


# # Notebook workflows
# 

# ## Parametrize notebooks
# To practice notebook parametrization, we will work with a Jupyter notebook called sklearn.ipynb.
# 
# This notebook can run any scikit-learn model on any built-in scikit-learn dataset.
# 
# The dataset and model that the notebook will use depend on the four parameters it receives.
# 
# To find the parameter names, we will use papermill to look at the source attribute of an nbformat NotebookNode object cell.
# 
# We will need the parameter names to create a dictionary of parameters that we will then use to execute the notebook.

# ### code

# In[ ]:


# Read in the notebook to find the default parameter names
pprint(nbformat.read("sklearn.ipynb", as_version=4).cells[0].source)
keys = ["dataset_name", "model_type", "model_name", "hyperparameters"]
vals = ["diabetes", "ensemble", "RandomForestRegressor",
        dict(max_depth=3, n_estimators=100, random_state=0)]
parameter_dictionary = dict(zip(keys, vals))

# Execute the notebook with custom parameters
pprint(pm.execute_notebook(
    "sklearn.ipynb", "rf_diabetes.ipynb", 
    kernel_name="python3", parameters=parameter_dictionary
	))


# ## Summarize notebooks
# In the last notebook workflow exercise, we will use scrapbook to
# 
# read in a Jupyter notebook called rf_diabetes.ipynb
# create a dataframe that contains variables that were saved in the notebook with the glue() function
# create a second dataframe of parameters that were passed to the notebook by papermill
# This exercise demonstrates how we can use scrapbook to access notebook data.

# ### code

# In[ ]:


import scrapbook as sb

# Read in the notebook and assign the notebook object to nb
nb = sb.read_notebook("rf_diabetes.ipynb")

# Create a dataframe of scraps (recorded values)
scrap_df = nb.scrap_dataframe
print(scrap_df)


# # Parallel computing
# 

# ## Dask dataframe
# To practice working with Dask dataframes, we will
# 
# - read in a .csv file containing the diabetes dataset as Dask dataframe,
# - create a new binary variable from the age column, and
# - compute the means of all variables for the resulting two age groups.
# 
# The code in this exercise could easily be adapted to work with a Pandas dataframe instead of a Dask dataframe.

# ### init

# In[3]:


###################
##### file
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO_pushto_fileio('diabetes.csv')
"""

tobedownloaded="""
{numpy.ndarray: {'diabetes.csv': 'https://file.io/scSnKGFJ'}}
"""
prefixToc = '4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)


#  ### code

# In[7]:


import dask.dataframe as dd

# Read in a csv file using a dask.dataframe method
df = dd.read_csv(prefix+"diabetes.csv")

df["bin_age"] = (df.age > 0).astype(int)

# Compute the columns means in the two age groups
print(df.groupby("bin_age").mean().head())


# ## Joblib
# In the last exercise of this course, we will use the grid search technique to find the optimal hyperparameters for an elastic net model.
# 
# Grid search is computationally intensive. To speed up the search, we will use the joblib parallel_backend() function.
# 
# The scikit-learn GridSearchCV class has already been instantiated as engrid with a grid of two hyperparameters:
# 
# l1_ratio: the mix of Lasso (L1) and Ridge (L2) regression penalties used to shrink model coefficients
# alpha: the severity of the penalty
# Applying penalties to model coefficients helps to avoid overfitting and produce models that perform better on new data.
# 
# We will use the optimal l1_ratio to create a enet_path() plot that shows how coefficients shrink as alpha increases.

# ### code

# In[ ]:


# Set up a Dask client with 4 threads and 1 worker
Client(processes=False, threads_per_worker=4, n_workers=1)

# Run grid search using joblib and a Dask backend
with joblib.parallel_backend("dask"):
    engrid.fit(x_train, y_train)

plot_enet(*enet_path(x_test, y_test, eps=5e-5, fit_intercept=False,
                    l1_ratio=engrid.best_params_["l1_ratio"])[:2])

