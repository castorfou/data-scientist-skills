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
    "package": "{{ cookiecutter.project.____().replace(' ', '_') }}",
    # Fill in the default license value
    "license": ["____", "BSD", "GPL3"]
}))

pprint(json.loads(json_path.read_text()))

