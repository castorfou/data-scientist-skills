#!/usr/bin/env python
# coding: utf-8

# # DataFrames and Series

# # Clean and Validate
# 
# ```python
# pounds.value_counts().sort_index()
# 
# # Replace
# pounds = pounds.replace([98, 99], np.nan)
# ```

# # Filter and visualize
# 
# ```python
# # Histogram
# import matplotlib.pyplot as plt
# plt.hist(birth_weight.dropna(), bins=30)
# plt.xlabel('Birth weight (lb)')
# plt.ylabel('Fraction of births')
# plt.show()
# 
# # Filtering
# # Other logical operators:
# # & for AND (both must be true)
# # | for OR (either or both can be true)
# # Example:
# birth_weight[A & B] # both true
# birth_weight[A | B] # either or both true
# ```

# In[ ]:




