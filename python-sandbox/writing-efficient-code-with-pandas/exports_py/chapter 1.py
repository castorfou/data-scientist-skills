#!/usr/bin/env python
# coding: utf-8

# # The need for efficient coding I
# 

# ## Measuring time I
# In the lecture slides, you saw how the time.time() function can be loaded and used to assess the time required to perform a basic mathematical operation.
# 
# Now, you will use the same strategy to assess two different methods for solving a similar problem: calculate the sum of squares of all the positive integers from 1 to 1 million (1,000,000).
# 
# Similar to what you saw in the video, you will compare two methods; one that uses brute force and one more mathematically sophisticated.
# 
# In the function formula, we use the standard formula
# 
# N∗(N+1)(2N+1)6
# where N=1,000,000.
# 
# In the function brute_force we loop over each number from 1 to 1 million and add it to the result.

# In[ ]:




