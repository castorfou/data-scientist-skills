#!/usr/bin/env python
# coding: utf-8

# # Dates and times
# 
# ```python
# # Import datetime
# from datetime import datetime
# dt = datetime(2017, 10, 1, 15, 23, 25)
# 
# # Import datetime
# from datetime import datetime
# dt = datetime(year=2017, month=10, day=1,
# hour=15, minute=23, second=25,
# microsecond=500000)
# 
# # Replacing parts of a datetime
# print(dt)
# 2017-10-01 15:23:25.500000
# dt_hr = dt.replace(minute=0, second=0, microsecond=0)
# print(dt_hr)
# 2017-10-01 15:00:00
# ```

# ## Creating datetimes by hand
# > 
# > Often you create `datetime` objects based on outside data. Sometimes though, you want to create a `datetime` object from scratch.
# > 
# > You're going to create a few different `datetime` objects from scratch to get the hang of that process. These come from the bikeshare data set that you'll use throughout the rest of the chapter.

# In[1]:


# Import datetime
from datetime import datetime

# Create a datetime object
dt = datetime(2017, 10, 1, 15, 26, 26)

# Print the results in ISO 8601 format
print(dt.isoformat())


# In[2]:


# Import datetime
from datetime import datetime


# Create a datetime object
dt = datetime(2017, 12, 31, 15, 19, 13)

# Print the results in ISO 8601 format
print(dt.isoformat())


# In[3]:


# Replace the year with 1917
dt_old = dt.replace(year=1917)

# Print the results in ISO 8601 format
print(dt_old)


# ## Counting events before and after noon
# > 
# > In this chapter, you will be working with a list of all bike trips for one Capital Bikeshare bike, W20529, from October 1, 2017 to December 31, 2017. This list has been loaded as `onebike_datetimes`.
# > 
# > Each element of the list is a dictionary with two entries: `start` is a `datetime` object corresponding to the start of a trip (when a bike is removed from the dock) and `end` is a `datetime` object corresponding to the end of a trip (when a bike is put back into a dock).
# > 
# > You can use this data set to understand better how this bike was used. Did more trips start before noon or after noon?

# In[4]:


import datetime
onebike_datetimes=[{'end': datetime.datetime(2017, 10, 1, 15, 26, 26),
  'start': datetime.datetime(2017, 10, 1, 15, 23, 25)},
 {'end': datetime.datetime(2017, 10, 1, 17, 49, 59),
  'start': datetime.datetime(2017, 10, 1, 15, 42, 57)},
 {'end': datetime.datetime(2017, 10, 2, 6, 42, 53),
  'start': datetime.datetime(2017, 10, 2, 6, 37, 10)},
 {'end': datetime.datetime(2017, 10, 2, 9, 18, 3),
  'start': datetime.datetime(2017, 10, 2, 8, 56, 45)},
 {'end': datetime.datetime(2017, 10, 2, 18, 45, 5),
  'start': datetime.datetime(2017, 10, 2, 18, 23, 48)},
 {'end': datetime.datetime(2017, 10, 2, 19, 10, 54),
  'start': datetime.datetime(2017, 10, 2, 18, 48, 8)},
 {'end': datetime.datetime(2017, 10, 2, 19, 31, 45),
  'start': datetime.datetime(2017, 10, 2, 19, 18, 10)},
 {'end': datetime.datetime(2017, 10, 2, 19, 46, 37),
  'start': datetime.datetime(2017, 10, 2, 19, 37, 32)},
 {'end': datetime.datetime(2017, 10, 3, 8, 32, 27),
  'start': datetime.datetime(2017, 10, 3, 8, 24, 16)},
 {'end': datetime.datetime(2017, 10, 3, 18, 27, 46),
  'start': datetime.datetime(2017, 10, 3, 18, 17, 7)},
 {'end': datetime.datetime(2017, 10, 3, 19, 52, 8),
  'start': datetime.datetime(2017, 10, 3, 19, 24, 10)},
 {'end': datetime.datetime(2017, 10, 3, 20, 23, 52),
  'start': datetime.datetime(2017, 10, 3, 20, 17, 6)},
 {'end': datetime.datetime(2017, 10, 3, 20, 57, 10),
  'start': datetime.datetime(2017, 10, 3, 20, 45, 21)},
 {'end': datetime.datetime(2017, 10, 4, 7, 13, 31),
  'start': datetime.datetime(2017, 10, 4, 7, 4, 57)},
 {'end': datetime.datetime(2017, 10, 4, 7, 21, 54),
  'start': datetime.datetime(2017, 10, 4, 7, 13, 42)},
 {'end': datetime.datetime(2017, 10, 4, 14, 50),
  'start': datetime.datetime(2017, 10, 4, 14, 22, 12)},
 {'end': datetime.datetime(2017, 10, 4, 15, 44, 49),
  'start': datetime.datetime(2017, 10, 4, 15, 7, 27)},
 {'end': datetime.datetime(2017, 10, 4, 16, 32, 33),
  'start': datetime.datetime(2017, 10, 4, 15, 46, 41)},
 {'end': datetime.datetime(2017, 10, 4, 16, 46, 59),
  'start': datetime.datetime(2017, 10, 4, 16, 34, 44)},
 {'end': datetime.datetime(2017, 10, 4, 17, 31, 36),
  'start': datetime.datetime(2017, 10, 4, 17, 26, 6)},
 {'end': datetime.datetime(2017, 10, 4, 17, 50, 41),
  'start': datetime.datetime(2017, 10, 4, 17, 42, 3)},
 {'end': datetime.datetime(2017, 10, 5, 8, 12, 55),
  'start': datetime.datetime(2017, 10, 5, 7, 49, 2)},
 {'end': datetime.datetime(2017, 10, 5, 8, 29, 45),
  'start': datetime.datetime(2017, 10, 5, 8, 26, 21)},
 {'end': datetime.datetime(2017, 10, 5, 8, 38, 31),
  'start': datetime.datetime(2017, 10, 5, 8, 33, 27)},
 {'end': datetime.datetime(2017, 10, 5, 16, 51, 52),
  'start': datetime.datetime(2017, 10, 5, 16, 35, 35)},
 {'end': datetime.datetime(2017, 10, 5, 18, 16, 50),
  'start': datetime.datetime(2017, 10, 5, 17, 53, 31)},
 {'end': datetime.datetime(2017, 10, 6, 8, 38, 1),
  'start': datetime.datetime(2017, 10, 6, 8, 17, 17)},
 {'end': datetime.datetime(2017, 10, 6, 11, 50, 38),
  'start': datetime.datetime(2017, 10, 6, 11, 39, 40)},
 {'end': datetime.datetime(2017, 10, 6, 13, 13, 14),
  'start': datetime.datetime(2017, 10, 6, 12, 59, 54)},
 {'end': datetime.datetime(2017, 10, 6, 14, 14, 56),
  'start': datetime.datetime(2017, 10, 6, 13, 43, 5)},
 {'end': datetime.datetime(2017, 10, 6, 15, 9, 26),
  'start': datetime.datetime(2017, 10, 6, 14, 28, 15)},
 {'end': datetime.datetime(2017, 10, 6, 16, 12, 34),
  'start': datetime.datetime(2017, 10, 6, 15, 50, 10)},
 {'end': datetime.datetime(2017, 10, 6, 16, 39, 31),
  'start': datetime.datetime(2017, 10, 6, 16, 32, 16)},
 {'end': datetime.datetime(2017, 10, 6, 16, 48, 39),
  'start': datetime.datetime(2017, 10, 6, 16, 44, 8)},
 {'end': datetime.datetime(2017, 10, 6, 17, 9, 3),
  'start': datetime.datetime(2017, 10, 6, 16, 53, 43)},
 {'end': datetime.datetime(2017, 10, 7, 11, 53, 6),
  'start': datetime.datetime(2017, 10, 7, 11, 38, 55)},
 {'end': datetime.datetime(2017, 10, 7, 14, 7, 5),
  'start': datetime.datetime(2017, 10, 7, 14, 3, 36)},
 {'end': datetime.datetime(2017, 10, 7, 14, 27, 36),
  'start': datetime.datetime(2017, 10, 7, 14, 20, 3)},
 {'end': datetime.datetime(2017, 10, 7, 14, 44, 51),
  'start': datetime.datetime(2017, 10, 7, 14, 30, 50)},
 {'end': datetime.datetime(2017, 10, 8, 0, 30, 48),
  'start': datetime.datetime(2017, 10, 8, 0, 28, 26)},
 {'end': datetime.datetime(2017, 10, 8, 11, 33, 24),
  'start': datetime.datetime(2017, 10, 8, 11, 16, 21)},
 {'end': datetime.datetime(2017, 10, 8, 13, 1, 29),
  'start': datetime.datetime(2017, 10, 8, 12, 37, 3)},
 {'end': datetime.datetime(2017, 10, 8, 13, 57, 53),
  'start': datetime.datetime(2017, 10, 8, 13, 30, 37)},
 {'end': datetime.datetime(2017, 10, 8, 15, 7, 19),
  'start': datetime.datetime(2017, 10, 8, 14, 16, 40)},
 {'end': datetime.datetime(2017, 10, 8, 15, 50, 1),
  'start': datetime.datetime(2017, 10, 8, 15, 23, 50)},
 {'end': datetime.datetime(2017, 10, 8, 16, 17, 42),
  'start': datetime.datetime(2017, 10, 8, 15, 54, 12)},
 {'end': datetime.datetime(2017, 10, 8, 16, 35, 18),
  'start': datetime.datetime(2017, 10, 8, 16, 28, 52)},
 {'end': datetime.datetime(2017, 10, 8, 23, 33, 41),
  'start': datetime.datetime(2017, 10, 8, 23, 8, 14)},
 {'end': datetime.datetime(2017, 10, 8, 23, 45, 11),
  'start': datetime.datetime(2017, 10, 8, 23, 34, 49)},
 {'end': datetime.datetime(2017, 10, 9, 0, 10, 57),
  'start': datetime.datetime(2017, 10, 8, 23, 46, 47)},
 {'end': datetime.datetime(2017, 10, 9, 0, 36, 40),
  'start': datetime.datetime(2017, 10, 9, 0, 12, 58)},
 {'end': datetime.datetime(2017, 10, 9, 0, 53, 33),
  'start': datetime.datetime(2017, 10, 9, 0, 37, 2)},
 {'end': datetime.datetime(2017, 10, 9, 1, 48, 13),
  'start': datetime.datetime(2017, 10, 9, 1, 23, 29)},
 {'end': datetime.datetime(2017, 10, 9, 2, 13, 35),
  'start': datetime.datetime(2017, 10, 9, 1, 49, 25)},
 {'end': datetime.datetime(2017, 10, 9, 2, 29, 40),
  'start': datetime.datetime(2017, 10, 9, 2, 14, 11)},
 {'end': datetime.datetime(2017, 10, 9, 13, 13, 25),
  'start': datetime.datetime(2017, 10, 9, 13, 4, 32)},
 {'end': datetime.datetime(2017, 10, 9, 14, 38, 55),
  'start': datetime.datetime(2017, 10, 9, 14, 30, 10)},
 {'end': datetime.datetime(2017, 10, 9, 15, 11, 30),
  'start': datetime.datetime(2017, 10, 9, 15, 6, 47)},
 {'end': datetime.datetime(2017, 10, 9, 16, 45, 38),
  'start': datetime.datetime(2017, 10, 9, 16, 43, 25)},
 {'end': datetime.datetime(2017, 10, 10, 15, 51, 24),
  'start': datetime.datetime(2017, 10, 10, 15, 32, 58)},
 {'end': datetime.datetime(2017, 10, 10, 17, 3, 47),
  'start': datetime.datetime(2017, 10, 10, 16, 47, 55)},
 {'end': datetime.datetime(2017, 10, 10, 18, 0, 18),
  'start': datetime.datetime(2017, 10, 10, 17, 51, 5)},
 {'end': datetime.datetime(2017, 10, 10, 18, 19, 11),
  'start': datetime.datetime(2017, 10, 10, 18, 8, 12)},
 {'end': datetime.datetime(2017, 10, 10, 19, 14, 32),
  'start': datetime.datetime(2017, 10, 10, 19, 9, 35)},
 {'end': datetime.datetime(2017, 10, 10, 19, 23, 8),
  'start': datetime.datetime(2017, 10, 10, 19, 17, 11)},
 {'end': datetime.datetime(2017, 10, 10, 19, 44, 40),
  'start': datetime.datetime(2017, 10, 10, 19, 28, 11)},
 {'end': datetime.datetime(2017, 10, 10, 20, 11, 54),
  'start': datetime.datetime(2017, 10, 10, 19, 55, 35)},
 {'end': datetime.datetime(2017, 10, 10, 22, 33, 23),
  'start': datetime.datetime(2017, 10, 10, 22, 20, 43)},
 {'end': datetime.datetime(2017, 10, 11, 4, 59, 22),
  'start': datetime.datetime(2017, 10, 11, 4, 40, 52)},
 {'end': datetime.datetime(2017, 10, 11, 6, 40, 13),
  'start': datetime.datetime(2017, 10, 11, 6, 28, 58)},
 {'end': datetime.datetime(2017, 10, 11, 17, 1, 14),
  'start': datetime.datetime(2017, 10, 11, 16, 41, 7)},
 {'end': datetime.datetime(2017, 10, 12, 8, 35, 3),
  'start': datetime.datetime(2017, 10, 12, 8, 8, 30)},
 {'end': datetime.datetime(2017, 10, 12, 8, 59, 50),
  'start': datetime.datetime(2017, 10, 12, 8, 47, 2)},
 {'end': datetime.datetime(2017, 10, 12, 13, 37, 45),
  'start': datetime.datetime(2017, 10, 12, 13, 13, 39)},
 {'end': datetime.datetime(2017, 10, 12, 13, 48, 17),
  'start': datetime.datetime(2017, 10, 12, 13, 40, 12)},
 {'end': datetime.datetime(2017, 10, 12, 13, 53, 16),
  'start': datetime.datetime(2017, 10, 12, 13, 49, 56)},
 {'end': datetime.datetime(2017, 10, 12, 14, 39, 57),
  'start': datetime.datetime(2017, 10, 12, 14, 33, 18)},
 {'end': datetime.datetime(2017, 10, 13, 15, 59, 41),
  'start': datetime.datetime(2017, 10, 13, 15, 55, 39)},
 {'end': datetime.datetime(2017, 10, 17, 18, 1, 38),
  'start': datetime.datetime(2017, 10, 17, 17, 58, 48)},
 {'end': datetime.datetime(2017, 10, 19, 20, 29, 15),
  'start': datetime.datetime(2017, 10, 19, 20, 21, 45)},
 {'end': datetime.datetime(2017, 10, 19, 21, 29, 37),
  'start': datetime.datetime(2017, 10, 19, 21, 11, 39)},
 {'end': datetime.datetime(2017, 10, 19, 21, 47, 23),
  'start': datetime.datetime(2017, 10, 19, 21, 30, 1)},
 {'end': datetime.datetime(2017, 10, 19, 21, 57, 7),
  'start': datetime.datetime(2017, 10, 19, 21, 47, 34)},
 {'end': datetime.datetime(2017, 10, 19, 22, 9, 52),
  'start': datetime.datetime(2017, 10, 19, 21, 57, 24)},
 {'end': datetime.datetime(2017, 10, 21, 12, 36, 24),
  'start': datetime.datetime(2017, 10, 21, 12, 24, 9)},
 {'end': datetime.datetime(2017, 10, 21, 12, 42, 13),
  'start': datetime.datetime(2017, 10, 21, 12, 36, 37)},
 {'end': datetime.datetime(2017, 10, 22, 11, 9, 36),
  'start': datetime.datetime(2017, 10, 21, 13, 47, 43)},
 {'end': datetime.datetime(2017, 10, 22, 13, 31, 44),
  'start': datetime.datetime(2017, 10, 22, 13, 28, 53)},
 {'end': datetime.datetime(2017, 10, 22, 13, 56, 33),
  'start': datetime.datetime(2017, 10, 22, 13, 47, 5)},
 {'end': datetime.datetime(2017, 10, 22, 14, 32, 39),
  'start': datetime.datetime(2017, 10, 22, 14, 26, 41)},
 {'end': datetime.datetime(2017, 10, 22, 15, 9, 58),
  'start': datetime.datetime(2017, 10, 22, 14, 54, 41)},
 {'end': datetime.datetime(2017, 10, 22, 16, 51, 40),
  'start': datetime.datetime(2017, 10, 22, 16, 40, 29)},
 {'end': datetime.datetime(2017, 10, 22, 18, 28, 37),
  'start': datetime.datetime(2017, 10, 22, 17, 58, 46)},
 {'end': datetime.datetime(2017, 10, 22, 18, 50, 34),
  'start': datetime.datetime(2017, 10, 22, 18, 45, 16)},
 {'end': datetime.datetime(2017, 10, 22, 19, 11, 10),
  'start': datetime.datetime(2017, 10, 22, 18, 56, 22)},
 {'end': datetime.datetime(2017, 10, 23, 10, 35, 32),
  'start': datetime.datetime(2017, 10, 23, 10, 14, 8)},
 {'end': datetime.datetime(2017, 10, 23, 14, 38, 34),
  'start': datetime.datetime(2017, 10, 23, 11, 29, 36)},
 {'end': datetime.datetime(2017, 10, 23, 15, 32, 58),
  'start': datetime.datetime(2017, 10, 23, 15, 4, 52)},
 {'end': datetime.datetime(2017, 10, 23, 17, 6, 47),
  'start': datetime.datetime(2017, 10, 23, 15, 33, 48)},
 {'end': datetime.datetime(2017, 10, 23, 19, 31, 26),
  'start': datetime.datetime(2017, 10, 23, 17, 13, 16)},
 {'end': datetime.datetime(2017, 10, 23, 20, 25, 53),
  'start': datetime.datetime(2017, 10, 23, 19, 55, 3)},
 {'end': datetime.datetime(2017, 10, 23, 22, 18, 4),
  'start': datetime.datetime(2017, 10, 23, 21, 47, 54)},
 {'end': datetime.datetime(2017, 10, 23, 22, 48, 42),
  'start': datetime.datetime(2017, 10, 23, 22, 34, 12)},
 {'end': datetime.datetime(2017, 10, 24, 7, 2, 17),
  'start': datetime.datetime(2017, 10, 24, 6, 55, 1)},
 {'end': datetime.datetime(2017, 10, 24, 15, 3, 16),
  'start': datetime.datetime(2017, 10, 24, 14, 56, 7)},
 {'end': datetime.datetime(2017, 10, 24, 15, 59, 50),
  'start': datetime.datetime(2017, 10, 24, 15, 51, 36)},
 {'end': datetime.datetime(2017, 10, 24, 16, 55, 9),
  'start': datetime.datetime(2017, 10, 24, 16, 31, 10)},
 {'end': datetime.datetime(2017, 10, 28, 14, 32, 34),
  'start': datetime.datetime(2017, 10, 28, 14, 26, 14)},
 {'end': datetime.datetime(2017, 11, 1, 9, 52, 23),
  'start': datetime.datetime(2017, 11, 1, 9, 41, 54)},
 {'end': datetime.datetime(2017, 11, 1, 20, 32, 13),
  'start': datetime.datetime(2017, 11, 1, 20, 16, 11)},
 {'end': datetime.datetime(2017, 11, 2, 19, 50, 56),
  'start': datetime.datetime(2017, 11, 2, 19, 44, 29)},
 {'end': datetime.datetime(2017, 11, 2, 20, 30, 29),
  'start': datetime.datetime(2017, 11, 2, 20, 14, 37)},
 {'end': datetime.datetime(2017, 11, 2, 21, 38, 57),
  'start': datetime.datetime(2017, 11, 2, 21, 35, 47)},
 {'end': datetime.datetime(2017, 11, 3, 10, 11, 46),
  'start': datetime.datetime(2017, 11, 3, 9, 59, 27)},
 {'end': datetime.datetime(2017, 11, 3, 10, 32, 2),
  'start': datetime.datetime(2017, 11, 3, 10, 13, 22)},
 {'end': datetime.datetime(2017, 11, 3, 10, 50, 34),
  'start': datetime.datetime(2017, 11, 3, 10, 44, 25)},
 {'end': datetime.datetime(2017, 11, 3, 16, 44, 38),
  'start': datetime.datetime(2017, 11, 3, 16, 6, 43)},
 {'end': datetime.datetime(2017, 11, 3, 17, 0, 27),
  'start': datetime.datetime(2017, 11, 3, 16, 45, 54)},
 {'end': datetime.datetime(2017, 11, 3, 17, 35, 5),
  'start': datetime.datetime(2017, 11, 3, 17, 7, 15)},
 {'end': datetime.datetime(2017, 11, 3, 17, 46, 48),
  'start': datetime.datetime(2017, 11, 3, 17, 36, 5)},
 {'end': datetime.datetime(2017, 11, 3, 18, 0, 3),
  'start': datetime.datetime(2017, 11, 3, 17, 50, 31)},
 {'end': datetime.datetime(2017, 11, 3, 19, 45, 51),
  'start': datetime.datetime(2017, 11, 3, 19, 22, 56)},
 {'end': datetime.datetime(2017, 11, 4, 13, 26, 15),
  'start': datetime.datetime(2017, 11, 4, 13, 14, 10)},
 {'end': datetime.datetime(2017, 11, 4, 14, 30, 5),
  'start': datetime.datetime(2017, 11, 4, 14, 18, 37)},
 {'end': datetime.datetime(2017, 11, 4, 15, 3, 20),
  'start': datetime.datetime(2017, 11, 4, 14, 45, 59)},
 {'end': datetime.datetime(2017, 11, 4, 15, 44, 30),
  'start': datetime.datetime(2017, 11, 4, 15, 16, 3)},
 {'end': datetime.datetime(2017, 11, 4, 16, 58, 22),
  'start': datetime.datetime(2017, 11, 4, 16, 37, 46)},
 {'end': datetime.datetime(2017, 11, 4, 17, 34, 50),
  'start': datetime.datetime(2017, 11, 4, 17, 13, 19)},
 {'end': datetime.datetime(2017, 11, 4, 18, 58, 44),
  'start': datetime.datetime(2017, 11, 4, 18, 10, 34)},
 {'end': datetime.datetime(2017, 11, 5, 1, 1, 4),
  'start': datetime.datetime(2017, 11, 5, 1, 56, 50)},
 {'end': datetime.datetime(2017, 11, 5, 8, 53, 46),
  'start': datetime.datetime(2017, 11, 5, 8, 33, 33)},
 {'end': datetime.datetime(2017, 11, 5, 9, 3, 39),
  'start': datetime.datetime(2017, 11, 5, 8, 58, 8)},
 {'end': datetime.datetime(2017, 11, 5, 11, 30, 5),
  'start': datetime.datetime(2017, 11, 5, 11, 5, 8)},
 {'end': datetime.datetime(2017, 11, 6, 8, 59, 5),
  'start': datetime.datetime(2017, 11, 6, 8, 50, 18)},
 {'end': datetime.datetime(2017, 11, 6, 9, 13, 47),
  'start': datetime.datetime(2017, 11, 6, 9, 4, 3)},
 {'end': datetime.datetime(2017, 11, 6, 17, 2, 55),
  'start': datetime.datetime(2017, 11, 6, 16, 19, 36)},
 {'end': datetime.datetime(2017, 11, 6, 17, 34, 6),
  'start': datetime.datetime(2017, 11, 6, 17, 21, 27)},
 {'end': datetime.datetime(2017, 11, 6, 17, 57, 32),
  'start': datetime.datetime(2017, 11, 6, 17, 36, 1)},
 {'end': datetime.datetime(2017, 11, 6, 18, 15, 8),
  'start': datetime.datetime(2017, 11, 6, 17, 59, 52)},
 {'end': datetime.datetime(2017, 11, 6, 18, 21, 17),
  'start': datetime.datetime(2017, 11, 6, 18, 18, 36)},
 {'end': datetime.datetime(2017, 11, 6, 19, 37, 57),
  'start': datetime.datetime(2017, 11, 6, 19, 24, 31)},
 {'end': datetime.datetime(2017, 11, 6, 20, 3, 14),
  'start': datetime.datetime(2017, 11, 6, 19, 49, 16)},
 {'end': datetime.datetime(2017, 11, 7, 8, 1, 32),
  'start': datetime.datetime(2017, 11, 7, 7, 50, 48)},
 {'end': datetime.datetime(2017, 11, 8, 13, 18, 5),
  'start': datetime.datetime(2017, 11, 8, 13, 11, 51)},
 {'end': datetime.datetime(2017, 11, 8, 21, 46, 5),
  'start': datetime.datetime(2017, 11, 8, 21, 34, 47)},
 {'end': datetime.datetime(2017, 11, 8, 22, 4, 47),
  'start': datetime.datetime(2017, 11, 8, 22, 2, 30)},
 {'end': datetime.datetime(2017, 11, 9, 7, 12, 10),
  'start': datetime.datetime(2017, 11, 9, 7, 1, 11)},
 {'end': datetime.datetime(2017, 11, 9, 8, 8, 28),
  'start': datetime.datetime(2017, 11, 9, 8, 2, 2)},
 {'end': datetime.datetime(2017, 11, 9, 8, 32, 24),
  'start': datetime.datetime(2017, 11, 9, 8, 19, 59)},
 {'end': datetime.datetime(2017, 11, 9, 8, 48, 59),
  'start': datetime.datetime(2017, 11, 9, 8, 41, 31)},
 {'end': datetime.datetime(2017, 11, 9, 9, 9, 24),
  'start': datetime.datetime(2017, 11, 9, 9, 0, 6)},
 {'end': datetime.datetime(2017, 11, 9, 9, 24, 25),
  'start': datetime.datetime(2017, 11, 9, 9, 9, 37)},
 {'end': datetime.datetime(2017, 11, 9, 13, 25, 39),
  'start': datetime.datetime(2017, 11, 9, 13, 14, 37)},
 {'end': datetime.datetime(2017, 11, 9, 15, 31, 10),
  'start': datetime.datetime(2017, 11, 9, 15, 20, 7)},
 {'end': datetime.datetime(2017, 11, 9, 18, 53, 10),
  'start': datetime.datetime(2017, 11, 9, 18, 47, 8)},
 {'end': datetime.datetime(2017, 11, 9, 23, 43, 35),
  'start': datetime.datetime(2017, 11, 9, 23, 35, 2)},
 {'end': datetime.datetime(2017, 11, 10, 8, 2, 28),
  'start': datetime.datetime(2017, 11, 10, 7, 51, 33)},
 {'end': datetime.datetime(2017, 11, 10, 8, 42, 9),
  'start': datetime.datetime(2017, 11, 10, 8, 38, 28)},
 {'end': datetime.datetime(2017, 11, 11, 18, 13, 14),
  'start': datetime.datetime(2017, 11, 11, 18, 5, 25)},
 {'end': datetime.datetime(2017, 11, 11, 19, 46, 22),
  'start': datetime.datetime(2017, 11, 11, 19, 39, 12)},
 {'end': datetime.datetime(2017, 11, 11, 21, 16, 31),
  'start': datetime.datetime(2017, 11, 11, 21, 13, 19)},
 {'end': datetime.datetime(2017, 11, 12, 9, 51, 43),
  'start': datetime.datetime(2017, 11, 12, 9, 46, 19)},
 {'end': datetime.datetime(2017, 11, 13, 13, 54, 15),
  'start': datetime.datetime(2017, 11, 13, 13, 33, 42)},
 {'end': datetime.datetime(2017, 11, 14, 8, 55, 52),
  'start': datetime.datetime(2017, 11, 14, 8, 40, 29)},
 {'end': datetime.datetime(2017, 11, 15, 6, 30, 6),
  'start': datetime.datetime(2017, 11, 15, 6, 14, 5)},
 {'end': datetime.datetime(2017, 11, 15, 8, 23, 44),
  'start': datetime.datetime(2017, 11, 15, 8, 14, 59)},
 {'end': datetime.datetime(2017, 11, 15, 10, 33, 41),
  'start': datetime.datetime(2017, 11, 15, 10, 16, 44)},
 {'end': datetime.datetime(2017, 11, 15, 10, 54, 14),
  'start': datetime.datetime(2017, 11, 15, 10, 33, 58)},
 {'end': datetime.datetime(2017, 11, 15, 11, 14, 42),
  'start': datetime.datetime(2017, 11, 15, 11, 2, 15)},
 {'end': datetime.datetime(2017, 11, 16, 9, 38, 49),
  'start': datetime.datetime(2017, 11, 16, 9, 27, 41)},
 {'end': datetime.datetime(2017, 11, 16, 10, 18),
  'start': datetime.datetime(2017, 11, 16, 9, 57, 41)},
 {'end': datetime.datetime(2017, 11, 16, 17, 44, 47),
  'start': datetime.datetime(2017, 11, 16, 17, 25, 5)},
 {'end': datetime.datetime(2017, 11, 17, 16, 36, 56),
  'start': datetime.datetime(2017, 11, 17, 13, 45, 54)},
 {'end': datetime.datetime(2017, 11, 17, 19, 31, 15),
  'start': datetime.datetime(2017, 11, 17, 19, 12, 49)},
 {'end': datetime.datetime(2017, 11, 18, 10, 55, 45),
  'start': datetime.datetime(2017, 11, 18, 10, 49, 6)},
 {'end': datetime.datetime(2017, 11, 18, 11, 44, 16),
  'start': datetime.datetime(2017, 11, 18, 11, 32, 12)},
 {'end': datetime.datetime(2017, 11, 18, 18, 14, 31),
  'start': datetime.datetime(2017, 11, 18, 18, 9, 1)},
 {'end': datetime.datetime(2017, 11, 18, 19, 1, 29),
  'start': datetime.datetime(2017, 11, 18, 18, 53, 10)},
 {'end': datetime.datetime(2017, 11, 19, 14, 31, 49),
  'start': datetime.datetime(2017, 11, 19, 14, 15, 41)},
 {'end': datetime.datetime(2017, 11, 20, 21, 41, 9),
  'start': datetime.datetime(2017, 11, 20, 21, 19, 19)},
 {'end': datetime.datetime(2017, 11, 20, 23, 23, 37),
  'start': datetime.datetime(2017, 11, 20, 22, 39, 48)},
 {'end': datetime.datetime(2017, 11, 21, 17, 51, 32),
  'start': datetime.datetime(2017, 11, 21, 17, 44, 25)},
 {'end': datetime.datetime(2017, 11, 21, 18, 34, 51),
  'start': datetime.datetime(2017, 11, 21, 18, 20, 52)},
 {'end': datetime.datetime(2017, 11, 21, 18, 51, 50),
  'start': datetime.datetime(2017, 11, 21, 18, 47, 32)},
 {'end': datetime.datetime(2017, 11, 21, 19, 14, 33),
  'start': datetime.datetime(2017, 11, 21, 19, 7, 57)},
 {'end': datetime.datetime(2017, 11, 21, 20, 8, 54),
  'start': datetime.datetime(2017, 11, 21, 20, 4, 56)},
 {'end': datetime.datetime(2017, 11, 21, 22, 8, 12),
  'start': datetime.datetime(2017, 11, 21, 21, 55, 47)},
 {'end': datetime.datetime(2017, 11, 23, 23, 57, 56),
  'start': datetime.datetime(2017, 11, 23, 23, 47, 43)},
 {'end': datetime.datetime(2017, 11, 24, 6, 53, 15),
  'start': datetime.datetime(2017, 11, 24, 6, 41, 25)},
 {'end': datetime.datetime(2017, 11, 24, 7, 33, 24),
  'start': datetime.datetime(2017, 11, 24, 6, 58, 56)},
 {'end': datetime.datetime(2017, 11, 26, 12, 41, 36),
  'start': datetime.datetime(2017, 11, 26, 12, 25, 49)},
 {'end': datetime.datetime(2017, 11, 27, 5, 54, 13),
  'start': datetime.datetime(2017, 11, 27, 5, 29, 4)},
 {'end': datetime.datetime(2017, 11, 27, 6, 11, 1),
  'start': datetime.datetime(2017, 11, 27, 6, 6, 47)},
 {'end': datetime.datetime(2017, 11, 27, 6, 55, 39),
  'start': datetime.datetime(2017, 11, 27, 6, 45, 14)},
 {'end': datetime.datetime(2017, 11, 27, 9, 47, 43),
  'start': datetime.datetime(2017, 11, 27, 9, 39, 44)},
 {'end': datetime.datetime(2017, 11, 27, 11, 20, 46),
  'start': datetime.datetime(2017, 11, 27, 11, 9, 18)},
 {'end': datetime.datetime(2017, 11, 27, 11, 35, 44),
  'start': datetime.datetime(2017, 11, 27, 11, 31, 46)},
 {'end': datetime.datetime(2017, 11, 27, 12, 12, 36),
  'start': datetime.datetime(2017, 11, 27, 12, 7, 14)},
 {'end': datetime.datetime(2017, 11, 27, 12, 26, 44),
  'start': datetime.datetime(2017, 11, 27, 12, 21, 40)},
 {'end': datetime.datetime(2017, 11, 27, 17, 36, 7),
  'start': datetime.datetime(2017, 11, 27, 17, 26, 31)},
 {'end': datetime.datetime(2017, 11, 27, 18, 29, 4),
  'start': datetime.datetime(2017, 11, 27, 18, 11, 49)},
 {'end': datetime.datetime(2017, 11, 27, 19, 47, 17),
  'start': datetime.datetime(2017, 11, 27, 19, 36, 16)},
 {'end': datetime.datetime(2017, 11, 27, 20, 17, 33),
  'start': datetime.datetime(2017, 11, 27, 20, 12, 57)},
 {'end': datetime.datetime(2017, 11, 28, 8, 41, 53),
  'start': datetime.datetime(2017, 11, 28, 8, 18, 6)},
 {'end': datetime.datetime(2017, 11, 28, 19, 34, 1),
  'start': datetime.datetime(2017, 11, 28, 19, 17, 23)},
 {'end': datetime.datetime(2017, 11, 28, 19, 46, 24),
  'start': datetime.datetime(2017, 11, 28, 19, 34, 15)},
 {'end': datetime.datetime(2017, 11, 28, 21, 39, 32),
  'start': datetime.datetime(2017, 11, 28, 21, 27, 29)},
 {'end': datetime.datetime(2017, 11, 29, 7, 51, 18),
  'start': datetime.datetime(2017, 11, 29, 7, 47, 38)},
 {'end': datetime.datetime(2017, 11, 29, 9, 53, 44),
  'start': datetime.datetime(2017, 11, 29, 9, 50, 12)},
 {'end': datetime.datetime(2017, 11, 29, 17, 16, 21),
  'start': datetime.datetime(2017, 11, 29, 17, 3, 42)},
 {'end': datetime.datetime(2017, 11, 29, 18, 23, 43),
  'start': datetime.datetime(2017, 11, 29, 18, 19, 15)},
 {'end': datetime.datetime(2017, 12, 1, 17, 10, 12),
  'start': datetime.datetime(2017, 12, 1, 17, 3, 58)},
 {'end': datetime.datetime(2017, 12, 2, 8, 1, 1),
  'start': datetime.datetime(2017, 12, 2, 7, 55, 56)},
 {'end': datetime.datetime(2017, 12, 2, 9, 21, 18),
  'start': datetime.datetime(2017, 12, 2, 9, 16, 14)},
 {'end': datetime.datetime(2017, 12, 2, 19, 53, 18),
  'start': datetime.datetime(2017, 12, 2, 19, 48, 29)},
 {'end': datetime.datetime(2017, 12, 3, 15, 20, 9),
  'start': datetime.datetime(2017, 12, 3, 14, 36, 29)},
 {'end': datetime.datetime(2017, 12, 3, 16, 25, 30),
  'start': datetime.datetime(2017, 12, 3, 16, 4, 2)},
 {'end': datetime.datetime(2017, 12, 3, 16, 43, 58),
  'start': datetime.datetime(2017, 12, 3, 16, 40, 26)},
 {'end': datetime.datetime(2017, 12, 3, 18, 4, 33),
  'start': datetime.datetime(2017, 12, 3, 17, 20, 17)},
 {'end': datetime.datetime(2017, 12, 4, 8, 51),
  'start': datetime.datetime(2017, 12, 4, 8, 34, 24)},
 {'end': datetime.datetime(2017, 12, 4, 17, 53, 57),
  'start': datetime.datetime(2017, 12, 4, 17, 49, 26)},
 {'end': datetime.datetime(2017, 12, 4, 18, 50, 33),
  'start': datetime.datetime(2017, 12, 4, 18, 38, 52)},
 {'end': datetime.datetime(2017, 12, 4, 21, 46, 58),
  'start': datetime.datetime(2017, 12, 4, 21, 39, 20)},
 {'end': datetime.datetime(2017, 12, 4, 21, 56, 17),
  'start': datetime.datetime(2017, 12, 4, 21, 54, 21)},
 {'end': datetime.datetime(2017, 12, 5, 8, 52, 54),
  'start': datetime.datetime(2017, 12, 5, 8, 50, 50)},
 {'end': datetime.datetime(2017, 12, 6, 8, 24, 14),
  'start': datetime.datetime(2017, 12, 6, 8, 19, 38)},
 {'end': datetime.datetime(2017, 12, 6, 18, 28, 11),
  'start': datetime.datetime(2017, 12, 6, 18, 19, 19)},
 {'end': datetime.datetime(2017, 12, 6, 18, 33, 12),
  'start': datetime.datetime(2017, 12, 6, 18, 28, 55)},
 {'end': datetime.datetime(2017, 12, 6, 20, 21, 38),
  'start': datetime.datetime(2017, 12, 6, 20, 3, 29)},
 {'end': datetime.datetime(2017, 12, 6, 20, 39, 57),
  'start': datetime.datetime(2017, 12, 6, 20, 36, 42)},
 {'end': datetime.datetime(2017, 12, 7, 6, 1, 15),
  'start': datetime.datetime(2017, 12, 7, 5, 54, 51)},
 {'end': datetime.datetime(2017, 12, 8, 16, 55, 49),
  'start': datetime.datetime(2017, 12, 8, 16, 47, 18)},
 {'end': datetime.datetime(2017, 12, 8, 19, 29, 12),
  'start': datetime.datetime(2017, 12, 8, 19, 15, 2)},
 {'end': datetime.datetime(2017, 12, 9, 22, 47, 19),
  'start': datetime.datetime(2017, 12, 9, 22, 39, 37)},
 {'end': datetime.datetime(2017, 12, 9, 23, 5, 32),
  'start': datetime.datetime(2017, 12, 9, 23, 0, 10)},
 {'end': datetime.datetime(2017, 12, 10, 0, 56, 2),
  'start': datetime.datetime(2017, 12, 10, 0, 39, 24)},
 {'end': datetime.datetime(2017, 12, 10, 1, 8, 9),
  'start': datetime.datetime(2017, 12, 10, 1, 2, 42)},
 {'end': datetime.datetime(2017, 12, 10, 1, 11, 30),
  'start': datetime.datetime(2017, 12, 10, 1, 8, 57)},
 {'end': datetime.datetime(2017, 12, 10, 13, 51, 41),
  'start': datetime.datetime(2017, 12, 10, 13, 49, 9)},
 {'end': datetime.datetime(2017, 12, 10, 15, 18, 19),
  'start': datetime.datetime(2017, 12, 10, 15, 14, 29)},
 {'end': datetime.datetime(2017, 12, 10, 15, 36, 28),
  'start': datetime.datetime(2017, 12, 10, 15, 31, 7)},
 {'end': datetime.datetime(2017, 12, 10, 16, 30, 31),
  'start': datetime.datetime(2017, 12, 10, 16, 20, 6)},
 {'end': datetime.datetime(2017, 12, 10, 17, 14, 25),
  'start': datetime.datetime(2017, 12, 10, 17, 7, 54)},
 {'end': datetime.datetime(2017, 12, 10, 17, 45, 25),
  'start': datetime.datetime(2017, 12, 10, 17, 23, 47)},
 {'end': datetime.datetime(2017, 12, 11, 6, 34, 4),
  'start': datetime.datetime(2017, 12, 11, 6, 17, 6)},
 {'end': datetime.datetime(2017, 12, 11, 9, 12, 21),
  'start': datetime.datetime(2017, 12, 11, 9, 8, 41)},
 {'end': datetime.datetime(2017, 12, 11, 9, 20, 18),
  'start': datetime.datetime(2017, 12, 11, 9, 15, 41)},
 {'end': datetime.datetime(2017, 12, 12, 8, 59, 34),
  'start': datetime.datetime(2017, 12, 12, 8, 55, 53)},
 {'end': datetime.datetime(2017, 12, 13, 17, 18, 32),
  'start': datetime.datetime(2017, 12, 13, 17, 14, 56)},
 {'end': datetime.datetime(2017, 12, 13, 19, 0, 45),
  'start': datetime.datetime(2017, 12, 13, 18, 52, 16)},
 {'end': datetime.datetime(2017, 12, 14, 9, 11, 6),
  'start': datetime.datetime(2017, 12, 14, 9, 1, 10)},
 {'end': datetime.datetime(2017, 12, 14, 9, 19, 6),
  'start': datetime.datetime(2017, 12, 14, 9, 12, 59)},
 {'end': datetime.datetime(2017, 12, 14, 12, 2),
  'start': datetime.datetime(2017, 12, 14, 11, 54, 33)},
 {'end': datetime.datetime(2017, 12, 14, 14, 44, 40),
  'start': datetime.datetime(2017, 12, 14, 14, 40, 23)},
 {'end': datetime.datetime(2017, 12, 14, 15, 26, 24),
  'start': datetime.datetime(2017, 12, 14, 15, 8, 55)},
 {'end': datetime.datetime(2017, 12, 14, 18, 9, 4),
  'start': datetime.datetime(2017, 12, 14, 17, 46, 17)},
 {'end': datetime.datetime(2017, 12, 15, 9, 23, 45),
  'start': datetime.datetime(2017, 12, 15, 9, 8, 12)},
 {'end': datetime.datetime(2017, 12, 16, 9, 36, 17),
  'start': datetime.datetime(2017, 12, 16, 9, 33, 46)},
 {'end': datetime.datetime(2017, 12, 16, 11, 5, 4),
  'start': datetime.datetime(2017, 12, 16, 11, 2, 31)},
 {'end': datetime.datetime(2017, 12, 17, 10, 32, 3),
  'start': datetime.datetime(2017, 12, 17, 10, 9, 47)},
 {'end': datetime.datetime(2017, 12, 18, 8, 7, 34),
  'start': datetime.datetime(2017, 12, 18, 8, 2, 36)},
 {'end': datetime.datetime(2017, 12, 18, 16, 9, 20),
  'start': datetime.datetime(2017, 12, 18, 16, 3)},
 {'end': datetime.datetime(2017, 12, 18, 16, 53, 12),
  'start': datetime.datetime(2017, 12, 18, 16, 30, 7)},
 {'end': datetime.datetime(2017, 12, 18, 19, 22, 8),
  'start': datetime.datetime(2017, 12, 18, 19, 18, 23)},
 {'end': datetime.datetime(2017, 12, 18, 20, 17, 47),
  'start': datetime.datetime(2017, 12, 18, 20, 14, 46)},
 {'end': datetime.datetime(2017, 12, 19, 19, 23, 49),
  'start': datetime.datetime(2017, 12, 19, 19, 14, 8)},
 {'end': datetime.datetime(2017, 12, 19, 19, 43, 46),
  'start': datetime.datetime(2017, 12, 19, 19, 39, 36)},
 {'end': datetime.datetime(2017, 12, 20, 8, 10, 46),
  'start': datetime.datetime(2017, 12, 20, 8, 5, 14)},
 {'end': datetime.datetime(2017, 12, 20, 8, 29, 50),
  'start': datetime.datetime(2017, 12, 20, 8, 15, 45)},
 {'end': datetime.datetime(2017, 12, 20, 8, 38, 9),
  'start': datetime.datetime(2017, 12, 20, 8, 33, 32)},
 {'end': datetime.datetime(2017, 12, 20, 13, 54, 39),
  'start': datetime.datetime(2017, 12, 20, 13, 43, 36)},
 {'end': datetime.datetime(2017, 12, 20, 19, 6, 54),
  'start': datetime.datetime(2017, 12, 20, 18, 57, 53)},
 {'end': datetime.datetime(2017, 12, 21, 7, 32, 3),
  'start': datetime.datetime(2017, 12, 21, 7, 21, 11)},
 {'end': datetime.datetime(2017, 12, 21, 8, 6, 15),
  'start': datetime.datetime(2017, 12, 21, 8, 1, 58)},
 {'end': datetime.datetime(2017, 12, 21, 13, 33, 49),
  'start': datetime.datetime(2017, 12, 21, 13, 20, 54)},
 {'end': datetime.datetime(2017, 12, 21, 15, 34, 27),
  'start': datetime.datetime(2017, 12, 21, 15, 26, 8)},
 {'end': datetime.datetime(2017, 12, 21, 18, 38, 50),
  'start': datetime.datetime(2017, 12, 21, 18, 9, 46)},
 {'end': datetime.datetime(2017, 12, 22, 16, 21, 46),
  'start': datetime.datetime(2017, 12, 22, 16, 14, 21)},
 {'end': datetime.datetime(2017, 12, 22, 16, 34, 14),
  'start': datetime.datetime(2017, 12, 22, 16, 29, 17)},
 {'end': datetime.datetime(2017, 12, 25, 13, 18, 27),
  'start': datetime.datetime(2017, 12, 25, 12, 49, 51)},
 {'end': datetime.datetime(2017, 12, 25, 14, 20, 50),
  'start': datetime.datetime(2017, 12, 25, 13, 46, 44)},
 {'end': datetime.datetime(2017, 12, 26, 10, 53, 45),
  'start': datetime.datetime(2017, 12, 26, 10, 40, 16)},
 {'end': datetime.datetime(2017, 12, 27, 17, 17, 39),
  'start': datetime.datetime(2017, 12, 27, 16, 56, 12)},
 {'end': datetime.datetime(2017, 12, 29, 6, 12, 30),
  'start': datetime.datetime(2017, 12, 29, 6, 2, 34)},
 {'end': datetime.datetime(2017, 12, 29, 12, 46, 16),
  'start': datetime.datetime(2017, 12, 29, 12, 21, 3)},
 {'end': datetime.datetime(2017, 12, 29, 14, 43, 46),
  'start': datetime.datetime(2017, 12, 29, 14, 32, 55)},
 {'end': datetime.datetime(2017, 12, 29, 15, 18, 51),
  'start': datetime.datetime(2017, 12, 29, 15, 8, 26)},
 {'end': datetime.datetime(2017, 12, 29, 20, 38, 13),
  'start': datetime.datetime(2017, 12, 29, 20, 33, 34)},
 {'end': datetime.datetime(2017, 12, 30, 13, 54, 33),
  'start': datetime.datetime(2017, 12, 30, 13, 51, 3)},
 {'end': datetime.datetime(2017, 12, 30, 15, 19, 13),
  'start': datetime.datetime(2017, 12, 30, 15, 9, 3)}]


# In[5]:


# Create dictionary to hold results
trip_counts = {'AM': 0, 'PM': 0}
  
# Loop over all trips
for trip in onebike_datetimes:
  # Check to see if the trip starts before noon
  if trip['start'].hour < 12:
    # Increment the counter for before noon
    trip_counts['AM'] += 1
  else:
    # Increment the counter for after noon
    trip_counts['PM'] += 1
  
print(trip_counts)


# # Printing and parsing datetimes
# 
# ```python
# 
# # Printing datetimes
# print(dt.strftime("%H:%M:%S on %d/%m/%Y"))
# 15:19:13 on 2017/12/30
# 
# # Parsing datetimes with strptime (string parse time)
# # Import datetime
# from datetime import datetime
# dt = datetime.strptime("12/30/2017 15:19:13",
#                        "%m/%d/%Y %H:%M:%S")
# 
# # Parsing datetimes with strptime
# # Import datetime
# from datetime import datetime
# # Incorrect format string
# dt = datetime.strptime("2017-12-30 15:19:13", "%Y-%m-%d")
# ValueError: unconverted data remains:
# 15:19:13
#         
# # Parsing datetimes with timestamp
# # A timestamp
# ts = 1514665153.0
# # Convert to datetime and print
# print(datetime.fromtimestamp(ts))
# 2017-12-30 15:19:13
#         
#         
# ```

# ## Turning strings into datetimes
# > 
# > When you download data from the Internet, dates and times usually come to you as strings. Often the first step is to turn those strings into `datetime` objects.
# > 
# > In this exercise, you will practice this transformation.
# > 
# ![image.png](attachment:image.png)

# In[6]:


# Import the datetime class
from datetime import datetime

# Starting string, in YYYY-MM-DD HH:MM:SS format
s = '2017-02-03 00:00:01'

# Write a format string to parse s
fmt = '%Y-%m-%d %H:%M:%S'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)


# In[8]:


# Import the datetime class
from datetime import datetime

# Starting string, in YYYY-MM-DD format
s = '2030-10-15'

# Write a format string to parse s
fmt = '%Y-%m-%d'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)


# In[9]:


# Import the datetime class
from datetime import datetime

# Starting string, in MM/DD/YYYY HH:MM:SS format
s = '12/15/1986 08:00:00'

# Write a format string to parse s
fmt = '%m/%d/%Y %H:%M:%S'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)


# ## Parsing pairs of strings as datetimes
# > 
# > Up until now, you've been working with a pre-processed list of `datetime`s for W20529's trips. For this exercise, you're going to go one step back in the data cleaning pipeline and work with the strings that the data started as.
# > 
# > Explore `onebike_datetime_strings` in the IPython shell to determine the correct format. `datetime` has already been loaded for you.

# In[10]:


onebike_datetime_strings= [('2017-10-01 15:23:25', '2017-10-01 15:26:26'),
 ('2017-10-01 15:42:57', '2017-10-01 17:49:59'),
 ('2017-10-02 06:37:10', '2017-10-02 06:42:53'),
 ('2017-10-02 08:56:45', '2017-10-02 09:18:03'),
 ('2017-10-02 18:23:48', '2017-10-02 18:45:05'),
 ('2017-10-02 18:48:08', '2017-10-02 19:10:54'),
 ('2017-10-02 19:18:10', '2017-10-02 19:31:45'),
 ('2017-10-02 19:37:32', '2017-10-02 19:46:37'),
 ('2017-10-03 08:24:16', '2017-10-03 08:32:27'),
 ('2017-10-03 18:17:07', '2017-10-03 18:27:46'),
 ('2017-10-03 19:24:10', '2017-10-03 19:52:08'),
 ('2017-10-03 20:17:06', '2017-10-03 20:23:52'),
 ('2017-10-03 20:45:21', '2017-10-03 20:57:10'),
 ('2017-10-04 07:04:57', '2017-10-04 07:13:31'),
 ('2017-10-04 07:13:42', '2017-10-04 07:21:54'),
 ('2017-10-04 14:22:12', '2017-10-04 14:50:00'),
 ('2017-10-04 15:07:27', '2017-10-04 15:44:49'),
 ('2017-10-04 15:46:41', '2017-10-04 16:32:33'),
 ('2017-10-04 16:34:44', '2017-10-04 16:46:59'),
 ('2017-10-04 17:26:06', '2017-10-04 17:31:36'),
 ('2017-10-04 17:42:03', '2017-10-04 17:50:41'),
 ('2017-10-05 07:49:02', '2017-10-05 08:12:55'),
 ('2017-10-05 08:26:21', '2017-10-05 08:29:45'),
 ('2017-10-05 08:33:27', '2017-10-05 08:38:31'),
 ('2017-10-05 16:35:35', '2017-10-05 16:51:52'),
 ('2017-10-05 17:53:31', '2017-10-05 18:16:50'),
 ('2017-10-06 08:17:17', '2017-10-06 08:38:01'),
 ('2017-10-06 11:39:40', '2017-10-06 11:50:38'),
 ('2017-10-06 12:59:54', '2017-10-06 13:13:14'),
 ('2017-10-06 13:43:05', '2017-10-06 14:14:56'),
 ('2017-10-06 14:28:15', '2017-10-06 15:09:26'),
 ('2017-10-06 15:50:10', '2017-10-06 16:12:34'),
 ('2017-10-06 16:32:16', '2017-10-06 16:39:31'),
 ('2017-10-06 16:44:08', '2017-10-06 16:48:39'),
 ('2017-10-06 16:53:43', '2017-10-06 17:09:03'),
 ('2017-10-07 11:38:55', '2017-10-07 11:53:06'),
 ('2017-10-07 14:03:36', '2017-10-07 14:07:05'),
 ('2017-10-07 14:20:03', '2017-10-07 14:27:36'),
 ('2017-10-07 14:30:50', '2017-10-07 14:44:51'),
 ('2017-10-08 00:28:26', '2017-10-08 00:30:48'),
 ('2017-10-08 11:16:21', '2017-10-08 11:33:24'),
 ('2017-10-08 12:37:03', '2017-10-08 13:01:29'),
 ('2017-10-08 13:30:37', '2017-10-08 13:57:53'),
 ('2017-10-08 14:16:40', '2017-10-08 15:07:19'),
 ('2017-10-08 15:23:50', '2017-10-08 15:50:01'),
 ('2017-10-08 15:54:12', '2017-10-08 16:17:42'),
 ('2017-10-08 16:28:52', '2017-10-08 16:35:18'),
 ('2017-10-08 23:08:14', '2017-10-08 23:33:41'),
 ('2017-10-08 23:34:49', '2017-10-08 23:45:11'),
 ('2017-10-08 23:46:47', '2017-10-09 00:10:57'),
 ('2017-10-09 00:12:58', '2017-10-09 00:36:40'),
 ('2017-10-09 00:37:02', '2017-10-09 00:53:33'),
 ('2017-10-09 01:23:29', '2017-10-09 01:48:13'),
 ('2017-10-09 01:49:25', '2017-10-09 02:13:35'),
 ('2017-10-09 02:14:11', '2017-10-09 02:29:40'),
 ('2017-10-09 13:04:32', '2017-10-09 13:13:25'),
 ('2017-10-09 14:30:10', '2017-10-09 14:38:55'),
 ('2017-10-09 15:06:47', '2017-10-09 15:11:30'),
 ('2017-10-09 16:43:25', '2017-10-09 16:45:38'),
 ('2017-10-10 15:32:58', '2017-10-10 15:51:24'),
 ('2017-10-10 16:47:55', '2017-10-10 17:03:47'),
 ('2017-10-10 17:51:05', '2017-10-10 18:00:18'),
 ('2017-10-10 18:08:12', '2017-10-10 18:19:11'),
 ('2017-10-10 19:09:35', '2017-10-10 19:14:32'),
 ('2017-10-10 19:17:11', '2017-10-10 19:23:08'),
 ('2017-10-10 19:28:11', '2017-10-10 19:44:40'),
 ('2017-10-10 19:55:35', '2017-10-10 20:11:54'),
 ('2017-10-10 22:20:43', '2017-10-10 22:33:23'),
 ('2017-10-11 04:40:52', '2017-10-11 04:59:22'),
 ('2017-10-11 06:28:58', '2017-10-11 06:40:13'),
 ('2017-10-11 16:41:07', '2017-10-11 17:01:14'),
 ('2017-10-12 08:08:30', '2017-10-12 08:35:03'),
 ('2017-10-12 08:47:02', '2017-10-12 08:59:50'),
 ('2017-10-12 13:13:39', '2017-10-12 13:37:45'),
 ('2017-10-12 13:40:12', '2017-10-12 13:48:17'),
 ('2017-10-12 13:49:56', '2017-10-12 13:53:16'),
 ('2017-10-12 14:33:18', '2017-10-12 14:39:57'),
 ('2017-10-13 15:55:39', '2017-10-13 15:59:41'),
 ('2017-10-17 17:58:48', '2017-10-17 18:01:38'),
 ('2017-10-19 20:21:45', '2017-10-19 20:29:15'),
 ('2017-10-19 21:11:39', '2017-10-19 21:29:37'),
 ('2017-10-19 21:30:01', '2017-10-19 21:47:23'),
 ('2017-10-19 21:47:34', '2017-10-19 21:57:07'),
 ('2017-10-19 21:57:24', '2017-10-19 22:09:52'),
 ('2017-10-21 12:24:09', '2017-10-21 12:36:24'),
 ('2017-10-21 12:36:37', '2017-10-21 12:42:13'),
 ('2017-10-21 13:47:43', '2017-10-22 11:09:36'),
 ('2017-10-22 13:28:53', '2017-10-22 13:31:44'),
 ('2017-10-22 13:47:05', '2017-10-22 13:56:33'),
 ('2017-10-22 14:26:41', '2017-10-22 14:32:39'),
 ('2017-10-22 14:54:41', '2017-10-22 15:09:58'),
 ('2017-10-22 16:40:29', '2017-10-22 16:51:40'),
 ('2017-10-22 17:58:46', '2017-10-22 18:28:37'),
 ('2017-10-22 18:45:16', '2017-10-22 18:50:34'),
 ('2017-10-22 18:56:22', '2017-10-22 19:11:10'),
 ('2017-10-23 10:14:08', '2017-10-23 10:35:32'),
 ('2017-10-23 11:29:36', '2017-10-23 14:38:34'),
 ('2017-10-23 15:04:52', '2017-10-23 15:32:58'),
 ('2017-10-23 15:33:48', '2017-10-23 17:06:47'),
 ('2017-10-23 17:13:16', '2017-10-23 19:31:26'),
 ('2017-10-23 19:55:03', '2017-10-23 20:25:53'),
 ('2017-10-23 21:47:54', '2017-10-23 22:18:04'),
 ('2017-10-23 22:34:12', '2017-10-23 22:48:42'),
 ('2017-10-24 06:55:01', '2017-10-24 07:02:17'),
 ('2017-10-24 14:56:07', '2017-10-24 15:03:16'),
 ('2017-10-24 15:51:36', '2017-10-24 15:59:50'),
 ('2017-10-24 16:31:10', '2017-10-24 16:55:09'),
 ('2017-10-28 14:26:14', '2017-10-28 14:32:34'),
 ('2017-11-01 09:41:54', '2017-11-01 09:52:23'),
 ('2017-11-01 20:16:11', '2017-11-01 20:32:13'),
 ('2017-11-02 19:44:29', '2017-11-02 19:50:56'),
 ('2017-11-02 20:14:37', '2017-11-02 20:30:29'),
 ('2017-11-02 21:35:47', '2017-11-02 21:38:57'),
 ('2017-11-03 09:59:27', '2017-11-03 10:11:46'),
 ('2017-11-03 10:13:22', '2017-11-03 10:32:02'),
 ('2017-11-03 10:44:25', '2017-11-03 10:50:34'),
 ('2017-11-03 16:06:43', '2017-11-03 16:44:38'),
 ('2017-11-03 16:45:54', '2017-11-03 17:00:27'),
 ('2017-11-03 17:07:15', '2017-11-03 17:35:05'),
 ('2017-11-03 17:36:05', '2017-11-03 17:46:48'),
 ('2017-11-03 17:50:31', '2017-11-03 18:00:03'),
 ('2017-11-03 19:22:56', '2017-11-03 19:45:51'),
 ('2017-11-04 13:14:10', '2017-11-04 13:26:15'),
 ('2017-11-04 14:18:37', '2017-11-04 14:30:05'),
 ('2017-11-04 14:45:59', '2017-11-04 15:03:20'),
 ('2017-11-04 15:16:03', '2017-11-04 15:44:30'),
 ('2017-11-04 16:37:46', '2017-11-04 16:58:22'),
 ('2017-11-04 17:13:19', '2017-11-04 17:34:50'),
 ('2017-11-04 18:10:34', '2017-11-04 18:58:44'),
 ('2017-11-05 01:56:50', '2017-11-05 01:01:04'),
 ('2017-11-05 08:33:33', '2017-11-05 08:53:46'),
 ('2017-11-05 08:58:08', '2017-11-05 09:03:39'),
 ('2017-11-05 11:05:08', '2017-11-05 11:30:05'),
 ('2017-11-06 08:50:18', '2017-11-06 08:59:05'),
 ('2017-11-06 09:04:03', '2017-11-06 09:13:47'),
 ('2017-11-06 16:19:36', '2017-11-06 17:02:55'),
 ('2017-11-06 17:21:27', '2017-11-06 17:34:06'),
 ('2017-11-06 17:36:01', '2017-11-06 17:57:32'),
 ('2017-11-06 17:59:52', '2017-11-06 18:15:08'),
 ('2017-11-06 18:18:36', '2017-11-06 18:21:17'),
 ('2017-11-06 19:24:31', '2017-11-06 19:37:57'),
 ('2017-11-06 19:49:16', '2017-11-06 20:03:14'),
 ('2017-11-07 07:50:48', '2017-11-07 08:01:32'),
 ('2017-11-08 13:11:51', '2017-11-08 13:18:05'),
 ('2017-11-08 21:34:47', '2017-11-08 21:46:05'),
 ('2017-11-08 22:02:30', '2017-11-08 22:04:47'),
 ('2017-11-09 07:01:11', '2017-11-09 07:12:10'),
 ('2017-11-09 08:02:02', '2017-11-09 08:08:28'),
 ('2017-11-09 08:19:59', '2017-11-09 08:32:24'),
 ('2017-11-09 08:41:31', '2017-11-09 08:48:59'),
 ('2017-11-09 09:00:06', '2017-11-09 09:09:24'),
 ('2017-11-09 09:09:37', '2017-11-09 09:24:25'),
 ('2017-11-09 13:14:37', '2017-11-09 13:25:39'),
 ('2017-11-09 15:20:07', '2017-11-09 15:31:10'),
 ('2017-11-09 18:47:08', '2017-11-09 18:53:10'),
 ('2017-11-09 23:35:02', '2017-11-09 23:43:35'),
 ('2017-11-10 07:51:33', '2017-11-10 08:02:28'),
 ('2017-11-10 08:38:28', '2017-11-10 08:42:09'),
 ('2017-11-11 18:05:25', '2017-11-11 18:13:14'),
 ('2017-11-11 19:39:12', '2017-11-11 19:46:22'),
 ('2017-11-11 21:13:19', '2017-11-11 21:16:31'),
 ('2017-11-12 09:46:19', '2017-11-12 09:51:43'),
 ('2017-11-13 13:33:42', '2017-11-13 13:54:15'),
 ('2017-11-14 08:40:29', '2017-11-14 08:55:52'),
 ('2017-11-15 06:14:05', '2017-11-15 06:30:06'),
 ('2017-11-15 08:14:59', '2017-11-15 08:23:44'),
 ('2017-11-15 10:16:44', '2017-11-15 10:33:41'),
 ('2017-11-15 10:33:58', '2017-11-15 10:54:14'),
 ('2017-11-15 11:02:15', '2017-11-15 11:14:42'),
 ('2017-11-16 09:27:41', '2017-11-16 09:38:49'),
 ('2017-11-16 09:57:41', '2017-11-16 10:18:00'),
 ('2017-11-16 17:25:05', '2017-11-16 17:44:47'),
 ('2017-11-17 13:45:54', '2017-11-17 16:36:56'),
 ('2017-11-17 19:12:49', '2017-11-17 19:31:15'),
 ('2017-11-18 10:49:06', '2017-11-18 10:55:45'),
 ('2017-11-18 11:32:12', '2017-11-18 11:44:16'),
 ('2017-11-18 18:09:01', '2017-11-18 18:14:31'),
 ('2017-11-18 18:53:10', '2017-11-18 19:01:29'),
 ('2017-11-19 14:15:41', '2017-11-19 14:31:49'),
 ('2017-11-20 21:19:19', '2017-11-20 21:41:09'),
 ('2017-11-20 22:39:48', '2017-11-20 23:23:37'),
 ('2017-11-21 17:44:25', '2017-11-21 17:51:32'),
 ('2017-11-21 18:20:52', '2017-11-21 18:34:51'),
 ('2017-11-21 18:47:32', '2017-11-21 18:51:50'),
 ('2017-11-21 19:07:57', '2017-11-21 19:14:33'),
 ('2017-11-21 20:04:56', '2017-11-21 20:08:54'),
 ('2017-11-21 21:55:47', '2017-11-21 22:08:12'),
 ('2017-11-23 23:47:43', '2017-11-23 23:57:56'),
 ('2017-11-24 06:41:25', '2017-11-24 06:53:15'),
 ('2017-11-24 06:58:56', '2017-11-24 07:33:24'),
 ('2017-11-26 12:25:49', '2017-11-26 12:41:36'),
 ('2017-11-27 05:29:04', '2017-11-27 05:54:13'),
 ('2017-11-27 06:06:47', '2017-11-27 06:11:01'),
 ('2017-11-27 06:45:14', '2017-11-27 06:55:39'),
 ('2017-11-27 09:39:44', '2017-11-27 09:47:43'),
 ('2017-11-27 11:09:18', '2017-11-27 11:20:46'),
 ('2017-11-27 11:31:46', '2017-11-27 11:35:44'),
 ('2017-11-27 12:07:14', '2017-11-27 12:12:36'),
 ('2017-11-27 12:21:40', '2017-11-27 12:26:44'),
 ('2017-11-27 17:26:31', '2017-11-27 17:36:07'),
 ('2017-11-27 18:11:49', '2017-11-27 18:29:04'),
 ('2017-11-27 19:36:16', '2017-11-27 19:47:17'),
 ('2017-11-27 20:12:57', '2017-11-27 20:17:33'),
 ('2017-11-28 08:18:06', '2017-11-28 08:41:53'),
 ('2017-11-28 19:17:23', '2017-11-28 19:34:01'),
 ('2017-11-28 19:34:15', '2017-11-28 19:46:24'),
 ('2017-11-28 21:27:29', '2017-11-28 21:39:32'),
 ('2017-11-29 07:47:38', '2017-11-29 07:51:18'),
 ('2017-11-29 09:50:12', '2017-11-29 09:53:44'),
 ('2017-11-29 17:03:42', '2017-11-29 17:16:21'),
 ('2017-11-29 18:19:15', '2017-11-29 18:23:43'),
 ('2017-12-01 17:03:58', '2017-12-01 17:10:12'),
 ('2017-12-02 07:55:56', '2017-12-02 08:01:01'),
 ('2017-12-02 09:16:14', '2017-12-02 09:21:18'),
 ('2017-12-02 19:48:29', '2017-12-02 19:53:18'),
 ('2017-12-03 14:36:29', '2017-12-03 15:20:09'),
 ('2017-12-03 16:04:02', '2017-12-03 16:25:30'),
 ('2017-12-03 16:40:26', '2017-12-03 16:43:58'),
 ('2017-12-03 17:20:17', '2017-12-03 18:04:33'),
 ('2017-12-04 08:34:24', '2017-12-04 08:51:00'),
 ('2017-12-04 17:49:26', '2017-12-04 17:53:57'),
 ('2017-12-04 18:38:52', '2017-12-04 18:50:33'),
 ('2017-12-04 21:39:20', '2017-12-04 21:46:58'),
 ('2017-12-04 21:54:21', '2017-12-04 21:56:17'),
 ('2017-12-05 08:50:50', '2017-12-05 08:52:54'),
 ('2017-12-06 08:19:38', '2017-12-06 08:24:14'),
 ('2017-12-06 18:19:19', '2017-12-06 18:28:11'),
 ('2017-12-06 18:28:55', '2017-12-06 18:33:12'),
 ('2017-12-06 20:03:29', '2017-12-06 20:21:38'),
 ('2017-12-06 20:36:42', '2017-12-06 20:39:57'),
 ('2017-12-07 05:54:51', '2017-12-07 06:01:15'),
 ('2017-12-08 16:47:18', '2017-12-08 16:55:49'),
 ('2017-12-08 19:15:02', '2017-12-08 19:29:12'),
 ('2017-12-09 22:39:37', '2017-12-09 22:47:19'),
 ('2017-12-09 23:00:10', '2017-12-09 23:05:32'),
 ('2017-12-10 00:39:24', '2017-12-10 00:56:02'),
 ('2017-12-10 01:02:42', '2017-12-10 01:08:09'),
 ('2017-12-10 01:08:57', '2017-12-10 01:11:30'),
 ('2017-12-10 13:49:09', '2017-12-10 13:51:41'),
 ('2017-12-10 15:14:29', '2017-12-10 15:18:19'),
 ('2017-12-10 15:31:07', '2017-12-10 15:36:28'),
 ('2017-12-10 16:20:06', '2017-12-10 16:30:31'),
 ('2017-12-10 17:07:54', '2017-12-10 17:14:25'),
 ('2017-12-10 17:23:47', '2017-12-10 17:45:25'),
 ('2017-12-11 06:17:06', '2017-12-11 06:34:04'),
 ('2017-12-11 09:08:41', '2017-12-11 09:12:21'),
 ('2017-12-11 09:15:41', '2017-12-11 09:20:18'),
 ('2017-12-12 08:55:53', '2017-12-12 08:59:34'),
 ('2017-12-13 17:14:56', '2017-12-13 17:18:32'),
 ('2017-12-13 18:52:16', '2017-12-13 19:00:45'),
 ('2017-12-14 09:01:10', '2017-12-14 09:11:06'),
 ('2017-12-14 09:12:59', '2017-12-14 09:19:06'),
 ('2017-12-14 11:54:33', '2017-12-14 12:02:00'),
 ('2017-12-14 14:40:23', '2017-12-14 14:44:40'),
 ('2017-12-14 15:08:55', '2017-12-14 15:26:24'),
 ('2017-12-14 17:46:17', '2017-12-14 18:09:04'),
 ('2017-12-15 09:08:12', '2017-12-15 09:23:45'),
 ('2017-12-16 09:33:46', '2017-12-16 09:36:17'),
 ('2017-12-16 11:02:31', '2017-12-16 11:05:04'),
 ('2017-12-17 10:09:47', '2017-12-17 10:32:03'),
 ('2017-12-18 08:02:36', '2017-12-18 08:07:34'),
 ('2017-12-18 16:03:00', '2017-12-18 16:09:20'),
 ('2017-12-18 16:30:07', '2017-12-18 16:53:12'),
 ('2017-12-18 19:18:23', '2017-12-18 19:22:08'),
 ('2017-12-18 20:14:46', '2017-12-18 20:17:47'),
 ('2017-12-19 19:14:08', '2017-12-19 19:23:49'),
 ('2017-12-19 19:39:36', '2017-12-19 19:43:46'),
 ('2017-12-20 08:05:14', '2017-12-20 08:10:46'),
 ('2017-12-20 08:15:45', '2017-12-20 08:29:50'),
 ('2017-12-20 08:33:32', '2017-12-20 08:38:09'),
 ('2017-12-20 13:43:36', '2017-12-20 13:54:39'),
 ('2017-12-20 18:57:53', '2017-12-20 19:06:54'),
 ('2017-12-21 07:21:11', '2017-12-21 07:32:03'),
 ('2017-12-21 08:01:58', '2017-12-21 08:06:15'),
 ('2017-12-21 13:20:54', '2017-12-21 13:33:49'),
 ('2017-12-21 15:26:08', '2017-12-21 15:34:27'),
 ('2017-12-21 18:09:46', '2017-12-21 18:38:50'),
 ('2017-12-22 16:14:21', '2017-12-22 16:21:46'),
 ('2017-12-22 16:29:17', '2017-12-22 16:34:14'),
 ('2017-12-25 12:49:51', '2017-12-25 13:18:27'),
 ('2017-12-25 13:46:44', '2017-12-25 14:20:50'),
 ('2017-12-26 10:40:16', '2017-12-26 10:53:45'),
 ('2017-12-27 16:56:12', '2017-12-27 17:17:39'),
 ('2017-12-29 06:02:34', '2017-12-29 06:12:30'),
 ('2017-12-29 12:21:03', '2017-12-29 12:46:16'),
 ('2017-12-29 14:32:55', '2017-12-29 14:43:46'),
 ('2017-12-29 15:08:26', '2017-12-29 15:18:51'),
 ('2017-12-29 20:33:34', '2017-12-29 20:38:13'),
 ('2017-12-30 13:51:03', '2017-12-30 13:54:33'),
 ('2017-12-30 15:09:03', '2017-12-30 15:19:13')]


# In[11]:


# Write down the format string
fmt = "%Y-%m-%d %H:%M:%S"

# Initialize a list for holding the pairs of datetime objects
onebike_datetimes = []

# Loop over all trips
for (start, end) in onebike_datetime_strings:
  trip = {'start': datetime.strptime(start, fmt),
          'end': datetime.strptime(end, fmt)}
  
  # Append the trip
  onebike_datetimes.append(trip)


# ## Recreating ISO format with strftime()
# > 
# > In the last chapter, you used `strftime()` to create strings from `date` objects. Now that you know about `datetime` objects, let's practice doing something similar.
# > 
# > Re-create the `.isoformat()` method, using `.strftime()`, and print the first trip start in our data set.

# In[14]:


# Import datetime
from datetime import datetime

# Pull out the start of the first trip
first_start = onebike_datetimes[0]['start']

# Format to feed to strftime()
fmt = "%Y-%m-%dT%H:%M:%S"

# Print out date with .isoformat(), then with .strftime() to compare
print(first_start.isoformat())
print(first_start.strftime(fmt))


# ## Unix timestamps
# > 
# > Datetimes are sometimes stored as Unix timestamps: the number of seconds since January 1, 1970. This is especially common with computer infrastructure, like the log files that websites keep when they get visitors.

# In[15]:


# Import datetime
from datetime import datetime

# Starting timestamps
timestamps = [1514665153, 1514664543]

# Datetime objects
dts = []

# Loop
for ts in timestamps:
  dts.append(datetime.fromtimestamp(ts))
  
# Print results
print(dts)


# # Working with durations
# 
# ```python
# # Working with durations
# # Create example datetimes
# start = datetime(2017, 10, 8, 23, 46, 47)
# end = datetime(2017, 10, 9, 0, 10, 57)
# # Subtract datetimes to create a timedelta
# duration = end - start
# 
# # Subtract datetimes to create a timedelta
# print(duration.total_seconds())
# 
# # Creating timedeltas
# # Import timedelta
# from datetime import timedelta
# # Create a timedelta
# delta1 = timedelta(seconds=1)
# 
# 
# # Creating timedeltas
# print(start)
# 2017-10-08 23:46:47
# # One second later
# print(start + delta1)
# 2017-10-08 23:46:48
#         
# # Negative timedeltas
# # Create a negative timedelta of one week
# delta3 = timedelta(weeks=-1)
# print(start)
# 2017-10-08 23:46:47
# # One week earlier
# print(start + delta3)
# 2017-10-01 23:46:47
# ```

# ## Turning pairs of datetimes into durations
# > 
# > When working with timestamps, we often want to know how much time has elapsed between events. Thankfully, we can use `datetime` arithmetic to ask Python to do the heavy lifting for us so we don't need to worry about day, month, or year boundaries. Let's calculate the number of seconds that the bike was out of the dock for each trip.
# > 
# > Continuing our work from a previous coding exercise, the bike trip data has been loaded as the list `onebike_datetimes`. Each element of the list consists of two `datetime` objects, corresponding to the start and end of a trip, respectively.

# In[17]:


# Initialize a list for all the trip durations
onebike_durations = []

for trip in onebike_datetimes:
  # Create a timedelta object corresponding to the length of the trip
  trip_duration = trip['end'] - trip['start']
  
  # Get the total elapsed seconds in trip_duration
  trip_length_seconds = trip_duration.total_seconds()
  
  # Append the results to our list
  onebike_durations.append(trip_length_seconds)


# ## Average trip time
# > 
# > W20529 took 291 trips in our data set. How long were the trips on average? We can use the built-in Python functions `sum()` and `len()` to make this calculation.
# > 
# > Based on your last coding exercise, the data has been loaded as `onebike_durations`. Each entry is a number of seconds that the bike was out of the dock.

# In[18]:


# What was the total duration of all trips?
total_elapsed_time = sum(onebike_durations)

# What was the total number of trips?
number_of_trips = len(onebike_durations)
  
# Divide the total duration by the number of trips
print(total_elapsed_time / number_of_trips)


# ## The long and the short of why time is hard
# > 
# > Out of 291 trips taken by W20529, how long was the longest? How short was the shortest? Does anything look fishy?
# > 
# > As before, data has been loaded as `onebike_durations`.

# In[20]:


# Calculate shortest and longest trips
shortest_trip = min(onebike_durations)
longest_trip = max(onebike_durations)

# Print out the results
print("The shortest trip was " + str(shortest_trip) + " seconds")
print("The longest trip was " + str(longest_trip) + " seconds")


# In[ ]:




