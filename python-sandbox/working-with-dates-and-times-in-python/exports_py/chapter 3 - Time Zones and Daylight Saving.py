#!/usr/bin/env python
# coding: utf-8

# # UTC offsets
# 
# ```python
# 
# # UTC
# # Import relevant classes
# from datetime import datetime, timedelta, timezone
# 
# # US Eastern Standard time zone
# ET = timezone(timedelta(hours=-5))
# # Timezone-aware datetime
# dt = datetime(2017, 12, 30, 15, 9, 3, tzinfo = ET)
# print(dt)
# '2017-12-30 15:09:03-05:00'
# 
# # UTC
# # India Standard time zone
# IST = timezone(timedelta(hours=5, minutes=30))
# # Convert to IST
# print(dt.astimezone(IST))
# '2017-12-31 01:39:03+05:30'
# 
# # Adjusting timezone vs changing tzinfo
# print(dt)
# '2017-12-30 15:09:03-05:00'
# print(dt.replace(tzinfo=timezone.utc))
# '2017-12-30 15:09:03+00:00'
# # Change original to match UTC
# print(dt.astimezone(timezone.utc))
# '2017-12-30 20:09:03+00:00'
# 
# 
# ```

# ## Creating timezone aware datetimes
# > 
# > In this exercise, you will practice setting timezones manually.

# In[1]:


# Import datetime, timezone
from datetime import datetime, timezone

# October 1, 2017 at 15:26:26, UTC
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=timezone.utc)

# Print results
print(dt.isoformat())


# In[2]:


# Import datetime, timedelta, timezone
from datetime import datetime, timedelta, timezone

# Create a timezone for Pacific Standard Time, or UTC-8
pst = timezone(timedelta(hours=-8))

# October 1, 2017 at 15:26:26, UTC-8
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=pst)

# Print results
print(dt.isoformat())


# In[3]:


# Import datetime, timedelta, timezone
from datetime import datetime, timedelta, timezone

# Create a timezone for Australian Eastern Daylight Time, or UTC+11
aedt = timezone(timedelta(hours=+11))

# October 1, 2017 at 15:26:26, UTC+11
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=aedt)

# Print results
print(dt.isoformat())


# ## Setting timezones
# > 
# > Now that you have the hang of setting timezones one at a time, let's look at setting them for the first ten trips that W20529 took.
# > 
# > `timezone` and `timedelta` have already been imported. Make the change using `.replace()`

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


# Create a timezone object corresponding to UTC-4
edt = timezone(timedelta(hours=-4))

# Loop over trips, updating the start and end datetimes to be in UTC-4
for trip in onebike_datetimes[:10]:
  # Update trip['start'] and trip['end']
  trip['start'] = trip['start'].replace(tzinfo=edt)
  trip['end'] = trip['end'].replace(tzinfo=edt)


# ## What time did the bike leave in UTC?
# > 
# > Having set the timezone for the first ten rides that W20529 took, let's see what time the bike left in UTC. We've already loaded the results of the previous exercise into memory.

# In[7]:


# Loop over the trips
for trip in onebike_datetimes[:10]:
  # Pull out the start
  dt = trip['start']
  # Move dt to be in UTC
  dt = dt.astimezone(timezone.utc)
  
  # Print the start time in UTC
  print('Original:', trip['start'], '| UTC:', dt.isoformat())


# # Time zone database
# 
# ```python
# 
# # Time zone database
# # Imports
# from datetime import datetime
# from dateutil import tz
# 
# # Eastern time
# et = tz.gettz('America/New_York')
# 
# # Time zone database
# # Last ride
# last = datetime(2017, 12, 30, 15, 9, 3, tzinfo=et)
# print(last)
# '2017-12-30 15:09:03-05:00'
# 
# # First ride
# first = datetime(2017, 10, 1, 15, 23, 25, tzinfo=et)
# print(first)
# '2017-10-01 15:23:25-04:00'
# 
# ```

# ## Putting the bike trips into the right time zone
# > 
# > Instead of setting the timezones for W20529 by hand, let's assign them to their IANA timezone: 'America/New\_York'. Since we know their political jurisdiction, we don't need to look up their UTC offset. Python will do that for us.

# In[9]:


# Import tz
from dateutil import tz

# Create a timezone object for Eastern Time
et = tz.gettz('America/New_York')

# Loop over trips, updating the datetimes to be in Eastern Time
for trip in onebike_datetimes[:10]:
  # Update trip['start'] and trip['end']
  trip['start'] = trip['start'].replace(tzinfo=et)
  trip['end'] = trip['end'].replace(tzinfo=et)


# ## What time did the bike leave? (Global edition)
# > 
# > When you need to move a `datetime` from one timezone into another, use `.astimezone()` and `tz`. Often you will be moving things into UTC, but for fun let's try moving things from 'America/New\_York' into a few different time zones.

# In[10]:


# Create the timezone object
uk = tz.gettz('Europe/London')

# Pull out the start of the first trip
local = onebike_datetimes[0]['start']

# What time was it in the UK?
notlocal = local.astimezone(uk)

# Print them out and see the difference
print(local.isoformat())
print(notlocal.isoformat())


# In[11]:


# Create the timezone object
ist = tz.gettz('Asia/Kolkata')

# Pull out the start of the first trip
local = onebike_datetimes[0]['start']

# What time was it in India?
notlocal = local.astimezone(ist)

# Print them out and see the difference
print(local.isoformat())
print(notlocal.isoformat())


# In[13]:


# Create the timezone object
sm = tz.gettz('Pacific/Apia')

# Pull out the start of the first trip
local = onebike_datetimes[0]['start']

# What time was it in Samoa?
notlocal = local.astimezone(sm)

# Print them out and see the difference
print(local.isoformat())
print(notlocal.isoformat())


# # Starting daylight saving time
# 
# fiendish challenges
# 
# 
# ```python
# # Start of Daylight Saving Time
# spring_ahead_159am = datetime(2017, 3, 12, 1, 59, 59)
# spring_ahead_159am.isoformat()
# '2017-03-12T01:59:59'
# spring_ahead_3am = datetime(2017, 3, 12, 3, 0, 0)
# spring_ahead_3am.isoformat()
# '2017-03-12T03:00:00'
# (spring_ahead_3am - spring_ahead_159am).total_seconds()
# 3601
# 
# 
# # Start of Daylight Saving Time
# from datetime import timezone, timedelta
# EST = timezone(timedelta(hours=-5))
# EDT = timezone(timedelta(hours=-4))
# 
# spring_ahead_159am = spring_ahead_159am.replace(tzinfo = EST)
# spring_ahead_159am.isoformat()
# '2017-03-12T01:59:59-05:00'
# spring_ahead_3am = spring_ahead_159am.replace(tzinfo = EDT)
# spring_ahead_3am.isoformat()
# '2017-03-12T03:00:00-04:00'
# (spring_ahead_3am - spring_ahead_159am).seconds
# 1
# 
# # Start of Daylight Saving Time
# # Using dateutil
# # Import tz
# from dateutil import tz
# # Create eastern timezone
# eastern = tz.gettz('America/New_York')
# # 2017-03-12 01:59:59 in Eastern Time (EST)
# spring_ahead_159am = datetime(2017, 3, 12, 1, 59, 59,
# tzinfo = eastern)
# # 2017-03-12 03:00:00 in Eastern Time (EDT)
# spring_ahead_3am = datetime(2017, 3, 12, 3, 0, 0,
# tzinfo = eastern)
# 
# 
# ```

# ## How many hours elapsed around daylight saving?
# > 
# > Since our bike data takes place in the fall, you'll have to do something else to learn about the start of daylight savings time.
# > 
# > Let's look at March 12, 2017, in the Eastern United States, when Daylight Saving kicked in at 2 AM.
# > 
# > If you create a `datetime` for midnight that night, and add 6 hours to it, how much time will have elapsed?

# In[14]:


# Import datetime, timedelta, tz, timezone
from datetime import datetime, timedelta, timezone
from dateutil import tz

# Start on March 12, 2017, midnight, then add 6 hours
start = datetime(2017, 3, 12, tzinfo = tz.gettz('America/New_York'))
end = start + timedelta(hours=+6)
print(start.isoformat() + " to " + end.isoformat())


# In[16]:


# Import datetime, timedelta, tz, timezone
from datetime import datetime, timedelta, timezone
from dateutil import tz

# Start on March 12, 2017, midnight, then add 6 hours
start = datetime(2017, 3, 12, tzinfo = tz.gettz('America/New_York'))
end = start + timedelta(hours=6)
print(start.isoformat() + " to " + end.isoformat())

# How many hours have elapsed?
print((end - start).total_seconds()/(60*60))


# In[18]:


# What if we move to UTC?
print((end.astimezone(timezone.utc) - start.astimezone(timezone.utc))      .total_seconds()/(60*60))


# ![image.png](attachment:image.png)

# ## March 29, throughout a decade
# > 
# > Daylight Saving rules are complicated: they're different in different places, they change over time, and they usually start on a Sunday (and so they move around the calendar).
# > 
# > For example, in the United Kingdom, as of the time this lesson was written, Daylight Saving begins on the last Sunday in March. Let's look at the UTC offset for March 29, at midnight, for the years 2000 to 2010.

# In[20]:


# Import datetime and tz
from datetime import datetime
from dateutil import tz

# Create starting date
dt = datetime(2000, 3, 29, tzinfo = tz.gettz('Europe/London'))

# Loop over the dates, replacing the year, and print the ISO timestamp
for y in range(2000, 2011):
  print(dt.replace(year=y).isoformat())


# # Ending daylight saving time
# 
# ```python
# 
# # Ending Daylight Saving Time
# eastern = tz.gettz('US/Eastern')
# # 2017-11-05 01:00:00
# first_1am = datetime(2017, 11, 5, 1, 0, 0,
# tzinfo = eastern)
# tz.datetime_ambiguous(first_1am)
# True
# # 2017-11-05 01:00:00 again
# second_1am = datetime(2017, 11, 5, 1, 0, 0,
# tzinfo = eastern)
# second_1am = tz.enfold(second_1am)
# 
# # Ending Daylight Saving Time
# (first_1am - second_1am).total_seconds()
# 0.0
# first_1am = first_1am.astimezone(tz.UTC)
# second_1am = second_1am.astimezone(tz.UTC)
# (second_1am - first_1am).total_seconds()
# 3600.0
# 
# ```

# ## Finding ambiguous datetimes
# > 
# > At the end of lesson 2, we saw something anomalous in our bike trip duration data. Let's see if we can identify what the problem might be.
# > 
# > The data is loaded as `onebike_datetimes`, and `tz` has already been imported from `dateutil`.

# In[27]:


import datetime
from dateutil import tz

onebike_datetimes = [{'end': datetime.datetime(2017, 10, 1, 15, 26, 26, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 1, 15, 23, 25, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 1, 17, 49, 59, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 1, 15, 42, 57, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 2, 6, 42, 53, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 2, 6, 37, 10, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 2, 9, 18, 3, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 2, 8, 56, 45, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 2, 18, 45, 5, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 2, 18, 23, 48, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 2, 19, 10, 54, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 2, 18, 48, 8, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 2, 19, 31, 45, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 2, 19, 18, 10, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 2, 19, 46, 37, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 2, 19, 37, 32, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 3, 8, 32, 27, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 3, 8, 24, 16, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 3, 18, 27, 46, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 3, 18, 17, 7, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 3, 19, 52, 8, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 3, 19, 24, 10, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 3, 20, 23, 52, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 3, 20, 17, 6, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 3, 20, 57, 10, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 3, 20, 45, 21, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 4, 7, 13, 31, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 4, 7, 4, 57, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 4, 7, 21, 54, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 4, 7, 13, 42, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 4, 14, 50, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 4, 14, 22, 12, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 4, 15, 44, 49, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 4, 15, 7, 27, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 4, 16, 32, 33, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 4, 15, 46, 41, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 4, 16, 46, 59, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 4, 16, 34, 44, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 4, 17, 31, 36, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 4, 17, 26, 6, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 4, 17, 50, 41, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 4, 17, 42, 3, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 5, 8, 12, 55, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 5, 7, 49, 2, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 5, 8, 29, 45, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 5, 8, 26, 21, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 5, 8, 38, 31, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 5, 8, 33, 27, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 5, 16, 51, 52, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 5, 16, 35, 35, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 5, 18, 16, 50, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 5, 17, 53, 31, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 6, 8, 38, 1, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 6, 8, 17, 17, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 6, 11, 50, 38, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 6, 11, 39, 40, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 6, 13, 13, 14, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 6, 12, 59, 54, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 6, 14, 14, 56, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 6, 13, 43, 5, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 6, 15, 9, 26, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 6, 14, 28, 15, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 6, 16, 12, 34, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 6, 15, 50, 10, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 6, 16, 39, 31, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 6, 16, 32, 16, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 6, 16, 48, 39, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 6, 16, 44, 8, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 6, 17, 9, 3, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 6, 16, 53, 43, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 7, 11, 53, 6, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 7, 11, 38, 55, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 7, 14, 7, 5, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 7, 14, 3, 36, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 7, 14, 27, 36, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 7, 14, 20, 3, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 7, 14, 44, 51, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 7, 14, 30, 50, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 8, 0, 30, 48, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 8, 0, 28, 26, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 8, 11, 33, 24, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 8, 11, 16, 21, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 8, 13, 1, 29, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 8, 12, 37, 3, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 8, 13, 57, 53, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 8, 13, 30, 37, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 8, 15, 7, 19, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 8, 14, 16, 40, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 8, 15, 50, 1, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 8, 15, 23, 50, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 8, 16, 17, 42, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 8, 15, 54, 12, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 8, 16, 35, 18, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 8, 16, 28, 52, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 8, 23, 33, 41, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 8, 23, 8, 14, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 8, 23, 45, 11, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 8, 23, 34, 49, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 9, 0, 10, 57, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 8, 23, 46, 47, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 9, 0, 36, 40, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 9, 0, 12, 58, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 9, 0, 53, 33, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 9, 0, 37, 2, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 9, 1, 48, 13, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 9, 1, 23, 29, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 9, 2, 13, 35, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 9, 1, 49, 25, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 9, 2, 29, 40, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 9, 2, 14, 11, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 9, 13, 13, 25, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 9, 13, 4, 32, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 9, 14, 38, 55, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 9, 14, 30, 10, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 9, 15, 11, 30, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 9, 15, 6, 47, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 9, 16, 45, 38, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 9, 16, 43, 25, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 10, 15, 51, 24, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 10, 15, 32, 58, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 10, 17, 3, 47, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 10, 16, 47, 55, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 10, 18, 0, 18, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 10, 17, 51, 5, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 10, 18, 19, 11, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 10, 18, 8, 12, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 10, 19, 14, 32, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 10, 19, 9, 35, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 10, 19, 23, 8, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 10, 19, 17, 11, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 10, 19, 44, 40, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 10, 19, 28, 11, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 10, 20, 11, 54, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 10, 19, 55, 35, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 10, 22, 33, 23, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 10, 22, 20, 43, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 11, 4, 59, 22, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 11, 4, 40, 52, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 11, 6, 40, 13, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 11, 6, 28, 58, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 11, 17, 1, 14, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 11, 16, 41, 7, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 12, 8, 35, 3, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 12, 8, 8, 30, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 12, 8, 59, 50, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 12, 8, 47, 2, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 12, 13, 37, 45, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 12, 13, 13, 39, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 12, 13, 48, 17, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 12, 13, 40, 12, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 12, 13, 53, 16, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 12, 13, 49, 56, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 12, 14, 39, 57, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 12, 14, 33, 18, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 13, 15, 59, 41, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 13, 15, 55, 39, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 17, 18, 1, 38, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 17, 17, 58, 48, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 19, 20, 29, 15, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 19, 20, 21, 45, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 19, 21, 29, 37, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 19, 21, 11, 39, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 19, 21, 47, 23, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 19, 21, 30, 1, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 19, 21, 57, 7, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 19, 21, 47, 34, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 19, 22, 9, 52, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 19, 21, 57, 24, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 21, 12, 36, 24, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 21, 12, 24, 9, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 21, 12, 42, 13, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 21, 12, 36, 37, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 22, 11, 9, 36, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 21, 13, 47, 43, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 22, 13, 31, 44, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 22, 13, 28, 53, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 22, 13, 56, 33, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 22, 13, 47, 5, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 22, 14, 32, 39, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 22, 14, 26, 41, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 22, 15, 9, 58, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 22, 14, 54, 41, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 22, 16, 51, 40, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 22, 16, 40, 29, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 22, 18, 28, 37, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 22, 17, 58, 46, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 22, 18, 50, 34, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 22, 18, 45, 16, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 22, 19, 11, 10, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 22, 18, 56, 22, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 23, 10, 35, 32, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 23, 10, 14, 8, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 23, 14, 38, 34, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 23, 11, 29, 36, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 23, 15, 32, 58, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 23, 15, 4, 52, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 23, 17, 6, 47, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 23, 15, 33, 48, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 23, 19, 31, 26, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 23, 17, 13, 16, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 23, 20, 25, 53, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 23, 19, 55, 3, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 23, 22, 18, 4, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 23, 21, 47, 54, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 23, 22, 48, 42, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 23, 22, 34, 12, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 24, 7, 2, 17, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 24, 6, 55, 1, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 24, 15, 3, 16, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 24, 14, 56, 7, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 24, 15, 59, 50, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 24, 15, 51, 36, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 24, 16, 55, 9, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 24, 16, 31, 10, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 10, 28, 14, 32, 34, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 10, 28, 14, 26, 14, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 1, 9, 52, 23, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 1, 9, 41, 54, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 1, 20, 32, 13, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 1, 20, 16, 11, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 2, 19, 50, 56, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 2, 19, 44, 29, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 2, 20, 30, 29, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 2, 20, 14, 37, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 2, 21, 38, 57, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 2, 21, 35, 47, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 3, 10, 11, 46, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 3, 9, 59, 27, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 3, 10, 32, 2, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 3, 10, 13, 22, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 3, 10, 50, 34, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 3, 10, 44, 25, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 3, 16, 44, 38, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 3, 16, 6, 43, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 3, 17, 0, 27, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 3, 16, 45, 54, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 3, 17, 35, 5, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 3, 17, 7, 15, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 3, 17, 46, 48, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 3, 17, 36, 5, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 3, 18, 0, 3, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 3, 17, 50, 31, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 3, 19, 45, 51, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 3, 19, 22, 56, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 4, 13, 26, 15, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 4, 13, 14, 10, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 4, 14, 30, 5, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 4, 14, 18, 37, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 4, 15, 3, 20, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 4, 14, 45, 59, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 4, 15, 44, 30, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 4, 15, 16, 3, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 4, 16, 58, 22, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 4, 16, 37, 46, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 4, 17, 34, 50, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 4, 17, 13, 19, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 4, 18, 58, 44, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 4, 18, 10, 34, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 5, 1, 1, 4, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 5, 1, 56, 50, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 5, 8, 53, 46, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 5, 8, 33, 33, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 5, 9, 3, 39, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 5, 8, 58, 8, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 5, 11, 30, 5, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 5, 11, 5, 8, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 6, 8, 59, 5, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 6, 8, 50, 18, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 6, 9, 13, 47, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 6, 9, 4, 3, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 6, 17, 2, 55, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 6, 16, 19, 36, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 6, 17, 34, 6, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 6, 17, 21, 27, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 6, 17, 57, 32, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 6, 17, 36, 1, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 6, 18, 15, 8, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 6, 17, 59, 52, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 6, 18, 21, 17, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 6, 18, 18, 36, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 6, 19, 37, 57, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 6, 19, 24, 31, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 6, 20, 3, 14, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 6, 19, 49, 16, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 7, 8, 1, 32, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 7, 7, 50, 48, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 8, 13, 18, 5, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 8, 13, 11, 51, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 8, 21, 46, 5, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 8, 21, 34, 47, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 8, 22, 4, 47, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 8, 22, 2, 30, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 9, 7, 12, 10, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 9, 7, 1, 11, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 9, 8, 8, 28, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 9, 8, 2, 2, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 9, 8, 32, 24, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 9, 8, 19, 59, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 9, 8, 48, 59, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 9, 8, 41, 31, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 9, 9, 9, 24, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 9, 9, 0, 6, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 9, 9, 24, 25, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 9, 9, 9, 37, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 9, 13, 25, 39, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 9, 13, 14, 37, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 9, 15, 31, 10, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 9, 15, 20, 7, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 9, 18, 53, 10, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 9, 18, 47, 8, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 9, 23, 43, 35, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 9, 23, 35, 2, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 10, 8, 2, 28, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 10, 7, 51, 33, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 10, 8, 42, 9, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 10, 8, 38, 28, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 11, 18, 13, 14, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 11, 18, 5, 25, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 11, 19, 46, 22, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 11, 19, 39, 12, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 11, 21, 16, 31, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 11, 21, 13, 19, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 12, 9, 51, 43, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 12, 9, 46, 19, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 13, 13, 54, 15, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 13, 13, 33, 42, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 14, 8, 55, 52, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 14, 8, 40, 29, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 15, 6, 30, 6, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 15, 6, 14, 5, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 15, 8, 23, 44, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 15, 8, 14, 59, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 15, 10, 33, 41, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 15, 10, 16, 44, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 15, 10, 54, 14, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 15, 10, 33, 58, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 15, 11, 14, 42, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 15, 11, 2, 15, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 16, 9, 38, 49, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 16, 9, 27, 41, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 16, 10, 18, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 16, 9, 57, 41, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 16, 17, 44, 47, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 16, 17, 25, 5, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 17, 16, 36, 56, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 17, 13, 45, 54, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 17, 19, 31, 15, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 17, 19, 12, 49, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 18, 10, 55, 45, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 18, 10, 49, 6, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 18, 11, 44, 16, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 18, 11, 32, 12, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 18, 18, 14, 31, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 18, 18, 9, 1, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 18, 19, 1, 29, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 18, 18, 53, 10, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 19, 14, 31, 49, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 19, 14, 15, 41, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 20, 21, 41, 9, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 20, 21, 19, 19, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 20, 23, 23, 37, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 20, 22, 39, 48, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 21, 17, 51, 32, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 21, 17, 44, 25, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 21, 18, 34, 51, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 21, 18, 20, 52, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 21, 18, 51, 50, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 21, 18, 47, 32, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 21, 19, 14, 33, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 21, 19, 7, 57, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 21, 20, 8, 54, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 21, 20, 4, 56, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 21, 22, 8, 12, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 21, 21, 55, 47, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 23, 23, 57, 56, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 23, 23, 47, 43, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 24, 6, 53, 15, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 24, 6, 41, 25, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 24, 7, 33, 24, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 24, 6, 58, 56, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 26, 12, 41, 36, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 26, 12, 25, 49, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 5, 54, 13, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 5, 29, 4, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 6, 11, 1, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 6, 6, 47, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 6, 55, 39, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 6, 45, 14, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 9, 47, 43, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 9, 39, 44, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 11, 20, 46, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 11, 9, 18, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 11, 35, 44, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 11, 31, 46, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 12, 12, 36, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 12, 7, 14, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 12, 26, 44, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 12, 21, 40, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 17, 36, 7, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 17, 26, 31, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 18, 29, 4, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 18, 11, 49, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 19, 47, 17, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 19, 36, 16, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 27, 20, 17, 33, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 27, 20, 12, 57, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 28, 8, 41, 53, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 28, 8, 18, 6, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 28, 19, 34, 1, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 28, 19, 17, 23, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 28, 19, 46, 24, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 28, 19, 34, 15, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 28, 21, 39, 32, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 28, 21, 27, 29, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 29, 7, 51, 18, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 29, 7, 47, 38, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 29, 9, 53, 44, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 29, 9, 50, 12, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 29, 17, 16, 21, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 29, 17, 3, 42, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 11, 29, 18, 23, 43, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 11, 29, 18, 19, 15, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 1, 17, 10, 12, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 1, 17, 3, 58, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 2, 8, 1, 1, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 2, 7, 55, 56, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 2, 9, 21, 18, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 2, 9, 16, 14, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 2, 19, 53, 18, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 2, 19, 48, 29, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 3, 15, 20, 9, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 3, 14, 36, 29, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 3, 16, 25, 30, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 3, 16, 4, 2, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 3, 16, 43, 58, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 3, 16, 40, 26, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 3, 18, 4, 33, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 3, 17, 20, 17, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 4, 8, 51, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 4, 8, 34, 24, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 4, 17, 53, 57, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 4, 17, 49, 26, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 4, 18, 50, 33, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 4, 18, 38, 52, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 4, 21, 46, 58, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 4, 21, 39, 20, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 4, 21, 56, 17, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 4, 21, 54, 21, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 5, 8, 52, 54, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 5, 8, 50, 50, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 6, 8, 24, 14, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 6, 8, 19, 38, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 6, 18, 28, 11, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 6, 18, 19, 19, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 6, 18, 33, 12, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 6, 18, 28, 55, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 6, 20, 21, 38, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 6, 20, 3, 29, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 6, 20, 39, 57, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 6, 20, 36, 42, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 7, 6, 1, 15, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 7, 5, 54, 51, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 8, 16, 55, 49, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 8, 16, 47, 18, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 8, 19, 29, 12, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 8, 19, 15, 2, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 9, 22, 47, 19, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 9, 22, 39, 37, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 9, 23, 5, 32, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 9, 23, 0, 10, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 10, 0, 56, 2, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 10, 0, 39, 24, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 10, 1, 8, 9, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 10, 1, 2, 42, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 10, 1, 11, 30, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 10, 1, 8, 57, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 10, 13, 51, 41, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 10, 13, 49, 9, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 10, 15, 18, 19, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 10, 15, 14, 29, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 10, 15, 36, 28, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 10, 15, 31, 7, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 10, 16, 30, 31, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 10, 16, 20, 6, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 10, 17, 14, 25, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 10, 17, 7, 54, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 10, 17, 45, 25, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 10, 17, 23, 47, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 11, 6, 34, 4, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 11, 6, 17, 6, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 11, 9, 12, 21, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 11, 9, 8, 41, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 11, 9, 20, 18, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 11, 9, 15, 41, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 12, 8, 59, 34, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 12, 8, 55, 53, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 13, 17, 18, 32, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 13, 17, 14, 56, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 13, 19, 0, 45, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 13, 18, 52, 16, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 14, 9, 11, 6, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 14, 9, 1, 10, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 14, 9, 19, 6, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 14, 9, 12, 59, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 14, 12, 2, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 14, 11, 54, 33, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 14, 14, 44, 40, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 14, 14, 40, 23, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 14, 15, 26, 24, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 14, 15, 8, 55, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 14, 18, 9, 4, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 14, 17, 46, 17, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 15, 9, 23, 45, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 15, 9, 8, 12, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 16, 9, 36, 17, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 16, 9, 33, 46, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 16, 11, 5, 4, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 16, 11, 2, 31, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 17, 10, 32, 3, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 17, 10, 9, 47, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 18, 8, 7, 34, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 18, 8, 2, 36, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 18, 16, 9, 20, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 18, 16, 3, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 18, 16, 53, 12, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 18, 16, 30, 7, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 18, 19, 22, 8, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 18, 19, 18, 23, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 18, 20, 17, 47, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 18, 20, 14, 46, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 19, 19, 23, 49, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 19, 19, 14, 8, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 19, 19, 43, 46, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 19, 19, 39, 36, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 20, 8, 10, 46, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 20, 8, 5, 14, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 20, 8, 29, 50, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 20, 8, 15, 45, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 20, 8, 38, 9, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 20, 8, 33, 32, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 20, 13, 54, 39, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 20, 13, 43, 36, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 20, 19, 6, 54, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 20, 18, 57, 53, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 21, 7, 32, 3, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 21, 7, 21, 11, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 21, 8, 6, 15, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 21, 8, 1, 58, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 21, 13, 33, 49, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 21, 13, 20, 54, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 21, 15, 34, 27, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 21, 15, 26, 8, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 21, 18, 38, 50, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 21, 18, 9, 46, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 22, 16, 21, 46, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 22, 16, 14, 21, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 22, 16, 34, 14, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 22, 16, 29, 17, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 25, 13, 18, 27, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 25, 12, 49, 51, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 25, 14, 20, 50, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 25, 13, 46, 44, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 26, 10, 53, 45, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 26, 10, 40, 16, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 27, 17, 17, 39, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 27, 16, 56, 12, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 29, 6, 12, 30, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 29, 6, 2, 34, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 29, 12, 46, 16, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 29, 12, 21, 3, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 29, 14, 43, 46, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 29, 14, 32, 55, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 29, 15, 18, 51, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 29, 15, 8, 26, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 29, 20, 38, 13, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 29, 20, 33, 34, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 30, 13, 54, 33, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 30, 13, 51, 3, tzinfo=tz.gettz('US/Eastern'))},
 {'end': datetime.datetime(2017, 12, 30, 15, 19, 13, tzinfo=tz.gettz('US/Eastern')),
  'start': datetime.datetime(2017, 12, 30, 15, 9, 3, tzinfo=tz.gettz('US/Eastern'))}]


# In[28]:


# Loop over trips
for trip in onebike_datetimes:
  # Rides with ambiguous start
  if tz.datetime_ambiguous(trip['start']):
    print("Ambiguous start at " + str(trip['start']))
  # Rides with ambiguous end
  if tz.datetime_ambiguous(trip['end']):
    print("Ambiguous end at " + str(trip['end']))


# ## Cleaning daylight saving data with fold
# > 
# > As we've just discovered, there is a ride in our data set which is being messed up by a Daylight Savings shift. Let's clean up the data set so we actually have a correct minimum ride length. We can use the fact that we know the end of the ride happened after the beginning to fix up the duration messed up by the shift out of Daylight Savings.
# > 
# > Since Python does not handle `tz.enfold()` when doing arithmetic, we must put our datetime objects into UTC, where ambiguities have been resolved.
# > 
# > `onebike_datetimes` is already loaded and in the right timezone. `tz` and `timezone` have been imported. Use `tz.UTC` for the timezone.

# In[29]:


trip_durations = []
for trip in onebike_datetimes:
  # When the start is later than the end, set the fold to be 1
  if trip['start'] > trip['end']:
    trip['end'] = tz.enfold(trip['end'])
  # Convert to UTC
  start = trip['start'].astimezone(timezone.utc)
  end = trip['end'].astimezone(timezone.utc)

  # Subtract the difference
  trip_length_seconds = (end-start).total_seconds()
  trip_durations.append(trip_length_seconds)

# Take the shortest trip duration
print("Shortest trip: " + str(min(trip_durations)))


# In[ ]:




