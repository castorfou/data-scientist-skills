#!/usr/bin/env python
# coding: utf-8

# # Dates in Python
# 
# ```python
# # Creating date objects
# # Import date
# from datetime import date
# # Create dates
# two_hurricanes_dates = [date(2016, 10, 7), date(2017, 6, 21)]
# 
# # Attributes of a date
# # Import date
# from datetime import date
# # Create dates
# two_hurricanes_dates = [date(2016, 10, 7), date(2017, 6, 21)]
# print(two_hurricanes_dates[0].year)
# print(two_hurricanes_dates[0].month)
# print(two_hurricanes_dates[0].day)
# 
# # Finding the weekday of a date
# print(two_hurricanes_dates[0].weekday())
# 4
# # Weekdays in Python
# # 0 = Monday
# # 1 = Tuesday
# # 2 = Wednesday
# # ...
# # 6 = Sunday
# 
# 
# ```

# ## Which day of the week?
# > 
# > Hurricane Andrew, which hit Florida on August 24, 1992, was one of the costliest and deadliest hurricanes in US history. Which day of the week did it make landfall?
# > 
# > Let's walk through all of the steps to figure this out.

# In[1]:


# Import date from datetime
from datetime import date

# Create a date object
hurricane_andrew = date(1992, 8, 24)

# Which day of the week is the date?
print(hurricane_andrew.weekday())


# ## How many hurricanes come early?
# > 
# > In this chapter, you will work with a list of the hurricanes that made landfall in Florida from 1950 to 2017. There were 235 in total. Check out the variable `florida_hurricane_dates`, which has all of these dates.
# > 
# > Atlantic hurricane season officially begins on June 1. How many hurricanes since 1950 have made landfall in Florida before the official start of hurricane season?

# In[3]:


import datetime
florida_hurricane_dates=[datetime.date(1950, 8, 31),
 datetime.date(1950, 9, 5),
 datetime.date(1950, 10, 18),
 datetime.date(1950, 10, 21),
 datetime.date(1951, 5, 18),
 datetime.date(1951, 10, 2),
 datetime.date(1952, 2, 3),
 datetime.date(1952, 8, 30),
 datetime.date(1953, 6, 6),
 datetime.date(1953, 8, 29),
 datetime.date(1953, 9, 20),
 datetime.date(1953, 9, 26),
 datetime.date(1953, 10, 9),
 datetime.date(1955, 8, 21),
 datetime.date(1956, 7, 6),
 datetime.date(1956, 9, 24),
 datetime.date(1956, 10, 15),
 datetime.date(1957, 6, 8),
 datetime.date(1957, 9, 8),
 datetime.date(1958, 9, 4),
 datetime.date(1959, 6, 18),
 datetime.date(1959, 10, 8),
 datetime.date(1959, 10, 18),
 datetime.date(1960, 7, 29),
 datetime.date(1960, 9, 10),
 datetime.date(1960, 9, 15),
 datetime.date(1960, 9, 23),
 datetime.date(1961, 9, 11),
 datetime.date(1961, 10, 29),
 datetime.date(1962, 8, 26),
 datetime.date(1963, 10, 21),
 datetime.date(1964, 6, 6),
 datetime.date(1964, 8, 27),
 datetime.date(1964, 9, 10),
 datetime.date(1964, 9, 20),
 datetime.date(1964, 10, 5),
 datetime.date(1964, 10, 14),
 datetime.date(1965, 6, 15),
 datetime.date(1965, 9, 8),
 datetime.date(1965, 9, 30),
 datetime.date(1966, 6, 9),
 datetime.date(1966, 6, 30),
 datetime.date(1966, 7, 24),
 datetime.date(1966, 10, 4),
 datetime.date(1968, 6, 4),
 datetime.date(1968, 6, 18),
 datetime.date(1968, 7, 5),
 datetime.date(1968, 8, 10),
 datetime.date(1968, 8, 28),
 datetime.date(1968, 9, 26),
 datetime.date(1968, 10, 19),
 datetime.date(1969, 6, 9),
 datetime.date(1969, 8, 18),
 datetime.date(1969, 8, 29),
 datetime.date(1969, 9, 7),
 datetime.date(1969, 9, 21),
 datetime.date(1969, 10, 1),
 datetime.date(1969, 10, 2),
 datetime.date(1969, 10, 21),
 datetime.date(1970, 5, 25),
 datetime.date(1970, 7, 22),
 datetime.date(1970, 8, 6),
 datetime.date(1970, 9, 13),
 datetime.date(1970, 9, 27),
 datetime.date(1971, 8, 10),
 datetime.date(1971, 8, 13),
 datetime.date(1971, 8, 29),
 datetime.date(1971, 9, 1),
 datetime.date(1971, 9, 16),
 datetime.date(1971, 10, 13),
 datetime.date(1972, 5, 28),
 datetime.date(1972, 6, 19),
 datetime.date(1972, 9, 5),
 datetime.date(1973, 6, 7),
 datetime.date(1973, 6, 23),
 datetime.date(1973, 9, 3),
 datetime.date(1973, 9, 25),
 datetime.date(1974, 6, 25),
 datetime.date(1974, 9, 8),
 datetime.date(1974, 9, 27),
 datetime.date(1974, 10, 7),
 datetime.date(1975, 6, 27),
 datetime.date(1975, 7, 29),
 datetime.date(1975, 9, 23),
 datetime.date(1975, 10, 1),
 datetime.date(1975, 10, 16),
 datetime.date(1976, 5, 23),
 datetime.date(1976, 6, 11),
 datetime.date(1976, 8, 19),
 datetime.date(1976, 9, 13),
 datetime.date(1977, 8, 27),
 datetime.date(1977, 9, 5),
 datetime.date(1978, 6, 22),
 datetime.date(1979, 7, 11),
 datetime.date(1979, 9, 3),
 datetime.date(1979, 9, 12),
 datetime.date(1979, 9, 24),
 datetime.date(1980, 8, 7),
 datetime.date(1980, 11, 18),
 datetime.date(1981, 8, 17),
 datetime.date(1982, 6, 18),
 datetime.date(1982, 9, 11),
 datetime.date(1983, 8, 28),
 datetime.date(1984, 9, 9),
 datetime.date(1984, 9, 27),
 datetime.date(1984, 10, 26),
 datetime.date(1985, 7, 23),
 datetime.date(1985, 8, 15),
 datetime.date(1985, 10, 10),
 datetime.date(1985, 11, 21),
 datetime.date(1986, 6, 26),
 datetime.date(1986, 8, 13),
 datetime.date(1987, 8, 14),
 datetime.date(1987, 9, 7),
 datetime.date(1987, 10, 12),
 datetime.date(1987, 11, 4),
 datetime.date(1988, 5, 30),
 datetime.date(1988, 8, 4),
 datetime.date(1988, 8, 13),
 datetime.date(1988, 8, 23),
 datetime.date(1988, 9, 4),
 datetime.date(1988, 9, 10),
 datetime.date(1988, 9, 13),
 datetime.date(1988, 11, 23),
 datetime.date(1989, 9, 22),
 datetime.date(1990, 5, 25),
 datetime.date(1990, 10, 9),
 datetime.date(1990, 10, 12),
 datetime.date(1991, 6, 30),
 datetime.date(1991, 10, 16),
 datetime.date(1992, 6, 25),
 datetime.date(1992, 8, 24),
 datetime.date(1992, 9, 29),
 datetime.date(1993, 6, 1),
 datetime.date(1994, 7, 3),
 datetime.date(1994, 8, 15),
 datetime.date(1994, 10, 2),
 datetime.date(1994, 11, 16),
 datetime.date(1995, 6, 5),
 datetime.date(1995, 7, 27),
 datetime.date(1995, 8, 2),
 datetime.date(1995, 8, 23),
 datetime.date(1995, 10, 4),
 datetime.date(1996, 7, 11),
 datetime.date(1996, 9, 2),
 datetime.date(1996, 10, 8),
 datetime.date(1996, 10, 18),
 datetime.date(1997, 7, 19),
 datetime.date(1998, 9, 3),
 datetime.date(1998, 9, 20),
 datetime.date(1998, 9, 25),
 datetime.date(1998, 11, 5),
 datetime.date(1999, 8, 29),
 datetime.date(1999, 9, 15),
 datetime.date(1999, 9, 21),
 datetime.date(1999, 10, 15),
 datetime.date(2000, 8, 23),
 datetime.date(2000, 9, 9),
 datetime.date(2000, 9, 18),
 datetime.date(2000, 9, 22),
 datetime.date(2000, 10, 3),
 datetime.date(2001, 6, 12),
 datetime.date(2001, 8, 6),
 datetime.date(2001, 9, 14),
 datetime.date(2001, 11, 5),
 datetime.date(2002, 7, 13),
 datetime.date(2002, 8, 4),
 datetime.date(2002, 9, 4),
 datetime.date(2002, 9, 14),
 datetime.date(2002, 9, 26),
 datetime.date(2002, 10, 3),
 datetime.date(2002, 10, 11),
 datetime.date(2003, 4, 20),
 datetime.date(2003, 6, 30),
 datetime.date(2003, 7, 25),
 datetime.date(2003, 8, 14),
 datetime.date(2003, 8, 30),
 datetime.date(2003, 9, 6),
 datetime.date(2003, 9, 13),
 datetime.date(2004, 8, 12),
 datetime.date(2004, 8, 13),
 datetime.date(2004, 9, 5),
 datetime.date(2004, 9, 13),
 datetime.date(2004, 9, 16),
 datetime.date(2004, 10, 10),
 datetime.date(2005, 6, 11),
 datetime.date(2005, 7, 6),
 datetime.date(2005, 7, 10),
 datetime.date(2005, 8, 25),
 datetime.date(2005, 9, 12),
 datetime.date(2005, 9, 20),
 datetime.date(2005, 10, 5),
 datetime.date(2005, 10, 24),
 datetime.date(2006, 6, 13),
 datetime.date(2006, 8, 30),
 datetime.date(2007, 5, 9),
 datetime.date(2007, 6, 2),
 datetime.date(2007, 8, 23),
 datetime.date(2007, 9, 8),
 datetime.date(2007, 9, 13),
 datetime.date(2007, 9, 22),
 datetime.date(2007, 10, 31),
 datetime.date(2007, 12, 13),
 datetime.date(2008, 7, 16),
 datetime.date(2008, 7, 22),
 datetime.date(2008, 8, 18),
 datetime.date(2008, 8, 31),
 datetime.date(2008, 9, 2),
 datetime.date(2009, 8, 16),
 datetime.date(2009, 8, 21),
 datetime.date(2009, 11, 9),
 datetime.date(2010, 6, 30),
 datetime.date(2010, 7, 23),
 datetime.date(2010, 8, 10),
 datetime.date(2010, 8, 31),
 datetime.date(2010, 9, 29),
 datetime.date(2011, 7, 18),
 datetime.date(2011, 8, 25),
 datetime.date(2011, 9, 3),
 datetime.date(2011, 10, 28),
 datetime.date(2011, 11, 9),
 datetime.date(2012, 5, 28),
 datetime.date(2012, 6, 23),
 datetime.date(2012, 8, 25),
 datetime.date(2012, 10, 25),
 datetime.date(2015, 8, 30),
 datetime.date(2015, 10, 1),
 datetime.date(2016, 6, 6),
 datetime.date(2016, 9, 1),
 datetime.date(2016, 9, 14),
 datetime.date(2016, 10, 7),
 datetime.date(2017, 6, 21),
 datetime.date(2017, 7, 31),
 datetime.date(2017, 9, 10),
 datetime.date(2017, 10, 29)]


# In[4]:


# Counter for how many before June 1
early_hurricanes = 0

# We loop over the dates
for hurricane in florida_hurricane_dates:
  # Check if the month is before June (month number 6)
  if hurricane.month < 6:
    early_hurricanes = early_hurricanes + 1
    
print(early_hurricanes)


# # Math with dates
# 
# 
# ```python
# # Import date
# from datetime import date
# # Create our dates
# d1 = date(2017, 11, 5)
# d2 = date(2017, 12, 4)
# l = [d1, d2]
# print(min(l)) 
# 2017-11-05
# 
# # Subtract two dates
# delta = d2 - d1 #of type timedelta
# print(delta.days)
# 29
# 
# # Import timedelta
# from datetime import timedelta
# # Create a 29 day timedelta
# td = timedelta(days=29)
# print(d1 + td)
# 2017-12-04
# 
# 
# ```

# ## Subtracting dates
# > 
# > Python `date` objects let us treat calendar dates as something similar to numbers: we can compare them, sort them, add, and even subtract them. This lets us do math with dates in a way that would be a pain to do by hand.
# > 
# > The 2007 Florida hurricane season was one of the busiest on record, with 8 hurricanes in one year. The first one hit on May 9th, 2007, and the last one hit on December 13th, 2007. How many days elapsed between the first and last hurricane in 2007?

# In[5]:


# Import date
from datetime import date

# Create a date object for May 9th, 2007
start = date(2007, 5, 9)

# Create a date object for December 13th, 2007
end = date(2007, 12, 13)

# Subtract the two dates and print the number of days
print((end - start).days)


# ## Counting events per calendar month
# > 
# > Hurricanes can make landfall in Florida throughout the year. As we've already discussed, some months are more hurricane-prone than others.
# > 
# > Using `florida_hurricane_dates`, let's see how hurricanes in Florida were distributed across months throughout the year.
# > 
# > We've created a dictionary called `hurricanes_each_month` to hold your counts and set the initial counts to zero. You will loop over the list of hurricanes, incrementing the correct month in `hurricanes_each_month` as you go, and then print the result.

# In[6]:


# A dictionary to count hurricanes per calendar month
hurricanes_each_month = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6:0,
		  				 7: 0, 8:0, 9:0, 10:0, 11:0, 12:0}

# Loop over all hurricanes
for hurricane in florida_hurricane_dates:
  # Pull out the month
  month = hurricane.month
  # Increment the count in your dictionary by one
  hurricanes_each_month[month] +=1
  
print(hurricanes_each_month)


# ## Putting a list of dates in order
# > 
# > Much like numbers and strings, `date` objects in Python can be put in order. Earlier dates come before later ones, and so we can sort a list of `date` objects from earliest to latest.
# > 
# > What if our Florida hurricane dates had been scrambled? We've gone ahead and shuffled them so they're in random order and saved the results as `dates_scrambled`. Your job is to put them back in chronological order, and then print the first and last dates from this sorted list.

# In[7]:


import datetime
dates_scrambled = [datetime.date(1988, 8, 4),
 datetime.date(1990, 10, 12),
 datetime.date(2003, 4, 20),
 datetime.date(1971, 9, 1),
 datetime.date(1988, 8, 23),
 datetime.date(1994, 8, 15),
 datetime.date(2002, 8, 4),
 datetime.date(1988, 5, 30),
 datetime.date(2003, 9, 13),
 datetime.date(2009, 8, 21),
 datetime.date(1978, 6, 22),
 datetime.date(1969, 6, 9),
 datetime.date(1976, 6, 11),
 datetime.date(1976, 8, 19),
 datetime.date(1966, 6, 9),
 datetime.date(1968, 7, 5),
 datetime.date(1987, 11, 4),
 datetime.date(1988, 8, 13),
 datetime.date(2007, 12, 13),
 datetime.date(1994, 11, 16),
 datetime.date(2003, 9, 6),
 datetime.date(1971, 8, 13),
 datetime.date(1981, 8, 17),
 datetime.date(1998, 9, 25),
 datetime.date(1968, 9, 26),
 datetime.date(1968, 6, 4),
 datetime.date(1998, 11, 5),
 datetime.date(2008, 8, 18),
 datetime.date(1987, 8, 14),
 datetime.date(1988, 11, 23),
 datetime.date(2010, 9, 29),
 datetime.date(1985, 7, 23),
 datetime.date(2017, 7, 31),
 datetime.date(1955, 8, 21),
 datetime.date(1986, 6, 26),
 datetime.date(1963, 10, 21),
 datetime.date(2011, 10, 28),
 datetime.date(2011, 11, 9),
 datetime.date(1997, 7, 19),
 datetime.date(2007, 6, 2),
 datetime.date(2002, 9, 14),
 datetime.date(1992, 9, 29),
 datetime.date(1971, 10, 13),
 datetime.date(1962, 8, 26),
 datetime.date(1964, 8, 27),
 datetime.date(1984, 9, 27),
 datetime.date(1973, 9, 25),
 datetime.date(1969, 10, 21),
 datetime.date(1994, 7, 3),
 datetime.date(1958, 9, 4),
 datetime.date(1985, 11, 21),
 datetime.date(2011, 9, 3),
 datetime.date(1972, 6, 19),
 datetime.date(1991, 6, 30),
 datetime.date(2004, 8, 12),
 datetime.date(2007, 9, 8),
 datetime.date(1952, 2, 3),
 datetime.date(1965, 9, 30),
 datetime.date(2000, 9, 22),
 datetime.date(2002, 9, 26),
 datetime.date(1950, 9, 5),
 datetime.date(1966, 10, 4),
 datetime.date(1970, 5, 25),
 datetime.date(1979, 9, 24),
 datetime.date(1960, 9, 23),
 datetime.date(2007, 8, 23),
 datetime.date(2009, 8, 16),
 datetime.date(1996, 10, 18),
 datetime.date(2012, 10, 25),
 datetime.date(2011, 8, 25),
 datetime.date(1951, 5, 18),
 datetime.date(1980, 8, 7),
 datetime.date(1979, 9, 3),
 datetime.date(1953, 9, 26),
 datetime.date(1968, 10, 19),
 datetime.date(2009, 11, 9),
 datetime.date(1999, 8, 29),
 datetime.date(2015, 10, 1),
 datetime.date(2008, 9, 2),
 datetime.date(2004, 10, 10),
 datetime.date(2004, 9, 16),
 datetime.date(1992, 8, 24),
 datetime.date(2000, 9, 9),
 datetime.date(1971, 9, 16),
 datetime.date(1996, 9, 2),
 datetime.date(1998, 9, 3),
 datetime.date(1951, 10, 2),
 datetime.date(1979, 9, 12),
 datetime.date(2007, 10, 31),
 datetime.date(1953, 10, 9),
 datetime.date(1952, 8, 30),
 datetime.date(1969, 9, 7),
 datetime.date(2015, 8, 30),
 datetime.date(1959, 10, 8),
 datetime.date(2002, 7, 13),
 datetime.date(1961, 10, 29),
 datetime.date(2007, 5, 9),
 datetime.date(2016, 10, 7),
 datetime.date(1964, 9, 20),
 datetime.date(1979, 7, 11),
 datetime.date(1950, 10, 18),
 datetime.date(2008, 8, 31),
 datetime.date(2012, 8, 25),
 datetime.date(1966, 7, 24),
 datetime.date(2010, 8, 10),
 datetime.date(2005, 8, 25),
 datetime.date(2003, 6, 30),
 datetime.date(1956, 7, 6),
 datetime.date(1974, 9, 8),
 datetime.date(1966, 6, 30),
 datetime.date(2016, 9, 14),
 datetime.date(1968, 6, 18),
 datetime.date(1982, 9, 11),
 datetime.date(1976, 9, 13),
 datetime.date(1975, 7, 29),
 datetime.date(2007, 9, 13),
 datetime.date(1970, 9, 27),
 datetime.date(1969, 10, 2),
 datetime.date(2010, 8, 31),
 datetime.date(1995, 10, 4),
 datetime.date(1969, 8, 29),
 datetime.date(1984, 10, 26),
 datetime.date(1973, 9, 3),
 datetime.date(1976, 5, 23),
 datetime.date(2001, 11, 5),
 datetime.date(2010, 6, 30),
 datetime.date(1985, 10, 10),
 datetime.date(1970, 7, 22),
 datetime.date(1972, 5, 28),
 datetime.date(1982, 6, 18),
 datetime.date(2001, 8, 6),
 datetime.date(1953, 8, 29),
 datetime.date(1965, 9, 8),
 datetime.date(1964, 9, 10),
 datetime.date(1959, 10, 18),
 datetime.date(1957, 6, 8),
 datetime.date(1988, 9, 10),
 datetime.date(2005, 6, 11),
 datetime.date(1953, 6, 6),
 datetime.date(2003, 8, 30),
 datetime.date(2002, 10, 3),
 datetime.date(1968, 8, 10),
 datetime.date(1999, 10, 15),
 datetime.date(2002, 9, 4),
 datetime.date(2001, 6, 12),
 datetime.date(2017, 9, 10),
 datetime.date(2005, 10, 5),
 datetime.date(2005, 7, 10),
 datetime.date(1973, 6, 7),
 datetime.date(1999, 9, 15),
 datetime.date(2005, 9, 20),
 datetime.date(1995, 6, 5),
 datetime.date(2003, 7, 25),
 datetime.date(2004, 9, 13),
 datetime.date(1964, 6, 6),
 datetime.date(1973, 6, 23),
 datetime.date(2005, 9, 12),
 datetime.date(2012, 6, 23),
 datetime.date(1961, 9, 11),
 datetime.date(1990, 5, 25),
 datetime.date(2017, 6, 21),
 datetime.date(1975, 6, 27),
 datetime.date(1959, 6, 18),
 datetime.date(2004, 9, 5),
 datetime.date(1987, 10, 12),
 datetime.date(1995, 7, 27),
 datetime.date(1964, 10, 14),
 datetime.date(1970, 8, 6),
 datetime.date(1969, 10, 1),
 datetime.date(1996, 10, 8),
 datetime.date(1968, 8, 28),
 datetime.date(1956, 10, 15),
 datetime.date(1975, 9, 23),
 datetime.date(1970, 9, 13),
 datetime.date(1975, 10, 16),
 datetime.date(1990, 10, 9),
 datetime.date(2005, 10, 24),
 datetime.date(1950, 8, 31),
 datetime.date(2000, 10, 3),
 datetime.date(2002, 10, 11),
 datetime.date(1983, 8, 28),
 datetime.date(1960, 7, 29),
 datetime.date(1950, 10, 21),
 datetime.date(1995, 8, 2),
 datetime.date(1956, 9, 24),
 datetime.date(2016, 9, 1),
 datetime.date(1993, 6, 1),
 datetime.date(1987, 9, 7),
 datetime.date(2012, 5, 28),
 datetime.date(1995, 8, 23),
 datetime.date(1969, 8, 18),
 datetime.date(2001, 9, 14),
 datetime.date(2000, 8, 23),
 datetime.date(1974, 10, 7),
 datetime.date(1986, 8, 13),
 datetime.date(1977, 8, 27),
 datetime.date(2008, 7, 16),
 datetime.date(1996, 7, 11),
 datetime.date(1988, 9, 4),
 datetime.date(1975, 10, 1),
 datetime.date(2003, 8, 14),
 datetime.date(1957, 9, 8),
 datetime.date(2005, 7, 6),
 datetime.date(1960, 9, 15),
 datetime.date(1974, 9, 27),
 datetime.date(1965, 6, 15),
 datetime.date(1999, 9, 21),
 datetime.date(2004, 8, 13),
 datetime.date(1994, 10, 2),
 datetime.date(1971, 8, 10),
 datetime.date(2008, 7, 22),
 datetime.date(2000, 9, 18),
 datetime.date(1960, 9, 10),
 datetime.date(2006, 6, 13),
 datetime.date(2017, 10, 29),
 datetime.date(1972, 9, 5),
 datetime.date(1964, 10, 5),
 datetime.date(1991, 10, 16),
 datetime.date(1969, 9, 21),
 datetime.date(1998, 9, 20),
 datetime.date(1977, 9, 5),
 datetime.date(1988, 9, 13),
 datetime.date(1974, 6, 25),
 datetime.date(2010, 7, 23),
 datetime.date(2007, 9, 22),
 datetime.date(1984, 9, 9),
 datetime.date(1989, 9, 22),
 datetime.date(1992, 6, 25),
 datetime.date(1971, 8, 29),
 datetime.date(1953, 9, 20),
 datetime.date(1985, 8, 15),
 datetime.date(2016, 6, 6),
 datetime.date(2006, 8, 30),
 datetime.date(1980, 11, 18),
 datetime.date(2011, 7, 18)]


# In[8]:


# Print the first and last scrambled dates
print(dates_scrambled[0])
print(dates_scrambled[-1])


# In[10]:


# Put the dates in order
dates_ordered = sorted(dates_scrambled)

# Print the first and last ordered dates
print(dates_ordered[0])
print(dates_ordered[-1])


# # Turning dates into strings
# 
# ```python
# 
# # ISO 8601 format
# from datetime import date
# # Example date
# d = date(2017, 11, 5)
# # ISO format: YYYY-MM-DD
# print(d)
# 2017-11-05
# # Express the date in ISO 8601 format and put it in a list
# print( [d.isoformat()] )
# ['2017-11-05']
# 
# # Every other format
# d.strftime()
# 
# # Every other format: strftime
# # Example date
# d = date(2017, 1, 5)
# print(d.strftime("%Y"))
# 2017
# # Format string with more text in it
# print(d.strftime("Year is %Y"))
# Year is 2017
# 
# # Every other format: strftime
# # Format: YYYY/MM/DD
# print(d.strftime("%Y/%m/%d"))
# 2017/01/05
# ```

# ## Printing dates in a friendly format
# > 
# > Because people may want to see dates in many different formats, Python comes with very flexible functions for turning `date` objects into strings.
# > 
# > Let's see what event was recorded first in the Florida hurricane data set. In this exercise, you will format the earliest date in the `florida_hurriance_dates` list in two ways so you can decide which one you want to use: either the ISO standard or the typical US style.

# In[12]:


# Assign the earliest date to first_date
first_date = min(florida_hurricane_dates)

# Convert to ISO and US formats
iso = "Our earliest hurricane date: " + first_date.isoformat()
us = "Our earliest hurricane date: " + first_date.strftime("%m/%d/%Y")

print("ISO: " + iso)
print("US: " + us)


# ## Representing dates in different ways
# > 
# > `date` objects in Python have a great number of ways they can be printed out as strings. In some cases, you want to know the date in a clear, language-agnostic format. In other cases, you want something which can fit into a paragraph and flow naturally.
# > 
# > Let's try printing out the same date, August 26, 1992 (the day that Hurricane Andrew made landfall in Florida), in a number of different ways, to practice using the `.strftime()` method.
# > 
# > A date object called `andrew` has already been created.

# In[13]:


# Import date
from datetime import date

# Create a date object
andrew = date(1992, 8, 26)

# Print the date in the format 'YYYY-MM'
print(andrew.strftime("%Y-%m"))


# In[14]:


# Print the date in the format 'MONTH (YYYY)'
print(andrew.strftime("%B (%Y)"))


# In[15]:


# Print the date in the format 'YYYY-DDD'
print(andrew.strftime("%Y-%j"))


# In[ ]:




