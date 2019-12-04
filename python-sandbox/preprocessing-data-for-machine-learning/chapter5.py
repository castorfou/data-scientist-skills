#!/usr/bin/env python
# coding: utf-8

# # UFOs and preprocessing
# 

# ## Checking column types
# Take a look at the UFO dataset's column types using the dtypes attribute. Two columns jump out for transformation: the seconds column, which is a numeric column but is being read in as object, and the date column, which can be transformed into the datetime type. That will make our feature engineering efforts easier later on.

# ### init: 1 dataframe

# In[1]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(ufo)
tobedownloaded="{pandas.core.frame.DataFrame: {'ufo.csv': 'https://file.io/FBGUQ6'}}"
prefix='data_from_datacamp/Chap5-Exercise1.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[2]:


import pandas as pd
ufo=pd.read_csv(prefix+'ufo.csv',index_col=0)


# ### code

# - Print out the dtypes of the ufo dataset.
# - Change the type of the seconds column by passing the float type into the astype() method.
# - Change the type of the date column by passing ufo["date"] into the pd.to_datetime() function.
# - Print out the dtypes of the seconds and date columns, to make sure it worked.

# In[5]:


# Check the column types
ufo.info()

# Change the type of seconds to float
ufo["seconds"] = ufo['seconds'].astype('float')

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo['date'])

# Check the column types
print(ufo[['seconds', 'date']].dtypes)


# ## Dropping missing data
# Let's remove some of the rows where certain columns have missing values. We're going to look at the length_of_time column, the state column, and the type column. If any of the values in these columns are missing, we're going to drop the rows.

# ### code

# - Check how many values are missing in the length_of_time, state, and type columns, using isnull() to check for nulls and sum() to calculate how many exist.
# - Use boolean indexing to filter out the rows with those missing values, using notnull() to check the column. Here, we can chain together each column we want to check.
# - Print out the shape of the new ufo_no_missing dataset.

# In[7]:


ufo.isnull().sum()


# In[9]:


# Check how many values are missing in the length_of_time, state, and type columns
print(ufo[['length_of_time', 'state', 'type']].isnull().sum())

# Keep only rows where length_of_time, state, and type are not null
ufo_no_missing = ufo[ufo.length_of_time.notnull() & 
          ufo.state.notnull() & 
          ufo.type.notnull()]

# Print out the shape of the new dataset
print(ufo_no_missing.shape)


# # Categorical variables and standardization
# 

# ## Extracting numbers from strings
# The length_of_time field in the UFO dataset is a text field that has the number of minutes within the string. Here, you'll extract that number from that text field using regular expressions.

# - Pass \d+ into re.compile() in the pattern variable to designate that we want to grab as many digits as possible from the string.
# - Into re.match(), pass the pattern we just created, as well as the time_string we want to extract from.
# - Use lambda within the apply() method to perform the extraction.
# - Print out the head() of both the length_of_time and minutes columns to compare.

# ### init: 1 dataframe

# In[25]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(ufo)
tobedownloaded="{pandas.core.frame.DataFrame: {'ufo.csv': 'https://file.io/rinlM1'}}"
prefix='data_from_datacamp/Chap5-Exercise2.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[29]:


import pandas as pd
ufo=pd.read_csv(prefix+'ufo.csv',index_col=0,parse_dates=['date'])


# In[30]:


import re


# In[31]:


ufo.info()


# ### code

# In[32]:


def return_minutes(time_string):
    
    # We'll use \d+ to grab digits and match it to the column values
    pattern = re.compile(r"\d+")
        
    # Use match on the pattern and column
    num = re.match(pattern, time_string)
    if num is not None:
        return int(num.group(0))
        
# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(return_minutes)

# Take a look at the head of both of the columns
print(ufo[["length_of_time", "minutes"]].head())


# ## Identifying features for standardization
# In this section, you'll investigate the variance of columns in the UFO dataset to determine which features should be standardized. After taking a look at the variances of the seconds and minutes column, you'll see that the variance of the seconds column is extremely high. Because seconds and minutes are related to each other (an issue we'll deal with when we select features for modeling), let's log normlize the seconds column.

# In[33]:


import numpy as np


# In[34]:


# Check the variance of the seconds and minutes columns
print(ufo[['seconds','minutes']].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo['seconds'])

# Print out the variance of just the seconds_log column
print(ufo[['seconds_log']].var())


# # Engineering new features
# 

# ## Encoding categorical variables
# There are couple of columns in the UFO dataset that need to be encoded before they can be modeled through scikit-learn. You'll do that transformation here, using both binary and one-hot encoding methods.

# In[36]:


import pandas as pd


# In[43]:


# Use Pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda val: 1 if val == "us" else 0)

# Print the number of unique type values
print(len(ufo.type.unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo['type'])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)


# ## Features from dates
# Another feature engineering task to perform is month and year extraction. Perform this task on the date column of the ufo dataset.

# In[46]:


# Look at the first 5 rows of the date column
print(ufo[['date']].head())

# Extract the month from the date column
ufo["month"] = ufo["date"].apply(lambda val: val.month)

# Extract the year from the date column
ufo["year"] = ufo["date"].apply(lambda val: val.year)

# Take a look at the head of all three columns
print(ufo[['date', 'month', 'year']].head())


# ## Text vectorization
# Let's transform the desc column in the UFO dataset into tf/idf vectors, since there's likely something we can learn from this field.

# - Print out the head() of the ufo["desc"] column.
# - Set vec equal to the TfidfVectorizer() object.
# - Use vec's fit_transform() method on the ufo["desc"] column.
# - Print out the shape of the desc_tfidf vector, to take a look at the number of columns this created. The output is in the shape (rows, columns).

# In[47]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[48]:


# Take a look at the head of the desc field
print(ufo['desc'].head())

# Create the tfidf vectorizer object
vec = TfidfVectorizer()

# Use vec's fit_transform method on the desc field
desc_tfidf = vec.fit_transform(ufo['desc'])

# Look at the number of columns this creates
print(desc_tfidf.shape)


# ![image.png](attachment:image.png)

# # Feature selection and modeling
# 

# ## Selecting the ideal dataset
# Let's get rid of some of the unnecessary features. Because we have an encoded country column, country_enc, keep it and drop other columns related to location: city, country, lat, long, state.
# 
# We have columns related to month and year, so we don't need the date or recorded columns.
# 
# We vectorized desc, so we don't need it anymore. For now we'll keep type.
# 
# We'll keep seconds_log and drop seconds and minutes.
# 
# Let's also get rid of the length_of_time column, which is unnecessary after extracting minutes.

# ### init: 1 dataframe

# In[50]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(ufo)
tobedownloaded="{pandas.core.frame.DataFrame: {'ufo.csv': 'https://file.io/uMkfeT'}}"
prefix='data_from_datacamp/Chap5-Exercise4.1_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[51]:


import pandas as pd
ufo=pd.read_csv(prefix+'ufo.csv',index_col=0,parse_dates=['date'])


# In[60]:


# Add in the rest of the parameters
def return_weights(vocab, original_vocab, vector, vector_index,  top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    
    # Let's transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    
    # Let's sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]


# In[61]:


def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
    
        # Here we'll call the function from the previous exercise, and extend the list we're creating
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)


# In[62]:


vocab={1664: 'it', 3275: 'was', 1744: 'large', 147: '44', 3123: 'triangular', 2657: 'shaped', 1320: 'flying', 2134: 'object', 910: 'dancing', 1794: 'lights', 3002: 'that', 3379: 'would', 1319: 'fly', 395: 'around', 340: 'and', 3007: 'then', 1923: 'merge', 1645: 'into', 2173: 'one', 1787: 'light', 604: 'brilliant', 2188: 'orange', 2184: 'or', 718: 'chinese', 1738: 'lantern', 412: 'at', 1774: 'less', 3001: 'than', 15: '1000', 1363: 'ft', 2021: 'moving', 1102: 'east', 3050: 'to', 3298: 'west', 273: 'across', 2130: 'oakville', 2176: 'ontario', 1942: 'midnight', 1690: 'june', 251: '9th', 92: '2013', 596: 'bright', 2472: 'red', 2097: 'north', 1360: 'from', 3003: 'the', 1539: 'horizon', 3041: 'till', 1003: 'disapeared', 502: 'behind', 766: 'clouds', 2793: 'south', 1276: 'first', 2766: 'so', 873: 'craft', 1462: 'half', 1063: 'dozen', 2899: 'stragglers', 3015: 'they', 3296: 'were', 2943: 'surely', 2107: 'not', 2330: 'planes', 2094: 'nor', 449: 'ball', 2157: 'of', 2751: 'slowly', 2872: 'stationary', 2031: 'multicolored', 738: 'circular', 1926: 'met', 637: 'by', 351: 'another', 3315: 'which', 2435: 'raised', 1915: 'meet', 3057: 'too', 1025: 'displayed', 3153: 'ufo', 1421: 'going', 2737: 'sky', 3206: 'uso', 3280: 'watched', 3202: 'us', 1331: 'for', 864: 'couple', 1963: 'minutes', 1326: 'follows', 3085: 'train', 3077: 'tracks', 1607: 'in', 3350: 'winter', 46: '1931', 2586: 'saw', 1859: 'machine', 401: 'as', 2524: 'riding', 2171: 'on', 1543: 'horse', 1065: 'draw', 2265: 'pasture', 13: '10', 2343: 'pm', 23: '12', 249: '99', 2602: 'scottsdale', 386: 'arizona', 2778: 'something', 1482: 'have', 2067: 'never', 2623: 'seen', 498: 'before', 3097: 'traveling', 259: 'above', 1414: 'glow', 691: 'central', 1987: 'montana', 1928: 'metalic', 2822: 'sphere', 2551: 'rotating', 1240: 'fast', 2271: 'pattern', 3287: 'we', 1076: 'driving', 3069: 'town', 3011: 'there', 1656: 'is', 1424: 'golf', 865: 'course', 1530: 'holes', 760: 'close', 2535: 'road', 3312: 'when', 313: 'all', 2954: 'suuden', 2262: 'passing', 3137: 'turned', 2158: 'off', 474: 'bayou', 558: 'blvd', 2177: 'onto', 1420: 'godwinson', 337: 'an', 1209: 'extremely', 1061: 'down', 3355: 'with', 3081: 'trail', 2757: 'smok', 1337: 'formation', 167: '44counted', 31: '15', 2198: 'orbs', 3190: 'until', 3219: 'vanished', 628: 'bursts', 782: 'color', 811: 'concentrated', 384: 'area', 764: 'cloud', 486: 'beautiful', 2708: 'silver', 784: 'colored', 2583: 'saucer', 258: 'about', 2729: 'size', 2553: 'round', 42: '18', 3311: 'wheeler', 3141: 'turquoise', 2293: 'perimeter', 750: 'clearly', 125: '3997', 8: '05', 225: '64', 4: '02', 0: '00', 1808: 'little', 881: 'creek', 1723: 'ky', 1572: 'humming', 2085: 'noise', 1549: 'house', 2654: 'shaking', 2754: 'small', 1914: 'medium', 2730: 'sized', 536: 'black', 2471: 'rectangular', 2647: 'several', 3390: 'years', 3104: 'tree', 1800: 'line', 2622: 'seemingly', 627: 'burning', 3194: 'up', 2966: 'tail', 1032: 'dissipated', 3067: 'towards', 1141: 'end', 2698: 'sight', 524: 'big', 1966: 'miss', 187: '44orange', 2475: 'redish', 2136: 'objects', 1312: 'floating', 1533: 'home', 2643: 'sets', 3147: 'two', 2940: 'sunset', 2485: 'remained', 3316: 'while', 2211: 'other', 2013: 'moved', 2800: 'southwest', 2875: 'stayed', 3126: 'tripled', 2725: 'sitting', 671: 'car', 1831: 'looking', 2715: 'singal', 1147: 'engine', 2329: 'plane', 2932: 'suddenly', 1014: 'disc', 3295: 'went', 2263: 'past', 3240: 'very', 3319: 'white', 703: 'chased', 219: '52', 2054: 'near', 3380: 'wright', 2274: 'patterson', 291: 'air', 1332: 'force', 466: 'base', 2373: 'power', 2218: 'out', 652: 'came', 2620: 'seemed', 733: 'circle', 793: 'come', 442: 'back', 2421: 'quickly', 1695: 'just', 3218: 'vanishe', 3022: 'thirteen', 2038: 'my', 1437: 'grandmother', 2000: 'mother', 1483: 'having', 1593: 'ice', 880: 'cream', 2224: 'outside', 990: 'diner', 205: '45', 214: '50', 947: 'degrees', 1418: 'glows', 699: 'changing', 979: 'different', 289: 'again', 3393: 'yellow', 1556: 'hovering', 962: 'description', 2367: 'possible', 2700: 'sighting', 2886: 'still', 2377: 'present', 2857: 'star', 1796: 'like', 3052: 'together', 2571: 'same', 2816: 'speed', 987: 'dimmed', 1007: 'disappeared', 1289: 'flare', 713: 'chevron', 2705: 'silent', 2229: 'over', 108: '30', 1954: 'min', 682: 'caught', 3246: 'video', 2980: 'tape', 1270: 'fireball', 286: 'after', 1252: 'few', 1445: 'greenish', 970: 'diamond', 1279: 'five', 2755: 'smaller', 929: 'daytime', 2076: 'night', 964: 'desert', 1555: 'hovered', 2814: 'sped', 438: 'away', 1969: 'mississauga', 658: 'canada', 368: 'appeared', 477: 'be', 1517: 'high', 2368: 'possibly', 357: 'any', 1098: 'earthling', 1861: 'made', 1665: 'item', 1612: 'incredible', 2900: 'straight', 2084: 'no', 2788: 'sound', 1297: 'flat', 549: 'blinking', 1729: 'lake', 2210: 'oswego', 2425: 'quot', 1365: 'full', 2685: 'shortly', 21: '11', 35: '16', 2936: 'sunday', 943: 'defied', 282: 'aerodynamiccs', 1719: 'know', 554: 'blue', 2410: 'purple', 1219: 'faded', 909: 'danced', 885: 'cricle', 2529: 'rise', 1011: 'disapper', 2512: 'retiring', 1430: 'got', 494: 'bed', 2544: 'room', 911: 'dark', 3337: 'window', 2744: 'slightly', 2179: 'open', 1489: 'head', 1770: 'left', 1378: 'gateway', 298: 'airport', 1924: 'merged', 352: 'anouther', 489: 'became', 3040: 'tight', 197: '44straight', 374: 'approx', 124: '39', 2656: 'shape', 1630: 'intense', 1092: 'each', 770: 'cntr', 191: '44rest', 1324: 'followed', 2144: 'observer', 2889: 'stoppe', 616: 'brownwood', 2999: 'texas', 117: '33', 1900: 'mass', 2832: 'sporadically', 1611: 'inconsistent', 2791: 'sounds', 3284: 'waves', 106: '29', 151: '442008', 242: '8pm', 181: '44my', 3402: 'yr', 2168: 'old', 1398: 'girls', 407: 'asked', 3308: 'what', 1598: 'if', 2679: 'shooting', 3053: 'told', 3005: 'them', 2018: 'moves', 504: 'being', 1020: 'disk', 1254: 'field', 232: '70', 133: '39s', 2258: 'pass', 2335: 'play', 702: 'chase', 366: 'appear', 324: 'altitude', 1088: 'during', 749: 'clear', 926: 'day', 179: '44make', 1005: 'disappear', 1465: 'hammond', 1600: 'illinois', 2960: 'sylvania', 1907: 'mccord', 1691: 'junior', 2219: 'outer', 2803: 'space', 78: '20', 1829: 'looked', 2862: 'stars', 1292: 'flash', 599: 'brightlights', 336: 'amp', 3292: 'weird', 2088: 'noisies', 2902: 'strange', 1476: 'happennings', 3268: 'walking', 1817: 'local', 457: 'bar', 1896: 'mars', 490: 'because', 657: 'can', 3293: 'well', 2525: 'right', 2117: 'now', 2403: 'pulsating', 520: 'bethel', 2250: 'park', 2236: 'pa', 721: 'christmas', 1174: 'eve', 2572: 'san', 356: 'antonio', 653: 'camera', 945: 'definitely', 857: 'could', 1597: 'idetify', 3023: 'this', 2200: 'ordinary', 1306: 'flies', 2706: 'silently', 1176: 'evening', 2340: 'please', 2506: 'respect', 2386: 'privacy', 697: 'changed', 998: 'directions', 1196: 'experience', 3184: 'unkown', 3230: 'vegas', 856: 'couch', 2604: 'screen', 1053: 'door', 1475: 'happened', 2617: 'see', 296: 'airplane', 632: 'but', 762: 'closer', 2194: 'orb', 463: 'barnes', 2446: 'rd', 3269: 'wallingford', 2225: 'oval', 555: 'blueish', 3412: 'zig', 3409: 'zaging', 1028: 'dissapears', 2898: 'strage', 2163: 'ohio', 2533: 'river', 2746: 'slow', 2813: 'spectacular', 3117: 'triangle', 2619: 'seem', 2605: 'se', 973: 'did', 2012: 'move', 1822: 'long', 1105: 'eastern', 2439: 'random', 2405: 'pulse', 292: 'aircraft', 2381: 'pressure', 3283: 'wave', 290: 'ahead', 1444: 'green', 2922: 'strobing', 1086: 'duration', 2120: 'nuforc', 2108: 'note', 2717: 'sirius', 2277: 'pd', 508: 'bell', 1026: 'dissapear', 2458: 'reappear', 1985: 'monmouth', 863: 'county', 2068: 'new', 1675: 'jersey', 1886: 'maple', 1455: 'grove', 364: 'apparent', 1273: 'firey', 574: 'bottom', 2305: 'photos', 670: 'captured', 1080: 'drops', 1751: 'late', 287: 'afternoon', 2436: 'raleigh', 2052: 'nc', 377: 'april', 243: '8th', 2216: 'our', 1838: 'lost', 1996: 'morphed', 1008: 'disappearing', 2836: 'spotted', 790: 'columbus', 2024: 'ms', 1948: 'military', 293: 'aircrafts', 369: 'appearing', 1646: 'investigate', 1447: 'grey', 1354: 'friend', 1239: 'fashion', 2697: 'sideways', 994: 'direct', 1366: 'future', 445: 'backyard', 2767: 'soaring', 1295: 'flashing', 1165: 'erratically', 1846: 'low', 98: '22', 1494: 'headlights', 690: 'centered', 101: '25', 25: '13', 140: '40', 2844: 'squared', 3241: 'vessel', 1860: 'macon', 1370: 'ga', 2244: 'paper', 1685: 'journal', 2867: 'stated', 1989: 'month', 2578: 'satalites', 3256: 'visible', 137: '39winking', 3154: 'ufos', 700: 'charleston', 2592: 'sc', 2909: 'streaming', 2823: 'spheres', 955: 'des', 1978: 'moines', 1653: 'iowa', 453: 'balls', 1338: 'formations', 1677: 'jets', 3066: 'toward', 1702: 'kennedy', 689: 'center', 3249: 'viewed', 775: 'cocoa', 1281: 'fl', 1417: 'glowing', 476: 'bday', 2256: 'party', 1125: 'else', 328: 'am', 2498: 'reporting', 1457: 'gulf', 406: 'ashtabula', 2719: 'sister', 612: 'brother', 1757: 'law', 1478: 'hard', 1198: 'explain', 3031: 'three', 2928: 'suburb', 1269: 'fire', 510: 'beloit', 3324: 'wi', 570: 'border', 2278: 'pea', 2344: 'pod', 1345: 'four', 1790: 'lighted', 2281: 'peas', 3059: 'top', 3360: 'witnessed', 3044: 'times', 3110: 'tremendeous', 1164: 'erratic', 2014: 'moveme', 3095: 'travel', 2100: 'northern', 1314: 'florida', 2408: 'pulsing', 2686: 'shot', 2538: 'rocket', 3185: 'unlike', 1936: 'mi', 2789: 'soundless', 2727: 'six', 3025: 'those', 467: 'baseball', 2920: 'strobe', 1768: 'lebanon', 893: 'ct', 1586: 'i4', 1192: 'exit', 29: '14', 2824: 'spherical', 1463: 'halo', 380: 'arc', 2231: 'overhead', 2487: 'remember', 2240: 'pale', 743: 'city', 1087: 'durham', 1325: 'following', 996: 'direction', 1806: 'lite', 3034: 'through', 2143: 'observed', 593: 'brie', 1232: 'far', 1413: 'glod', 2111: 'nothig', 3398: 'you', 2027: 'much', 2074: 'nice', 1922: 'mercey', 1545: 'hot', 2840: 'springs', 2827: 'spinning', 2773: 'solid', 2674: 'shinny', 3092: 'transparent', 2833: 'spot', 1492: 'heading', 1566: 'huge', 2189: 'orangeish', 1340: 'forming', 898: 'curved', 1460: 'had', 2781: 'somewhere', 688: 'cen', 36: '160', 1993: 'more', 3150: 'type', 1339: 'formed', 2615: 'seconds', 3286: 'way', 1791: 'lighthouse', 1043: 'does', 729: 'cigar', 1910: 'me', 1120: 'electric', 1616: 'indian', 673: 'carolina', 2260: 'passenger', 2609: 'seat', 844: 'conversion', 3214: 'van', 1981: 'mom', 2694: 'siblings', 408: 'asleep', 1134: 'emitted', 927: 'daylight', 2566: 'rush', 1547: 'hour', 1441: 'gray', 439: 'awesome', 3396: 'yet', 882: 'creepy', 2831: 'spooky', 2669: 'shined', 2671: 'shiniest', 722: 'chrome', 1182: 'ever', 805: 'completely', 2422: 'quiet', 2699: 'sighted', 1746: 'las', 2496: 'report', 1179: 'event', 3294: 'wells', 198: '44texas', 3358: 'without', 109: '300', 141: '400', 1245: 'feet', 2299: 'phoenix', 441: 'az', 2186: 'orage', 1405: 'glenville', 3384: 'wv', 1557: 'hovers', 1552: 'houston', 3148: 'tx', 1227: 'fall', 60: '1973', 2834: 'spotlight', 1971: 'mist', 3199: 'upward', 2002: 'motion', 1967: 'missile', 1754: 'launch', 200: '44then', 2187: 'orang', 3281: 'watching', 2010: 'mountain', 1231: 'fanwood', 2081: 'nj', 2638: 'series', 2801: 'southwestern', 2985: 'teardrop', 3282: 'water', 1805: 'lit', 1301: 'flew', 573: 'bothell', 3261: 'wa', 1929: 'metallic', 1042: 'dodge', 2911: 'street', 1206: 'exprsway', 2170: 'omaha', 2053: 'ne', 1487: 'hbccufo', 659: 'canadian', 974: 'didn', 134: '39t', 2946: 'surrounded', 2064: 'neon', 1553: 'hover', 914: 'dart', 1902: 'massive', 785: 'colorful', 1624: 'instantly', 1520: 'highway', 112: '31', 1617: 'indiana', 957: 'descend', 3106: 'trees', 1521: 'hill', 999: 'directly', 3236: 'vernon', 1415: 'glowball', 1361: 'front', 496: 'bedroom', 1247: 'felt', 633: 'butterfly', 2628: 'sensation', 712: 'chest', 3000: 'th', 901: 'cylinder', 414: 'atlantic', 2152: 'ocean', 774: 'coastal', 1529: 'hobe', 1937: 'miami', 310: 'alien', 2339: 'playing', 1343: 'forth', 1117: 'egg', 5: '03', 1724: 'l7', 1548: 'hours', 3221: 'vanishing', 1767: 'leaving', 560: 'bobbing', 3183: 'unknown', 3144: 'twinkling', 849: 'corners', 3168: 'underneath', 393: 'arou', 3017: 'thin', 2321: 'pinkish', 2528: 'rings', 1069: 'drifting', 2026: 'mt', 3210: 'va', 2371: 'potomac', 1096: 'early', 50: '1963', 51: '1964', 2490: 'reno', 2913: 'streets', 3130: 'trying', 1265: 'find', 1546: 'hotel', 2114: 'noticed', 2312: 'pie', 1472: 'hanging', 739: 'circus', 2251: 'parki', 57: '197', 1490: 'headed', 3373: 'work', 1538: 'horizion', 2309: 'pics', 938: 'decide', 3400: 'yourself', 564: 'bolingbrook', 1904: 'may', 41: '17th', 81: '2001', 1912: 'meadows', 1771: 'length', 1303: 'flickered', 2494: 'replace', 2923: 'strong', 2238: 'pacific', 1799: 'lincoln', 2606: 'sea', 1459: 'gypsy', 478: 'beach', 796: 'coming', 3100: 'travels', 1962: 'minute', 404: 'ascends', 1408: 'gliding', 1877: 'manhattan', 991: 'dinner', 2775: 'some', 858: 'couldn', 503: 'beileve', 1836: 'los', 342: 'angeles', 1888: 'march', 83: '2004', 1241: 'faster', 359: 'anything', 2029: 'multi', 787: 'coloured', 541: 'blasts', 2667: 'shifting', 2957: 'swaying', 93: '2055', 1561: 'hrs', 1114: 'edt', 2716: 'single', 3111: 'tremendously', 1203: 'explodes', 276: 'activity', 1700: 'keller', 2580: 'satellites', 410: 'associated', 708: 'chemtrails', 3096: 'traveled', 3192: 'unusual', 1880: 'manner', 2345: 'point', 3410: 'zags', 1834: 'loops', 1959: 'minnesotas', 3371: 'woods', 2056: 'nearly', 1157: 'equilateral', 848: 'corner', 3063: 'total', 2707: 'sillouette', 2693: 'si', 1601: 'illuminated', 1021: 'disks', 2011: 'mountains', 1284: 'flame', 2098: 'northeast', 1819: 'location', 1385: 'georgia', 1208: 'extreme', 1304: 'flickering', 3132: 'tubular', 950: 'delta', 572: 'both', 934: 'debris', 2701: 'sightings', 1714: 'kingstown', 2521: 'ri', 2155: 'odd', 1778: 'lewisville', 2563: 'running', 1760: 'leader', 2565: 'rural', 2997: 'tests', 2771: 'soil', 1364: 'fuel', 686: 'cell', 3222: 'vapor', 1047: 'dogs', 461: 'barking', 1842: 'loudly', 1287: 'flanked', 2174: 'ones', 2721: 'sited', 2314: 'pier', 2276: 'pawleys', 1659: 'island', 491: 'become', 420: 'auburn', 1244: 'federal', 1851: 'luminous', 2325: 'pittsburgh', 3389: 'year', 1036: 'distinct', 2045: 'naked', 1210: 'eye', 3405: 'yuma', 2015: 'movement', 3347: 'winn', 1040: 'dixie', 2252: 'parking', 1839: 'lot', 1825: 'longwood', 3317: 'whippany', 1998: 'morris', 1747: 'last', 2760: 'smoking', 731: 'cigarette', 327: 'always', 1041: 'do', 2493: 'repeated', 1140: 'encounter', 65: '1980', 483: 'beardstown', 818: 'confirmed', 2049: 'nature', 3079: 'traditional', 1336: 'format', 2359: 'port', 645: 'california', 3229: 'veers', 3220: 'vanishes', 1885: 'many', 1056: 'dots', 2413: 'put', 2222: 'outlined', 3099: 'travelling', 2675: 'shiny', 1006: 'disappeards', 982: 'dim', 322: 'also', 888: 'crossed', 1442: 'great', 1034: 'distances', 2683: 'short', 335: 'amount', 3043: 'time', 2459: 'reappeare', 735: 'circles', 2829: 'splitting', 2722: 'siting', 3215: 'vancouver', 1485: 'haze', 248: '96', 405: 'ashland', 2202: 'oregon', 72: '1996', 1797: 'lima', 1927: 'metal', 862: 'country', 2866: 'state', 332: 'amber', 1765: 'least', 224: '6000', 233: '7000', 262: 'absolutely', 1280: 'fixed', 2044: 'nailed', 2326: 'place', 475: 'bb', 1504: 'held', 391: 'arms', 110: '30am', 3027: 'thought', 499: 'began', 530: 'binoculars', 1578: 'husband', 497: 'been', 2618: 'seeing', 3302: 'western', 231: '6pm', 1183: 'every', 1580: 'hw', 1814: 'lo', 2507: 'rest', 1293: 'flashed', 2516: 'revealed', 121: '35ish', 3328: 'wife', 2681: 'shopping', 446: 'bag', 741: 'ciruclar', 1079: 'dropping', 1291: 'flares', 1423: 'golden', 511: 'below', 3386: 'wyoming', 1257: 'figeting', 2509: 'resturant', 647: 'called', 227: '66', 14: '100', 2695: 'side', 1033: 'distance', 1872: 'making', 2758: 'smoke', 2890: 'stopped', 3264: 'waiting', 74: '1999', 875: 'crafts', 3116: 'triange', 2629: 'separate', 3357: 'within', 2994: 'ten', 562: 'body', 2786: 'sort', 317: 'almost', 581: 'bouy', 2175: 'only', 1199: 'explainable', 930: 'dayton', 763: 'closing', 2443: 'rapidly', 797: 'commercial', 3054: 'tomah', 3351: 'wisconsin', 2769: 'softly', 969: 'diameter', 3102: 'traversed', 154: '44800', 965: 'detectable', 1832: 'looks', 814: 'condor', 887: 'cross', 532: 'bird', 1077: 'drone', 1094: 'eagle', 2415: 'qu', 2828: 'split', 1258: 'fighter', 1676: 'jet', 1636: 'intercepted', 2614: 'second', 3062: 'toronto', 2885: 'steps', 361: 'apartment', 3103: 'travling', 3414: 'zigzagging', 2182: 'opposite', 600: 'brightly', 1639: 'intermittently', 2091: 'non', 2955: 'sw', 952: 'denver', 2738: 'skyline', 2518: 'rex', 859: 'couldnt', 506: 'believe', 3018: 'thing', 786: 'colors', 171: '44gold', 1330: 'football', 1426: 'gondola', 2925: 'structure', 3080: 'traffic', 822: 'congestion', 447: 'bakersfield', 639: 'ca', 1671: 'january', 67: '1983', 1425: 'golfing', 3291: 'weekend', 2040: 'myrtle', 422: 'august', 85: '2006', 3354: 'witc', 2390: 'pronged', 223: '60', 1777: 'level', 2577: 'sat', 834: 'contemplated', 1558: 'how', 2600: 'school', 1752: 'later', 1256: 'fiery', 1166: 'erraticly', 2028: 'mufon', 783: 'colorado', 3361: 'witnesses', 634: 'buzz', 38: '17', 3093: 'transport', 3211: 'vail', 815: 'cone', 2917: 'strip', 1913: 'medford', 2372: 'potterville', 236: '7pm', 96: '21', 1816: 'lobed', 3301: 'westerly', 1761: 'leading', 2963: 'tacoma', 1660: 'isle', 715: 'child', 1560: 'however', 1516: 'hig', 2468: 'rectangle', 2003: 'motionless', 1394: 'gilbert', 1309: 'flipping', 1588: 'i5freeway', 1218: 'fade', 2455: 'reapear', 2327: 'places', 1905: 'maybe', 103: '26', 868: 'cousins', 448: 'balcony', 479: 'beachfront', 435: 'avon', 2797: 'southern', 1362: 'fruitville', 2575: 'sarasota', 1097: 'earth', 256: 'abnormal', 1673: 'jefferson', 2587: 'say', 1513: 'here', 3395: 'yes', 383: 'are', 606: 'bristol', 3049: 'tn', 1263: 'film', 3322: 'who', 1721: 'knows', 1574: 'hundreds', 2285: 'people', 825: 'considered', 427: 'authentic', 1480: 'has', 966: 'determined', 1527: 'hoax', 3260: 'visual', 3248: 'view', 1947: 'miles', 2515: 'returned', 3345: 'wingless', 3131: 'tube', 2290: 'perfectly', 1311: 'floated', 2703: 'signal', 563: 'boise', 1594: 'id', 2642: 'set', 1818: 'located', 654: 'campbells', 1074: 'drive', 2670: 'shiney', 1737: 'lane', 2096: 'normal', 1541: 'horizontal', 3042: 'tilting', 1735: 'landed', 1925: 'mesa', 3326: 'width', 1090: 'dusk', 1178: 'evenly', 323: 'alternated', 522: 'between', 1650: 'invisible', 3098: 'travelled', 3084: 'trails', 3120: 'triangles', 184: '44objects', 915: 'darted', 3406: 'zag', 568: 'boomerang', 1294: 'flashes', 2696: 'sides', 480: 'beam', 2672: 'shining', 3065: 'touching', 1450: 'ground', 1655: 'irregular', 1994: 'morning', 2673: 'shinning', 3353: 'wispy', 2197: 'orbiting', 607: 'broad', 3359: 'witness', 1017: 'discovers', 1893: 'marks', 904: 'cylindrical', 2474: 'reddish', 827: 'consistent', 443: 'background', 2731: 'sk', 622: 'bunch', 737: 'circlular', 960: 'descends', 3161: 'uncertain', 3314: 'whether', 2639: 'serious', 2318: 'pinal', 165: '44circular', 190: '44red', 1221: 'fading', 174: '44huge', 195: '44silent', 1961: 'mintues', 2853: 'stalled', 297: 'airplanes', 2658: 'shapes', 876: 'cranston', 2462: 'reappears', 976: 'diff', 20: '10th', 76: '1pm', 1419: 'go', 2935: 'sun', 3335: 'wilsonville', 176: '44il', 64: '1979', 1720: 'known', 1667: 'its', 2430: 'radioactive', 3278: 'waste', 1084: 'dump', 2646: 'seventies', 2676: 'ship', 2557: 'row', 745: 'classic', 1050: 'dome', 129: '39dancing', 3234: 'ventura', 2541: 'role', 2161: 'offshore', 2033: 'multiple', 980: 'differing', 1644: 'intervention', 2907: 'streaking', 1271: 'fireballs', 732: 'cincinnati', 2162: 'oh', 773: 'coast', 846: 'copper', 971: 'diamonds', 3299: 'westbound', 3149: 'tyler', 1138: 'en', 693: 'cette', 345: 'ann', 1093: 'eacute', 1672: 'je', 951: 'demeur', 430: 'avec', 1983: 'mon', 845: 'copain', 2398: 'puis', 126: '39avais', 3072: 'trac', 3158: 'un', 692: 'cercle', 933: 'de', 2279: 'peace', 1845: 'love', 1686: 'jours', 428: 'avant', 878: 'crayons', 1251: 'feutre', 1696: 'juste', 2245: 'par', 3014: 'these', 1537: 'hoovering', 1242: 'father', 2782: 'son', 611: 'brookyln', 2125: 'ny', 58: '1970', 590: 'breaks', 3401: 'ypsilanti', 1991: 'moon', 977: 'differen', 2799: 'southwards', 2545: 'rose', 2720: 'site', 1881: 'manuever', 2209: 'orlando', 698: 'changes', 2360: 'portal', 379: 'aptos', 2289: 'perfect', 1661: 'isosceles', 804: 'complete', 3180: 'unison', 2124: 'nw', 3138: 'turning', 2684: 'shorted', 769: 'cluster', 1637: 'interchanging', 362: 'apeared', 3037: 'thrust', 1031: 'dissappeared', 2460: 'reappeared', 809: 'con', 2971: 'taking', 1044: 'dog', 2855: 'standing', 940: 'deck', 1215: 'faces', 1082: 'due', 1525: 'his', 937: 'decently', 2492: 'repeat', 2688: 'show', 80: '2000', 756: 'cloaked', 808: 'components', 1640: 'international', 2645: 'seven', 3019: 'things', 1135: 'emitting', 1841: 'loud', 2690: 'shrieking', 2192: 'orangey', 2320: 'pink', 1622: 'inside', 119: '33right', 2295: 'persistent', 381: 'arced', 556: 'bluish', 1651: 'involved', 2904: 'strangley', 53: '1966', 239: '8211', 537: 'blackfoot', 3213: 'valley', 2227: 'ovando', 1237: 'farmland', 3399: 'young', 3365: 'woman', 3266: 'walk', 1933: 'metro', 2267: 'path', 484: 'bearing', 319: 'along', 1802: 'lines', 1175: 'even', 3368: 'woode', 372: 'approached', 1865: 'main', 2180: 'opened', 2338: 'played', 2206: 'original', 2310: 'picture', 3378: 'worth', 3372: 'words', 148: '44000', 3032: 'threw', 2540: 'rocks', 2234: 'ovoid', 3394: 'yellowish', 2037: 'mutiney', 473: 'bay', 1573: 'hundred', 1322: 'folks', 685: 'celebrating', 213: '4th', 1688: 'july', 3177: 'union', 906: 'dad', 1716: 'kite', 2484: 'relized', 547: 'blinked', 2636: 'sequence', 193: '44saw', 2916: 'string', 2199: 'order', 1807: 'litghts', 2843: 'square', 1710: 'kiltered', 1318: 'fluttering', 373: 'approaching', 2795: 'southeast', 2480: 'region', 1236: 'farmington', 544: 'blimp', 3276: 'washington', 2610: 'seattle', 1863: 'madrona', 1454: 'groups', 73: '1998', 3273: 'wanted', 2379: 'press', 2119: 'nuclear', 2448: 'reactor', 1507: 'helicopters', 2870: 'static', 2440: 'randomly', 94: '20min', 2172: 'once', 1081: 'drove', 440: 'awoken', 669: 'capsule', 1734: 'land', 2968: 'take', 2508: 'restaurant', 2093: 'noon', 836: 'continued', 2451: 'ready', 2399: 'pull', 1075: 'driveway', 1428: 'good', 1035: 'distant', 1118: 'eight', 1643: 'intervals', 2034: 'murfreesboro', 450: 'ballls', 2733: 'skies', 1559: 'howad', 3217: 'vanish', 867: 'cousin', 1512: 'her', 2976: 'tall', 2319: 'pine', 748: 'cleaning', 1277: 'fish', 2411: 'purpule', 1687: 'ju', 1222: 'faint', 2444: 'rate', 2625: 'seethrough', 1124: 'elongated', 2284: 'pentagon', 2351: 'poles', 2947: 'surrounding', 1972: 'misterious', 1532: 'hollywood', 91: '2012', 2929: 'suburban', 1957: 'minneapolis', 250: '9pm', 2441: 'range', 2714: 'since', 460: 'barely', 928: 'days', 2450: 'reading', 1375: 'garden', 3387: 'yard', 802: 'compared', 253: 'abilene', 1267: 'finished', 2500: 'rerun', 1390: 'ghost', 1575: 'hunters', 1742: 'laptop', 3207: 'usual', 1429: 'google', 894: 'cube', 2888: 'stop', 1078: 'dropped', 1376: 'gas', 2871: 'station', 953: 'departed', 2191: 'oranges', 1184: 'exact', 360: 'apart', 1931: 'meteors', 2447: 're', 1152: 'entering', 3174: 'unexplained', 2336: 'playa', 948: 'del', 2519: 'rey', 2952: 'suspicious', 1486: 'hazy', 1939: 'michigan', 546: 'blink', 1427: 'gone', 3157: 'umo', 70: '1988', 2153: 'oct', 1260: 'figures', 621: 'buildings', 2526: 'rigid', 281: 'aerial', 272: 'acrobatics', 2743: 'slight', 2953: 'sutle', 3167: 'under', 2819: 'spencerport', 3385: 'wx', 684: 'cavu', 648: 'calm', 139: '3rd', 626: 'burned', 1307: 'flight', 958: 'descended', 2873: 'stationery', 3026: 'though', 3235: 'venus', 3421: 'zooming', 635: 'buzzing', 2086: 'noiseless', 344: 'angles', 424: 'aura', 2878: 'steady', 1409: 'glimmering', 861: 'counties', 1941: 'middle', 2397: 'puget', 2102: 'northward', 2783: 'sonora', 1916: 'meets', 2212: 'others', 1010: 'disappears', 390: 'armada', 454: 'baltimore', 514: 'beltway', 2148: 'observing', 2881: 'steelers', 1372: 'game', 2286: 'peoria', 2055: 'nearby', 2363: 'ports', 2103: 'northwest', 87: '2008', 355: 'anthony', 1374: 'gap', 1119: 'el', 2257: 'paso', 2: '00pm', 1980: 'moline', 1228: 'falling', 1099: 'earths', 417: 'atomsphere', 1736: 'landing', 1970: 'missouri', 3257: 'vision', 2601: 'scope', 2558: 'rows', 525: 'bigger', 598: 'brighter', 2633: 'seperate', 1569: 'hum', 696: 'change', 2860: 'starlike', 1067: 'dribbled', 2915: 'stright', 852: 'corona', 3152: 'uah', 1576: 'huntsville', 301: 'al', 2023: 'mph', 2072: 'next', 667: 'cape', 777: 'cod', 1341: 'forms', 2118: 'nowhere', 2776: 'somehow', 2942: 'sure', 3313: 'where', 3115: 'triang', 2748: 'slowely', 917: 'darts', 2637: 'sequentially', 382: 'arched', 3411: 'zenith', 2110: 'notheastern', 1965: 'mirror', 905: 'cylndrical', 1062: 'downtown', 2794: 'southbound', 925: 'davie', 131: '39nt', 1887: 'marble', 3251: 'viewpoint', 2348: 'points', 1491: 'heades', 2846: 'sse', 3107: 'treetop', 2488: 'remote', 681: 'catskill', 2661: 'sharp', 343: 'angle', 456: 'banos', 1581: 'hwy', 104: '27', 34: '15am', 2423: 'quietly', 398: 'arrow', 526: 'biggest', 1142: 'ended', 1698: 'keep', 936: 'decended', 402: 'ascended', 1335: 'form', 1019: 'dishpan', 1400: 'glacier', 2280: 'peak', 2483: 'releasing', 3047: 'tiny', 1811: 'littleton', 2895: 'storms', 3223: 'various', 821: 'conformations', 3306: 'westward', 1137: 'empty', 2842: 'spurts', 694: 'chain', 1155: 'equally', 2805: 'spaced', 830: 'constant', 676: 'cascades', 1745: 'larger', 1982: 'moment', 2133: 'obj', 1662: 'iss', 2331: 'planet', 2547: 'roseville', 211: '48066', 210: '48', 1169: 'est', 2926: 'student', 1054: 'doors', 392: 'arnold', 1975: 'mo', 1706: 'kettleman', 772: 'coalinga', 2596: 'scarry', 2933: 'summer', 2660: 'shapped', 2543: 'rooftops', 1506: 'helicopter', 2342: 'plus', 1467: 'hampton', 99: '23', 3058: 'took', 3420: 'zoomed', 1510: 'help', 1641: 'interrupted', 7: '04', 3254: 'visable', 1159: 'eratic', 501: 'behavior', 1753: 'lauderdale', 1170: 'estimated', 1579: 'huvering', 177: '44jerked', 199: '44than', 3021: 'third', 2438: 'rancho', 895: 'cucamonga', 2138: 'oblong', 2463: 'rear', 209: '460', 720: 'christiansburg', 539: 'blacksburg', 2287: 'peppers', 1250: 'ferry', 583: 'box', 997: 'directional', 1106: 'eastery', 2077: 'nights', 3105: 'treeline', 2608: 'season', 62: '1975', 668: 'capitan', 215: '500', 1946: 'mile', 1642: 'interstate', 981: 'digital', 649: 'camara', 3205: 'use', 2593: 'scan', 2739: 'skys', 1990: 'months', 2112: 'nothing', 1657: 'isaw', 2337: 'playback', 1150: 'enlarging', 1435: 'graham', 674: 'caroline', 235: '75', 620: 'building', 631: 'busy', 130: '39m', 3381: 'writing', 120: '35', 1960: 'mins', 370: 'appears', 1386: 'get', 2581: 'satillite', 1296: 'flashlight', 1682: 'joined', 752: 'clifton', 2083: 'nne', 2761: 'smoky', 1974: 'mnts', 1456: 'guelph', 138: '3bright', 3122: 'trianglular', 2847: 'st', 680: 'catharines', 1027: 'dissapeared', 3036: 'thru', 2080: 'nite', 2993: 'temecula', 164: '44ca', 2183: 'ops', 972: 'dice', 740: 'cirlcle', 758: 'clockface', 1988: 'montauk', 2835: 'spotlighting', 2664: 'shed', 158: '44at', 851: 'cornwall', 2548: 'rotate', 1161: 'erractic', 3225: 'varying', 2818: 'speeds', 2009: 'mount', 1758: 'lbert', 1055: 'dot', 3242: 'vey', 1514: 'hi', 1212: 'faa', 3208: 'usually', 2763: 'smoothly', 2051: 'navigation', 1857: 'ma', 3330: 'will', 3140: 'turns', 270: 'accross', 2635: 'september', 90: '2011', 1317: 'fluorescent', 3253: 'virginia', 932: 'dc', 2691: 'shut', 1246: 'fell', 1329: 'footage', 2568: 'sacramento', 2884: 'stepped', 1681: 'job', 589: 'break', 3267: 'walked', 3129: 'trunk', 1216: 'facing', 2975: 'talking', 3407: 'zagged', 2564: 'runway', 1197: 'experienced', 122: '360', 2531: 'rising', 1323: 'follow', 1201: 'explanations', 2937: 'sunny', 602: 'brihgte', 1712: 'kinda', 2467: 'recktangular', 2864: 'started', 2427: 'rad', 1143: 'ends', 266: 'accelerates', 423: 'augusta', 2810: 'sparkling', 839: 'contrai', 3064: 'totally', 3179: 'unique', 2383: 'previously', 114: '31st', 220: '55', 1113: 'edina', 3203: 'usa', 2868: 'staten', 183: '44ny', 2598: 'scattered', 1352: 'fresno', 1592: 'i95', 2522: 'richmond', 2298: 'phillipsburg', 1107: 'easton', 2453: 'realized', 268: 'accompanied', 2941: 'super', 1801: 'linear', 3305: 'westside', 2207: 'originally', 2735: 'skipped', 791: 'columns', 172: '44green', 153: '447', 246: '95', 162: '44boone', 2196: 'orbital', 2190: 'orangelights', 1422: 'gold', 2996: 'tessellated', 2520: 'rhomboidal', 2243: 'panes', 1288: 'flap', 2412: 'pursuit', 3228: 'vector', 3173: 'unexplainably', 3303: 'westminster', 1908: 'md', 1875: 'man', 436: 'awakened', 2380: 'pressed', 3004: 'their', 1618: 'indicator', 2313: 'pieces', 1692: 'junk', 2477: 'reentering', 2924: 'strs', 843: 'converging', 3198: 'upwad', 493: 'becoming', 1453: 'grouping', 452: 'balloons', 278: 'adult', 1248: 'female', 535: 'bizarre', 2499: 'reports', 2353: 'polished', 326: 'aluminum', 2105: 'nose', 419: 'attitude', 1051: 'domed', 1496: 'heard', 1163: 'erradically', 2848: 'stabilized', 1668: 'itself', 1605: 'images', 2552: 'roughly', 2203: 'organge', 734: 'circled', 1786: 'ligh', 3175: 'unidentified', 3388: 'yards', 2261: 'passes', 1809: 'littlefield', 2495: 'replacement', 3143: 'twice', 2366: 'positioned', 2874: 'stay', 2486: 'remaining', 3: '01', 10: '07', 263: 'abt', 1535: 'honey', 2332: 'planetary', 2401: 'pullman', 1389: 'gettysburg', 1810: 'littlestown', 1523: 'him', 3200: 'upwards', 884: 'crew', 2624: 'sees', 2385: 'prior', 52: '1965', 538: 'blackout', 280: 'advanced', 2273: 'patterns', 1316: 'fluid', 1570: 'human', 1469: 'handle', 2796: 'southeastern', 1321: 'foley', 302: 'alabama', 2723: 'sitings', 132: '39re', 1940: 'mid', 3133: 'tucson', 550: 'blinks', 1149: 'enjoying', 623: 'burger', 309: 'ali', 1620: 'inn', 240: '8230', 156: '44and', 3416: 'zipping', 1870: 'make', 1404: 'glendale', 1282: 'fl200', 1898: 'masking', 2239: 'pacing', 295: 'airliner', 2908: 'streaks', 3074: 'tracers', 2678: 'shoot', 2747: 'slowed', 1705: 'kept', 1205: 'expose', 2792: 'source', 2934: 'summverville', 2511: 'retired', 2927: 'submariner', 3172: 'unexplainable', 2479: 'reflective', 2887: 'stood', 2549: 'rotated', 2001: 'motio', 307: 'alexandria', 761: 'closely', 1451: 'group', 2078: 'nighttime', 1629: 'intelligent', 437: 'aware', 2653: 'shaft', 3078: 'tractor', 3109: 'trek', 723: 'chrysler', 2611: 'sebring', 2523: 'ridge', 426: 'austin', 3087: 'trajectory', 1357: 'frightening', 146: '43', 2226: 'ovals', 2016: 'movements', 371: 'approach', 2402: 'pulsated', 3415: 'zipped', 3289: 'wednesday', 2375: 'preparing', 2969: 'taken', 3127: 'truck', 3340: 'winds', 2597: 'scary', 2726: 'situation', 1563: 'hudson', 2376: 'presence', 1678: 'jetted', 2288: 'perceived', 2768: 'soft', 3134: 'tues', 2680: 'shoots', 1613: 'incredibly', 828: 'const', 3232: 'vehicles', 1350: 'freeway', 1733: 'lamps', 714: 'chicago', 1599: 'il', 3197: 'upright', 288: 'aftewr', 462: 'barn', 1355: 'friends', 128: '39clock', 1728: 'lafever', 1826: 'loo', 1855: 'lycoming', 1874: 'mall', 3332: 'williamsport', 2283: 'pennsylvania', 1866: 'maine', 1431: 'goto', 2464: 'reason', 1781: 'li', 1654: 'iridescent', 2087: 'noises', 575: 'boulder', 1571: 'humanoid', 2977: 'tampa', 421: 'aug', 89: '2010', 1095: 'earliest', 2249: 'paralyzing', 1782: 'liberty', 482: 'beams', 726: 'church', 2641: 'services', 261: 'abruptly', 3038: 'thunderstorm', 609: 'bronx', 3397: 'york', 1568: 'hulman', 1626: 'institute', 2986: 'technology', 3307: 'wetlands', 2048: 'national', 2514: 'return', 2382: 'previous', 1764: 'leaped', 1072: 'drips', 1917: 'memorial', 2640: 'service', 2264: 'pastor', 1589: 'i70e', 771: 'co', 1780: 'lg', 877: 'crashed', 314: 'allegedly', 285: 'affected', 918: 'dashboard', 1180: 'events', 2630: 'separated', 2983: 'teal', 3238: 'vertical', 1499: 'heavy', 1471: 'hangglider', 2879: 'stealth', 2650: 'sha', 789: 'columbia', 2644: 'setting', 334: 'amended', 920: 'date', 2949: 'suspect', 2358: 'porch', 1921: 'mentioned', 1253: 'fiance', 1488: 'he', 2569: 'said', 1243: 'feb', 646: 'call', 907: 'dallas', 1830: 'lookes', 3121: 'triangluar', 2753: 'slowy', 3270: 'walmart', 664: 'canyon', 1812: 'lived', 1381: 'general', 433: 'aviation', 3362: 'witnessing', 829: 'constan', 1964: 'mir', 311: 'aligned', 1718: 'knew', 3277: 'wasn', 1997: 'morphing', 870: 'covered', 244: '90', 376: 'approximately', 3318: 'whit', 3142: 'twenty', 2223: 'outlining', 2585: 'savannah', 1112: 'edges', 2292: 'perhaps', 2275: 'paused', 992: 'dipped', 2349: 'polar', 2195: 'orbit', 2692: 'shuttle', 3101: 'traverse', 150: '442001', 1882: 'manueverablilty', 3024: 'thornhill', 897: 'cured', 2599: 'scepticisim', 1083: 'dull', 3329: 'wildest', 136: '39ve', 1699: 'keeping', 1167: 'escort', 1827: 'look', 2470: 'rectanglular', 2246: 'parachute', 3124: 'tried', 2764: 'snap', 855: 'cou', 2141: 'obserbed', 3196: 'upper', 2417: 'quadrant', 3341: 'windshield', 444: 'backing', 2787: 'sough', 1108: 'eastward', 2626: 'self', 169: '44daughter', 1278: 'fishing', 767: 'cloudy', 1992: 'moonlight', 869: 'cover', 180: '44mind', 507: 'believer', 267: 'accidentally', 2302: 'photographed', 842: 'converged', 2851: 'staight', 736: 'circling', 1412: 'globes', 2303: 'photographic', 350: 'anomaly', 3325: 'wide', 1658: 'ish', 161: '44blue', 2215: 'ou', 1012: 'disappered', 1585: 'i35', 2724: 'sititng', 832: 'constellation', 1116: 'effortlessly', 207: '45min', 1542: 'horizontally', 3239: 'vertically', 212: '495', 247: '95s', 2315: 'pilot', 1022: 'dispatched', 2574: 'santa', 458: 'barbara', 2759: 'smokey', 1707: 'kfc', 1057: 'doughter', 1743: 'lar', 2075: 'niceville', 1889: 'marina', 1018: 'discovery', 43: '180', 946: 'degree', 3136: 'turn', 455: 'band', 354: 'anteloe', 2988: 'tehachapi', 19: '10pm', 1766: 'leavin', 2317: 'pin', 3083: 'trailing', 1059: 'douhgnut', 2131: 'ob', 2135: 'object0', 226: '65', 3245: 'vicinity', 61: '1974', 2702: 'sign', 2663: 'she', 168: '44craft', 16: '100ft', 1085: 'dumps', 2316: 'pima', 1955: 'mine', 963: 'desendes', 988: 'dimming', 597: 'brightening', 2534: 'riverside', 2607: 'searching', 3336: 'wind', 847: 'copters', 2903: 'strangely', 1840: 'lots', 3285: 'waxhaw', 3419: 'zody', 954: 'dept', 2893: 'store', 661: 'canoga', 451: 'balloon', 975: 'diego', 88: '2009', 1290: 'flared', 415: 'atleast', 3028: 'thousand', 509: 'belmont', 1185: 'exactly', 551: 'blocked', 1275: 'firing', 2241: 'palm', 942: 'deep', 1160: 'erie', 902: 'cylinders', 348: 'anoka', 1958: 'minnesota', 778: 'cohesive', 3181: 'unit', 2065: 'net', 1448: 'grid', 3370: 'woodinville', 1577: 'hurry', 2821: 'spher', 935: 'december', 221: '58pm', 1272: 'fireworks', 1052: 'don', 1071: 'drink', 2820: 'spere', 3344: 'winged', 1315: 'flowing', 1432: 'government', 1234: 'farmer', 1595: 'idaho', 2627: 'semi', 578: 'bound', 1683: 'jordan', 2201: 'ore', 418: 'atp', 1918: 'memphis', 2145: 'observers', 316: 'almond', 1211: 'eyes', 601: 'brightness', 2394: 'proximity', 638: 'c130', 2918: 'stripes', 3247: 'videotaping', 2576: 'saratoga', 3186: 'unlit', 178: '44lit', 819: 'confirming', 2369: 'post', 1358: 'frisbee', 329: 'amazed', 587: 'bradenton', 1590: 'i80', 485: 'beatrice', 2057: 'nebraska', 1848: 'lumenescent', 707: 'chemtrail', 916: 'darting', 333: 'ambiant', 378: 'aprox', 1850: 'luminescent', 1792: 'lighting', 2378: 'president', 630: 'bush', 2815: 'speech', 2741: 'sleek', 3118: 'triangled', 2362: 'portland', 2165: 'okanagan', 2817: 'speeding', 2655: 'shap', 557: 'blurred', 1342: 'fort', 1949: 'mill', 701: 'charlotte', 1725: 'la', 1213: 'face', 2449: 'read', 1603: 'illumination', 2713: 'simultaneously', 203: '44white', 2931: 'sudden', 2454: 'really', 3071: 'tr', 2041: 'myself', 613: 'brothers', 2089: 'noisless', 3068: 'tower', 1704: 'kentucky', 229: '68', 237: '80', 577: 'bounced', 204: '44with', 961: 'describe', 2387: 'probe', 1813: 'living', 2204: 'organism', 1200: 'explanation', 2476: 'redlands', 1911: 'mead', 640: 'cabazon', 2631: 'separates', 792: 'combine', 1584: 'i294', 2129: 'oaklawn', 2921: 'strobed', 1871: 'makes', 3119: 'triangler', 603: 'brillant', 1748: 'lasting', 1380: 'gave', 717: 'chills', 2807: 'span', 2892: 'stops', 2510: 'resumes', 1518: 'higher', 1634: 'intensly', 2030: 'multicolor', 2742: 'sleeping', 1873: 'male', 1647: 'investigated', 3164: 'unconfirmed', 3112: 'tri', 1190: 'exhibited', 349: 'anomalous', 194: '44shinny', 234: '737', 874: 'crafted', 1943: 'midtown', 413: 'atlanta', 3162: 'uncle', 55: '1968', 2914: 'striations', 1793: 'lightning', 566: 'bolts', 1132: 'eminating', 102: '25mph', 2962: 'tablet', 985: 'dimley', 32: '150', 1327: 'fomation', 2559: 'rr', 2099: 'northeastern', 2166: 'oklahoma', 986: 'dimly', 1226: 'fairly', 1388: 'getting', 1171: 'eubank', 588: 'brake', 365: 'apparently', 3231: 'vehicle', 1004: 'disapp', 2880: 'steel', 1395: 'girder', 1536: 'hoovered', 1564: 'hudsonville', 1515: 'hides', 2005: 'motivation', 1609: 'included', 919: 'database', 1440: 'graveyard', 1920: 'mendon', 2527: 'ring', 394: 'aroun', 3165: 'uncontrollable', 2116: 'november', 218: '50pm', 1173: 'eureka', 3051: 'tobacco', 1214: 'faceing', 1436: 'grandmesa', 1869: 'major', 2008: 'motorists', 3139: 'turnpike', 1534: 'homes', 2352: 'police', 2046: 'name', 2429: 'radiating', 1153: 'entire', 487: 'beavercreek', 1935: 'mexico', 2047: 'naples', 1162: 'erracticly', 1666: 'itermittant', 3413: 'zigzag', 2004: 'motions', 989: 'dims', 1145: 'enfield', 751: 'cleveland', 2334: 'plasma', 3338: 'windowi', 2945: 'surpr', 523: 'beyond', 665: 'capabilities', 1879: 'manmade', 993: 'dipper', 2178: 'ooltewah', 1399: 'gives', 931: 'dazzling', 2613: 'seco', 3271: 'walton', 33: '1500', 2682: 'shoreline', 608: 'broke', 3113: 'tria', 2704: 'silence', 3265: 'wake', 400: 'arvada', 69: '1986', 2882: 'steelville', 59: '1971', 1230: 'family', 1225: 'fairfield', 3212: 'vallejo', 2254: 'partially', 396: 'arranged', 2950: 'suspected', 1689: 'jumpers', 1493: 'headlight', 2555: 'roused', 650: 'cambria', 1333: 'forest', 2497: 'reported', 683: 'caused', 1189: 'exercise', 1741: 'lapse', 580: 'bouse', 1479: 'harquahala', 1049: 'doing', 2990: 'teleporting', 531: 'birch', 1401: 'glance', 2017: 'moven', 2433: 'rainbow', 411: 'asu', 1823: 'longer', 2428: 'radiant', 1565: 'hue', 1391: 'giant', 2323: 'pitch', 2536: 'roar', 429: 'ave', 1731: 'lakewood', 1359: 'frist', 1308: 'fling', 655: 'campfire', 879: 'crazy', 2230: 'overflight', 1701: 'kendall', 1795: 'ligts', 2780: 'sometimes', 2876: 'stays', 79: '200', 3391: 'yelled', 3163: 'uncomperhensable', 666: 'capablities', 1759: 'lead', 2311: 'pictures', 1503: 'heiskell', 2849: 'stadium', 2790: 'soundlessly', 1111: 'edge', 2668: 'shimmering', 3363: 'wobbled', 3155: 'uknown', 624: 'burgettstown', 188: '44pa', 2059: 'needle', 2582: 'saturday', 118: '33pm', 2101: 'northest', 1123: 'elmwood', 618: 'buffalo', 1285: 'flames', 2062: 'neighbors', 2217: 'ours', 339: 'anchorage', 304: 'alaska', 967: 'detroit', 3342: 'windsor', 678: 'castle', 1458: 'gunnison', 2785: 'soon', 3012: 'thereafter', 2228: 'ove', 488: 'beaverton', 3255: 'visibility', 1944: 'midvale', 2570: 'salt', 3209: 'utah', 2296: 'personal', 2982: 'taylorville', 2501: 'research', 2560: 'rt', 45: '190', 1773: 'leominster', 518: 'bernardino', 1030: 'dissapered', 1999: 'most', 1104: 'easterly', 2762: 'smooth', 548: 'blinkers', 201: '44two', 3383: 'wsw', 1168: 'ese', 2811: 'sparland', 2856: 'standstill', 1844: 'louisville', 710: 'chesapeake', 2347: 'pointed', 896: 'cull', 2634: 'seperately', 2595: 'scarey', 3060: 'tops', 1403: 'gleaming', 1951: 'milroy', 2556: 'route', 116: '322', 318: 'alone', 2150: 'occasions', 636: 'bwi', 3020: 'think', 1181: 'eventually', 2752: 'slows', 274: 'acted', 375: 'approximate', 2905: 'streak', 416: 'atmosphere', 2850: 'staggered', 1847: 'lower', 3297: 'wesleyan', 3182: 'university', 949: 'delaware', 2469: 'rectangler', 2891: 'stopping', 1906: 'maze', 941: 'decoration', 464: 'barrel', 2322: 'pinpoint', 1300: 'fleet', 823: 'connected', 2998: 'tethers', 1310: 'float', 3375: 'working', 26: '130', 779: 'cold', 730: 'cigarett', 2958: 'swiftly', 1540: 'horizonal', 1238: 'fas', 533: 'birds', 1852: 'lunar', 1110: 'eclipse', 2838: 'spring', 54: '1967', 2432: 'railroad', 3075: 'track', 1531: 'hollandale', 1973: 'mn', 2901: 'strang', 2826: 'spining', 1619: 'initially', 1397: 'girlfriend', 1524: 'hinckley', 641: 'cabin', 2354: 'polution', 1608: 'incident', 2974: 'talked', 3366: 'wondered', 1631: 'intensely', 1446: 'grew', 1115: 'eerie', 100: '24', 1798: 'limbs', 2255: 'partly', 810: 'concealed', 2979: 'tandem', 754: 'climbed', 2550: 'rotates', 3159: 'unaffected', 2948: 'surroundings', 512: 'belpre', 206: '45714', 1373: 'gandy', 592: 'bridge', 135: '39tl', 299: 'airpt', 2050: 'navarre', 2711: 'similar', 1060: 'dover', 284: 'afb', 143: '41', 727: 'chute', 3039: 'tiger', 660: 'canal', 321: 'alpine', 765: 'cloudless', 30: '141', 1582: 'hyw', 3033: 'throug', 367: 'appearances', 3250: 'viewing', 1856: 'm42', 2991: 'telescope', 338: 'anamolous', 2106: 'noss', 331: 'amazing', 841: 'conventional', 2779: 'sometime', 315: 'allemands', 2020: 'movies', 2750: 'slowing', 2961: 'synchronized', 2221: 'outline', 2482: 'relatively', 3061: 'topsail', 3086: 'traingular', 2364: 'pos', 2333: 'planets', 1068: 'drifted', 1130: 'emerge', 142: '408', 2841: 'spur', 3392: 'yelling', 2588: 'saying', 2400: 'pulled', 1064: 'drapes', 68: '1985', 1930: 'meteor', 2689: 'shower', 2039: 'mylar', 1717: 'kites', 173: '44headlights', 2809: 'sparking', 1410: 'glittering', 3321: 'whittier', 2573: 'sanikiluaq', 2123: 'nunavut', 18: '101', 2930: 'such', 724: 'chubby', 900: 'cylender', 1632: 'intensified', 1502: 'heights', 2728: 'sixties', 3108: 'treetops', 2297: 'philadelphia', 908: 'dance', 1878: 'manitoba', 2839: 'springfield', 1551: 'housing', 1481: 'hatteras', 1328: 'foot', 1938: 'miamisburg', 2772: 'soldiers', 1302: 'flicker', 1625: 'instead', 2740: 'slanted', 984: 'diminishing', 1776: 'lethbridge', 306: 'alberta', 2965: 'tahoe', 1127: 'emanated', 1509: 'hell', 1351: 'fresh', 1730: 'lakes', 1708: 'kids', 24: '12th', 1313: 'floor', 813: 'condo', 1148: 'enjoy', 1016: 'discoloration', 540: 'blacktop', 363: 'apopka', 1037: 'distinctly', 3091: 'transluscent', 1129: 'ember', 3029: 'thousands', 2384: 'pri', 115: '32', 39: '170', 644: 'calaveras', 3135: 'tumbling', 569: 'booster', 3216: 'vandenburge', 2242: 'palms', 1890: 'marine', 1259: 'figure', 505: 'belgrade', 71: '1992', 186: '44or', 1498: 'heat', 806: 'complety', 2424: 'quincy', 84: '2005', 629: 'burton', 1353: 'friday', 2154: 'october', 86: '2007', 1023: 'disperse', 1968: 'missing', 1474: 'happend', 2079: 'nine', 3376: 'works', 3204: 'usaf', 1713: 'kingman', 1274: 'firie', 1703: 'kennesaw', 2554: 'rounded', 3048: 'tips', 3343: 'wing', 1679: 'jetties', 2765: 'snow', 2894: 'storm', 1349: 'freaky', 1156: 'equidistant', 889: 'crossing', 2784: 'sons', 552: 'blocks', 2539: 'rockford', 160: '44blinked', 663: 'cant', 1202: 'explane', 1402: 'glaring', 1382: 'generated', 2006: 'motor', 565: 'bolted', 872: 'crack', 3170: 'unearthly', 576: 'bounce', 2391: 'property', 2590: 'says', 619: 'buglike', 2912: 'streetlight', 1526: 'hit', 2537: 'rock', 1649: 'investigators', 3056: 'tonight', 2291: 'performing', 3191: 'unusal', 1884: 'manuvers', 3364: 'woke', 471: 'bathed', 755: 'clinton', 3070: 'township', 1934: 'metroparkway', 397: 'arrived', 2594: 'scared', 1500: 'heck', 2648: 'sf', 1001: 'disapear', 2649: 'sh', 3090: 'transform', 2416: 'quad', 719: 'chopper', 824: 'consecutive', 1606: 'impossible', 1876: 'maneuvers', 757: 'cloaking', 469: 'bass', 1477: 'harbor', 1864: 'magnitude', 1867: 'maintianed', 1154: 'equal', 2632: 'separati', 2121: 'number', 2812: 'specific', 75: '19th', 22: '11am', 2567: 'ruth', 472: 'battle', 529: 'biking', 432: 'average', 2481: 'regrouped', 1984: 'monday', 709: 'cherry', 107: '2nd', 2837: 'spotting', 2865: 'starts', 3076: 'tracking', 2270: 'patten', 2247: 'parade', 347: 'annex', 2392: 'prospect', 610: 'brooklyn', 1775: 'let', 2115: 'noticing', 747: 'clayton', 1177: 'evenings', 1674: 'jellyfish', 675: 'carrier', 2414: 'pyramid', 1897: 'marysville', 3331: 'willamette', 711: 'chesepeake', 2938: 'sunrise', 3193: 'unusually', 3195: 'upon', 2142: 'observation', 241: '85', 1903: 'mauldin', 3145: 'twirling', 617: 'buena', 265: 'accelerated', 2489: 'rendezvous', 1151: 'enormous', 2269: 'patrol', 871: 'covina', 3156: 'ultra', 2687: 'should', 1815: 'loading', 3187: 'unloading', 1058: 'douglas', 2967: 'tails', 886: 'criss', 582: 'bowling', 3279: 'watch', 425: 'aurora', 2662: 'shawnee', 1722: 'ks', 2589: 'saylorsburg', 794: 'comes', 2970: 'takes', 1843: 'louis', 959: 'descending', 672: 'carmel', 2213: 'otherwise', 1186: 'exception', 1693: 'jupiter', 890: 'cruising', 1268: 'firballs', 1788: 'lightbulb', 3417: 'zips', 1334: 'forked', 1977: 'moderate', 1623: 'instantaneous', 1377: 'gaseous', 571: 'borealis', 2151: 'occurred', 515: 'bend', 1858: 'mach', 2395: 'pst', 271: 'acelerating', 1986: 'monrovia', 1286: 'flaming', 105: '28', 2164: 'oject', 853: 'cosmic', 2959: 'switched', 1784: 'lifted', 2127: 'oak', 40: '172nd', 3334: 'wilson', 1371: 'gabriel', 643: 'cajon', 2060: 'neighbor', 883: 'crete', 3176: 'unimaginable', 44: '19', 2169: 'older', 409: 'assend', 260: 'abright', 2095: 'norhwest', 826: 'consisted', 817: 'configuration', 465: 'barrie', 1207: 'extra', 2995: 'terrestrial', 1144: 'energy', 1024: 'display', 2214: 'ottawa', 1379: 'gatineau', 2104: 'norwalk', 170: '44glowing', 2579: 'satellite', 1633: 'intensity', 2092: 'none', 1684: 'jose', 795: 'comet', 2259: 'passed', 403: 'ascending', 1131: 'emerged', 1347: 'francisco', 2712: 'simple', 1194: 'expanded', 2896: 'story', 2542: 'roof', 1501: 'height', 2651: 'shadow', 2061: 'neighborhood', 1348: 'franklin', 3262: 'wached', 519: 'bernie', 1188: 'executing', 1383: 'geometric', 1976: 'model', 3030: 'thr', 1803: 'link', 3125: 'trinagle', 3346: 'wings', 3189: 'unsusal', 820: 'confirms', 3237: 'verona', 983: 'dimed', 385: 'argos', 1868: 'mainville', 1899: 'mason', 579: 'bourbonnais', 706: 'chautauqua', 1604: 'ilm', 159: '44before', 3046: 'tinted', 49: '1958', 1122: 'eleven', 1396: 'girl', 1511: 'hemet', 1038: 'distortion', 3016: 'thier', 330: 'amazement', 3304: 'weston', 2473: 'redding', 1522: 'hillsboro', 3290: 'week', 3088: 'tranlucent', 2734: 'skin', 1126: 'elyria', 2457: 'reappe', 2981: 'taped', 3243: 'vibrating', 840: 'contrail', 1255: 'fields', 801: 'companions', 1002: 'disapeard', 238: '800', 1583: 'i20', 2877: 'steadily', 325: 'altitudes', 591: 'brentwood', 1452: 'grouped', 3114: 'triagular', 2137: 'objest', 1121: 'electricity', 798: 'common', 1591: 'i805', 66: '1981', 222: '5pm', 1835: 'loose', 860: 'counter', 759: 'clockwise', 2324: 'pitched', 1046: 'doggy', 2181: 'operated', 2235: 'own', 1368: 'fying', 3009: 'thens', 1100: 'eases', 585: 'boyton', 202: '44which', 3404: 'yukon', 2718: 'sirun', 2910: 'streams', 27: '135', 2147: 'observes', 2035: 'muskoka', 854: 'cottagers', 2513: 'retreats', 3377: 'world', 3244: 'vibration', 2561: 'rudder', 1195: 'expect', 1505: 'heli', 1711: 'kind', 1638: 'interesting', 2156: 'odor', 677: 'case', 1411: 'glob', 1128: 'emanating', 387: 'arkansas', 228: '67', 1891: 'marker', 1628: 'intelligence', 1635: 'intercept', 2113: 'notice', 2389: 'process', 1740: 'laps', 2205: 'origin', 1952: 'milton', 2304: 'photographs', 269: 'account', 1039: 'disturbing', 679: 'catch', 725: 'chunks', 614: 'brown', 1979: 'mold', 264: 'acad', 1433: 'grad', 561: 'bobs', 12: '09', 2562: 'rungs', 1726: 'ladder', 517: 'bent', 559: 'boarder', 1995: 'morph', 2984: 'tear', 1956: 'mineola', 2109: 'noted', 2503: 'residence', 2621: 'seeming', 1567: 'hugging', 642: 'cabos', 1789: 'lighte', 2442: 'rapid', 1101: 'easily', 1009: 'disappearred', 1820: 'locations', 1932: 'meters', 2964: 'tag', 2616: 'sedona', 1953: 'milwaukee', 1089: 'durning', 1136: 'emitts', 816: 'conected', 2883: 'steep', 753: 'climb', 2858: 'stared', 2122: 'numerous', 3349: 'winnipeg', 2248: 'parallel', 1139: 'encircling', 149: '4410000', 2798: 'southward', 1550: 'houses', 799: 'commute', 2478: 'reflecting', 1229: 'falls', 542: 'blazed', 3082: 'trailed', 1369: 'fyling', 831: 'constantly', 312: 'alignment', 275: 'action', 3263: 'wading', 2356: 'pool', 2677: 'ships', 1950: 'million', 3259: 'visitors', 216: '50am', 3367: 'woodbridge', 2806: 'spaceship', 3339: 'windows', 1015: 'discernible', 2301: 'photo', 812: 'conclusive', 3408: 'zagging', 788: 'colours', 2770: 'sohn', 3226: 'vassar', 2185: 'ora', 1443: 'greece', 1439: 'grants', 2082: 'nm', 553: 'blood', 2897: 'strafe', 2409: 'pure', 3045: 'tinged', 341: 'angel', 3166: 'uncovered', 1406: 'glide', 1894: 'marlborough', 1652: 'inward', 1892: 'markings', 2863: 'start', 1013: 'disapwar', 2504: 'residential', 1862: 'madison', 705: 'chattanooga', 3233: 'venice', 2193: 'orangish', 152: '444', 95: '20pm', 346: 'annapolis', 3224: 'varnville', 1392: 'gigantic', 17: '100kts', 3356: 'withfour', 768: 'cloverleaf', 1193: 'exiting', 155: '44a', 923: 'daughter', 528: 'bike', 1709: 'kildonan', 922: 'daugh', 1614: 'independent', 2233: 'overwhelme', 3369: 'wooded', 434: 'avoided', 1384: 'george', 2978: 'tan', 2944: 'surface', 2612: 'secluded', 2437: 'ranch', 389: 'arlington', 1750: 'lat', 892: 'cst', 254: 'ability', 2951: 'suspend', 3006: 'themselves', 2370: 'posted', 255: 'able', 2419: 'queens', 1283: 'fla', 3010: 'theory', 2361: 'porthole', 656: 'campsite', 320: 'alongside', 492: 'becomes', 2665: 'shell', 3146: 'twitching', 2869: 'stati', 837: 'contoocook', 2073: 'nh', 3252: 'village', 1785: 'lifts', 3073: 'trace', 468: 'basketball', 2374: 'practice', 2308: 'pick', 9: '06', 2069: 'newfoundland', 1554: 'hoverd', 1945: 'might', 2208: 'orion', 513: 'belt', 1610: 'incomprhensible', 2859: 'starlight', 2465: 'recently', 2132: 'obect', 1416: 'glowed', 2852: 'staionary', 1464: 'halt', 995: 'directio', 1694: 'jus', 1070: 'drifts', 2090: 'noislessly', 1220: 'fades', 2774: 'som', 2845: 'squid', 3327: 'wierd', 1739: 'lanterns', 1755: 'launched', 279: 'adults', 1438: 'granite', 185: '44on', 1833: 'loop', 2989: 'telephone', 2350: 'pole', 595: 'briefly', 3013: 'thermal', 1663: 'issaquah', 56: '1969', 2709: 'silvery', 833: 'containing', 2019: 'movie', 2546: 'rosebud', 1233: 'farm', 144: '413', 1393: 'gila', 1596: 'identify', 1680: 'jig', 1669: 'jag', 1849: 'luminescence', 728: 'cicular', 1146: 'engaged', 1045: 'dogfight', 744: 'clarksville', 1602: 'illuminating', 127: '39ball', 3169: 'underside', 3333: 'wilmington', 1299: 'flattened', 1224: 'fairfax', 2365: 'position', 1824: 'longmont', 3300: 'wester', 850: 'cornfield', 2407: 'pulses', 2066: 'nevada', 1587: 'i580', 2128: 'oakland', 1484: 'hayward', 1763: 'leandro', 277: 'actually', 2973: 'talk', 2992: 'tell', 358: 'anyone', 3323: 'why', 2532: 'risk', 1562: 'hu', 3352: 'wish', 3382: 'written', 3272: 'want', 3374: 'worked', 2159: 'office', 807: 'complex', 605: 'bringing', 3094: 'trash', 662: 'cans', 1670: 'jan', 2854: 'standard', 1109: 'eating', 1853: 'lunch', 781: 'collor', 944: 'defies', 2307: 'physics', 2070: 'newport', 1762: 'leads', 1837: 'loss', 2160: 'officers', 2042: 'mysterious', 1158: 'equinox', 77: '1st', 1621: 'insects', 470: 'bat', 2445: 'ray', 2268: 'patio', 866: 'court', 495: 'bedford', 803: 'compelled', 2491: 'rep', 500: 'begin', 835: 'continue', 97: '2145', 1466: 'hampshire', 3348: 'winnepasauke', 1497: 'heart', 1191: 'exhibiting', 2802: 'spa', 1821: 'lone', 2043: 'mystery', 3188: 'unorderly', 3418: 'zizzaged', 912: 'darkened', 1029: 'dissapeered', 2294: 'period', 305: 'albany', 2282: 'peculiar', 2659: 'shapeshifting', 2987: 'teenagers', 2906: 'streaked', 303: 'alarming', 2652: 'shadowey', 516: 'beneath', 1000: 'disa', 2956: 'swarmed', 1901: 'massachusetts', 1783: 'lie', 913: 'darkness', 2406: 'pulsed', 2530: 'rises', 924: 'daughters', 1235: 'farmhouse', 2584: 'saucers', 1407: 'glided', 899: 'cycle', 111: '30ftlength', 217: '50feet', 2420: 'question', 1172: 'eugene', 1772: 'lentil', 1091: 'ea', 704: 'chasing', 716: 'chillicothe', 1261: 'fill', 2149: 'occasionally', 252: 'abduction', 1473: 'happen', 481: 'beamed', 388: 'arlee', 543: 'bliking', 48: '1953', 6: '0300', 534: 'birmingham', 903: 'cylindical', 1508: 'helium', 1262: 'filled', 3274: 'warm', 2036: 'must', 3201: 'urbana', 157: '44around', 1: '00am', 175: '44i', 1715: 'kitchen', 166: '44circuler', 1749: 'lasts', 1519: 'highly', 594: 'brief', 2505: 'residue', 742: 'cirular', 2022: 'movingtoward', 3008: 'thenhorizontally', 2272: 'patterned', 584: 'boyfriend', 3055: 'tone', 2388: 'proce', 208: '45pm', 687: 'cemetary', 1756: 'laure', 2341: 'plume', 47: '1951', 1449: 'grosse', 2346: 'pointe', 2232: 'overlooking', 1804: 'linked', 1223: 'fairchild', 294: 'airforce', 1854: 'lustrous', 625: 'burien', 3151: 'tysons', 1895: 'marriot', 1217: 'factory', 545: 'blinding', 978: 'difference', 1828: 'looke', 776: 'coconut', 2393: 'prove', 2058: 'need', 1048: 'doi', 800: 'commuting', 521: 'bethesda', 1461: 'hagerstown', 2710: 'simi', 2517: 'reviewing', 2418: 'quarter', 3160: 'unbelievable', 2357: 'pop', 1919: 'mena', 300: 'akansas', 2452: 'realeases', 2756: 'smells', 2126: 'nyc', 245: '911', 113: '311', 1187: 'exchange', 615: 'brownish', 2032: 'multipe', 2466: 'recived', 2300: 'phone', 353: 'answered', 3309: 'whats', 399: 'arrowhead', 2139: 'oblonged', 2736: 'skipping', 2502: 'resembles', 1387: 'gets', 2025: 'mst', 2426: 'rachel', 123: '375', 1468: 'han', 3227: 've', 3089: 'transco', 1204: 'exploding', 2808: 'sparatically', 1697: 'kalih', 695: 'chandler', 2355: 'pomona', 1367: 'fwy', 63: '1978', 2830: 'spokane', 2167: 'olathe', 28: '135th', 1627: 'instructed', 2396: 'pu', 2253: 'parkland', 1495: 'heads', 2749: 'slower', 2237: 'pace', 308: 'algona', 37: '167', 2591: 'sb', 2146: 'observery', 921: 'datetime', 746: 'classical', 956: 'desacends', 1133: 'emits', 1779: 'lexington', 459: 'barbourville', 2071: 'newton', 651: 'camden', 1266: 'fine', 3288: 'weaverville', 2939: 'sunroof', 2861: 'starring', 2777: 'somethi', 2063: 'neighbours', 283: 'afar', 3320: 'whitish', 1615: 'independently', 2431: 'raid', 196: '44slow', 3403: 'yucaipa', 2220: 'outerspace', 2434: 'rainer', 82: '2003', 3171: 'unexpected', 2461: 'reappearing', 3035: 'throughout', 1727: 'lafayette', 2603: 'scottsville', 2804: 'spacecraft', 1732: 'laminating', 586: 'bozeman', 1264: 'finally', 1769: 'led', 1883: 'manuevers', 838: 'contorting', 163: '44bright', 189: '44rectangular', 2328: 'plain', 527: 'bight', 2456: 'reapearing', 2825: 'spiltting', 1344: 'fortunate', 567: 'bonneville', 1298: 'flats', 1648: 'investigator', 2404: 'pulsations', 2007: 'motorcycle', 1544: 'hospital', 3310: 'whatsoever', 1470: 'hang', 1356: 'friendship', 1528: 'hobbs', 2266: 'patches', 1103: 'easter', 2666: 'sheridan', 1909: 'mdt', 11: '08', 230: '696', 1249: 'ferndale', 1066: 'dream', 3128: 'true', 2919: 'strob', 780: 'college', 1346: 'fourth', 2972: 'talahasee', 1434: 'gradually', 1305: 'flickers', 3258: 'visiting', 145: '41st', 431: 'avenue', 182: '44myself', 939: 'decided', 2732: 'ski', 2306: 'physical', 257: 'abnormalities', 2745: 'sligtly', 2140: 'obscurred', 3178: 'uniontown', 968: 'diam', 192: '44round', 891: 'cruses', 1073: 'driv'}


# ### code

# - Use .corr() to run the correlation on seconds, seconds_log, and minutes in the ufo DataFrame.
# - Make a list of columns to drop, in alphabetical order.
# - Use drop() to drop the columns.
# - Use the words_to_filter() function we created previously. Pass in vocab, vec.vocabulary_, desc_tfidf, and let's keep the top 4 words as the last parameter.

# In[65]:


# Check the correlation between the seconds, seconds_log, and minutes columns
print(ufo[['seconds', 'seconds_log', 'minutes']].corr())

# Make a list of features to drop
to_drop = ['minutes', 'seconds','city', 'country', 'lat', 'long', 'state', 'date',  'recorded', 'desc', 'length_of_time' ]

# Drop those features
ufo_dropped = ufo.drop(to_drop, axis=1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)


# In[66]:


filtered_words


# ## Modeling the UFO dataset, part 1
# In this exercise, we're going to build a k-nearest neighbor model to predict which country the UFO sighting took place in. Our X dataset has the log-normalized seconds column, the one-hot encoded type columns, as well as the month and year when the sighting took place. The y labels are the encoded country column, where 1 is us and 0 is ca.

# ### init: 1 dataframe, 1 serie, 

# In[67]:


from uploadfromdatacamp import saveFromFileIO

#uploadToFileIO(X, y)
tobedownloaded="{pandas.core.frame.DataFrame: {'X.csv': 'https://file.io/2gTwrd'}, pandas.core.series.Series: {'y.csv': 'https://file.io/iGKwZU'}}"
prefix='data_from_datacamp/Chap5-Exercise4.2_'
#saveFromFileIO(tobedownloaded, prefix=prefix, proxy="10.225.92.1:80")


# In[68]:


import pandas as pd
X=pd.read_csv(prefix+'X.csv',index_col=0)
y=pd.read_csv(prefix+'y.csv',index_col=0, header=None,squeeze=True)


# In[69]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()


# ### code

# In[70]:


# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets using train_test_split, setting stratify=y
train_X, test_X, train_y, test_y = train_test_split(X, y, stratify=y)

# Fit knn to the training sets
knn.fit(train_X, train_y)

# Print the score of knn on the test sets
print(knn.score(test_X, test_y))


# ## Modeling the UFO dataset, part 2
# Finally, let's build a model using the text vector we created, desc_tfidf, using the filtered_words list to create a filtered text vector. Let's see if we can predict the type of the sighting based on the text. We'll use a Naive Bayes model for this.

# ### code

# - On the desc_tfidf vector, filter by passing a list of filtered_words into the index.
# - Split up the X and y sets using train_test_split(). Remember to convert filtered_text using toarray(). Pass the y set to the stratify= parameter, since we have imbalanced classes here.
# - Use the nb model's fit() to fit train_X and train_y.
# - Print out the .score() of the nb model on the test_X and test_y sets.

# In[71]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()


# In[72]:


# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y 
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify=y)

# Fit nb to the training sets
nb.fit(train_X, train_y)

# Print the score of nb on the test sets
print(nb.score(test_X, test_y))


# ![image.png](attachment:image.png)

# In[ ]:




