#!/usr/bin/env python
# coding: utf-8

# # Efficiently combining, counting, and iterating
# 

# ## Combining Pokémon names and types
# Three lists have been loaded into your session from a dataset that contains 720 Pokémon:
# 
# - The names list contains the names of each Pokémon.
# - The primary_types list contains the corresponding primary type of each Pokémon.
# - The secondary_types list contains the corresponding secondary type of each Pokémon (nan if the Pokémon has only one type).
# 
# We want to combine each Pokémon's name and types together so that you easily see a description of each Pokémon. Practice using zip() to accomplish this task.

# ### init

# In[1]:


###################
##### list of strings
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(names,  primary_types,  secondary_types)
"""

tobedownloaded="""
{list: {'names.txt': 'https://file.io/R4BqN0',
  'primary_types.txt': 'https://file.io/j1mLT0',
  'secondary_types.txt': 'https://file.io/WIzaWQ'}}
  """
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadListFromTxt
names = loadListFromTxt(prefix+'names.txt')
primary_types = loadListFromTxt(prefix+'primary_types.txt')
secondary_types = loadListFromTxt(prefix+'secondary_types.txt')


# ### code

# In[3]:


# Combine names and primary_types
names_type1 = [*zip(names, primary_types)]

print(*names_type1[:5], sep='\n')


# In[4]:


# Combine all three lists together
names_types = [*zip(names, primary_types, secondary_types)]

print(*names_types[:5], sep='\n')


# In[7]:


# Combine five items from names and three items from primary_types
differing_lengths = [*zip(names[:5], primary_types[:3])]

print(*differing_lengths, sep='\n')


# ## Counting Pokémon from a sample
# A sample of 500 Pokémon has been generated, and three lists from this sample have been loaded into your session:
# 
# - The names list contains the names of each Pokémon in the sample.
# - The primary_types list containing the corresponding primary type of each Pokémon in the sample.
# - The generations list contains the corresponding generation of each Pokémon in the sample.
# 
# You want to quickly gather a few counts from these lists to better understand the sample that was generated. Use Counter from the collections module to explore what types of Pokémon are in your sample, what generations they come from, and how many Pokémon have a name that starts with a specific letter.
# 
# Counter has already been imported into your session for convenience.

# ### init

# In[8]:


from collections import Counter


# In[9]:


###################
##### list of strings
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(generations)
"""

tobedownloaded="""
{list: {'generations.txt': 'https://file.io/rka7fD'}}
  """
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadListFromTxt
generations = loadListFromTxt(prefix+'generations.txt')


# ### code

# In[10]:


# Collect the count of primary types
type_count = Counter(primary_types)
print(type_count, '\n')

# Collect the count of generations
gen_count = Counter(generations)
print(gen_count, '\n')

# Use list comprehension to get each Pokémon's starting letter
starting_letters = [name[0] for name in names]

# Collect the count of Pokémon for each starting_letter
starting_letters_count = Counter(starting_letters)
print(starting_letters_count)


# ## Combinations of Pokémon
# Ash, a Pokémon trainer, encounters a group of five Pokémon. These Pokémon have been loaded into a list within your session (called pokemon) and printed into the console for your convenience.
# 
# Ash would like to try to catch some of these Pokémon, but his Pokédex can only store two Pokémon at a time. Let's use combinations from the itertools module to see what the possible pairs of Pokémon are that Ash could catch.

# ### init

# In[11]:


###################
##### list of strings
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(pokemon)
"""

tobedownloaded="""
{list: {'pokemon.txt': 'https://file.io/02MNAT'}}
"""
prefixToc='1.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadListFromTxt
pokemon = loadListFromTxt(prefix+'pokemon.txt')


# In[14]:


from itertools import combinations


# ### code

# In[18]:


# Import combinations from itertools
from itertools import combinations

# Create a combination object with pairs of Pokémon
combos_obj = combinations(pokemon, 2)
print(type(combos_obj), '\n')

# Convert combos_obj to a list by unpacking
combos_2 = [*combos_obj]
print(combos_2, '\n')

# Collect all possible combinations of 4 Pokémon directly into a list
combos_4 = [*combinations(pokemon, 4)]
print(combos_4)


# # Set theory
# 

# ## Comparing Pokédexes
# Two Pokémon trainers, Ash and Misty, would like to compare their individual collections of Pokémon. Let's see what Pokémon they have in common and what Pokémon Ash has that Misty does not.
# 
# Both Ash and Misty's Pokédex (their collection of Pokémon) have been loaded into your session as lists called ash_pokedex and misty_pokedex. They have been printed into the console for your convenience.

# 
# ### init

# In[19]:


###################
##### list of strings
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(ash_pokedex , misty_pokedex)
"""

tobedownloaded="""
{list: {'ash_pokedex.txt': 'https://file.io/cWYxUC',
  'misty_pokedex.txt': 'https://file.io/dtC7sg'}}
  """
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadListFromTxt
ash_pokedex = loadListFromTxt(prefix+'ash_pokedex.txt')
misty_pokedex = loadListFromTxt(prefix+'misty_pokedex.txt')


# ### code

# In[20]:


# Convert both lists to sets
ash_set = set(ash_pokedex)
misty_set = set(misty_pokedex)

# Find the Pokémon that exist in both sets
both = ash_set.intersection(misty_set)
print(both)

# Find the Pokémon that Ash has and Misty does not have
ash_only = ash_set.difference(misty_set)
print(ash_only)

# Find the Pokémon that are in only one set (not both)
unique_to_set = ash_set.symmetric_difference(misty_set)
print(unique_to_set)


# ## Searching for Pokémon
# Two Pokémon trainers, Ash and Brock, have a collection of ten Pokémon each. Each trainer's Pokédex (their collection of Pokémon) has been loaded into your session as lists called ash_pokedex and brock_pokedex respectively.
# 
# You'd like to see if certain Pokémon are members of either Ash or Brock's Pokédex.
# 
# Let's compare using a set versus using a list when performing this membership testing.

# ### init

# In[21]:


###################
##### list of strings
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(ash_pokedex , brock_pokedex)
"""

tobedownloaded="""
{list: {'ash_pokedex.txt': 'https://file.io/8oUDoz',
  'brock_pokedex.txt': 'https://file.io/iMrlNd'}}
  """
prefixToc='2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadListFromTxt
ash_pokedex = loadListFromTxt(prefix+'ash_pokedex.txt')
brock_pokedex = loadListFromTxt(prefix+'brock_pokedex.txt')


# ### init

# In[22]:


# Convert Brock's Pokédex to a set
brock_pokedex_set = set(brock_pokedex)
print(brock_pokedex_set)


# In[24]:


# Check if Psyduck is in Ash's list and Brock's set
print('Psyduck' in  ash_pokedex)
print('Psyduck' in brock_pokedex_set)


# In[25]:


# Check if Machop is in Ash's list and Brock's set
print('Machop' in ash_pokedex)
print('Machop' in brock_pokedex_set)


# Question
# Within your IPython console, use %timeit to compare membership testing for 'Psyduck' in ash_pokedex, 'Psyduck' in brock_pokedex_set, 'Machop' in ash_pokedex, and 'Machop' in brock_pokedex_set (a total of four different timings).
# 
# Don't include the print() function. Only time the commands that you wrote inside the print() function in the previous steps.
# 
# Which membership testing was faster?

# In[26]:


get_ipython().run_line_magic('timeit', "'Psyduck' in  ash_pokedex")


# In[27]:


get_ipython().run_line_magic('timeit', "'Psyduck' in brock_pokedex_set")


# In[28]:


get_ipython().run_line_magic('timeit', "'Machop' in ash_pokedex")


# In[29]:


get_ipython().run_line_magic('timeit', "'Machop' in brock_pokedex_set")


# ## Gathering unique Pokémon
# A sample of 500 Pokémon has been created with replacement (meaning a Pokémon could be selected more than once and duplicates exist within the sample).
# 
# Three lists have been loaded into your session:
# 
# - The names list contains the names of each Pokémon in the sample.
# - The primary_types list containing the corresponding primary type of each Pokémon in the sample.
# - The generations list contains the corresponding generation of each Pokémon in the sample.
# 
# The below function was written to gather unique values from each list:
# 
# ```
# def find_unique_items(data):
#     uniques = []
# 
#     for item in data:
#         if item not in uniques:
#             uniques.append(item)
# 
#     return uniques
# ```
# Let's compare the above function to using the set data type for collecting unique items.

# ### init

# In[30]:


def find_unique_items(data):
    uniques = []

    for item in data:
        if item not in uniques:
            uniques.append(item)

    return uniques


# In[31]:


###################
##### list of strings
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(names , primary_types, generations)
"""

tobedownloaded="""
{list: {'names.txt': 'https://file.io/daT9iz',
  'primary_types.txt': 'https://file.io/qOMN5u',
  'generations.txt': 'https://file.io/shBRGt'}}
  """
prefixToc='2.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadListFromTxt
names = loadListFromTxt(prefix+'names.txt')
primary_types = loadListFromTxt(prefix+'primary_types.txt')
generations = loadListFromTxt(prefix+'generations.txt')


# ### code

# In[32]:


# Use the provided function to collect unique Pokémon names
uniq_names_func = find_unique_items(names)
print(len(uniq_names_func))


# In[33]:


# Convert the names list to a set to collect unique Pokémon names
uniq_names_set = set(names)
print(len(uniq_names_set))

# Check that both unique collections are equivalent
print(sorted(uniq_names_func) == sorted(uniq_names_set))


# Question
# Within your IPython console, use %timeit to compare the find_unique_items() function with using a set data type to collect unique Pokémon character names in names.
# 
# Which membership testing was faster?

# In[34]:


get_ipython().run_line_magic('timeit', 'find_unique_items(names)')


# In[35]:


get_ipython().run_line_magic('timeit', 'set(names)')


# In[36]:


# Use the best approach to collect unique primary types and generations
uniq_types = set(primary_types) 
uniq_gens = set(generations)
print(uniq_types, uniq_gens, sep='\n') 


# # Eliminating loops
# 

# ## Gathering Pokémon without a loop
# A list containing 720 Pokémon has been loaded into your session as poke_names. Another list containing each Pokémon's corresponding generation has been loaded as poke_gens.
# 
# A for loop has been created to filter the Pokémon that belong to generation one or two, and collect the number of letters in each Pokémon's name:
# ```
# gen1_gen2_name_lengths_loop = []
# 
# for name,gen in zip(poke_names, poke_gens):
#     if gen < 3:
#         name_length = len(name)
#         poke_tuple = (name, name_length)
#         gen1_gen2_name_lengths_loop.append(poke_tuple)
# ```

# ### init

# In[37]:


###################
##### list of strings
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(poke_names , poke_gens)
"""

tobedownloaded="""
{list: {'poke_names.txt': 'https://file.io/G8huuC',
  'poke_gens.txt': 'https://file.io/omS1QK'}}
  """
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadListFromTxt
poke_names = loadListFromTxt(prefix+'poke_names.txt')
poke_gens = loadListFromTxt(prefix+'poke_gens.txt')


# In[38]:


gen1_gen2_name_lengths_loop = []

for name,gen in zip(poke_names, poke_gens):
    if gen < 3:
        name_length = len(name)
        poke_tuple = (name, name_length)
        gen1_gen2_name_lengths_loop.append(poke_tuple)


# ### code

# In[40]:


# Collect Pokémon that belong to generation 1 or generation 2
gen1_gen2_pokemon = [name for name,gen in zip(poke_names, poke_gens) if gen in [1,2]]

# Create a map object that stores the name lengths
name_lengths_map = map(len, gen1_gen2_pokemon)

# Combine gen1_gen2_pokemon and name_lengths_map into a list
gen1_gen2_name_lengths = [*zip(gen1_gen2_pokemon, name_lengths_map)]

print(gen1_gen2_name_lengths_loop[:5])
print(gen1_gen2_name_lengths[:5])


# ![image.png](attachment:image.png)

# ## Pokémon totals and averages without a loop
# A list of 720 Pokémon has been loaded into your session called names. Each Pokémon's corresponding statistics has been loaded as a NumPy array called stats. Each row of stats corresponds to a Pokémon in names and each column represents an individual Pokémon stat (HP, Attack, Defense, Special Attack, Special Defense, and Speed respectively.)
# 
# You want to gather each Pokémon's total stat value (i.e., the sum of each row in stats) and each Pokémon's average stat value (i.e., the mean of each row in stats) so that you find the strongest Pokémon.
# 
# The below for loop was written to collect these values:
# 
# ```
# poke_list = []
# 
# for pokemon,row in zip(names, stats):
#     total_stats = np.sum(row)
#     avg_stats = np.mean(row)
#     poke_list.append((pokemon, total_stats, avg_stats))
#     ```

# ### init

# In[43]:


###################
##### list of strings
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(names , stats, poke_list)
"""

tobedownloaded="""
{list: {'names.txt': 'https://file.io/LXbjUY'},
 numpy.ndarray: {'stats.csv': 'https://file.io/jA2nHl'}}
  """
prefixToc='3.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadListFromTxt
names = loadListFromTxt(prefix+'names.txt')
from downloadfromFileIO import loadNDArrayFromCsv
stats = loadNDArrayFromCsv(prefix+'stats.csv', dtype='int64')


# In[44]:


stats


# ### code

# In[46]:


# Create a total stats array
total_stats_np = stats.sum(axis=1)

# Create an average stats array
avg_stats_np = stats.mean(axis=1)

# Combine names, total_stats_np, and avg_stats_np into a list
poke_list_np = [*zip(names, total_stats_np, avg_stats_np)]

#print(poke_list_np == poke_list, '\n')
print(poke_list_np[:3])
#print(poke_list[:3], '\n')
top_3 = sorted(poke_list_np, key=lambda x: x[1], reverse=True)[:3]
print('3 strongest Pokémon:\n{}'.format(top_3))


# # Writing better loops
# 

# ## One-time calculation loop
# A list of integers that represents each Pokémon's generation has been loaded into your session called generations. You'd like to gather the counts of each generation and determine what percentage each generation accounts for out of the total count of integers.
# 
# The below loop was written to accomplish this task:
# ```
# for gen,count in gen_counts.items():
#     total_count = len(generations)
#     gen_percent = round(count / total_count * 100, 2)
#     print(
#       'generation {}: count = {:3} percentage = {}'
#       .format(gen, count, gen_percent)
#     )
# ```
# Let's make this loop more efficient by moving a one-time calculation outside the loop.

# ### init

# In[47]:


###################
##### list of int
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(generations)
"""

tobedownloaded="""
{list: {'generations.txt': 'https://file.io/Q6QqAV'}}
"""
prefixToc='4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadListFromTxt
generations = loadListFromTxt(prefix+'generations.txt')


# ### code

# In[51]:


# Import Counter
from collections import Counter

# Collect the count of each generation
gen_counts = Counter(generations)

# Improve for loop by moving one calculation above the loop
total_count = len(generations)


for gen,count in gen_counts.items():
    gen_percent = round(count / total_count * 100, 2)
    print('generation {}: count = {:3} percentage = {}'
          .format(gen, count, gen_percent))


# ## Holistic conversion loop
# A list of all possible Pokémon types has been loaded into your session as pokemon_types. It's been printed in the console for convenience.
# 
# You'd like to gather all the possible pairs of Pokémon types. You want to store each of these pairs in an individual list with an enumerated index as the first element of each list. This allows you to see the total number of possible pairs and provides an indexed label for each pair.
# 
# The below loop was written to accomplish this task:
# 
# ```
# enumerated_pairs = []
# 
# for i,pair in enumerate(possible_pairs, 1):
#     enumerated_pair_tuple = (i,) + pair
#     enumerated_pair_list = list(enumerated_pair_tuple)
#     enumerated_pairs.append(enumerated_pair_list)
# Let's make this loop more efficient using a holistic conversion.
# ```

# ### init

# In[52]:


pokemon_types=['Bug',
 'Dark',
 'Dragon',
 'Electric',
 'Fairy',
 'Fighting',
 'Fire',
 'Flying',
 'Ghost',
 'Grass',
 'Ground',
 'Ice',
 'Normal',
 'Poison',
 'Psychic',
 'Rock',
 'Steel',
 'Water']


# In[53]:


from itertools import combinations


# ### code

# In[61]:


# Collect all possible pairs using combinations()
possible_pairs = [*combinations(pokemon_types, 2)]

# Create an empty list called enumerated_tuples
enumerated_tuples = []

# Add a line to append each enumerated_pair_tuple to the empty list above
for i,pair in enumerate(possible_pairs, 1):
    enumerated_pair_tuple = (i,) + pair
    enumerated_tuples.append(enumerated_pair_tuple)

# Convert all tuples in enumerated_tuples to a list
enumerated_pairs = [*map(list, enumerated_tuples)]
print(enumerated_pairs)


# ## Bringing it all together: Pokémon z-scores
# A list of 720 Pokémon has been loaded into your session as names. Each Pokémon's corresponding Health Points is stored in a NumPy array called hps. You want to analyze the Health Points using the [z-score](https://en.wikipedia.org/wiki/Standard_score) to see how many standard deviations each Pokémon's HP is from the mean of all HPs.
# 
# The below code was written to calculate the HP z-score for each Pokémon and gather the Pokémon with the highest HPs based on their z-scores:
# 
# ```
# poke_zscores = []
# 
# for name,hp in zip(names, hps):
#     hp_avg = hps.mean()
#     hp_std = hps.std()
#     z_score = (hp - hp_avg)/hp_std
#     poke_zscores.append((name, hp, z_score))
# highest_hp_pokemon = []
# 
# for name,hp,zscore in poke_zscores:
#     if zscore > 2:
#         highest_hp_pokemon.append((name, hp, zscore))
#         
# ```

# ### init

# In[62]:


###################
##### list of strings
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(names , hps)
"""

tobedownloaded="""
{list: {'names.txt': 'https://file.io/5QbqLv'},
 numpy.ndarray: {'hps.csv': 'https://file.io/SAgoXo'}}

  """
prefixToc='4.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadListFromTxt
names = loadListFromTxt(prefix+'names.txt')

from downloadfromFileIO import loadNDArrayFromCsv
hps = loadNDArrayFromCsv(prefix+'hps.csv')


# ### code

# In[63]:


# Calculate the total HP avg and total HP standard deviation
hp_avg = hps.mean()
hp_std = hps.std()

# Use NumPy to eliminate the previous for loop
z_scores = (hps - hp_avg)/hp_std

# Combine names, hps, and z_scores
poke_zscores2 = [*zip(names, hps, z_scores)]
print(*poke_zscores2[:3], sep='\n')


# In[64]:


# Use list comprehension with the same logic as the highest_hp_pokemon code block
highest_hp_pokemon2 = [(name, hp, z_score) for name,hp,z_score in poke_zscores2 if z_score > 2]
print(*highest_hp_pokemon2, sep='\n')


# In[ ]:




