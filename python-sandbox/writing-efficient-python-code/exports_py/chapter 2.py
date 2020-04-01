#!/usr/bin/env python
# coding: utf-8

# # Examining runtime
# 

# In[1]:


get_ipython().run_line_magic('lsmagic', '')


# ## Using %timeit: your turn!
# You'd like to create a list of integers from 0 to 50 using the `range()` function. However, you are unsure whether using list comprehension or unpacking the range object into a list is faster. Let's use `%timeit` to find the best implementation.
# 
# For your convenience, a reference table of time orders of magnitude is provided below (faster at the top).
# ![image.png](attachment:image.png)

# ### code

# In[2]:


# Create a list of integers (0-50) using list comprehension
nums_list_comp = [num for num in range(51)]
print(nums_list_comp)


# In[4]:


# Create a list of integers (0-50) by unpacking range
nums_unpack = [*(range(51))]
print(nums_unpack)


# Question
# Use `%timeit` within your IPython console (i.e. not within the script.py window) to compare the runtimes for creating a list of integers from 0 to 50 using list comprehension vs. unpacking the range object. Don't include the `print()` statements when timing.
# 
# Which method was faster?

# In[5]:


get_ipython().run_line_magic('timeit', 'nums_list_comp = [num for num in range(51)]')


# In[6]:


get_ipython().run_line_magic('timeit', 'nums_unpack = [*(range(51))]')


# ## Using %timeit: specifying number of runs and loops
# A list of 480 superheroes has been loaded into your session (called `heroes`). You'd like to analyze the runtime for converting this `heroes` list into a set. Instead of relying on the default settings for `%timeit`, you'd like to only use 5 runs and 25 loops per each run.
# 
# What is the correct syntax when using %timeit and only using 5 runs with 25 loops per each run?

# In[8]:


heroes=['A-Bomb',
 'Abe Sapien',
 'Abin Sur',
 'Abomination',
 'Absorbing Man',
 'Adam Strange',
 'Agent 13',
 'Agent Bob',
 'Agent Zero',
 'Air-Walker',
 'Ajax',
 'Alan Scott',
 'Alfred Pennyworth',
 'Alien',
 'Amazo',
 'Ammo',
 'Angel',
 'Angel Dust',
 'Angel Salvadore',
 'Animal Man',
 'Annihilus',
 'Ant-Man',
 'Ant-Man II',
 'Anti-Venom',
 'Apocalypse',
 'Aqualad',
 'Aquaman',
 'Arachne',
 'Archangel',
 'Arclight',
 'Ardina',
 'Ares',
 'Ariel',
 'Armor',
 'Atlas',
 'Atom',
 'Atom Girl',
 'Atom II',
 'Aurora',
 'Azazel',
 'Bane',
 'Banshee',
 'Bantam',
 'Batgirl',
 'Batgirl IV',
 'Batgirl VI',
 'Batman',
 'Batman II',
 'Battlestar',
 'Beak',
 'Beast',
 'Beast Boy',
 'Beta Ray Bill',
 'Big Barda',
 'Big Man',
 'Binary',
 'Bishop',
 'Bizarro',
 'Black Adam',
 'Black Bolt',
 'Black Canary',
 'Black Cat',
 'Black Knight III',
 'Black Lightning',
 'Black Mamba',
 'Black Manta',
 'Black Panther',
 'Black Widow',
 'Black Widow II',
 'Blackout',
 'Blackwing',
 'Blackwulf',
 'Blade',
 'Bling!',
 'Blink',
 'Blizzard II',
 'Blob',
 'Bloodaxe',
 'Blue Beetle II',
 'Boom-Boom',
 'Booster Gold',
 'Box III',
 'Brainiac',
 'Brainiac 5',
 'Brother Voodoo',
 'Buffy',
 'Bullseye',
 'Bumblebee',
 'Cable',
 'Callisto',
 'Cannonball',
 'Captain America',
 'Captain Atom',
 'Captain Britain',
 'Captain Mar-vell',
 'Captain Marvel',
 'Captain Marvel II',
 'Carnage',
 'Cat',
 'Catwoman',
 'Cecilia Reyes',
 'Century',
 'Chamber',
 'Changeling',
 'Cheetah',
 'Cheetah II',
 'Cheetah III',
 'Chromos',
 'Citizen Steel',
 'Cloak',
 'Clock King',
 'Colossus',
 'Copycat',
 'Corsair',
 'Cottonmouth',
 'Crimson Dynamo',
 'Crystal',
 'Cyborg',
 'Cyclops',
 'Cypher',
 'Dagger',
 'Daredevil',
 'Darkhawk',
 'Darkseid',
 'Darkstar',
 'Darth Vader',
 'Dash',
 'Dazzler',
 'Deadman',
 'Deadpool',
 'Deadshot',
 'Deathlok',
 'Deathstroke',
 'Demogoblin',
 'Destroyer',
 'Diamondback',
 'Doc Samson',
 'Doctor Doom',
 'Doctor Doom II',
 'Doctor Fate',
 'Doctor Octopus',
 'Doctor Strange',
 'Domino',
 'Donna Troy',
 'Doomsday',
 'Doppelganger',
 'Drax the Destroyer',
 'Elastigirl',
 'Electro',
 'Elektra',
 'Elongated Man',
 'Emma Frost',
 'Enchantress',
 'Etrigan',
 'Evil Deadpool',
 'Evilhawk',
 'Exodus',
 'Fabian Cortez',
 'Falcon',
 'Feral',
 'Fin Fang Foom',
 'Firebird',
 'Firelord',
 'Firestar',
 'Firestorm',
 'Flash',
 'Flash II',
 'Flash III',
 'Flash IV',
 'Forge',
 'Franklin Richards',
 'Franklin Storm',
 'Frenzy',
 'Frigga',
 'Galactus',
 'Gambit',
 'Gamora',
 'Genesis',
 'Ghost Rider',
 'Giganta',
 'Gladiator',
 'Goblin Queen',
 'Goku',
 'Goliath IV',
 'Gorilla Grodd',
 'Granny Goodness',
 'Gravity',
 'Green Arrow',
 'Green Goblin',
 'Green Goblin II',
 'Green Goblin III',
 'Green Goblin IV',
 'Groot',
 'Guy Gardner',
 'Hal Jordan',
 'Han Solo',
 'Harley Quinn',
 'Havok',
 'Hawk',
 'Hawkeye',
 'Hawkeye II',
 'Hawkgirl',
 'Hawkman',
 'Hawkwoman',
 'Hawkwoman III',
 'Heat Wave',
 'Hela',
 'Hellboy',
 'Hellcat',
 'Hellstorm',
 'Hercules',
 'Hobgoblin',
 'Hope Summers',
 'Howard the Duck',
 'Hulk',
 'Human Torch',
 'Huntress',
 'Husk',
 'Hybrid',
 'Hydro-Man',
 'Hyperion',
 'Iceman',
 'Impulse',
 'Ink',
 'Invisible Woman',
 'Iron Fist',
 'Iron Man',
 'Jack of Hearts',
 'Jack-Jack',
 'James T. Kirk',
 'Jean Grey',
 'Jennifer Kale',
 'Jessica Jones',
 'Jigsaw',
 'John Stewart',
 'John Wraith',
 'Joker',
 'Jolt',
 'Jubilee',
 'Juggernaut',
 'Justice',
 'Kang',
 'Karate Kid',
 'Killer Croc',
 'Kilowog',
 'Kingpin',
 'Klaw',
 'Kraven II',
 'Kraven the Hunter',
 'Krypto',
 'Kyle Rayner',
 'Lady Deathstrike',
 'Leader',
 'Legion',
 'Lex Luthor',
 'Light Lass',
 'Lightning Lad',
 'Lightning Lord',
 'Living Brain',
 'Lizard',
 'Lobo',
 'Loki',
 'Longshot',
 'Luke Cage',
 'Luke Skywalker',
 'Mach-IV',
 'Machine Man',
 'Magneto',
 'Man-Thing',
 'Man-Wolf',
 'Mandarin',
 'Mantis',
 'Martian Manhunter',
 'Marvel Girl',
 'Master Brood',
 'Maverick',
 'Maxima',
 'Medusa',
 'Meltdown',
 'Mephisto',
 'Mera',
 'Metallo',
 'Metamorpho',
 'Metron',
 'Micro Lad',
 'Mimic',
 'Miss Martian',
 'Mister Fantastic',
 'Mister Freeze',
 'Mister Sinister',
 'Mockingbird',
 'MODOK',
 'Molten Man',
 'Monarch',
 'Moon Knight',
 'Moonstone',
 'Morlun',
 'Morph',
 'Moses Magnum',
 'Mr Immortal',
 'Mr Incredible',
 'Ms Marvel II',
 'Multiple Man',
 'Mysterio',
 'Mystique',
 'Namor',
 'Namora',
 'Namorita',
 'Naruto Uzumaki',
 'Nebula',
 'Nick Fury',
 'Nightcrawler',
 'Nightwing',
 'Northstar',
 'Nova',
 'Odin',
 'Omega Red',
 'Omniscient',
 'One Punch Man',
 'Onslaught',
 'Oracle',
 'Paul Blart',
 'Penance II',
 'Penguin',
 'Phantom Girl',
 'Phoenix',
 'Plantman',
 'Plastic Man',
 'Plastique',
 'Poison Ivy',
 'Polaris',
 'Power Girl',
 'Predator',
 'Professor X',
 'Professor Zoom',
 'Psylocke',
 'Punisher',
 'Purple Man',
 'Pyro',
 'Question',
 'Quicksilver',
 'Quill',
 "Ra's Al Ghul",
 'Raven',
 'Ray',
 'Razor-Fist II',
 'Red Arrow',
 'Red Hood',
 'Red Hulk',
 'Red Robin',
 'Red Skull',
 'Red Tornado',
 'Rhino',
 'Rick Flag',
 'Ripcord',
 'Robin',
 'Robin II',
 'Robin III',
 'Robin V',
 'Rocket Raccoon',
 'Rogue',
 'Ronin',
 'Rorschach',
 'Sabretooth',
 'Sage',
 'Sandman',
 'Sasquatch',
 'Scarecrow',
 'Scarlet Spider',
 'Scarlet Spider II',
 'Scarlet Witch',
 'Scorpion',
 'Sentry',
 'Shadow King',
 'Shadow Lass',
 'Shadowcat',
 'Shang-Chi',
 'Shatterstar',
 'She-Hulk',
 'She-Thing',
 'Shocker',
 'Shriek',
 'Sif',
 'Silver Surfer',
 'Silverclaw',
 'Sinestro',
 'Siren',
 'Siryn',
 'Skaar',
 'Snowbird',
 'Solomon Grundy',
 'Songbird',
 'Space Ghost',
 'Spawn',
 'Spider-Girl',
 'Spider-Gwen',
 'Spider-Man',
 'Spider-Woman',
 'Spider-Woman III',
 'Spider-Woman IV',
 'Spock',
 'Spyke',
 'Star-Lord',
 'Starfire',
 'Stargirl',
 'Static',
 'Steel',
 'Steppenwolf',
 'Storm',
 'Sunspot',
 'Superboy',
 'Superboy-Prime',
 'Supergirl',
 'Superman',
 'Swarm',
 'Synch',
 'T-1000',
 'Taskmaster',
 'Tempest',
 'Thanos',
 'The Comedian',
 'Thing',
 'Thor',
 'Thor Girl',
 'Thunderbird',
 'Thunderbird III',
 'Thunderstrike',
 'Thundra',
 'Tiger Shark',
 'Tigra',
 'Tinkerer',
 'Toad',
 'Toxin',
 'Trickster',
 'Triplicate Girl',
 'Triton',
 'Two-Face',
 'Ultragirl',
 'Ultron',
 'Utgard-Loki',
 'Vagabond',
 'Valerie Hart',
 'Valkyrie',
 'Vanisher',
 'Vegeta',
 'Venom',
 'Venom II',
 'Venom III',
 'Vertigo II',
 'Vibe',
 'Vindicator',
 'Violet Parr',
 'Vision',
 'Vision II',
 'Vixen',
 'Vulture',
 'Walrus',
 'War Machine',
 'Warbird',
 'Warlock',
 'Warp',
 'Warpath',
 'Wasp',
 'White Queen',
 'Winter Soldier',
 'Wiz Kid',
 'Wolfsbane',
 'Wolverine',
 'Wonder Girl',
 'Wonder Man',
 'Wonder Woman',
 'Wyatt Wingfoot',
 'X-23',
 'X-Man',
 'Yellow Claw',
 'Yellowjacket',
 'Yellowjacket II',
 'Yoda',
 'Zatanna',
 'Zoom']


# In[9]:


get_ipython().run_line_magic('timeit', '-r5 -n25 set(heroes)')


# ## Using %timeit: formal name or literal syntax
# Python allows you to create data structures using either a formal name or a literal syntax. In this exercise, you'll explore how using a literal syntax for creating a data structure can speed up runtimes.
# ![image.png](attachment:image.png)

# In[10]:


# Create a list using the formal name
formal_list = list()
print(formal_list)

# Create a list using the literal syntax
literal_list = []
print(literal_list)


# In[11]:


# Print out the type of formal_list
print(type(formal_list))

# Print out the type of literal_list
print(type(literal_list))


# Question
# Use `%timeit` in your IPython console to compare runtimes between creating a list using the formal name (`list()`) and the literal syntax (`[]`). Don't include the `print()` statements when timing.
# 
# Which naming convention is faster?

# In[14]:


get_ipython().run_line_magic('timeit', 'formal_list = list()')


# In[15]:


get_ipython().run_line_magic('timeit', 'literal_list = []')


# ## Using cell magic mode (%%timeit)
# From here on out, you'll be working with a superheroes dataset. For this exercise, a list of each hero's weight in kilograms (called `wts`) is loaded into your session. You'd like to convert these weights into pounds.
# 
# You could accomplish this using the below for loop:
# ```
# hero_wts_lbs = []
# for wt in wts:
#     hero_wts_lbs.append(wt * 2.20462)
# ```
# Or you could use a numpy array to accomplish this task:
# ```
# wts_np = np.array(wts)
# hero_wts_lbs_np = wts_np * 2.20462
# ```
# Use `%%timeit` in your IPython console to compare runtimes between these two approaches. Make sure to press `SHIFT+ENTER` after the magic command to add a new line before writing the code you wish to time. After you've finished coding, answer the following question:
# 
# Which of the above techniques is faster?

# ### code

# In[16]:


wts = [441.0,
 65.0,
 90.0,
 441.0,
 122.0,
 88.0,
 61.0,
 81.0,
 104.0,
 108.0,
 90.0,
 90.0,
 72.0,
 169.0,
 173.0,
 101.0,
 68.0,
 57.0,
 54.0,
 83.0,
 90.0,
 122.0,
 86.0,
 358.0,
 135.0,
 106.0,
 146.0,
 63.0,
 68.0,
 57.0,
 98.0,
 270.0,
 59.0,
 50.0,
 101.0,
 68.0,
 54.0,
 81.0,
 63.0,
 67.0,
 180.0,
 77.0,
 54.0,
 57.0,
 52.0,
 61.0,
 95.0,
 79.0,
 133.0,
 63.0,
 181.0,
 68.0,
 216.0,
 135.0,
 71.0,
 54.0,
 124.0,
 155.0,
 113.0,
 95.0,
 58.0,
 54.0,
 86.0,
 90.0,
 52.0,
 92.0,
 90.0,
 59.0,
 61.0,
 104.0,
 86.0,
 88.0,
 97.0,
 68.0,
 56.0,
 77.0,
 230.0,
 495.0,
 86.0,
 55.0,
 97.0,
 110.0,
 135.0,
 61.0,
 99.0,
 52.0,
 90.0,
 59.0,
 158.0,
 74.0,
 81.0,
 108.0,
 90.0,
 116.0,
 108.0,
 74.0,
 74.0,
 86.0,
 61.0,
 61.0,
 62.0,
 97.0,
 63.0,
 81.0,
 50.0,
 55.0,
 54.0,
 86.0,
 170.0,
 70.0,
 78.0,
 225.0,
 67.0,
 79.0,
 99.0,
 104.0,
 50.0,
 173.0,
 88.0,
 68.0,
 52.0,
 90.0,
 81.0,
 817.0,
 56.0,
 135.0,
 27.0,
 52.0,
 90.0,
 95.0,
 91.0,
 178.0,
 101.0,
 95.0,
 383.0,
 90.0,
 171.0,
 187.0,
 132.0,
 89.0,
 110.0,
 81.0,
 54.0,
 63.0,
 412.0,
 104.0,
 306.0,
 56.0,
 74.0,
 59.0,
 80.0,
 65.0,
 57.0,
 203.0,
 95.0,
 106.0,
 88.0,
 96.0,
 108.0,
 50.0,
 18.0,
 56.0,
 99.0,
 56.0,
 91.0,
 81.0,
 88.0,
 86.0,
 52.0,
 81.0,
 45.0,
 92.0,
 104.0,
 167.0,
 16.0,
 81.0,
 77.0,
 86.0,
 99.0,
 630.0,
 268.0,
 50.0,
 62.0,
 90.0,
 270.0,
 115.0,
 79.0,
 88.0,
 83.0,
 77.0,
 88.0,
 79.0,
 4.0,
 95.0,
 90.0,
 79.0,
 63.0,
 79.0,
 89.0,
 104.0,
 57.0,
 61.0,
 88.0,
 54.0,
 65.0,
 81.0,
 225.0,
 158.0,
 61.0,
 81.0,
 146.0,
 83.0,
 48.0,
 18.0,
 630.0,
 77.0,
 59.0,
 58.0,
 77.0,
 119.0,
 207.0,
 65.0,
 65.0,
 81.0,
 54.0,
 79.0,
 191.0,
 79.0,
 14.0,
 77.0,
 52.0,
 55.0,
 56.0,
 113.0,
 90.0,
 88.0,
 86.0,
 49.0,
 52.0,
 855.0,
 81.0,
 104.0,
 72.0,
 356.0,
 324.0,
 203.0,
 97.0,
 99.0,
 106.0,
 18.0,
 79.0,
 58.0,
 63.0,
 59.0,
 95.0,
 54.0,
 65.0,
 95.0,
 360.0,
 230.0,
 288.0,
 236.0,
 36.0,
 191.0,
 77.0,
 79.0,
 383.0,
 86.0,
 225.0,
 90.0,
 97.0,
 52.0,
 135.0,
 56.0,
 81.0,
 110.0,
 72.0,
 59.0,
 54.0,
 140.0,
 72.0,
 90.0,
 90.0,
 86.0,
 77.0,
 101.0,
 61.0,
 81.0,
 86.0,
 128.0,
 61.0,
 338.0,
 248.0,
 90.0,
 101.0,
 59.0,
 79.0,
 79.0,
 72.0,
 70.0,
 158.0,
 61.0,
 70.0,
 79.0,
 54.0,
 125.0,
 85.0,
 101.0,
 54.0,
 83.0,
 99.0,
 88.0,
 79.0,
 83.0,
 86.0,
 293.0,
 191.0,
 65.0,
 69.0,
 405.0,
 59.0,
 117.0,
 89.0,
 79.0,
 54.0,
 52.0,
 87.0,
 80.0,
 55.0,
 50.0,
 52.0,
 81.0,
 234.0,
 86.0,
 81.0,
 70.0,
 90.0,
 74.0,
 68.0,
 83.0,
 79.0,
 56.0,
 97.0,
 50.0,
 70.0,
 117.0,
 83.0,
 81.0,
 630.0,
 56.0,
 108.0,
 146.0,
 320.0,
 85.0,
 72.0,
 79.0,
 101.0,
 56.0,
 38.0,
 25.0,
 54.0,
 104.0,
 63.0,
 171.0,
 61.0,
 203.0,
 900.0,
 63.0,
 74.0,
 113.0,
 59.0,
 310.0,
 87.0,
 149.0,
 54.0,
 50.0,
 79.0,
 88.0,
 315.0,
 153.0,
 79.0,
 52.0,
 191.0,
 101.0,
 50.0,
 92.0,
 72.0,
 52.0,
 180.0,
 49.0,
 437.0,
 65.0,
 113.0,
 405.0,
 54.0,
 56.0,
 74.0,
 59.0,
 55.0,
 58.0,
 81.0,
 83.0,
 79.0,
 71.0,
 62.0,
 63.0,
 131.0,
 91.0,
 57.0,
 77.0,
 68.0,
 77.0,
 54.0,
 101.0,
 47.0,
 74.0,
 146.0,
 99.0,
 54.0,
 443.0,
 101.0,
 225.0,
 288.0,
 143.0,
 101.0,
 74.0,
 288.0,
 158.0,
 203.0,
 81.0,
 54.0,
 76.0,
 97.0,
 81.0,
 59.0,
 86.0,
 82.0,
 105.0,
 331.0,
 58.0,
 54.0,
 56.0,
 214.0,
 79.0,
 73.0,
 117.0,
 50.0,
 334.0,
 52.0,
 71.0,
 54.0,
 41.0,
 135.0,
 135.0,
 63.0,
 79.0,
 162.0,
 95.0,
 54.0,
 108.0,
 67.0,
 158.0,
 50.0,
 65.0,
 117.0,
 39.0,
 473.0,
 135.0,
 51.0,
 171.0,
 74.0,
 117.0,
 50.0,
 61.0,
 95.0,
 83.0,
 52.0,
 17.0,
 57.0,
 81.0]


# In[17]:


get_ipython().run_cell_magic('timeit', '', 'hero_wts_lbs = []\nfor wt in wts:\n    hero_wts_lbs.append(wt * 2.20462)')


# In[18]:


get_ipython().run_cell_magic('timeit', '', 'wts_np = np.array(wts)\nhero_wts_lbs_np = wts_np * 2.20462')


# # Code profiling for runtime
# 

# ## Using %lprun: spot bottlenecks
# Profiling a function allows you to dig deeper into the function's source code and potentially spot bottlenecks. When you see certain lines of code taking up the majority of the function's runtime, it is an indication that you may want to deploy a different, more efficient technique.
# 
# Lets dig deeper into the `convert_units()` function.
# ```
# def convert_units(heroes, heights, weights):
# 
#     new_hts = [ht * 0.39370  for ht in heights]
#     new_wts = [wt * 2.20462  for wt in weights]
# 
#     hero_data = {}
# 
#     for i,hero in enumerate(heroes):
#         hero_data[hero] = (new_hts[i], new_wts[i])
# 
#     return hero_data
# ```
# Load the `line_profiler` package into your IPython session. Then, use `%lprun` to profile the `convert_units()` function acting on your superheroes data. Remember to use the special syntax for working with `%lprun` (you'll have to provide a `-f` flag specifying the function you'd like to profile).
# 
# The `convert_units()` function, `heroes` list, `hts` array, and `wts` array have been loaded into your session. After you've finished coding, answer the following question:
# 
# What percentage of time is spent on the `new_hts` list comprehension line of code relative to the total amount of time spent in the `convert_units()` function?

# ### init

# In[24]:


heroes = ['A-Bomb',
 'Abe Sapien',
 'Abin Sur',
 'Abomination',
 'Absorbing Man',
 'Adam Strange',
 'Agent 13',
 'Agent Bob',
 'Agent Zero',
 'Air-Walker',
 'Ajax',
 'Alan Scott',
 'Alfred Pennyworth',
 'Alien',
 'Amazo',
 'Ammo',
 'Angel',
 'Angel Dust',
 'Angel Salvadore',
 'Animal Man',
 'Annihilus',
 'Ant-Man',
 'Ant-Man II',
 'Anti-Venom',
 'Apocalypse',
 'Aqualad',
 'Aquaman',
 'Arachne',
 'Archangel',
 'Arclight',
 'Ardina',
 'Ares',
 'Ariel',
 'Armor',
 'Atlas',
 'Atom',
 'Atom Girl',
 'Atom II',
 'Aurora',
 'Azazel',
 'Bane',
 'Banshee',
 'Bantam',
 'Batgirl',
 'Batgirl IV',
 'Batgirl VI',
 'Batman',
 'Batman II',
 'Battlestar',
 'Beak',
 'Beast',
 'Beast Boy',
 'Beta Ray Bill',
 'Big Barda',
 'Big Man',
 'Binary',
 'Bishop',
 'Bizarro',
 'Black Adam',
 'Black Bolt',
 'Black Canary',
 'Black Cat',
 'Black Knight III',
 'Black Lightning',
 'Black Mamba',
 'Black Manta',
 'Black Panther',
 'Black Widow',
 'Black Widow II',
 'Blackout',
 'Blackwing',
 'Blackwulf',
 'Blade',
 'Bling!',
 'Blink',
 'Blizzard II',
 'Blob',
 'Bloodaxe',
 'Blue Beetle II',
 'Boom-Boom',
 'Booster Gold',
 'Box III',
 'Brainiac',
 'Brainiac 5',
 'Brother Voodoo',
 'Buffy',
 'Bullseye',
 'Bumblebee',
 'Cable',
 'Callisto',
 'Cannonball',
 'Captain America',
 'Captain Atom',
 'Captain Britain',
 'Captain Mar-vell',
 'Captain Marvel',
 'Captain Marvel II',
 'Carnage',
 'Cat',
 'Catwoman',
 'Cecilia Reyes',
 'Century',
 'Chamber',
 'Changeling',
 'Cheetah',
 'Cheetah II',
 'Cheetah III',
 'Chromos',
 'Citizen Steel',
 'Cloak',
 'Clock King',
 'Colossus',
 'Copycat',
 'Corsair',
 'Cottonmouth',
 'Crimson Dynamo',
 'Crystal',
 'Cyborg',
 'Cyclops',
 'Cypher',
 'Dagger',
 'Daredevil',
 'Darkhawk',
 'Darkseid',
 'Darkstar',
 'Darth Vader',
 'Dash',
 'Dazzler',
 'Deadman',
 'Deadpool',
 'Deadshot',
 'Deathlok',
 'Deathstroke',
 'Demogoblin',
 'Destroyer',
 'Diamondback',
 'Doc Samson',
 'Doctor Doom',
 'Doctor Doom II',
 'Doctor Fate',
 'Doctor Octopus',
 'Doctor Strange',
 'Domino',
 'Donna Troy',
 'Doomsday',
 'Doppelganger',
 'Drax the Destroyer',
 'Elastigirl',
 'Electro',
 'Elektra',
 'Elongated Man',
 'Emma Frost',
 'Enchantress',
 'Etrigan',
 'Evil Deadpool',
 'Evilhawk',
 'Exodus',
 'Fabian Cortez',
 'Falcon',
 'Feral',
 'Fin Fang Foom',
 'Firebird',
 'Firelord',
 'Firestar',
 'Firestorm',
 'Flash',
 'Flash II',
 'Flash III',
 'Flash IV',
 'Forge',
 'Franklin Richards',
 'Franklin Storm',
 'Frenzy',
 'Frigga',
 'Galactus',
 'Gambit',
 'Gamora',
 'Genesis',
 'Ghost Rider',
 'Giganta',
 'Gladiator',
 'Goblin Queen',
 'Goku',
 'Goliath IV',
 'Gorilla Grodd',
 'Granny Goodness',
 'Gravity',
 'Green Arrow',
 'Green Goblin',
 'Green Goblin II',
 'Green Goblin III',
 'Green Goblin IV',
 'Groot',
 'Guy Gardner',
 'Hal Jordan',
 'Han Solo',
 'Harley Quinn',
 'Havok',
 'Hawk',
 'Hawkeye',
 'Hawkeye II',
 'Hawkgirl',
 'Hawkman',
 'Hawkwoman',
 'Hawkwoman III',
 'Heat Wave',
 'Hela',
 'Hellboy',
 'Hellcat',
 'Hellstorm',
 'Hercules',
 'Hobgoblin',
 'Hope Summers',
 'Howard the Duck',
 'Hulk',
 'Human Torch',
 'Huntress',
 'Husk',
 'Hybrid',
 'Hydro-Man',
 'Hyperion',
 'Iceman',
 'Impulse',
 'Ink',
 'Invisible Woman',
 'Iron Fist',
 'Iron Man',
 'Jack of Hearts',
 'Jack-Jack',
 'James T. Kirk',
 'Jean Grey',
 'Jennifer Kale',
 'Jessica Jones',
 'Jigsaw',
 'John Stewart',
 'John Wraith',
 'Joker',
 'Jolt',
 'Jubilee',
 'Juggernaut',
 'Justice',
 'Kang',
 'Karate Kid',
 'Killer Croc',
 'Kilowog',
 'Kingpin',
 'Klaw',
 'Kraven II',
 'Kraven the Hunter',
 'Krypto',
 'Kyle Rayner',
 'Lady Deathstrike',
 'Leader',
 'Legion',
 'Lex Luthor',
 'Light Lass',
 'Lightning Lad',
 'Lightning Lord',
 'Living Brain',
 'Lizard',
 'Lobo',
 'Loki',
 'Longshot',
 'Luke Cage',
 'Luke Skywalker',
 'Mach-IV',
 'Machine Man',
 'Magneto',
 'Man-Thing',
 'Man-Wolf',
 'Mandarin',
 'Mantis',
 'Martian Manhunter',
 'Marvel Girl',
 'Master Brood',
 'Maverick',
 'Maxima',
 'Medusa',
 'Meltdown',
 'Mephisto',
 'Mera',
 'Metallo',
 'Metamorpho',
 'Metron',
 'Micro Lad',
 'Mimic',
 'Miss Martian',
 'Mister Fantastic',
 'Mister Freeze',
 'Mister Sinister',
 'Mockingbird',
 'MODOK',
 'Molten Man',
 'Monarch',
 'Moon Knight',
 'Moonstone',
 'Morlun',
 'Morph',
 'Moses Magnum',
 'Mr Immortal',
 'Mr Incredible',
 'Ms Marvel II',
 'Multiple Man',
 'Mysterio',
 'Mystique',
 'Namor',
 'Namora',
 'Namorita',
 'Naruto Uzumaki',
 'Nebula',
 'Nick Fury',
 'Nightcrawler',
 'Nightwing',
 'Northstar',
 'Nova',
 'Odin',
 'Omega Red',
 'Omniscient',
 'One Punch Man',
 'Onslaught',
 'Oracle',
 'Paul Blart',
 'Penance II',
 'Penguin',
 'Phantom Girl',
 'Phoenix',
 'Plantman',
 'Plastic Man',
 'Plastique',
 'Poison Ivy',
 'Polaris',
 'Power Girl',
 'Predator',
 'Professor X',
 'Professor Zoom',
 'Psylocke',
 'Punisher',
 'Purple Man',
 'Pyro',
 'Question',
 'Quicksilver',
 'Quill',
 "Ra's Al Ghul",
 'Raven',
 'Ray',
 'Razor-Fist II',
 'Red Arrow',
 'Red Hood',
 'Red Hulk',
 'Red Robin',
 'Red Skull',
 'Red Tornado',
 'Rhino',
 'Rick Flag',
 'Ripcord',
 'Robin',
 'Robin II',
 'Robin III',
 'Robin V',
 'Rocket Raccoon',
 'Rogue',
 'Ronin',
 'Rorschach',
 'Sabretooth',
 'Sage',
 'Sandman',
 'Sasquatch',
 'Scarecrow',
 'Scarlet Spider',
 'Scarlet Spider II',
 'Scarlet Witch',
 'Scorpion',
 'Sentry',
 'Shadow King',
 'Shadow Lass',
 'Shadowcat',
 'Shang-Chi',
 'Shatterstar',
 'She-Hulk',
 'She-Thing',
 'Shocker',
 'Shriek',
 'Sif',
 'Silver Surfer',
 'Silverclaw',
 'Sinestro',
 'Siren',
 'Siryn',
 'Skaar',
 'Snowbird',
 'Solomon Grundy',
 'Songbird',
 'Space Ghost',
 'Spawn',
 'Spider-Girl',
 'Spider-Gwen',
 'Spider-Man',
 'Spider-Woman',
 'Spider-Woman III',
 'Spider-Woman IV',
 'Spock',
 'Spyke',
 'Star-Lord',
 'Starfire',
 'Stargirl',
 'Static',
 'Steel',
 'Steppenwolf',
 'Storm',
 'Sunspot',
 'Superboy',
 'Superboy-Prime',
 'Supergirl',
 'Superman',
 'Swarm',
 'Synch',
 'T-1000',
 'Taskmaster',
 'Tempest',
 'Thanos',
 'The Comedian',
 'Thing',
 'Thor',
 'Thor Girl',
 'Thunderbird',
 'Thunderbird III',
 'Thunderstrike',
 'Thundra',
 'Tiger Shark',
 'Tigra',
 'Tinkerer',
 'Toad',
 'Toxin',
 'Trickster',
 'Triplicate Girl',
 'Triton',
 'Two-Face',
 'Ultragirl',
 'Ultron',
 'Utgard-Loki',
 'Vagabond',
 'Valerie Hart',
 'Valkyrie',
 'Vanisher',
 'Vegeta',
 'Venom',
 'Venom II',
 'Venom III',
 'Vertigo II',
 'Vibe',
 'Vindicator',
 'Violet Parr',
 'Vision',
 'Vision II',
 'Vixen',
 'Vulture',
 'Walrus',
 'War Machine',
 'Warbird',
 'Warlock',
 'Warp',
 'Warpath',
 'Wasp',
 'White Queen',
 'Winter Soldier',
 'Wiz Kid',
 'Wolfsbane',
 'Wolverine',
 'Wonder Girl',
 'Wonder Man',
 'Wonder Woman',
 'Wyatt Wingfoot',
 'X-23',
 'X-Man',
 'Yellow Claw',
 'Yellowjacket',
 'Yellowjacket II',
 'Yoda',
 'Zatanna',
 'Zoom']


# In[26]:


hts = np.array([203. , 191. , 185. , 203. , 193. , 185. , 173. , 178. , 191. ,
       188. , 193. , 180. , 178. , 244. , 257. , 188. , 183. , 165. ,
       163. , 183. , 180. , 211. , 183. , 229. , 213. , 178. , 185. ,
       175. , 183. , 173. , 193. , 185. , 165. , 163. , 183. , 178. ,
       168. , 183. , 180. , 183. , 203. , 183. , 165. , 170. , 165. ,
       168. , 188. , 178. , 198. , 175. , 180. , 173. , 201. , 188. ,
       165. , 180. , 198. , 191. , 191. , 188. , 165. , 178. , 183. ,
       185. , 170. , 188. , 183. , 170. , 170. , 191. , 185. , 188. ,
       188. , 168. , 165. , 175. , 178. , 218. , 183. , 165. , 196. ,
       193. , 198. , 170. , 183. , 157. , 183. , 170. , 203. , 175. ,
       183. , 188. , 193. , 198. , 188. , 180. , 175. , 185. , 173. ,
       175. , 170. , 201. , 175. , 180. , 163. , 170. , 175. , 185. ,
       183. , 226. , 178. , 226. , 183. , 191. , 183. , 180. , 168. ,
       198. , 191. , 175. , 165. , 183. , 185. , 267. , 168. , 198. ,
       122. , 173. , 183. , 188. , 185. , 193. , 193. , 185. , 188. ,
       193. , 198. , 201. , 201. , 188. , 175. , 188. , 173. , 175. ,
       244. , 196. , 193. , 168. , 180. , 175. , 185. , 178. , 168. ,
       193. , 188. , 191. , 183. , 196. , 188. , 175. , 975. , 165. ,
       193. , 173. , 188. , 180. , 183. , 183. , 157. , 183. , 142. ,
       188. , 211. , 180. , 876. , 185. , 183. , 185. , 188. ,  62.5,
       198. , 168. , 175. , 183. , 198. , 178. , 178. , 188. , 180. ,
       178. , 183. , 178. , 701. , 188. , 188. , 183. , 170. , 183. ,
       185. , 191. , 165. , 175. , 185. , 175. , 170. , 180. , 213. ,
       259. , 173. , 185. , 196. , 180. , 168. ,  79. , 244. , 178. ,
       180. , 170. , 175. , 188. , 183. , 173. , 170. , 180. , 168. ,
       180. , 198. , 155. ,  71. , 178. , 168. , 168. , 170. , 188. ,
       185. , 183. , 196. , 165. , 165. , 287. , 178. , 191. , 173. ,
       244. , 234. , 201. , 188. , 191. , 183. ,  64. , 180. , 175. ,
       178. , 175. , 188. , 165. , 155. , 191. , 198. , 203. , 229. ,
       193. , 188. , 198. , 168. , 180. , 183. , 188. , 213. , 188. ,
       188. , 168. , 201. , 170. , 183. , 193. , 180. , 180. , 165. ,
       198. , 175. , 196. , 185. , 185. , 183. , 188. , 178. , 185. ,
       183. , 196. , 175. , 366. , 196. , 193. , 188. , 180. , 188. ,
       178. , 175. , 188. , 201. , 173. , 180. , 180. , 178. , 188. ,
       180. , 168. , 168. , 185. , 185. , 175. , 178. , 180. , 185. ,
       206. , 211. , 180. , 175. , 305. , 178. , 170. , 183. , 157. ,
       168. , 168. , 183. , 185. , 168. , 168. , 170. , 180. , 213. ,
       183. , 180. , 180. , 183. , 180. , 178. , 188. , 183. , 163. ,
       193. , 165. , 178. , 191. , 180. , 183. , 213. , 165. , 188. ,
       185. , 196. , 185. , 180. , 178. , 183. , 165. , 137. , 122. ,
       173. , 191. , 168. , 198. , 170. , 185. , 305. , 183. , 178. ,
       193. , 170. , 211. , 188. , 185. , 173. , 168. , 178. , 191. ,
       201. , 183. , 175. , 173. , 188. , 193. , 157. , 201. , 175. ,
       168. , 198. , 178. , 279. , 165. , 188. , 211. , 170. , 165. ,
       178. , 178. , 173. , 178. , 185. , 183. , 188. , 193. , 165. ,
       170. , 201. , 183. , 180. , 173. , 170. , 180. , 165. , 191. ,
       196. , 180. , 183. , 188. , 163. , 201. , 188. , 183. , 198. ,
       175. , 185. , 175. , 198. , 218. , 185. , 178. , 163. , 175. ,
       188. , 183. , 168. , 188. , 183. , 168. , 206. ,  15.2, 168. ,
       175. , 191. , 165. , 168. , 191. , 175. , 229. , 168. , 178. ,
       165. , 137. , 191. , 191. , 175. , 180. , 183. , 185. , 180. ,
       188. , 173. , 218. , 163. , 178. , 175. , 140. , 366. , 160. ,
       165. , 188. , 183. , 196. , 155. , 175. , 188. , 183. , 165. ,
        66. , 170. , 185. ])
wts = np.array([441.,  65.,  90., 441., 122.,  88.,  61.,  81., 104., 108.,  90.,
        90.,  72., 169., 173., 101.,  68.,  57.,  54.,  83.,  90., 122.,
        86., 358., 135., 106., 146.,  63.,  68.,  57.,  98., 270.,  59.,
        50., 101.,  68.,  54.,  81.,  63.,  67., 180.,  77.,  54.,  57.,
        52.,  61.,  95.,  79., 133.,  63., 181.,  68., 216., 135.,  71.,
        54., 124., 155., 113.,  95.,  58.,  54.,  86.,  90.,  52.,  92.,
        90.,  59.,  61., 104.,  86.,  88.,  97.,  68.,  56.,  77., 230.,
       495.,  86.,  55.,  97., 110., 135.,  61.,  99.,  52.,  90.,  59.,
       158.,  74.,  81., 108.,  90., 116., 108.,  74.,  74.,  86.,  61.,
        61.,  62.,  97.,  63.,  81.,  50.,  55.,  54.,  86., 170.,  70.,
        78., 225.,  67.,  79.,  99., 104.,  50., 173.,  88.,  68.,  52.,
        90.,  81., 817.,  56., 135.,  27.,  52.,  90.,  95.,  91., 178.,
       101.,  95., 383.,  90., 171., 187., 132.,  89., 110.,  81.,  54.,
        63., 412., 104., 306.,  56.,  74.,  59.,  80.,  65.,  57., 203.,
        95., 106.,  88.,  96., 108.,  50.,  18.,  56.,  99.,  56.,  91.,
        81.,  88.,  86.,  52.,  81.,  45.,  92., 104., 167.,  16.,  81.,
        77.,  86.,  99., 630., 268.,  50.,  62.,  90., 270., 115.,  79.,
        88.,  83.,  77.,  88.,  79.,   4.,  95.,  90.,  79.,  63.,  79.,
        89., 104.,  57.,  61.,  88.,  54.,  65.,  81., 225., 158.,  61.,
        81., 146.,  83.,  48.,  18., 630.,  77.,  59.,  58.,  77., 119.,
       207.,  65.,  65.,  81.,  54.,  79., 191.,  79.,  14.,  77.,  52.,
        55.,  56., 113.,  90.,  88.,  86.,  49.,  52., 855.,  81., 104.,
        72., 356., 324., 203.,  97.,  99., 106.,  18.,  79.,  58.,  63.,
        59.,  95.,  54.,  65.,  95., 360., 230., 288., 236.,  36., 191.,
        77.,  79., 383.,  86., 225.,  90.,  97.,  52., 135.,  56.,  81.,
       110.,  72.,  59.,  54., 140.,  72.,  90.,  90.,  86.,  77., 101.,
        61.,  81.,  86., 128.,  61., 338., 248.,  90., 101.,  59.,  79.,
        79.,  72.,  70., 158.,  61.,  70.,  79.,  54., 125.,  85., 101.,
        54.,  83.,  99.,  88.,  79.,  83.,  86., 293., 191.,  65.,  69.,
       405.,  59., 117.,  89.,  79.,  54.,  52.,  87.,  80.,  55.,  50.,
        52.,  81., 234.,  86.,  81.,  70.,  90.,  74.,  68.,  83.,  79.,
        56.,  97.,  50.,  70., 117.,  83.,  81., 630.,  56., 108., 146.,
       320.,  85.,  72.,  79., 101.,  56.,  38.,  25.,  54., 104.,  63.,
       171.,  61., 203., 900.,  63.,  74., 113.,  59., 310.,  87., 149.,
        54.,  50.,  79.,  88., 315., 153.,  79.,  52., 191., 101.,  50.,
        92.,  72.,  52., 180.,  49., 437.,  65., 113., 405.,  54.,  56.,
        74.,  59.,  55.,  58.,  81.,  83.,  79.,  71.,  62.,  63., 131.,
        91.,  57.,  77.,  68.,  77.,  54., 101.,  47.,  74., 146.,  99.,
        54., 443., 101., 225., 288., 143., 101.,  74., 288., 158., 203.,
        81.,  54.,  76.,  97.,  81.,  59.,  86.,  82., 105., 331.,  58.,
        54.,  56., 214.,  79.,  73., 117.,  50., 334.,  52.,  71.,  54.,
        41., 135., 135.,  63.,  79., 162.,  95.,  54., 108.,  67., 158.,
        50.,  65., 117.,  39., 473., 135.,  51., 171.,  74., 117.,  50.,
        61.,  95.,  83.,  52.,  17.,  57.,  81.])


# In[27]:


def convert_units(heroes, heights, weights):

    new_hts = [ht * 0.39370  for ht in heights]
    new_wts = [wt * 2.20462  for wt in weights]

    hero_data = {}

    for i,hero in enumerate(heroes):
        hero_data[hero] = (new_hts[i], new_wts[i])

    return hero_data


# ### code

# In[21]:


get_ipython().run_line_magic('load_ext', 'line_profiler')


# In[28]:


get_ipython().run_line_magic('lprun', '-f convert_units convert_units(heroes, hts, wts)')


# ![image.png](attachment:image.png)

# ## Using %lprun: fix the bottleneck
# In the previous exercise, you profiled the `convert_units()` function and saw that the `new_hts` list comprehension could be a potential bottleneck. Did you notice that the `new_wts` list comprehension also accounted for a similar percentage of the runtime? This is an indication that you may want to create the `new_hts` and `new_wts` objects using a different technique.
# 
# Since the height and weight of each hero is stored in a `numpy` array, you can use array broadcasting rather than list comprehension to convert the heights and weights. This has been implemented in the below function:
# ```
# def convert_units_broadcast(heroes, heights, weights):
# 
#     # Array broadcasting instead of list comprehension
#     new_hts = heights * 0.39370
#     new_wts = weights * 2.20462
# 
#     hero_data = {}
# 
#     for i,hero in enumerate(heroes):
#         hero_data[hero] = (new_hts[i], new_wts[i])
# 
#     return hero_data
# ```
# Load the `line_profiler` package into your IPython session. Then, use `%lprun` to profile the `convert_units_broadcast()` function acting on your superheroes data. The `convert_units_broadcast()` function, `heroes` list, `hts` array, and `wts` array have been loaded into your session. After you've finished coding, answer the following question:
# 
# What percentage of time is spent on the new_hts array broadcasting line of code relative to the total amount of time spent in the `convert_units_broadcast()` function?
# 

# ### init

# In[29]:


def convert_units_broadcast(heroes, heights, weights):

    # Array broadcasting instead of list comprehension
    new_hts = heights * 0.39370
    new_wts = weights * 2.20462

    hero_data = {}

    for i,hero in enumerate(heroes):
        hero_data[hero] = (new_hts[i], new_wts[i])

    return hero_data


# ### code

# In[30]:


get_ipython().run_line_magic('load_ext', 'line_profiler')
get_ipython().run_line_magic('lprun', '-f convert_units_broadcast convert_units_broadcast(heroes, hts, wts)')


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # Code profiling for memory usage
# 

# ## Using %mprun: Hero BMI
# 
# You'd like to calculate the body mass index (BMI) for a selected sample of heroes. BMI can be calculated using the below formula:
# 
# BMI = mass(kg) / height(m)^2
# 
# A random sample of 25,000 superheroes has been loaded into your session as an array called sample_indices. This sample is a list of indices that corresponds to each superhero's index selected from the heroes list.
# 
# A function named calc_bmi_lists has also been created and saved to a file titled bmi_lists.py. For convenience, it is displayed below:
# ```
# def calc_bmi_lists(sample_indices, hts, wts):
# 
#     # Gather sample heights and weights as lists
#     s_hts = [hts[i] for i in sample_indices]
#     s_wts = [wts[i] for i in sample_indices]
# 
#     # Convert heights from cm to m and square with list comprehension
#     s_hts_m_sqr = [(ht / 100) ** 2 for ht in s_hts]
# 
#     # Calculate BMIs as a list with list comprehension
#     bmis = [s_wts[i] / s_hts_m_sqr[i] for i in range(len(sample_indices))]
# 
#     return bmis
# ```
# Notice that this function performs all necessary calculations using list comprehension (hence the name calc_bmi_lists()). Dig deeper into this function and analyze the memory footprint for performing your calculations using lists:
# 
# Load the memory_profiler package into your IPython session.
# Import calc_bmi_lists from bmi_lists.
# Once you've completed the above steps, use %mprun to profile the calc_bmi_lists() function acting on your superheroes data. The hts array and wts array have already been loaded into your session.
# After you've finished coding, answer the following question:
# 
# How much memory do the list comprehension lines of code consume in the calc_bmi_lists() function? (i.e., what is the total sum of the Increment column for these four lines of code?)

# ### init

# In[39]:


###################
##### numpy ndarray float
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(sample_indices)
"""

tobedownloaded="""
{numpy.ndarray: {'sample_indices.csv': 'https://file.io/RCm5CD'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

from downloadfromFileIO import loadNDArrayFromCsv
sample_indices = loadNDArrayFromCsv(prefix+'sample_indices.csv', dtype='int64')


# ### code

# In[40]:


from bmi_lists import calc_bmi_lists


# In[41]:


get_ipython().run_line_magic('load_ext', 'memory_profiler')
get_ipython().run_line_magic('mprun', '-f calc_bmi_lists calc_bmi_lists(sample_indices, hts, wts)')


# ![image.png](attachment:image.png)

# ## Using %mprun: Hero BMI 2.0
# 
# Let's see if using a different approach to calculate the BMIs can save some memory. If you remember, each hero's height and weight is stored in a numpy array. That means you can use NumPy's handy array indexing capabilities and broadcasting to perform your calculations. A function named calc_bmi_arrays has been created and saved to a file titled bmi_arrays.py. For convenience, it is displayed below:
# ```
# def calc_bmi_arrays(sample_indices, hts, wts):
# 
#     # Gather sample heights and weights as arrays
#     s_hts = hts[sample_indices]
#     s_wts = wts[sample_indices]
# 
#     # Convert heights from cm to m and square with broadcasting
#     s_hts_m_sqr = (s_hts / 100) ** 2
# 
#     # Calculate BMIs as an array using broadcasting
#     bmis = s_wts / s_hts_m_sqr
# 
#     return bmis
# ```
# Notice that this function performs all necessary calculations using arrays.
# 
# Let's see if this updated array approach decreases your memory footprint:
# 
# Load the memory_profiler package into your IPython session.
# Import calc_bmi_arrays from bmi_arrays.
# Once you've completed the above steps, use %mprun to profile the calc_bmi_arrays() function acting on your superheroes data. The sample_indices array, hts array, and wts array have been loaded into your session.
# After you've finished coding, answer the following question:
# 
# How much memory do the array indexing and broadcasting lines of code consume in the calc_bmi_array() function? (i.e., what is the total sum of the Increment column for these four lines of code?)

# ### code

# In[43]:


from bmi_arrays import calc_bmi_arrays


# In[44]:


get_ipython().run_line_magic('load_ext', 'memory_profiler')
get_ipython().run_line_magic('mprun', '-f calc_bmi_arrays calc_bmi_arrays(sample_indices, hts, wts)')


# ![image.png](attachment:image.png)

# ## Bringing it all together: Star Wars profiling
# 
# A list of 480 superheroes has been loaded into your session (called heroes) as well as a list of each hero's corresponding publisher (called publishers).
# 
# You'd like to filter the heroes list based on a hero's specific publisher, but are unsure which of the below functions is more efficient.
# ```
# def get_publisher_heroes(heroes, publishers, desired_publisher):
# 
#     desired_heroes = []
# 
#     for i,pub in enumerate(publishers):
#         if pub == desired_publisher:
#             desired_heroes.append(heroes[i])
# 
#     return desired_heroes
# ```
# 
# ```
# def get_publisher_heroes_np(heroes, publishers, desired_publisher):
# 
#     heroes_np = np.array(heroes)
#     pubs_np = np.array(publishers)
# 
#     desired_heroes = heroes_np[pubs_np == desired_publisher]
# 
#     return desired_heroes
# ```

# ### init

# In[45]:


publishers = ['Marvel Comics',
 'Dark Horse Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Dark Horse Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Dark Horse Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Team Epic TV',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'George Lucas',
 'Dark Horse Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Dark Horse Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Shueisha',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'George Lucas',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Dark Horse Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Dark Horse Comics',
 'Star Trek',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'George Lucas',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Team Epic TV',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Dark Horse Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Shueisha',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Team Epic TV',
 'Shueisha',
 'Marvel Comics',
 'DC Comics',
 'Sony Pictures',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Dark Horse Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Image Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Star Trek',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Dark Horse Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Team Epic TV',
 'Marvel Comics',
 'Marvel Comics',
 'Shueisha',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Dark Horse Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'DC Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'Marvel Comics',
 'George Lucas',
 'DC Comics',
 'DC Comics']


# In[46]:


def get_publisher_heroes(heroes, publishers, desired_publisher):

    desired_heroes = []

    for i,pub in enumerate(publishers):
        if pub == desired_publisher:
            desired_heroes.append(heroes[i])

    return desired_heroes
def get_publisher_heroes_np(heroes, publishers, desired_publisher):

    heroes_np = np.array(heroes)
    pubs_np = np.array(publishers)

    desired_heroes = heroes_np[pubs_np == desired_publisher]

    return desired_heroes


# ### code

# In[47]:


# Use get_publisher_heroes() to gather Star Wars heroes
star_wars_heroes = get_publisher_heroes(heroes, publishers, 'George Lucas')

print(star_wars_heroes)
print(type(star_wars_heroes))

# Use get_publisher_heroes_np() to gather Star Wars heroes
star_wars_heroes_np = get_publisher_heroes_np(heroes, publishers, 'George Lucas')

print(star_wars_heroes_np)
print(type(star_wars_heroes_np))


# Question
# Within your IPython console, load the line_profiler and use %lprun to profile the two functions for line-by-line runtime. When using %lprun, use each function to gather the Star Wars heroes as you did in the previous step. After you've finished profiling, answer the following question:
# Which function has the fastest runtime?

# In[49]:


get_ipython().run_line_magic('load_ext', 'line_profiler')
get_ipython().run_line_magic('lprun', '-f get_publisher_heroes get_publisher_heroes(heroes, publishers, "George Lucas")')


# ![image.png](attachment:image.png)

# In[50]:


get_ipython().run_line_magic('load_ext', 'line_profiler')
get_ipython().run_line_magic('lprun', '-f get_publisher_heroes_np get_publisher_heroes_np(heroes, publishers, "George Lucas")')


# ![image.png](attachment:image.png)

# Question
# Within your IPython console, load the memory_profiler and use %mprun to profile the two functions for line-by-line memory consumption.
# The get_publisher_heroes() function and get_publisher_heroes_np() function have been saved within a file titled hero_funcs.py (i.e., you can import both functions from hero_funcs). When using %mprun, use each function to gather the Star Wars heroes as you did in the previous step. After you've finished profiling, answer the following question:
# 
# Which function uses the least amount of memory?

# In[ ]:




