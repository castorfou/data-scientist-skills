#!/usr/bin/env python
# coding: utf-8

# # Calculating probabilities of two events
# 

# ## Measuring a sample
# Let's work with a sample of coin flips to calculate some probabilities. You will calculate absolute and relative frequency and check the theoretical value from the distribution of the sample data.
# 
# The array sample_of_two_coin_flips has 1,000 experiments, each consisting of two fair coin flips. For each experiment, we record the number of heads out of the two coin flips: 0, 1, or 2.
# 
# We've preloaded the binom object and the find_repeats() and relfreq() methods from the scipy.stats library for you. You'll need these to calculate the probabilities in this exercise.

# ### init

# In[5]:


###################
##### numpy ndarray float
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(sample_of_two_coin_flips)
"""

tobedownloaded="""
{numpy.ndarray: {'sample_of_two_coin_flips.csv': 'https://file.io/uG0HmTRf'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc)

#initialisation

from downloadfromFileIO import loadNDArrayFromCsv
sample_of_two_coin_flips = loadNDArrayFromCsv(prefix+'sample_of_two_coin_flips.csv', dtype='int')


# ### code

# In[6]:


from scipy.stats import binom, find_repeats, relfreq


# In[7]:


# Count how many times you got 2 heads from the sample data
count_2_heads = find_repeats(sample_of_two_coin_flips).counts[2]

# Divide the number of heads by the total number of draws
prob_2_heads = count_2_heads / len(sample_of_two_coin_flips)

# Display the result
print(prob_2_heads)


# In[9]:


# Get the relative frequency from sample_of_two_coin_flips
# Set numbins as 3
# Extract frequency
rel_freq = relfreq(sample_of_two_coin_flips, numbins=3).frequency
print(rel_freq)


# In[13]:


# Probability of getting 0, 1, or 2 from the distribution
probabilities = binom.pmf(k=[0, 1, 2], n=2, p=0.5)
print(probabilities)


# ## Joint probabilities
# In this exercise we're going to calculate joint probabilities using the following table:
# 
# ![image.png](attachment:image.png)
# Joint probability table
# 
# Take the values from the table, create variables, and calculate the probability of the event in each step.

# ### code

# In[14]:


# Individual probabilities
P_Eng_works = 0.99
P_GearB_works = 0.995

# Joint probability calculation
P_both_works = P_Eng_works * P_GearB_works

print(P_both_works)


# In[15]:


# Individual probabilities
P_Eng_fails = 0.01
P_Eng_works = 1-P_Eng_fails
P_GearB_fails = 1-P_GearB_works
P_GearB_works = 0.995

# Joint probability calculation
P_only_GearB_fails = P_GearB_fails*P_Eng_works
P_only_Eng_fails = P_Eng_fails*P_GearB_works

# Calculate result
P_one_fails = P_only_Eng_fails+P_only_GearB_fails

print(P_one_fails)


# In[16]:


# Individual probabilities
P_Eng_fails = 0.01
P_Eng_works = 0.99
P_GearB_fails = 0.005
P_GearB_works = 0.995

# Joint probability calculation
P_EngW_GearBW = P_Eng_works * P_GearB_works
P_EngF_GearBF = P_Eng_fails * P_GearB_fails

# Calculate result
P_fails_or_works = P_EngW_GearBW + P_EngF_GearBF

print(P_fails_or_works)


# ## Deck of cards
# In this exercise, you'll use the following deck of cards to calculate some probabilities in each step:
# 
# ![image.png](attachment:image.png)
# Deck of cards

# ### code

# In[18]:


# Ace probability
P_Ace = 4/52

# Not Ace probability
P_not_Ace = 1 - P_Ace

print(P_not_Ace)


# In[19]:


# Figure probabilities
P_Hearts = 13/52
P_Diamonds = 13/52

# Probability of red calculation
P_Red = P_Hearts + P_Diamonds

print(P_Red)


# In[20]:


# Figure probabilities
P_Jack = 4/52
P_Spade = 13/52

# Joint probability
P_Jack_n_Spade = 1/52

# Probability of Jack or spade
P_Jack_or_Spade = P_Jack + P_Spade - P_Jack_n_Spade

print(P_Jack_or_Spade)


# In[21]:


# Figure probabilities
P_King = 4/52
P_Queen = 4/52

# Joint probability
P_King_n_Queen = 0

# Probability of King or Queen
P_King_or_Queen = P_King + P_Queen - P_King_n_Queen

print(P_King_or_Queen)


# # Conditional probabilities
# 

# ## Delayed flights
# A certain airline offers flights departing to New York on Tuesdays and Fridays, but sometimes the flights are delayed:
# 
# ![image.png](attachment:image.png)
# At the bottom of the Delayed column you have a total of 35, which means there were 35 delayed flights out of the total of 276 departures in the sample. Of these, 24 were on Tuesday and 11 on Friday.
# 
# Given the table, answer the following questions:

# ### code

# What is the probability of a flight being on time?

# In[22]:


# Needed quantities
On_time = 241
Total_departures = 276

# Probability calculation
P_On_time = On_time / Total_departures

print(P_On_time)


# Every departure is on time with probability P_On_time. What is the probability of a flight being delayed?

# In[23]:


# Needed quantities
P_On_time = 241 / 276

# Probability calculation
P_Delayed = 1 - P_On_time

print(P_Delayed)


# Given that it's Tuesday, what is the probability of a flight being delayed (P(Delayed|Tuesday))?

# In[25]:


# Needed quantities
Delayed_on_Tuesday = 24
On_Tuesday = 138

# Probability calculation
P_Delayed_g_Tuesday = Delayed_on_Tuesday / On_Tuesday

print(P_Delayed_g_Tuesday)


# Given that it's Friday, what is the probability of a flight being delayed (P(Delayed|Friday))?

# In[26]:


# Needed quantities
Delayed_on_Friday = 11
On_Friday = 138

# Probability calculation
P_Delayed_g_Friday = Delayed_on_Friday / On_Friday

print(P_Delayed_g_Friday)


# ## Contingency table
# The following table shows the numbers of red and black cards in a deck that are Aces and non-Aces:
# 
# ![image.png](attachment:image.png)
# The total in the Red column is 26, which means there are 26 red cards in the deck. Of these, 2 are Aces and 24 are non-Aces. There are 52 cards in a deck. Use the values in the table to calculate some conditional probabilities.

# ### code

# Calculate P(Ace|Red).

# In[28]:


# Individual probabilities
P_Red = 1/2
P_Red_n_Ace = 2/52

# Conditional probability calculation
P_Ace_given_Red = P_Red_n_Ace / P_Red

print(P_Ace_given_Red)


# Calculate P(Black|Ace)

# In[29]:


# Individual probabilities
P_Ace = 4/52
P_Ace_n_Black = 2/52

# Conditional probability calculation
P_Black_given_Ace = P_Ace_n_Black / P_Ace

print(P_Black_given_Ace)


# Calculate P(Non Ace|Black).
# 
# 

# In[30]:


# Individual probabilities
P_Black = 26/52
P_Black_n_Non_ace = 24/52

# Conditional probability calculation
P_Non_ace_given_Black = P_Black_n_Non_ace / P_Black

print(P_Non_ace_given_Black)


# Calculate P(Red|Non Ace).

# In[32]:


# Individual probabilities
P_Non_ace = 48/52
P_Non_ace_n_Red = 24/52

# Conditional probability calculation
P_Red_given_Non_ace = P_Non_ace_n_Red / P_Non_ace

print(P_Red_given_Non_ace)


# ## More cards
# Now let's use the deck of cards to calculate some conditional probabilities.
# 
# ![image.png](attachment:image.png)Deck of cards

# ### code

# Calculate the probability of getting two Jacks (P(Jack and Jack)).

# In[33]:


# Needed probabilities
P_first_Jack = 4/52
P_Jack_given_Jack = 3/51

# Joint probability calculation
P_two_Jacks = P_first_Jack * P_Jack_given_Jack

print(P_two_Jacks)


# Calculate P(Ace|Spade).

# In[34]:


# Needed probabilities
P_Spade = 1/4
P_Spade_n_Ace = 1/52

# Conditional probability calculation
P_Ace_given_Spade = P_Spade_n_Ace / P_Spade

print(P_Ace_given_Spade)


# Calculate P(Queen|Face card).

# In[35]:


# Needed probabilities
P_Face_card = 12/52
P_Face_card_n_Queen = 4/52

# Conditional probability calculation
P_Queen_given_Face_card = P_Face_card_n_Queen / P_Face_card

print(P_Queen_given_Face_card)


# # Total probability law
# 

# ## Formula 1 engines
# Suppose that two manufacturers, A and B, supply the engines for Formula 1 racing cars, with the following characteristics:
# 
# - 99% of the engines from factory A last more than 5,000 km.
# - Factory B manufactures engines that last more than 5,000 km with 95% probability.
# - 70% of the engines are from manufacturer A, and the rest are produced by manufacturer B.
# 
# What is the chance that an engine will last more than 5,000 km?

# ### code

# In[36]:


# Needed probabilities
P_A = 0.7
P_last5000_g_A = 0.99
P_B = 0.3
P_last5000_g_B = 0.95

# Total probability calculation
P_last_5000 = P_A*P_last5000_g_A + P_B*P_last5000_g_B

print(P_last_5000)


# ## Voters
# Of the total population of three states X, Y, and Z, 43% are from state X, 25% are from state Y, and 32% are from state Z. A poll is taken and the result is the following:
# 
# - 53% of the voters support John Doe in state X.
# - 67% of the voters support John Doe in state Y.
# - 32% of the voters support John Doe in state Z.
# 
# Given that a voter supports John Doe, answer the following questions.

# ### code

# What is the probability that the voter lives in state X and supports John Doe?
# 
# 

# In[37]:


# Individual probabilities
P_X = 0.43

# Conditional probabilities
P_Support_g_X = 0.53

# Total probability calculation
P_X_n_Support = P_X * P_Support_g_X
print(P_X_n_Support)


# What is the probability that the voter lives in state Z and does not support John Doe?

# In[38]:


# Individual probabilities
P_Z = 0.32

# Conditional probabilities
P_Support_g_Z = 0.32
P_NoSupport_g_Z = 1 - P_Support_g_Z

# Total probability calculation
P_Z_n_NoSupport = P_Z * P_NoSupport_g_Z
print(P_Z_n_NoSupport)


# What is the total percentage of voters that support John Doe?

# In[39]:


# Individual probabilities
P_X = 0.43
P_Y = 0.25
P_Z = 0.32

# Conditional probabilities
P_Support_g_X = 0.53
P_Support_g_Y = 0.67
P_Support_g_Z = 0.32

# Total probability calculation
P_Support = P_X * P_Support_g_X + P_Y * P_Support_g_Y + P_Z * P_Support_g_Z
print(P_Support)


# # Bayes' rule
# 

# ## Factories and parts
# A certain electronic part is manufactured by three different vendors named V1, V2, and V3.
# 
# Half of the parts are produced by V1, 25% by V2, and the rest by V3. The probability of a part being damaged given that it was produced by V1 is 1%, while it's 2% for V2 and 3% for V3.
# 
# If a part taken at random is damaged, answer the following questions.
# 
# 

# ### code

# What is the probability that the part was manufactured by V1?

# In[40]:


# Individual probabilities & conditional probabilities
P_V1 = 0.5
P_V2 = 0.25
P_V3 = 0.25
P_D_g_V1 = 0.01
P_D_g_V2 = 0.02
P_D_g_V3 = 0.03

# Probability of Damaged
P_Damaged = (P_V1 * P_D_g_V1) + (P_V2 * P_D_g_V2) + (P_V3 * P_D_g_V3)

# Bayes' rule for P(V1|D)
P_V1_g_D = (P_D_g_V1 * P_V1) / P_Damaged

print(P_V1_g_D)


# What is the probability that it was manufactured by V2?

# In[41]:


# Probability of Damaged
P_Damaged = (P_V1 * P_D_g_V1) + (P_V2 * P_D_g_V2) + (P_V3 * P_D_g_V3)

# Bayes' rule for P(V2|D)
P_V2_g_D = (P_V2 * P_D_g_V2) / P_Damaged

print(P_V2_g_D)


# What is the probability that the part was manufactured by V3?

# In[42]:


# Probability of Damaged
P_Damaged = (P_V1 * P_D_g_V1) + (P_V2 * P_D_g_V2) + (P_V3 * P_D_g_V3)

# Bayes' rule for P(V3|D)
P_V3_g_D = (P_V3 * P_D_g_V3) / P_Damaged

print(P_V3_g_D)


# ## Swine flu blood test
# You go to the doctor about a strong headache. The doctor randomly selects you for a blood test for swine flu, which is suspected to affect 1 in 9,000 people in your city. The accuracy of the test is 99%, meaning that the probability of a false positive is 1%. The probability of a false negative is zero.
# 
# Given that you test positive, answer the following questions.

# ### code

# What is the probability that you have swine flu?
# 
# 

# In[45]:


# Individual probabilities & conditional probabilities
P_Swine_flu = 1./9000
P_no_Swine_flu = 1 - P_Swine_flu
P_Positive_g_Swine_flu = 1
P_Positive_g_no_Swine_flu = 0.01

# Probability of Positive
P_Positive = (P_Swine_flu * P_Positive_g_Swine_flu) + (P_no_Swine_flu * P_Positive_g_no_Swine_flu)

# Bayes' rule for P(Swine_flu|Positive)
P_Swine_flu_g_Positive = (P_Positive_g_Swine_flu * P_Swine_flu) / P_Positive

print(P_Swine_flu_g_Positive)


# You went to Miami and 1 in 350 people came back with swine flu. Calculate the new probability that you'll test positive.

# In[46]:


# Individual probabilities & conditional probabilities
P_Swine_flu = 1./350
P_no_Swine_flu = 1 - P_Swine_flu
P_Positive_g_Swine_flu = 1
P_Positive_g_no_Swine_flu = 0.01

# Probability of Positive
P_Positive = (P_Swine_flu * P_Positive_g_Swine_flu) + (P_no_Swine_flu * P_Positive_g_no_Swine_flu)

# Bayes' rule for P(Swine_flu|Positive)
P_Swine_flu_g_Positive = (P_Positive_g_Swine_flu * P_Swine_flu) / P_Positive

print(P_Swine_flu_g_Positive)


# If the probability of a false positive is 2%, what is the new probability that you have swine flu after your vacation?

# In[47]:


# Individual probabilities & conditional probabilities
P_Swine_flu = 1./350
P_no_Swine_flu = 1 - P_Swine_flu
P_Positive_g_Swine_flu = 1
P_Positive_g_no_Swine_flu = 0.02

# Probability of Positive
P_Positive = P_Swine_flu * P_Positive_g_Swine_flu + P_no_Swine_flu * P_Positive_g_no_Swine_flu

# Bayes' rule for P(Swine_flu|Positive)
P_Swine_flu_g_Positive = (P_Swine_flu * P_Positive_g_Swine_flu) / P_Positive

print(P_Swine_flu_g_Positive)


# In[ ]:




