#!/usr/bin/env python
# coding: utf-8

# # From sample mean to population mean
# 

# ## Generating a sample
# A hospital's planning department is investigating different treatments for newborns. As a data scientist you are hired to simulate the sex of 250 newborn children, and you are told that on average 50.50% are males.

# ### code

# In[1]:


# Import the binom object
from scipy.stats import binom

# Generate a sample of 250 newborn children
sample = binom.rvs(n=1, p=0.505, size=250, random_state=42)

# Show the sample values
print(sample)


# ## Calculating the sample mean
# Now you can calculate the sample mean for this generated sample by taking some elements from the sample.
# 
# Using the sample variable you just created, you'll calculate the sample means of the first 10, 50, and 250 samples.
# 
# The binom object and describe() method from scipy.stats have been imported for your convenience.

# ### code

# In[2]:


from scipy.stats import describe


# In[3]:


# Print the sample mean of the first 10 samples
print(describe(sample[0:10]).mean)


# In[4]:


# Print the sample mean of the first 50 samples
print(describe(sample[0:50]).mean)


# In[5]:


# Print the sample mean of the first 250 samples
print(describe(sample[0:250]).mean)


# ## Plotting the sample mean
# Now let's plot the sample mean, so you can see more clearly how it evolves as more data becomes available.
# 
# For this exercise we'll again use the sample you generated earlier, which is available in the sample variable. The binom object and describe() function have already been imported for you from scipy.stats, and matplotlib.pyplot is available as plt.

# ### code

# In[9]:


import matplotlib.pyplot as plt


# In[10]:


# Calculate sample mean and store it on averages array
averages = []
for i in range(2, 251):
    averages.append(describe(sample[0:i]).mean)


# In[15]:


# Add population mean line and sample mean plot
plt.axhline(binom.mean(n=1, p=0.505), color='red')
plt.plot(averages, '-')

# Add legend
plt.legend(("Population mean","Sample mean"), loc='upper right')
plt.show()


# # Adding random variables
# 

# ## Sample means
# An important result in probability and statistics is that the shape of the distribution of the means of random variables tends to a normal distribution, which happens when you add random variables with any distribution with the same expected value and variance.
# 
# For your convenience, we've loaded binom and describe() from the scipy.stats library and imported matplotlib.pyplot as plt and numpy as np. We generated a simulated population with size 1,000 that follows a binomial distribution for 10 fair coin flips and is available in the population variable.

# ### code

# In[2]:


from scipy.stats import binom, describe
import matplotlib.pyplot as plt
import numpy as np
population = binom.rvs(n=10, p=0.5, size=1000)


# In[5]:


# Create list for sample means
sample_means = []
for _ in range(1500):
	# Take 20 values from the population
    sample = np.random.choice(population, 20)
    # Calculate the sample mean
    sample_means.append(describe(sample).mean)


# In[6]:



# Plot the histogram
plt.hist(sample_means)
plt.xlabel("Sample mean values")
plt.ylabel("Frequency")
plt.show()


# ## Sample means follow a normal distribution
# In the previous exercise, we generated a population that followed a binomial distribution, chose 20 random samples from the population, and calculated the sample mean. Now we're going to test some other probability distributions to see the shape of the sample means.
# 
# From the scipy.stats library, we've loaded the poisson and geom objects and the describe() function. We've also imported matplotlib.pyplot as plt and numpy as np.
# 
# As you'll see, the shape of the distribution of the means is the same even though the samples are generated from different distributions.

# ### code

# In[7]:


from scipy.stats import geom


# In[8]:


# Generate the population
population = geom.rvs(p=0.5, size=1000)

# Create list for sample means
sample_means = []
for _ in range(3000):
	# Take 20 values from the population
    sample = np.random.choice(population, 20)
    # Calculate the sample mean
    sample_means.append(describe(sample).mean)

# Plot the histogram
plt.hist(sample_means)
plt.show()


# In[9]:


from scipy.stats import poisson


# In[10]:


# Generate the population
population = poisson.rvs(mu=2, size=1000)

# Create list for sample means
sample_means = []
for _ in range(1500):
	# Take 20 values from the population
    sample = np.random.choice(population, 20)
    # Calculate the sample mean
    sample_means.append(describe(sample).mean)

# Plot the histogram
plt.hist(sample_means)
plt.show()


# ## Adding dice rolls
# To illustrate the central limit theorem, we are going to work with dice rolls. We'll generate the samples and then add them to plot the outcome.
# 
# You're provided with a function named roll_dice() that will generate the sample dice rolls. numpy is already imported as np for your convenience: you have to use np.add(sample1, sample2) to add samples. Also, matplotlib.pyplot is imported as plt so you can plot the histograms.

# ### init

# In[13]:


###################
##### inspect Function
###################

""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
import inspect
print_func(roll_dice)
"""
import random

def roll_dice(num_rolls):
    """Generate dice roll simulations

    Parameters
    ----------
    num_rolls : int
        The number of dice rolls to simulate

    Returns
    -------
    list
        a list with num_rolls simulations of dice rolls
    """
    
    sample = []
    for i in range(num_rolls):
        sample.append(random.randint(1,6))
    return(sample)


# ### code

# In[14]:


# Configure random generator
np.random.seed(42)

# Generate the sample
sample1 = roll_dice(2000)

# Plot the sample
plt.hist(sample1, bins=range(1, 8), width=0.9)
plt.show()  


# In[15]:


# Configure random generator
np.random.seed(42)

# Generate two samples of 2000 dice rolls
sample1 = roll_dice(2000)
sample2 = roll_dice(2000)

# Add the first two samples
sum_of_1_and_2 = np.add(sample1, sample2)

# Plot the sum
plt.hist(sum_of_1_and_2, bins=range(2, 14), width=0.9)
plt.show()


# In[16]:


# Configure random generator
np.random.seed(42)

# Generate the samples
sample1 = roll_dice(2000)
sample2 = roll_dice(2000)
sample3 = roll_dice(2000)

# Add the first two samples
sum_of_1_and_2 = np.add(sample1, sample2)

# Add the first two with the third sample
sum_of_3_samples = np.add(sum_of_1_and_2, sample3)

# Plot the result
plt.hist(sum_of_3_samples, bins=range(3, 20), width=0.9)
plt.show() 


# # Linear regression
# 

# ## Fitting a model
# A university has provided you with data that shows a relationship between the hours of study and the scores that the students get on a given test.
# 
# You have access to the data through the variables hours_of_study and scores. Use a linear model to learn from the data.

# ### code

# In[17]:


hours_of_study = [4, 8, 8, 12, 8, 9, 6, 11, 13, 13, 19, 16, 17, 17, 21, 21, 23, 27, 30, 24]
scores = [52, 54, 61, 63, 63, 60, 61, 70, 75, 77, 76, 79, 81, 83, 85, 86, 88, 90, 95, 93]


# In[18]:


# Import the linregress() function
from scipy.stats import linregress

# Get the model parameters
slope, intercept, r_value, p_value, std_err = linregress(hours_of_study, scores)

# Print the linear model parameters
print('slope:', slope)
print('intercept:', intercept)


# ## Predicting test scores
# With the relationship between the hours of study and the scores that students got on a given test, you already got the parameters of a linear model, slope and intercept. With those parameters, let's predict the test score for a student who studies for 10 hours.
# 
# For this exercise, the linregress() function has been imported for you from scipy.stats.

# ### code

# In[19]:


# Get the predicted test score for given hours of study
score = slope*10 + intercept
print('score:', score)


# In[20]:


# Get the predicted test score for given hours of study
score = slope*9 + intercept
print('score:', score)


# In[21]:


# Get the predicted test score for given hours of study
score = slope*12 + intercept
print('score:', score)


# ## Studying residuals
# To implement a linear model you must study the residuals, which are the distances between the predicted outcomes and the data.
# 
# Three conditions must be met:
# 
# - The mean should be 0.
# - The variance must be constant.
# - The distribution must be normal.
# 
# We will work with data of test scores for two schools, A and B, on the same subject. model_A and model_B were fitted with hours_of_study_A and test_scores_A and hours_of_study_B and test_scores_B, respectively.
# 
# matplotlib.pyplot has been imported as plt, numpy as np and LinearRegression from sklearn.linear_model.

# ### init

# In[36]:


from sklearn.linear_model import LinearRegression
hours_of_study_A = np.array([[ 4],       [ 9],       [ 7],       [12],       [ 3],       [ 9],       [ 6],       [11],       [13],       [13],       [19],       [16],       [17],       [17],       [13],       [21],       [23],       [27],       [30],       [24]])
hours_of_study_values_A = np.array([[ 1. ],       [ 1.5],       [ 2. ],       [ 2.5],       [ 3. ],       [ 3.5],       [ 4. ],       [ 4.5],       [ 5. ],       [ 5.5],       [ 6. ],       [ 6.5],       [ 7. ],       [ 7.5],       [ 8. ],       [ 8.5],       [ 9. ],       [ 9.5],       [10. ],       [10.5],       [11. ],       [11.5],       [12. ],       [12.5],       [13. ],       [13.5],       [14. ],       [14.5],       [15. ],       [15.5],       [16. ],       [16.5],       [17. ],       [17.5],       [18. ],       [18.5],       [19. ],       [19.5],       [20. ],       [20.5],       [21. ],       [21.5],       [22. ],       [22.5],       [23. ],       [23.5],       [24. ],       [24.5],       [25. ],       [25.5],       [26. ],       [26.5],       [27. ],       [27.5],       [28. ],       [28.5],       [29. ],       [29.5],       [30. ],       [30.5]])
hours_of_study_values_B = hours_of_study_values_A
hours_of_study_B = np.array([[ 4],       [ 9],       [ 7],       [12],       [ 3],       [ 9],       [ 6],       [11],       [13],       [13],       [19],       [16],       [17],       [17],       [13],       [21],       [23],       [27],       [30],       [24],       [17],       [17],       [19],       [19],       [19],       [19]])

test_scores_B = [58, 70, 60, 65, 57, 63, 63, 73, 65, 77, 58, 62, 62, 90, 85, 95, 97, 95, 65, 65, 70, 75, 65, 75, 85, 93]
test_scores_A = [52, 56, 59, 60, 61, 62, 63, 73, 75, 77, 76, 79, 81, 83, 85, 87, 89, 89, 89, 93]
model_A = LinearRegression()
model_A.fit(hours_of_study_A, test_scores_A)
model_B = LinearRegression()
model_B.fit(hours_of_study_B, test_scores_B)


# ### code

# In[40]:


# Scatterplot of hours of study and test scores
plt.scatter(hours_of_study_A, test_scores_A)

# Plot of hours_of_study_values_A and predicted values
plt.plot(hours_of_study_values_A, model_A.predict(hours_of_study_values_A))
plt.title("Model A", fontsize=25)
plt.show()


# In[38]:


# Calculate the residuals
residuals_A = model_A.predict(hours_of_study_A) - test_scores_A

# Make a scatterplot of residuals of model_A
plt.scatter(hours_of_study_A, residuals_A)

# Add reference line and title and show plot
plt.hlines(0, 0, 30, colors='r', linestyles='--')
plt.title("Residuals plot of Model A", fontsize=25)
plt.show()


# In[41]:


# Scatterplot of hours of study and test scores
plt.scatter(hours_of_study_B, test_scores_B)

# Plot of hours_of_study_values_B and predicted values
plt.plot(hours_of_study_values_B, model_B.predict(hours_of_study_values_B))
plt.title("Model B", fontsize=25)
plt.show()


# In[42]:


# Calculate the residuals
residuals_B = model_B.predict(hours_of_study_B) - test_scores_B

# Make a scatterplot of residuals of model_B
plt.scatter(hours_of_study_B, residuals_B)

# Add reference line and title and show plot
plt.hlines(0, 0, 30, colors='r', linestyles='--')
plt.title("Residuals plot of Model B", fontsize=25)
plt.show()


# # Logistic regression
# 

# ## Fitting a logistic model
# The university studying the relationship between hours of study and outcomes on a given test has provided you with a data set containing the number of hours the students studied and whether they failed or passed the test, and asked you to fit a model to predict future performance.
# 
# The data is provided in the variables hours_of_study and outcomes. Use this data to fit a LogisticRegression model. numpy has been imported as np for your convenience.

# ### code

# In[43]:


outcomes= [False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True]
hours_of_study = np.array([[ 4],       [ 8],       [ 8],       [12],       [ 8],       [ 9],       [ 6],       [11],       [13],       [13],       [19],       [16],       [17],       [17],       [21],       [21],       [23],       [27],       [30],       [24]])


# In[44]:


# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# sklearn logistic model
model = LogisticRegression(C=1e9)
model.fit(hours_of_study, outcomes)

# Get parameters
beta1 = model.coef_[0][0]
beta0 = model.intercept_[0]

# Print parameters
print(beta1, beta0)


# ## Predicting if students will pass
# In the previous exercise you calculated the parameters of the logistic regression model that fits the data of hours of study and test outcomes.
# 
# With those parameters you can predict the performance of students based on their hours of study. Use model.predict() to get the outcomes based on the logistic regression.
# 
# For your convenience, LogisticRegression has been imported from sklearn.linear_model and numpy has been imported as np.

# ### code

# In[45]:


# Specify values to predict
hours_of_study_test = [[10], [11], [12], [13], [14]]

# Pass values to predict
predicted_outcomes = model.predict(hours_of_study_test)
print(predicted_outcomes)

# Set value in array
value = np.asarray(11).reshape(-1,1)
# Probability of passing the test with 11 hours of study
print("Probability of passing test ", model.predict_proba(value)[:,1])


# ## Passing two tests
# Put yourself in the shoes of one of the university students. You have two tests coming up in different subjects, and you're running out of time to study. You want to know how much time you have to study each subject to maximize the probability of passing both tests. Fortunately, there's data that you can use.
# 
# For subject A, you already fitted a logistic model in model_A, and for subject B you fitted a model in model_B. As well as preloading LogisticRegression from sklearn.linear_model and numpy as np, expit(), the inverse of the logistic function, has been imported for you from scipy.special.

# ### init

# In[6]:


from scipy.special import expit
import numpy as np

scores_subject_A = [60, 65, 59, 70, 61, 68, 63, 73, 75, 77, 86, 79, 81, 83, 85, 87, 89, 89, 89, 93]
hours_of_study_subject_A=np.array([[ 8],       [ 9],       [ 7],       [12],       [ 6],       [11],       [ 7],       [11],       [13],       [13],       [19],       [16],       [17],       [17],       [16],       [17],       [18],       [16],       [20],       [19]])
scores_subject_B=[60, 65, 59, 70, 61, 68, 63, 73, 75, 77, 86, 79, 81, 83, 85, 87, 89, 89, 89, 93]
hours_of_study_subject_B=np.array([[ 4],       [ 5],       [ 4],       [ 6],       [ 3],       [ 6],       [ 4],       [ 6],       [ 7],       [ 7],       [10],       [ 8],       [ 9],       [ 9],       [ 8],       [ 9],       [ 9],       [ 8],       [10],       [10]])
outcome_A=[False, True, False, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True] 
outcome_B=[False, True, False, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True] 
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression


# sklearn logistic model
model_A = LogisticRegression()
model_A.fit(hours_of_study_subject_A, outcome_A)
model_B = LogisticRegression()
model_B.fit(hours_of_study_subject_B, outcome_B)


# ### code

# In[7]:


# Specify values to predict
hours_of_study_test_A = [[6], [7], [8], [9], [10]]

# Pass values to predict
predicted_outcomes_A = model_A.predict(hours_of_study_test_A)
print(predicted_outcomes_A)

# Specify values to predict
hours_of_study_test_B = [[3], [4], [5], [6]]

# Pass values to predict
predicted_outcomes_B = model_B.predict(hours_of_study_test_B)
print(predicted_outcomes_B)


# In[8]:


#Get the probability of passing for test A with 8.6 hours of study and test B with 4.7 hours of study.

# Set value in array
value_A = np.asarray(8.6).reshape(-1,1)
# Probability of passing test A with 8.6 hours of study
print("The probability of passing test A with 8.6 hours of study is ", model_A.predict_proba(value_A)[:,1])

# Set value in array
value_B = np.asarray(4.7).reshape(-1,1)
# Probability of passing test B with 4.7 hours of study
print("The probability of passing test B with 4.7 hours of study is ", model_B.predict_proba(value_B)[:,1])


# In[9]:


# Calculate the hours you need to study to have 0.5 probability of passing the test using the formula -intercept/slope
# Print the hours required to have 0.5 probability on model_A
print("Minimum hours of study for test A are ", -model_A.intercept_/model_A.coef_)

# Print the hours required to have 0.5 probability on model_B
print("Minimum hours of study for test B are ", -model_B.intercept_/model_B.coef_)


# In[11]:


# Calculate the joint probability of passing test A and test B.
study_hours_A=np.array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,        5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 10.5,       11. , 11.5, 12. , 12.5, 13. , 13.5])
study_hours_B=np.array([14. , 13.5, 13. , 12.5, 12. , 11.5, 11. , 10.5, 10. ,  9.5,  9. ,        8.5,  8. ,  7.5,  7. ,  6.5,  6. ,  5.5,  5. ,  4.5,  4. ,  3.5,        3. ,  2.5,  2. ,  1.5,  1. ,  0.5])

# Probability calculation for each value of study_hours
prob_passing_A = model_A.predict_proba(study_hours_A.reshape(-1,1))[:,1]
prob_passing_B = model_B.predict_proba(study_hours_B.reshape(-1,1))[:,1]

# Calculate the probability of passing both tests
prob_passing_A_and_B = prob_passing_A * prob_passing_B

# Maximum probability value
max_prob = max(prob_passing_A_and_B)

# Position where we get the maximum value
max_position = np.where(prob_passing_A_and_B == max_prob)[0][0]

# Study hours for each test
print("Study {:1.0f} hours for the first and {:1.0f} hours for the second test and you will pass both tests with {:01.2f} probability.".format(study_hours_A[max_position], study_hours_B[max_position], max_prob))


# In[ ]:




