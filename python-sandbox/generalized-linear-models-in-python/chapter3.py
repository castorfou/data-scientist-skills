# -*- coding: utf-8 -*-
"""
chapter3
Created on Tue Jul 30 15:25:29 2019

@author: N561507
"""

#%% Exercise - Visualize the response - init
import pandas as pd
crab=pd.read_csv('crab.csv', index_col=0)


#%% Exercise - Visualize the response
# Import libraries
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt

# Plot sat variable
sns.distplot(crab['sat'])

# Display the plot
plt.show()

#%% Exercise - Fitting a Poisson regression
# Import libraries
import statsmodels.api as sm
from statsmodels.formula.api import glm

# Fit Poisson regression of sat by weight
model = glm('sat ~ weight', data = crab, family = sm.families.Poisson()).fit()

# Display model results
print(model.summary())

#%% Exercise - Estimate parameter lambda

# Import libraries
import statsmodels.api as sm
from statsmodels.formula.api import glm

# Fit Poisson regression of sat by width
model =glm('sat ~ width', data = crab, family = sm.families.Poisson()).fit()

# Display model results
print(model.summary())

# Compute average crab width
mean_width = np.mean(crab['width'])

# Print the compute mean
print('Average width: ', round(mean_width, 3))

# Extract coefficients
intercept, slope = model.params

# Compute the estimated mean of y (lambda) at the average width
est_lambda = np.exp(intercept) * np.exp(slope * mean_width)

# Print estimated mean of y
print('Estimated mean of y at average width: ', round(est_lambda, 3))

#%% Exercise - Is the mean equal to the variance?

# Compute and print sample mean of the number of satellites: sat_mean
sat_mean = np.mean(crab.sat)

print('Sample mean:', round(sat_mean, 3))

# Compute and print sample variance of the number of satellites: sat_var
sat_var = np.var(crab.sat)
print('Sample variance:', round(sat_var, 3))

# Compute ratio of variance to mean
print('Ratio:', round(sat_var/sat_mean, 3))

#%% Exercise - Computing expected number of counts - init
import math
#%% Exercise - Computing expected number of counts
# Expected number of zero counts
exp_zero_cnt = ((sat_mean**0)*np.exp(-sat_mean))/math.factorial(0)

# Print exp_zero_counts
print('Expected zero counts given mean of ', round(sat_mean,3), 
      'is ', round(exp_zero_cnt,3)*100)

# Number of zero counts in sat variable
actual_zero_ant = sum(crab['sat']  == 0)

# Number of observations in crab dataset
num_obs = len(crab['sat'])

# Print the percentage of zero count observations in the sample
print('Actual zero counts in the sample: ', round(actual_zero_ant/ num_obs,3)*100)


#%% Exercise - Checking for overdispersion - init
crab_pois = model
from statsmodels.formula.api import glm

#%% Exercise - Checking for overdispersion
# Compute and print the overdispersion approximation
print(crab_pois.pearson_chi2/ crab_pois.df_resid)

#%% Exercise - Fitting negative binomial
# Define the formula for the model fit
formula = 'sat ~ width'

# Fit the GLM negative binomial model using log link function
crab_NB = glm(formula = formula, data = crab, 
				  family = sm.families.NegativeBinomial()).fit()

# Print Poisson model's summary
print(crab_pois.summary())

# Print the negative binomial model's summary
print(crab_NB.summary())

#%% Exercise - Confidence intervals for negative Binomial model
# Compute confidence intervals for crab_Pois model
print('Confidence intervals for the Poisson model')
print(crab_pois.conf_int())

# Compute confidence intervals for crab_NB model
print('Confidence intervals for the Negative Binomial model')
print(crab_NB.conf_int())

#%%Exercise - Exercise - Plotting data and linear model fit

# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the data points and linear model fit
sns.regplot('width', 'sat', data = crab,
            y_jitter = 0.3,
            fit_reg = True,
            line_kws = {'color':'green', 
                        'label':'LM fit'})

# Print plot
plt.show()

#%% Exercise - Plotting fitted values - init
model = crab_pois

#%% Exercise - Plotting fitted values

# Add fitted values to the fit_values column of crab dataframe
crab['fit_values'] = model.fittedvalues

# Plot data points
sns.regplot('width', 'sat', data = crab,
            y_jitter = 0.3,
            fit_reg = True, 
            line_kws = {'color':'green', 
                        'label':'LM fit'})

# Poisson regression fitted values
sns.scatterplot('width','fit_values', data = crab,
           color = 'red', label = 'Poisson')

# Print plot          
plt.show()

