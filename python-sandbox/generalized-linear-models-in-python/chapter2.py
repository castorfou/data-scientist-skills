# -*- coding: utf-8 -*-
"""
chapter 2
Created on Mon Jul 29 21:58:52 2019

@author: N561507
"""

#%% Exercise - Fit logistic regression - init
import pandas as pd
wells=pd.read_csv('wells.csv', index_col=0)

#%% Exercise - Fit logistic regression
# Load libraries and functions
import statsmodels.api as sm
from statsmodels.formula.api import glm

# Fit logistic regression model
model_GLM = glm(formula = 'switch ~ arsenic',
                data = wells,
                family = sm.families.Binomial()).fit() 

# Print model summary
print(model_GLM.summary())

#%% Exercise - Coefficients in terms of odds
# Load libraries and functions
import statsmodels.api as sm
from statsmodels.formula.api import glm
import numpy as np

# Fit logistic regression model
model_GLM = glm(formula = 'switch ~ distance100',
                data = wells,
                family = sm.families.Binomial()).fit() 

# Extract model coefficients
print('Model coefficients: \n', model_GLM.params)

# Compute the multiplicative effect on the odds
print('Odds: \n', np.exp(model_GLM.params))

#%% Exercise - Rate of change in probability - init
wells_GLM=model_GLM

#%% Exercise - Rate of change in probability
# Define x at 1.5
x = 1.5

# Extract intercept & slope from the fitted model
intercept, slope = wells_GLM.params

# Compute and print the estimated probability
est_prob = np.exp(intercept + slope*x)/(1+np.exp(intercept + slope*x))
print('Estimated probability at x = 1.5: ', round(est_prob, 4))

# Compute the slope of the tangent line for parameter beta at x
slope_tan = slope * est_prob * (1 - est_prob)
print('The rate of change in probability: ', round(slope_tan,4))

#%% Exercise - Statistical significance - init
import pandas as pd
crab=pd.read_csv('crab.csv', index_col=0)

#%% Exercise - Statistical significance
# Import libraries and th glm function
import statsmodels.api as sm
from statsmodels.formula.api import glm

# Fit logistic regression and save as crab_GLM
crab_GLM = glm('y ~ width', data = crab, family = sm.families.Binomial()).fit()

# Print model summary
print(crab_GLM.summary())

#%% Exercise - Computing Wald statistic
# Extract coefficients
intercept, slope = crab_GLM.params

# Estimated covariance matrix: crab_cov
crab_cov = crab_GLM.cov_params()
print(crab_cov)

# Compute standard error (SE): std_error
std_error = np.sqrt(crab_cov.loc['width', 'width'])
print('SE: ', round(std_error, 4))

# Compute Wald statistic
wald_stat = slope/std_error
print('Wald statistic: ', round(wald_stat,4))

#%% Exercise - Confidence intervals
# Extract and print confidence intervals
print(crab_GLM.conf_int())

# Compute confidence intervals for the odds
print(np.exp(crab_GLM.conf_int()))

#%% Exercise - Visualize model fit using regplot() - init
import seaborn as sns
import matplotlib.pyplot as plt

#%% Exercise - Visualize model fit using regplot()
# Plot distance and switch and add overlay with the logistic fit
sns.regplot(x = 'arsenic', y = 'switch', 
            y_jitter = 0.03,
            data = wells, 
            logistic = True,
            ci = None)

# Display the plot
plt.show()

#%% Exercise - Compute predictions - init
#wells_test.to_csv('wells_test.csv')
#!curl -F "file=@wells_test.csv" https://file.io
#{"success":true,"key":"elqnww","link":"https://file.io/elqnww","expiry":"14 days"}
wells_test=pd.read_csv('wells_test.csv', index_col=0)
wells_fit=model_GLM

#%% Exercise - Compute predictions
# Compute predictions for the test sample wells_test and save as prediction
prediction = wells_fit.predict(exog = wells_test)

# Add prediction to the existing data frame wells_test and assign column name prediction
wells_test['prediction'] = prediction

# Examine the first 5 computed predictions
print(wells_test[['switch', 'arsenic', 'prediction']].head())

#%% Exercise - Compute confusion matrix - init
import numpy as np

#%% Exercise - Compute confusion matrix
# Define the cutoff
cutoff = 0.5

# Compute class predictions: y_prediction
y_prediction = np.where(prediction > cutoff, 1, 0)

# Assign actual class labels from the test sample to y_actual
y_actual = wells_test['switch']

# Compute and print confusion matrix using crosstab function
conf_mat = pd.crosstab(y_actual, y_prediction, 
                       rownames=['Actual'], 
                       colnames=['Predicted'], 
                       margins = True)
                      
# Print the confusion matrix
print(conf_mat)

