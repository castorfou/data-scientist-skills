# -*- coding: utf-8 -*-
"""
chapter 1
Created on Mon Jul 29 12:40:40 2019

@author: N561507
"""

#%% Exercice - Linear model, a special case of GLM - init
import pandas as pd
salary=pd.read_csv('salary.csv', index_col=0)


#%% Exercice - Linear model, a special case of GLM

import statsmodels.api as sm
from statsmodels.formula.api import ols, glm

# Fit a linear model
model_lm = ols(formula = 'Salary ~ Experience',
               data = salary).fit()

# View model coefficients
print(model_lm.params)

# Fit a GLM
model_glm = glm(formula = 'Salary ~ Experience',
                data = salary,
                family = sm.families.Gaussian()).fit()

# View model coefficients
print(model_glm.params)

#%% Exercise - Linear model and a binary response variable - init
#import pandas as pd
#crab.to_csv('crab.csv')
#!curl --upload-file ./crab.csv https://transfer.sh/crab.csv 
#https://transfer.sh/fRqAY/crab.csv
import pandas as pd
crab=pd.read_csv('crab.csv', index_col=0)


#%% Exercise - Linear model and a binary response variable
# Define model formula
formula = 'y ~ width'

# Define probability distribution for the response variable for 
# the linear (LM) and logistic (GLM) model
family_LM = sm.families.Gaussian()
family_GLM = sm.families.Binomial()

# Define and fit a linear regression model
model_LM = glm(formula = formula, data = crab, family = family_LM).fit()
print(model_LM.summary())

# Define and fit a logistic regression model
model_GLM = glm(formula = formula, data = crab, family = family_GLM).fit()
print(model_GLM.summary())

#%% Exercise - Comparing predicted values - init
width_init=[17.8,24.6,28.1,32.0,33.7]
y_init=[0,0,1,1,1]
test=pd.DataFrame(list(zip(width_init,y_init)),columns=['width','y'])

#%% Exercise - Comparing predicted values - init
# View test set
print(test)

# Compute estimated probabilities for linear model: pred_lm
pred_lm = model_LM.predict(test)

# Compute estimated probabilities for GLM model: pred_glm
pred_glm = model_GLM.predict(test)

# Create dataframe of predictions for linear and GLM model: predictions
predictions = pd.DataFrame({'Pred_LM': pred_lm, 'Pred_GLM': pred_glm})

# Concatenate test sample and predictions and view the results
all_data = pd.concat([test, predictions], axis = 1)
print(all_data)

#%% Exercise - Model fitting step-by-step - init
#import pandas as pd
#wells.to_csv('wells.csv')
#!curl --upload-file ./wells.csv https://transfer.sh/wells.csv
#!curl -F "file=@wells.csv" https://file.io

import pandas as pd
wells=pd.read_csv('wells.csv', index_col=0)
from statsmodels.formula.api import glm
import statsmodels.api as sm


#%% Exercise - Model fitting step-by-step
# Define the formula the the logistic model
model_formula = 'switch ~ distance100'

# Define the correct probability distribution and the link function of the response variable
link_function = sm.families.links.logit
model_family = sm.families.Binomial(link = link_function)

# Fit the model
wells_fit = glm(formula = model_formula, 
                 data = wells, 
                 family = model_family).fit()

#%% Exercise - Results of the model fit using summary()
# View the results of the wells_fit model
print(wells_fit.summary())

#%% Exercise - Extracting parameter estimates
# Extract coefficients from the fitted model wells_fit
intercept, slope = wells_fit.params

# Print coefficients
print('Intercept =', intercept)
print('Slope =', slope)

# Extract and print confidence intervals
print(wells_fit.conf_int())
