# -*- coding: utf-8 -*-
"""
chapter 4
Created on Wed Jul 31 08:44:11 2019

@author: N561507
"""

#%% Exercise - Fit a multivariable logistic regression - init
import pandas as pd
crab=pd.read_csv('crab.csv', index_col=0)


#%% Exercise - Fit a multivariable logistic regression
# Import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import glm

# Define model formula
formula = 'y ~ width + color'

# Fit GLM
model = glm(formula, data = crab, family = sm.families.Binomial()).fit()

# Print model summary
print(model.summary())

#%% Exercise - The effect of multicollinearity - init
import pandas as pd
crab=pd.read_csv('crab.csv', index_col=0)

#%% Exercise - The effect of multicollinearity
# Import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import glm

# Define model formula
formula = 'y ~ weight + width'

# Fit GLM
model = glm(formula, data = crab, family = sm.families.Binomial()).fit()

# Print model summary
print(model.summary())

#%% Exercise - Compute VIF - init
import pandas as pd
crab=pd.read_csv('crab.csv', index_col=0)
import statsmodels.api as sm
from statsmodels.formula.api import glm
# Define model formula
formula = 'y ~ weight + width'
# Fit GLM
model = glm(formula, data = crab, family = sm.families.Binomial()).fit()


#%% Exercise - Compute VIF
# Import functions
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term
X = crab[['weight', 'width', 'color']]
X['Intercept'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# View results using print
print(vif)

#%% Exercise - Checking model fit - init
import pandas as pd
wells=pd.read_csv('wells.csv', index_col=0)

#%% Exercise - Checking model fit
# Import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import glm

# Define model formula
formula = 'switch ~ distance100 + arsenic'

# Fit GLM
model_dist_ars = glm(formula, data = wells, family = sm.families.Binomial()).fit()

# Compare deviance of null and residual model
diff_deviance = model_dist_ars.null_deviance - model_dist_ars.deviance

# Print the computed difference in deviance
print(diff_deviance)

#%% Exercise - Compare two models - init
import pandas as pd
wells=pd.read_csv('wells.csv', index_col=0)
# Define model formula
formula_dist_ars = 'switch ~ distance100 + arsenic'
formula_dist = 'switch ~ distance100'
# Fit GLM
model_dist_ars = glm(formula_dist_ars, data = wells, family = sm.families.Binomial()).fit()
model_dist = glm(formula_dist, data = wells, family = sm.families.Binomial()).fit()


#%% Exercise - Compare two models
# Compute the difference in adding distance100 variable
diff_deviance_distance = model_dist.null_deviance - model_dist.deviance

# Print the computed difference in deviance
print('Adding distance100 to the null model reduces deviance by: ', 
      round(diff_deviance_distance,3))

# Compute the difference in adding arsenic variable
diff_deviance_arsenic = model_dist.deviance - model_dist_ars.deviance

# Print the computed difference in deviance
print('Adding arsenic to the distance model reduced deviance further by: ', 
      round(diff_deviance_arsenic,3))


#%% Exercise - Deviance and linear transformation - init
import pandas as pd
wells=pd.read_csv('wells.csv', index_col=0)
# Define model formula
formula_dist = 'switch ~ distance100'
# Fit GLM
model_dist = glm(formula_dist, data = wells, family = sm.families.Binomial()).fit()



#%% Exercise - Deviance and linear transformation
# Import functions
import statsmodels.api as sm
from statsmodels.formula.api import glm

# Fit logistic regression model as save as model_dist_1
model_dist_1 = glm('switch ~ distance', data = wells, family = sm.families.Binomial()).fit()

# Check the difference in deviance of model_dist_1 and model_dist
print('Difference in deviance is: ', round(model_dist_1.deviance - model_dist.deviance,3))

#%% Exercise - Model matrix for continuous variables - init
import pandas as pd
wells=pd.read_csv('wells.csv', index_col=0)
# Define model formula
formula = 'switch ~ arsenic'
# Fit GLM
model = glm(formula, data = wells, family = sm.families.Binomial()).fit()

#%% Exercise - Model matrix for continuous variables
# Import function dmatrix()
from patsy import dmatrix

# Construct model matrix with arsenic
model_matrix = dmatrix('arsenic', data = wells, return_type = 'dataframe')
print(model_matrix.head())

# Construct model matrix with arsenic and distance100
model_matrix = dmatrix('arsenic + distance100', data = wells, return_type = 'dataframe')
print(model_matrix.head())

#%% Exercise - Variable transformation - init
import pandas as pd
wells=pd.read_csv('wells.csv', index_col=0)
# Define model formula
formula = 'switch ~ arsenic'
# Fit GLM
model_ars = glm(formula, data = wells, family = sm.families.Binomial()).fit()

#%% Exercise - Variable transformation
import numpy as np
from patsy import dmatrix

# Construct model matrix for arsenic with log transformation
dmatrix('np.log(arsenic)', data = wells,
       return_type = 'dataframe').head()

# Import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import glm
import numpy as np

# Define model formula
formula = 'switch ~ np.log(arsenic)'

# Fit GLM
model_log_ars = glm(formula, data = wells, 
                     family = sm.families.Binomial()).fit()

# Print model summary
print(model_log_ars.summary())

#%% Exercise - Coding categorical variables - init
import pandas as pd
crab=pd.read_csv('crab.csv', index_col=0)

#%% Exercise - Coding categorical variables
# Import function dmatrix
from patsy import dmatrix

# Construct and print model matrix for color as categorical variable
print(dmatrix('C(color)', data = crab,
     	   return_type = 'dataframe').head())
# Construct and print the model matrix for color with reference group 3
print(dmatrix('C(color, Treatment(3))', 
     	  data = crab,
     	  return_type = 'dataframe').head())

#%%Exercise - Modeling with categorical variable
# Construct model matrix
model_matrix = dmatrix('C(color, Treatment(4))' , data = crab, 
                       return_type = 'dataframe')

# Print first 5 rows of model matrix dataframe
print(model_matrix.head())

# Fit and print the results of a glm model with the above model matrix configuration
model = glm('y ~ C(color, Treatment(4))', data = crab, 
            family = sm.families.Binomial()).fit()

print(model.summary())

# Construct model matrix
model_matrix = dmatrix('C(color, Treatment(4)) + width' , data = crab, 
                       return_type = 'dataframe')

# Print first 5 rows of model matrix
print(model_matrix.head())

# Fit and print the results of a glm model with the above model matrix configuration
model = glm('y ~ C(color, Treatment(4)) + width', data = crab, 
            family = sm.families.Binomial()).fit()

print(model.summary())

#%% Exercise - Interaction terms - init
import pandas as pd
wells=pd.read_csv('wells.csv', index_col=0)
# Define model formula
formula_dist_ars = 'switch ~ distance100 + arsenic'
# Fit GLM
model_dist_ars = glm(formula_dist_ars, data = wells, family = sm.families.Binomial()).fit()


#%% Exercise - Interaction terms
# Import libraries
import statsmodels.api as sm
from statsmodels.formula.api import glm

# Fit GLM and print model summary
model_int = glm('switch ~ center(distance100) + center(arsenic) + center(distance100):center(arsenic)', 
                data = wells, family = sm.families.Binomial()).fit()

# View model results
print(model_int.summary())