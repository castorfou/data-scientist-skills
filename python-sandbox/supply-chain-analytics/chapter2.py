# -*- coding: utf-8 -*-
"""
chapter2
Created on Thu Jul 25 09:06:12 2019

@author: N561507
"""

#%% LpVariable.dicts with List Comprehension
from pulp import *

#Transportation Optimization
# Define Decision Variables
customers = ['East','South','Midwest','West']
warehouse = ['New York','Atlanta']
transport = LpVariable.dicts("route", [(w,c) for w in warehouse for c in customers], lowBound=0, cat='Integer')
# Define Objective
model += lpSum([cost[(w,c)]*transport[(w,c)] for w in warehouse for c in customers])

#%% Exercise - Logistics Planning Problem 2 - init
from pulp import *
costs = {('New York', 'East'): 211, ('New York', 'South'): 232, ('New York', 'Midwest'): 240, ('New York', 'West'): 300, ('Atlanta', 'East'): 232, ('Atlanta', 'South'): 212, ('Atlanta', 'Midwest'): 230, ('Atlanta', 'West'): 280}
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
warehouse = ['New York', 'Atlanta']
customers  = ['East', 'South', 'Midwest', 'West']

#%% Exercise - Logistics Planning Problem 2
# Define decision variables
key = [(m, w, c) for m in months for w in warehouse for c in customers]
var_dict = LpVariable.dicts('num_of_shipments', key, lowBound=0, cat='Integer')

# Use the LpVariable dictionary variable to define objective
model += lpSum([costs[(w, c)] * var_dict[(m, w, c)] for m in months for w in warehouse for c in customers])

#%% Exercise - Travelling Salesman Problem (TSP) - init
from pulp import *
import pandas as pd
n = 15
cities = range(0,15)
#!curl --upload-file ./dist.csv https://transfer.sh/dist.csv 
dist = pd.read_csv('dist.csv',index_col=0)

#%% Exercise - Travelling Salesman Problem (TSP)
# Define Decision Variables
x = LpVariable.dicts('X', [(c1, c2) for c1 in cities for c2 in cities], cat='Binary')
u = LpVariable.dicts('U', [c1 for c1 in cities], lowBound=0, upBound=(n-1), cat='Integer')

# Define Objective
model += lpSum([dist.iloc[c1, c2] * x[(c1, c2)] for c1 in cities for c2 in cities])
# Define Constraints
for c2 in cities:
    model += lpSum([x[(c1, c2)] for c1 in cities]) == 1
for c1 in cities:
    model += lpSum([x[(c1, c2)] for c2 in cities]) == 1
    
#%% Exercise - Scheduling Workers Problem    
from pulp import *

# The class has been initialize, and days defined
model = LpProblem("Minimize Staffing", LpMinimize)
days = list(range(7))

# Define Decision Variables
x = LpVariable.dicts('staff_', days, lowBound=0, cat='Integer')
model += lpSum([x[i] for i in days])

# Define Constraints
model += x[0] + x[3] + x[4] + x[5] + x[6] >= 31
model += x[0] + x[1] + x[4] + x[5] + x[6] >= 45
model += x[0] + x[1] + x[2] + x[5] + x[6] >= 40
model += x[0] + x[1] + x[2] + x[3] + x[6] >= 40
model += x[0] + x[1] + x[2] + x[3] + x[4] >= 48
model += x[1] + x[2] + x[3] + x[4] + x[5] >= 30
model += x[2] + x[3] + x[4] + x[5] + x[6] >= 25

model.solve()


#%%Capacitated Plant Location Model
from pulp import *
# Initialize Class
model = LpProblem("Capacitated Plant Location Model", LpMinimize)
# Define Decision Variables
loc = ['A', 'B', 'C', 'D', 'E']
size = ['Low_Cap','High_Cap']
x = LpVariable.dicts("production",[(i,j) for i in loc for j in loc],lowBound=0, upBound=None, cat='Continous')
y = LpVariable.dicts("plant",[(i,s) for s in size for i in loc], cat='Binary')
# Define objective function
model += (lpSum([fix_cost.loc[i,s]*y[(i,s)] for s in size for i in loc])+ lpSum([var_cost.loc[i,j]*x[(i,j)] for i in loc for j in loc]))

