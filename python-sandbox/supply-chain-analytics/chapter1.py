# -*- coding: utf-8 -*-
"""
Chapter1
Created on Thu Jul 25 08:29:09 2019

@author: N561507
"""

#%% PuLP Example – Resource Scheduling

from pulp import *
# Initialize Class
model = LpProblem("Maximize Bakery Profits", LpMaximize)
# Define Decision Variables
A = LpVariable('A', lowBound=0, cat='Integer')
B = LpVariable('B', lowBound=0, cat='Integer')
# Define Objective Function
model += 20 * A + 40 * B
# Define Constraints
model += 0.5 * A + 1 * B <= 30
model += 1 * A + 2.5 * B <= 60
model += 1 * A + 2 * B <= 22
# Solve Model
model.solve()
print("Produce {} Cake A".format(A.varValue))
print("Produce {} Cake B".format(B.varValue))

#%% Exercice - Simple Resource Scheduling Exercise - init
from pulp import *

#%% Exercice - Simple Resource Scheduling Exercise
# Initialize Class
model = LpProblem("Maximize Glass Co. Profits", LpMaximize)

# Define Decision Variables
wine = LpVariable('Wine', lowBound=0, upBound=None, cat='Integer')
beer = LpVariable('Beer', lowBound=0, upBound=None, cat='Integer')

# Define Objective Function
model += 5 * wine + 4.5 * beer

# Define Constraints
model += 6 * wine + 5 * beer <= 60
model += 10 * wine + 20 * beer <= 150
model += wine <= 6

# Solve Model
model.solve()
print("Produce {} batches of wine glasses".format(wine.varValue))
print("Produce {} batches of beer glasses".format(beer.varValue))

#%% - lpSum with List Comprehension
# Define Objective Function
cake_types = ["A", "B", "C", "D", "E", "F"]
profit_by_cake = {"A":20, "B":40, "C":33, "D":14, "E":6, "F":60}
var_dict = {"A":A, "B":B, "C":C, "D":D, "E":E, "F":F}
model += lpSum([profit_by_cake[type] * var_dict[type] for type in cake_types])

#%% - Exercise - Trying out lpSum - init
from pulp import *
ingredient= ['cream', 'milk', 'sugar']
prod_type=['premium', 'budget']
var_dict={('premium', 'cream'): cp, ('premium', 'milk'): mp, ('premium', 'sugar'): sp, ('budget', 'cream'): cb, ('budget', 'milk'): mb, ('budget', 'sugar'): sb}


#%% - Exercise - Trying out lpSum
# Define Objective Function
model += lpSum([1.5 * var_dict[(i, 'cream')] + 0.125 * var_dict[(i, 'milk')] + 0.10 * var_dict[(i, 'sugar')] for i in prod_type])


#%% Exercise - Logistics Planning Problem - init
from pulp import *
costs = {('New York', 'East'): 211, ('New York', 'South'): 232, ('New York', 'Midwest'): 240, ('New York', 'West'): 300, ('Atlanta', 'East'): 232, ('Atlanta', 'South'): 212, ('Atlanta', 'Midwest'): 230, ('Atlanta', 'West'): 280}
var_dict = {('New York', 'East'): ne, ('New York', 'South'): ns, ('New York', 'Midwest'): nm, ('New York', 'West'): nw, ('Atlanta', 'East'): atle, ('Atlanta', 'South'): atls, ('Atlanta', 'Midwest'): atlm, ('Atlanta', 'West'): atlw}

#%% Exercise - Logistics Planning Problem
# Initialize Model
model = LpProblem("Minimize Transportation Costs", LpMinimize)

# Build the lists and the demand dictionary
warehouse = ['New York', 'Atlanta']
customers = ['East', 'South', 'Midwest', 'West']
regional_demand = [1800, 1200, 1100, 1000]
demand = dict(zip(customers, regional_demand))

# Define Objective
model += lpSum([costs[(w, c)] * var_dict[(w, c)] for c in customers for w in warehouse])

# For each customer, sum warehouse shipments and set equal to customer demand
for c in customers:
    model += lpSum([var_dict[(w, c)] for w in warehouse]) == demand[c]