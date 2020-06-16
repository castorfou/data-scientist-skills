#!/usr/bin/env python
# coding: utf-8

# # working case

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import matplotlib as mpl


import mlflow
import mlflow.xgboost

mpl.use('Agg')
# prepare train and test data
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# enable auto logging
mlflow.xgboost.autolog(importance_types=['weight', 'gain'])

with mlflow.start_run():

    # train model
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'learning_rate': 0.7,
        'eval_metric': 'mlogloss',
        'colsample_bytree': 1.0,
        'subsample': 1,
        'seed': 42,
    }
    model = xgb.train(params, dtrain,10,  evals=[(dtrain, 'train')])

    # evaluate model
    y_proba = model.predict(dtest)
    y_pred = y_proba.argmax(axis=1)
    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    # log metrics
    mlflow.log_metrics({'log_loss': loss, 'accuracy': acc})


# # non working case

# In[6]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import matplotlib as mpl


import mlflow
import mlflow.xgboost

#mlflow.set_tracking_uri('file:models_mlflow/mlruns')
mlflow.set_tracking_uri('file:models_mlflow/mlruns')

mpl.use('Agg')
# prepare train and test data
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# enable auto logging
mlflow.xgboost.autolog(importance_types=['weight', 'gain'])

with mlflow.start_run():

    # train model
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'learning_rate': 0.7,
        'eval_metric': 'mlogloss',
        'colsample_bytree': 1.0,
        'subsample': 1,
        'seed': 42,
    }
    model = xgb.train(params, dtrain,10,  evals=[(dtrain, 'train')])

    # evaluate model
    y_proba = model.predict(dtest)
    y_pred = y_proba.argmax(axis=1)
    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    # log metrics
    mlflow.log_metrics({'log_loss': loss, 'accuracy': acc})


# In[ ]:



