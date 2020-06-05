#!/usr/bin/env python
# coding: utf-8

# # test basique

# In[4]:


import os
from mlflow import log_metric, log_param, log_artifact, end_run

# Log a parameter (key-value pair)
log_param("param1", 5)

# Log a metric; metrics can be updated throughout the run
log_metric("foo", 1)
log_metric("foo", 2)
log_metric("foo", 3)

# Log an artifact (output file)
with open("output.txt", "w") as f:
    f.write("Hello world!")
log_artifact("output.txt")
end_run()


# # log structure

# In[12]:


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
    
    importance_types=['weight', 'gain']
    for imp_type in importance_types:
        imp = model.get_score(importance_type=imp_type)
        features, importance = zip(*imp.items())
        print(imp_type, features, importance)


# # test scikit-learn

# https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html

# In[5]:


import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



warnings.filterwarnings("ignore")
np.random.seed(40)

# Read the wine-quality csv file from the URL
csv_url =    'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
try:
    data = pd.read_csv(csv_url, sep=';')
except Exception as e:
    logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s", e)

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

alpha = 0.6
l1_ratio = 0.55

with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
    else:
        mlflow.sklearn.log_model(lr, "model")


# # 2e test

# In[14]:


import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
y = np.array([0, 0, 1, 1, 1, 0])
lr = LogisticRegression()
lr.fit(X, y)
score = lr.score(X, y)
print("Score: %s" % score)

mlflow.set_tracking_uri('file:/mnt/z/temp/mlruns')
mlflow.set_experiment('Dureté Shore')

mlflow.log_param("learning_rate", "0.7")
mlflow.set_tag("source","SCML,METEO,QMP")
mlflow.set_tag("notebook","76")

mlflow.log_metric("score", score)
mlflow.sklearn.log_model(lr, "model")
print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
mlflow.end_run()


# In[6]:





# In[4]:


mlflow.active_run()


# In[ ]:




