import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn import datasets
import xgboost as xgb
import matplotlib as mpl

import mlflow
import mlflow.xgboost

mpl.use('Agg')


def get_latest_run():
    client = mlflow.tracking.MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id='0')[0].run_id)


@pytest.fixture(scope="session")
def bst_params():
    return {
        'objective': 'multi:softprob',
        'num_class': 3,
    }


@pytest.fixture(scope="session")
def dtrain():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    return xgb.DMatrix(X, y)


@pytest.mark.large
def test_xgb_autolog_ends_auto_created_run(bst_params, dtrain):
    mlflow.xgboost.autolog()
    xgb.train(bst_params, dtrain)
    assert mlflow.active_run() is None


@pytest.mark.large
def test_xgb_autolog_persists_manually_created_run(bst_params, dtrain):
    mlflow.xgboost.autolog()
    with mlflow.start_run() as run:
        xgb.train(bst_params, dtrain)
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.mark.large
def test_xgb_autolog_logs_default_params(bst_params, dtrain):
    mlflow.xgboost.autolog()
    xgb.train(bst_params, dtrain)
    run = get_latest_run()
    params = run.data.params

    expected_params = {
        'num_boost_round': 10,
        'maximize': False,
        'early_stopping_rounds': None,
        'verbose_eval': True,
    }
    expected_params.update(bst_params)

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    unlogged_params = ['dtrain', 'evals', 'obj', 'feval', 'evals_result',
                       'xgb_model', 'callbacks', 'learning_rates']

    for param in unlogged_params:
        assert param not in params


@pytest.mark.large
def test_xgb_autolog_logs_specified_params(bst_params, dtrain):
    mlflow.xgboost.autolog()
    expected_params = {
        'num_boost_round': 20,
        'early_stopping_rounds': 5,
        'verbose_eval': False,
    }
    xgb.train(bst_params, dtrain, evals=[(dtrain, 'train')], **expected_params)
    run = get_latest_run()
    params = run.data.params

    expected_params.update(bst_params)

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    unlogged_params = ['dtrain', 'evals', 'obj', 'feval', 'evals_result',
                       'xgb_model', 'callbacks', 'learning_rates']

    for param in unlogged_params:
        assert param not in params


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_validation_data(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    xgb.train(bst_params, dtrain, num_boost_round=20,
              evals=[(dtrain, 'train')], evals_result=evals_result)
    run = get_latest_run()
    data = run.data
    metric_key = 'train-merror'
    client = mlflow.tracking.MlflowClient()
    metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
    assert metric_key in data.metrics
    assert len(metric_history) == 20
    assert metric_history == evals_result['train']['merror']


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_multi_validation_data(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    evals = [(dtrain, 'train'), (dtrain, 'valid')]
    xgb.train(bst_params, dtrain, num_boost_round=20, evals=evals, evals_result=evals_result)
    run = get_latest_run()
    data = run.data
    client = mlflow.tracking.MlflowClient()
    for eval_name in [e[1] for e in evals]:
        metric_key = '{}-merror'.format(eval_name)
        metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
        assert metric_key in data.metrics
        assert len(metric_history) == 20
        assert metric_history == evals_result[eval_name]['merror']


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_multi_metrics(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    params = {'eval_metric': ['merror', 'mlogloss']}
    params.update(bst_params)
    xgb.train(params, dtrain, num_boost_round=20,
              evals=[(dtrain, 'train')], evals_result=evals_result)
    run = get_latest_run()
    data = run.data
    client = mlflow.tracking.MlflowClient()
    for metric_name in params['eval_metric']:
        metric_key = 'train-{}'.format(metric_name)
        metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
        assert metric_key in data.metrics
        assert len(metric_history) == 20
        assert metric_history == evals_result['train'][metric_name]


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_multi_validation_data_and_metrics(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    params = {'eval_metric': ['merror', 'mlogloss']}
    params.update(bst_params)
    evals = [(dtrain, 'train'), (dtrain, 'valid')]
    xgb.train(params, dtrain, num_boost_round=20, evals=evals, evals_result=evals_result)
    run = get_latest_run()
    data = run.data
    client = mlflow.tracking.MlflowClient()
    for eval_name in [e[1] for e in evals]:
        for metric_name in params['eval_metric']:
            metric_key = '{}-{}'.format(eval_name, metric_name)
            metric_history = [x.value for x
                              in client.get_metric_history(run.info.run_id, metric_key)]
            assert metric_key in data.metrics
            assert len(metric_history) == 20
            assert metric_history == evals_result[eval_name][metric_name]


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_early_stopping(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    params = {'eval_metric': ['merror', 'mlogloss']}
    params.update(bst_params)
    evals = [(dtrain, 'train'), (dtrain, 'valid')]
    model = xgb.train(params, dtrain, num_boost_round=20, early_stopping_rounds=5,
                      evals=evals, evals_result=evals_result)
    run = get_latest_run()
    data = run.data

    assert 'best_iteration' in data.metrics
    assert int(data.metrics['best_iteration']) == model.best_iteration
    assert 'stopped_iteration' in data.metrics
    assert int(data.metrics['stopped_iteration']) == len(evals_result['train']['merror']) - 1
    client = mlflow.tracking.MlflowClient()

    for eval_name in [e[1] for e in evals]:
        for metric_name in params['eval_metric']:
            metric_key = '{}-{}'.format(eval_name, metric_name)
            metric_history = [x.value for x
                              in client.get_metric_history(run.info.run_id, metric_key)]
            assert metric_key in data.metrics
            assert len(metric_history) == 20 + 1

            best_metrics = evals_result[eval_name][metric_name][model.best_iteration]
            assert metric_history == evals_result[eval_name][metric_name] + [best_metrics]


@pytest.mark.large
def test_xgb_autolog_logs_feature_importance(bst_params, dtrain):
    mlflow.xgboost.autolog()
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace('file://', '')
    client = mlflow.tracking.MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    importance_type = 'weight'
    plot_name = 'feature_importance_{}.png'.format(importance_type)
    assert plot_name in artifacts

    json_name = 'feature_importance_{}.json'.format(importance_type)
    assert json_name in artifacts

    json_path = os.path.join(artifacts_dir, json_name)
    with open(json_path, 'r') as f:
        loaded_imp = json.load(f)

    assert loaded_imp == model.get_score(importance_type=importance_type)


@pytest.mark.large
def test_xgb_autolog_logs_specified_feature_importance(bst_params, dtrain):
    importance_types = ['weight', 'total_gain']
    mlflow.xgboost.autolog(importance_types)
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace('file://', '')
    client = mlflow.tracking.MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    for imp_type in importance_types:
        plot_name = 'feature_importance_{}.png'.format(imp_type)
        assert plot_name in artifacts

        json_name = 'feature_importance_{}.json'.format(imp_type)
        assert json_name in artifacts

        json_path = os.path.join(artifacts_dir, json_name)
        with open(json_path, 'r') as f:
            loaded_imp = json.load(f)

        assert loaded_imp == model.get_score(importance_type=imp_type)


@pytest.mark.large
def test_no_figure_is_opened_after_logging(bst_params, dtrain):
    mlflow.xgboost.autolog()
    xgb.train(bst_params, dtrain)
    assert mpl.pyplot.get_fignums() == []


@pytest.mark.large
def test_xgb_autolog_loads_model_from_artifact(bst_params, dtrain):
    mlflow.xgboost.autolog()
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id

    loaded_model = mlflow.xgboost.load_model('runs:/{}/model'.format(run_id))
    np.testing.assert_array_almost_equal(model.predict(dtrain), loaded_model.predict(dtrain))
