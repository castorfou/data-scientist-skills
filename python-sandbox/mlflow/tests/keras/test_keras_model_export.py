# pep8: disable=E501

import h5py
import os
import json
import pytest
import shutil
import importlib
from keras.models import Sequential
from keras.layers import Layer, Dense
from keras import backend as K
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import yaml
import mock

import mlflow
import mlflow.keras
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from tests.helper_functions import pyfunc_serve_and_score_model
from tests.helper_functions import score_model_in_sagemaker_docker_container
from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import
from tests.pyfunc.test_spark import score_model_as_udf


@pytest.fixture(scope='module')
def data():
    iris = datasets.load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
    y = data['target']
    x = data.drop('target', axis=1)
    return x, y


@pytest.fixture(scope='module')
def model(data):
    x, y = data
    model = Sequential()
    model.add(Dense(3, input_dim=4))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='SGD')
    model.fit(x, y)
    return model


@pytest.fixture(scope='module')
def tf_keras_model(data):
    x, y = data
    from tensorflow.keras.models import Sequential as TfSequential
    from tensorflow.keras.layers import Dense as TfDense
    model = TfSequential()
    model.add(TfDense(3, input_dim=4))
    model.add(TfDense(1))
    model.compile(loss='mean_squared_error', optimizer='SGD')
    model.fit(x, y)
    return model


@pytest.fixture(scope='module')
def predicted(model, data):
    return model.predict(data[0])


@pytest.fixture(scope='module')
def custom_layer():
    class MyDense(Layer):
        def __init__(self, output_dim, **kwargs):
            self.output_dim = output_dim
            super(MyDense, self).__init__(**kwargs)

        def build(self, input_shape):
            self.kernel = self.add_weight(name='kernel',
                                          shape=(input_shape[1], self.output_dim),
                                          initializer='uniform',
                                          trainable=True)
            super(MyDense, self).build(input_shape)

        def call(self, x):
            return K.dot(x, self.kernel)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_dim)

        def get_config(self):
            return {'output_dim': self.output_dim}

    return MyDense


@pytest.fixture(scope='module')
def custom_model(data, custom_layer):
    x, y = data
    x, y = x.values, y.values
    model = Sequential()
    model.add(custom_layer(6))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='SGD')
    model.fit(x, y, epochs=1)
    return model


@pytest.fixture(scope='module')
def custom_predicted(custom_model, data):
    return custom_model.predict(data[0])


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(tmpdir.strpath, "model")


@pytest.fixture
def keras_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(
        conda_env,
        additional_conda_deps=["keras", "tensorflow", "pytest"])
    return conda_env


def test_that_keras_module_arg_works(model_path):
    class MyModel(object):
        def __init__(self, x):
            self._x = x

        def __eq__(self, other):
            return self._x == other._x

        def save(self, path, **kwargs):
            with h5py.File(path, "w") as f:
                f.create_dataset(name="x", data=self._x)

    class FakeKerasModule(object):
        __name__ = "some.test.keras.module"
        __version__ = "42.42.42"

        @staticmethod
        def load_model(file, **kwars):
            return MyModel(file.get("x").value)

    original_import = importlib.import_module

    def _import_module(name, **kwargs):
        if name.startswith(FakeKerasModule.__name__):
            return FakeKerasModule
        else:
            return original_import(name, **kwargs)

    with mock.patch("importlib.import_module") as import_module_mock:
        import_module_mock.side_effect = _import_module
        x = MyModel("x123")
        path0 = os.path.join(model_path, "0")
        with pytest.raises(MlflowException):
            mlflow.keras.save_model(x, path0)
        mlflow.keras.save_model(x, path0, keras_module=FakeKerasModule)
        y = mlflow.keras.load_model(path0)
        assert x == y
        path1 = os.path.join(model_path, "1")
        mlflow.keras.save_model(x, path1, keras_module=FakeKerasModule.__name__)
        z = mlflow.keras.load_model(path1)
        assert x == z
        # Tests model log
        with mlflow.start_run() as active_run:
            with pytest.raises(MlflowException):
                mlflow.keras.log_model(x, "model0")
            mlflow.keras.log_model(x, "model0", keras_module=FakeKerasModule)
            a = mlflow.keras.load_model("runs:/{}/model0".format(active_run.info.run_id))
            assert x == a
            mlflow.keras.log_model(x, "model1", keras_module=FakeKerasModule.__name__)
            b = mlflow.keras.load_model("runs:/{}/model1".format(active_run.info.run_id))
            assert x == b


@pytest.mark.parametrize("build_model", [model, tf_keras_model])
@pytest.mark.large
def test_model_save_load(build_model, model_path, data):
    x, _ = data
    keras_model = build_model(data)
    if build_model == tf_keras_model:
        model_path = os.path.join(model_path, "tf")
    else:
        model_path = os.path.join(model_path, "plain")
    expected = keras_model.predict(x)
    mlflow.keras.save_model(keras_model, model_path)
    # Loading Keras model
    model_loaded = mlflow.keras.load_model(model_path)
    assert type(keras_model) == type(model_loaded)
    assert all(expected == model_loaded.predict(x))
    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    assert all(pyfunc_loaded.predict(x).values == expected)

    # pyfunc serve
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pd.DataFrame(x),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    print(scoring_response.content)
    assert all(pd.read_json(scoring_response.content, orient="records",
                            encoding="utf8").values.astype(np.float32)
               == expected)
    # test spark udf
    spark_udf_preds = score_model_as_udf(model_uri=os.path.abspath(model_path),
                                         pandas_df=pd.DataFrame(x),
                                         result_type="float")
    np.allclose(
        np.array(spark_udf_preds), expected.reshape(len(spark_udf_preds)))


@pytest.mark.large
def test_signature_and_examples_are_saved_correctly(model, data):
    signature_ = infer_signature(*data)
    example_ = data[0].head(3)
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.keras.save_model(model, path=path,
                                        signature=signature,
                                        input_example=example)
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


@pytest.mark.large
def test_custom_model_save_load(custom_model, custom_layer, data, custom_predicted, model_path):
    x, _ = data
    custom_objects = {'MyDense': custom_layer}
    mlflow.keras.save_model(custom_model, model_path, custom_objects=custom_objects)

    # Loading Keras model
    model_loaded = mlflow.keras.load_model(model_path)
    assert all(model_loaded.predict(x) == custom_predicted)
    # pyfunc serve
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pd.DataFrame(x),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    assert np.allclose(
        pd.read_json(scoring_response.content, orient="records",
                     encoding="utf8").values.astype(np.float32),
        custom_predicted,
        rtol=1e-5,
        atol=1e-9)
    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    assert all(pyfunc_loaded.predict(x).values == custom_predicted)
    # test spark udf
    spark_udf_preds = score_model_as_udf(model_uri=os.path.abspath(model_path),
                                         pandas_df=pd.DataFrame(x),
                                         result_type="float")
    np.allclose(
        np.array(spark_udf_preds), custom_predicted.reshape(len(spark_udf_preds)))


def test_custom_model_save_respects_user_custom_objects(custom_model, custom_layer, model_path):
    class DifferentCustomLayer():
        def __init__(self):
            pass

        def __call__(self):
            pass

    incorrect_custom_objects = {'MyDense': DifferentCustomLayer()}
    correct_custom_objects = {'MyDense': custom_layer}
    mlflow.keras.save_model(custom_model, model_path, custom_objects=incorrect_custom_objects)
    model_loaded = mlflow.keras.load_model(model_path, custom_objects=correct_custom_objects)
    assert model_loaded is not None
    with pytest.raises(TypeError):
        model_loaded = mlflow.keras.load_model(model_path)


@pytest.mark.large
def test_model_load_from_remote_uri_succeeds(model, model_path, mock_s3_bucket, data, predicted):
    x, _ = data
    mlflow.keras.save_model(model, model_path)

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    model_loaded = mlflow.keras.load_model(model_uri=model_uri)
    assert all(model_loaded.predict(x) == predicted)


@pytest.mark.large
def test_model_log(model, data, predicted):
    x, _ = data
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        try:
            if should_start_run:
                mlflow.start_run()
            artifact_path = "keras_model"
            mlflow.keras.log_model(model, artifact_path=artifact_path)
            model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id,
                artifact_path=artifact_path)

            # Load model
            model_loaded = mlflow.keras.load_model(model_uri=model_uri)
            assert all(model_loaded.predict(x) == predicted)

            # Loading pyfunc model
            pyfunc_loaded = mlflow.pyfunc.load_model(model_uri=model_uri)
            assert all(pyfunc_loaded.predict(x).values == predicted)
        finally:
            mlflow.end_run()


def test_log_model_calls_register_model(model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.keras.log_model(model, artifact_path=artifact_path,
                               registered_model_name="AdsModel1")
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=mlflow.active_run().info.run_id,
                                                            artifact_path=artifact_path)
        mlflow.register_model.assert_called_once_with(model_uri, "AdsModel1")


def test_log_model_no_registered_model_name(model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.keras.log_model(model, artifact_path=artifact_path)
        mlflow.register_model.assert_not_called()


@pytest.mark.large
def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
        model, model_path, keras_custom_env):
    mlflow.keras.save_model(keras_model=model, path=model_path, conda_env=keras_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != keras_custom_env

    with open(keras_custom_env, "r") as f:
        keras_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == keras_custom_env_parsed


@pytest.mark.large
def test_model_save_accepts_conda_env_as_dict(model, model_path):
    conda_env = dict(mlflow.keras.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.keras.save_model(keras_model=model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


@pytest.mark.large
def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(model, keras_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.keras.log_model(
            keras_model=model, artifact_path=artifact_path, conda_env=keras_custom_env)
        model_path = _download_artifact_from_uri("runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path))

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != keras_custom_env

    with open(keras_custom_env, "r") as f:
        keras_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == keras_custom_env_parsed


@pytest.mark.large
def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        model, model_path):
    mlflow.keras.save_model(keras_model=model, path=model_path, conda_env=None)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.keras.get_default_conda_env()


@pytest.mark.large
def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.keras.log_model(keras_model=model, artifact_path=artifact_path, conda_env=None)
        model_path = _download_artifact_from_uri("runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path))

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.keras.get_default_conda_env()


@pytest.mark.large
def test_model_load_succeeds_with_missing_data_key_when_data_exists_at_default_path(
        model, model_path, data, predicted):
    """
    This is a backwards compatibility test to ensure that models saved in MLflow version <= 0.8.0
    can be loaded successfully. These models are missing the `data` flavor configuration key.
    """
    mlflow.keras.save_model(keras_model=model, path=model_path)
    shutil.move(
        os.path.join(model_path, 'data', 'model.h5'),
        os.path.join(model_path, 'model.h5'))
    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    flavor_conf = model_conf.flavors.get(mlflow.keras.FLAVOR_NAME, None)
    assert flavor_conf is not None
    del flavor_conf['data']
    model_conf.save(model_conf_path)

    model_loaded = mlflow.keras.load_model(model_path)
    assert all(model_loaded.predict(data[0]) == predicted)


@pytest.mark.release
def test_sagemaker_docker_model_scoring_with_default_conda_env(model, model_path, data, predicted):
    mlflow.keras.save_model(keras_model=model, path=model_path, conda_env=None)

    scoring_response = score_model_in_sagemaker_docker_container(
        model_uri=model_path,
        data=data[0],
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        flavor=mlflow.pyfunc.FLAVOR_NAME,
        activity_polling_timeout_seconds=500)
    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content))

    np.testing.assert_array_almost_equal(
        deployed_model_preds.values,
        predicted,
        decimal=4)
