"""
Simple unit tests to confirm that ModelRegistryClient properly calls the registry Store methods
and returns values when required.
"""
from mock import ANY
import pytest
import mock

from mlflow.entities.model_registry import ModelVersion, RegisteredModel, \
    RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.tracking._model_registry.client import ModelRegistryClient


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracking._model_registry.utils._get_store") as mock_get_store:
        yield mock_get_store.return_value


def newModelRegistryClient():
    return ModelRegistryClient("uri:/fake")


def _model_version(name, version, stage, source="some:/source", run_id="run13579"):
    return ModelVersion(name, version, "2345671890", "234567890",
                        "some description", "UserID", stage, source, run_id)


# Registered Model API
def test_create_registered_model(mock_store):
    mock_store.create_registered_model.return_value = RegisteredModel("Model 1")
    result = newModelRegistryClient().create_registered_model("Model 1")
    mock_store.create_registered_model.assert_called_once_with("Model 1")
    assert result.name == "Model 1"


def test_update_and_rename_registered_model(mock_store):
    name = "Model 1"
    new_description = "New Description"
    new_description_2 = "New Description 2"
    mock_store.update_registered_model.return_value = RegisteredModel(name,
                                                                      description=new_description)

    result = newModelRegistryClient().update_registered_model(
        name=name,
        description=new_description)
    mock_store.update_registered_model.assert_called_with(name=name, description=new_description)

    assert result.description == new_description
    newModelRegistryClient().update_registered_model(
        name=name,
        description=new_description_2)
    mock_store.update_registered_model.assert_called_with(name=name,
                                                          description="New Description 2")


def test_rename_registered_model(mock_store):
    name = "Model 1"
    new_name = "New Name"
    mock_store.rename_registered_model.return_value = RegisteredModel(new_name)
    result = newModelRegistryClient().rename_registered_model(
        name=name,
        new_name=new_name)
    mock_store.rename_registered_model.assert_called_with(name=name, new_name=new_name)
    assert result.name == "New Name"

    mock_store.update_registered_model.return_value = RegisteredModel("New Name 2")
    newModelRegistryClient().rename_registered_model(
        name=name,
        new_name="New Name 2")
    mock_store.rename_registered_model.assert_called_with(name=name, new_name="New Name 2")


def test_update_registered_model_validation_errors_on_empty_new_name(mock_store):
    with pytest.raises(MlflowException):
        newModelRegistryClient().rename_registered_model("Model 1", " ")


def test_delete_registered_model(mock_store):
    newModelRegistryClient().delete_registered_model("Model 1")
    mock_store.delete_registered_model.assert_called_once()


def test_list_registered_models(mock_store):
    mock_store.list_registered_models.return_value = [
        RegisteredModel("Model 1"),
        RegisteredModel("Model 2")
    ]
    result = newModelRegistryClient().list_registered_models()
    mock_store.list_registered_models.assert_called_once()
    assert len(result) == 2


def test_get_registered_model_details(mock_store):
    name = "Model 1"
    mock_store.get_registered_model.return_value = RegisteredModel(
        name, "1263283747835", "1283168374623874", "I am a model",
        [_model_version("Model 1", 3, "None"),
         _model_version("Model 1", 2, "Staging"),
         _model_version("Model 1", 1, "Production")]
    )
    result = newModelRegistryClient().get_registered_model(name)
    mock_store.get_registered_model.assert_called_once()
    assert result.name == name
    assert len(result.latest_versions) == 3


def test_get_latest_versions(mock_store):
    mock_store.get_latest_versions.return_value = [
        _model_version("Model 1", 3, "None"),
        _model_version("Model 1", 2, "Staging"),
        _model_version("Model 1", 1, "Production")
    ]
    result = newModelRegistryClient().get_latest_versions("Model 1", ["Stage1", "Stage2"])
    mock_store.get_latest_versions.assert_called_once_with(ANY, ["Stage1", "Stage2"])
    assert len(result) == 3


# Model Version API


def test_create_model_version(mock_store):
    name = "Model 1"
    version = "1"
    mock_store.create_model_version.return_value = ModelVersion(name=name, version=version,
                                                                creation_timestamp=123)
    result = newModelRegistryClient().create_model_version(name, "uri:/for/source", "run123")
    mock_store.create_model_version.assert_called_once_with(name, "uri:/for/source", "run123")
    assert result.name == name
    assert result.version == version


def test_update_model_version(mock_store):
    name = "Model 1"
    version = "12"
    description = "new description"
    expected_result = ModelVersion(name, version, creation_timestamp=123, description=description)
    mock_store.update_model_version.return_value = expected_result
    actal_result = newModelRegistryClient().update_model_version(name, version, "new description")
    mock_store.update_model_version.assert_called_once_with(name=name, version=version,
                                                            description="new description")
    assert expected_result == actal_result


def test_transition_model_version_stage(mock_store):
    name = "Model 1"
    version = "12"
    stage = "Production"
    expected_result = ModelVersion(name, version, creation_timestamp=123, current_stage=stage)
    mock_store.transition_model_version_stage.return_value = expected_result
    actual_result = newModelRegistryClient().transition_model_version_stage(name, version, stage)
    mock_store.transition_model_version_stage.assert_called_once_with(
        name=name, version=version, stage=stage, archive_existing_versions=False)
    assert expected_result == actual_result


def test_transition_model_version_stage_validation_errors(mock_store):
    with pytest.raises(MlflowException):
        newModelRegistryClient().transition_model_version_stage("Model 1", "12", stage=" ")


def test_delete_model_version(mock_store):
    newModelRegistryClient().delete_model_version("Model 1", 12)
    mock_store.delete_model_version.assert_called_once()


def test_get_model_version_details(mock_store):
    mock_store.get_model_version.return_value = _model_version("Model 1", "12",
                                                               "Production")
    result = newModelRegistryClient().get_model_version("Model 1", "12")
    mock_store.get_model_version.assert_called_once()
    assert result.name == "Model 1"


def test_get_model_version_download_uri(mock_store):
    mock_store.get_model_version_download_uri.return_value = "some:/uri/here"
    result = newModelRegistryClient().get_model_version_download_uri("Model 1", 12)
    mock_store.get_model_version_download_uri.assert_called_once()
    assert result == "some:/uri/here"


def test_search_model_versions(mock_store):
    mock_store.search_model_versions.return_value = [
        ModelVersion(name="Model 1", version="1", creation_timestamp=123),
        ModelVersion(name="Model 1", version="2", creation_timestamp=124)
    ]
    result = newModelRegistryClient().search_model_versions("name=Model 1")
    mock_store.search_model_versions.assert_called_once_with("name=Model 1")
    assert len(result) == 2


def test_get_model_version_stages(mock_store):
    mock_store.get_model_version_stages.return_value = ["Stage A", "Stage B"]
    result = newModelRegistryClient().get_model_version_stages("Model 1", 1)
    mock_store.get_model_version_stages.assert_called_once()
    assert len(result) == 2
    assert "Stage A" in result
    assert "Stage B" in result
