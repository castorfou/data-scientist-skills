import unittest

import json
import mock
import pytest
import uuid

from mlflow.protos.model_registry_pb2 import CreateRegisteredModel, \
    UpdateRegisteredModel, DeleteRegisteredModel, ListRegisteredModels, \
    GetRegisteredModel, GetLatestVersions, CreateModelVersion, UpdateModelVersion, \
    DeleteModelVersion, GetModelVersion, GetModelVersionDownloadUri, SearchModelVersions, \
    RenameRegisteredModel, TransitionModelVersionStage
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import MlflowHostCreds


@pytest.fixture(scope="class")
def request_fixture():
    with mock.patch('requests.request') as request_mock:
        response = mock.MagicMock
        response.status_code = 200
        response.text = '{}'
        request_mock.return_value = response
        yield request_mock


@pytest.mark.usefixtures("request_fixture")
class TestRestStore(unittest.TestCase):
    def setUp(self):
        self.creds = MlflowHostCreds('https://hello')
        self.store = RestStore(lambda: self.creds)

    def tearDown(self):
        pass

    def _args(self, host_creds, endpoint, method, json_body):
        res = {'host_creds': host_creds,
               'endpoint': "/api/2.0/preview/mlflow/%s" % endpoint,
               'method': method}
        if method == "GET":
            res["params"] = json.loads(json_body)
        else:
            res["json"] = json.loads(json_body)
        return res

    def _verify_requests(self, http_request, endpoint, method, proto_message):
        print(http_request.call_args_list)
        json_body = message_to_json(proto_message)
        http_request.assert_any_call(**(self._args(self.creds, endpoint, method, json_body)))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_create_registered_model(self, mock_http):
        self.store.create_registered_model("model_1")
        self._verify_requests(mock_http, "registered-models/create", "POST",
                              CreateRegisteredModel(name="model_1"))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_update_registered_model_name(self, mock_http):
        name = "model_1"
        new_name = "model_2"
        self.store.rename_registered_model(name=name, new_name=new_name)
        self._verify_requests(mock_http, "registered-models/rename", "POST",
                              RenameRegisteredModel(name=name, new_name=new_name))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_update_registered_model_description(self, mock_http):
        name = "model_1"
        description = "test model"
        self.store.update_registered_model(name=name, description=description)
        self._verify_requests(mock_http, "registered-models/update", "PATCH",
                              UpdateRegisteredModel(name=name, description=description))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_delete_registered_model(self, mock_http):
        name = "model_1"
        self.store.delete_registered_model(name=name)
        self._verify_requests(mock_http, "registered-models/delete", "DELETE",
                              DeleteRegisteredModel(name=name))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_list_registered_model(self, mock_http):
        self.store.list_registered_models()
        self._verify_requests(mock_http, "registered-models/list", "GET", ListRegisteredModels())

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_get_registered_model(self, mock_http):
        name = "model_1"
        self.store.get_registered_model(name=name)
        self._verify_requests(mock_http, "registered-models/get", "GET",
                              GetRegisteredModel(name=name))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_get_latest_versions(self, mock_http):
        name = "model_1"
        self.store.get_latest_versions(name=name)
        self._verify_requests(mock_http, "registered-models/get-latest-versions", "GET",
                              GetLatestVersions(name=name))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_get_latest_versions_with_stages(self, mock_http):
        name = "model_1"
        self.store.get_latest_versions(name=name, stages=["blaah"])
        self._verify_requests(mock_http, "registered-models/get-latest-versions", "GET",
                              GetLatestVersions(name=name, stages=["blaah"]))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_create_model_version(self, mock_http):
        run_id = uuid.uuid4().hex
        self.store.create_model_version("model_1", "path/to/source", run_id)
        self._verify_requests(mock_http, "model-versions/create", "POST",
                              CreateModelVersion(name="model_1", source="path/to/source",
                                                 run_id=run_id))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_transition_model_version_stage(self, mock_http):
        name = "model_1"
        version = "5"
        self.store.transition_model_version_stage(name=name, version=version, stage="prod",
                                                  archive_existing_versions=True)
        self._verify_requests(mock_http, "model-versions/transition-stage", "POST",
                              TransitionModelVersionStage(name=name, version=version,
                                                          stage="prod",
                                                          archive_existing_versions=True))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_update_model_version_decription(self, mock_http):
        name = "model_1"
        version = "5"
        description = "test model version"
        self.store.update_model_version(name=name, version=version, description=description)
        self._verify_requests(mock_http, "model-versions/update", "PATCH",
                              UpdateModelVersion(name=name, version=version,
                                                 description="test model version"))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_delete_model_version(self, mock_http):
        name = "model_1"
        version = "12"
        self.store.delete_model_version(name=name, version=version)
        self._verify_requests(mock_http, "model-versions/delete", "DELETE",
                              DeleteModelVersion(name=name, version=version))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_get_model_version_details(self, mock_http):
        name = "model_11"
        version = "8"
        self.store.get_model_version(name=name, version=version)
        self._verify_requests(mock_http, "model-versions/get", "GET",
                              GetModelVersion(name=name, version=version))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_get_model_version_download_uri(self, mock_http):
        name = "model_11"
        version = "8"
        self.store.get_model_version_download_uri(name=name, version=version)
        self._verify_requests(mock_http, "model-versions/get-download-uri", "GET",
                              GetModelVersionDownloadUri(name=name, version=version))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_search_model_versions(self, mock_http):
        self.store.search_model_versions(filter_string="name='model_12'")
        self._verify_requests(mock_http, "model-versions/search", "GET",
                              SearchModelVersions(filter="name='model_12'"))
