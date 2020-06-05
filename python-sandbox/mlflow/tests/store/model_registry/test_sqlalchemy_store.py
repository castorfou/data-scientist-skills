import os
import unittest

import mock
import tempfile
import uuid

import mlflow
import mlflow.db
import mlflow.store.db.base_sql_model
from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, RESOURCE_DOES_NOT_EXIST, \
    INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from tests.helper_functions import random_str

DB_URI = 'sqlite:///'


class TestSqlAlchemyStoreSqlite(unittest.TestCase):
    def _get_store(self, db_uri=''):
        return SqlAlchemyStore(db_uri)

    def setUp(self):
        self.maxDiff = None  # print all differences on assert failures
        fd, self.temp_dbfile = tempfile.mkstemp()
        # Close handle immediately so that we can remove the file later on in Windows
        os.close(fd)
        self.db_url = "%s%s" % (DB_URI, self.temp_dbfile)
        self.store = self._get_store(self.db_url)

    def tearDown(self):
        mlflow.store.db.base_sql_model.Base.metadata.drop_all(self.store.engine)
        os.remove(self.temp_dbfile)

    def _rm_maker(self, name):
        return self.store.create_registered_model(name)

    def _mv_maker(self, name, source="path/to/source", run_id=uuid.uuid4().hex):
        return self.store.create_model_version(name, source, run_id)

    def _extract_latest_by_stage(self, latest_versions):
        return {mvd.current_stage: mvd.version for mvd in latest_versions}

    def test_create_registered_model(self):
        name = random_str() + "abCD"
        rm1 = self._rm_maker(name)
        self.assertEqual(rm1.name, name)

        # error on duplicate
        with self.assertRaises(MlflowException) as exception_context:
            self._rm_maker(name)
        assert exception_context.exception.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)

        # slightly different name is ok
        for name2 in [name + "extra", name.lower(), name.upper(), name + name]:
            rm2 = self._rm_maker(name2)
            self.assertEqual(rm2.name, name2)

    def test_get_registered_model(self):
        name = "model_1"
        # use fake clock
        with mock.patch("time.time") as mock_time:
            mock_time.return_value = 1234
            rm = self._rm_maker(name)
            self.assertEqual(rm.name, name)
        rmd = self.store.get_registered_model(name=name)
        self.assertEqual(rmd.name, name)
        self.assertEqual(rmd.creation_timestamp, 1234000)
        self.assertEqual(rmd.last_updated_timestamp, 1234000)
        self.assertEqual(rmd.description, None)
        self.assertEqual(rmd.latest_versions, [])

    def test_update_registered_model(self):
        name1 = "model_for_update_RM"
        name2 = "NewName"
        rm1 = self._rm_maker(name1)
        rmd1 = self.store.get_registered_model(name=name1)
        self.assertEqual(rm1.name, name1)
        self.assertEqual(rmd1.description, None)

        # update name
        rm2 = self.store.rename_registered_model(name=name1, new_name=name2)
        rmd2 = self.store.get_registered_model(name=name2)
        self.assertEqual(rm2.name, "NewName")
        self.assertEqual(rmd2.name, "NewName")
        self.assertEqual(rmd2.description, None)

        # update description
        rm3 = self.store.update_registered_model(name=name2, description="test model")
        rmd3 = self.store.get_registered_model(name=name2)
        self.assertEqual(rm3.name, "NewName")
        self.assertEqual(rmd3.name, "NewName")
        self.assertEqual(rmd3.description, "test model")

        # new models with old names
        self._rm_maker(name1)
        # cannot rename model to conflict with an existing model
        with self.assertRaises(MlflowException) as exception_context:
            self.store.rename_registered_model(name=name2, new_name=name1)
        assert exception_context.exception.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)

    def test_delete_registered_model(self):
        name = "model_for_delete_RM"
        rm = self._rm_maker(name)
        rmd1 = self.store.get_registered_model(name=name)
        self.assertEqual(rmd1.name, name)

        # delete model
        self.store.delete_registered_model(name=name)

        # cannot get model
        with self.assertRaises(MlflowException) as exception_context:
            self.store.get_registered_model(name=name)
        assert exception_context.exception.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # cannot update a delete model
        with self.assertRaises(MlflowException) as exception_context:
            self.store.update_registered_model(name=name, description="deleted")
        assert exception_context.exception.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # cannot delete it again
        with self.assertRaises(MlflowException) as exception_context:
            self.store.delete_registered_model(name=name)
        assert exception_context.exception.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    def test_list_registered_model(self):
        self._rm_maker("A")
        registered_models = self.store.list_registered_models()
        self.assertEqual(len(registered_models), 1)
        self.assertEqual(registered_models[0].name, "A")
        self.assertIsInstance(registered_models[0], RegisteredModel)

        self._rm_maker("B")
        self.assertEqual(set([rm.name for rm in self.store.list_registered_models()]),
                         set(["A", "B"]))

        self._rm_maker("BB")
        self._rm_maker("BA")
        self._rm_maker("AB")
        self._rm_maker("BBC")
        self.assertEqual(set([rm.name for rm in self.store.list_registered_models()]),
                         set(["A", "B", "BB", "BA", "AB", "BBC"]))

        # list should not return deleted models
        self.store.delete_registered_model(name="BA")
        self.store.delete_registered_model(name="B")
        self.assertEqual(set([rm.name for rm in self.store.list_registered_models()]),
                         set(["A", "BB", "AB", "BBC"]))

    def test_get_latest_versions(self):
        name = "test_for_latest_versions"
        self._rm_maker(name)
        rmd1 = self.store.get_registered_model(name=name)
        self.assertEqual(rmd1.latest_versions, [])

        mv1 = self._mv_maker(name)
        self.assertEqual(mv1.version, 1)
        rmd2 = self.store.get_registered_model(name=name)
        self.assertEqual(self._extract_latest_by_stage(rmd2.latest_versions), {"None": 1})

        # add a bunch more
        mv2 = self._mv_maker(name)
        self.assertEqual(mv2.version, 2)
        self.store.transition_model_version_stage(
            name=mv2.name, version=mv2.version, stage="Production",
            archive_existing_versions=False)

        mv3 = self._mv_maker(name)
        self.assertEqual(mv3.version, 3)
        self.store.transition_model_version_stage(name=mv3.name, version=mv3.version,
                                                  stage="Production",
                                                  archive_existing_versions=False)
        mv4 = self._mv_maker(name)
        self.assertEqual(mv4.version, 4)
        self.store.transition_model_version_stage(
            name=mv4.name, version=mv4.version, stage="Staging",
            archive_existing_versions=False)

        # test that correct latest versions are returned for each stage
        rmd4 = self.store.get_registered_model(name=name)
        self.assertEqual(self._extract_latest_by_stage(rmd4.latest_versions),
                         {"None": 1, "Production": 3, "Staging": 4})

        # delete latest Production, and should point to previous one
        self.store.delete_model_version(name=mv3.name, version=mv3.version)
        rmd5 = self.store.get_registered_model(name=name)
        self.assertEqual(self._extract_latest_by_stage(rmd5.latest_versions),
                         {"None": 1, "Production": 2, "Staging": 4})

    def test_create_model_version(self):
        name = "test_for_update_MV"
        self._rm_maker(name)
        run_id = uuid.uuid4().hex
        with mock.patch("time.time") as mock_time:
            mock_time.return_value = 456778
            mv1 = self._mv_maker(name, "a/b/CD", run_id)
            self.assertEqual(mv1.name, name)
            self.assertEqual(mv1.version, 1)

        mvd1 = self.store.get_model_version(mv1.name, mv1.version)
        self.assertEqual(mvd1.name, name)
        self.assertEqual(mvd1.version, 1)
        self.assertEqual(mvd1.current_stage, "None")
        self.assertEqual(mvd1.creation_timestamp, 456778000)
        self.assertEqual(mvd1.last_updated_timestamp, 456778000)
        self.assertEqual(mvd1.description, None)
        self.assertEqual(mvd1.source, "a/b/CD")
        self.assertEqual(mvd1.run_id, run_id)
        self.assertEqual(mvd1.status, "READY")
        self.assertEqual(mvd1.status_message, None)

        # new model versions for same name autoincrement versions
        mv2 = self._mv_maker(name)
        mvd2 = self.store.get_model_version(name=mv2.name, version=mv2.version)
        self.assertEqual(mv2.version, 2)
        self.assertEqual(mvd2.version, 2)

        mv3 = self._mv_maker(name)
        mvd3 = self.store.get_model_version(name=mv3.name, version=mv3.version)
        self.assertEqual(mv3.version, 3)
        self.assertEqual(mvd3.version, 3)

    def test_update_model_version(self):
        name = "test_for_update_MV"
        self._rm_maker(name)
        mv1 = self._mv_maker(name)
        mvd1 = self.store.get_model_version(name=mv1.name, version=mv1.version)
        self.assertEqual(mvd1.name, name)
        self.assertEqual(mvd1.version, 1)
        self.assertEqual(mvd1.current_stage, "None")

        # update stage
        self.store.transition_model_version_stage(name=mv1.name, version=mv1.version,
                                                  stage="Production",
                                                  archive_existing_versions=False)
        mvd2 = self.store.get_model_version(name=mv1.name, version=mv1.version)
        self.assertEqual(mvd2.name, name)
        self.assertEqual(mvd2.version, 1)
        self.assertEqual(mvd2.current_stage, "Production")
        self.assertEqual(mvd2.description, None)

        # update description
        self.store.update_model_version(name=mv1.name, version=mv1.version,
                                        description="test model version")
        mvd3 = self.store.get_model_version(name=mv1.name, version=mv1.version)
        self.assertEqual(mvd3.name, name)
        self.assertEqual(mvd3.version, 1)
        self.assertEqual(mvd3.current_stage, "Production")
        self.assertEqual(mvd3.description, "test model version")

        # only valid stages can be set
        with self.assertRaises(MlflowException) as exception_context:
            self.store.transition_model_version_stage(mv1.name, mv1.version,
                                                      stage="unknown",
                                                      archive_existing_versions=False)
        assert exception_context.exception.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        # stages are case-insensitive and auto-corrected to system stage names
        for stage_name in ["STAGING", "staging", "StAgInG"]:
            self.store.transition_model_version_stage(
                name=mv1.name, version=mv1.version,
                stage=stage_name, archive_existing_versions=False)
            mvd5 = self.store.get_model_version(name=mv1.name, version=mv1.version)
            self.assertEqual(mvd5.current_stage, "Staging")

    def test_delete_model_version(self):
        name = "test_for_update_MV"
        self._rm_maker(name)
        mv = self._mv_maker(name)
        mvd = self.store.get_model_version(name=mv.name, version=mv.version)
        self.assertEqual(mvd.name, name)

        self.store.delete_model_version(name=mv.name, version=mv.version)

        # cannot get a deleted model version
        with self.assertRaises(MlflowException) as exception_context:
            self.store.get_model_version(name=mv.name, version=mv.version)
        assert exception_context.exception.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # cannot update a delete
        with self.assertRaises(MlflowException) as exception_context:
            self.store.update_model_version(mv.name, mv.version, description="deleted!")
        assert exception_context.exception.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

        # cannot delete it again
        with self.assertRaises(MlflowException) as exception_context:
            self.store.delete_model_version(name=mv.name, version=mv.version)
        assert exception_context.exception.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    def test_get_model_version_download_uri(self):
        name = "test_for_update_MV"
        self._rm_maker(name)
        source_path = "path/to/source"
        mv = self._mv_maker(name, source=source_path, run_id=uuid.uuid4().hex)
        mvd1 = self.store.get_model_version(name=mv.name, version=mv.version)
        self.assertEqual(mvd1.name, name)
        self.assertEqual(mvd1.source, source_path)

        # download location points to source
        self.assertEqual(self.store.get_model_version_download_uri(name=mv.name,
                                                                   version=mv.version), source_path)

        # download URI does not change even if model version is updated
        self.store.transition_model_version_stage(
            name=mv.name, version=mv.version,
            stage="Production",
            archive_existing_versions=False)
        self.store.update_model_version(name=mv.name, version=mv.version,
                                        description="Test for Path")
        mvd2 = self.store.get_model_version(name=mv.name, version=mv.version)
        self.assertEqual(mvd2.source, source_path)
        self.assertEqual(self.store.get_model_version_download_uri(
            name=mv.name, version=mv.version), source_path)

        # cannot retrieve download URI for deleted model versions
        self.store.delete_model_version(name=mv.name, version=mv.version)
        with self.assertRaises(MlflowException) as exception_context:
            self.store.get_model_version_download_uri(name=mv.name, version=mv.version)
        assert exception_context.exception.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    def test_search_model_versions(self):
        # create some model versions
        name = "test_for_search_MV"
        self._rm_maker(name)
        run_id_1 = uuid.uuid4().hex
        run_id_2 = uuid.uuid4().hex
        run_id_3 = uuid.uuid4().hex
        mv1 = self._mv_maker(name=name, source="A/B", run_id=run_id_1)
        self.assertEqual(mv1.version, 1)
        mv2 = self._mv_maker(name=name, source="A/C", run_id=run_id_2)
        self.assertEqual(mv2.version, 2)
        mv3 = self._mv_maker(name=name, source="A/D", run_id=run_id_2)
        self.assertEqual(mv3.version, 3)
        mv4 = self._mv_maker(name=name, source="A/D", run_id=run_id_3)
        self.assertEqual(mv4.version, 4)

        def search_versions(filter_string):
            return [mvd.version for mvd in self.store.search_model_versions(filter_string)]

        # search using name should return all 4 versions
        self.assertEqual(set(search_versions("name='%s'" % name)), set([1, 2, 3, 4]))

        # search using run_id_1 should return version 1
        self.assertEqual(set(search_versions("run_id='%s'" % run_id_1)), set([1]))

        # search using run_id_2 should return versions 2 and 3
        self.assertEqual(set(search_versions("run_id='%s'" % run_id_2)), set([2, 3]))

        # search using source_path "A/D" should return version 3 and 4
        self.assertEqual(set(search_versions("source_path = 'A/D'")), set([3, 4]))

        # search using source_path "A" should not return anything
        self.assertEqual(len(search_versions("source_path = 'A'")), 0)
        self.assertEqual(len(search_versions("source_path = 'A/'")), 0)
        self.assertEqual(len(search_versions("source_path = ''")), 0)

        # delete mv4. search should not return version 4
        self.store.delete_model_version(name=mv4.name, version=mv4.version)
        self.assertEqual(set(search_versions("")), set([1, 2, 3]))

        self.assertEqual(set(search_versions(None)), set([1, 2, 3]))

        self.assertEqual(set(search_versions("name='%s'" % name)), set([1, 2, 3]))

        self.assertEqual(set(search_versions("source_path = 'A/D'")), set([3]))

        self.store.transition_model_version_stage(
            name=mv1.name, version=mv1.version, stage="production",
            archive_existing_versions=False
        )

        self.store.update_model_version(
            name=mv1.name, version=mv1.version, description="Online prediction model!")

        mvds = self.store.search_model_versions("run_id = '%s'" % run_id_1)
        assert 1 == len(mvds)
        assert isinstance(mvds[0], ModelVersion)
        assert mvds[0].current_stage == "Production"
        assert mvds[0].run_id == run_id_1
        assert mvds[0].source == "A/B"
        assert mvds[0].description == "Online prediction model!"
