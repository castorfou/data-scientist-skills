import os
import posixpath

import pytest

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository

from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import


@pytest.fixture
def s3_artifact_root(mock_s3_bucket):
    return "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)


def teardown_function():
    if 'MLFLOW_S3_UPLOAD_EXTRA_ARGS' in os.environ:
        del os.environ['MLFLOW_S3_UPLOAD_EXTRA_ARGS']


def test_file_artifact_is_logged_and_downloaded_successfully(s3_artifact_root, tmpdir):
    file_name = "test.txt"
    file_path = os.path.join(str(tmpdir), file_name)
    file_text = "Hello world!"

    with open(file_path, "w") as f:
        f.write(file_text)

    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    repo.log_artifact(file_path)
    downloaded_text = open(repo.download_artifacts(file_name)).read()
    assert downloaded_text == file_text


def test_file_and_directories_artifacts_are_logged_and_downloaded_successfully_in_batch(
        s3_artifact_root, tmpdir):
    subdir_path = str(tmpdir.mkdir("subdir"))
    nested_path = os.path.join(subdir_path, "nested")
    os.makedirs(nested_path)
    with open(os.path.join(subdir_path, "a.txt"), "w") as f:
        f.write("A")
    with open(os.path.join(subdir_path, "b.txt"), "w") as f:
        f.write("B")
    with open(os.path.join(nested_path, "c.txt"), "w") as f:
        f.write("C")

    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    repo.log_artifacts(subdir_path)

    # Download individual files and verify correctness of their contents
    downloaded_file_a_text = open(repo.download_artifacts("a.txt")).read()
    assert downloaded_file_a_text == "A"
    downloaded_file_b_text = open(repo.download_artifacts("b.txt")).read()
    assert downloaded_file_b_text == "B"
    downloaded_file_c_text = open(repo.download_artifacts("nested/c.txt")).read()
    assert downloaded_file_c_text == "C"

    # Download the nested directory and verify correctness of its contents
    downloaded_dir = repo.download_artifacts("nested")
    assert os.path.basename(downloaded_dir) == "nested"
    text = open(os.path.join(downloaded_dir, "c.txt")).read()
    assert text == "C"

    # Download the root directory and verify correctness of its contents
    downloaded_dir = repo.download_artifacts("")
    dir_contents = os.listdir(downloaded_dir)
    assert "nested" in dir_contents
    assert os.path.isdir(os.path.join(downloaded_dir, "nested"))
    assert "a.txt" in dir_contents
    assert "b.txt" in dir_contents


def test_file_and_directories_artifacts_are_logged_and_listed_successfully_in_batch(
        s3_artifact_root, tmpdir):
    subdir_path = str(tmpdir.mkdir("subdir"))
    nested_path = os.path.join(subdir_path, "nested")
    os.makedirs(nested_path)
    with open(os.path.join(subdir_path, "a.txt"), "w") as f:
        f.write("A")
    with open(os.path.join(subdir_path, "b.txt"), "w") as f:
        f.write("B")
    with open(os.path.join(nested_path, "c.txt"), "w") as f:
        f.write("C")

    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    repo.log_artifacts(subdir_path)

    root_artifacts_listing = sorted(
        [(f.path, f.is_dir, f.file_size) for f in repo.list_artifacts()])
    assert root_artifacts_listing == [
        ("a.txt", False, 1),
        ("b.txt", False, 1),
        ("nested", True, None),
    ]

    nested_artifacts_listing = sorted(
        [(f.path, f.is_dir, f.file_size) for f in repo.list_artifacts("nested")])
    assert nested_artifacts_listing == [("nested/c.txt", False, 1)]


def test_download_directory_artifact_succeeds_when_artifact_root_is_s3_bucket_root(
        s3_artifact_root, tmpdir):
    file_a_name = "a.txt"
    file_a_text = "A"
    subdir_path = str(tmpdir.mkdir("subdir"))
    nested_path = os.path.join(subdir_path, "nested")
    os.makedirs(nested_path)
    with open(os.path.join(nested_path, file_a_name), "w") as f:
        f.write(file_a_text)

    repo = get_artifact_repository(s3_artifact_root)
    repo.log_artifacts(subdir_path)

    downloaded_dir_path = repo.download_artifacts("nested")
    assert file_a_name in os.listdir(downloaded_dir_path)
    with open(os.path.join(downloaded_dir_path, file_a_name), "r") as f:
        assert f.read() == file_a_text


def test_download_file_artifact_succeeds_when_artifact_root_is_s3_bucket_root(
        s3_artifact_root, tmpdir):
    file_a_name = "a.txt"
    file_a_text = "A"
    file_a_path = os.path.join(str(tmpdir), file_a_name)
    with open(file_a_path, "w") as f:
        f.write(file_a_text)

    repo = get_artifact_repository(s3_artifact_root)
    repo.log_artifact(file_a_path)

    downloaded_file_path = repo.download_artifacts(file_a_name)
    with open(downloaded_file_path, "r") as f:
        assert f.read() == file_a_text


def test_get_s3_file_upload_extra_args():
    os.environ.setdefault('MLFLOW_S3_UPLOAD_EXTRA_ARGS',
                          '{"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "123456"}')

    parsed_args = S3ArtifactRepository.get_s3_file_upload_extra_args()

    assert parsed_args == {'ServerSideEncryption': 'aws:kms', 'SSEKMSKeyId': '123456'}


def test_get_s3_file_upload_extra_args_env_var_not_present():
    parsed_args = S3ArtifactRepository.get_s3_file_upload_extra_args()

    assert parsed_args is None


def test_get_s3_file_upload_extra_args_invalid_json():
    os.environ.setdefault('MLFLOW_S3_UPLOAD_EXTRA_ARGS',
                          '"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "123456"}')

    with pytest.raises(ValueError):
        S3ArtifactRepository.get_s3_file_upload_extra_args()
