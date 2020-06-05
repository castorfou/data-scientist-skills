#!/bin/bash
set -ex
if ! [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$TRAVIS_BUILD_STAGE_NAME" == "Nightly" ]]
then
  if [[ "$TRAVIS_OS_NAME" == "windows" ]]
  then
     echo "skipping this step on windows."
  else
    ./travis/run-large-python-tests.sh
    ./travis/test-anaconda-compatibility.sh "anaconda3:2020.02"
    ./travis/test-anaconda-compatibility.sh "anaconda3:2019.03"
  fi
fi
CHANGED_FILES=$(git diff --name-only master..HEAD | grep "tests/examples\|examples") || true
if [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$CHANGED_FILES" == *"examples"* ]] && [[ "$TRAVIS_BUILD_STAGE_NAME" == "Nightly" ]]
then
  pytest --verbose tests/examples --large;
fi
if [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$CHANGED_FILES" == *"Dockerfile"* ]] && [[ "$TRAVIS_BUILD_STAGE_NAME" == "Nightly" ]]
then
  docker build -t mlflow_test_build . && docker images | grep mlflow_test_build
fi
