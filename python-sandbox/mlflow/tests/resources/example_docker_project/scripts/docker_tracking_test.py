""" Example script that calls tracking APIs within / outside of a start_run() block. """
import mlflow
import os
import sys
import tempfile


def call_tracking_apis():
    mlflow.log_metric("some_key", 3)
    with tempfile.NamedTemporaryFile("w") as temp_file:
        temp_file.write("Temporary content.")
        mlflow.log_artifact(temp_file.name)


def main(use_start_run):

    if use_start_run:
        print("Running with start_run API")
        with mlflow.start_run():
            call_tracking_apis()
    else:
        print("Running without start_run API")
        call_tracking_apis()


if __name__ == "__main__":
    main(use_start_run=int(sys.argv[1]))
