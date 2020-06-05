=============================================
MLflow: A Machine Learning Lifecycle Platform
=============================================

MLflow is a platform to streamline machine learning development, including tracking experiments, packaging code
into reproducible runs, and sharing and deploying models. MLflow offers a set of lightweight APIs that can be
used with any existing machine learning application or library (TensorFlow, PyTorch, XGBoost, etc), wherever you
currently run ML code (e.g. in notebooks, standalone applications or the cloud). MLflow's current components are:

* `MLflow Tracking <https://mlflow.org/docs/latest/tracking.html>`_: An API to log parameters, code, and
  results in machine learning experiments and compare them using an interactive UI.
* `MLflow Projects <https://mlflow.org/docs/latest/projects.html>`_: A code packaging format for reproducible
  runs using Conda and Docker, so you can share your ML code with others.
* `MLflow Models <https://mlflow.org/docs/latest/models.html>`_: A model packaging format and tools that let
  you easily deploy the same model (from any ML library) to batch and real-time scoring on platforms such as
  Docker, Apache Spark, Azure ML and AWS SageMaker.
* `MLflow Model Registry <https://mlflow.org/docs/latest/model-registry.html>`_: A centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of MLflow Models.

|docs| |travis| |pypi| |conda-forge| |cran| |maven| |license| |downloads|

.. |docs| image:: https://img.shields.io/badge/docs-latest-success.svg
    :target: https://mlflow.org/docs/latest/index.html
    :alt: Latest Docs
.. |travis| image:: https://img.shields.io/travis/mlflow/mlflow.svg
    :target: https://travis-ci.org/mlflow/mlflow
    :alt: Build Status
.. |pypi| image:: https://img.shields.io/pypi/v/mlflow.svg
    :target: https://pypi.org/project/mlflow/
    :alt: Latest Python Release
.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/mlflow.svg
    :target: https://anaconda.org/conda-forge/mlflow
    :alt: Latest Conda Release
.. |cran| image:: https://img.shields.io/cran/v/mlflow.svg
    :target: https://cran.r-project.org/package=mlflow
    :alt: Latest CRAN Release
.. |maven| image:: https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg
    :target: https://mvnrepository.com/artifact/org.mlflow
    :alt: Maven Central
.. |license| image:: https://img.shields.io/badge/license-Apache%202-brightgreen.svg
    :target: https://github.com/mlflow/mlflow/blob/master/LICENSE.txt
    :alt: Apache 2 License
.. |downloads| image:: https://pepy.tech/badge/mlflow
    :target: https://pepy.tech/project/mlflow
    :alt: Total Downloads

Installing
----------
Install MLflow from PyPI via ``pip install mlflow``

MLflow requires ``conda`` to be on the ``PATH`` for the projects feature.

Nightly snapshots of MLflow master are also available `here <https://mlflow-snapshots.s3-us-west-2.amazonaws.com/>`_.

Documentation
-------------
Official documentation for MLflow can be found at https://mlflow.org/docs/latest/index.html.

Community
---------
For help or questions about MLflow usage (e.g. "how do I do X?") see the `docs <https://mlflow.org/docs/latest/index.html>`_
or `Stack Overflow <https://stackoverflow.com/questions/tagged/mlflow>`_.

To report a bug, file a documentation issue, or submit a feature request, please open a GitHub issue.

For release announcements and other discussions, please subscribe to our mailing list (mlflow-users@googlegroups.com)
or join us on `Slack <https://mlflow-users.slack.com/join/shared_invite/enQtMzkxMTAwNTcyODM5LTkzMDFhNzliNjExOGQ1ZGI1ZmFlMGE5YWE1OTI4ZGM1ZWZmYzc3NGNiZTM3YjgwOTdlODAzMjJhZTdiN2Y3MWY>`_.

Running a Sample App With the Tracking API
------------------------------------------
The programs in ``examples`` use the MLflow Tracking API. For instance, run::

    python examples/quickstart/mlflow_tracking.py

This program will use `MLflow Tracking API <https://mlflow.org/docs/latest/tracking.html>`_,
which logs tracking data in ``./mlruns``. This can then be viewed with the Tracking UI.


Launching the Tracking UI
-------------------------
The MLflow Tracking UI will show runs logged in ``./mlruns`` at `<http://localhost:5000>`_.
Start it with::

    mlflow ui

**Note:** Running ``mlflow ui`` from within a clone of MLflow is not recommended - doing so will
run the dev UI from source. We recommend running the UI from a different working directory,
specifying a backend store via the ``--backend-store-uri`` option. Alternatively, see
instructions for running the dev UI in the `contributor guide <CONTRIBUTING.rst>`_.


Running a Project from a URI
----------------------------
The ``mlflow run`` command lets you run a project packaged with a MLproject file from a local path
or a Git URI::

    mlflow run examples/sklearn_elasticnet_wine -P alpha=0.4

    mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=0.4

See ``examples/sklearn_elasticnet_wine`` for a sample project with an MLproject file.


Saving and Serving Models
-------------------------
To illustrate managing models, the ``mlflow.sklearn`` package can log scikit-learn models as
MLflow artifacts and then load them again for serving. There is an example training application in
``examples/sklearn_logistic_regression/train.py`` that you can run as follows::

    $ python examples/sklearn_logistic_regression/train.py
    Score: 0.666
    Model saved in run <run-id>

    $ mlflow models serve --model-uri runs:/<run-id>/model

    $ curl -d '{"columns":[0],"index":[0,1],"data":[[1],[-1]]}' -H 'Content-Type: application/json'  localhost:5000/invocations


Contributing
------------
We happily welcome contributions to MLflow. Please see our `contribution guide <CONTRIBUTING.rst>`_
for details.
