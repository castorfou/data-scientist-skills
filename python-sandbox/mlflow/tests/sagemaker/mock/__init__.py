import time
import json
from collections import namedtuple
from datetime import datetime

from moto.core import BaseBackend, BaseModel
from moto.core.responses import BaseResponse
from moto.ec2 import ec2_backends

from moto.iam.models import ACCOUNT_ID
from moto.core.models import base_decorator, deprecated_base_decorator

SageMakerResourceWithArn = namedtuple("SageMakerResourceWithArn", ["resource", "arn"])


class SageMakerResponse(BaseResponse):
    """
    A collection of handlers for SageMaker API calls that produce API-conforming
    JSON responses.
    """

    @property
    def sagemaker_backend(self):
        return sagemaker_backends[self.region]

    @property
    def request_params(self):
        return json.loads(self.body)

    def create_endpoint_config(self):
        """
        Handler for the SageMaker "CreateEndpointConfig" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateEndpointConfig.html.
        """
        config_name = self.request_params["EndpointConfigName"]
        production_variants = self.request_params.get("ProductionVariants")
        tags = self.request_params.get("Tags", [])
        new_config = self.sagemaker_backend.create_endpoint_config(
                config_name=config_name, production_variants=production_variants, tags=tags,
                region_name=self.region)
        return json.dumps({
            'EndpointConfigArn': new_config.arn
        })

    def describe_endpoint_config(self):
        """
        Handler for the SageMaker "DescribeEndpoint" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpoint.html.
        """
        config_name = self.request_params["EndpointConfigName"]
        config_description = self.sagemaker_backend.describe_endpoint_config(config_name)
        return json.dumps(config_description.response_object)

    def delete_endpoint_config(self):
        """
        Handler for the SageMaker "DeleteEndpointConfig" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteEndpointConfig.html.
        """
        config_name = self.request_params["EndpointConfigName"]
        self.sagemaker_backend.delete_endpoint_config(config_name)
        return ""

    def create_endpoint(self):
        """
        Handler for the SageMaker "CreateEndpoint" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateEndpoint.html.
        """
        endpoint_name = self.request_params["EndpointName"]
        endpoint_config_name = self.request_params["EndpointConfigName"]
        tags = self.request_params.get("Tags", [])
        new_endpoint = self.sagemaker_backend.create_endpoint(
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_config_name,
                tags=tags, region_name=self.region)
        return json.dumps({
            'EndpointArn': new_endpoint.arn
        })

    def describe_endpoint(self):
        """
        Handler for the SageMaker "DescribeEndpoint" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpoint.html.
        """
        endpoint_name = self.request_params["EndpointName"]
        endpoint_description = self.sagemaker_backend.describe_endpoint(endpoint_name)
        return json.dumps(endpoint_description.response_object)

    def update_endpoint(self):
        """
        Handler for the SageMaker "UpdateEndpoint" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_UpdateEndpoint.html.
        """
        endpoint_name = self.request_params["EndpointName"]
        new_config_name = self.request_params["EndpointConfigName"]
        updated_endpoint = self.sagemaker_backend.update_endpoint(
                endpoint_name=endpoint_name, new_config_name=new_config_name)
        return json.dumps({
            'EndpointArn': updated_endpoint.arn
        })

    def delete_endpoint(self):
        """
        Handler for the SageMaker "DeleteEndpoint" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteEndpoint.html.
        """
        endpoint_name = self.request_params["EndpointName"]
        self.sagemaker_backend.delete_endpoint(endpoint_name)
        return ""

    def list_endpoints(self):
        """
        Handler for the SageMaker "ListEndpoints" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpoints.html.

        This function does not support pagination. All endpoint configs are returned in a
        single response.
        """
        endpoint_summaries = self.sagemaker_backend.list_endpoints()
        return json.dumps({
            'Endpoints': [summary.response_object for summary in endpoint_summaries]
        })

    def list_endpoint_configs(self):
        """
        Handler for the SageMaker "ListEndpointConfigs" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpointConfigs.html.

        This function does not support pagination. All endpoint configs are returned in a
        single response.
        """
        # Note:
        endpoint_config_summaries = self.sagemaker_backend.list_endpoint_configs()
        return json.dumps({
            'EndpointConfigs': [summary.response_object for summary in endpoint_config_summaries]
        })

    def list_models(self):
        """
        Handler for the SageMaker "ListModels" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListModels.html.

        This function does not support pagination. All endpoint configs are returned in a
        single response.
        """
        model_summaries = self.sagemaker_backend.list_models()
        return json.dumps({
            'Models': [summary.response_object for summary in model_summaries]
        })

    def create_model(self):
        """
        Handler for the SageMaker "CreateModel" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateModel.html.
        """
        model_name = self.request_params["ModelName"]
        primary_container = self.request_params["PrimaryContainer"]
        execution_role_arn = self.request_params["ExecutionRoleArn"]
        tags = self.request_params.get("Tags", [])
        vpc_config = self.request_params.get("VpcConfig", None)
        new_model = self.sagemaker_backend.create_model(
                model_name=model_name, primary_container=primary_container,
                execution_role_arn=execution_role_arn, tags=tags, vpc_config=vpc_config,
                region_name=self.region)
        return json.dumps({
            'ModelArn': new_model.arn
        })

    def describe_model(self):
        """
        Handler for the SageMaker "DescribeModel" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeModel.html.
        """
        model_name = self.request_params["ModelName"]
        model_description = self.sagemaker_backend.describe_model(model_name)
        return json.dumps(model_description.response_object)

    def delete_model(self):
        """
        Handler for the SageMaker "DeleteModel" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteModel.html.
        """
        model_name = self.request_params["ModelName"]
        self.sagemaker_backend.delete_model(model_name)
        return ""


class SageMakerBackend(BaseBackend):
    """
    A mock backend for managing and exposing SageMaker resource state.
    """

    BASE_SAGEMAKER_ARN = "arn:aws:sagemaker:{region_name}:{account_id}:"

    def __init__(self):
        self.models = {}
        self.endpoints = {}
        self.endpoint_configs = {}
        self._endpoint_update_latency_seconds = 0

    def set_endpoint_update_latency(self, latency_seconds):
        """
        Sets the latency for the following operations that update endpoint state:
        - "create_endpoint"
        - "update_endpoint"
        """
        self._endpoint_update_latency_seconds = latency_seconds

    def set_endpoint_latest_operation(self, endpoint_name, operation):
        if endpoint_name not in self.endpoints:
            raise ValueError(
                "Attempted to manually set the latest operation for an endpoint"
                " that does not exist!")
        self.endpoints[endpoint_name].resource.latest_operation = operation

    @property
    def _url_module(self):
        """
        Required override from the Moto "BaseBackend" object that reroutes requests from the
        specified SageMaker URLs to the mocked SageMaker backend.
        """
        urls_module_name = "tests.sagemaker.mock.mock_sagemaker_urls"
        urls_module = __import__(urls_module_name, fromlist=['url_bases', 'url_paths'])
        return urls_module

    def _get_base_arn(self, region_name):
        """
        :return: A SageMaker ARN prefix that can be prepended to a resource name.
        """
        return SageMakerBackend.BASE_SAGEMAKER_ARN.format(
                region_name=region_name, account_id=ACCOUNT_ID)

    def create_endpoint_config(self, config_name, production_variants, tags, region_name):
        """
        Modifies backend state during calls to the SageMaker "CreateEndpointConfig" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateEndpointConfig.html.
        """
        if config_name in self.endpoint_configs:
            raise ValueError("Attempted to create an endpoint configuration with name:"
                             " {config_name}, but an endpoint configuration with this"
                             " name already exists.".format(config_name=config_name))
        for production_variant in production_variants:
            if "ModelName" not in production_variant:
                raise ValueError("Production variant must specify a model name.")
            elif production_variant["ModelName"] not in self.models:
                raise ValueError(
                        "Production variant specifies a model name that does not exist"
                        " Model name: '{model_name}'".format(
                            model_name=production_variant["ModelName"]))

        new_config = EndpointConfig(config_name=config_name,
                                    production_variants=production_variants,
                                    tags=tags)
        new_config_arn = self._get_base_arn(region_name=region_name) + new_config.arn_descriptor
        new_resource = SageMakerResourceWithArn(resource=new_config, arn=new_config_arn)
        self.endpoint_configs[config_name] = new_resource
        return new_resource

    def describe_endpoint_config(self, config_name):
        """
        Modifies backend state during calls to the SageMaker "DescribeEndpointConfig" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpointConfig.html.
        """
        if config_name not in self.endpoint_configs:
            raise ValueError("Attempted to describe an endpoint config with name: `{config_name}`"
                             " that does not exist.".format(config_name=config_name))

        config = self.endpoint_configs[config_name]
        return EndpointConfigDescription(config=config.resource, arn=config.arn)

    def delete_endpoint_config(self, config_name):
        """
        Modifies backend state during calls to the SageMaker "DeleteEndpointConfig" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteEndpointConfig.html.
        """
        if config_name not in self.endpoint_configs:
            raise ValueError("Attempted to delete an endpoint config with name: `{config_name}`"
                             " that does not exist.".format(config_name=config_name))

        del self.endpoint_configs[config_name]

    def create_endpoint(self, endpoint_name, endpoint_config_name, tags, region_name):
        """
        Modifies backend state during calls to the SageMaker "CreateEndpoint" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateEndpoint.html.
        """
        if endpoint_name in self.endpoints:
            raise ValueError("Attempted to create an endpoint with name: `{endpoint_name}`"
                             " but an endpoint with this name already exists.".format(
                                 endpoint_name=endpoint_name))

        if endpoint_config_name not in self.endpoint_configs:
            raise ValueError("Attempted to create an endpoint with a configuration named:"
                             " `{config_name}` However, this configuration does not exist.".format(
                                config_name=endpoint_config_name))

        new_endpoint = Endpoint(endpoint_name=endpoint_name,
                                config_name=endpoint_config_name,
                                tags=tags,
                                latest_operation=EndpointOperation.create_successful(
                                    latency_seconds=self._endpoint_update_latency_seconds))
        new_endpoint_arn = self._get_base_arn(region_name=region_name) + new_endpoint.arn_descriptor
        new_resource = SageMakerResourceWithArn(resource=new_endpoint, arn=new_endpoint_arn)
        self.endpoints[endpoint_name] = new_resource
        return new_resource

    def describe_endpoint(self, endpoint_name):
        """
        Modifies backend state during calls to the SageMaker "DescribeEndpoint" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpoint.html.
        """
        if endpoint_name not in self.endpoints:
            raise ValueError("Attempted to describe an endpoint with name: `{endpoint_name}`"
                             " that does not exist.".format(endpoint_name=endpoint_name))

        endpoint = self.endpoints[endpoint_name]
        config = self.endpoint_configs[endpoint.resource.config_name]
        return EndpointDescription(
                endpoint=endpoint.resource, config=config.resource, arn=endpoint.arn)

    def update_endpoint(self, endpoint_name, new_config_name):
        """
        Modifies backend state during calls to the SageMaker "UpdateEndpoint" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_UpdateEndpoint.html.
        """
        if endpoint_name not in self.endpoints:
            raise ValueError("Attempted to update an endpoint with name: `{endpoint_name}`"
                             " that does not exist.".format(endpoint_name=endpoint_name))

        if new_config_name not in self.endpoint_configs:
            raise ValueError("Attempted to update an endpoint named `{endpoint_name}` with a new"
                             " configuration named: `{config_name}`. However, this configuration"
                             " does not exist.".format(
                                endpoint_name=endpoint_name, config_name=new_config_name))

        endpoint = self.endpoints[endpoint_name]
        endpoint.resource.latest_operation = EndpointOperation.update_successful(
            latency_seconds=self._endpoint_update_latency_seconds)
        endpoint.resource.config_name = new_config_name
        return endpoint

    def delete_endpoint(self, endpoint_name):
        """
        Modifies backend state during calls to the SageMaker "DeleteEndpoint" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteEndpoint.html.
        """
        if endpoint_name not in self.endpoints:
            raise ValueError("Attempted to delete an endpoint with name: `{endpoint_name}`"
                             " that does not exist.".format(endpoint_name=endpoint_name))

        del self.endpoints[endpoint_name]

    def list_endpoints(self):
        """
        Modifies backend state during calls to the SageMaker "ListEndpoints" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpoints.html.
        """
        summaries = []
        for _, endpoint in self.endpoints.items():
            summary = EndpointSummary(endpoint=endpoint.resource, arn=endpoint.arn)
            summaries.append(summary)
        return summaries

    def list_endpoint_configs(self):
        """
        Modifies backend state during calls to the SageMaker "ListEndpointConfigs" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpointConfigs.html.
        """
        summaries = []
        for _, endpoint_config in self.endpoint_configs.items():
            summary = EndpointConfigSummary(
                    config=endpoint_config.resource, arn=endpoint_config.arn)
            summaries.append(summary)
        return summaries

    def list_models(self):
        """
        Modifies backend state during calls to the SageMaker "ListModels" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListModels.html.
        """
        summaries = []
        for _, model in self.models.items():
            summary = ModelSummary(model=model.resource, arn=model.arn)
            summaries.append(summary)
        return summaries

    def create_model(self, model_name, primary_container, execution_role_arn, tags, region_name,
                     vpc_config=None):
        """
        Modifies backend state during calls to the SageMaker "CreateModel" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateModel.html.
        """
        if model_name in self.models:
            raise ValueError("Attempted to create a model with name: `{model_name}`"
                             " but a model with this name already exists.".format(
                                model_name=model_name))

        new_model = Model(model_name=model_name, primary_container=primary_container,
                          execution_role_arn=execution_role_arn, tags=tags, vpc_config=vpc_config)
        new_model_arn = self._get_base_arn(region_name=region_name) + new_model.arn_descriptor
        new_resource = SageMakerResourceWithArn(resource=new_model, arn=new_model_arn)
        self.models[model_name] = new_resource
        return new_resource

    def describe_model(self, model_name):
        """
        Modifies backend state during calls to the SageMaker "DescribeModel" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeModel.html.
        """
        if model_name not in self.models:
            raise ValueError("Attempted to describe a model with name: `{model_name}`"
                             " that does not exist.".format(model_name=model_name))

        model = self.models[model_name]
        return ModelDescription(model=model.resource, arn=model.arn)

    def delete_model(self, model_name):
        """
        Modifies backend state during calls to the SageMaker "DeleteModel" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteModel.html.
        """
        if model_name not in self.models:
            raise ValueError("Attempted to delete an model with name: `{model_name}`"
                             " that does not exist.".format(model_name=model_name))

        del self.models[model_name]


class TimestampedResource(BaseModel):

    TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

    def __init__(self):
        curr_time = datetime.now().strftime(TimestampedResource.TIMESTAMP_FORMAT)
        self.creation_time = curr_time
        self.last_modified_time = curr_time


class Endpoint(TimestampedResource):
    """
    Object representing a SageMaker endpoint. The SageMakerBackend will create
    and manage Endpoints.
    """

    STATUS_IN_SERVICE = "InService"
    STATUS_FAILED = "Failed"
    STATUS_CREATING = "Creating"
    STATUS_UPDATING = "Updating"

    def __init__(self, endpoint_name, config_name, tags, latest_operation):
        """
        :param endpoint_name: The name of the Endpoint.
        :param config_name: The name of the EndpointConfiguration to associate with the Endpoint.
        :param tags: Arbitrary tags to associate with the endpoint.
        :param latest_operation: The most recent operation that was invoked on the endpoint,
                                 represented as an EndpointOperation object.
        """
        super(Endpoint, self).__init__()
        self.endpoint_name = endpoint_name
        self.config_name = config_name
        self.tags = tags
        self.latest_operation = latest_operation

    @property
    def arn_descriptor(self):
        return ":endpoint/{endpoint_name}".format(endpoint_name=self.endpoint_name)

    @property
    def status(self):
        return self.latest_operation.status()


class EndpointOperation:
    """
    Object representing a SageMaker endpoint operation ("create" or "update"). Every
    Endpoint is associated with the operation that was most recently invoked on it.
    """

    def __init__(self, latency_seconds, pending_status, completed_status):
        """
        :param latency: The latency of the operation, in seconds. Before the time window specified
                        by this latency elapses, the operation will have the status specified by
                        ``pending_status``. After the time window elapses, the operation will
                        have the status  specified by ``completed_status``.
        :param pending_status: The status that the operation should reflect *before* the latency
                               window has elapsed.
        :param completed_status: The status that the operation should reflect *after* the latency
                                 window has elapsed.
        """
        self.latency_seconds = latency_seconds
        self.pending_status = pending_status
        self.completed_status = completed_status
        self.start_time = time.time()

    def status(self):
        if time.time() - self.start_time < self.latency_seconds:
            return self.pending_status
        else:
            return self.completed_status

    @classmethod
    def create_successful(cls, latency_seconds):
        return cls(latency_seconds=latency_seconds, pending_status=Endpoint.STATUS_CREATING,
                   completed_status=Endpoint.STATUS_IN_SERVICE)

    @classmethod
    def create_unsuccessful(cls, latency_seconds):
        return cls(latency_seconds=latency_seconds, pending_status=Endpoint.STATUS_CREATING,
                   completed_status=Endpoint.STATUS_FAILED)

    @classmethod
    def update_successful(cls, latency_seconds):
        return cls(latency_seconds=latency_seconds, pending_status=Endpoint.STATUS_UPDATING,
                   completed_status=Endpoint.STATUS_IN_SERVICE)

    @classmethod
    def update_unsuccessful(cls, latency_seconds):
        return cls(latency_seconds=latency_seconds, pending_status=Endpoint.STATUS_UPDATING,
                   completed_status=Endpoint.STATUS_FAILED)


class EndpointSummary:
    """
    Object representing an endpoint entry in the endpoints list returned by
    SageMaker's "ListEndpoints" API:
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpoints.html.
    """

    def __init__(self, endpoint, arn):
        self.endpoint = endpoint
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'EndpointName': self.endpoint.endpoint_name,
            'CreationTime': self.endpoint.creation_time,
            'LastModifiedTime': self.endpoint.last_modified_time,
            'EndpointStatus': self.endpoint.status,
            'EndpointArn': self.arn,
        }
        return response


class EndpointDescription:
    """
    Object representing an endpoint description returned by SageMaker's
    "DescribeEndpoint" API:
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpoint.html.
    """

    def __init__(self, endpoint, config, arn):
        self.endpoint = endpoint
        self.config = config
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'EndpointName': self.endpoint.endpoint_name,
            'EndpointArn': self.arn,
            'EndpointConfigName': self.endpoint.config_name,
            'ProductionVariants': self.config.production_variants,
            'EndpointStatus': self.endpoint.status,
            'CreationTime': self.endpoint.creation_time,
            'LastModifiedTime': self.endpoint.last_modified_time,
        }
        return response


class EndpointConfig(TimestampedResource):
    """
    Object representing a SageMaker endpoint configuration. The SageMakerBackend will create
    and manage EndpointConfigs.
    """

    def __init__(self, config_name, production_variants, tags):
        super(EndpointConfig, self).__init__()
        self.config_name = config_name
        self.production_variants = production_variants
        self.tags = tags

    @property
    def arn_descriptor(self):
        return ":endpoint-config/{config_name}".format(config_name=self.config_name)


class EndpointConfigSummary:
    """
    Object representing an endpoint configuration entry in the configurations list returned by
    SageMaker's "ListEndpointConfigs" API:
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpointConfigs.html.
    """

    def __init__(self, config, arn):
        self.config = config
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'EndpointConfigName': self.config.config_name,
            'EndpointArn': self.arn,
            'CreationTime': self.config.creation_time,
        }
        return response


class EndpointConfigDescription:
    """
    Object representing an endpoint configuration description returned by SageMaker's
    "DescribeEndpointConfig" API:
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpointConfig.html.
    """

    def __init__(self, config, arn):
        self.config = config
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'EndpointConfigName': self.config.config_name,
            'EndpointConfigArn': self.arn,
            'ProductionVariants': self.config.production_variants,
            'CreationTime': self.config.creation_time,
        }
        return response


class Model(TimestampedResource):
    """
    Object representing a SageMaker model. The SageMakerBackend will create and manage Models.
    """

    def __init__(self, model_name, primary_container, execution_role_arn, tags, vpc_config):
        super(Model, self).__init__()
        self.model_name = model_name
        self.primary_container = primary_container
        self.execution_role_arn = execution_role_arn
        self.tags = tags
        self.vpc_config = vpc_config

    @property
    def arn_descriptor(self):
        return ":model/{model_name}".format(model_name=self.model_name)


class ModelSummary:
    """
    Object representing a model entry in the models list returned by SageMaker's
    "ListModels" API: https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListModels.html.
    """

    def __init__(self, model, arn):
        self.model = model
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'ModelArn': self.arn,
            'ModelName': self.model.model_name,
            'CreationTime': self.model.creation_time,
        }
        return response


class ModelDescription:
    """
    Object representing a model description returned by SageMaker's
    "DescribeModel" API: https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeModel.html.
    """

    def __init__(self, model, arn):
        self.model = model
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'ModelArn': self.arn,
            'ModelName': self.model.model_name,
            'PrimaryContainer': self.model.primary_container,
            'ExecutionRoleArn': self.model.execution_role_arn,
            'VpcConfig': self.model.vpc_config if self.model.vpc_config else {},
            'CreationTime': self.model.creation_time,
        }
        return response


# Create a SageMaker backend for each EC2 region
sagemaker_backends = {}
for region, ec2_backend in ec2_backends.items():
    new_backend = SageMakerBackend()
    sagemaker_backends[region] = new_backend

mock_sagemaker = base_decorator(sagemaker_backends)
