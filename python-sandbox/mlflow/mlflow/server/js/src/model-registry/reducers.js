import {
  LIST_REGISTRED_MODELS,
  SEARCH_MODEL_VERSIONS,
  GET_REGISTERED_MODEL,
  GET_MODEL_VERSION,
  DELETE_MODEL_VERSION,
  DELETE_REGISTERED_MODEL,
} from './actions';
import { getProtoField } from './utils';
import _ from 'lodash';
import { fulfilled } from '../common/utils/ActionUtils';

const modelByName = (state = {}, action) => {
  switch (action.type) {
    case fulfilled(LIST_REGISTRED_MODELS): {
      const models = action.payload[getProtoField('registered_models')];
      const nameToModelMap = {};
      if (models) {
        models.forEach((model) => (nameToModelMap[model.name] = model));
      }
      return {
        ...nameToModelMap,
      };
    }
    case fulfilled(GET_REGISTERED_MODEL): {
      const detailedModel = action.payload[getProtoField('registered_model')];
      const { modelName } = action.meta;
      const modelWithUpdatedMetadata = {
        ...state[modelName],
        ...detailedModel,
      };
      return {
        ...state,
        ...{ [modelName]: modelWithUpdatedMetadata },
      };
    }
    case fulfilled(DELETE_REGISTERED_MODEL): {
      const { model } = action.meta;
      return _.omit(state, model.name);
    }
    default:
      return state;
  }
};

// 2-levels lookup for model version indexed by (modelName, version)
const modelVersionsByModel = (state = {}, action) => {
  switch (action.type) {
    case fulfilled(GET_MODEL_VERSION): {
      const modelVersion = action.payload[getProtoField('model_version')];
      const { modelName } = action.meta;
      const updatedMap = {
        ...state[modelName],
        [modelVersion.version]: modelVersion,
      };
      return {
        ...state,
        [modelName]: updatedMap,
      };
    }
    case fulfilled(SEARCH_MODEL_VERSIONS): {
      const modelVersions = action.payload[getProtoField('model_versions')];
      if (!modelVersions) {
        return state;
      }

      // Merge all modelVersions into the store
      return modelVersions.reduce(
        (newState, modelVersion) => {
          const { name, version } = modelVersion;
          return {
            ...newState,
            [name]: {
              ...newState[name],
              [version]: modelVersion,
            },
          };
        },
        { ...state },
      );
    }
    case fulfilled(DELETE_MODEL_VERSION): {
      const { modelName, version } = action.meta;
      const modelVersionByVersion = state[modelName];
      return {
        ...state,
        [modelName]: _.omit(modelVersionByVersion, version),
      };
    }
    case fulfilled(DELETE_REGISTERED_MODEL): {
      const { model } = action.meta;
      return _.omit(state, model.name);
    }
    default:
      return state;
  }
};

export const getModelVersion = (state, modelName, version) => {
  const modelVersions = state.entities.modelVersionsByModel[modelName];
  return modelVersions && modelVersions[version];
};

export const getModelVersions = (state, modelName) => {
  const modelVersions = state.entities.modelVersionsByModel[modelName];
  return modelVersions && Object.values(modelVersions);
};

export const getAllModelVersions = (state) => {
  return _.flatMap(Object.values(state.entities.modelVersionsByModel), (modelVersionByVersion) =>
    Object.values(modelVersionByVersion),
  );
};

export default {
  modelByName,
  modelVersionsByModel,
};
