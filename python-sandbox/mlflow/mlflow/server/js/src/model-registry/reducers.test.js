import ModelRegistryReducers, {
  getAllModelVersions,
  getModelVersion,
  getModelVersions,
} from './reducers';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from './test-utils';
import {
  DELETE_MODEL_VERSION,
  DELETE_REGISTERED_MODEL,
  GET_MODEL_VERSION,
  GET_REGISTERED_MODEL,
  LIST_REGISTRED_MODELS,
  SEARCH_MODEL_VERSIONS,
} from './actions';
import { fulfilled } from '../common/utils/ActionUtils';

const { modelByName, modelVersionsByModel } = ModelRegistryReducers;

describe('test modelByName', () => {
  test('initial state', () => {
    expect(modelByName(undefined, {})).toEqual({});
  });

  test('LIST_REGISTRED_MODELS handles empty state correctly', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const modelB = mockRegisteredModelDetailed('modelB');
    const state = {};
    const action = {
      type: fulfilled(LIST_REGISTRED_MODELS),
      payload: {
        registered_models: [modelA, modelB],
      },
    };
    expect(modelByName(state, action)).toEqual({ modelA, modelB });
  });

  test('LIST_REGISTRED_MODELS flushes previous loaded models in state (1)', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const modelB = mockRegisteredModelDetailed('modelB');
    const modelC = mockRegisteredModelDetailed('modelC');
    const state = { modelA };
    const action = {
      type: fulfilled(LIST_REGISTRED_MODELS),
      payload: {
        registered_models: [modelB, modelC],
      },
    };
    expect(modelByName(state, action)).toEqual({ modelB, modelC });
  });

  test('LIST_REGISTRED_MODELS flushes previous loaded models in state (2)', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const modelB = mockRegisteredModelDetailed('modelB');
    const modelC = mockRegisteredModelDetailed('modelC');
    const state = { modelA, modelB };
    const action = {
      type: fulfilled(LIST_REGISTRED_MODELS),
      payload: {
        registered_models: [modelB, modelC],
      },
    };
    expect(modelByName(state, action)).toEqual({ modelB, modelC });
  });

  test('LIST_REGISTRED_MODELS flushes previous loaded models in state (3)', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const modelB = mockRegisteredModelDetailed('modelB');
    const state = { modelA, modelB };
    const action = {
      type: fulfilled(LIST_REGISTRED_MODELS),
      payload: {
        registered_models: [],
      },
    };
    expect(modelByName(state, action)).toEqual({});
  });

  test('LIST_REGISTRED_MODELS should have no effect on valid state', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const modelB = mockRegisteredModelDetailed('modelB');
    const state = { modelA, modelB };
    const action = {
      type: fulfilled(LIST_REGISTRED_MODELS),
      payload: {
        registered_models: [modelB, modelA],
      },
    };
    expect(modelByName(state, action)).toEqual({ modelB, modelA });
  });

  test('GET_REGISTERED_MODEL updates empty state correctly', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const state = {};
    const action = {
      type: fulfilled(GET_REGISTERED_MODEL),
      meta: { modelName: 'modelA' },
      payload: {
        registered_model: modelA,
      },
    };
    expect(modelByName(state, action)).toEqual({ modelA: modelA });
  });

  test('GET_REGISTERED_MODEL updates incorrect state', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const state = { modelA: undefined };
    const action = {
      type: fulfilled(GET_REGISTERED_MODEL),
      meta: { modelName: 'modelA' },
      payload: {
        registered_model: modelA,
      },
    };
    expect(modelByName(state, action)).toEqual({ modelA: modelA });
  });

  test('GET_REGISTERED_MODEL does not affect other models', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const modelB = mockRegisteredModelDetailed('modelA');
    const state = { modelB: modelB };
    const action = {
      type: fulfilled(GET_REGISTERED_MODEL),
      meta: { modelName: 'modelA' },
      payload: {
        registered_model: modelA,
      },
    };
    expect(modelByName(state, action)).toEqual({ modelA: modelA, modelB: modelB });
  });

  test('DELETE_REGISTERED_MODEL should handle empty state correctly', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const state = {};
    const action = {
      type: fulfilled(DELETE_REGISTERED_MODEL),
      meta: { model: modelA },
    };
    expect(modelByName(state, action)).toEqual({});
  });

  test('DELETE_REGISTERED_MODEL cleans out state correctly', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const state = { modelA: modelA };
    const action = {
      type: fulfilled(DELETE_REGISTERED_MODEL),
      meta: { model: modelA },
    };
    expect(modelByName(state, action)).toEqual({});
  });

  test('DELETE_REGISTERED_MODEL does not remove other models from state', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const modelB = mockRegisteredModelDetailed('modelB');
    const state = { modelA: modelA, modelB: modelB };
    const action = {
      type: fulfilled(DELETE_REGISTERED_MODEL),
      meta: { model: modelA },
    };
    expect(modelByName(state, action)).toEqual({ modelB: modelB });
  });

  test('DELETE_REGISTERED_MODEL does not remove other models with similar name from state', () => {
    const modelA = mockRegisteredModelDetailed('modelA');
    const modelAA = mockRegisteredModelDetailed('modelAA');
    const state = { modelA: modelA, modelAA: modelAA };
    const action = {
      type: fulfilled(DELETE_REGISTERED_MODEL),
      meta: { model: modelA },
    };
    expect(modelByName(state, action)).toEqual({ modelAA: modelAA });
  });
});

describe('test modelVersionsByModel', () => {
  test('initial state (1)', () => {
    expect(modelVersionsByModel(undefined, {})).toEqual({});
  });

  test('initial state (2)', () => {
    const versionA = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    expect(modelVersionsByModel({ 1: versionA }, {})).toEqual({ 1: versionA });
  });

  test('GET_MODEL_VERSION updates empty state correctly', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const state = {};
    const action = {
      type: fulfilled(GET_MODEL_VERSION),
      meta: { modelName: 'modelA' },
      payload: {
        model_version: version1,
      },
    };
    expect(modelVersionsByModel(state, action)).toEqual({ modelA: { 1: version1 } });
  });

  test('GET_MODEL_VERSION updates non-empty state correctly', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const version2 = mockModelVersionDetailed('modelA', 2, 'Staging', 'READY');
    const state = { modelA: { 1: version1 } };
    const action = {
      type: fulfilled(GET_MODEL_VERSION),
      meta: { modelName: 'modelA' },
      payload: {
        model_version: version2,
      },
    };
    expect(modelVersionsByModel(state, action)).toEqual({
      modelA: {
        1: version1,
        2: version2,
      },
    });
  });

  test('DELETE_MODEL_VERSION handles missing versions correctly', () => {
    const state = { modelA: {} };
    const action = {
      meta: { modelName: 'modelA', version: 1 },
      type: fulfilled(DELETE_MODEL_VERSION),
    };
    expect(modelVersionsByModel(state, action)).toEqual({ ...state });
  });

  test('DELETE_MODEL_VERSION updates state correctly (1)', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const state = { modelA: { 1: version1 } };
    const action = {
      meta: { modelName: 'modelA', version: 1 },
      type: fulfilled(DELETE_MODEL_VERSION),
    };
    expect(modelVersionsByModel(state, action)).toEqual({ modelA: {} });
  });

  test('DELETE_MODEL_VERSION updates state correctly (2)', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const version2 = mockModelVersionDetailed('modelA', 2, 'Staging', 'READY');
    const state = { modelA: { 1: version1, 2: version2 } };
    const action = {
      meta: { modelName: 'modelA', version: 1 },
      type: fulfilled(DELETE_MODEL_VERSION),
    };
    expect(modelVersionsByModel(state, action)).toEqual({ modelA: { 2: version2 } });
  });

  test('DELETE_MODEL_VERSION does not mess with other registered models', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const version2 = mockModelVersionDetailed('modelA', 2, 'Staging', 'READY');
    const version3 = mockModelVersionDetailed('modelB', 2, 'Staging', 'READY');
    const state = { modelA: { 1: version1, 2: version2 }, modelB: { 2: version3 } };
    const action = {
      meta: { modelName: 'modelA', version: 2 },
      type: fulfilled(DELETE_MODEL_VERSION),
    };
    expect(modelVersionsByModel(state, action)).toEqual({
      modelA: { 1: version1 },
      modelB: { 2: version3 },
    });
  });

  test('SEARCH_MODEL_VERSION handles empty state (1)', () => {
    const state = {};
    const action = {
      type: fulfilled(SEARCH_MODEL_VERSIONS),
      payload: {
        model_versions: [],
      },
    };
    expect(modelVersionsByModel(state, action)).toEqual({});
  });

  test('SEARCH_MODEL_VERSION handles empty state (2)', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const version2 = mockModelVersionDetailed('modelA', 2, 'Staging', 'READY');
    const state = {};
    const action = {
      type: fulfilled(SEARCH_MODEL_VERSIONS),
      payload: {
        model_versions: [version2, version1],
      },
    };
    expect(modelVersionsByModel(state, action)).toEqual({
      modelA: {
        1: version1,
        2: version2,
      },
    });
  });

  test('SEARCH_MODEL_VERSION updates states correctly', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const version2 = mockModelVersionDetailed('modelA', 2, 'Staging', 'READY');
    const state = { modelX: {} };
    const action = {
      type: fulfilled(SEARCH_MODEL_VERSIONS),
      payload: {
        model_versions: [version2, version1],
      },
    };
    expect(modelVersionsByModel(state, action)).toEqual({
      modelA: {
        1: version1,
        2: version2,
      },
      modelX: {},
    });
  });

  test('SEARCH_MODEL_VERSION refreshes state with new models', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const version2 = mockModelVersionDetailed('modelA', 2, 'Staging', 'READY');
    const version3 = mockModelVersionDetailed('modelA', 3, 'Staging', 'READY');
    const state = { modelA: { 1: version1, 2: version2 } };
    const action = {
      type: fulfilled(SEARCH_MODEL_VERSIONS),
      payload: {
        model_versions: [version3],
      },
    };
    expect(modelVersionsByModel(state, action)).toEqual({
      modelA: {
        1: version1,
        2: version2,
        3: version3,
      },
    });
  });
});

describe('test getModelVersion', () => {
  test('getModelVersion handles empty state', () => {
    const state = {
      entities: {
        modelVersionsByModel: { undefined: {} },
      },
    };
    expect(getModelVersion(state, 'modelA', 1)).toEqual(undefined);
  });

  test('getModelVersion handles missing model', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const state = {
      entities: {
        modelVersionsByModel: { modelX: { 1: version1 } },
      },
    };
    expect(getModelVersion(state, 'modelA', 1)).toEqual(undefined);
  });

  test('getModelVersion handles missing version', () => {
    const version2 = mockModelVersionDetailed('modelA', 2, 'Production', 'READY');
    const state = {
      entities: {
        modelVersionsByModel: { modelA: { 2: version2 } },
      },
    };
    expect(getModelVersion(state, 'modelA', 1)).toEqual(undefined);
  });

  test('getModelVersion returns correct version (1)', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const version2 = mockModelVersionDetailed('modelA', 2, 'Staging', 'READY');
    const state = {
      entities: {
        modelVersionsByModel: { modelA: { 1: version1, 2: version2 } },
      },
    };
    expect(getModelVersion(state, 'modelA', 1)).toEqual(version1);
  });

  test('getModelVersion returns correct version (2)', () => {
    const versionA1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const versionB1 = mockModelVersionDetailed('modelB', 1, 'Staging', 'READY');
    const state = {
      entities: {
        modelVersionsByModel: { modelA: { 1: versionA1 }, modelB: { 1: versionB1 } },
      },
    };
    expect(getModelVersion(state, 'modelA', 1)).toEqual(versionA1);
  });
});

describe('test getModelVersions', () => {
  test('getModelVersions handles empty state', () => {
    const state = {
      entities: {
        modelVersionsByModel: { undefined: {} },
      },
    };
    expect(getModelVersions(state, 'modelA', 1)).toEqual(undefined);
  });

  test('getModelVersions handles missing model', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const state = {
      entities: {
        modelVersionsByModel: { modelX: { 1: version1 } },
      },
    };
    expect(getModelVersions(state, 'modelA')).toEqual(undefined);
  });

  test('getModelVersions returns correct versions (1)', () => {
    const version2 = mockModelVersionDetailed('modelA', 2, 'Production', 'READY');
    const state = {
      entities: {
        modelVersionsByModel: { modelA: { 2: version2 } },
      },
    };
    expect(getModelVersions(state, 'modelA')).toEqual([version2]);
  });

  test('getModelVersions returns correct versions (2)', () => {
    const version1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const version2 = mockModelVersionDetailed('modelA', 2, 'Staging', 'READY');
    const state = {
      entities: {
        modelVersionsByModel: { modelA: { 1: version1, 2: version2 } },
      },
    };
    expect(getModelVersions(state, 'modelA')).toEqual([version1, version2]);
  });

  test('getModelVersions returns correct versions (3)', () => {
    const versionA1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const versionB1 = mockModelVersionDetailed('modelB', 1, 'Staging', 'READY');
    const state = {
      entities: {
        modelVersionsByModel: { modelA: { 1: versionA1 }, modelB: { 1: versionB1 } },
      },
    };
    expect(getModelVersions(state, 'modelA')).toEqual([versionA1]);
  });
});

describe('test getAllModelVersions', () => {
  test('getAllModelVersions handles empty state', () => {
    const state = {
      entities: {
        modelVersionsByModel: { undefined: {} },
      },
    };
    expect(getAllModelVersions(state)).toEqual([]);
  });

  test('getAllModelVersions returns versions of all models', () => {
    const versionA1 = mockModelVersionDetailed('modelA', 1, 'Production', 'READY');
    const versionB1 = mockModelVersionDetailed('modelB', 1, 'Staging', 'READY');
    const state = {
      entities: {
        modelVersionsByModel: { modelA: { 1: versionA1 }, modelB: { 1: versionB1 } },
      },
    };
    expect(getAllModelVersions(state)).toEqual([versionA1, versionB1]);
  });
});
